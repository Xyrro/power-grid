"""PGLib-UC California: train the label-free Model 1 variants and the error-cost (harm) model; no full MILP.

    python3 scripts/uc_pglib_train.py --stage bce      # imitation of the repaired LP relaxation (label-free)
    python3 scripts/uc_pglib_train.py --stage rl       # + REINFORCE with the dispatch LP as critic
    python3 scripts/uc_pglib_train.py --stage st       # one self-training round (reduced MILPs as teacher)
    python3 scripts/uc_pglib_train.py --stage harm     # error-cost labels (dispatch LPs) + harm ensemble
    python3 scripts/uc_pglib_train.py --stage probs    # probabilities of every model on val / test (+ timing)

Uses data/generated/pglib_ca/{train,val,test}.npz; early stopping uses only label-free quantities
(BCE to the repaired relaxation labels of 16 held-out training instances); no MILP label is read here.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.pglib import load_bases, load_npz, subset  # noqa: E402
from otsl.pglib_ml import (DispatchOracle, PGFeaturizer, SolverPool, build_model1, fix_array, harm_job,  # noqa: E402
                           inst_arrays, reduced_job, repair_fn)
from otsl.selftrain import label_fixings, train_bce_fixed  # noqa: E402
from otsl.ucml import train_uc_bce, train_uc_reinforce  # noqa: E402

ROOT, OUT = "data/generated/pglib_ca", "results/pglib"
LOG = {}


def log(msg):
    print(msg, flush=True)


def save_model(m1, name):
    torch.save(m1.net.state_dict(), os.path.join(OUT, f"pglib_{name}.pt"))


def load_model(sysm, feat, name, seed=0):
    m1 = build_model1(sysm, feat, seed=seed)
    m1.net.load_state_dict(torch.load(os.path.join(OUT, f"pglib_{name}.pt")))
    m1.net.eval()
    return m1


def stats_path():
    return os.path.join(OUT, "pglib_train_stats.json")


def update_stats(**kw):
    p = stats_path()
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.update(kw)
    json.dump(st, open(p, "w"), indent=1, default=float)


def strip(d):
    """training view without any MILP field; obj := LP relaxation cost (the REINFORCE reference is per instance and
    cancels in the leave-one-out baseline)"""
    out = {k: v for k, v in d.items() if k not in ("u", "obj", "bound", "gap", "time", "inc_t", "inc_obj", "inc_bound")}
    out["obj"] = d["c_rel"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--rl_steps", type=int, default=40)
    ap.add_argument("--rl_bs", type=int, default=8)
    ap.add_argument("--rl_samples", type=int, default=4)
    ap.add_argument("--st_n", type=int, default=60)
    ap.add_argument("--st_ratio", type=float, default=0.9)
    ap.add_argument("--st_tl", type=float, default=30.0)
    ap.add_argument("--harm_n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_hold", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    sysm, _ = load_bases(a.T)
    full = strip(load_npz(os.path.join(ROOT, "train.npz")))
    n_all = len(full["load"])
    tr = subset(full, np.arange(n_all - a.n_hold))          # training instances
    va = subset(full, np.arange(n_all - a.n_hold, n_all))   # label-free hold-out for early stopping (no MILP)
    va["u_target"] = va["y_lf"]
    feat = PGFeaturizer(sysm, tr)
    cfg = dict(T=a.T)
    rep = repair_fn(sysm)

    if a.stage == "bce":
        m1 = build_model1(sysm, feat, seed=a.seed)
        t0 = time.time()
        train_uc_bce(m1, tr, va, tr["y_lf"], epochs=a.epochs, seed=a.seed, log_every=5, bs=16)
        save_model(m1, "m1_lf")
        p = m1.predict(va)
        update_stats(bce_s=time.time() - t0, bce_val_acc_to_lf=float(((p > 0.5) == va["y_lf"]).mean()))
        log(f"saved m1_lf ({time.time() - t0:.0f}s)")

    elif a.stage == "rl":
        m1 = load_model(sysm, feat, "m1_lf", a.seed)
        oracle = DispatchOracle(cfg, a.workers)
        t0 = time.time()
        _, hist = train_uc_reinforce(m1, tr, oracle, steps=a.rl_steps, bs=a.rl_bs, n_samples=a.rl_samples, seed=a.seed,
                                     log_every=5, repair=rep)
        save_model(m1, "m1_rl")
        update_stats(rl_s=time.time() - t0, rl_lps=oracle.n, rl_hist=hist)
        oracle.close()
        log(f"saved m1_rl ({time.time() - t0:.0f}s, {oracle.n} LPs)")

    elif a.stage == "st":
        # one self-training round on the first st_n training instances: fix the confident decisions on which the
        # model agrees with the current label (repaired relaxation), reduced MILP (short limit), keep if cheaper
        m1 = load_model(sysm, feat, "m1_lf", a.seed)
        n = min(a.st_n, len(tr["load"]))
        idx = np.arange(n)
        p = m1.predict(subset(tr, idx))
        jobs = []
        for k, i in enumerate(idx):
            fix = label_fixings(p[k], tr["y_lf"][i], a.st_ratio, agree=True)
            jobs.append((int(i), inst_arrays(tr, i), fix_array(fix), a.st_tl, 1e-3))
        pool = SolverPool(cfg, a.workers)
        t0 = time.time()
        res = pool.run(reduced_job, jobs, log=log, every=10)
        pool.close()
        lab = tr["y_lf"].copy()
        lab_cost = tr["c_lf"].copy()
        better = 0
        for r in res:
            if r["u"] is not None and r["cost"] < lab_cost[r["i"]] - 1e-9:
                lab[r["i"]] = r["u"]
                lab_cost[r["i"]] = r["cost"]
                better += 1
        t_st = time.time() - t0
        np.savez_compressed(os.path.join(ROOT, "st_labels.npz"), idx=idx, u=lab[idx], cost=lab_cost[idx],
                            c_lf=tr["c_lf"][idx], c_rel=tr["c_rel"][idx],
                            milp_s=np.array([r["milp_s"] for r in sorted(res, key=lambda r: r["i"])]))
        impr = (tr["c_lf"][idx] - lab_cost[idx]) / tr["c_lf"][idx]
        log(f"self-training labels: {better}/{n} improved, mean label cost change {impr.mean() * 100:.3f}% "
            f"({t_st:.0f}s)")
        # retrain on the improved pool (all training instances; the rest keep the relaxation label)
        m2 = build_model1(sysm, feat, seed=a.seed)
        t1 = time.time()
        train_bce_fixed(m2, tr, lab, epochs=a.epochs, seed=a.seed, log_every=5, bs=16)
        save_model(m2, "m1_st")
        update_stats(st_label_s=t_st, st_core_s=float(sum(r["milp_s"] for r in res)), st_improved=better, st_n=n,
                     st_gap_lf_to_st_mean=float(impr.mean()), st_train_s=time.time() - t1)

    elif a.stage == "harm":
        # error-cost labels on training instances: reference = self-training label (no MILP), probabilities of the
        # self-trained model; one dispatch LP per wrong decision
        from otsl.fixpolicy import FixFeaturizer, HarmEnsemble, HarmModel
        L = np.load(os.path.join(ROOT, "st_labels.npz"))
        idx = L["idx"][:a.harm_n]
        m1 = load_model(sysm, feat, "m1_st", a.seed)
        p = m1.predict(subset(tr, idx))
        jobs = [(int(i), inst_arrays(tr, i), L["u"][k], p[k]) for k, i in enumerate(idx)]
        pool = SolverPool(cfg, a.workers)
        t0 = time.time()
        res = {r["i"]: r for r in pool.run(harm_job, jobs, log=log, every=10)}
        pool.close()
        t_h = time.time() - t0
        ff = FixFeaturizer(sysm, a.T)
        X = np.stack([ff(p[k], tr, i) for k, i in enumerate(idx)])
        H = np.zeros(p.shape)
        for k, i in enumerate(idx):
            for (t, g), h in res[int(i)]["single"].items():
                H[k, t, g] = min(h, 10.0) if np.isfinite(h) else 10.0
        n_tr = int(0.8 * len(idx))
        F_ = X.shape[-1]
        members = []
        for sd in range(3):
            hm = HarmModel(F_, 64, seed=sd)
            hm.fit(X[:n_tr].reshape(-1, F_), H[:n_tr].reshape(-1), X[n_tr:], H[n_tr:], epochs=30, seed=sd, log=log)
            members.append(dict(state=hm.net.state_dict(), mu=hm.mu, sd=hm.sd))
        torch.save(dict(d_in=F_, hidden=64, members=members), os.path.join(OUT, "pglib_harm.pt"))
        update_stats(harm_label_s=t_h, harm_n=len(idx), harm_lps=int(sum(r["n_wrong"] + 1 for r in res.values())),
                     harm_wrong_per_inst=float(np.mean([r["n_wrong"] for r in res.values()])),
                     harm_positive_share=float((H > 1e-4).sum() / max(1, (H != 0).sum())))
        log(f"harm model saved ({t_h:.0f}s labels)")

    elif a.stage == "probs":
        # probabilities of every model and of the paper's kNN on val / test, with per-instance inference times;
        # validation criterion for the probability source of Learning to Fix on our model: error rate of the 95 %
        # most confident decisions against the (aligned) validation MILP schedules
        from otsl.fixpolicy import align_to_prediction
        from otsl.pglib_ml import KNNProb
        out = {}
        val_full = load_npz(os.path.join(ROOT, "val.npz"))
        splits = [("va", strip(val_full))]
        if os.path.exists(os.path.join(ROOT, "test.npz")):
            splits.append(("te", strip(load_npz(os.path.join(ROOT, "test.npz")))))
        L = np.load(os.path.join(ROOT, "st_labels.npz"))
        kl = subset(tr, L["idx"])
        knn = KNNProb(sysm, kl, L["u"], k=50)
        for split, d in splits:
            t0 = time.time()
            out[f"knn_{split}"] = knn.predict(d)
            out[f"knn_{split}_s"] = np.array((time.time() - t0) / len(d["load"]))
        crit = {}
        for name in ("m1_lf", "m1_rl", "m1_st"):
            if not os.path.exists(os.path.join(OUT, f"pglib_{name}.pt")):
                continue
            m1 = load_model(sysm, feat, name, a.seed)
            for split, d in splits:
                t0 = time.time()
                out[f"{name}_{split}"] = m1.predict(d, bs=1)
                out[f"{name}_{split}_s"] = np.array((time.time() - t0) / len(d["load"]))
        for name in ("knn", "m1_lf", "m1_rl", "m1_st"):
            if f"{name}_va" not in out:
                continue
            p = out[f"{name}_va"]
            errs = []
            for i in range(len(p)):
                yhat = (p[i] > 0.5).astype(np.int8)
                ua = align_to_prediction(sysm, val_full["u"][i], yhat, val_full["u0"][i])
                conf = np.abs(p[i] - 0.5)
                k = int(round(0.95 * conf.size))
                top = np.argsort(-conf.reshape(-1), kind="stable")[:k]
                errs.append(float((yhat.reshape(-1)[top] != ua.reshape(-1)[top]).mean()))
                if i == 0:
                    pass
            crit[name] = dict(err95=float(np.mean(errs)),
                              acc=float(np.mean([((p[i] > 0.5) == align_to_prediction(sysm, val_full["u"][i], (p[i] > 0.5).astype(np.int8),
                                                                                       val_full["u0"][i])).mean() for i in range(len(p))])))
        best = min((v["err95"], k) for k, v in crit.items() if k != "knn")[1]
        update_stats(val_criterion=crit, val_selected_source=best)
        np.savez_compressed(os.path.join(ROOT, "pglib_probs.npz"), **out)
        log(f"val criterion {crit} -> selected {best}")
        log("saved probabilities " + ", ".join(k for k in out if not k.endswith("_s")))
