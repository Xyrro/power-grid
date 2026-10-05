"""Constrained (Lagrangian) fine-tuning of Model 1 for the end-to-end mode on B2 (uc12), with
baselines under the same LP budget (plain REINFORCE, a He et al. 2026-style repair policy, kNN-20).

    python scripts/uc_constrained.py --stage bce                       # label-free BCE predictor
    python scripts/uc_constrained.py --stage rl  --name rl_lf          # plain REINFORCE (current method)
    python scripts/uc_constrained.py --stage lag --name lag_B --muprop 1 --window 1
    python scripts/uc_constrained.py --stage he  --name he_bc          # BC + PPO repair policy
    python scripts/uc_constrained.py --stage test --models rl_lf,lag_B,he_bc,...   # final test evaluation

Every stage writes results/uc12/constrained_<stage/name>_*.json. Hyperparameters are selected on
val; the test stage evaluates all 120 test instances once. Label-free stages never read MILP fields
of the training or validation set (reference cost = LP relaxation cost c_rel); MILP costs are used
only to score test (and, for reporting, val) results.
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
from otsl.constrained import (HourlyOracle, RepairEnv, RepairPolicy, adequacy_repair_blocks, repair_probs,  # noqa: E402
                              train_constrained, train_reinforce_plain, train_repair_ppo)
from otsl.uc import adequacy_repair, load_rts_gmlc, repair_min_updown  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, train_uc_bce, train_uc_reinforce, uc_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_gen import UC_CONFIGS  # noqa: E402
from uc_model1 import candidates_from_probs, knn_candidates  # noqa: E402

MILP_KEYS = ("u", "p", "va", "r", "obj", "time", "gap", "opt", "shed", "alt_u", "alt_c")
OUT = os.path.join("results", "uc12")
KEY = {"train": 0, "val": 10 ** 6, "test": 2 * 10 ** 6}


class _ScoreAdapter:
    """ucml.train_uc_reinforce expects oracle.evaluate -> (cost, shed, short)."""

    def __init__(self, o):
        self.o = o

    @property
    def n(self):
        return self.o.n

    def evaluate(self, d, idx, us, keys):
        return self.o.scores(d, idx, us, np.asarray(keys))


def setup(cfg_name="uc12"):
    cfg = UC_CONFIGS[cfg_name]
    root = os.path.join("data", "generated", cfg_name)
    tr_full, va_full, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    sysm = load_rts_gmlc()
    rep = lambda u, u0: repair_min_updown(u, u0, sysm.min_up, sysm.min_dn)
    return cfg, tr_full, va_full, te, sysm, rep


def make_repair(kind, d, sysm, rep):
    """repair(u, i) for instance i of dataset d: min up/down repair, preceded by the block adequacy repair
    when kind == 'blocks'."""
    if kind == "blocks":
        return lambda u, i: rep(adequacy_repair_blocks(u, d["u0"][i], d["load"][i], d["avail"][i], d["sr"][i], sysm),
                                d["u0"][i])
    return lambda u, i: rep(u, d["u0"][i])


def strip(d):
    """remove every MILP-derived field; the reference cost becomes the LP relaxation cost"""
    return dict({k: v for k, v in d.items() if k not in MILP_KEYS}, obj=d["c_rel"])


def evaluate_probs(name, p, d, split, oracle, sysm, rep, seed=0, screening=True, arrays=None):
    """One-shot top-1, top-1 + adequacy repair, and candidate screening, all through the exact LP."""
    n = len(d["load"])
    keys = np.arange(n) + KEY[split]
    fixrep = lambda U, U0: np.array([rep(U[i], U0[i]) for i in range(len(U))])

    def adequate(U):
        V = np.array([adequacy_repair(U[i], d["load"][i], d["avail"][i], d["sr"][i], sysm) for i in range(len(U))])
        return fixrep(V, d["u0"])

    rows = []

    def add(label, c, sh, so, U, lps):
        hard = (sh < 1e-6) & (so < 1e-6)
        r = uc_metrics(c, sh, so, d, U, f"{name}: {label}", LPs_per_instance=lps)
        # label-free selection criterion: cost relative to the LP relaxation on served instances
        g_rel = (c - d["c_rel"]) / d["c_rel"] * 100
        r["gap_to_relax_mean_served_%"] = float(g_rel[hard].mean()) if hard.any() else float("nan")
        r["split"] = split
        rows.append(r)
        if arrays is not None:
            arrays[r["method"]] = np.stack([c, sh, so])
        print(f"  [{split}] {r['method']:70s} served {r['no_shed_no_shortfall_%']:5.1f}%  median {r['gap_median_%']:7.3f}%  "
              f"served-mean {r['gap_mean_served_%']:6.3f}%  mean {r['gap_mean_%']:8.2f}%  units {r['units_on']:.2f}  "
              f"LPs {lps:.1f}", flush=True)

    U = fixrep((p > 0.5).astype(np.int8), d["u0"])
    c, sh, so = oracle.scores(d, np.arange(n), U, keys)
    add("top-1 -> LP", c, sh, so, U, 1.0)
    U = adequate((p > 0.5).astype(np.int8))
    c, sh, so = oracle.scores(d, np.arange(n), U, keys)
    add("top-1 + adequacy repair -> LP", c, sh, so, U, 1.0)
    rb = make_repair("blocks", d, sysm, rep)
    U = np.array([rb((p[i] > 0.5).astype(np.int8), i) for i in range(n)])
    c, sh, so = oracle.scores(d, np.arange(n), U, keys)
    add("top-1 + block adequacy repair -> LP", c, sh, so, U, 1.0)
    if screening:
        cl = candidates_from_probs(p, 8, np.random.default_rng(seed))
        flat_i = np.concatenate([np.full(len(x), i) for i, x in enumerate(cl)])
        flat_u = fixrep(np.concatenate(cl), d["u0"][flat_i])
        c, sh, so = oracle.scores(d, flat_i, flat_u, keys[flat_i])
        best = np.array([np.where(flat_i == i)[0][np.argmin(c[flat_i == i])] for i in range(n)])
        add("candidate screening -> LP", c[best], sh[best], so[best], flat_u[best], float(np.mean([len(x) for x in cl])))
    return rows


def load_model1(path, sysm, tr_feat, T=12):
    feat = UCFeaturizer(sysm, tr_feat, relax=True, sym=True)
    m1 = build_uc_model1(sysm, feat, T, "gnn", seed=0)
    m1.net.load_state_dict(torch.load(path))
    return m1


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


MODEL_PATHS = {
    "lf_bce": os.path.join(OUT, "constrained_lf_bce.pt"),
    "milp_bce": os.path.join(OUT, "uc_model1_4.pt"),
    "milp_rl": os.path.join(OUT, "uc_model1_rl.pt"),
}


def model_path(name):
    return MODEL_PATHS.get(name, os.path.join(OUT, f"constrained_{name}.pt"))


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def model_probs(name, d, sysm, tr, tr_full, T=12):
    """commitment probabilities of a saved Model 1, or of a BC predictor + repair policy"""
    obj = torch.load(model_path(name))
    if isinstance(obj, dict) and "pol" in obj:
        bc = load_model1(model_path(obj["init"]), sysm, tr, T)
        env = RepairEnv(sysm, tr_full, k=obj["k"])
        pol = RepairPolicy(obj["d_in"])
        pol.load_state_dict(obj["pol"])
        return repair_probs(pol, env, d, bc.predict(d))
    return load_model1(model_path(name), sysm, tr, T).predict(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bce", "rl", "lag", "he", "eval", "test", "calib"])
    ap.add_argument("--name", default="")
    ap.add_argument("--init", default="lf_bce", help="lf_bce | milp_bce")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--lam_lr", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=1, help="-1 = whole horizon (no per-hour credit)")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--muprop", type=int, default=1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam_max", type=float, default=1e3)
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--repair", default="mud", choices=["mud", "blocks"],
                    help="repair in the training loop: min up/down only, or block adequacy + min up/down")
    ap.add_argument("--k_repair", type=int, default=48)
    ap.add_argument("--he_lr", type=float, default=1e-3)
    ap.add_argument("--models", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=0)
    ap.add_argument("--no_screen", action="store_true")
    ap.add_argument("--screen_models", default="", help="test stage: only these models get candidate screening")
    a = ap.parse_args()
    cfg, tr_full, va_full, te, sysm, rep = setup()
    T = cfg["T"]
    tr, va = strip(tr_full), strip(va_full)
    va_eval = va_full if not a.n_val else {k: v[:a.n_val] for k, v in va_full.items()}
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(a.seed)

    if a.stage == "bce":
        # label-free predictor of scripts/uc_label_free.py (same labels, features, epochs, seed)
        fixrep = lambda U, U0: np.array([rep(U[i], U0[i]) for i in range(len(U))])
        adequate = lambda U, d: fixrep(np.array([adequacy_repair(U[i], d["load"][i], d["avail"][i], d["sr"][i], sysm)
                                                 for i in range(len(U))]), d["u0"])
        y_tr = adequate((tr["u_rel"] > 0.5).astype(np.int8), tr)
        va["u_target"] = adequate((va["u_rel"] > 0.5).astype(np.int8), va)
        feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
        m1 = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
        t0 = time.time()
        train_uc_bce(m1, tr, va, y_tr, epochs=80, seed=a.seed)
        torch.save(m1.net.state_dict(), MODEL_PATHS["lf_bce"])
        oracle = HourlyOracle(cfg, a.workers)
        rows = evaluate_probs("LF-BCE", m1.predict(va_eval), va_eval, "val", oracle, sysm, rep)
        dump(os.path.join(OUT, "constrained_val_lf_bce.json"), {"rows": rows, "train_s": time.time() - t0})
        oracle.close()

    elif a.stage in ("rl", "lag"):
        m1 = load_model1(model_path(a.init), sysm, tr, T)
        oracle = HourlyOracle(cfg, a.workers)
        rv = make_repair(a.repair, va_eval, sysm, rep)
        vstep = [0]

        def val_fn(m):
            """deployed decoder on val (label-free numbers) + checkpoint of this step"""
            vstep[0] += 30
            torch.save(m.net.state_dict(), os.path.join(OUT, f"constrained_{a.name}_s{vstep[0]}.pt"))
            p = m.predict(va_eval)
            U = np.array([rv((p[i] > 0.5).astype(np.int8), i) for i in range(len(p))])
            o = oracle.evaluate(va_eval, np.arange(len(p)), U, np.arange(len(p)) + KEY["val"])
            hard = (o["shed"].sum(1) < 1e-6) & (o["short"].sum(1) < 1e-6)
            g = (o["obj"] - va_eval["c_rel"]) / va_eval["c_rel"] * 100
            v = {"step": vstep[0], "served_%": float(hard.mean() * 100),
                 "gap_to_relax_served_%": float(g[hard].mean()) if hard.any() else float("nan"),
                 "mean_cost_over_relax": float(np.mean(o["obj"] / va_eval["c_rel"])), "units_on": float(U.sum((1, 2)).mean() / T)}
            print("    val (deployed decoder):", {k: round(x, 3) for k, x in v.items()}, flush=True)
            return v
        t0 = time.time()
        if a.stage == "rl":
            ref = tr["obj"] if a.init == "lf_bce" else tr_full["obj"]
            tr_use = tr if a.init == "lf_bce" else dict(tr, obj=tr_full["obj"])
            if a.repair == "mud":     # exactly the current method (otsl.ucml)
                _, hist = train_uc_reinforce(m1, tr_use, _ScoreAdapter(oracle), steps=a.steps, bs=a.bs, n_samples=6,
                                             lr=a.lr, seed=a.seed, repair=rep, log_every=10)
            else:
                _, hist = train_reinforce_plain(m1, tr_use, oracle, ref, make_repair(a.repair, tr_use, sysm, rep),
                                                steps=a.steps, bs=a.bs, n_samples=6, lr=a.lr, seed=a.seed,
                                                val_fn=val_fn, val_every=30)
            hp = dict(steps=a.steps, bs=a.bs, n_samples=6, lr=a.lr, init=a.init, repair=a.repair)
        else:
            ref = tr["c_rel"] if a.init == "lf_bce" else tr_full["obj"]

            hp = dict(steps=a.steps, bs=a.bs, n_samples=5, lr=a.lr, eps=a.eps, lam0=a.lam0, lam_lr=a.lam_lr,
                      window=None if a.window < 0 else a.window, kappa=a.kappa, tau=a.tau, muprop=bool(a.muprop),
                      beta=a.beta, init=a.init, lam_max=a.lam_max, kl=a.kl, repair=a.repair)
            _, hist = train_constrained(m1, tr, oracle, ref, make_repair(a.repair, tr, sysm, rep), steps=a.steps, bs=a.bs, n_samples=5, lr=a.lr,
                                        seed=a.seed, eps=a.eps, lam0=a.lam0, lam_lr=a.lam_lr,
                                        window=hp["window"], kappa=a.kappa, tau=a.tau, muprop=bool(a.muprop),
                                        beta=a.beta, val_fn=val_fn, val_every=30, lam_max=a.lam_max, kl=a.kl)
        train_s, train_lps = time.time() - t0, oracle.n
        torch.save(m1.net.state_dict(), model_path(a.name))
        rows = evaluate_probs(a.name, m1.predict(va_eval), va_eval, "val", oracle, sysm, rep)
        dump(os.path.join(OUT, f"constrained_val_{a.name}.json"),
             {"rows": rows, "hp": hp, "train_s": train_s, "train_LPs": train_lps, "hist": hist})
        oracle.close()

    elif a.stage == "he":
        # He et al. 2026-style: behaviour cloning of MILP commitments (the MILP-label BCE model) + PPO repair
        # of the k least reliable decisions with LP cost / feasibility reward. Reference cost: MILP cost
        # (the BC stage needs MILP labels anyway).
        bc = load_model1(model_path(a.init), sysm, tr, T)
        p_tr, p_va = bc.predict(tr_full), bc.predict(va_eval)
        env = RepairEnv(sysm, tr_full, k=a.k_repair)
        _, x0, _ = env.build(tr_full, np.arange(2), p_tr[:2])
        pol = RepairPolicy(x0.shape[-1])
        oracle = HourlyOracle(cfg, a.workers)
        ref = tr_full["obj"] if a.init == "milp_bce" else tr["c_rel"]
        t0 = time.time()
        _, hist = train_repair_ppo(pol, env, p_tr, tr_full, oracle, ref, rep, steps=a.steps, bs=a.bs, n_samples=6,
                                   lr=a.he_lr, seed=a.seed)
        train_s, train_lps = time.time() - t0, oracle.n
        torch.save({"pol": pol.state_dict(), "k": a.k_repair, "init": a.init, "d_in": x0.shape[-1]}, model_path(a.name))
        rows = evaluate_probs(f"{a.name} (BC only)", p_va, va_eval, "val", oracle, sysm, rep, screening=False)
        rows += evaluate_probs(a.name, repair_probs(pol, env, va_eval, p_va), va_eval, "val", oracle, sysm, rep)
        dump(os.path.join(OUT, f"constrained_val_{a.name}.json"),
             {"rows": rows, "hp": dict(k=a.k_repair, lr=a.he_lr, steps=a.steps, bs=a.bs, n_samples=6, init=a.init),
              "train_s": train_s, "train_LPs": train_lps, "hist": hist})
        oracle.close()

    elif a.stage == "test":
        oracle = HourlyOracle(cfg, a.workers)
        rows, arrays = [], {}
        n_te = len(te["load"])
        tag = a.name or "all"

        def save_test():
            dump(os.path.join(OUT, f"constrained_test_{tag}.json"), {"rows": rows})
            np.savez_compressed(os.path.join(OUT, f"constrained_test_{tag}_arrays.npz"),
                                **{k.replace("/", "_"): v for k, v in arrays.items()}, names=np.array(list(arrays.keys())))
        for name in [x for x in a.models.split(",") if x]:
            save_test()
            t0 = time.time()
            if name == "milp":     # reference: the MILP's own commitment through the same LP
                c, sh, so = oracle.scores(te, np.arange(n_te), te["u"], KEY["test"] + np.arange(n_te))
                r = uc_metrics(c, sh, so, te, te["u"], "MILP commitment (reference)", LPs_per_instance=0.0)
                hard = (sh < 1e-6) & (so < 1e-6)
                r["gap_to_relax_mean_served_%"] = float(((c - te["c_rel"]) / te["c_rel"] * 100)[hard].mean())
                r["split"] = "test"
                rows.append(r)
                arrays[r["method"]] = np.stack([c, sh, so])
                continue
            if name == "knn20":
                cl = knn_candidates(tr_full, te, 20)
                flat_i = np.concatenate([np.full(len(x), i) for i, x in enumerate(cl)])
                flat_u = np.concatenate(cl)
                flat_u = np.array([rep(flat_u[j], te["u0"][flat_i[j]]) for j in range(len(flat_u))])
                c, sh, so = oracle.scores(te, flat_i, flat_u, KEY["test"] + flat_i)
                best = np.array([np.where(flat_i == i)[0][np.argmin(c[flat_i == i])] for i in range(n_te)])
                r = uc_metrics(c[best], sh[best], so[best], te, flat_u[best], "kNN-20 + LP check", LPs_per_instance=20.0)
                hard = (sh[best] < 1e-6) & (so[best] < 1e-6)
                g_rel = (c[best] - te["c_rel"]) / te["c_rel"] * 100
                r["gap_to_relax_mean_served_%"] = float(g_rel[hard].mean())
                r["split"] = "test"
                rows.append(r)
                arrays[r["method"]] = np.stack([c[best], sh[best], so[best]])
                print(f"  [test] kNN-20 served {r['no_shed_no_shortfall_%']:.1f}% median {r['gap_median_%']:.3f}%", flush=True)
                continue
            p = model_probs(name, te, sysm, tr, tr_full, T)
            scr = not a.screen_models or name in a.screen_models.split(",")
            rows += evaluate_probs(name, p, te, "test", oracle, sysm, rep, arrays=arrays, screening=scr)
            cal_p = os.path.join(OUT, "constrained_calib.json")
            cal = json.load(open(cal_p)) if os.path.exists(cal_p) else {}
            for dec, lab in (("mud", ""), ("blocks", " + block adequacy repair")):
                if name not in cal or cal[name][dec]["threshold"] == 0.5:
                    continue
                th = cal[name][dec]["threshold"]
                rr = make_repair(dec, te, sysm, rep)
                U = np.array([rr((p[i] > th).astype(np.int8), i) for i in range(n_te)])
                c, sh, so = oracle.scores(te, np.arange(n_te), U, KEY["test"] + np.arange(n_te))
                r = uc_metrics(c, sh, so, te, U, f"{name}: threshold {th} (val-calibrated){lab} -> LP", LPs_per_instance=1.0)
                hard = (sh < 1e-6) & (so < 1e-6)
                r["gap_to_relax_mean_served_%"] = float(((c - te["c_rel"]) / te["c_rel"] * 100)[hard].mean()) if hard.any() else float("nan")
                r["split"] = "test"
                rows.append(r)
                arrays[r["method"]] = np.stack([c, sh, so])
                print(f"  [test] {r['method']:70s} served {r['no_shed_no_shortfall_%']:5.1f}%  median {r['gap_median_%']:7.3f}%  "
                      f"served-mean {r['gap_mean_served_%']:6.3f}%", flush=True)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
        save_test()
        oracle.close()

    elif a.stage == "calib":
        # decoding threshold per model, chosen on val by the label-free expected cost:
        # mean over instances of (LP cost incl. shedding / reserve penalties) / (LP relaxation cost)
        oracle = HourlyOracle(cfg, a.workers)
        path = os.path.join(OUT, "constrained_calib.json")
        out = json.load(open(path)) if os.path.exists(path) else {}
        n = len(va_eval["load"])
        for name in [x for x in a.models.split(",") if x]:
            p = model_probs(name, va_eval, sysm, tr, tr_full, T)
            out[name] = {}
            for dec in ("mud", "blocks"):
                rr = make_repair(dec, va_eval, sysm, rep)
                res = {}
                for th in THRESHOLDS:
                    U = np.array([rr((p[i] > th).astype(np.int8), i) for i in range(n)])
                    c, sh, so = oracle.scores(va_eval, np.arange(n), U, np.arange(n) + KEY["val"])
                    hard = (sh < 1e-6) & (so < 1e-6)
                    res[str(th)] = {"mean_cost_over_relax": float(np.mean(c / va_eval["c_rel"])),
                                    "served_%": float(hard.mean() * 100),
                                    "gap_to_relax_served_%": float(((c / va_eval["c_rel"] - 1) * 100)[hard].mean()) if hard.any() else None}
                best = min(res, key=lambda k: res[k]["mean_cost_over_relax"])
                out[name][dec] = {"threshold": float(best), "val": res}
                print(name, dec, "threshold", best, {k: (round(v["served_%"], 1), round(v["mean_cost_over_relax"], 4)) for k, v in res.items()}, flush=True)
        dump(path, out)
        oracle.close()

    elif a.stage == "eval":   # val evaluation of saved models
        oracle = HourlyOracle(cfg, a.workers)
        rows = []
        for name in [x for x in a.models.split(",") if x]:
            rows += evaluate_probs(name, model_probs(name, va_eval, sysm, tr, tr_full, T), va_eval, "val", oracle, sysm,
                                   rep, screening=not a.no_screen)
        dump(os.path.join(OUT, f"constrained_val_eval_{a.name or 'misc'}.json"), {"rows": rows})
        oracle.close()
