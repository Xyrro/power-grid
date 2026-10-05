"""B3 (24-hour UC): label-free training of Model 1 and end-to-end evaluation.

    python scripts/uc_b3_train.py --stage bce,rl,select,test

No full MILP is used for training:
* LF-BCE      imitate the repaired LP relaxation (rounded, per-hour adequacy repair, min up/down repair; one LP
              relaxation per label, computed by scripts/uc_b3_gen.py); early stopping on val BCE against the same
              label-free targets.
* REINFORCE   fine-tune with the exact dispatch LP as critic (RLOO baseline; reward -log(LP cost / LP-relaxation
              cost)); min up/down repair of every sample. Checkpoints every --val_every steps.
* select      (val, full-MILP reference) pre-declared rules:
                checkpoint   lowest mean val gap of "top-1 + block repair -> LP";
                threshold    for each model, the decoding threshold in {0.3, ..., 0.9} with the lowest mean val gap
                             with block repair.
* test        end-to-end on test: one LP with block repair (val threshold), and candidate screening.
Results: results/uc24/b3_train_*.json, b3_e2e_results.{md,json}; weights results/uc24/b3_*.pt.
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
from otsl.b3 import load_b3, rep_block, rep_minud, subset  # noqa: E402
from otsl.constrained import HourlyOracle, train_reinforce_plain  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, train_uc_bce, uc_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model1 import fmt_table  # noqa: E402
from uc_model1 import candidates_from_probs  # noqa: E402

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MILP_KEYS = ("u", "obj", "time", "cpu", "gap", "bound", "opt", "nodes", "shed", "short", "obj_lp", "inc_t", "inc_obj",
             "n_inc")
OUT = os.path.join("results", "uc24")


def strip(d):
    """remove every full-MILP field; the reference cost for REINFORCE logs is the LP relaxation"""
    return dict({k: v for k, v in d.items() if k not in MILP_KEYS}, obj=d["c_rel"].copy())


class Evaluator:
    """decode probabilities into commitments and price them with the dispatch LP (HourlyOracle, memoised)"""

    def __init__(self, oracle, sysm, d, key_offset):
        self.o, self.s, self.d, self.k = oracle, sysm, d, key_offset
        self.n = len(d["load"])

    def decode(self, p, th=0.5, block=True):
        U = (p > th).astype(np.int8)
        if block:
            return np.array([rep_block(U[i], self.d, i, self.s) for i in range(self.n)])
        return np.array([rep_minud(U[i], self.d["u0"][i], self.s) for i in range(self.n)])

    def price(self, U, idx=None):
        idx = np.arange(self.n) if idx is None else idx
        return self.o.scores(self.d, idx, U, idx + self.k)

    def metrics(self, U, label, **extra):
        c, sh, so = self.price(U)
        return uc_metrics(c, sh, so, self.d, U if "u" in self.d else None, label, **extra), c

    def screening(self, p, rng, n_samples=8):
        cl = candidates_from_probs(p, n_samples, rng)
        flat_i = np.concatenate([np.full(len(x), i) for i, x in enumerate(cl)])
        cand = np.concatenate(cl)
        flat_u = np.array([rep_block(cand[j], self.d, flat_i[j], self.s) for j in range(len(cand))])
        c, sh, so = self.o.scores(self.d, flat_i, flat_u, flat_i + self.k)
        best = np.array([np.where(flat_i == i)[0][np.argmin(c[flat_i == i])] for i in range(self.n)])
        return flat_u[best], float(np.mean([len(x) for x in cl]))


def row_short(r):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
            if k in ("method", "no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%", "units_on")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="bce,rl,select,test")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rl_steps", type=int, default=100)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--val_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=0, help="debug: first n training instances")
    ap.add_argument("--n_val", type=int, default=0, help="debug: first n val instances")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cfg", default="uc24")
    ap.add_argument("--rl_init", default="", help="continue REINFORCE from these weights (default: the LF-BCE model)")
    a = ap.parse_args()
    stages = a.stage.split(",")
    root = os.path.join("data", "generated", a.cfg)
    if a.cfg != "uc24":
        OUT = os.path.join("results", "uc24", "b3_debug")
    os.makedirs(OUT, exist_ok=True)
    tr = load_b3(os.path.join(root, "train.npz"))
    if a.n_train:
        tr = subset(tr, np.arange(a.n_train))
    sysm = load_rts_gmlc()
    T = tr["load"].shape[1]
    cfg = {"T": T, "network": True}
    tr = strip(tr)
    assert np.isnan(tr.get("time", np.array([np.nan]))).all()
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    tag = a.tag
    log = {}
    logp = os.path.join(OUT, f"b3_train{tag}.json")
    if os.path.exists(logp):
        log = json.load(open(logp))

    def save_log():
        json.dump(log, open(logp, "w"), indent=1, default=float)

    va_full = None
    if os.path.exists(os.path.join(root, "val.npz")):
        va_full = load_b3(os.path.join(root, "val.npz"))
        if a.n_val:
            va_full = subset(va_full, np.arange(a.n_val))
    va_lf = strip(va_full) if va_full is not None else strip(subset(load_b3(os.path.join(root, "train.npz")), np.arange(20)))
    va_lf["u_target"] = va_lf["y_lf"]

    # --------------------------------------------------------------- LF-BCE (imitation of the repaired relaxation)
    # two label-free targets: y_lf (per-hour adequacy + min up/down repair, as in B2) and y_blk (block adequacy
    # repair); both are trained and the one with the lower mean val gap (top-1 + block repair -> LP) is kept.
    oracle = HourlyOracle(cfg, a.workers)
    # label-free selection: val commitments are priced against the val LP-relaxation bound (obj := c_rel), so no
    # full MILP is used for training or selection
    ev_va = Evaluator(oracle, sysm, strip(va_full), 7 * 10 ** 7) if va_full is not None else None
    m_bce = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
    p_bce = os.path.join(OUT, f"b3_lf_bce{tag}.pt")
    if "bce" in stages:
        log["label_s_mean"] = float(tr["t_label"].mean())
        log["relax_s_mean"] = float(tr["t_rel"].mean())
        cands = {}
        for target in ("y_lf", "y_blk"):
            m_ = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
            va_lf["u_target"] = va_lf[target]
            t0 = time.time()
            train_uc_bce(m_, tr, va_lf, tr[target], epochs=a.epochs, seed=a.seed)
            log[f"bce_{target}_train_s"] = time.time() - t0
            torch.save(m_.net.state_dict(), os.path.join(OUT, f"b3_lf_bce_{target}{tag}.pt"))
            r, _ = ev_va.metrics(ev_va.decode(m_.predict(va_full), 0.5, True), f"val LF-BCE {target} top-1 + block")
            log[f"bce_{target}_val"] = row_short(r)
            print("  ", target, row_short(r), flush=True)
            cands[target] = (r["gap_mean_%"], m_)
            save_log()
        best_t = min(cands, key=lambda k: cands[k][0])
        m_bce = cands[best_t][1]
        log["bce_target_selected"] = best_t
        log["lf_bce_train_s"] = log[f"bce_{best_t}_train_s"]
        torch.save(m_bce.net.state_dict(), p_bce)
        save_log()
        print("LF-BCE selected target", best_t, flush=True)
    else:
        if not os.path.exists(p_bce):          # tagged runs share the untagged imitation model
            p_bce = os.path.join(OUT, "b3_lf_bce.pt")
        m_bce.net.load_state_dict(torch.load(p_bce))

    def val_fn(m1):
        p = m1.predict(va_full)
        r, _ = ev_va.metrics(ev_va.decode(p, 0.5, True), "val top-1 + block")
        r0, _ = ev_va.metrics(ev_va.decode(p, 0.5, False), "val top-1")
        out = {"block": row_short(r), "plain": row_short(r0)}
        print("   [val]", out, flush=True)
        return out

    # --------------------------------------------------------------- REINFORCE with the LP critic
    p_rl = os.path.join(OUT, f"b3_lf_rl{tag}.pt")
    if "rl" in stages:
        m_rl = copy.deepcopy(m_bce)
        if a.rl_init:
            m_rl.net.load_state_dict(torch.load(a.rl_init))
            log["rl_init"] = a.rl_init
        ckpts = {}

        def val_ckpt(m1):
            out = val_fn(m1) if ev_va is not None else {}
            step = len(ckpts) * a.val_every + a.val_every
            torch.save(m1.net.state_dict(), os.path.join(OUT, f"b3_lf_rl{tag}_s{step}.pt"))
            ckpts[step] = out
            log["rl_ckpts"] = ckpts
            save_log()
            return out

        if ev_va is not None:
            log["rl_ckpt0"] = val_fn(m_bce)
        t0 = time.time()
        rep = lambda u, i: rep_minud(u, tr["u0"][i], sysm)
        _, hist = train_reinforce_plain(m_rl, tr, oracle, tr["c_rel"], rep, steps=a.rl_steps, bs=a.bs,
                                        n_samples=a.n_samples, seed=a.seed, log_every=5, val_fn=val_ckpt,
                                        val_every=a.val_every)
        log["rl_train_s"] = time.time() - t0
        log["rl_LPs"] = oracle.n
        log["rl_hist"] = hist
        torch.save(m_rl.net.state_dict(), p_rl)
        save_log()
        print("REINFORCE done", log["rl_train_s"], oracle.n, flush=True)

    # --------------------------------------------------------------- selection on val
    if "select" in stages:
        ck = log["rl_ckpts"]
        score = lambda r: r["block"]["gap_mean_%"]
        best = min(ck, key=lambda s: score(ck[s]))
        log["rl_selected_step"] = int(best)
        print("selected RL checkpoint step", best, ck[best], flush=True)
        m_sel = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
        m_sel.net.load_state_dict(torch.load(os.path.join(OUT, f"b3_lf_rl{tag}_s{best}.pt")))
        torch.save(m_sel.net.state_dict(), os.path.join(OUT, f"b3_rl_selected{tag}.pt"))
        sel = {}
        for name, m1 in (("lf_bce", m_bce), ("lf_rl", m_sel)):
            p = m1.predict(va_full)
            res = {}
            for th in THRESHOLDS:
                r, _ = ev_va.metrics(ev_va.decode(p, th, True), f"{name} th {th} + block")
                res[str(th)] = row_short(r)
            bth = min(res, key=lambda k: res[k]["gap_mean_%"])
            sel[name] = {"threshold": float(bth), "val": res}
            print(name, "threshold", bth, res[bth], flush=True)
        log["thresholds"] = sel
        save_log()

    # --------------------------------------------------------------- test (end-to-end)
    if "test" in stages:
        te = load_b3(os.path.join(root, "test.npz"))
        ev = Evaluator(oracle, sysm, te, 9 * 10 ** 7)
        m_sel = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
        m_sel.net.load_state_dict(torch.load(os.path.join(OUT, f"b3_rl_selected{tag}.pt")))
        rows, rng = [], np.random.default_rng(a.seed)
        n = len(te["load"])
        ok = np.ones(n, bool)
        rows.append(uc_metrics(te["obj"], te["shed"], te["short"], te, te["u"], "full MILP (reference)",
                               LPs_per_instance=0.0, time_s=float(te["time"].mean())))
        # heuristics without learning: repaired relaxation (the training label) and block-repaired relaxation
        for key, lab in (("y_lf", "rounded LP relaxation + adequacy + min up/down repair (training label)"),
                         ("y_blk", "rounded LP relaxation + block repair")):
            r, _ = ev.metrics(te[key], lab, LPs_per_instance=1.0)
            rows.append(r)
        t_rel = float(te["t_rel"].mean())
        for name, m1 in (("LF-BCE (imitation, no MILP)", m_bce), ("LF-BCE + REINFORCE (LP critic, no MILP)", m_sel)):
            key = "lf_bce" if m1 is m_bce else "lf_rl"
            t0 = time.time()
            p = m1.predict(te)
            t_inf = (time.time() - t0) / n
            np.save(os.path.join(OUT, f"b3_probs_test_{key}{tag}.npy"), p)
            th = log["thresholds"][key]["threshold"]
            for lab, U in ((f"{name}: top-1 -> LP", ev.decode(p, 0.5, False)),
                           (f"{name}: top-1 + block repair -> LP", ev.decode(p, 0.5, True)),
                           (f"{name}: threshold {th} (val) + block repair -> LP", ev.decode(p, th, True))):
                t0 = time.time()
                r, _ = ev.metrics(U, lab, LPs_per_instance=1.0)
                rows.append(r)
                print("  ", row_short(r), flush=True)
            Ub, nl = ev.screening(p, rng)
            r, _ = ev.metrics(Ub, f"{name}: candidate screening (block repair) -> LP", LPs_per_instance=nl)
            rows.append(r)
            print("  ", row_short(r), flush=True)
            log.setdefault("inference_s", {})[key] = t_inf
        cols = ["method", "no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%",
                "matches_or_beats_milp_%", "units_on", "LPs_per_instance"]
        md = fmt_table(rows, cols)
        print(md, flush=True)
        with open(os.path.join(OUT, f"b3_e2e_results{tag}.md"), "w") as f:
            f.write(f"{n} test instances (uc24, T=24); gaps vs the full MILP reference\n\n" + md)
        json.dump({"rows": rows, "t_rel_s": t_rel}, open(os.path.join(OUT, f"b3_e2e_results{tag}.json"), "w"),
                  indent=1, default=float)
        save_log()
    oracle.close()
