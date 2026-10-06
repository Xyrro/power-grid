"""Stacking follow-up, fused learned + LP rule (12 h): guard-aware Learning to Fix tuning (otsl.hybrid.HybridTuner,
adequacy + min up/down row guards inside the check, as the hybrid) on the hybrid's error-cost score with an LP veto
(otsl.base.lp_veto_score: decisions whose LP-relaxation value is integral and contradicts the learned rounding are
demoted to the least confident level of their direction).

    python3 scripts/uc_base_fuse.py --eps 0.01 --budget_min 60

Validation: val + val_extra (the first 180 instances, as the faithful runs); scores from results/uc12/hybrid_probs.npz
(BCE GNN seed 0 and its error-cost ensemble, normalisation of results/uc12/hybrid_probs.json) and the stored LP
relaxations. Settings of the hybrid runs: Q = 20, K_max = 10, check MILPs 60 s, relaxation MILPs 6 s.
Output: results/uc12/base_tune_fused_<eps%>.{json,log}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import load_b3  # noqa: E402
from otsl.base import lp_veto_score  # noqa: E402
from otsl.hybrid import HybridTuner, harm_to_score  # noqa: E402
from otsl.ltfx import fix_masks  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402

FIELDS = ("load", "avail", "u0", "sr", "obj", "u", "day", "start", "gap", "time", "u_rel")
GUARDS = ("adeq", "rows")


def fused_val(n_val=180):
    parts = [load_b3("data/generated/uc12/val.npz"), load_b3("data/generated/uc12/val_extra.npz")]
    d = {k: np.concatenate([p[k] for p in parts])[:n_val] for k in FIELDS}
    P = np.load("results/uc12/hybrid_probs.npz")
    mu, sd = json.load(open("results/uc12/hybrid_probs.json"))["harm_norm_logh_mean_sd"]["bce_s0"]
    p = np.concatenate([P["bce_s0_va"], P["bce_s0_vx"]])[:n_val].astype(np.float64)
    h = np.concatenate([P["harm_bce_s0_va"], P["harm_bce_s0_vx"]])[:n_val]
    s = harm_to_score(h, p, mu, sd)
    f = lp_veto_score(s, p, d["u_rel"])
    return d, f, s, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--n_val", type=int, default=180)
    ap.add_argument("--budget_min", type=float, default=60.0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    d, pi, s, p = fused_val(a.n_val)
    veto = np.abs(pi - s) > 1e-12
    out = os.path.join("results", "uc12", f"base_tune_fused_{a.eps * 100:g}")
    logf = open(out + ".log", "w")

    def log(x):
        logf.write(x + "\n")
        logf.flush()
    log(f"fused LP-veto score, guard-aware LtF eps={a.eps} n_val={len(pi)} guards={GUARDS}; vetoed decisions per "
        f"instance {veto.reshape(len(pi), -1).sum(1).mean():.1f} ({veto.mean() * 100:.2f} %)")
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    t0, c0 = time.time(), time.process_time()
    tu = HybridTuner(m, sysm, d, pi, a.eps, guards=GUARDS, cut_log=out + "_cuts.jsonl", K_max=10, Q=20, check_tl=60.0,
                     relax_tl=6.0, master_tl=120.0, log=log, time_budget=a.budget_min * 60)
    lo, hi = tu.run()
    conv = tu.history[-1].get("result") == "all instances feasible"
    ver = tu.verify(lo, hi)
    off, on = fix_masks(pi, lo, hi)
    po, pn = tu.guarded_masks(lo, hi)
    res = dict(score="harm_bce_s0 + LP veto", eps=a.eps, guards=list(GUARDS), converged=conv, lo=lo.tolist(), hi=hi.tolist(),
               n_val=len(pi), val_fixed_pre_guard=float((off | on).mean()), val_fixed=float((po | pn).mean()),
               val_fixed_off=float(po.mean()), val_fixed_on=float(pn.mean()), veto_share=float(veto.mean()),
               n_iter=len(tu.history), n_cut_sets=len(tu.master.cuts), verify_max=float(np.max(ver)),
               n_verify_fail=int((ver > a.eps * (1 + 1e-6)).sum()), stats=tu.stats, history=tu.history,
               wall_s=time.time() - t0, cpu_s=time.process_time() - c0)
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log(f"done: converged={conv} val fixed {res['val_fixed'] * 100:.2f}% (pre-guard {res['val_fixed_pre_guard'] * 100:.2f}%) "
        f"verify max {res['verify_max'] * 100:.3f}% wall {res['wall_s']:.0f}s")
    print(f"done: converged={conv} val fixed {res['val_fixed'] * 100:.2f}% wall {res['wall_s']:.0f}s", flush=True)
