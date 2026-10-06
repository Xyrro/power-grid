"""No-learning baselines: Learning to Fix's joint threshold tuning (Algorithm 1 + 2, otsl.ltfx) with the LP-relaxation
values as the "probabilities" (learning-free Learning to Fix), on the validation instances of uc12 or uc24.

    python3 scripts/uc_base_tune.py --bench uc12 --eps 0.01          # 180 val instances, settings of ltfx_tune_knn_1
    python3 scripts/uc_base_tune.py --bench uc24 --eps 0.01          # 40 val instances, settings of uc24ltf_tune_knn_1

pi = u_rel (the LP relaxation's commitment values of each validation instance; no model, no training data). Tuner and
settings are those of the faithful kNN runs it is compared with: uc12 otsl.ltfx.LtFTuner (Q = 20, K_max = 10, check
MILPs 60 s, relaxation MILPs 6 s, 180 instances = val + val_extra); uc24 otsl.uc24ltf.StartLtFTuner (relaxation MILPs
8 s, 40 instances of uc24ltf_val). Budget: --budget_min (a run that hits it stops and is marked not converged).
Output: results/<bench>/base_tune_lp_<eps%>.{json,log}.
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
from otsl.ltfx import LtFTuner, fix_masks  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.uc24ltf import StartLtFTuner  # noqa: E402

FIELDS = ("load", "avail", "u0", "sr", "obj", "u", "day", "start", "gap", "time", "u_rel")
CFG = {"uc12": dict(T=12, parts=["data/generated/uc12/val.npz", "data/generated/uc12/val_extra.npz"], n_val=180,
                    relax_tl=6.0, tuner="LtFTuner"),
       "uc24": dict(T=24, parts=["data/generated/uc24/uc24ltf_val.npz"], n_val=40, relax_tl=8.0, tuner="StartLtFTuner")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=["uc12", "uc24"])
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--budget_min", type=float, default=75.0)
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--Q", type=int, default=20)
    ap.add_argument("--check_tl", type=float, default=60.0)
    ap.add_argument("--master_tl", type=float, default=120.0)
    ap.add_argument("--n_val", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    cfg = dict(CFG[a.bench])
    n_val = a.n_val or cfg["n_val"]
    parts = [load_b3(p) for p in cfg["parts"]]
    d = {k: np.concatenate([p[k] for p in parts])[:n_val] for k in FIELDS}
    pi = d["u_rel"].astype(np.float64)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=cfg["T"], network=True)
    tag = f"base_tune_lp_{a.eps * 100:g}"
    out = os.path.join("results", a.bench, tag)
    logf = open(out + ".log", "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    run_cfg = dict(bench=a.bench, n_val=len(pi), K_max=a.K_max, Q=a.Q, relax_tl=cfg["relax_tl"], check_tl=a.check_tl,
                   master_tl=a.master_tl, budget_s=a.budget_min * 60, tuner=cfg["tuner"], parts=cfg["parts"])
    log(f"learning-free LtF (pi = LP relaxation) {json.dumps(run_cfg)} eps={a.eps}")
    t_cpu0, t0 = time.process_time(), time.time()
    cls = LtFTuner if cfg["tuner"] == "LtFTuner" else StartLtFTuner
    tu = cls(m, sysm, d, pi, a.eps, K_max=a.K_max, Q=a.Q, check_tl=a.check_tl, relax_tl=cfg["relax_tl"],
             master_tl=a.master_tl, log=log, time_budget=a.budget_min * 60)
    lo, hi = tu.run()
    off, on = fix_masks(pi, lo, hi)
    conv = tu.history[-1].get("result") == "all instances feasible"
    ver = tu.verify(lo, hi)
    log(f"verification: max rel. cost increase {np.max(ver) * 100:.4f}% (eps {a.eps * 100:g}%), instances above eps: "
        f"{int((ver > a.eps * (1 + 1e-6)).sum())}")
    res = dict(model="lprelax", eps=a.eps, converged=conv, lo=lo.tolist(), hi=hi.tolist(), n_val=len(pi),
               val_fixed_share=float((off | on).mean()), val_fixed_off=float(off.mean()), val_fixed_on=float(on.mean()),
               n_iter=len(tu.history), n_cut_sets=len(tu.master.cuts), verify_rel_increase=ver.tolist(),
               verify_max=float(np.max(ver)), verify_ok=bool(np.all(ver <= a.eps * (1 + 1e-6))),
               n_verify_fail=int((ver > a.eps * (1 + 1e-6)).sum()), stats=tu.stats, history=tu.history, cfg=run_cfg,
               wall_s=time.time() - t0, cpu_s=time.process_time() - t_cpu0,
               failed_instances=sorted({h["failed"] for h in tu.history if "failed" in h}))
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% wall {res['wall_s']:.0f}s "
        f"cpu {res['cpu_s']:.0f}s")
    print(f"done {tag}: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% wall {res['wall_s']:.0f}s", flush=True)
