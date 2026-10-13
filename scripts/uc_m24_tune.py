"""m24 study: guard-aware Learning to Fix tuning (otsl.m24.M24Tuner) of the temporal-GNN ensemble's probabilities on
the 40 uc24 validation instances with full MILPs (data/generated/uc24/uc24ltf_val.npz).

    python3 scripts/uc_m24_tune.py --run hg_fin --src fin_ens [--eps 0.01] [--budget_min 120]

Settings of the uc24ltf runs (Q = 20, K_max = 10, relaxation MILPs 8 s, check MILPs 60 s with the (1 + eps) C* cut-off,
warm-start candidates of StartLtFTuner with 15 s); C* = the validation MILP objective (300 s, 0.1 %). Guards inside the
check, the cuts and the verification: adequacy guard + minimal min up/down conflict release (otsl.m24.guard_chain).
Output: results/uc24/m24_tune_<run>.json / .log / _cuts.jsonl. A converged run is skipped on rerun.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import load_b3  # noqa: E402
from otsl.ltfx import fix_masks  # noqa: E402
from otsl.m24 import M24Tuner  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402

U24, DATA, RES = "data/generated/uc24", "data/generated/uc24_m24", "results/uc24"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src", required=True, help="probability source: data/generated/uc24_m24/probs/<src>_probs.npz")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--budget_min", type=float, default=120.0)
    ap.add_argument("--relax_tl", type=float, default=8.0)
    ap.add_argument("--check_tl", type=float, default=60.0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    out = os.path.join(RES, f"m24_tune_{a.run}.json")
    if os.path.exists(out) and json.load(open(out))["converged"]:
        print(f"{a.run}: already converged, skipped", flush=True)
        sys.exit(0)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=24, network=True)
    d = load_b3(os.path.join(U24, "uc24ltf_val.npz"))
    pi = np.load(os.path.join(DATA, "probs", f"{a.src}_probs.npz"))["val"].astype(np.float64)
    assert len(pi) == len(d["obj"]) == 40
    logf = open(os.path.join(RES, f"m24_tune_{a.run}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    log(f"m24 tuning {a.run}: src {a.src} eps={a.eps} guards adeq+rowmin n_val=40 relax_tl={a.relax_tl} "
        f"check_tl={a.check_tl}")
    t0, c0 = time.time(), time.process_time()
    tu = M24Tuner(m, sysm, d, pi, a.eps, K_max=10, Q=20, check_tl=a.check_tl, relax_tl=a.relax_tl, master_tl=120.0,
                  log=log, time_budget=a.budget_min * 60, start_tl=15.0)
    lo, hi = tu.run()
    conv = tu.history[-1].get("result") == "all instances feasible"
    ver = tu.verify(lo, hi)
    off, on = fix_masks(pi, lo, hi)
    po, pn = tu.guarded_masks(lo, hi)
    res = dict(run=a.run, src=a.src, eps=a.eps, guards=["adeq", "rowmin"], converged=conv, lo=lo.tolist(), hi=hi.tolist(),
               n_val=40, val_fixed_share_pre_guard=float((off | on).mean()), val_fixed_share=float((po | pn).mean()),
               val_fixed_off=float(po.mean()), val_fixed_on=float(pn.mean()), n_iter=len(tu.history),
               n_cut_sets=len(tu.master.cuts), verify_rel_increase=ver.tolist(), verify_max=float(np.max(ver)),
               verify_ok=bool(np.all(ver <= a.eps * (1 + 1e-6))), n_verify_fail=int((ver > a.eps * (1 + 1e-6)).sum()),
               stats={k: v for k, v in tu.stats.items()}, history=tu.history, wall_s=time.time() - t0,
               cpu_s=time.process_time() - c0,
               cfg=dict(K_max=10, Q=20, relax_tl=a.relax_tl, check_tl=a.check_tl, budget_s=a.budget_min * 60, start_tl=15.0))
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% after guards "
        f"({res['val_fixed_share_pre_guard'] * 100:.2f}% before), verify max {res['verify_max'] * 100:.3f}% "
        f"({res['n_verify_fail']} above eps), wall {res['wall_s']:.0f}s")
    print(f"{a.run}: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% "
          f"(pre {res['val_fixed_share_pre_guard'] * 100:.2f}%) wall {res['wall_s']:.0f}s", flush=True)
