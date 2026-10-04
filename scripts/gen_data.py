"""Generate DC-OTS datasets.

    python scripts/gen_data.py --cfg case118 --split train --n 1500 --seed 1
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import generate, save  # noqa: E402

_118 = {"case": "case118_ieee", "line_scale": 1.0, "budget": 3, "time_limit": 30, "mip_gap": 1e-4}
_30 = {"case": "case30_ieee", "line_scale": 1.0, "budget": 3, "time_limit": 30, "mip_gap": 1e-5}
CONFIGS = {
    # IEEE 118 (PGLib typical operations), nominal ratings, switching budget 3, per-switch cost
    # of 0.02% of the nominal OPF cost ("parsimonious" labels: only switch when it pays)
    "case118": dict(_118, switch_cost_rel=2e-4),
    "case118_raw": dict(_118, switch_cost_rel=0.0),   # the plain DC-OTS labels of the original framework
    # IEEE 30, nominal ratings: large switching benefit, tiny MILPs (fast secondary benchmark)
    "case30": dict(_30, switch_cost_rel=2e-4),
    "case30_raw": dict(_30, switch_cost_rel=0.0),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_alt", type=int, default=0, help="alternative optima to enumerate (label ambiguity)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = dict(CONFIGS[a.cfg], n_alt=a.n_alt)
    t0 = time.time()
    data = generate(cfg, a.n, a.seed, a.workers)
    out = os.path.join("data", "generated", a.cfg, f"{a.split}.npz")
    save(out, data)
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump({"cfg": cfg, "n_requested": a.n, "n_kept": int(len(data["pd"])), "seed": a.seed,
                   "wall_s": time.time() - t0, "milp_time_mean": float(data["ots_time"].mean()),
                   "milp_optimal_frac": float(data["ots_opt"].mean()),
                   "benefit_mean_%": float(((data["c0"] - data["c_ots"]) / data["c0"]).mean() * 100)}, f, indent=1)
    print(f"saved {out} in {time.time() - t0:.0f}s")
