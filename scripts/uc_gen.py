"""Generate unit-commitment datasets on RTS-GMLC.

    python scripts/uc_gen.py --cfg uc1 --split train --n 4000 --seed 1

Days of 2020 are split by calendar: every 5th day is held out for testing (seasonal coverage, no
leakage of a test day's profile into training); validation uses every 5th day offset by 2.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.ucdata import generate, save  # noqa: E402

UC_CONFIGS = {
    # B1: single-period network-constrained UC (the framework's PD -> on/off -> LP -> PG, VA)
    "uc1": {"T": 1, "network": True, "time_limit": 30, "mip_gap": 1e-5},
    # B2: 12-hour look-ahead UC with min up/down, ramping, start-up costs
    "uc12": {"T": 12, "network": True, "time_limit": 60, "mip_gap": 1e-3},
    "uc1_smoke": {"T": 1, "network": True, "time_limit": 30, "mip_gap": 1e-5},
}
DAYS = np.arange(366)
SPLITS = {"test": DAYS[DAYS % 5 == 0], "val": DAYS[DAYS % 5 == 2],
          "train": DAYS[(DAYS % 5 != 0) & (DAYS % 5 != 2)]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="uc1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_alt", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = dict(UC_CONFIGS[a.cfg], n_alt=a.n_alt)
    t0 = time.time()
    data = generate(cfg, SPLITS[a.split.split("_")[0]], a.n, a.seed, a.workers)
    out = os.path.join("data", "generated", a.cfg, f"{a.split}.npz")
    save(out, data)
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump({"cfg": cfg, "n": int(len(data["obj"])), "wall_s": time.time() - t0,
                   "milp_time_mean": float(data["time"].mean()), "milp_opt_frac": float(data["opt"].mean()),
                   "relax_gap_%": float(((data["obj"] - data["c_rel"]) / data["obj"]).mean() * 100),
                   "persistence_gap_%": float(((data["c_u0"] - data["obj"]) / data["obj"]).mean() * 100),
                   "round_gap_%": float(((data["c_round"] - data["obj"]) / data["obj"]).mean() * 100)}, f, indent=1)
    print("saved", out, f"{time.time() - t0:.0f}s")
