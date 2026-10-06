"""Robustness study, step 1: shift specifications and instance lists (no MILPs here; the full MILP of every shifted
instance is solved in scripts/uc_ood_eval.py, back to back with the methods, on the same core).

    taskset -c 2 python3 scripts/uc_ood_gen.py

* Training coverage: what the training distribution spans (days, window starts, loads, renewables, initial status),
  so that each shift is out of distribution by construction; written to data/generated/uc12_ood/coverage_train.json.
* Line-outage diagnostic: lines whose single outage raises the LP relaxation cost by > 0.1 % on at least one of 12
  training instances (bridges excluded; written to specs.json as "line_pool"). A first design drew the outages from this
  pool; on the first instances it changed the relaxation cost by ~0.01 %, as did the most loaded lines, so the shift
  draws instead from the 4 lines whose single outage raises each instance's LP relaxation cost most (N-1 screening of
  its 20 most loaded lines; "line_top" / "line_screen", otsl.ood.with_loaded_line_outage).
* Instance lists: per shift, n jobs (day, window start, generator seed) drawn from the test calendar days
  (day % 5 == 0, the days of test / test_fresh; never a training or validation day) with a fixed seed per shift.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.ood import OODModel, critical_lines  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

ROOT, OUTD = "data/generated/uc12", "data/generated/uc12_ood"
DAYS = np.arange(366)
TEST_DAYS = DAYS[DAYS % 5 == 0]
# shift name -> (seed, make_ood_scenario keyword arguments, window starts)
SHIFTS = {
    "load_hi": (101, {"load_mult": 1.15}, (0, 12)),
    "load_lo": (102, {"load_mult": 0.85}, (0, 12)),
    "ren_hi": (103, {"vre_mult": 1.5}, (0, 12)),
    "gen_out": (104, {"n_gen_out": [2, 3]}, (0, 12)),
    "line_out": (105, {"n_line_out": [1, 2], "line_top": 4, "line_screen": 20}, (0, 12)),
    "night": (106, {}, (13, 23)),
}


def coverage(sysm, d):
    L = d["load"].sum(2) * 100
    rt = np.asarray(sysm.r_type)
    W = d["avail"][:, :, rt == "WIND"].sum(2) * 100
    S = d["avail"][:, :, np.isin(rt, ["PV", "RTPV"])].sum(2) * 100
    N = L - d["avail"].sum(2) * 100
    base = sysm.area_load[d["day"]].sum(2).max(1) * 100          # day peak of the calendar profile
    return {"n": int(len(L)), "n_days": int(np.unique(d["day"]).size), "day_mod5": sorted(set((d["day"] % 5).tolist())),
            "months": sorted(set(int(sysm.dates[x][1]) for x in np.unique(d["day"]))),
            "window_start": [int(d["start"].min()), int(d["start"].max())],
            "load_MW": [float(L.min()), float(L.max())], "net_load_MW": [float(N.min()), float(N.max())],
            "wind_MW_max": float(W.max()), "solar_MW_max": float(S.max()),
            "load_over_day_peak": [float((L.max(1) / base).min()), float((L.max(1) / base).max())],
            "u0_units_on": [int(d["u0"].sum(1).min()), int(d["u0"].sum(1).max())],
            "thermal_capacity_MW": float(sysm.pmax.sum() * 100),
            "vre_capacity_MW": float(sysm.r_pmax[np.isin(rt, ["WIND", "PV", "RTPV"])].sum() * 100)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n_line_inst", type=int, default=12)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    sysm = load_rts_gmlc()
    tr = load(os.path.join(ROOT, "train.npz"))
    cov = {"train": coverage(sysm, tr)}
    for sp in ("val", "val_extra", "val_extra2", "test_fresh"):
        cov[sp] = coverage(sysm, load(os.path.join(ROOT, f"{sp}.npz")))
    cov["note"] = ("training days are every day of 2020 with day % 5 in {1, 3, 4} (all 12 months); validation day % 5 == 2; "
                   "test day % 5 == 0. Window starts 0-12 in every split, so no 12-hour window crosses midnight.")
    json.dump(cov, open(os.path.join(OUTD, "coverage_train.json"), "w"), indent=1)
    print(json.dumps(cov["train"], indent=1), flush=True)
    # ---------------- line pool from training instances
    m = OODModel(sysm, T=12)
    rng = np.random.default_rng(7)
    idx = sorted(rng.choice(len(tr["load"]), a.n_line_inst, replace=False).tolist())
    pool, inc, base = critical_lines(sysm, m, [scenario_from(tr, i) for i in idx], thr=0.1)
    print(f"line pool ({len(pool)} of {sysm.L} lines): {pool} ({time.time() - t0:.0f}s)", flush=True)
    specs = {"days": "test calendar days (day % 5 == 0)", "n": a.n, "line_pool": pool,
             "line_pool_rule": f"single outage raises the LP relaxation cost by > 0.1 % on at least one of {a.n_line_inst} "
                               f"training instances {idx}; bridges (islanding) excluded",
             "line_max_increase_pct": {int(l): float(inc[:, l].max()) for l in pool}, "shifts": {}}
    for name, (seed, kw, (s0, s1)) in SHIFTS.items():
        r = np.random.default_rng(seed)
        days = TEST_DAYS[TEST_DAYS + 1 < 366] if s1 + 12 > 24 else TEST_DAYS
        jobs = []
        for k in range(a.n):
            day = int(r.choice(days))
            start = int(r.integers(s0, s1 + 1))
            jobs.append({"k": k, "day": day, "start": start, "seed": int(r.integers(1 << 31))})
        specs["shifts"][name] = {"seed": seed, "kw": kw, "starts": [s0, s1], "jobs": jobs}
    json.dump(specs, open(os.path.join(OUTD, "specs.json"), "w"), indent=1)
    print("saved", os.path.join(OUTD, "specs.json"), f"{time.time() - t0:.0f}s")
