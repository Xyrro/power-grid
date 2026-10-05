"""B3 probe: how long does the full 24-hour network-constrained UC MILP take?

    python scripts/uc_b3_probe.py --T 24 --limit 900 --gap 1e-3 --days 16,108,196,288

Solves the full MILP (highspy, 1 thread per solve, at most 2 solves in parallel) on a few training-split days
with start hour 0 and the network on, and logs the incumbent and dual-bound trajectories, so a time limit /
gap configuration can be chosen from the gap reached over time.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.b3 import _init, _W, instance  # noqa: E402


def _probe(job):
    r = instance(job, _W["cfg"], _W["s"], _W["m"])
    keep = ["day", "start", "seed", "c_rel", "c_lf", "c_blk", "c_round", "obj", "obj_lp", "time", "gap", "bound",
            "opt", "nodes", "shed", "short", "t_rel", "t_label"]
    out = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keep}
    out["inc"], out["trace"] = r["inc"], r["trace"]
    out["units_on"] = float(r["u"].sum() / r["u"].shape[0])
    return out


def gap_at(trace, t):
    """relative gap (primal - dual) / primal at time t from a (time, primal, dual) trace"""
    best = None
    for tt, pr, du in trace:
        if tt <= t:
            best = (pr - du) / abs(pr) if np.isfinite(pr) and abs(pr) < 1e20 else np.inf
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--limit", type=float, default=900)
    ap.add_argument("--gap", type=float, default=1e-3)
    ap.add_argument("--days", default="16,108,196,288")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/uc24/b3_probe_T24.json")
    a = ap.parse_args()
    cfg = {"T": a.T, "network": True, "time_limit": a.limit, "mip_gap": a.gap, "mode": "milp", "trace_every": 10.0}
    rng = np.random.default_rng(a.seed)
    jobs = [(int(d), 0, int(rng.integers(1 << 31))) for d in a.days.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    res = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for r in pool.imap(_probe, jobs, chunksize=1):
            r["gap_at"] = {str(t): gap_at(r["trace"], t) for t in (30, 60, 120, 180, 300, 450, 600, 900)}
            res.append(r)
            print(f"day {r['day']}: {r['time']:.0f}s gap {r['gap'] * 100:.3f}% opt {r['opt']} obj {r['obj']:.0f} "
                  f"(lp {r['obj_lp']:.0f}) relax {r['c_rel']:.0f} lf-label {r['c_lf']:.0f} block {r['c_blk']:.0f} "
                  f"incumbents {len(r['inc'])} gap@t {r['gap_at']} ({time.time() - t0:.0f}s)", flush=True)
            json.dump({"cfg": cfg, "rows": res}, open(a.out, "w"), indent=1, default=float)
    print("done", time.time() - t0)
