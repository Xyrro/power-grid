"""PGLib-UC California: generate perturbed instances (data/generated/pglib_ca/<split>.npz).

    python3 scripts/uc_pglib_gen.py --split train --n 200 --seed 1 --mode lp   --workers 2
    python3 scripts/uc_pglib_gen.py --split val   --n 30  --seed 2 --mode milp --tl 900 --gap 0.001
    python3 scripts/uc_pglib_gen.py --split test  --n 30  --seed 3 --mode milp --tl 900 --gap 0.001
    python3 scripts/uc_pglib_gen.py --split knn   --n 100 --seed 4 --mode label --tl 60 --gap 0.01

mode lp: scenario + LP relaxation + repaired-relaxation label (no MILP); milp / label: + full MILP with its
incumbent log (label = loose gap / short limit for the kNN's labelled set).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.pglib import generate, save_npz  # noqa: E402

ROOT = "data/generated/pglib_ca"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", default="lp")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--gap", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = dict(T=a.T, mode=a.mode, time_limit=a.tl, mip_gap=a.gap)
    seeds = np.random.default_rng(a.seed).integers(1 << 31, size=a.n)
    path = os.path.join(ROOT, f"{a.split}.npz")
    t0 = time.time()
    data = generate(cfg, seeds, workers=a.workers, log=lambda s: print(s, flush=True),
                    partial_path=path.replace(".npz", ".partial.npz"))
    data["cfg_T"], data["cfg_tl"], data["cfg_gap"] = np.array(a.T), np.array(a.tl), np.array(a.gap)
    save_npz(path, data)
    print(f"saved {path}: {a.n} instances in {time.time() - t0:.0f}s", flush=True)
