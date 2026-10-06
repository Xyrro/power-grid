"""PGLib-UC California: Learning to Fix (faithful; otsl.ltfx Algorithm 1 + 2) tuned on the validation instances.

    python3 scripts/uc_pglib_ltf.py --jobs knn:0.01,knn:0.05,st:0.01 --workers 2 --n_val 30 --relax_tl 20

Probabilities: kNN (k = 50, Table II features, labelled set data/generated/pglib_ca/knn.npz) and our models
(results/pglib/pglib_probs.npz). C* = the validation MILP objective. One tuning run per process (HiGHS single
thread). Writes results/pglib/pglib_ltf_<model>_<eps>.{json,log} and the cheap thresholds (constant, worst case).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.ltfx import fix_masks  # noqa: E402
from otsl.pglib import PGModel, load_bases, load_npz, subset  # noqa: E402

ROOT, OUT = "data/generated/pglib_ca", "results/pglib"


def tag(name, eps):
    return f"{name}_{eps * 100:g}"


def val_probs(name, n_val):
    P = np.load(os.path.join(OUT, "pglib_probs.npz"))
    return P[f"{name}_va"][:n_val]


def run_job(args):
    name, eps, cfg = args
    os.chdir(os.path.dirname(HERE))
    from otsl.pglib_ml import PGLtFTuner
    t0, c0 = time.time(), time.process_time()
    sysm, _ = load_bases(cfg["T"])
    m = PGModel(sysm, T=cfg["T"])
    d = subset(load_npz(os.path.join(ROOT, "val.npz")), np.arange(cfg["n_val"]))
    pi = val_probs(name, cfg["n_val"])
    logf = open(os.path.join(OUT, f"pglib_ltf_{tag(name, eps)}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()

    tuner = PGLtFTuner(m, sysm, d, pi, eps, K_max=cfg["K_max"], check_tl=cfg["check_tl"], relax_tl=cfg["relax_tl"],
                       master_tl=cfg["master_tl"], log=log, time_budget=cfg["budget_min"] * 60)
    lo, hi = tuner.run()
    ver = tuner.verify(lo, hi)
    off, on = fix_masks(pi, lo, hi)
    out = dict(model=name, eps=eps, lo=lo.tolist(), hi=hi.tolist(), stats=tuner.stats, history=tuner.history,
               val_fixed_share=float((off | on).mean()), val_off_share=float(off.mean()), val_on_share=float(on.mean()),
               verify_max=float(np.max(ver)), verify=ver.tolist(), n_val=cfg["n_val"], wall_s=time.time() - t0,
               cpu_s=time.process_time() - c0, budget_stop=any(h.get("result") == "budget stop" for h in tuner.history))
    json.dump(out, open(os.path.join(OUT, f"pglib_ltf_{tag(name, eps)}.json"), "w"), default=float)
    log(f"done: val fixed {out['val_fixed_share'] * 100:.2f}%, verify max {out['verify_max'] * 100:.3f}%, "
        f"wall {out['wall_s']:.0f}s")
    return name, eps, out["val_fixed_share"], out["wall_s"], out["budget_stop"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="knn:0.01,knn:0.05,st:0.01")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--check_tl", type=float, default=120.0)
    ap.add_argument("--relax_tl", type=float, default=20.0)
    ap.add_argument("--master_tl", type=float, default=120.0)
    ap.add_argument("--budget_min", type=float, default=90.0)
    a = ap.parse_args()
    cfg = dict(n_val=a.n_val, T=a.T, K_max=a.K_max, check_tl=a.check_tl, relax_tl=a.relax_tl, master_tl=a.master_tl,
               budget_min=a.budget_min)
    jobs = [(j.split(":")[0], float(j.split(":")[1]), cfg) for j in a.jobs.split(",")]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            print("finished", r, flush=True)
