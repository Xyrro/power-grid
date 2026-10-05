"""Learning to Fix reconstruction: how close is the LP impact proxy to the reduced MILP it bounds?

    python scripts/uc_ltf_proxycheck.py --n 30

Samples (seed 0) calibration cells (val instance i, generator g, threshold level theta) of the kNN impact curves
with a compensated LP impact > 0.1 %, and solves the reduced MILP in which only g's decisions with confidence >=
theta are fixed to the kNN prediction (all other decisions free; 60 s, 0.1 %). Its relative cost increase over
the dataset MILP is the impact the paper's calibration would measure with reduced MILPs; the LP impacts are upper
bounds of it (up to the MIP gap / time limit). Output: results/uc12/ltf_proxycheck.json
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

_W = {}


def _init():
    s = load_rts_gmlc()
    _W.update(m=UCModel(s, T=12, network=True))


def _job(args):
    cell, sc, fix = args
    sol = _W["m"].solve_uc(sc, time_limit=60.0, mip_gap=1e-3, z_fix=fix)
    return cell, (float(sol.obj) if sol.u is not None else float("inf")), float(sol.time), sol.status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    va = load("data/generated/uc12/val.npz")
    C = pickle.load(open("results/uc12/ltf_curves.pkl", "rb"))
    P = np.load("results/uc12/ltf_probs.npz")["knn_va"]
    cells = [(i, g, th, nc, cp, C[("knn_va", i)]["base"]) for i in range(60) for g, cv in C[("knn_va", i)]["curves"].items()
             for th, nc, cp in cv if np.isfinite(cp) and cp > 1e-3]
    pick = np.random.default_rng(0).choice(len(cells), min(a.n, len(cells)), replace=False)
    jobs = []
    for j in pick:
        i, g, th, nc, cp, base = cells[j]
        p = P[i]
        conf = np.maximum(p, 1 - p)
        fix = {(t, int(g)): int(p[t, g] > 0.5) for t in range(12) if conf[t, g] >= th}
        jobs.append(((int(i), int(g), float(th), float(nc), float(cp), float(base)), scenario_from(va, int(i)), fix))
    out = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for cell, obj, sec, st in pool.imap_unordered(_job, jobs, chunksize=1):
            i, g, th, nc, cp, base = cell
            milp = (obj - va["obj"][i]) / va["obj"][i]
            out.append(dict(i=i, g=g, theta=th, lp_nocomp=nc, lp_comp=cp, milp=milp, base_vs_obj=(base - va["obj"][i]) / va["obj"][i],
                            time=sec, status=st))
            print(f"i={i} g={g} th={th:.3f}  LP {nc * 100:.3f}%  LP+comp {cp * 100:.3f}%  reduced MILP {milp * 100:.3f}%  ({sec:.1f}s)",
                  flush=True)
    lc, lm = np.array([o["lp_comp"] for o in out]), np.array([o["milp"] for o in out])
    summ = dict(n=len(out), mean_lp_nocomp=float(np.mean([o["lp_nocomp"] for o in out])), mean_lp_comp=float(lc.mean()),
                mean_milp=float(lm.mean()), median_ratio_milp_over_lpcomp=float(np.median(lm / lc)),
                share_milp_below_half_lpcomp=float(np.mean(lm < 0.5 * lc)),
                spearman=float(np.corrcoef(np.argsort(np.argsort(lc)), np.argsort(np.argsort(lm)))[0, 1]))
    print(summ, flush=True)
    json.dump(dict(cells=out, summary=summ), open("results/uc12/ltf_proxycheck.json", "w"), indent=1)
