"""Learning to Fix reconstruction, step 2: per-generator impact curves of fixing errors on calibration instances.

    python scripts/uc_ltf_curves.py --sets knn_va,bce_va,st_va,rl_va [--workers 2]
    python scripts/uc_ltf_curves.py --sets knn_oof_tr --n 500

For every calibration instance and every generator with a wrong rounded prediction (against the MILP optimum
aligned inside identical-unit groups), the dispatch LP prices the optimum with all of that generator's
decisions above each error's confidence level fixed to the prediction, with and without compensation by other
units (otsl/ltf.py: ltf_curves). Output: results/uc12/ltf_curves.pkl ({(set, i): result}; resumable).
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.ltf import ltf_curves  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

_W = {}


def _init():
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=12, network=True), groups=s.identical_groups())


def _job(args):
    key, sc, u_star, p = args
    t0 = time.time()
    r = ltf_curves(_W["m"], _W["s"], sc, u_star, p, _W["groups"])
    r["key"], r["sec"] = key, time.time() - t0
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="knn_va,bce_va,st_va,rl_va")
    ap.add_argument("--n", type=int, default=0, help="first n instances of each set (0 = all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="ltf_curves.pkl")
    a = ap.parse_args()
    root, out_dir = "data/generated/uc12", "results/uc12"
    data = {"va": load(os.path.join(root, "val.npz")), "tr": load(os.path.join(root, "train.npz"))}
    P = np.load(os.path.join(out_dir, "ltf_probs.npz"))
    path = os.path.join(out_dir, a.out)
    out = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    jobs = []
    for sname in a.sets.split(","):
        d = data[sname.split("_")[-1]]
        p_all = P[sname]
        n = len(p_all) if not a.n else min(a.n, len(p_all))
        for i in range(n):
            if (sname, i) not in out:
                jobs.append(((sname, i), scenario_from(d, i), d["u"][i], p_all[i]))
    print(f"{len(jobs)} instances to price ({len(out)} done)", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            out[r["key"]] = r
            if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                print(f"[curves] {k + 1}/{len(jobs)} LPs {sum(v['n_lp'] for v in out.values())} ({time.time() - t0:.0f}s)",
                      flush=True)
                with open(path + ".tmp", "wb") as f:
                    pickle.dump(out, f)
                os.replace(path + ".tmp", path)
    print("done", flush=True)
