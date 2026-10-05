"""Solver-outcome labels for the fixing policy (B2, uc12): dispatch-LP harm of every wrong fix.

    python scripts/uc_fixpolicy_label.py --workers 2

Train instances use out-of-fold probabilities (scripts/uc_fixpolicy_crossfit.py), val instances the
saved model's probabilities. For each instance: the optimum aligned to the prediction inside identical-
unit groups, the relative cost increase of forcing each wrongly predicted decision (one LP each), and
Learning-to-Fix impact curves per generator. Output: results/uc12/fixpolicy_labels.pkl
(--comp: second pass, harms of wrong OFF fixes with replacement units -> fixpolicy_labels_comp.pkl)
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.fixpolicy import compensated_harms, harm_labels  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

_W = {}


def _init():
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=12, network=True), groups=s.identical_groups())


def _job(args):
    key, sc, u_star, p, comp = args
    t0 = time.time()
    if comp:
        single, n_lp = compensated_harms(_W["m"], _W["s"], sc, u_star, p, _W["groups"], uncomp=comp["single"],
                                         base=comp["base"])
        r = dict(single_comp=single, n_lp=n_lp)
    else:
        r = harm_labels(_W["m"], _W["s"], sc, u_star, p, _W["groups"])
    r["key"], r["sec"] = key, time.time() - t0
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--comp", action="store_true", help="compensated harms of wrong OFF fixes (second pass)")
    a = ap.parse_args()
    root, out_dir = "data/generated/uc12", "results/uc12"
    tr, va = (load(os.path.join(root, f"{s}.npz")) for s in ("train", "val"))
    P = np.load(os.path.join(out_dir, "fixpolicy_probs.npz"))
    first = {}
    if a.comp:                                   # reuse the first pass (uncompensated harms, base cost)
        with open(os.path.join(out_dir, "fixpolicy_labels.pkl"), "rb") as f:
            first = {k: dict(single=v["single"], base=v["base"]) for k, v in pickle.load(f).items()}
    jobs = [(("va", i), scenario_from(va, i), va["u"][i], P["bce_va"][i], first.get(("va", i), a.comp))
            for i in range(len(va["load"]))]
    jobs += [(("tr", i), scenario_from(tr, i), tr["u"][i], P["bce_oof_tr"][i], first.get(("tr", i), a.comp))
             for i in range(min(a.n_train, len(tr["load"])))]
    out, t0 = {}, time.time()
    path = os.path.join(out_dir, "fixpolicy_labels_comp.pkl" if a.comp else "fixpolicy_labels.pkl")
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            out[r["key"]] = r
            if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                n_lp = sum(v["n_lp"] for v in out.values())
                print(f"[label] {k + 1}/{len(jobs)}  LPs {n_lp}  ({time.time() - t0:.0f}s)", flush=True)
                with open(path, "wb") as f:
                    pickle.dump(out, f)
    print("done", flush=True)
