"""Learning to Fix reconstruction, step 3b: sequential (joint) decomposition of the generator thresholds.

    python scripts/uc_ltf_seq.py --plan knn:seq:0.01,0.04 knn:budgetJ:0.01,0.02 --boot 3

seq:     generators are visited in ascending order of their stand-alone compensated impact of fixing all their
         decisions (from the per-generator curves, results/uc12/ltf_curves.pkl); each threshold is lowered while the
         joint validation cost increase (upper bound by the dispatch LP with capacity compensation; otsl/ltf.py:
         JointCalibrator) stays <= tau.
budgetJ: the separable budget (Lagrangian) decomposition proposes the thresholds; the joint validation cost of the
         proposal is checked with the same upper bound and the internal tolerance bisected until it is <= tau.
Both end with the exact joint check (bisection of the internal tolerance, <= 3 reruns).
--boot s: s bootstrap resamples of the 60 calibration instances (seeds 0..s-1), kNN, tau = 1 %, mode --boot_mode.
Output: results/uc12/ltf_thresholds_seq.json ({model: {"seq:comp:<tau>": {...}}}), log on stdout.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.ltf import JointCalibrator, _jinit, generator_options, generator_tables, thresholds_budget  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402

OUT = "results/uc12"


def standalone_order(curves, G):
    tabs = generator_tables(curves, G, "comp")
    key = [(0.0 if len(lv) == 0 else (ms[-1] if np.isfinite(ms[-1]) else 1e9)) for lv, ms in tabs]
    return list(np.argsort(key, kind="stable"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", default=["knn:0.01,0.02,0.04"])
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--boot_mode", default="budgetJ")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nocomp", action="store_true")
    ap.add_argument("--out", default="ltf_thresholds_seq.json")
    a = ap.parse_args()
    sysm = load_rts_gmlc()
    va = load("data/generated/uc12/val.npz")
    va = {k: v[:a.n] for k, v in va.items() if isinstance(v, np.ndarray) and len(v) == 60}
    P = np.load(os.path.join(OUT, "ltf_probs.npz"))
    C = pickle.load(open(os.path.join(OUT, "ltf_curves.pkl"), "rb"))
    path = os.path.join(OUT, a.out)
    res = json.load(open(path)) if os.path.exists(path) else {}
    imp = "nocomp" if a.nocomp else "comp"
    splits = [s for s in ("va", "te", "tf") if f"knn_{s}" in P.files]
    share = lambda p, th: float((np.maximum(p, 1 - p) >= np.asarray(th)[None, None, :]).mean())
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_jinit) as pool:
        jobs = [(spec.split(":")[0], spec.split(":")[1], [float(x) for x in spec.split(":")[2].split(",")], None)
                for spec in a.plan]
        jobs += [("knn", a.boot_mode, [0.01], s) for s in range(a.boot)]
        cals = {}
        for model, mode, taus, boot in jobs:
            if model not in cals:
                cals[model] = JointCalibrator(sysm, va, P[f"{model}_va"][:a.n], pool, comp=not a.nocomp,
                                              groups=sysm.identical_groups())
            cal = cals[model]
            curves = [C[(f"{model}_va", i)]["curves"] for i in range(a.n)]
            order = standalone_order(curves, sysm.G)
            w = None
            name = model
            conf = np.maximum(P[f"{model}_va"][:a.n], 1 - P[f"{model}_va"][:a.n])
            if boot is not None:
                w = np.bincount(np.random.default_rng(boot).integers(0, a.n, a.n), minlength=a.n)
                name = f"knn_boot{boot}"
                rep = np.repeat(np.arange(a.n), w)            # bootstrap sample for the per-generator curves
                curves, conf = [curves[i] for i in rep], conf[rep]
            opts = generator_options(generator_tables(curves, sysm.G, imp), conf)
            propose = None if mode == "seq" else (lambda t, opts=opts: thresholds_budget(opts, t)[:2])
            for tau in taus:
                key = f"{mode}:{imp}:{tau:g}"
                if key in res.get(name, {}):
                    continue
                print(f"== {name} {key}", flush=True)
                theta, final, hist = cal.calibrate(tau, order, w, log=lambda s: print(s, flush=True), propose=propose,
                                                   max_rerun=3 if mode == "seq" else 5)
                r = dict(theta=[float(x) if np.isfinite(x) else 2.0 for x in theta], impact_cal=final, hist=hist,
                         share_cal=share(P[f"{model}_va"][:a.n], np.where(np.isfinite(theta), theta, 2.0)))
                for s in splits:
                    r[f"share_{s}"] = share(P[f"{model}_{s}"], r["theta"])
                res.setdefault(name, {})[key] = r
                print(f"{name} {key}: joint val impact {final * 100:.3f}%  share val {r['share_va'] * 100:.1f}%  "
                      + "  ".join(f"{s} {r[f'share_{s}'] * 100:.1f}%" for s in splits) + f"  LPs {cal.n_lp} ({time.time() - t0:.0f}s)",
                      flush=True)
                with open(path, "w") as f:
                    json.dump(res, f)
    print("done", flush=True)
