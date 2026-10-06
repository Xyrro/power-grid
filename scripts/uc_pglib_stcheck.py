"""PGLib-UC: quality of the self-training labels (the kNN's labelled set and the error-cost reference), measured on
validation instances whose full MILP is known: the same labelling step as uc_pglib_train.py --stage st (label-free
model, 90 % most confident decisions that agree with the repaired-relaxation label fixed, reduced MILP 60 s, keep if
cheaper), then the label's cost against the MILP objective.

    python3 scripts/uc_pglib_stcheck.py --n 30 --workers 1
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.pglib import load_bases, load_npz  # noqa: E402
from otsl.pglib_ml import PGFeaturizer, SolverPool, build_model1, fix_array, inst_arrays, reduced_job  # noqa: E402
from otsl.selftrain import label_fixings  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    sysm, _ = load_bases(24)
    tr = load_npz("data/generated/pglib_ca/train.npz")
    va = load_npz("data/generated/pglib_ca/val.npz")
    feat = PGFeaturizer(sysm, {k: v[:-16] for k, v in tr.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(tr["load"])})
    m1 = build_model1(sysm, feat)
    m1.net.load_state_dict(torch.load("results/pglib/pglib_m1_lf.pt"))
    p = m1.predict(va)
    n = min(a.n, len(va["load"]))
    jobs = [(i, inst_arrays(va, i), fix_array(label_fixings(p[i], va["y_lf"][i], 0.9, agree=True)), 60.0, 1e-3)
            for i in range(n)]
    pool = SolverPool(dict(T=24), a.workers)
    res = sorted(pool.run(reduced_job, jobs), key=lambda r: r["i"])
    pool.close()
    lab = np.array([min(r["cost"], va["c_lf"][r["i"]]) for r in res])
    g_st = (lab / va["obj"][:n] - 1) * 100
    g_lf = (va["c_lf"][:n] / va["obj"][:n] - 1) * 100
    out = dict(n=n, st_gap_mean=float(g_st.mean()), st_gap_median=float(np.median(g_st)), st_gap_max=float(g_st.max()),
               lf_gap_mean=float(g_lf.mean()), lf_gap_median=float(np.median(g_lf)), lf_gap_max=float(g_lf.max()),
               st_time_mean=float(np.mean([r["milp_s"] for r in res])),
               st_better_than_milp=int((g_st < -1e-6).sum()))
    json.dump(out, open("results/pglib/pglib_stcheck.json", "w"), indent=1)
    print(out)
