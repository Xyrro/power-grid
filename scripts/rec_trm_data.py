"""Training data for the TRM study at scale: scenarios without full MILPs and teacher-polished labels, by chunk, so
that a cluster array job can produce 50-100k labelled instances (one array task = one chunk, one core).

    python3 scripts/rec_trm_data.py --chunk 7 --n_total 100000 --out data/generated/recursive/pace   # one chunk
    python3 scripts/rec_trm_data.py --chunk 7 --n_total 100000 --out ... --stage gen                 # scenarios only
    python3 scripts/rec_trm_data.py --chunk 7 --n_total 100000 --out ... --stage polish              # labels only

Same sampler and labels as the m1x study (scripts/uc_m1x_data.py, which is unchanged): otsl.m1x.scenario_jobs on the
training days (seed --seed, default 211, disjoint from train.npz (11) and the m1x extra scenarios (111)); per scenario
the LP relaxation and the label-free label (otsl.m1x.lf_instance); teacher-polished label (otsl.m1x.polish with the m1x
setting: the teacher's 90 % most confident decisions, adequacy guard, min up/down row release, LP-relaxation guard,
reduced MILP 5 s at 0.1 %). Teacher: mean of the three MILP-label BCE GNNs (results/uc12/uc_model1_4.pt,
hybrid_bce_s1.pt, hybrid_bce_s2.pt; retrain with scripts/uc_hybrid_train.py --seeds 0,1,2 and copy hybrid_bce_s0.pt to
uc_model1_4.pt if absent). Output: <out>/extra_<k>.npz and <out>/polish_<k>.npz in the m1x format (read by
otsl.trm.train_data(n, m1x=<out>)). Cost: ~0.6 s + ~5 s per instance on one core.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.fixpolicy import lp_guard  # noqa: E402
from otsl.m1x import lf_instance, polish, scenario, scenario_jobs  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402

ROOT = os.path.join("data", "generated", "uc12")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--n_total", type=int, default=100000)
    ap.add_argument("--chunk_size", type=int, default=250)
    ap.add_argument("--seed", type=int, default=211)
    ap.add_argument("--out", default=os.path.join("data", "generated", "recursive", "pace"))
    ap.add_argument("--stage", default="all", choices=["all", "gen", "polish"])
    ap.add_argument("--tl", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=0.9)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    from uc_gen import SPLITS
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    k = a.chunk
    src, dst = os.path.join(a.out, f"extra_{k}.npz"), os.path.join(a.out, f"polish_{k}.npz")
    if a.stage in ("all", "gen") and not os.path.exists(src):
        jobs = scenario_jobs(a.n_total, a.seed, SPLITS["train"], T=12)[k * a.chunk_size:(k + 1) * a.chunk_size]
        t0 = time.time()
        recs = [lf_instance(m, sysm, j) for j in jobs]
        d = {key: np.array([r[key] for r in recs]) for key in recs[0]}
        np.savez_compressed(src + ".tmp.npz", **d)
        os.replace(src + ".tmp.npz", src)
        print(f"chunk {k}: {len(recs)} scenarios, {time.time() - t0:.0f}s", flush=True)
    if a.stage in ("all", "polish") and not os.path.exists(dst):
        from uc_constrained import strip
        from uc_m1x_data import teacher
        pred = teacher(sysm, strip(load(os.path.join(ROOT, "train.npz"))))
        d = dict(np.load(src))
        t0 = time.time()
        P = pred(d)
        Y, recs = [], []
        for i in range(len(P)):
            sc = scenario(d, i)
            g = (lambda f, sc=sc: lp_guard(m, sysm, sc, f)[0])
            y, r = polish(m, sysm, sc, P[i], d["y_lf"][i], d["c_lf"][i], ratio=a.ratio, time_limit=a.tl, rank="conf",
                          lp_guard_fn=g)
            Y.append(y)
            recs.append(r)
        cost = np.array([min(r.get("cost", np.inf), c) for r, c in zip(recs, d["c_lf"])])
        np.savez_compressed(dst + ".tmp.npz", y=np.array(Y, np.int8), p_teacher=P.astype(np.float32), cost=cost,
                            used_milp=np.array([r["used"] == "milp" for r in recs]),
                            total_s=np.array([r["total_s"] for r in recs]), milp_s=np.array([r["milp_s"] for r in recs]),
                            n_fixed=np.array([r["n_fixed"] for r in recs]),
                            hit_limit=np.array([r["status"] == "time_limit" for r in recs]),
                            served=np.array([r.get("shed", 1) < 1e-6 and r.get("short", 1) < 1e-6 for r in recs]),
                            t_inf=0.0, cfg=json.dumps(dict(ratio=a.ratio, rank="conf", tl=a.tl, lp=True)))
        os.replace(dst + ".tmp.npz", dst)
        print(f"polish chunk {k}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
