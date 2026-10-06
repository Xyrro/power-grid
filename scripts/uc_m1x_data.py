"""m1x study, data: extra 12-hour training scenarios without full MILPs, label-free and teacher-polished labels.

    python3 scripts/uc_m1x_data.py --pilot            # choose the polishing setting on 24 training instances (OOF probs)
    python3 scripts/uc_m1x_data.py --gen              # scenarios + LP relaxation + label-free labels, chunks of 250
    python3 scripts/uc_m1x_data.py --polish           # teacher-polished labels for every generated chunk

Scenarios: otsl.m1x.scenario_jobs(3500, seed 111, training days) - the sampler of otsl.ucdata.generate (train.npz
used seed 11), so the distribution is that of the existing training set. One process, HiGHS single-threaded.
Output (git-ignored): data/generated/uc12_m1x/extra_<k>.npz (k = 0..13, 250 instances each),
polish_<k>.npz (labels + records), pilot.json; every chunk is written once complete (resume = skip existing files).
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
from uc_constrained import load_model1, strip  # noqa: E402
from uc_gen import SPLITS  # noqa: E402

ROOT, OUT, RES = "data/generated/uc12", "data/generated/uc12_m1x", "results/uc12"
N_EXTRA, CHUNK, SEED = 3500, 250, 111
TEACHERS = ("uc_model1_4.pt", "hybrid_bce_s1.pt", "hybrid_bce_s2.pt")   # MILP-label BCE GNN, seeds 0-2


def teacher(sysm, trs):
    ms = [load_model1(os.path.join(RES, f), sysm, trs, 12) for f in TEACHERS]
    return lambda d: np.mean([m.predict(d).astype(np.float64) for m in ms], 0)


def pilot(m, sysm, tr, n=24, only=()):
    """settings for the polishing MILP, on training instances with out-of-fold BCE probabilities (realistic errors;
    the training MILP objective is used only to score label quality, as in W2)"""
    P = np.load(os.path.join(RES, "fixpolicy_probs.npz"))["bce_oof_tr"]
    idx = np.random.default_rng(0).choice(len(tr["load"]), n, replace=False)
    cfgs = {"asym q0.90 tl5": dict(ratio=0.90, rank="asym", tl=5.0), "asym q0.95 tl5": dict(ratio=0.95, rank="asym", tl=5.0),
            "conf q0.90 tl5": dict(ratio=0.90, rank="conf", tl=5.0),
            "asym q0.95 tl5 +lp": dict(ratio=0.95, rank="asym", tl=5.0, lp=True),
            "conf q0.90 tl5 +lp": dict(ratio=0.90, rank="conf", tl=5.0, lp=True)}
    if only:
        cfgs = {k: v for k, v in cfgs.items() if k in only}
    out = {}
    lf = {}
    for i in idx:
        sc = scenario(tr, i)
        from otsl.m1x import lf_label
        y = lf_label(tr["u_rel"][i], sc, sysm)
        lf[i] = (y, m.solve_dispatch(sc, y).obj)
    out["lf"] = {"gap_median_%": float(np.median([(lf[i][1] / tr["obj"][i] - 1) * 100 for i in idx])),
                 "gap_mean_%": float(np.mean([(lf[i][1] / tr["obj"][i] - 1) * 100 for i in idx]))}
    print("label-free", out["lf"], flush=True)
    for name, c in cfgs.items():
        recs = []
        for i in idx:
            sc = scenario(tr, i)
            g = (lambda f, sc=sc: lp_guard(m, sysm, sc, f)[0]) if c.get("lp") else None
            y, r = polish(m, sysm, sc, P[i], lf[i][0], lf[i][1], ratio=c["ratio"], time_limit=c["tl"], rank=c["rank"],
                          lp_guard_fn=g)
            r["gap_%"] = (min(r["cost"], lf[i][1]) / tr["obj"][i] - 1) * 100
            r["agree_milp_%"] = float((y == tr["u"][i]).mean() * 100)
            recs.append(r)
        g = np.array([r["gap_%"] for r in recs])
        out[name] = {"time_mean_s": float(np.mean([r["total_s"] for r in recs])), "gap_median_%": float(np.median(g)),
                     "gap_mean_%": float(g.mean()), "n_gt_1%": int((g > 1).sum()), "hit_limit": int(sum(r["status"] == "time_limit" for r in recs)),
                     "served_%": float(np.mean([r.get("shed", 1) < 1e-6 and r.get("short", 1) < 1e-6 for r in recs]) * 100),
                     "used_milp_%": float(np.mean([r["used"] == "milp" for r in recs]) * 100)}
        print(name, out[name], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot_cfgs", default="", help="comma-separated pilot settings (default: all)")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--rank", default="asym")
    ap.add_argument("--tl", type=float, default=5.0)
    ap.add_argument("--lp", action="store_true", help="LP-relaxation guard on the polishing fixings")
    ap.add_argument("--chunks", default="", help="comma-separated chunk ids (default: all)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    tr = load(os.path.join(ROOT, "train.npz"))
    chunks = [int(x) for x in a.chunks.split(",") if x] or list(range(N_EXTRA // CHUNK))
    if a.pilot:
        pj = os.path.join(OUT, "pilot.json")
        res = json.load(open(pj)) if os.path.exists(pj) else {}
        res.update(pilot(m, sysm, tr, only=[c for c in a.pilot_cfgs.split(",") if c]))
        json.dump(res, open(pj, "w"), indent=1)
    if a.gen:
        jobs = scenario_jobs(N_EXTRA, SEED, SPLITS["train"], T=12)
        for k in chunks:
            path = os.path.join(OUT, f"extra_{k}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            recs = [lf_instance(m, sysm, j) for j in jobs[k * CHUNK:(k + 1) * CHUNK]]
            d = {key: np.array([r[key] for r in recs]) for key in recs[0]}
            np.savez_compressed(path, **d)
            print(f"chunk {k}: {len(recs)} instances, {time.time() - t0:.0f}s, LP relaxation {d['t_rel'].mean():.2f}s, "
                  f"label-free label {d['t_lf'].mean():.2f}s", flush=True)
    if a.polish:
        trs = strip(tr)
        pred = teacher(sysm, trs)
        for k in chunks:
            path, src = os.path.join(OUT, f"polish_{k}.npz"), os.path.join(OUT, f"extra_{k}.npz")
            if os.path.exists(path) or not os.path.exists(src):
                continue
            d = dict(np.load(src))
            t0 = time.time()
            ti = time.time()
            P = pred(d)
            t_inf = (time.time() - ti) / len(P)
            Y, recs = [], []
            for i in range(len(P)):
                sc = scenario(d, i)
                g = (lambda f, sc=sc: lp_guard(m, sysm, sc, f)[0]) if a.lp else None
                y, r = polish(m, sysm, sc, P[i], d["y_lf"][i], d["c_lf"][i], ratio=a.ratio, time_limit=a.tl,
                              rank=a.rank, lp_guard_fn=g)
                Y.append(y)
                recs.append(r)
            cost = np.array([min(r.get("cost", np.inf), c) for r, c in zip(recs, d["c_lf"])])
            np.savez_compressed(path, y=np.array(Y, np.int8), p_teacher=P.astype(np.float32), cost=cost,
                                used_milp=np.array([r["used"] == "milp" for r in recs]),
                                total_s=np.array([r["total_s"] for r in recs]), milp_s=np.array([r["milp_s"] for r in recs]),
                                n_fixed=np.array([r["n_fixed"] for r in recs]),
                                hit_limit=np.array([r["status"] == "time_limit" for r in recs]),
                                served=np.array([r.get("shed", 1) < 1e-6 and r.get("short", 1) < 1e-6 for r in recs]),
                                t_inf=t_inf, cfg=json.dumps(dict(ratio=a.ratio, rank=a.rank, tl=a.tl, lp=a.lp)))
            print(f"polish chunk {k}: {time.time() - t0:.0f}s, label from MILP "
                  f"{np.mean([r['used'] == 'milp' for r in recs]) * 100:.0f}%, mean {np.mean([r['total_s'] for r in recs]):.2f}s, "
                  f"label vs label-free cost {np.mean(cost / d['c_lf'] - 1) * 100:.2f}% (median "
                  f"{np.median(cost / d['c_lf'] - 1) * 100:.2f}%)", flush=True)
