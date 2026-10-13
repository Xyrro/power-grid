"""hard benchmark, data: RTS-GMLC 24 h with line ratings at 60 % (otsl.hard: CFG, bench_jobs).

    python scripts/uc_hard_data.py --gen train --chunks 0-29     # LP relaxation + label-free labels, no MILP (50 / chunk)
    python scripts/uc_hard_data.py --gen val --chunks 0,2,4      # + full MILP (0.1 %, 600 s, bound trace) (5 / chunk)
    python scripts/uc_hard_data.py --pilot --teacher t0          # polishing settings on validation instances 0-19
    python scripts/uc_hard_data.py --polish --teacher t0 --chunks 0-14 --tl 10

Every instance: otsl.b3.instance (scenario of a calendar day of the split, LP relaxation, label-free labels y_lf /
y_blk with their dispatch LPs; validation and test also the full MILP through highspy, one thread). Polished labels:
otsl.hard.polish with the teacher's mean probability. Output (git-ignored): data/generated/hard/<split>_<kk>.npz
(+ _trace.json: the MILP incumbent logs and bound traces), polish_<kk>.npz, pilot.json. Resume = skip existing files.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import instance, load_b3, save_b3, to_arrays  # noqa: E402
from otsl.hard import CFG, DATA_DIR, SPLIT_N, bench_jobs, bench_system, polish, sc_of  # noqa: E402

CHUNK = {"train": 50, "val": 5, "test": 5}


def parse(s, n):
    if not s:
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            x, y = part.split("-")
            out += list(range(int(x), int(y) + 1))
        else:
            out.append(int(part))
    return [k for k in out if k < n]


def teacher(name, sysm):
    from uc_hard_train import featurizer, load_member
    feat = featurizer(sysm)
    ms = [load_member(p, sysm, feat) for p in sorted(glob.glob(os.path.join(DATA_DIR, "models", f"{name}_s*.pt")))]
    assert ms, name
    return lambda d: np.mean([m.predict(d).astype(np.float64) for m in ms], 0), len(ms)


KEYS = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="")
    ap.add_argument("--chunks", default="")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--teacher", default="t0")
    ap.add_argument("--tl", type=float, default=10.0)
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--pilot_cfgs", default="0.9:10,0.9:20,0.8:10")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(DATA_DIR, exist_ok=True)
    sysm, m = bench_system()
    if a.gen:
        split = a.gen
        jobs = bench_jobs(split)
        cfg = dict(CFG, mode="lp" if split == "train" else "milp")
        C = CHUNK[split]
        for k in parse(a.chunks, (SPLIT_N[split] + C - 1) // C):
            path = os.path.join(DATA_DIR, f"{split}_{k:02d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = []
            for j in jobs[k * C:(k + 1) * C]:
                r = instance(j, cfg, sysm, m)
                res.append(r)
                if split != "train":
                    print(f"[{split} {k}] day {r['day']} MILP {r['time']:.1f}s gap {r['gap'] * 100:.3f}% "
                          f"{'opt' if r['opt'] else 'limit'} obj {r['obj']:.1f} lp-check {r['obj_lp']:.1f} c_rel "
                          f"{r['c_rel']:.1f} shed {r['shed']:.4f} short {r['short']:.4f} load {r['loadavg']:.1f}", flush=True)
            save_b3(path, to_arrays(res))
            json.dump(dict(inc=[r["inc"] for r in res], trace=[r.get("trace", []) for r in res], jobs=jobs[k * C:(k + 1) * C],
                           cpu=sorted(os.sched_getaffinity(0))), open(path.replace(".npz", "_trace.json"), "w"))
            print(f"{split} chunk {k}: {len(res)} instances {time.time() - t0:.0f}s", flush=True)
    if a.pilot:
        from otsl.hard import load_split
        v = load_split("val")
        idx = list(range(20))
        pred, nm = teacher(a.teacher, sysm)
        P = pred({k: v[k][idx] for k in KEYS})
        out = {"teacher": a.teacher, "members": nm, "idx": idx,
               "label_free_blk": {"gap_median_%": float(np.median((v["c_blk"][idx] / v["obj"][idx] - 1) * 100)),
                                  "gap_mean_%": float(np.mean((v["c_blk"][idx] / v["obj"][idx] - 1) * 100))}}
        print("label-free y_blk vs full MILP:", out["label_free_blk"], flush=True)
        for spec in a.pilot_cfgs.split(","):
            ratio, tl = (float(x) for x in spec.split(":"))
            recs, Y = [], []
            for k, i in enumerate(idx):
                y, r = polish(m, sysm, sc_of(v, i), P[k], v["y_blk"][i], v["c_blk"][i], ratio=ratio, time_limit=tl)
                recs.append(r)
                Y.append(y)
            cost = np.array([min(r.get("cost", np.inf), c) for r, c in zip(recs, v["c_blk"][idx])])
            g = (cost / v["obj"][idx] - 1) * 100
            out[spec] = {"ratio": ratio, "tl": tl, "time_mean_s": float(np.mean([r["total_s"] for r in recs])),
                         "gap_median_%": float(np.median(g)), "gap_mean_%": float(g.mean()), "gap_max_%": float(g.max()),
                         "n_gt_0.5%": int((g > 0.5).sum()), "n_gt_1%": int((g > 1).sum()),
                         "hit_limit": int(sum(r["status"] != "optimal" for r in recs)),
                         "used_milp_%": float(np.mean([r["used"] == "milp" for r in recs]) * 100),
                         "fixed_%": float(np.mean([r["n_fixed"] for r in recs]) / (24 * sysm.G) * 100),
                         "agree_milp_%": float(np.mean([(Y[k] == v["u"][i]).mean() for k, i in enumerate(idx)]) * 100)}
            print(spec, out[spec], flush=True)
        json.dump(out, open(os.path.join(DATA_DIR, "pilot.json"), "w"), indent=1)
    if a.polish:
        pred, nm = teacher(a.teacher, sysm)
        cfg = dict(teacher=a.teacher, members=nm, ratio=a.ratio, tl=a.tl, gap=1e-3, guards="adeq+rowmin+lp")
        for k in parse(a.chunks, (SPLIT_N["train"] + CHUNK["train"] - 1) // CHUNK["train"]):
            path, src = os.path.join(DATA_DIR, f"polish_{k:02d}.npz"), os.path.join(DATA_DIR, f"train_{k:02d}.npz")
            if os.path.exists(path) or not os.path.exists(src):
                continue
            d = load_b3(src)
            t0 = time.time()
            P = pred({kk: d[kk] for kk in KEYS})
            t_inf = (time.time() - t0) / len(P)
            Y, recs = [], []
            for i in range(len(P)):
                y, r = polish(m, sysm, sc_of(d, i), P[i], d["y_blk"][i], d["c_blk"][i], ratio=a.ratio, time_limit=a.tl)
                Y.append(y)
                recs.append(r)
            cost = np.array([min(r.get("cost", np.inf), c) for r, c in zip(recs, d["c_blk"])])
            np.savez_compressed(path, y=np.array(Y, np.int8), p_teacher=P.astype(np.float32), cost=cost,
                                used_milp=np.array([r["used"] == "milp" for r in recs]),
                                total_s=np.array([r["total_s"] for r in recs]), milp_s=np.array([r["milp_s"] for r in recs]),
                                n_fixed=np.array([r["n_fixed"] for r in recs]),
                                hit_limit=np.array([r["status"] != "optimal" for r in recs]),
                                served=np.array([r.get("shed", 1) < 1e-6 and r.get("short", 1) < 1e-6 for r in recs]),
                                t_inf=t_inf, cfg=json.dumps(cfg))
            print(f"polish {k}: {len(P)} in {time.time() - t0:.0f}s, from MILP "
                  f"{np.mean([r['used'] == 'milp' for r in recs]) * 100:.0f}%, mean {np.mean([r['total_s'] for r in recs]):.1f}s, "
                  f"vs y_blk median {np.median(cost / d['c_blk'] - 1) * 100:.2f}%", flush=True)
