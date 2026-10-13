"""m24 study, data: extra 24-hour training scenarios (no full MILP) and teacher-polished labels.

    python3 scripts/uc_m24_data.py --pilot --teacher t0p          # polishing settings on training instances 100-119
    python3 scripts/uc_m24_data.py --gen --chunks 0-5             # new scenarios, 50 per chunk
    python3 scripts/uc_m24_data.py --polish --teacher t0 --src orig,extra --tl 10

Scenarios: otsl.b3.make_jobs(training days, 600, seed 404) and otsl.b3.instance in mode "lp" - the generator of
data/generated/uc24/train.npz (seed 101) with another seed: LP relaxation, label-free labels (y_lf, y_blk) and their
dispatch LPs, no MILP. Polished labels (otsl.m24.polish): the teacher's mean probability (members
data/generated/uc24_m24/models/<teacher>_s*.pt), its 90 % most confident decisions fixed, adequacy guard, minimal
min up/down conflict release, LP-relaxation guard, reduced MILP (highspy, one thread, --tl s, 0.1 %), kept if the
dispatch LP prices it below the block-repaired label-free label y_blk. Sources: orig = training instances 120-299 of
train.npz (instances 0-119 have 0.5 %-gap MILP schedules), extra = the new scenarios.
Output (git-ignored): data/generated/uc24_m24/extra_<k>.npz, polish_<src>_<k>.npz, pilot.json; resume = skip
existing files.
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
sys.path.insert(0, HERE)
from otsl.b3 import instance, load_b3, make_jobs, save_b3, to_arrays  # noqa: E402
from otsl.m24 import polish  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402
from uc_gen import SPLITS  # noqa: E402
from uc_m24_train import KEYS, featurizer, load_member  # noqa: E402

U24, DATA = "data/generated/uc24", "data/generated/uc24_m24"
CFG = {"T": 24, "start": 0, "network": True, "time_limit": 300.0, "mip_gap": 1e-3, "mode": "lp"}
N_MAX, SEED, CHUNK, ORIG_CHUNK = 600, 404, 50, 45


def parse(s, n):
    if not s:
        return list(range(n))
    if "-" in s:
        x, y = s.split("-")
        return list(range(int(x), int(y) + 1))
    return [int(x) for x in s.split(",")]


def teacher(name, sysm):
    feat = featurizer(sysm)
    ms = [load_member(p, sysm, feat) for p in sorted(glob.glob(os.path.join(DATA, "models", f"{name}_s*.pt")))]
    assert ms, name
    return lambda d: np.mean([m.predict(d).astype(np.float64) for m in ms], 0), len(ms)


def sc_of(d, i):
    return UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])


def polish_set(m, sysm, d, idx, P, tl, ratio=0.9):
    Y, recs = [], []
    for k, i in enumerate(idx):
        y, r = polish(m, sysm, sc_of(d, i), P[k], d["y_blk"][i], d["c_blk"][i], ratio=ratio, time_limit=tl)
        Y.append(y)
        recs.append(r)
    return np.array(Y, np.int8), recs


def save_polish(path, idx, Y, recs, d, P, t_inf, cfg):
    cost = np.array([min(r.get("cost", np.inf), c) for r, c in zip(recs, d["c_blk"][idx])])
    np.savez_compressed(path, idx=np.asarray(idx), y=Y, p_teacher=P.astype(np.float32), cost=cost,
                        used_milp=np.array([r["used"] == "milp" for r in recs]),
                        total_s=np.array([r["total_s"] for r in recs]), milp_s=np.array([r["milp_s"] for r in recs]),
                        n_fixed=np.array([r["n_fixed"] for r in recs]),
                        hit_limit=np.array([r["status"] != "optimal" for r in recs]),
                        released_rows=np.array([r.get("released_rows", 0) for r in recs]),
                        released_lp=np.array([r.get("released_lp", 0) for r in recs]),
                        served=np.array([r.get("shed", 1) < 1e-6 and r.get("short", 1) < 1e-6 for r in recs]),
                        t_inf=t_inf, cfg=json.dumps(cfg))
    return cost


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--teacher", default="t0")
    ap.add_argument("--src", default="orig,extra")
    ap.add_argument("--chunks", default="")
    ap.add_argument("--tl", type=float, default=10.0)
    ap.add_argument("--tls", default="10,20", help="pilot time limits")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(DATA, exist_ok=True)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=24, network=True)
    if a.pilot:
        lab = load_b3(os.path.join(U24, "uc24ltf_train_lab.npz"))
        tr = load_b3(os.path.join(U24, "train.npz"))
        idx = list(range(100, 120))
        pred, nm = teacher(a.teacher, sysm)
        P = pred({k: tr[k][idx] for k in KEYS})
        out = {"teacher": a.teacher, "members": nm, "idx": idx,
               "label_free_blk": {"gap_median_%": float(np.median((tr["c_blk"][idx] / lab["obj"][idx] - 1) * 100)),
                                  "gap_mean_%": float(np.mean((tr["c_blk"][idx] / lab["obj"][idx] - 1) * 100))}}
        print("label-free y_blk vs 0.5 %-gap MILP:", out["label_free_blk"], flush=True)
        for tl in [float(x) for x in a.tls.split(",")]:
            Y, recs = polish_set(m, sysm, tr, idx, P, tl)
            cost = np.array([min(r.get("cost", np.inf), c) for r, c in zip(recs, tr["c_blk"][idx])])
            g = (cost / lab["obj"][idx] - 1) * 100
            out[f"tl{tl:g}"] = {"time_mean_s": float(np.mean([r["total_s"] for r in recs])),
                                "gap_median_%": float(np.median(g)), "gap_mean_%": float(g.mean()),
                                "n_gt_0.5%": int((g > 0.5).sum()), "n_gt_1%": int((g > 1).sum()),
                                "hit_limit": int(sum(r["status"] != "optimal" for r in recs)),
                                "used_milp_%": float(np.mean([r["used"] == "milp" for r in recs]) * 100),
                                "fixed_%": float(np.mean([r["n_fixed"] for r in recs]) / (24 * sysm.G) * 100),
                                "agree_milp_%": float(np.mean([(Y[k] == lab["u"][i]).mean() for k, i in enumerate(idx)]) * 100)}
            print(f"tl {tl:g}:", out[f"tl{tl:g}"], flush=True)
        json.dump(out, open(os.path.join(DATA, "pilot.json"), "w"), indent=1)
    if a.gen:
        jobs = make_jobs(SPLITS["train"], N_MAX, SEED, T=24, start=0)
        for k in parse(a.chunks, N_MAX // CHUNK):
            path = os.path.join(DATA, f"extra_{k}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = [instance(j, CFG, sysm, m) for j in jobs[k * CHUNK:(k + 1) * CHUNK]]
            save_b3(path, to_arrays(res))
            print(f"extra chunk {k}: {len(res)} instances {time.time() - t0:.0f}s "
                  f"(label {np.mean([r['t_label'] for r in res]):.2f}s each)", flush=True)
    if a.polish:
        pred, nm = teacher(a.teacher, sysm)
        cfg = dict(teacher=a.teacher, members=nm, ratio=0.9, tl=a.tl, gap=1e-3, guards="adeq+rowmin+lp")
        for src in [x for x in a.src.split(",") if x]:
            if src == "orig":
                d = load_b3(os.path.join(U24, "train.npz"))
                all_idx = list(range(120, 300))
                n_ch = (len(all_idx) + ORIG_CHUNK - 1) // ORIG_CHUNK
                get = lambda k: (d, all_idx[k * ORIG_CHUNK:(k + 1) * ORIG_CHUNK])
            else:
                n_ch = N_MAX // CHUNK
                get = lambda k: (load_b3(os.path.join(DATA, f"extra_{k}.npz")), list(range(CHUNK)))
            for k in parse(a.chunks if src == "extra" else "", n_ch):
                path = os.path.join(DATA, f"polish_{src}_{k}.npz")
                if os.path.exists(path) or (src == "extra" and not os.path.exists(os.path.join(DATA, f"extra_{k}.npz"))):
                    continue
                dd, idx = get(k)
                t0 = time.time()
                P = pred({kk: dd[kk][idx] for kk in KEYS})
                t_inf = (time.time() - t0) / len(idx)
                Y, recs = polish_set(m, sysm, dd, idx, P, a.tl)
                cost = save_polish(path, idx, Y, recs, dd, P, t_inf, cfg)
                print(f"polish {src} {k}: {len(idx)} in {time.time() - t0:.0f}s, from MILP "
                      f"{np.mean([r['used'] == 'milp' for r in recs]) * 100:.0f}%, mean {np.mean([r['total_s'] for r in recs]):.1f}s, "
                      f"vs y_blk median {np.median(cost / dd['c_blk'][idx] - 1) * 100:.2f}%, rows released "
                      f"{np.mean([r.get('released_rows', 0) for r in recs]):.1f}", flush=True)
