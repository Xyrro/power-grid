"""hard benchmark, training: temporal GNNs (otsl.hard.build_gnnt: CommitGNNT, dilations 1, 2, 4, 8) and their
validation metrics. No full MILP is used for training.

    python scripts/uc_hard_train.py --name t0 --seeds 0,1,2 --label lf --n 1500      # teacher: label-free labels
    python scripts/uc_hard_train.py --name fin --seeds 0,1,2,3,4 --label pol          # final: polished labels
    python scripts/uc_hard_train.py --ensemble fin                                     # mean of the trained seeds

Training data: the finished chunks data/generated/hard/train_<kk>.npz (in order, first --n instances); labels lf = the
block-repaired label-free label y_blk (rounded LP relaxation -> block adequacy repair -> min up/down repair), pol = the
polished labels of polish_<kk>.npz (chunks without polished labels are left out). Labels canonicalised inside
identical-unit groups. Features normalised with the first 300 training instances. BCE, AdamW 1e-3, batch 64, cosine
schedule, early stopping on the log-loss of the validation instances (full-MILP schedules; otsl.ucml.train_uc_bce).
Output: data/generated/hard/models/<name>_s<seed>.pt, probs/<name>_s<seed>_probs.npz (validation and test
probabilities), results/hard/hard_train.json.
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
from otsl.b3 import load_b3  # noqa: E402
from otsl.hard import DATA_DIR, T_H, bench_system, build_gnnt, load_split  # noqa: E402
from otsl.m1x import aligned_labels, prob_metrics  # noqa: E402
from otsl.ucml import UCFeaturizer, canonical_labels, train_uc_bce  # noqa: E402

RES = "results/hard"
KEYS = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel")
LOG = os.path.join(RES, "hard_train.json")
N_NORM = 300


def featurizer(sysm):
    tr = load_b3(os.path.join(DATA_DIR, "train_00.npz"))
    k = 1
    while len(tr["load"]) < N_NORM:
        e = load_b3(os.path.join(DATA_DIR, f"train_{k:02d}.npz"))
        tr = {key: np.concatenate([tr[key], e[key]]) for key in tr}
        k += 1
    return UCFeaturizer(sysm, {key: tr[key][:N_NORM] for key in KEYS}, relax=True, sym=True)


def train_set(sysm, label, n):
    parts, ys = [], []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "train_[0-9][0-9].npz"))):
        if label == "pol":
            pf = f.replace("train_", "polish_")
            if not os.path.exists(pf):
                continue
            y = np.load(pf)["y"]
        d = load_b3(f)
        parts.append({k: d[k] for k in KEYS})
        ys.append(d["y_blk"] if label == "lf" else y)
    d = {k: np.concatenate([p[k] for p in parts])[:n] for k in KEYS}
    y = np.concatenate(ys)[:n]
    d["obj"] = d["c_rel"]
    return d, canonical_labels(sysm, y, d["u0"]).astype(np.float32)


def load_member(path, sysm, feat):
    m1 = build_gnnt(sysm, feat, T_H, seed=0)
    m1.net.load_state_dict(torch.load(path))
    m1.net.eval()
    return m1


def metrics(sysm, p, v, std=None):
    ya = aligned_labels(sysm, v["u"], v["u0"], p)
    yc = canonical_labels(sysm, v["u"], v["u0"])
    return prob_metrics(p, ya, yc, std)


def update_log(tag, rec):
    os.makedirs(RES, exist_ok=True)
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    if rec.get("train_s", 0) is None and log.get(tag, {}).get("train_s"):
        rec["train_s"] = log[tag]["train_s"]
    log[tag] = rec
    json.dump(log, open(LOG, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--label", default="lf", choices=["lf", "pol"])
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--ensemble", default="", help="name: average the probabilities of its trained seeds")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "probs"), exist_ok=True)
    sysm, _ = bench_system()
    v = load_split("val")
    te = load_split("test")
    if a.ensemble:
        fs = sorted(glob.glob(os.path.join(DATA_DIR, "probs", f"{a.ensemble}_s*_probs.npz")))
        P = np.stack([np.load(f)["val"].astype(np.float64) for f in fs])
        p, std = P.mean(0), P.std(0)
        out = dict(val=p.astype(np.float32), std=std.astype(np.float32))
        if all("test" in np.load(f) for f in fs):
            out["test"] = np.mean([np.load(f)["test"].astype(np.float64) for f in fs], 0).astype(np.float32)
        np.savez_compressed(os.path.join(DATA_DIR, "probs", f"{a.ensemble}_ens_probs.npz"), **out)
        update_log(f"{a.ensemble}_ens", dict(members=[os.path.basename(f) for f in fs], val=metrics(sysm, p, v, std)))
        print(a.ensemble, "ensemble of", len(fs), json.dumps(json.load(open(LOG))[f"{a.ensemble}_ens"]["val"]), flush=True)
        sys.exit(0)
    feat = featurizer(sysm)
    t0 = time.time()
    tr, y = train_set(sysm, a.label, a.n)
    vt = {k: v[k] for k in KEYS}
    vt["obj"] = v["obj"]
    vt["u_target"] = canonical_labels(sysm, v["u"], v["u0"])
    print(f"{a.name}: {len(y)} training instances (labels {a.label}), data {time.time() - t0:.0f}s", flush=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        tag = f"{a.name}_s{s}"
        path = os.path.join(DATA_DIR, "models", f"{tag}.pt")
        if os.path.exists(path):
            m1, t_train = load_member(path, sysm, feat), None
        else:
            m1 = build_gnnt(sysm, feat, T_H, seed=s)
            t1 = time.time()
            train_uc_bce(m1, tr, vt, y, epochs=a.epochs, seed=s, log_every=10)
            t_train = time.time() - t1
            torch.save(m1.net.state_dict(), path)
        t1 = time.time()
        p = m1.predict({k: v[k] for k in KEYS}).astype(np.float64)
        t_inf = (time.time() - t1) / len(p)
        out = dict(val=p.astype(np.float32))
        if te:
            out["test"] = m1.predict({k: te[k] for k in KEYS}).astype(np.float32)
        np.savez_compressed(os.path.join(DATA_DIR, "probs", f"{tag}_probs.npz"), **out)
        rec = dict(name=a.name, seed=s, n_train=int(len(y)), label=a.label, epochs=a.epochs, train_s=t_train,
                   infer_s_per_inst=t_inf, val=metrics(sysm, p, v))
        update_log(tag, rec)
        print(tag, f"train {t_train}s", json.dumps(rec["val"]), flush=True)
