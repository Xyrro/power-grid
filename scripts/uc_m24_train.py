"""m24 study, training: temporal GNNs (otsl.m24.build_gnnt) on 24-hour instances, and their validation metrics.

    python3 scripts/uc_m24_train.py --name t0 --seeds 0,1,2 --lab 120                  # teacher: the 120 MILP labels
    python3 scripts/uc_m24_train.py --name t0p --seeds 0 --lab 100                     # pilot teacher (20 held out)
    python3 scripts/uc_m24_train.py --name fin --seeds 0,1,2,3,4 --lab 120 --pol orig,extra --epochs 40
    python3 scripts/uc_m24_train.py --ensemble fin                                     # mean of the trained seeds

Training data: the first --lab instances of data/generated/uc24/train.npz with their 0.5 %-gap MILP schedules
(uc24ltf_train_lab.npz: the first 120 training instances), plus the teacher-polished labels of
data/generated/uc24_m24/polish_<src>_<k>.npz (src: orig = training instances 120-299, extra = new scenarios). Labels
canonicalised inside identical-unit groups. Features normalised with the 300 original training instances. BCE, AdamW
1e-3, batch 64, cosine schedule, early stopping on the log-loss of the 40 validation instances with MILP schedules
(uc24ltf_val.npz). Output: data/generated/uc24_m24/models/<name>_s<seed>.pt, probs/<name>_s<seed>_probs.npz (val),
results/uc24/m24_train.json.
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
from otsl.m1x import aligned_labels, prob_metrics  # noqa: E402
from otsl.m24 import build_gnnt  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucml import UCFeaturizer, canonical_labels, train_uc_bce  # noqa: E402

U24, DATA, RES = "data/generated/uc24", "data/generated/uc24_m24", "results/uc24"
KEYS = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel")
LOG = os.path.join(RES, "m24_train.json")
T = 24


def featurizer(sysm):
    tr = load_b3(os.path.join(U24, "train.npz"))
    return UCFeaturizer(sysm, {k: tr[k] for k in KEYS}, relax=True, sym=True)


def polished(src):
    """scenarios and labels of every finished polish chunk of a source (orig / extra)"""
    tr = load_b3(os.path.join(U24, "train.npz"))
    parts, ys = [], []
    for p in sorted(glob.glob(os.path.join(DATA, f"polish_{src}_*.npz")), key=lambda x: int(x.rsplit("_", 1)[1][:-4])):
        z = np.load(p)
        if src == "orig":
            idx = z["idx"]
            parts.append({k: tr[k][idx] for k in KEYS})
        else:
            e = load_b3(p.replace(f"polish_{src}_", "extra_"))
            parts.append({k: e[k] for k in KEYS})
        ys.append(z["y"])
    return parts, ys


def train_set(sysm, n_lab, pols):
    tr = load_b3(os.path.join(U24, "train.npz"))
    lab = load_b3(os.path.join(U24, "uc24ltf_train_lab.npz"))
    assert np.array_equal(lab["seed"], tr["seed"][:120])
    parts, ys = [{k: lab[k][:n_lab] for k in KEYS}], [lab["u"][:n_lab]]
    for src in pols:
        p, y = polished(src)
        parts += p
        ys += y
    d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    d["obj"] = d["c_rel"]
    y = np.concatenate(ys)
    return d, canonical_labels(sysm, y, d["u0"]).astype(np.float32)


def val_set():
    v = load_b3(os.path.join(U24, "uc24ltf_val.npz"))
    return v


def load_member(path, sysm, feat):
    m1 = build_gnnt(sysm, feat, T, seed=0)
    m1.net.load_state_dict(torch.load(path))
    m1.net.eval()
    return m1


def metrics(sysm, p, v, std=None):
    ya = aligned_labels(sysm, v["u"], v["u0"], p)
    yc = canonical_labels(sysm, v["u"], v["u0"])
    return prob_metrics(p, ya, yc, std)


def update_log(tag, rec):
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    if rec.get("train_s", 0) is None and log.get(tag, {}).get("train_s"):
        rec["train_s"] = log[tag]["train_s"]
    log[tag] = rec
    json.dump(log, open(LOG, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lab", type=int, default=120)
    ap.add_argument("--pol", default="", help="comma-separated polished sources (orig, extra)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ensemble", default="", help="name: average the val probabilities of its trained seeds")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(os.path.join(DATA, "models"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "probs"), exist_ok=True)
    sysm = load_rts_gmlc()
    v = val_set()
    if a.ensemble:
        fs = sorted(glob.glob(os.path.join(DATA, "probs", f"{a.ensemble}_s*_probs.npz")))
        P = np.stack([np.load(f)["val"].astype(np.float64) for f in fs])
        p, std = P.mean(0), P.std(0)
        np.savez_compressed(os.path.join(DATA, "probs", f"{a.ensemble}_ens_probs.npz"), val=p.astype(np.float32),
                            std=std.astype(np.float32))
        update_log(f"{a.ensemble}_ens", dict(members=[os.path.basename(f) for f in fs], val=metrics(sysm, p, v, std)))
        print(a.ensemble, "ensemble of", len(fs), json.dumps(json.load(open(LOG))[f"{a.ensemble}_ens"]["val"]), flush=True)
        sys.exit(0)
    feat = featurizer(sysm)
    pols = [x for x in a.pol.split(",") if x]
    t0 = time.time()
    tr, y = train_set(sysm, a.lab, pols)
    vt = {k: v[k] for k in KEYS}
    vt["obj"] = v["obj"]
    vt["u_target"] = canonical_labels(sysm, v["u"], v["u0"])
    print(f"{a.name}: {len(y)} training instances ({a.lab} MILP-labelled, polished {pols}), data {time.time() - t0:.0f}s",
          flush=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        tag = f"{a.name}_s{s}"
        path = os.path.join(DATA, "models", f"{tag}.pt")
        if os.path.exists(path):
            m1, t_train = load_member(path, sysm, feat), None
        else:
            m1 = build_gnnt(sysm, feat, T, seed=s)
            t1 = time.time()
            train_uc_bce(m1, tr, vt, y, epochs=a.epochs, seed=s, log_every=10)
            t_train = time.time() - t1
            torch.save(m1.net.state_dict(), path)
        t1 = time.time()
        p = m1.predict({k: v[k] for k in KEYS}).astype(np.float64)
        t_inf = (time.time() - t1) / len(p)
        np.savez_compressed(os.path.join(DATA, "probs", f"{tag}_probs.npz"), val=p.astype(np.float32))
        rec = dict(name=a.name, seed=s, n_train=int(len(y)), n_lab=a.lab, pol=pols, epochs=a.epochs, train_s=t_train,
                   infer_s_per_inst=t_inf, val=metrics(sysm, p, v))
        update_log(tag, rec)
        print(tag, f"train {t_train}s", json.dumps(rec["val"]), flush=True)
