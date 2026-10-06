"""m1x study, training: Model 1 probability models on 500-4000 12-hour instances, and their validation metrics.

    python3 scripts/uc_m1x_train.py --label pol --n 2000 --kind gnn --seed 0
    python3 scripts/uc_m1x_train.py --metrics_only TAG [TAG ...]       # recompute validation metrics

Training data: the 500 instances of train.npz plus the first (n - 500) / 250 chunks of data/generated/uc12_m1x
(scenarios drawn like train.npz, no full MILP). Labels (always canonicalised inside identical-unit groups):
  lf   label-free labels everywhere (rounded LP relaxation -> adequacy -> min up/down repair; V4)
  pol  MILP labels of the 500 original instances + teacher-polished labels of the extra ones (uc_m1x_data.py --polish)
Model kinds: gnn (CommitGNN, the recipe of uc_model1_4.pt), gnnt (CommitGNNT: + temporal head), mlp (CommitMLP).
Features normalised with the statistics of the 500 original training instances (the existing models' featurizer).
BCE, AdamW 1e-3, batch 64, cosine schedule, early stopping on the log-loss of val.npz (60, canonical MILP labels);
epochs by size: 500 -> 80, 1000 -> 60, 2000 -> 40, 4000 -> 30 (more steps for more data).
Output: data/generated/uc12_m1x/models/<tag>.pt, data/generated/uc12_m1x/probs/<tag>_probs.npz (validation:
va 60 + vx 120 + vx2 180, in that order), results/uc12/m1x_train.json (per tag: data, time, validation metrics).
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
from otsl.m1x import RELAX_KEYS, aligned_labels, build_model, lf_label, prob_metrics, scenario  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import UCFeaturizer, canonical_labels, train_uc_bce  # noqa: E402
from uc_constrained import strip  # noqa: E402

ROOT, DATA, RES = "data/generated/uc12", "data/generated/uc12_m1x", "results/uc12"
EPOCHS = {500: 80, 1000: 60, 2000: 40, 4000: 30}
VAL = (("va", "val"), ("vx", "val_extra"), ("vx2", "val_extra2"))
LOG = os.path.join(RES, "m1x_train.json")
REG = os.path.join(DATA, "sources.json")
REF_MODELS = {"ref_bce_s0": ["uc_model1_4.pt"], "ref_bce_s1": ["hybrid_bce_s1.pt"], "ref_bce_s2": ["hybrid_bce_s2.pt"],
              "ref_bce_ens3": ["uc_model1_4.pt", "hybrid_bce_s1.pt", "hybrid_bce_s2.pt"]}


def register(name, members):
    """probability source -> list of (model path, kind); the eval recomputes (and times) every member"""
    reg = json.load(open(REG)) if os.path.exists(REG) else {}
    reg[name] = members
    json.dump(reg, open(REG, "w"), indent=1)


def train_set(sysm, n, label):
    tr = load(os.path.join(ROOT, "train.npz"))
    parts = [{k: tr[k] for k in RELAX_KEYS}]
    if label == "lf":
        ys = [np.stack([lf_label(tr["u_rel"][i], scenario(tr, i), sysm) for i in range(len(tr["load"]))])]
    else:
        ys = [tr["u"]]
    k = 0
    while sum(len(p["load"]) for p in parts) < n:
        d = dict(np.load(os.path.join(DATA, f"extra_{k}.npz")))
        parts.append({key: d[key] for key in RELAX_KEYS})
        if label == "lf":
            ys.append(d["y_lf"])
        else:
            ys.append(np.load(os.path.join(DATA, f"polish_{k}.npz"))["y"])
        k += 1
    d = {key: np.concatenate([p[key] for p in parts])[:n] for key in RELAX_KEYS}
    y = np.concatenate(ys)[:n]
    d["obj"] = d["c_rel"]
    return d, canonical_labels(sysm, y, d["u0"]).astype(np.float32)


def val_sets():
    parts = [load(os.path.join(ROOT, f"{v}.npz")) for _, v in VAL]
    return {k: np.concatenate([p[k] for p in parts]) for k in RELAX_KEYS + ("u", "obj")}


def featurizer(sysm):
    return UCFeaturizer(sysm, strip(load(os.path.join(ROOT, "train.npz"))), relax=True, sym=True)


def load_net(path, kind, sysm, feat):
    m1 = build_model(kind, sysm, feat, 12, seed=0)
    m1.net.load_state_dict(torch.load(path))
    m1.net.eval()
    return m1


def val_metrics(sysm, p, va, std=None):
    ya = aligned_labels(sysm, va["u"], va["u0"], p)
    yc = canonical_labels(sysm, va["u"], va["u0"])
    out = {"all360": prob_metrics(p, ya, yc, std)}
    out["vx2_180"] = prob_metrics(p[180:], ya[180:], yc[180:])
    return out


def update_log(tag, rec):
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    if rec.get("train_s", 0) is None and log.get(tag, {}).get("train_s"):    # re-evaluated checkpoint: keep its time
        rec["train_s"] = log[tag]["train_s"]
    log[tag] = rec
    json.dump(log, open(LOG, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="pol", choices=["lf", "pol"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--kind", default="gnn", choices=["gnn", "gnnt", "mlp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metrics_only", nargs="*")
    ap.add_argument("--ensemble", nargs=2, metavar=("NAME", "TAGS"))
    ap.add_argument("--refs", action="store_true", help="metrics of the existing BCE / self-trained / kNN probabilities")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    os.makedirs(os.path.join(DATA, "models"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "probs"), exist_ok=True)
    sysm = load_rts_gmlc()
    feat = featurizer(sysm)
    va = val_sets()
    if a.refs:      # existing probability sources (results/uc12/hybrid_probs.npz), same metrics
        H = np.load(os.path.join(RES, "hybrid_probs.npz"))
        cat = lambda k: np.concatenate([H[f"{k}_{s}"] for s, _ in VAL]).astype(np.float64)
        srcs = {f"ref_{k}": cat(k) for k in ("bce_s0", "bce_s1", "bce_s2", "st_s0", "knn")}
        srcs["ref_bce_ens3"] = np.mean([srcs[f"ref_bce_s{s}"] for s in range(3)], 0)
        for tag, fs in REF_MODELS.items():
            register(tag, [[os.path.join(RES, f), "gnn"] for f in fs])
        for tag, p in srcs.items():
            np.savez_compressed(os.path.join(DATA, "probs", f"{tag}_probs.npz"), val=p.astype(np.float32))
            std = np.std([srcs[f"ref_bce_s{s}"] for s in range(3)], 0) if tag == "ref_bce_ens3" else None
            update_log(tag, dict(label="milp" if "bce" in tag else tag[4:], n=500, kind="ref", val=val_metrics(sysm, p, va, std)))
            print(tag, json.dumps(json.load(open(LOG))[tag]["val"]["all360"]), flush=True)
        sys.exit(0)
    if a.ensemble:  # NAME TAG1,TAG2,...: mean probability of the members, disagreement = std over members
        name, tags = a.ensemble[0], a.ensemble[1].split(",")
        reg = json.load(open(REG))
        P = np.stack([np.load(os.path.join(DATA, "probs", f"{t}_probs.npz"))["val"].astype(np.float64) for t in tags])
        p, std = P.mean(0), P.std(0)
        np.savez_compressed(os.path.join(DATA, "probs", f"{name}_probs.npz"), val=p.astype(np.float32), std=std.astype(np.float32))
        register(name, [mem for t in tags for mem in reg[t]])
        log = json.load(open(LOG))
        update_log(name, dict(label=log[tags[0]].get("label"), n=log[tags[0]].get("n"), kind="ens:" + ",".join(tags),
                              val=val_metrics(sysm, p, va, std)))
        print(name, json.dumps(json.load(open(LOG))[name]["val"]["all360"]), flush=True)
        sys.exit(0)
    if a.metrics_only is not None:
        log = json.load(open(LOG))
        for tag in a.metrics_only:
            p = np.load(os.path.join(DATA, "probs", f"{tag}_probs.npz"))["val"]
            log[tag]["val"] = val_metrics(sysm, p, va)
            update_log(tag, log[tag])
            print(tag, json.dumps(log[tag]["val"]["all360"]), flush=True)
        sys.exit(0)
    tag = a.tag or f"{a.label}_n{a.n}_{a.kind}_s{a.seed}"
    path = os.path.join(DATA, "models", f"{tag}.pt")
    epochs = a.epochs or EPOCHS[a.n]
    t0 = time.time()
    tr, y = train_set(sysm, a.n, a.label)
    vtr = load(os.path.join(ROOT, "val.npz"))
    vtr["u_target"] = canonical_labels(sysm, vtr["u"], vtr["u0"])
    t_data = time.time() - t0
    if os.path.exists(path):
        m1 = load_net(path, a.kind, sysm, feat)
        t_train = None
    else:
        m1 = build_model(a.kind, sysm, feat, 12, seed=a.seed)
        t1 = time.time()
        train_uc_bce(m1, tr, vtr, y, epochs=epochs, seed=a.seed, log_every=10)
        t_train = time.time() - t1
        torch.save(m1.net.state_dict(), path)
    t1 = time.time()
    p = m1.predict(va).astype(np.float64)
    t_pred = (time.time() - t1) / len(p)
    np.savez_compressed(os.path.join(DATA, "probs", f"{tag}_probs.npz"), val=p.astype(np.float32))
    rec = dict(label=a.label, n=a.n, kind=a.kind, seed=a.seed, epochs=epochs, train_s=t_train, data_s=t_data,
               infer_s_per_inst=t_pred, params=int(sum(q.numel() for q in m1.net.parameters())),
               label_agree_train_milp_500=None, val=val_metrics(sysm, p, va))
    update_log(tag, rec)
    register(tag, [[path, a.kind]])
    print(tag, f"train {t_train}s", json.dumps(rec["val"]["all360"]), flush=True)
