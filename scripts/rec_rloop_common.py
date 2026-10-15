"""rloop (arm B): shared paths, data loading and model I/O for scripts/rec_rloop_*.py.

Benchmarks (all read only):
  uc12      data/generated/uc12 (train 500 MILP labels; val + val_extra + val_extra2 = 360; test_fresh 120) and the
            1,500 polished extra instances of data/generated/uc12_m1x (extra_k + polish_k, k = 0..5)
  uc24      data/generated/uc24/test.npz (40 instances, 24 h)
  hard      data/generated/hard/{val,test}_XX.npz (60 + 60, 12 h, line ratings x 0.6)
  line_out  data/generated/uc12_ood/line_out/NNN.npz (30, 1-2 critical lines out)
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from otsl import rloop  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402

D12, DM1X = os.path.join(ROOT, "data/generated/uc12"), os.path.join(ROOT, "data/generated/uc12_m1x")
OUT = os.path.join(ROOT, "data/generated/recursive/rloop")          # git-ignored: checkpoints, caches
RES = os.path.join(ROOT, "results/recursive")
KEYS = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel")
VAL_SETS = ("val", "val_extra", "val_extra2")
LINE_SCALE = {"uc12": 1.0, "uc24": 1.0, "hard": 0.6, "line_out": 1.0}


def _npz(path):
    return dict(np.load(path, allow_pickle=False))


def label_chunks(data_dirs=()):
    """(extra, polish) file pairs: uc12_m1x's 6 polished chunks, then those of each extra data directory (same format:
    extra_<k>.npz with the LP-relaxation fields of KEYS, polish_<k>.npz with the label y)"""
    pairs = [(os.path.join(DM1X, f"extra_{k}.npz"), os.path.join(DM1X, f"polish_{k}.npz")) for k in range(6)]
    for dd in data_dirs:
        for f in sorted(glob.glob(os.path.join(dd, "extra_*.npz"))):
            pf = os.path.join(dd, "polish_" + os.path.basename(f)[len("extra_"):])
            if os.path.exists(pf):
                pairs.append((f, pf))
    return pairs


def train_data(n=2000, data_dirs=()):
    """training instances: 500 with MILP labels (train.npz) + 1,500 with polished labels (uc12_m1x) [+ polished
    chunks of data_dirs, for larger runs]"""
    tr = _npz(os.path.join(D12, "train.npz"))
    parts = [{k: tr[k] for k in KEYS}]
    ys = [tr["u"]]
    for fe, fp in label_chunks(data_dirs):
        if sum(len(p["load"]) for p in parts) >= n:
            break
        d = _npz(fe)
        parts.append({key: d[key] for key in KEYS})
        ys.append(_npz(fp)["y"])
    d = {key: np.concatenate([p[key] for p in parts])[:n] for key in KEYS}
    y = np.concatenate(ys)[:n].astype(np.int8)
    return d, y


def label_costs(d, y, tag="train2000"):
    """dispatch-LP cost of each label (best known cost; cached under OUT)"""
    path = os.path.join(OUT, f"label_costs_{tag}.npy")
    if os.path.exists(path):
        c = np.load(path)
        if len(c) == len(y):
            return c
    sysm = load_rts_gmlc()
    lp = rloop.FastLP(sysm, y.shape[1])
    c = np.array([lp.solve(rloop.scenario_of(x), y[i])["obj"] for i, x in enumerate(insts_of(d))])
    os.makedirs(OUT, exist_ok=True)
    np.save(path, c)
    return c


def val_data():
    """val + val_extra + val_extra2 (360), with the stored MILP objective, its dual bound and label"""
    parts = [_npz(os.path.join(D12, f"{v}.npz")) for v in VAL_SETS]
    d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("u", "obj", "gap")}
    d["db"] = d["obj"] * (1 - d["gap"])
    return d


def test_fresh():
    d = _npz(os.path.join(D12, "test_fresh.npz"))
    d["db"] = d["obj"] * (1 - d["gap"])
    return d


def bench_data(name, split="test"):
    """dict of stacked arrays for an evaluation set; 'db' = stored dual bound (for selection only; the test runs
    re-solve the full MILP)"""
    if name == "uc12":
        return test_fresh() if split == "test" else val_data()
    if name == "uc24":
        d = _npz(os.path.join(ROOT, "data/generated/uc24", f"{split}.npz"))
        d["db"] = d["bound"]
        return d
    if name == "hard":
        fs = sorted(glob.glob(os.path.join(ROOT, "data/generated/hard", f"{split}_[0-9][0-9].npz")))
        parts = [_npz(f) for f in fs]
        d = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("u", "obj", "bound", "gap", "time")}
        d["db"] = d["bound"]
        return d
    if name == "line_out":
        fs = sorted(glob.glob(os.path.join(ROOT, "data/generated/uc12_ood/line_out", "[0-9][0-9][0-9].npz")))
        parts = [_npz(f) for f in fs]
        d = {k: np.stack([p[k] for p in parts]) for k in KEYS}
        d["l_out"] = [p["l_out"] for p in parts]
        d["g_out"] = [p["g_out"] for p in parts]
        d["obj"] = np.array([float(p["obj_full"]) for p in parts])
        d["db"] = np.array([float(p["bound_full"]) for p in parts])
        return d
    raise ValueError(name)


def insts_of(d, idx=None):
    n = len(d["load"])
    idx = range(n) if idx is None else idx
    out = []
    for i in idx:
        x = {k: d[k][i] for k in KEYS}
        x["c_rel"] = float(x["c_rel"])
        for k in ("l_out", "g_out"):
            if k in d:
                x[k] = np.asarray(d[k][i], int)
        out.append(x)
    return out


def system(name="uc12"):
    return load_rts_gmlc(line_scale=LINE_SCALE[name])


def feats_for(sysm0=None, sym_rank=False):
    """static featurizer, normalised on the first 400 training instances (original system)"""
    sysm0 = sysm0 or load_rts_gmlc()
    tr, _ = train_data(500)
    return rloop.StaticFeats(sysm0, insts_of(tr, range(400)), sym_rank=sym_rank)


def model_cfg(preset, **over):
    cfg = dict(rloop.PRESETS[preset])
    cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg


def save_ckpt(path, net, ema, opt, sched, cfg, rec, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(dict(net=net.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict() if opt else None,
                    sched=sched.state_dict() if sched else None, cfg=cfg, rec=rec, extra=extra or {}), tmp)
    os.replace(tmp, path)


def load_model(path, sysm, feats=None, device="cpu", use_ema=True):
    """(net, checkpoint, featurizer); the featurizer is rebuilt from the checkpoint's config unless given"""
    ck = torch.load(path, map_location=device, weights_only=False)
    feats = feats or feats_for(load_rts_gmlc(), sym_rank=ck["cfg"].get("sym_rank", False))
    net = rloop.build_net(sysm, feats, ck["cfg"]).to(device)
    net.load_state_dict(ck["ema"] if use_ema else ck["net"])
    net.eval()
    return net, ck, feats


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []
