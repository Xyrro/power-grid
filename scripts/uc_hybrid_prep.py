"""Hybrid study, step 1: probabilities and error-cost scores on the validation and test instances.

    python3 scripts/uc_hybrid_prep.py

Splits: va (val, 60), vx (val_extra, 120), vx2 (val_extra2, 180; if generated), tf (test_fresh, 120).
kNN of the paper (k = 50, Table II features; otsl.ltfx.KNNProb). Probability sources, seeds 0-2: MILP-label BCE GNN (seed 0 = uc_model1_4.pt, seeds 1-2 = hybrid_bce_s<s>.pt),
self-trained GNN (combo_st_s<s>.pt). Error-cost scores: harm ensemble seed s (combo_harm_s<s>.pt) on the features of
the same seed's probabilities (otsl.combo.harm_scores). Normalisation of log h for the score transform of
otsl.hybrid.harm_to_score: mean / std over the decisions of the first 180 validation instances (val + val_extra),
fixed once, so 180- and 360-instance tunings share it.
Output: results/uc12/hybrid_probs.npz (keys <src>_s<seed>_<split>, harm_<src>_s<seed>_<split>),
results/uc12/hybrid_probs.json (norms, log-losses, wrong decisions at 0.5).
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import harm_from_file, harm_scores  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import harm_norm  # noqa: E402
from otsl.ltfx import KNNProb  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import canonical_labels  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
SPLITS = {"va": "val", "vx": "val_extra", "vx2": "val_extra2", "tf": "test_fresh"}


def model_paths():
    out = {}
    for s in range(3):
        f = "uc_model1_4.pt" if s == 0 else f"hybrid_bce_s{s}.pt"
        if os.path.exists(os.path.join(OUT, f)):
            out[f"bce_s{s}"] = f
        out[f"st_s{s}"] = f"combo_st_s{s}.pt"
    return out


def logloss(p, y, clip=1e-3):
    p = np.clip(p, clip, 1 - clip)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    t0 = time.time()
    sysm = load_rts_gmlc()
    tr = load(os.path.join(ROOT, "train.npz"))
    trs = strip(tr)
    data = {k: load(os.path.join(ROOT, f"{v}.npz")) for k, v in SPLITS.items() if os.path.exists(os.path.join(ROOT, f"{v}.npz"))}
    ff = FixFeaturizer(sysm, 12)
    out, info = {}, {"splits": {k: int(len(d["load"])) for k, d in data.items()}}
    knn = KNNProb(sysm, tr, k=50)                     # the paper's classifier (for held-out checks of its thresholds)
    for k, d in data.items():
        out[f"knn_{k}"] = knn.predict(d)
    for name, f in model_paths().items():
        m1 = load_model1(os.path.join(OUT, f), sysm, trs, 12)
        src, seed = name.split("_s")
        harm = harm_from_file(os.path.join(OUT, f"combo_harm_s{seed}.pt"))
        for k, d in data.items():
            p = m1.predict(d).astype(np.float64)
            out[f"{name}_{k}"] = p
            out[f"harm_{name}_{k}"] = harm_scores(harm, ff, p, d, list(range(len(p)))).astype(np.float64)
            if k != "tf":
                y = canonical_labels(sysm, d["u"], d["u0"])
                info[f"{name}_{k}"] = {"logloss": logloss(p, y), "wrong_at_0.5_per_inst": float(((p > 0.5) != y).sum((1, 2)).mean())}
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
    norms = {}
    for name in model_paths():
        hv = np.concatenate([out[f"harm_{name}_va"], out[f"harm_{name}_vx"]])
        norms[name] = harm_norm(hv)
    info["harm_norm_logh_mean_sd"] = norms
    old = np.load(os.path.join(OUT, "ltfx_probs.npz"))
    info["max_abs_diff_vs_ltfx_probs"] = {f"{a}_{k}": float(np.abs(out[f"{a}_{k}"] - old[f"{b}_{k}"]).max())
                                          for a, b in (("bce_s0", "bce"), ("knn", "knn")) for k in ("va", "vx", "tf")}
    np.savez_compressed(os.path.join(OUT, "hybrid_probs.npz"), **out)
    info["wall_s"] = time.time() - t0
    json.dump(info, open(os.path.join(OUT, "hybrid_probs.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
