"""uc24ltf, step 2: on-probabilities for Learning to Fix on uc24 (validation with full MILPs, and test).

    python3 scripts/uc_uc24ltf_prep.py [--train_lab data/generated/uc24/uc24ltf_train_lab.npz]

* kNN (the paper's classifier, otsl.ltfx.KNNProb): k = 50, Euclidean distance on the Table II features (z-scored on
  the labelled training subset), inverse-distance weights (eq. 13), labels = the 0.5 %-gap MILP schedules of the
  labelled training subset (scripts/uc_uc24ltf_gen.py), canonicalised inside groups of identical units.
* our label-free GNNs (no MILP label in training or selection; scripts/uc_b3_train.py):
  bce = imitation of the block-repaired LP relaxation (results/uc24/b3_lf_bce.pt),
  rl  = + REINFORCE with the dispatch-LP critic (results/uc24/b3_rl_selected.pt).
  Pre-declared choice of THE GNN for Learning to Fix: the lower validation log-loss against the canonical
  validation MILP schedules (clip 1e-3), the calibration the threshold tuning relies on.
Diagnostics on validation: log-loss, wrong decisions at 0.5, fixed share of the cheap thresholds (constant
[r, 1 - r], eq. 6; worst-case misprediction, eq. 7), kNN log-loss for other k (not used).
Output: results/uc24/uc24ltf_probs.npz (keys <model>_<va|te>), results/uc24/uc24ltf_probs.json,
results/uc24/uc24ltf_cheap_thresholds.json.
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
from otsl.b3 import load_b3  # noqa: E402
from otsl.ltfx import KNNProb, constant_thresholds, fix_masks, worst_case_thresholds  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.uc24ltf import logloss  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, canonical_labels  # noqa: E402
from uc_b3_train import strip  # noqa: E402

ROOT, OUT = os.path.join("data", "generated", "uc24"), os.path.join("results", "uc24")
GNNS = {"bce": "b3_lf_bce.pt", "rl": "b3_rl_selected.pt"}


def load_train_lab(path):
    d = load_b3(path)
    ok = np.isfinite(d["obj"]) & (d["u"].min(axis=(1, 2)) >= 0)
    return {k: (v[ok] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(ok) else v) for k, v in d.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_lab", default=os.path.join(ROOT, "uc24ltf_train_lab.npz"))
    ap.add_argument("--val", default=os.path.join(ROOT, "uc24ltf_val.npz"))
    ap.add_argument("--no_knn", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    t0 = time.time()
    sysm = load_rts_gmlc()
    va = load_b3(a.val)
    te = load_b3(os.path.join(ROOT, "test.npz"))
    data = {"va": va, "te": te}
    out, info = {}, {"val_file": a.val, "n_val": int(len(va["obj"]))}
    old = dict(np.load(os.path.join(OUT, "uc24ltf_probs.npz"))) if os.path.exists(os.path.join(OUT, "uc24ltf_probs.npz")) else {}
    out.update({k: v for k, v in old.items() if k.startswith("knn")})
    yv = canonical_labels(sysm, va["u"], va["u0"])
    if not a.no_knn and os.path.exists(a.train_lab):
        trl = load_train_lab(a.train_lab)
        info["n_train_lab"] = int(len(trl["obj"]))
        info["train_lab_gap_mean_%"] = float(trl["gap"].mean() * 100)
        info["train_lab_time_mean_s"] = float(trl["time"].mean())
        knn = KNNProb(sysm, trl, k=50)
        for k, d in data.items():
            out[f"knn_{k}"] = knn.predict(d)
        diag = {}
        for kk in (5, 10, 20, 50, 100):
            if kk <= len(trl["obj"]):
                diag[str(kk)] = logloss(KNNProb(sysm, trl, k=kk).predict(va), yv)
        info["knn_val_logloss_by_k (diagnostic, not used)"] = diag
    tr = strip(load_b3(os.path.join(ROOT, "train.npz")))
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    for name, f in GNNS.items():
        m1 = build_uc_model1(sysm, feat, 24, "gnn", seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(OUT, f)))
        for k, d in data.items():
            out[f"{name}_{k}"] = m1.predict(d).astype(np.float64)
    ref = np.load(os.path.join(OUT, "b3_probs_test_lf_bce.npy"))
    info["max_abs_diff_vs_b3_probs_test_bce"] = float(np.abs(out["bce_te"] - ref).max())
    models = [m for m in ("knn", "bce", "rl") if f"{m}_va" in out]
    cheap = {}
    for name in models:
        p = out[f"{name}_va"]
        info[f"{name}_va"] = {"logloss": logloss(p, yv), "wrong_at_0.5_per_inst": float(((p > 0.5) != yv).sum((1, 2)).mean()),
                              "share_exact_0_or_1": float(((p == 0) | (p == 1)).mean()),
                              "share_in_[0.01,0.99]": float(((p > 0.01) & (p < 0.99)).mean())}
        lo, hi = worst_case_thresholds(p, yv)
        off, on = fix_masks(p, lo, hi)
        cheap[f"{name}_worst"] = dict(lo=lo.tolist(), hi=hi.tolist(), val_fixed_share=float((off | on).mean()))
        for r in (0.1, 0.05, 0.01):
            lo, hi = constant_thresholds(sysm.G, r)
            off, on = fix_masks(p, lo, hi)
            cheap[f"{name}_const{r:g}"] = dict(lo=lo.tolist(), hi=hi.tolist(), val_fixed_share=float((off | on).mean()),
                                               val_wrong_fixed_per_inst=float((((off & (yv == 1)) | (on & (yv == 0)))).sum((1, 2)).mean()))
        info[f"{name}_va"]["fixed_share_worst"] = cheap[f"{name}_worst"]["val_fixed_share"]
    gl = {g: info[f"{g}_va"]["logloss"] for g in GNNS}
    info["gnn_choice"] = min(gl, key=gl.get)
    info["gnn_choice_rule"] = "lower validation log-loss against the canonical validation MILP schedules (clip 1e-3)"
    np.savez_compressed(os.path.join(OUT, "uc24ltf_probs.npz"), **out)
    json.dump(cheap, open(os.path.join(OUT, "uc24ltf_cheap_thresholds.json"), "w"), indent=1)
    info["wall_s"] = time.time() - t0
    json.dump(info, open(os.path.join(OUT, "uc24ltf_probs.json"), "w"), indent=1)
    print(json.dumps(info, indent=1), flush=True)
