"""Out-of-fold commitment probabilities for the fixing-policy study (B2, uc12).

    python scripts/uc_fixpolicy_crossfit.py --folds 4

The fixing policy is trained on the errors the probability model makes on instances it has NOT seen.
In-sample predictions of the saved model are too accurate (train: 8.8 wrong decisions per instance,
val: 14.8), so the train set is re-predicted by K models trained exactly like uc_model1_4.pt (GNN +
symmetry rank + LP-relaxation features, BCE on canonical labels, 80 epochs, early stopping on val) on
K-1 folds, with folds grouped by calendar day. Val/test probabilities come from the saved weights
(BCE and its REINFORCE fine-tune). Output: results/uc12/fixpolicy_probs.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, canonical_labels, train_uc_bce  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    root, out_dir, T = "data/generated/uc12", "results/uc12", 12
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ("train", "val", "test"))
    sysm = load_rts_gmlc()
    out = {}
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    for tag, path in (("bce", "uc_model1_4.pt"), ("rl", "uc_model1_rl.pt")):
        m1 = build_uc_model1(sysm, feat, T, "gnn", seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(out_dir, path)))
        for name, d in (("tr", tr), ("va", va), ("te", te)):
            out[f"{tag}_{name}"] = m1.predict(d).astype(np.float32)
    canon_tr, canon_va = canonical_labels(sysm, tr["u"], tr["u0"]), canonical_labels(sysm, va["u"], va["u0"])
    days = np.unique(tr["day"])
    rng = np.random.default_rng(0)
    fold_of_day = dict(zip(rng.permutation(days), np.arange(len(days)) % a.folds))
    fold = np.array([fold_of_day[d] for d in tr["day"]])
    oof = np.zeros_like(out["bce_tr"])
    for k in range(a.folds):
        t0 = time.time()
        idx_in, idx_out = np.where(fold != k)[0], np.where(fold == k)[0]
        sub = {key: v[idx_in] for key, v in tr.items() if isinstance(v, np.ndarray) and len(v) == len(tr["load"])}
        feat_k = UCFeaturizer(sysm, sub, relax=True, sym=True)
        m1 = build_uc_model1(sysm, feat_k, T, "gnn", seed=0)
        train_uc_bce(m1, sub, dict(va, u_target=canon_va), canon_tr[idx_in], epochs=a.epochs, seed=0, log_every=40)
        oof[idx_out] = m1.predict({key: v[idx_out] for key, v in tr.items()
                                   if isinstance(v, np.ndarray) and len(v) == len(tr["load"])})
        wrong = ((oof[idx_out] > 0.5) != canon_tr[idx_out]).sum((1, 2)).mean()
        print(f"fold {k}: {len(idx_in)} train / {len(idx_out)} held out, wrong decisions per instance "
              f"{wrong:.2f} ({time.time() - t0:.0f}s)", flush=True)
    out["bce_oof_tr"] = oof
    out["fold"] = fold
    for name, p, y in (("in-sample train", out["bce_tr"], canon_tr), ("out-of-fold train", oof, canon_tr),
                       ("val", out["bce_va"], canon_va)):
        print(f"{name:18s} wrong per instance {((p > 0.5) != y).sum((1, 2)).mean():.2f}", flush=True)
    np.savez_compressed(os.path.join(out_dir, "fixpolicy_probs.npz"), **out)
    print("saved", flush=True)
