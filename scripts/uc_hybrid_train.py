"""Hybrid study, step 0: seed replicates of the MILP-label BCE GNN (the probability source chosen for Learning to Fix
by the paper's own criterion, the largest validation fixed share at eps = 1 %, see docs/methods/hybrid.md).

    python3 scripts/uc_hybrid_train.py --seeds 1,2

Recipe of results/uc12/uc_model1_4.pt (scripts/uc_model1.py, variant "GNN + symmetry + LP-relaxation features"):
UCFeaturizer(relax=True, sym=True) on the training set, build_uc_model1(kind="gnn", seed=s), train_uc_bce on the
canonical MILP labels, 80 epochs, AdamW 1e-3, batch 64, early stopping on the validation log-loss (val.npz, 60
instances, canonical labels). Seed 0 is the existing checkpoint uc_model1_4.pt; seeds 1, 2 are new.
Output: results/uc12/hybrid_bce_s<seed>.pt, results/uc12/hybrid_train.json
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
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, canonical_labels, train_uc_bce  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    tr, va = load(os.path.join(ROOT, "train.npz")), load(os.path.join(ROOT, "val.npz"))
    canon_tr, canon_va = canonical_labels(sysm, tr["u"], tr["u0"]), canonical_labels(sysm, va["u"], va["u0"])
    path = os.path.join(OUT, "hybrid_train.json")
    info = json.load(open(path)) if os.path.exists(path) else {}
    for s in [int(x) for x in a.seeds.split(",")]:
        t0 = time.time()
        feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
        m1 = build_uc_model1(sysm, feat, 12, "gnn", seed=s)
        train_uc_bce(m1, tr, dict(va, u_target=canon_va), canon_tr, epochs=a.epochs, seed=s)
        torch.save(m1.net.state_dict(), os.path.join(OUT, f"hybrid_bce_s{s}.pt"))
        info[f"bce_s{s}"] = {"train_s": time.time() - t0, "epochs": a.epochs}
        print(f"seed {s}: {time.time() - t0:.0f}s", flush=True)
        json.dump(info, open(path, "w"), indent=1)
