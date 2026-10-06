"""Learning to Fix (faithful), step 1: on-probabilities of the kNN (paper settings) and of our three GNNs on the
validation instances (val + val_extra) and on test_fresh.

    python3 scripts/uc_ltfx_prep.py

kNN: k = 50, Euclidean distance on the Table II features (z-scored on train), inverse-distance weights (eq. 13),
trained on the 500 training instances. GNNs (existing checkpoints, as in docs/methods/ltf.md): MILP-label BCE
(uc_model1_4.pt), self-trained (selftrain_m3.pt), MILP-label REINFORCE (uc_model1_rl.pt).
Output: results/uc12/ltfx_probs.npz (keys <model>_<split>, split in va, vx, tf), results/uc12/ltfx_probs.json.
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
from otsl.ltfx import KNNProb  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import canonical_labels  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
GNNS = {"bce": "uc_model1_4.pt", "st": "selftrain_m3.pt", "rl": "uc_model1_rl.pt"}
SPLITS = {"va": "val", "vx": "val_extra", "tf": "test_fresh"}


def logloss(p, y, clip=1e-3):
    p = np.clip(p, clip, 1 - clip)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    t0 = time.time()
    sysm = load_rts_gmlc()
    tr = load(os.path.join(ROOT, "train.npz"))
    data = {k: load(os.path.join(ROOT, f"{v}.npz")) for k, v in SPLITS.items()}
    out, info = {}, {}
    knn = KNNProb(sysm, tr, k=50)
    for k, d in data.items():
        out[f"knn_{k}"] = knn.predict(d)
    trs = strip(tr)
    for name, f in GNNS.items():
        m1 = load_model1(os.path.join(OUT, f), sysm, trs, 12)
        for k, d in data.items():
            out[f"{name}_{k}"] = m1.predict(d).astype(np.float64)
    old = np.load(os.path.join(OUT, "ltf_probs.npz"))
    for name in GNNS:
        for k in ("va", "tf"):
            info[f"max_abs_diff_vs_ltf_probs_{name}_{k}"] = float(np.abs(out[f"{name}_{k}"] - old[f"{name}_{k}"]).max())
    for k in ("va", "vx"):
        d = data[k]
        y = canonical_labels(sysm, d["u"], d["u0"])
        for name in ["knn"] + list(GNNS):
            p = out[f"{name}_{k}"]
            info[f"{name}_{k}"] = {"logloss": logloss(p, y), "wrong_at_0.5_per_inst": float(((p > 0.5) != y).sum((1, 2)).mean()),
                                   "share_exact_0_or_1": float(((p == 0) | (p == 1)).mean())}
    np.savez_compressed(os.path.join(OUT, "ltfx_probs.npz"), **out)
    info["wall_s"] = time.time() - t0
    json.dump(info, open(os.path.join(OUT, "ltfx_probs.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
