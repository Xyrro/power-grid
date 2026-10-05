"""Learning to Fix reconstruction, step 1: probabilities of every model on val / test / test_fresh, and the kNN
classifier's hyperparameters chosen on val.

    python scripts/uc_ltf_prep.py

kNN grid (val only): feature set {sys, bus} x k {5, 11, 21, 41, 81} x initial-status weight {0.5, 1, 2} x
{uniform, 1/distance}; criterion: val log-loss (probabilities clipped to [1e-3, 1 - 1e-3]) against the
canonicalised MILP labels. GNN probabilities: uc_model1_4.pt (MILP-label BCE), selftrain_m3.pt (self-trained,
round 3), uc_model1_rl.pt (REINFORCE), all rebuilt with UCFeaturizer(train, relax=True, sym=True).
Output: results/uc12/ltf_probs.npz, results/uc12/ltf_knn_select.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.ltf import KNNCommit  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, canonical_labels  # noqa: E402

ROOT, OUT, T = "data/generated/uc12", "results/uc12", 12
GNNS = {"bce": "uc_model1_4.pt", "st": "selftrain_m3.pt", "rl": "uc_model1_rl.pt"}


def logloss(p, y, clip=1e-3):
    p = np.clip(p, clip, 1 - clip)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def conf_err_curve(p, y, shares=(0.8, 0.9, 0.95)):
    """wrong decisions per instance among the share of most confident decisions (ties broken at random)"""
    n = len(p)
    conf = np.maximum(p, 1 - p).reshape(n, -1) + 1e-9 * np.random.default_rng(0).random((n, p[0].size))
    wrong = ((p > 0.5) != y).reshape(n, -1)
    out = {}
    for q in shares:
        k = int(q * conf.shape[1])
        o = np.argsort(-conf, 1)[:, :k]
        out[q] = float(np.take_along_axis(wrong, o, 1).sum(1).mean())
    return out


if __name__ == "__main__":
    t0 = time.time()
    sysm = load_rts_gmlc()
    tr, va, te = (load(os.path.join(ROOT, f"{s}.npz")) for s in ("train", "val", "test"))
    splits = {"va": va, "te": te}
    fresh = os.path.join(ROOT, "test_fresh.npz")
    if os.path.exists(fresh):
        splits["tf"] = load(fresh)
    y_va = canonical_labels(sysm, va["u"], va["u0"])
    rows = []
    for kind in ("sys", "bus"):
        for k in (5, 11, 21, 41, 81):
            for w in (0.5, 1.0, 2.0):
                for weighted in (False, True):
                    knn = KNNCommit(sysm, tr, k=k, kind=kind, w_u0=w, weighted=weighted)
                    p = knn.predict(va)
                    ce = conf_err_curve(p, y_va)
                    rows.append(dict(kind=kind, k=k, w_u0=w, weighted=weighted, logloss=logloss(p, y_va),
                                     acc=float(((p > 0.5) == y_va).mean()), wrong=float(((p > 0.5) != y_va).sum((1, 2)).mean()),
                                     wrong90=ce[0.9], wrong95=ce[0.95]))
    rows.sort(key=lambda r: r["logloss"])
    for r in rows[:10]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    best = rows[0]
    knn = KNNCommit(sysm, tr, k=best["k"], kind=best["kind"], w_u0=best["w_u0"], weighted=best["weighted"])
    out = {}
    for name, d in splits.items():
        out[f"knn_{name}"] = knn.predict(d)
    out["knn_oof_tr"] = knn.predict(tr, exclude_same_day=True)
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    for tag, path in GNNS.items():
        m1 = build_uc_model1(sysm, feat, T, "gnn", seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(OUT, path)))
        for name, d in splits.items():
            out[f"{tag}_{name}"] = m1.predict(d).astype(np.float32)
    old = np.load(os.path.join(OUT, "fixpolicy_probs.npz"))
    print("check vs fixpolicy_probs: bce_te max abs diff", float(np.abs(old["bce_te"] - out["bce_te"]).max()),
          "rl_te", float(np.abs(old["rl_te"] - out["rl_te"]).max()), flush=True)
    summ = {}
    for name, d in splits.items():
        y = canonical_labels(sysm, d["u"], d["u0"])
        for tag in ("knn", "bce", "st", "rl"):
            p = out[f"{tag}_{name}"]
            summ[f"{tag}_{name}"] = dict(logloss=logloss(p, y), wrong=float(((p > 0.5) != y).sum((1, 2)).mean()),
                                         **{f"wrong_top{int(q * 100)}": v for q, v in conf_err_curve(p, y).items()})
            print(tag, name, {k: round(v, 3) for k, v in summ[f"{tag}_{name}"].items()}, flush=True)
    np.savez_compressed(os.path.join(OUT, "ltf_probs.npz"), **out)
    with open(os.path.join(OUT, "ltf_knn_select.json"), "w") as f:
        json.dump(dict(best=best, grid=rows, summary=summ, splits=list(splits)), f, indent=1)
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)
