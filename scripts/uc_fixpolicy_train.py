"""Train the harm-ranked fixing policy and calibrate the Learning-to-Fix baseline (B2, uc12).

    python scripts/uc_fixpolicy_train.py

Inputs: results/uc12/fixpolicy_probs.npz (out-of-fold / saved-model probabilities) and
results/uc12/fixpolicy_labels.pkl (dispatch-LP harm of every wrong fix, LtF impact curves).
Train set: 500 train instances (out-of-fold predictions); model selection: 60 val instances.
Output: results/uc12/fixpolicy_policy.pt (harm model, LtF threshold table) and val proxy metrics
(results/uc12/fixpolicy_train.json): measured harm of the decisions each rule would fix.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.fixpolicy import FixFeaturizer, HarmEnsemble, HarmModel, fix_from_thresholds, fixed_harm, ltf_sweeps, ltf_thresholds  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402


def harm_array(labels, split, ids, T, G, comp=None):
    """measured harm per decision (0 if the prediction agrees with the aligned optimum); with comp, wrong OFF
    fixes take the compensated harm (replacement units allowed)"""
    H = np.zeros((len(ids), T, G))
    W = np.zeros((len(ids), T, G), bool)
    for k, i in enumerate(ids):
        c = comp[(split, i)]["single_comp"] if comp is not None else {}
        for (t, g), h in labels[(split, i)]["single"].items():
            H[k, t, g] = max(c.get((t, g), h), 0.0)
            W[k, t, g] = True
    return H, W


def ltf_table(curves, G, p_cal, taus):
    """thresholds per tolerance and the mean fixed share they give on the calibration probabilities"""
    rows = []
    sw = ltf_sweeps(curves, G)
    for tau in taus:
        th = ltf_thresholds(sw, tau)
        share = float((np.maximum(p_cal, 1 - p_cal) >= th[None, None, :]).mean())
        rows.append((float(tau), th, float(share)))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="uncomp", choices=["uncomp", "comp"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--w_reg", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    root, out_dir, T = "data/generated/uc12", "results/uc12", 12
    tr, va = (load(os.path.join(root, f"{s}.npz")) for s in ("train", "val"))
    sysm = load_rts_gmlc()
    G = sysm.G
    P = np.load(os.path.join(out_dir, "fixpolicy_probs.npz"))
    with open(os.path.join(out_dir, "fixpolicy_labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    comp = None
    if a.labels == "comp":
        with open(os.path.join(out_dir, "fixpolicy_labels_comp.pkl"), "rb") as f:
            comp = pickle.load(f)
    ids_tr = sorted(k[1] for k in labels if k[0] == "tr" and (comp is None or k in comp))
    n_tr, n_va = len(ids_tr), len(va["load"])
    p_tr, p_va = P["bce_oof_tr"], P["bce_va"]
    H_tr, W_tr = harm_array(labels, "tr", ids_tr, T, G, comp)
    H_va, W_va = harm_array(labels, "va", range(n_va), T, G, comp)
    ff = FixFeaturizer(sysm, T)
    X_tr = np.stack([ff(p_tr[i], tr, i) for i in ids_tr])
    X_va = np.stack([ff(p_va[i], va, i) for i in range(n_va)])
    info = {"labels": a.labels, "n_train": n_tr, "n_val": n_va, "features": FixFeaturizer.names,
            "wrong_per_instance_train": float(W_tr.sum((1, 2)).mean()),
            "wrong_per_instance_val": float(W_va.sum((1, 2)).mean()),
            "harmful_per_instance_train": float((H_tr > 1e-4).sum((1, 2)).mean()),
            "harmful_per_instance_val": float((H_va > 1e-4).sum((1, 2)).mean())}
    print(info, flush=True)

    # ---------------------------------------------------------------- harm model
    members = []
    for seed in range(a.seeds):
        hm = HarmModel(X_tr.shape[-1], hidden=a.hidden, seed=seed)
        hm.fit(X_tr.reshape(-1, X_tr.shape[-1]), H_tr.reshape(-1), X_va, H_va, epochs=a.epochs, lr=a.lr, wd=a.wd,
               w_reg=a.w_reg, seed=seed, log=None)
        print(f"  [harm] seed {seed}: best val harm-in-fixed-set {hm.best_val:.4f} "
              f"(curve {' '.join(f'{v:.2f}' for v in hm.hist)})", flush=True)
        members.append(hm)
    ens = HarmEnsemble(members)
    r_va = ens.score(X_va)
    info["val_crit"] = {"ensemble": fixed_harm(r_va, H_va), "RACLearn": fixed_harm(np.minimum(p_va, 1 - p_va), H_va),
                        "members": [m.best_val for m in members]}
    print("  val criterion", info["val_crit"], flush=True)

    # ---------------------------------------------------------------- Learning to Fix thresholds
    curves_tr = [labels[("tr", i)]["ltf"] for i in ids_tr]
    curves_va = [labels[("va", i)]["ltf"] for i in range(n_va)]
    taus = np.r_[0.0, np.geomspace(1e-7, 1.0, 400)]
    ltf_pool = ltf_table(curves_tr + curves_va, G, p_va, taus)
    ltf_val = ltf_table(curves_va, G, p_va, taus)

    # ---------------------------------------------------------------- val proxies (no MILP)
    def proxy(score, ratio):
        k = int(round(ratio * T * G))
        o = np.argsort(score.reshape(n_va, -1), 1, kind="stable")[:, :k]
        h = np.take_along_axis(H_va.reshape(n_va, -1), o, 1)
        w = np.take_along_axis(W_va.reshape(n_va, -1), o, 1)
        return float(h.sum(1).mean() * 100), float(w.sum(1).mean()), float((h > 1e-4).sum(1).mean())

    def proxy_fix(fixes):
        h = [sum(H_va[i, t, g] for (t, g) in fx) for i, fx in enumerate(fixes)]
        w = [sum(W_va[i, t, g] for (t, g) in fx) for i, fx in enumerate(fixes)]
        return float(np.mean(h) * 100), float(np.mean(w)), float(np.mean([len(fx) / (T * G) for fx in fixes]))

    conf_err = np.minimum(p_va, 1 - p_va)
    rows = []
    for ratio in (0.8, 0.9, 0.95, 0.97):
        for name, sc in (("RACLearn (margin)", conf_err), ("asymmetric k=10 (no guard)", conf_err * np.where(p_va < 0.5, 10, 1)),
                         ("harm policy", r_va)):
            hs, nw, nh = proxy(sc, ratio)
            rows.append(dict(ratio=ratio, method=name, harm_sum_pct=hs, wrong_fixed=nw, harmful_fixed=nh))
        for lab, tab in (("LtF (pooled calib)", ltf_pool), ("LtF (val calib)", ltf_val)):
            j = int(np.argmin([abs(s - ratio) for _, _, s in tab]))
            fixes = [fix_from_thresholds(p_va[i], tab[j][1]) for i in range(n_va)]
            hs, nw, sh = proxy_fix(fixes)
            rows.append(dict(ratio=ratio, method=f"{lab} tau={tab[j][0]:.2g} share={sh:.3f}", harm_sum_pct=hs, wrong_fixed=nw))
    for r in rows:
        print(f"  {r['ratio']:.2f}  {r['method']:44s} measured harm in fixed set {r['harm_sum_pct']:9.3f}%  "
              f"wrong fixes {r['wrong_fixed']:.2f}", flush=True)
    torch.save({"members": [{"state": m.net.state_dict(), "mu": m.mu, "sd": m.sd} for m in members],
                "hidden": a.hidden, "d_in": X_tr.shape[-1], "args": vars(a),
                "ltf_pool": [(t, th.tolist(), s) for t, th, s in ltf_pool],
                "ltf_val": [(t, th.tolist(), s) for t, th, s in ltf_val]}, os.path.join(out_dir, f"fixpolicy_policy_{a.labels}.pt"))
    info["val_proxy"] = rows
    info["ltf_pool_share_by_tau"] = [(t, s) for t, _, s in ltf_pool]
    info["train_s"] = time.time() - t0
    with open(os.path.join(out_dir, f"fixpolicy_train_{a.labels}.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)
