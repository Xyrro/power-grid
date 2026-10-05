"""Learning to Fix reconstruction, step 3: generator-specific thresholds from the impact curves.

    python scripts/uc_ltf_calib.py

For every probability model (kNN, MILP-label BCE GNN, self-trained GNN, REINFORCE GNN), impact measure
(LP with / without compensation) and decomposition ("budget": sum of generator impacts <= tau; "each": every
generator's impact <= tau), thresholds are computed on the calibration instances (val, 60) for a grid of
tolerances tau. Seeded variant "knn_bs<s>": bootstrap resample (seed s) of the 60 val calibration instances.
Not run (needs knn_oof_tr curves from uc_ltf_curves.py): "knn_mix<s>", calibration on val + 120 out-of-fold
training instances drawn with seed s from the first 200 (kNN predictions without same-day neighbours).
Output: results/uc12/ltf_thresholds.json (thresholds, calibration impact, fixed share on val / test / test_fresh).
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.ltf import generator_options, generator_tables, thresholds_budget, thresholds_each  # noqa: E402

OUT = "results/uc12"
TAUS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
G = 73


def calibrate(curves, conf_cal, taus=TAUS, impacts=("comp", "nocomp")):
    res = {}
    for which in impacts:
        opts = generator_options(generator_tables(curves, G, which), conf_cal)
        for tau in taus:
            th_b, used, sh = thresholds_budget(opts, tau)
            res[f"budget:{which}:{tau:g}"] = dict(theta=th_b.tolist(), impact_cal=used, share_cal=sh)
            th_e = thresholds_each(opts, tau)
            imp_e = 0.0
            for g, o in enumerate(opts):
                j = [x[0] for x in o].index(th_e[g])
                imp_e += o[j][1]
            res[f"each:{which}:{tau:g}"] = dict(theta=th_e.tolist(), impact_cal=float(imp_e),
                                                share_cal=float((conf_cal >= th_e[None, None]).mean()))
    return res


def share(p, theta):
    return float((np.maximum(p, 1 - p) >= np.asarray(theta)[None, None, :]).mean())


if __name__ == "__main__":
    C = pickle.load(open(os.path.join(OUT, "ltf_curves.pkl"), "rb"))
    P = np.load(os.path.join(OUT, "ltf_probs.npz"))
    splits = [s for s in ("va", "te", "tf") if f"knn_{s}" in P.files]
    out = {}
    for model in ("knn", "bce", "st", "rl"):
        keys = [(f"{model}_va", i) for i in range(60)]
        if not all(k in C for k in keys):
            print("missing curves for", model)
            continue
        curves = [C[k]["curves"] for k in keys]
        conf = np.maximum(P[f"{model}_va"], 1 - P[f"{model}_va"])
        res = calibrate(curves, conf)
        for name, r in res.items():
            for s in splits:
                r[f"share_{s}"] = share(P[f"{model}_{s}"], r["theta"])
        out[model] = res
        n_wrong = np.mean([C[k]["n_wrong"] for k in keys])
        print(f"{model}: {n_wrong:.1f} wrong / val instance", flush=True)
        for name in res:
            if name.startswith("budget:comp") or name.startswith("each:comp"):
                r = res[name]
                print(f"  {model:4s} {name:22s} impact_cal {r['impact_cal'] * 100:7.3f}%  share cal {r['share_cal'] * 100:5.1f}%  "
                      + "  ".join(f"{s} {r[f'share_{s}'] * 100:5.1f}%" for s in splits), flush=True)
    # seeded variant: bootstrap resamples of the 60 calibration instances (seeds 0, 1, 2), kNN, separable budget
    if all(("knn_va", i) in C for i in range(60)):
        conf_va = np.maximum(P["knn_va"], 1 - P["knn_va"])
        for seed in range(3):
            rep = np.random.default_rng(seed).integers(0, 60, 60)
            res = calibrate([C[("knn_va", i)]["curves"] for i in rep], conf_va[rep], taus=[0.005, 0.01, 0.02],
                            impacts=("comp",))
            for name, r in res.items():
                for s in splits:
                    r[f"share_{s}"] = share(P[f"knn_{s}"], r["theta"])
            out[f"knn_bs{seed}"] = res
            for name in res:
                r = res[name]
                if name.startswith("budget"):
                    print(f"  knn_bs{seed} {name:22s} impact_cal {r['impact_cal'] * 100:7.3f}%  share cal {r['share_cal'] * 100:5.1f}%  "
                          + "  ".join(f"{s} {r[f'share_{s}'] * 100:5.1f}%" for s in splits), flush=True)
    tr_keys = [("knn_oof_tr", i) for i in range(200)]
    if all(k in C for k in tr_keys):
        conf_tr = np.maximum(P["knn_oof_tr"], 1 - P["knn_oof_tr"])
        conf_va = np.maximum(P["knn_va"], 1 - P["knn_va"])
        for seed in range(3):
            pick = np.sort(np.random.default_rng(seed).choice(200, 120, replace=False))
            curves = [C[("knn_va", i)]["curves"] for i in range(60)] + [C[tr_keys[i]]["curves"] for i in pick]
            conf = np.concatenate([conf_va, conf_tr[pick]])
            res = calibrate(curves, conf, taus=[0.005, 0.01, 0.02], impacts=("comp",))
            for name, r in res.items():
                for s in splits:
                    r[f"share_{s}"] = share(P[f"knn_{s}"], r["theta"])
            out[f"knn_mix{seed}"] = res
            for name in res:
                r = res[name]
                print(f"  knn_mix{seed} {name:22s} impact_cal {r['impact_cal'] * 100:7.3f}%  share cal {r['share_cal'] * 100:5.1f}%  "
                      + "  ".join(f"{s} {r[f'share_{s}'] * 100:5.1f}%" for s in splits), flush=True)
    with open(os.path.join(OUT, "ltf_thresholds.json"), "w") as f:
        json.dump(out, f)
    print("saved", flush=True)
