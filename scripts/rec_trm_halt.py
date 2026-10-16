"""Post-hoc halting head on LP-priced targets (PLAN.md, risk 2): "the schedule of step k is within 0.5 % of the best
known cost". Fitted on training instances, applied on validation.

    python3 scripts/rec_trm_halt.py --tag main --n_fit 400

* fit: the first --n_fit training instances of the polished part (never used for selection); for every step
  k = 1..K the schedule decoded with the model's validation-selected repaired pipeline (decoder, threshold, LP repair)
  is priced; best known cost = min(label cost, every step's cost); target good_k = cost_k <= 1.005 x best; a logistic
  regression on the mean-pooled answer embedding y after step k (the input of the trained halting head).
* validation: stop at the first step with a positive logit (cap K); mean gap and mean steps against fixed step counts.
Outputs: results/recursive/trm_halt_<tag>.json (weights, fit and validation metrics).
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
from otsl import trm  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from rec_trm_val import Pricer  # noqa: E402

OUT = os.path.join("data", "generated", "recursive")
RES = os.path.join("results", "recursive")


def priced_steps(model, feats, cfg, sel, d, sysm, K, price):
    xt, xb = feats(d)
    H = trm.hidden_at_steps(model, xt, xb, K, bs=40)
    P, _ = trm.run_steps(model, xt, xb, K, bs=40)
    sgs = trm.subgroups(sysm, d["u0"]) if cfg.sym == "count" else [None] * len(P)
    C = np.zeros((len(P), K, 3))
    for i in range(len(P)):
        for k in range(K):
            u = trm.decode(P[i, k], d, i, sysm, sel["th"], sel["decoder"], sgs[i])
            C[i, k] = price.repaired(i, u)[:3]
    return H, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_fit", type=int, default=400)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    model, feats, cfg, _ = trm.load_model(os.path.join(OUT, "models", f"{a.tag}.pt"), sysm)
    sel = json.load(open(os.path.join(RES, f"trm_val_{a.tag}.json")))["selection"]["rep"]
    t0 = time.time()
    tr = trm.train_data(2000)
    fit = trm.subset(tr, np.arange(500, 500 + a.n_fit))          # polished instances
    Hf, Cf = priced_steps(model, feats, cfg, sel, fit, sysm, a.K, Pricer(sysm, fit))
    best = np.minimum(fit["c_lab"], Cf[:, :, 0].min(1))
    good = Cf[:, :, 0] <= (1 + a.tol) * best[:, None]
    w, b = trm.fit_halting(Hf.reshape(-1, Hf.shape[-1]), good.reshape(-1).astype(float))
    t_fit = time.time() - t0
    va = trm.val_data()
    Hv, Cv = priced_steps(model, feats, cfg, sel, va, sysm, a.K, Pricer(sysm, va))
    logit = Hv @ w + b                                            # [N, K]
    kh = np.array([int(np.argmax(l > 0)) + 1 if (l > 0).any() else a.K for l in logit])
    pick = Cv[np.arange(len(Cv)), kh - 1]
    gap = (pick[:, 0] - va["obj"]) / va["obj"] * 100
    served = (pick[:, 1] < 1e-6) & (pick[:, 2] < 1e-6)
    fixed = {k: dict(gap_mean=float(((Cv[:, k - 1, 0] - va["obj"]) / va["obj"] * 100).mean()),
                     served=float(((Cv[:, k - 1, 1] < 1e-6) & (Cv[:, k - 1, 2] < 1e-6)).mean() * 100))
             for k in range(1, a.K + 1)}
    oracle = (Cv[:, :, 0].min(1) - va["obj"]) / va["obj"] * 100
    out = dict(tag=a.tag, n_fit=a.n_fit, K=a.K, tol=a.tol, w=w.tolist(), b=b, good_rate_fit=float(good.mean()),
               good_by_step_fit=good.mean(0).tolist(), fit_s=t_fit,
               val=dict(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), served=float(served.mean() * 100),
                        steps_mean=float(kh.mean()), steps_median=float(np.median(kh)),
                        steps_hist=np.bincount(kh, minlength=a.K + 1)[1:].tolist()),
               val_fixed_steps=fixed, val_oracle_best_step_gap=float(oracle.mean()))
    with open(os.path.join(RES, f"trm_halt_{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("w", "val_fixed_steps")}, default=float))
    print("fixed steps:", {k: round(v["gap_mean"], 3) for k, v in fixed.items()})


if __name__ == "__main__":
    main()
