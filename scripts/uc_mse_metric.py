"""Is the framework's test metric (MSE of PG / VA against the MILP solution) aligned with cost?

    python scripts/uc_mse_metric.py --subset tied   # hours with an exactly tied alternative optimum
    python scripts/uc_mse_metric.py --subset all    # the first 300 test hours
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otsl.uc import adequacy_repair, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from uc_gen import UC_CONFIGS  # noqa: E402
from uc_model2 import dispatch_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="tied", choices=["tied", "all"])
    a = ap.parse_args()
    cfg, s = UC_CONFIGS["uc1"], load_rts_gmlc()
    te = load("data/generated/uc1/test.npz")
    rel = (te["alt_c"][:, 0] - te["obj"]) / te["obj"]
    idx = np.where(rel <= 1e-6)[0][:300] if a.subset == "tied" else np.arange(300)
    adq = lambda U: np.array([adequacy_repair(U[k], te["load"][i], te["avail"][i], te["sr"][i], s) for k, i in enumerate(idx)])
    meths = {} if a.subset == "all" else {"alternative optimal commitment (gap 0)": te["alt_u"][idx, 0]}
    meths.update({"rounded LP relaxation + adequacy repair": adq((te["u_rel"][idx] > 0.5).astype(np.int8)),
                  "persistence + adequacy repair": adq(te["u0"][idx][:, None]),
                  "merit-order priority list": adq(np.zeros_like(te["u"][idx])),
                  "persistence (last hour's units)": te["u0"][idx][:, None]})
    out = {}
    for name, U in meths.items():
        obj, p, va, r, sh, so = dispatch_all(cfg, te, idx, U, 2)
        gap = (obj - te["obj"][idx]) / te["obj"][idx] * 100
        mse = ((p - te["p"][idx]) ** 2).mean((1, 2))
        ok = (sh < 1e-6) & (so < 1e-6)
        rho = np.corrcoef(np.argsort(np.argsort(mse)), np.argsort(np.argsort(gap)))[0, 1]
        out[name] = dict(pg_mse=float(mse.mean()), va_mse=float(((va - te["va"][idx]) ** 2).mean()), gap_median=float(np.median(gap)),
                         gap_mean=float(gap.mean()), no_shed=float(ok.mean() * 100), spearman_mse_gap=float(rho))
        print(f"{name:45s} PG MSE {mse.mean():.4f}  VA MSE {out[name]['va_mse']:.5f}  median gap {np.median(gap):.3f}%  mean gap {gap.mean():.1f}%  no-shed {ok.mean()*100:.1f}%  rho {rho:.2f}", flush=True)
    json.dump(out, open(f"results/uc1/mse_vs_gap_{a.subset}.json", "w"), indent=1)
