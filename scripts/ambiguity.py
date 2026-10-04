"""Label ambiguity of DC-OTS: how often is the MILP's switching decision non-unique?

    python scripts/ambiguity.py case30 case30_raw case118 case118_raw

Uses the alternative optima enumerated with no-good cuts at data-generation time (--n_alt).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import load  # noqa: E402

if __name__ == "__main__":
    out = {}
    for cfg in sys.argv[1:]:
        path = os.path.join("data", "generated", cfg, "test.npz")
        if not os.path.exists(path):
            continue
        d = load(path)
        sc = float(d.get("switch_cost", 0.0))
        n_open = (1 - d["z"]).sum(1)
        cs = d["c_ots"] + sc * n_open
        rel = (d["alt_c"] - cs[:, None]) / cs[:, None]          # 2nd/3rd best topology vs best
        ben = (d["c0"] - cs) / d["c0"]
        r = {"n": int(len(cs)), "switch_cost": sc,
             "mean_benefit_%": float(ben.mean() * 100),
             "frac_zero_benefit(<0.01%)": float((ben < 1e-4).mean()),
             "mean_lines_opened": float(n_open.mean()),
             "frac_opening_full_budget": float((n_open == n_open.max()).mean())}
        for tol in [1e-6, 1e-5, 1e-4]:
            r[f"frac_alt_topology_within_{tol:g}_rel"] = float((rel <= tol).any(1).mean())
        # among scenarios with a real benefit, is the 2nd-best topology essentially as good?
        mb = ben > 1e-3
        r["frac_alt_within_1%_of_benefit|benefit>0.1%"] = float(
            ((d["alt_c"][mb, 0] - cs[mb]) <= 0.01 * (d["c0"][mb] - cs[mb])).mean()) if mb.any() else None
        # how different are the alternative optima? (Hamming distance of 2nd-best to best)
        r["mean_hamming_best_vs_2nd"] = float((d["alt_z"][:, 0] != d["z"]).sum(1).mean())
        out[cfg] = r
        print(cfg, json.dumps(r, indent=1))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "ambiguity.json"), "w") as f:
        json.dump(out, f, indent=1)
