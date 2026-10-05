"""Diagnostics of fine-tuned commitment models on the validation set (no LP solves):
how far each model moved from its initial (imitation) policy, which decisions it changed, and how the
changes relate to the MILP schedule.

    python scripts/uc_constrained_diag.py --pairs lag_B:lf_bce,rl_lf:lf_bce,milp_rl:milp_bce
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_constrained import OUT, model_probs, setup, strip  # noqa: E402


def kl(p, q):
    p, q = np.clip(p, 1e-6, 1 - 1e-6), np.clip(q, 1e-6, 1 - 1e-6)
    return (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))).sum((1, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    cfg, tr_full, va, te, sysm, rep = setup()
    tr = strip(tr_full)
    cache, rows = {}, []
    P = lambda n: cache.setdefault(n, model_probs(n, va, sysm, tr, tr_full))
    milp = va["u"]
    for pr in a.pairs.split(","):
        m, init = pr.split(":")
        p, p0 = P(m), P(init)
        on, on0 = p > 0.5, p0 > 0.5
        add, rem = on & ~on0, ~on & on0
        r = {"model": m, "init": init, "KL_to_init_per_inst": float(kl(p, p0).mean()),
             "added_per_inst": float(add.sum((1, 2)).mean()), "removed_per_inst": float(rem.sum((1, 2)).mean()),
             "added_and_milp_on": float((add & (milp == 1)).sum((1, 2)).mean()),
             "added_and_milp_off": float((add & (milp == 0)).sum((1, 2)).mean()),
             "removed_and_milp_off": float((rem & (milp == 0)).sum((1, 2)).mean()),
             "uncertain_decisions_%": float(((p > 0.05) & (p < 0.95)).mean() * 100),
             "units_on": float(on.sum((1, 2)).mean() / 12), "units_on_init": float(on0.sum((1, 2)).mean() / 12),
             "units_on_milp": float(milp.sum((1, 2)).mean() / 12),
             "missed_milp_on_per_inst": float((~on & (milp == 1)).sum((1, 2)).mean()),
             "extra_vs_milp_per_inst": float((on & (milp == 0)).sum((1, 2)).mean())}
        for lo, hi in [(0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5)]:
            r[f"added_with_p0_in_[{lo},{hi})"] = float((add & (p0 >= lo) & (p0 < hi)).sum((1, 2)).mean())
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(OUT, "constrained_diag.json"), "w") as f:
        json.dump(rows, f, indent=1)
