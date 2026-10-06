"""m1x study, downstream check on validation: a guarded fixing rule with reduced MILPs for each probability source.

    python3 scripts/uc_m1x_valfix.py --tags ref_bce_s0,pol_n1000_gnn_s0 [--n 60] [--ratio 0.95]

Rule (fixed in advance, the "error-cost + both guards" rule of X2): learned error-cost ranking (harm ensemble
combo_harm_s0.pt on features of the source's probabilities), fix the `ratio` share to round(p), adequacy guard,
min up/down row release, LP-relaxation guard, reduced MILP (60 s, 0.1 %). Instances: the first n of val_extra2
(validation days, never used for early stopping). Gap to the stored validation MILP objective (60 s, 0.1 %).
Output: results/uc12/m1x_valfix.jsonl (one record per source and instance; resume skips finished pairs).
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
from otsl.combo import harm_from_file, harm_scores, rule_fixings, solver_guards  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.m1x import scenario  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402

ROOT, DATA, RES = "data/generated/uc12", "data/generated/uc12_m1x", "results/uc12"
OFFSET = 180                     # val_extra2 starts at index 180 of the 360 validation instances

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ratio", type=float, default=0.95)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    d = load(os.path.join(ROOT, "val_extra2.npz"))
    ff = FixFeaturizer(sysm, 12)
    harm = harm_from_file(os.path.join(RES, "combo_harm_s0.pt"))
    path = os.path.join(RES, "m1x_valfix.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["tag"], r["i"], r["ratio"]) for r in map(json.loads, open(path))}
    for tag in a.tags.split(","):
        P = np.load(os.path.join(DATA, "probs", f"{tag}_probs.npz"))["val"][OFFSET:OFFSET + a.n].astype(np.float64)
        H = harm_scores(harm, ff, P, d, list(range(a.n)))
        t0 = time.time()
        for i in range(a.n):
            if (tag, i, a.ratio) in done:
                continue
            sc = scenario(d, i)
            fix = rule_fixings("harm", a.ratio, P[i], sc, sysm, H[i])
            fix, info, lp_s = solver_guards(m, sysm, sc, fix)
            sol = m.solve_uc(sc, time_limit=60.0, mip_gap=1e-3, z_fix=fix or None)
            rec = dict(tag=tag, i=i, ratio=a.ratio, fixed=len(fix) / P[i].size, feasible=sol.u is not None,
                       gap=float((sol.obj - d["obj"][i]) / d["obj"][i] * 100) if sol.u is not None else None,
                       milp_s=sol.time, guard_s=lp_s, shed=sol.shed, short=sol.short, status=sol.status, **info)
            with open(path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        recs = [r for r in map(json.loads, open(path)) if r["tag"] == tag and r["ratio"] == a.ratio]
        g = np.array([r["gap"] if r["feasible"] else np.nan for r in recs])
        print(f"{tag}: n {len(recs)} mean gap {np.nanmean(g):.3f}% median {np.nanmedian(g):.3f}% >1% {int(np.sum(g > 1))} "
              f"fixed {np.mean([r['fixed'] for r in recs]) * 100:.1f}% time {np.mean([r['milp_s'] + r['guard_s'] for r in recs]):.2f}s "
              f"({time.time() - t0:.0f}s)", flush=True)
