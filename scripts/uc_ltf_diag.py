"""Learning to Fix reconstruction: where do the fixing rules fail? Wrong fixes (against the dataset MILP solution,
aligned inside identical-unit groups) by direction, on instances with and without a large gap.

    python scripts/uc_ltf_diag.py --tag fresh --split test_fresh

Output: results/uc12/ltf_diag_<tag>.json and a table on stdout.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otsl.fixpolicy import align_to_prediction  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_ltf_eval import RuleMaker  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fresh")
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--rules", default="ltf:knn:budget:comp:0.01,ltf:rl:budget:comp:0.01,rac@0.95,harm@0.95,rl@0.95")
    a = ap.parse_args()
    sysm = load_rts_gmlc()
    d = load(os.path.join("data/generated/uc12", f"{a.split}.npz"))
    recs = [json.loads(line) for line in open(os.path.join("results/uc12", f"ltf_eval_{a.tag}.jsonl"))]
    gap = {(r["rule"], r["i"]): (r["obj"] - d["obj"][r["i"]]) / d["obj"][r["i"]] * 100 for r in recs}
    mk = RuleMaker(a.split, d, sysm)
    groups = sysm.identical_groups()
    out = {}
    print("| rule | instances | gap > 10 %: n, wrong OFF / ON fixes per instance | gap <= 1 %: n, wrong OFF / ON |")
    print("|---|---|---|---|")
    for rule in a.rules.split(","):
        rows = []
        for i in sorted({i for (r, i) in gap if r == rule}):
            sc = scenario_from(d, i)
            fix, _ = mk(rule, i, sc)
            yh = np.zeros((12, sysm.G), np.int8)
            for t, g, v in fix:
                yh[t, g] = v
            ua = align_to_prediction(sysm, d["u"][i], yh, sc.u0, groups)
            off = sum(1 for t, g, v in fix if v == 0 and ua[t, g] == 1)
            on = sum(1 for t, g, v in fix if v == 1 and ua[t, g] == 0)
            rows.append(dict(i=i, gap=gap[(rule, i)], wrong_off=off, wrong_on=on, n_fixed=len(fix)))
        bad = [r for r in rows if r["gap"] > 10]
        good = [r for r in rows if r["gap"] <= 1]
        f = lambda rs, k: np.mean([r[k] for r in rs]) if rs else float("nan")
        print(f"| {rule} | {len(rows)} | {len(bad)}: {f(bad, 'wrong_off'):.1f} / {f(bad, 'wrong_on'):.1f} | "
              f"{len(good)}: {f(good, 'wrong_off'):.1f} / {f(good, 'wrong_on'):.1f} |")
        out[rule] = rows
    json.dump(out, open(os.path.join("results/uc12", f"ltf_diag_{a.tag}.json"), "w"), indent=1, default=float)
