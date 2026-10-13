"""Cost-aware check study: what goes wrong on the pipeline's worst test instances? (diagnostic, test data read only)

    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_deep.py

For the instances where the pipeline is more than 1 pp worse than the full MILP at 0.5 % (from the m1x records): the
pipeline's reduced MILP and the full MILP at 0.5 % are re-solved to obtain both schedules; the fixings that contradict
the full MILP's schedule (aligned inside groups of identical units) are listed with their unit, direction, LP-relaxation
value, first-order saving in the relaxed reduced LP and in the dispatch LP at the pipeline's solution (and the rank of
each among all fixings). Nothing here feeds a decision. Output: results/uc12/cguard_deep.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uc_cguard_common import BHG, RES, ROOT, data_instance, load, shifted_instance  # noqa: E402
from otsl.cguard import relaxed_reduced_lp, solve_with_start  # noqa: E402
from otsl.fixpolicy import align_to_prediction  # noqa: E402
from otsl.ood import OODModel  # noqa: E402
from otsl.solver import flip_savings  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402

if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    D = [json.loads(l) for l in open(os.path.join(RES, "cguard_diag_test.jsonl"))]
    bad = sorted([d for d in D if d["gap_pipe"] - d["gap_full5"] > 1.0], key=lambda d: -(d["gap_pipe"] - d["gap_full5"]))
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    bhg = BHG(sysm)
    tf = load(os.path.join(ROOT, "test_fresh.npz"))
    out = []
    for d in bad:
        s, k = d["set"], d["k"]
        sc = data_instance(tf, k) if s == "tf" else shifted_instance(s, k)
        rel = m.solve_dispatch(sc, None, relax=True)
        _, fix, _ = bhg.run(m, sc, rel)
        red = solve_with_start(m, sc, fix, 60.0, 5e-3)
        full = solve_with_start(m, sc, {}, 60.0, 5e-3)
        ua = align_to_prediction(sysm, full["u"], red["u"], sc.u0)
        rr = relaxed_reduced_lp(m, sc, fix)
        sav_d, _ = flip_savings(m, sc, red["u"])
        keys = list(fix)
        rank_rr = {tg: i for i, tg in enumerate(sorted(keys, key=lambda tg: -rr["saving"][tg]))}
        rank_d = {tg: i for i, tg in enumerate(sorted(keys, key=lambda tg: -sav_d[tg]))}
        wrong = [(t, g, v) for (t, g), v in fix.items() if ua[t, g] != v]
        free_diff = int(((red["u"] != ua) & ~np.isin(np.arange(12 * sysm.G).reshape(12, sysm.G),
                                                       [t * sysm.G + g for (t, g) in fix])).sum())
        units = sorted(set(g for _, g, _ in wrong))
        rec = dict(set=s, k=k, gap_pipe=d["gap_pipe"], gap_full5=d["gap_full5"], n_fix=len(fix), n_wrong=len(wrong),
                   n_free_differ=free_diff, cost_red=red["obj"], cost_full5=full["obj"], z_lp=rel.obj,
                   wrong_units=[dict(g=int(g), type=str(sysm.utype[g]), pmax_MW=float(sysm.pmax[g] * 100),
                                     hours=[int(t) for t, gg, _ in wrong if gg == g],
                                     fixed_to=int([v for _, gg, v in wrong if gg == g][0]),
                                     u_rel=[round(float(rel.u[t, g]), 3) for t, gg, _ in wrong if gg == g],
                                     rank_rr=min(rank_rr[(t, g)] for t, gg, _ in wrong if gg == g),
                                     rank_dispatch=min(rank_d[(t, g)] for t, gg, _ in wrong if gg == g)) for g in units])
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(out, open(os.path.join(RES, "cguard_deep.json"), "w"), indent=1, default=float)
