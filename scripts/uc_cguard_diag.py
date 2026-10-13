"""Cost-aware check study, step 1: which signal separates the pipeline's expensive failures? (LPs only, no MILP)

    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_diag.py --sets test        # test_fresh 0-119 + 6 shifted sets
    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_diag.py --sets val         # val + val_extra + val_extra2 (360)

Per instance: LP relaxation, the pipeline's fixings (rebuilt; the fixed share is checked against the pipeline record on
test), and signals computed from them:
  delta      (z_RR - z_LP) / z_LP: the relaxed reduced LP (fixings applied, the rest relaxed) against the full relaxation
  sav_max    largest first-order saving of releasing one fixing (bound marginal in the relaxed reduced LP), / z_LP
  sav_sum    sum of the positive savings, / z_LP
  n_on_vs_lp ON fixings where the LP relaxation is exactly 0; n_off_vs_lp OFF fixings where it is exactly 1
  sav_on / sav_off   sum of positive savings over ON / OFF fixings, / z_LP
On test the pipeline's outcome (its gap and the full MILP at 0.5 %, from results/uc12/m1x_pipeline_eval_s*.jsonl) is
attached for the analysis; on validation the dataset MILP objective. Output: results/uc12/cguard_diag_<sets>.jsonl.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uc_cguard_common import (BHG, R_FULL, R_FULL5, R_PIPE, RES, ROOT, data_instance, fixed_share, load,  # noqa: E402
                              pipeline_records, shifted_instance, slack, test_jobs, timed)
from otsl.cguard import lp_increase, relaxed_reduced_lp  # noqa: E402
from otsl.ood import OODModel  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402


def signals(m, sc, rel, fix):
    rr = relaxed_reduced_lp(m, sc, fix)
    z = rel.obj
    sav = rr["saving"]
    u = rel.u
    pos = {tg: s for tg, s in sav.items() if s > 0}
    return dict(z_lp=float(z), z_rr=rr["cost"], delta=float(lp_increase(rr["cost"], z)), rr_s=rr["seconds"],
                rr_slack=float(np.sum(rr["slack"])),
                sav_max=float(max(pos.values()) / z) if pos else 0.0, sav_sum=float(sum(pos.values()) / z),
                sav_on=float(sum(s for (t, g), s in pos.items() if fix[(t, g)] == 1) / z),
                sav_off=float(sum(s for (t, g), s in pos.items() if fix[(t, g)] == 0) / z),
                n_on_vs_lp=int(sum(1 for (t, g), v in fix.items() if v == 1 and u[t, g] <= 1e-6)),
                n_off_vs_lp=int(sum(1 for (t, g), v in fix.items() if v == 0 and u[t, g] >= 1 - 1e-6)),
                n_frac_fixed=int(sum(1 for (t, g) in fix if 1e-6 < u[t, g] < 1 - 1e-6)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", choices=["test", "val"], required=True)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    bhg = BHG(sysm)
    out = os.path.join(RES, f"cguard_diag_{a.sets}.jsonl")
    done = {(r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    if a.sets == "test":
        tf = load(os.path.join(ROOT, "test_fresh.npz"))
        R = pipeline_records()
        jobs = test_jobs()
        get = lambda s, k: data_instance(tf, k) if s == "tf" else shifted_instance(s, k)
    else:
        parts = [(p, load(os.path.join(ROOT, f"{p}.npz"))) for p in ("val", "val_extra", "val_extra2")]
        V = {"val": parts[0][1], "val_extra": parts[1][1], "val_extra2": parts[2][1]}
        jobs = [(p, k) for p, d in parts for k in range(len(d["load"]))]
        get = lambda s, k: data_instance(V[s], k)
    jobs = [j for j in jobs if j not in done]
    bhg.warm(m, get(*jobs[0])) if jobs else None
    print(f"{len(jobs)} instances", flush=True)
    for n, (s, k) in enumerate(jobs):
        sc = get(s, k)
        rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
        fix0, fix, info = bhg.run(m, sc, rel)
        rec = dict(set=s, k=k, relax_s=relax_s, pre_s=info["pre_s"], guard_s=info["guard_s"],
                   released_adeq=info.get("released_adeq"), released_rows=info.get("released_rows"),
                   released_lp=info.get("released_lp"), fixed=fixed_share(fix, sc.g_out, 12, sysm.G),
                   n_fix=len(fix), **signals(m, sc, rel, fix))
        if a.sets == "test":
            rr = R[(s, k)]
            F = rr[R_FULL]
            DB = F["bound"] if np.isfinite(F["bound"]) else F["obj"] * (1 - (F["mip_gap"] or 0))
            g = lambda r: (r["obj"] - DB) / r["obj"] * 100
            rec.update(fixed_match=abs(rec["fixed"] - rr[R_PIPE]["fixed_share"]) < 1e-9, gap_pipe=g(rr[R_PIPE]),
                       gap_full5=g(rr[R_FULL5]), gap_full=g(F), pipe_slack=slack(rr[R_PIPE]),
                       t_full=F["time"], t_pipe=rr[R_PIPE]["time"])
        else:
            d = V[s]
            rec.update(obj=float(d["obj"][k]), db=float(d["obj"][k] * (1 - d["gap"][k])))
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if n % 20 == 0:
            print(f"[{n + 1}/{len(jobs)}] {s} {k} delta {rec['delta'] * 100:.3f}% "
                  + (f"pipe {rec['gap_pipe']:.2f}% full5 {rec['gap_full5']:.2f}% match {rec['fixed_match']}" if a.sets == "test" else ""),
                  flush=True)
