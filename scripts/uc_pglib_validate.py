"""PGLib-UC: validate otsl/pglib.py against the reference Pyomo formulation (uc_model.py of pglib-uc).

    PYTHONPATH=<dir with pyomo> python3 scripts/uc_pglib_validate.py --out results/pglib/pglib_validate.json

Checks, per instance (optionally truncated to T periods):
  1. LP relaxation value: reference model (hard balance, binaries relaxed) vs ours (soft balance) - must agree;
  2. a given commitment priced by both: the reference model with u fixed (all other binaries free) vs our dispatch LP;
  3. MILP optimum on small instances: reference model solved by HiGHS vs ours (delta continuous and binary).
The reference model is built by executing uc_model.py's model-construction code unchanged.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.pglib import PGLIB, PGModel, load_case  # noqa: E402
from otsl.uc import repair_min_updown  # noqa: E402

REF = os.path.join(PGLIB, "uc_model.py")


def truncated(path, T, tmpdir):
    d = json.load(open(path))
    if T is None or T >= d["time_periods"]:
        return path
    d["time_periods"] = T
    d["demand"], d["reserves"] = d["demand"][:T], d["reserves"][:T]
    for g in d["renewable_generators"].values():
        g["power_output_minimum"] = g["power_output_minimum"][:T]
        g["power_output_maximum"] = g["power_output_maximum"][:T]
    out = os.path.join(tmpdir, f"pglib_{T}_" + os.path.basename(path))
    json.dump(d, open(out, "w"))
    return out


def build_reference(path):
    """execute uc_model.py up to (not including) its solve; returns the Pyomo model and its data"""
    src = open(REF).read()
    src = src.split("from pyomo.opt import SolverFactory")[0]
    g = {"__name__": "pglib_ref"}
    argv = sys.argv
    sys.argv = ["uc_model.py", path]
    try:
        exec(compile(src, REF, "exec"), g)
    finally:
        sys.argv = argv
    return g["m"], g["data"]


def solve_ref(m, gap=1e-6, tl=600.0, relax=False, fix_u=None, names=None):
    import pyomo.environ as pe
    from pyomo.contrib.appsi.solvers import Highs
    if relax:
        pe.TransformationFactory("core.relax_integer_vars").apply_to(m)
    if fix_u is not None:
        for (g, t), v in fix_u.items():
            m.ug[g, t].fix(v)
    opt = Highs()
    opt.config.time_limit = tl
    opt.config.mip_gap = gap
    opt.config.load_solution = True
    opt.highs_options = {"threads": 1}
    t0 = time.time()
    res = opt.solve(m)
    return float(pe.value(m.obj)), time.time() - t0, str(res.termination_condition), \
        (float(res.best_objective_bound) if res.best_objective_bound is not None else np.nan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pglib/pglib_validate.json")
    ap.add_argument("--lp", nargs="+", default=["ca/2014-09-01_reserves_3.json", "ca/Scenario400_reserves_5.json",
                                                 "rts_gmlc/2020-01-27.json"])
    ap.add_argument("--milp", nargs="+", default=["rts_gmlc/2020-01-27.json:24", "ca/2015-03-01_reserves_1.json:12"])
    ap.add_argument("--tl", type=float, default=600.0)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="pglib_val_", dir=os.environ.get("PGLIB_TMP", None))
    out = {"lp": [], "fixed": [], "milp": []}
    for f in a.lp:
        path = os.path.join(PGLIB, f)
        s, sc, d = load_case(path)
        T = d["time_periods"]
        m = PGModel(s, T=T)
        rel = m.solve_dispatch(sc, None, relax=True)
        ref, _ = build_reference(path)
        v, dt, st, _ = solve_ref(ref, relax=True)
        row = dict(file=f, T=T, ours=rel.obj, ours_s=rel.time, ref=v, ref_s=dt, ref_status=st,
                   rel_diff=(rel.obj - v) / abs(v), ours_slack=rel.shed + rel.short)
        # 2. a heuristic commitment (rounded relaxation, min up/down repaired) priced by both models
        u = repair_min_updown((rel.u > 0.5).astype(np.int8), sc.u0, s.min_up, s.min_dn)
        lp = m.solve_dispatch(sc, u)
        ref2, _ = build_reference(path)
        fix = {(s.names_all[g], t + 1): int(u[t, k]) for k, g in enumerate(s.free) for t in range(T)}
        v2, dt2, st2, _ = solve_ref(ref2, gap=1e-9, tl=a.tl, fix_u=fix)
        row2 = dict(file=f, T=T, ours_dispatch=lp.obj, ours_slack=lp.shed + lp.short, ref_fixed_u=v2, ref_status=st2,
                    rel_diff=(lp.obj - v2) / abs(v2) if np.isfinite(v2) else None)
        print("LP", row, "\nFIXED", row2, flush=True)
        out["lp"].append(row)
        out["fixed"].append(row2)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
    for spec in a.milp:
        f, T = spec.split(":")
        T = int(T)
        path = truncated(os.path.join(PGLIB, f), T, tmp)
        s, sc, d = load_case(path)
        row = dict(file=f, T=T)
        for db in (False, True):
            m = PGModel(s, T=T, delta_binary=db)
            sol = m.solve_milp(sc, time_limit=a.tl, mip_gap=1e-6, incumbents=False)
            row[f"ours_delta_{'bin' if db else 'cont'}"] = dict(obj=sol["obj"], bound=sol["bound"], time=sol["time"],
                                                               status=sol["status"], slack=sol["shed"] + sol["short"])
        ref, _ = build_reference(path)
        v, dt, st, bd = solve_ref(ref, gap=1e-6, tl=a.tl)
        row["ref"] = dict(obj=v, bound=bd, time=dt, status=st)
        row["rel_diff"] = (row["ours_delta_cont"]["obj"] - v) / abs(v)
        print("MILP", row, flush=True)
        out["milp"].append(row)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
