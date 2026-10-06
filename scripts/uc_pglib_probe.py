"""PGLib-UC: probe full-MILP solve times with HiGHS (single thread) on base California instances.

    python3 scripts/uc_pglib_probe.py --files 2014-09-01_reserves_3 Scenario400_reserves_3 --T 48 --gap 0.001 \
        --tl 900 --workers 2 --out results/pglib/pglib_probe.jsonl

Per instance: model size, LP relaxation (value, time), full MILP (status, objective, dual bound, gap, time,
incumbent log), the dispatch LP of the MILP schedule (must reproduce the objective) and the time at which the
incumbent first came within 0.1 / 0.25 / 0.5 / 1 % of the final objective.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(job):
    f, T, gap, tl, delta_binary, scale = job
    from otsl.pglib import PGLIB, PGModel, load_case
    path = f if f.endswith(".json") else os.path.join(PGLIB, "ca", f + ".json")
    s, sc, _ = load_case(path, T=T)
    if scale != 1.0:
        sc.load = sc.load * scale
        sc.sr = sc.sr * scale
    t0 = time.time()
    m = PGModel(s, T=T, delta_binary=delta_binary)
    t_build = time.time() - t0
    rel = m.solve_dispatch(sc, None, relax=True)
    sol = m.solve_milp(sc, time_limit=tl, mip_gap=gap, incumbents=True)
    out = dict(file=os.path.basename(path), T=T, gap_target=gap, tl=tl, delta_binary=delta_binary, scale=scale,
               nv=m.nv, nrow=m.A.shape[0], nnz=m.A.nnz, n_int=int(m.integ.sum()), t_build=t_build,
               lp_rel=rel.obj, t_lp_rel=rel.time, status=sol["status"], obj=sol["obj"], bound=sol["bound"],
               gap=sol["gap"], time=sol["time"], nodes=sol["nodes"], shed=sol["shed"], short=sol["short"],
               inc=sol["inc"], loadavg=os.getloadavg()[0])
    if sol["u"] is not None:
        lp = m.solve_dispatch(sc, sol["u"])
        out.update(obj_lp=lp.obj, t_dispatch=lp.time, units_on=float(sol["u"].sum() / T),
                   rel_gap_to_lp_relax=(sol["obj"] - rel.obj) / sol["obj"])
        for q in (0.001, 0.0025, 0.005, 0.01):
            tt = [t for t, o, _ in sol["inc"] if o <= sol["obj"] * (1 + q) + 1e-9]
            out[f"t_within_{q * 100:g}%"] = tt[0] if tt else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--gap", type=float, default=1e-3)
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delta_binary", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="results/pglib/pglib_probe.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = [(f, a.T, a.gap, a.tl, bool(a.delta_binary), a.scale) for f in a.files]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run, jobs):
            print(f"{r['file']} T={r['T']} {r['status']} obj {r['obj']:.2f} bound {r['bound']:.2f} gap {r['gap'] * 100:.3f}% "
                  f"time {r['time']:.1f}s nodes {r['nodes']} incumbents {len(r['inc'])} lp-relax {r['lp_rel']:.2f} "
                  f"({r['t_lp_rel']:.1f}s) obj_lp {r.get('obj_lp')} t0.5% {r.get('t_within_0.5%')}", flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
