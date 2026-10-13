"""Screening of candidate benchmarks with a weaker LP relaxation (see otsl/hard.py).

One process per case (peak memory = this process's maximum resident set size). Per instance: LP relaxation, full
MILP (HiGHS, 1 thread, mip_rel_gap 0.1 %, bound trace every second), relaxation statistics against the MILP
schedule, served share, and for RTS-GMLC the line loading of the MILP schedule's dispatch LP. Records are appended to
results/hard/hard_screen.jsonl; instances already there are skipped (resume).

  python scripts/uc_hard_screen.py --case rts --line_scale 0.8 --n 4
  python scripts/uc_hard_screen.py --case rts --reserve 0.06 --n 3
  python scripts/uc_hard_screen.py --case rts --T 48 --n 2
  python scripts/uc_hard_screen.py --case ferc --files 2015-01-01_hw 2015-07-01_hw --T 24
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from otsl.hard import make_scenario_h, pg_milp_trace, relax_stats, split_days, trace_gap_times  # noqa: E402

OUT = "results/hard/hard_screen.jsonl"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                keys.add((r["case"], r["idx"]))
            except Exception:
                pass
    return keys


def append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def common(rec, rel_obj, sol, rs):
    obj, db = sol["obj"], sol["bound"]
    rec.update(rs)
    rec.update(milp_time=sol["time"], milp_cpu=sol.get("cpu", np.nan), milp_status=sol["status"], obj=obj, bound=db,
               mip_gap=sol["gap"], nodes=sol["nodes"], n_inc=len(sol["inc"]),
               lp_gap_obj=float((obj - rel_obj) / obj * 100) if np.isfinite(obj) else None,
               lp_gap_db=float((db - rel_obj) / db * 100))
    rec.update(trace_gap_times(sol.get("trace", [])))
    rec["first_inc"] = sol["inc"][0][0] if sol["inc"] else None
    rec["first_inc_gap"] = float((sol["inc"][0][1] - db) / sol["inc"][0][1] * 100) if sol["inc"] else None
    return rec


def run_rts(a):
    from otsl.b3 import make_jobs, solve_milp_hs
    from otsl.uc import UCModel, load_rts_gmlc
    s = load_rts_gmlc(line_scale=a.line_scale)
    s1 = load_rts_gmlc(line_scale=1.0)
    m = UCModel(s, T=a.T, network=True)
    case = a.label or f"rts{a.T}_ls{a.line_scale:g}_rf{a.reserve:g}"
    days = split_days(s, a.T)["train"]
    jobs = make_jobs(days, a.n, a.seed, T=min(a.T, 24), start=0 if a.T >= 24 else None)   # < 24 h: random start
    done = done_keys(a.out)
    for i, (day, st, seed) in enumerate(jobs):
        if (case, i) in done:
            continue
        t0 = time.time()
        sc = make_scenario_h(s, day, np.random.default_rng(seed), T=a.T, start=st, reserve_frac=a.reserve)
        rel = m.solve_dispatch(sc, None, relax=True)
        sol = solve_milp_hs(m, sc, time_limit=a.time_limit, mip_gap=a.gap, threads=1, trace_every=1.0)
        rec = dict(case=case, idx=i, bench="rts", T=a.T, line_scale=a.line_scale, reserve=a.reserve, day=int(day),
                   start=int(st), seed=int(seed), G=int(s.G), t_rel=rel.time, c_rel=rel.obj, rel_shed=rel.shed,
                   rel_short=rel.short, load_total=float(sc.load.sum()))
        if sol["u"] is None:
            rec.update(milp_status=sol["status"], milp_time=sol["time"])
            append(a.out, rec)
            continue
        rs = relax_stats(rel.u, sol["u"], s, sc.u0)
        common(rec, rel.obj, sol, rs)
        d = m.solve_dispatch(sc, sol["u"])
        f = np.abs(d.flow)                                         # [T, L]
        rec.update(shed=float(sol["shed"]), short=float(sol["short"]),
                   served=float(1 - sol["shed"] / sc.load.sum()), obj_lp=d.obj)
        rec["bind_share"] = float((f >= 0.999 * s.fmax[None, :]).mean())        # line-hours at this system's limit
        rec["n_lines_bind"] = int((f >= 0.999 * s.fmax[None, :]).any(0).sum())
        lo = f / s1.fmax[None, :]                                  # loading relative to the original rating
        for k in (1.0, 0.8, 0.7, 0.6):
            rec[f"above_{k:g}"] = float((lo >= 0.999 * k).mean())  # line-hours loaded >= k x original rating
        rec["max_loading_orig"] = float(lo.max())
        rec["wall"] = time.time() - t0
        rec["maxrss_mb"] = rss_mb()
        append(a.out, rec)
        print(f"[{case}] {i} day {day} rel {rel.time:.1f}s gap_obj {rec['lp_gap_obj']:.3f}% gap_db {rec['lp_gap_db']:.3f}% "
              f"int {rs['int_share'] * 100:.1f}% frac {rs['n_frac']} wrong {rs['n_wrong']} | MILP {sol['time']:.1f}s "
              f"{sol['status']} gap {sol['gap'] * 100:.3f}% t1% {rec['t_gap_1']} | served {rec['served'] * 100:.2f}% "
              f"short {rec['short']:.3f} bind {rec['bind_share'] * 100:.2f}% lines {rec['n_lines_bind']} | "
              f"rss {rec['maxrss_mb']:.0f} MB", flush=True)


def run_ferc(a):
    from otsl.pglib import PGLIB, PGModel, load_case
    for i, fn in enumerate(a.files):
        case = a.label or f"ferc{a.T}"
        if (case, fn) in done_keys(a.out):
            continue
        t0 = time.time()
        sysm, sc, _ = load_case(os.path.join(PGLIB, "ferc", fn + ".json"), T=a.T)
        m = PGModel(sysm, T=a.T)
        tb = time.time() - t0
        rel = m.solve_dispatch(sc, None, relax=True)
        print(f"[{case}] {fn} build {tb:.1f}s rel {rel.time:.1f}s obj {rel.obj:.1f} rss {rss_mb():.0f} MB", flush=True)
        sol = pg_milp_trace(m, sc, time_limit=a.time_limit, mip_gap=a.gap, trace_every=1.0)
        rec = dict(case=case, idx=fn, bench="ferc", T=a.T, G=int(sysm.G), G_all=int(sysm.G_all), t_build=tb,
                   t_rel=rel.time, c_rel=rel.obj, rel_shed=float(rel.shed), rel_short=float(rel.short),
                   load_total=float(sc.load.sum()), nv=int(m.nv), nrow=int(m.A.shape[0]))
        if sol["u"] is None:
            rec.update(milp_status=sol["status"], milp_time=sol["time"], maxrss_mb=rss_mb())
            append(a.out, rec)
            print(f"[{case}] {fn} no MILP solution ({sol['status']}, {sol['time']:.0f}s)", flush=True)
            continue
        rs = relax_stats(rel.u, sol["u"], sysm, sc.u0)
        common(rec, rel.obj, sol, rs)
        rec.update(shed=float(sol["shed"]), short=float(sol["short"]), served=float(1 - sol["shed"] / sc.load.sum()))
        rec["wall"] = time.time() - t0
        rec["maxrss_mb"] = rss_mb()
        append(a.out, rec)
        print(f"[{case}] {fn} gap_obj {rec['lp_gap_obj']:.3f}% gap_db {rec['lp_gap_db']:.3f}% int "
              f"{rs['int_share'] * 100:.2f}% frac {rs['n_frac']} wrong {rs['n_wrong']} | MILP {sol['time']:.1f}s "
              f"{sol['status']} gap {sol['gap'] * 100:.3f}% t1% {rec['t_gap_1']} t0.5% {rec['t_gap_0.5']} | served "
              f"{rec['served'] * 100:.3f}% | rss {rec['maxrss_mb']:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["rts", "ferc"], required=True)
    ap.add_argument("--line_scale", type=float, default=1.0)
    ap.add_argument("--reserve", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--time_limit", type=float, default=300.0)
    ap.add_argument("--gap", type=float, default=1e-3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    (run_rts if a.case == "rts" else run_ferc)(a)


if __name__ == "__main__":
    main()
