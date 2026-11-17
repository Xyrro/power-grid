"""Round 8 on the PGLib-UC California instances as distributed (48 h, 610 units, no perturbation; docs/methods/agg.md):
the full MILP against the learning-free certified pipeline, back to back in one single-threaded process.

    PGLIB=<pglib-uc> OTSL_THREADS=1 python3 scripts/agg_pglib_base.py --file ca/2014-09-01_reserves_0.json --out x.jsonl

Methods (no learning, nothing trained or tuned on these instances):
  full@g                the full MILP at gap g (full@0.1% is the reference; --tl seconds)
  pcert:[r]lp:<d>:<e>   the certified pipeline on the LP relaxation's values (scripts/agg_pglib_eval.py): with 'r', the
                        rounded and repaired LP values first (0.5 and 0.1; one dispatch LP each), delivered when the LP
                        bound proves them within e %; otherwise every value with min(u, 1 - u) <= d fixed, rows min up/down
                        cannot complete released, reduced MILP at e, the LP bound, else the full MILP at e warm-started.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.fixpolicy import release_conflicting_rows  # noqa: E402
from otsl.pglib import PGLIB, PGModel, check_no_carryover, highs_run, load_case, rep_block  # noqa: E402

ROUND_TH = (0.5, 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="path inside the pglib-uc repository, e.g. ca/2014-09-01_reserves_0.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--tl", type=float, default=1800.0)
    ap.add_argument("--pcert", default="rlp:0.05:0.5,rlp:0.05:0.2,rlp:0.05:0.1,lp:0.05:0.5")
    ap.add_argument("--full", default="0.1,0.5")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if os.path.exists(a.out) and any(json.loads(l)["file"] == a.file for l in open(a.out)):
        print(f"{a.file} already in {a.out}")
        return
    sysm, sc, _ = load_case(os.path.join(PGLIB, a.file), T=a.T)
    assert check_no_carryover(sysm), "repairs assume every initial unit has served its minimum time"
    m = PGModel(sysm, T=a.T)
    inst = dict(load=sc.load, avail=sc.avail, sr=sc.sr, u0=sc.u0)
    m.solve_dispatch(sc, None, relax=True)                          # warm-up
    t0 = time.perf_counter()
    rel = m.solve_dispatch(sc, None, relax=True)
    t_rel = time.perf_counter() - t0

    def milp(fix, gap, start=None, tl=a.tl):
        lo, hi, lb, ub = m.bounds_fixed(sc, fix)
        return highs_run(m.c, m.A, lo, hi, lb, ub, m.integ, tl, gap, 1, False, start=start)

    def u_of(x):
        return np.round(m._unpack(x)[0]).astype(np.int8)

    meth = {}
    for g in (float(x) for x in a.full.split(",")):
        def full(g=g / 100):
            t0 = time.perf_counter()
            r = milp(None, g)
            return (u_of(r["x"]) if r["x"] is not None else None), time.perf_counter() - t0, dict(
                obj=r["obj"], bound=r["bound"], status=r["status"])
        meth[f"full@{g:g}%"] = full
    for spec in a.pcert.split(","):
        rnd = spec.startswith("r")
        _, dl, ep = spec.lstrip("r").split(":")

        def pcert(dl=float(dl), eps=float(ep) / 100, rnd=rnd):
            p = np.clip(rel.u, 0, 1)
            t0 = time.perf_counter()
            if rnd:
                best = None
                for th in ROUND_TH:
                    u0 = rep_block((p > th).astype(np.int8), inst, sysm)
                    c0 = m.solve_dispatch(sc, u0).obj
                    if best is None or c0 < best[1]:
                        best = (u0, c0)
                g0 = (best[1] - rel.obj) / best[1]
                if g0 <= eps:
                    return best[0], t_rel + time.perf_counter() - t0, dict(cert="round", lp_gap=g0, n_fixed=0)
            fix = {(int(t), int(g)): int(p[t, g] > 0.5) for t, g in np.argwhere(np.minimum(p, 1 - p) <= dl)}
            fix, _ = release_conflicting_rows(fix, sysm, sc.u0)
            r = milp(fix, eps)
            info = dict(n_fixed=len(fix), red_obj=r["obj"], red_s=time.perf_counter() - t0)
            if r["x"] is None:
                r = milp(None, eps)
                info["fallback"] = True
            C = r["obj"]
            g1 = (C - rel.obj) / C
            info["lp_gap"] = g1
            if g1 <= eps:
                info["cert"] = "lp"
                return u_of(r["x"]), t_rel + time.perf_counter() - t0, info
            z = milp(None, eps, start=r["x"], tl=max(a.tl - (time.perf_counter() - t0), 1.0))
            x = z["x"] if z["x"] is not None and z["obj"] <= C else r["x"]
            info.update(cert="milp", cert_obj=z["obj"], cert_bound=z["bound"], cert_status=z["status"])
            return u_of(x), t_rel + time.perf_counter() - t0, info
        meth[f"pcert:{spec}"] = pcert

    order = sorted(meth)
    np.random.default_rng(abs(hash(a.file)) % (2 ** 32)).shuffle(order)
    import highspy
    h = highspy.Highs()
    rec = dict(file=a.file, T=a.T, c_rel=rel.obj, t_rel=t_rel, host=socket.gethostname(),
               highs=f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}", order=order, runs={})
    try:
        rec["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    for name in order:
        u, dt, info = meth[name]()
        pr = m.solve_dispatch(sc, u) if u is not None else None
        rec["runs"][name] = dict(time=dt, cost=pr.obj if pr else float("inf"),
                                 served=bool(pr is not None and pr.shed + pr.short < 1e-6), **info)
        print(f"  {name:22s} {dt:8.2f}s  cost {rec['runs'][name]['cost']:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    main()
