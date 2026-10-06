"""No-learning baselines and fair solver budgets for partial fixing in network-constrained UC (RTS-GMLC, 12 h / 24 h).

Baselines (no model, no training data; the only input is the instance's LP relaxation):

* lp_integral_fixings   fix every commitment u[t, g] whose LP-relaxation value is within `tol` of 0 or 1 (the
                        "fix the integral values" rule of docs/methods/pglib.md, ported to RTS-GMLC); optionally
                        followed by our guards (otsl.hybrid.apply_guards).
* LP values as LtF probabilities: pi = u_rel fed to otsl.ltfx.LtFTuner unchanged (learning-free Learning to Fix);
                        nothing here, the tuner is used as is.
* lp_round_screen       end-to-end without a MILP (the PGLib "round the LP relaxation, repair, 5 LPs" baseline):
                        round u_rel at each threshold of E2E_TH, block (min up/down-aware) adequacy repair, min up/down
                        repair, exact dispatch LP of every distinct candidate, keep the cheapest.

Fair-budget helpers: incumbent_at / gap_within read the full MILP's incumbent log (what the full MILP has found
after a given time), solve_reduced wraps the highspy MILP path of otsl.b3 (incumbent log, optional bound trace).

Metric helpers (paper metrics of Learning to Fix, Sec. IV-C): gap (C - DB) / C to the full MILP's dual bound DB,
per-instance speed-up T_MILP / T_method, statistics over feasible instances; instance-bootstrap CIs and paired
bootstrap differences.
"""
from __future__ import annotations

import time

import numpy as np

from .constrained import adequacy_repair_blocks
from .uc import UCScenario, repair_min_updown

E2E_TH = (0.001, 0.05, 0.2, 0.5, 0.8)       # rounding thresholds of the PGLib end-to-end baseline (pglib.md, §5)


# ============================================================================ baselines
def lp_integral_fixings(u_rel, tol=1e-6):
    """{(t, g): v} for every decision whose LP-relaxation value is within tol of v in {0, 1}"""
    u = np.asarray(u_rel, float)
    lo, hi = u <= tol, u >= 1.0 - tol
    out = {(int(t), int(g)): 0 for t, g in zip(*np.where(lo))}
    out.update({(int(t), int(g)): 1 for t, g in zip(*np.where(hi & ~lo))})
    return out


def lp_round_candidates(u_rel, sc: UCScenario, sysm, thresholds=E2E_TH):
    """[(threshold, schedule)]: u_rel > threshold, block adequacy repair, min up/down repair (otsl.b3.rep_block)"""
    out = []
    for th in thresholds:
        u = (np.asarray(u_rel) > th).astype(np.int8)
        u = adequacy_repair_blocks(u, sc.u0, sc.load, sc.avail, sc.sr, sysm)
        out.append((th, repair_min_updown(u, sc.u0, sysm.min_up, sysm.min_dn)))
    return out


def lp_round_screen(m, sysm, sc: UCScenario, u_rel, thresholds=E2E_TH):
    """the cheapest (by the exact dispatch LP, penalties included) of the repaired roundings of u_rel. Identical
    candidate schedules are priced once. Returns dict(u, cost, shed, short, th, n_lp, t_repair, t_lp, costs)."""
    t0 = time.time()
    cands = lp_round_candidates(u_rel, sc, sysm, thresholds)
    t_rep = time.time() - t0
    seen, costs, best = {}, [], None
    t_lp = 0.0
    for th, u in cands:
        key = u.tobytes()
        if key not in seen:
            t1 = time.time()
            sol = m.solve_dispatch(sc, u)
            t_lp += time.time() - t1
            seen[key] = (float(sol.obj), float(sol.shed), float(sol.short))
        c, sh, so = seen[key]
        costs.append(c)
        if best is None or c < best["cost"] - 1e-9:
            best = dict(u=u, cost=c, shed=sh, short=so, th=th)
    best.update(n_lp=len(seen), t_repair=t_rep, t_lp=t_lp, costs=costs)
    return best


# ============================================================================ solver wrapper
def solve_reduced(m, sc: UCScenario, fix=None, time_limit=60.0, mip_gap=1e-3, trace_every=0.0):
    """full (fix empty) or reduced MILP through otsl.b3.solve_milp_hs (highspy, 1 thread, incumbent log). Returns a
    JSON-friendly record: feasible, obj, time, cpu, mip_gap, bound, status, nodes, shed, short, inc [(s, obj, bound)],
    trace [(s, primal, dual)] (if trace_every > 0), n_fixed."""
    from .b3 import solve_milp_hs
    sol = solve_milp_hs(m, sc, time_limit=time_limit, mip_gap=mip_gap, z_fix=fix or None, threads=1,
                        trace_every=trace_every)
    ok = sol["u"] is not None
    return dict(feasible=bool(ok), obj=float(sol["obj"]) if ok else None, time=float(sol["time"]),
                cpu=float(sol["cpu"]), mip_gap=float(sol["gap"]) if ok else None, bound=float(sol["bound"]),
                status=str(sol["status"]), nodes=int(sol.get("nodes", -1)),
                shed=float(sol.get("shed", np.nan)) if ok else None, short=float(sol.get("short", np.nan)) if ok else None,
                inc=[[float(a) for a in x] for x in sol["inc"]], trace=[[float(a) for a in x] for x in sol.get("trace", [])],
                n_fixed=len(fix or {}))


# ============================================================================ anytime view of the full MILP
def incumbent_at(inc, t):
    """best objective in an incumbent log [(seconds, objective, ...)] found by time t (inf if none)"""
    best = np.inf
    for row in inc:
        if row[0] <= t + 1e-12:
            best = min(best, float(row[1]))
    return best


def gap_within(inc, t, db):
    """paper gap (C - DB) / C in % of the full MILP's best incumbent after t seconds; nan if it has none yet"""
    c = incumbent_at(inc, t)
    return float((c - db) / c * 100) if np.isfinite(c) else np.nan


def time_to_gap(trace, g):
    """first time in a (seconds, primal, dual) trace at which (primal - dual) / |primal| <= g (nan if never)"""
    for t, pb, db in trace:
        if np.isfinite(pb) and abs(pb) > 0 and (pb - db) / abs(pb) <= g + 1e-12:
            return float(t)
    return np.nan


# ============================================================================ paper metrics and bootstrap
def gap_db(cost, db):
    """(C - DB) / C in %, elementwise; nan where cost is missing"""
    c = np.asarray([np.nan if x is None else x for x in np.atleast_1d(cost)], float)
    return (c - np.asarray(db, float)) / c * 100


def boot_ci(x, B=2000, seed=0):
    """instance-bootstrap 95 % interval of the mean (finite values only)"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def paired(a, b, B=2000, seed=0):
    """mean of a - b over instances where both are finite, with its bootstrap 95 % interval"""
    x = np.asarray(a, float) - np.asarray(b, float)
    x = x[np.isfinite(x)]
    return dict(diff=float(x.mean()) if len(x) else np.nan, ci=boot_ci(x, B, seed), n=int(len(x)))


def paper_stats(feasible, cost, db, t_method, t_full, fixed=None, served=None):
    """Learning to Fix's Table I statistics: feasibility rate; over feasible instances the gap to DB (mean, median,
    max, CI), runtime, per-instance speed-up T_full / T_method (mean, median, max, CI), ratio of mean times, fixed
    share and served share. Inputs are per-instance arrays (cost may contain None for infeasible instances)."""
    f = np.asarray(feasible, bool)
    out = dict(n=int(len(f)), feasible=float(f.mean() * 100), n_infeasible=int((~f).sum()))
    if not f.any():
        return out
    g = gap_db(np.asarray(cost, object)[f], np.asarray(db, float)[f])
    tm, tf = np.asarray(t_method, float)[f], np.asarray(t_full, float)[f]
    sp = tf / tm
    out.update(gap_mean=float(g.mean()), gap_median=float(np.median(g)), gap_max=float(g.max()), gap_ci=boot_ci(g),
               time_mean=float(tm.mean()), time_median=float(np.median(tm)), speedup_mean=float(sp.mean()),
               speedup_median=float(np.median(sp)), speedup_max=float(sp.max()), speedup_ci=boot_ci(sp),
               speedup_ratio_of_means=float(tf.mean() / tm.mean()))
    if fixed is not None:
        out["fixed_mean"] = float(np.asarray(fixed, float)[f].mean() * 100)
    if served is not None:
        out["served"] = float(np.asarray(served, bool)[f].mean() * 100)
    return out
