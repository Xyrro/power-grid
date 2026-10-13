"""Cost-aware checks for fixed commitments before / after the reduced MILP (12-hour UC).

The recommended pipeline (docs/methods/m1x.md, last section) fixes B-hg decisions, applies the adequacy guard, the min
up/down row release and the LP-relaxation guard, and solves the reduced MILP at a 0.5 % gap. The guards catch fixings
that make the relaxed reduced problem shed load or miss reserve. They cannot catch a wrong fixing that is merely
expensive: the reduced problem stays feasible without penalties but its optimum is far above the full problem's.

Pre-solve (LP) check: Delta = (z_RR - z_LP) / z_LP, the LP relaxation of the reduced problem against the full LP
relaxation solved anyway as model input (z_RR bounds the reduced MILP, z_LP the full MILP); response: release fixings by
their first-order saving in the relaxed reduced LP (release_by_saving), or keep only the fixings that agree with the
relaxation (consensus_fixings). The study (docs/methods/cguard.md) found that this check does not see the failures.

Post-solve bound check: g = (C_red - z_LP) / C_red of the reduced MILP's solution (bound_gap; free to compute);
responses: gradient polish around the solution (gradient_polish, otsl.solver's dispatch-LP derivative) or the full
MILP (solved by the caller). The bound check with a full-MILP fallback is the check that works.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def relaxed_reduced_lp(m, sc, fix):
    """LP relaxation of the reduced problem: u[t, g] = v for (t, g) in fix, every other u in [0, 1]; v, w, dispatch
    continuous. Returns dict(cost, slack [T] (shed + spill + shortfall per hour), saving {(t, g): first-order cost
    saving of releasing that fixing, from the bound marginals}, u [T, G], seconds)."""
    t0 = time.time()
    T, G = m.T, m.dims["G"]
    lo, hi, lb, ub = m._rhs_bounds(sc)
    for (t, g), v in fix.items():
        i = m.ix("u", t, g)
        lb[i] = ub[i] = v
    eq = lo == hi
    res = linprog(m.c, A_ub=sp.vstack([m.A[~eq & np.isfinite(hi)], -m.A[~eq & np.isfinite(lo)]]),
                  b_ub=np.r_[hi[~eq & np.isfinite(hi)], -lo[~eq & np.isfinite(lo)]],
                  A_eq=m.A[eq], b_eq=lo[eq], bounds=np.c_[lb, ub], method="highs")
    if res.status != 0:
        return dict(cost=np.inf, slack=np.full(T, np.inf), saving={}, u=None, seconds=time.time() - t0)
    x = res.x
    get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T].reshape(T, -1)
    marg = res.lower.marginals + res.upper.marginals          # d cost / d (bound value) of each column
    o, _ = m.off["u"]
    saving = {}
    for (t, g), v in fix.items():
        d = marg[o + t * G + g]
        saving[(t, g)] = float(d if v == 1 else -d)            # ON: lowering u saves d; OFF: raising u saves -d
    slack = get("shed").sum(1) + get("spill").sum(1) + get("short").sum(1)
    return dict(cost=float(res.fun), slack=slack, saving=saving, u=get("u"), seconds=time.time() - t0)


def lp_increase(z_rr, z_lp):
    return (z_rr - z_lp) / abs(z_lp)


def release_by_saving(m, sc, fix, z_lp, delta, batch=0.1, max_iter=6, min_batch=5):
    """Gradient release on the relaxed reduced LP: while (z_RR - z_LP) / z_LP > delta, release the fixings with the
    largest positive first-order saving (a share `batch` of the fixings with positive saving, at least min_batch) and
    re-solve. Returns (fix, info) with info: delta_before, delta_after, released, n_lp, seconds."""
    t0 = time.time()
    fix = dict(fix)
    r = relaxed_reduced_lp(m, sc, fix)
    d0 = lp_increase(r["cost"], z_lp)
    d, n_lp, released = d0, 1, 0
    while d > delta and n_lp <= max_iter:
        cand = sorted(((s, tg) for tg, s in r["saving"].items() if s > 0), reverse=True)
        if not cand:
            break
        k = max(min_batch, int(np.ceil(batch * len(cand))))
        for _, tg in cand[:k]:
            del fix[tg]
            released += 1
        r = relaxed_reduced_lp(m, sc, fix)
        d = lp_increase(r["cost"], z_lp)
        n_lp += 1
    return fix, dict(delta_before=float(d0), delta_after=float(d), released=released, n_lp=n_lp,
                     seconds=time.time() - t0)


def consensus_fixings(fix, u_rel, tol=1e-6):
    """keep only the fixings that agree with an integral LP-relaxation value (|u_rel - v| <= tol)"""
    u = np.asarray(u_rel, float)
    return {(t, g): v for (t, g), v in fix.items() if abs(u[t, g] - v) <= tol}


# ============================================================================ post-solve check and its responses
def bound_gap(cost, z_lp):
    """(C - z_LP) / C: how far a solution of cost C can be from the optimum at most, by the LP relaxation bound"""
    return (cost - z_lp) / abs(cost)


def solve_with_start(m, sc, fix, time_limit=60.0, gap=5e-3, start=None):
    """(reduced) MILP through highspy (one thread) with an optional full primal start; returns the record of
    otsl.ood.solve_milp plus the primal vector x (None if no solution)."""
    from .ltfx import highs_solve
    lo, hi, lb, ub = m._rhs_bounds(sc)
    for (t, g), v in (fix or {}).items():
        i = m.ix("u", t, g)
        lb[i] = ub[i] = v
    t0 = time.time()
    r = highs_solve(m.c, m.A, lo, hi, lb, ub, m.integ, time_limit, gap, start=start)
    dt = time.time() - t0
    out = dict(feasible=r["x"] is not None, obj=float(r["obj"]), bound=float(r["bound"]), time=float(dt),
               status=r["status"], mip_gap=float(r["gap"]) if np.isfinite(r["gap"]) else None, x=r["x"])
    if r["x"] is not None:
        x, T = r["x"], m.T
        get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T]
        out.update(shed=float(get("shed").sum()), spill=float(get("spill").sum()), short=float(get("short").sum()),
                   u=np.round(get("u")).reshape(T, -1).astype(np.int8))
    return out


def gradient_polish(m, sc, fix, x_hat, u_hat, n_release=60, time_limit=60.0, gap=5e-3):
    """otsl.solver's gradient release around an integer solution: the exact derivative of the dispatch-LP cost with
    respect to u at u_hat predicts the saving of flipping each decision; the n_release fixings with the largest positive
    predicted saving are released and the reduced MILP is re-solved from x_hat (still a feasible start). Returns
    (record, info) with info: released, grad_s."""
    from .solver import flip_savings, gradient_release
    sav, grad_s = flip_savings(m, sc, u_hat)
    fix2, rel = gradient_release(fix, sav, n_release)
    r = solve_with_start(m, sc, fix2, time_limit, gap, start=x_hat)
    return r, dict(released=len(rel), grad_s=float(grad_s), n_fix=len(fix2))
