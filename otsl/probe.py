"""Lagrangian probing: exact fixings from the LP relaxation's prices (docs/methods/agg.md).

Price the system rows of the LP relaxation (bus balances, reserve requirement) at their optimal duals y and every
thermal unit becomes its own problem: choose an on/off schedule that meets its minimum up/down times, paying in each
hour it is on its no-load cost minus the price of its minimum output, the best use of its headroom (segments of output
bought at their marginal cost and sold at the bus price, reserve sold at the reserve price), and its start-up cost.
A forward-backward dynamic programme over the unit's states (on or off for d hours) gives, for every hour, the cheapest
schedule that is on and the cheapest that is off; their difference is the hour's margin.

Why it is exact. For any y, dualising the system rows gives a valid lower bound (Lagrangian relaxation); at the LP's
optimal duals the bound with each unit's LP subproblem equals z_LP, and each unit's integer subproblem is no cheaper
than its LP one. Forcing hour t of unit g to the other state raises unit g's subproblem by its margin and leaves the
others unchanged, so every schedule that flips (t, g) costs at least z_LP + margin(t, g). Given any schedule of cost C,
the decisions with margin > C - z_LP therefore hold in every schedule cheaper than C: fixing them cannot cut off an
improvement, and a bound computed with them fixed is a bound for the full problem as far as C is concerned (the optimum
either respects them or costs more than C). This is reduced-cost fixing with the unit's whole schedule in view: a
single variable's LP reduced cost is the slope along one edge and is usually far smaller than the cost of the minimum
up or down block a flip forces.
"""
from __future__ import annotations

import time

import highspy
import numpy as np
import scipy.sparse as sp


def lp_relaxation_duals(m, sc):
    """LP relaxation of the model m (otsl.uc.UCModel / OODModel, network or copper plate) with HiGHS. Returns dict(obj,
    x, row_dual, lmp [T, N] (bus balance duals; copper plate: [T, 1]), mu [T] (reserve requirement duals), time)."""
    lo, hi, lb, ub = m._rhs_bounds(sc)
    A = sp.csc_matrix(m.A)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(m.c, float)
    lp.col_lower_, lp.col_upper_ = lb, ub
    lp.row_lower_, lp.row_upper_ = lo, hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    sol = h.getSolution()
    y = np.array(sol.row_dual)
    T = m.T
    if m.network:
        N = m.dims["N"]
        lmp = np.array([[y[m.row_tag[("kcl", t, n)][0]] for n in range(N)] for t in range(T)])
    else:
        lmp = np.array([[y[m.row_tag[("bal", t)][0]]] for t in range(T)])
    mu = np.array([y[m.row_tag[("sr", t)][0]] for t in range(T)])
    return dict(obj=float(h.getInfo().objective_function_value), x=np.array(sol.col_value), row_dual=y, lmp=lmp, mu=mu,
                time=dt)


def hourly_on_cost(sysm, lmp, mu, gbus=None):
    """[T, G] cost of being on in each hour at prices (lmp [T, N] or [T, 1], mu [T]): no-load cost minus the value of the
    minimum output plus the cheapest use of the headroom pmax - pmin (output segments at marginal cost minus price,
    reserve at minus mu; a fractional knapsack whose capacity the segments exactly fill)."""
    s = sysm
    T, G = lmp.shape[0], s.G
    lam = lmp[:, s.gbus] if lmp.shape[1] > 1 else np.repeat(lmp, G, axis=1)          # [T, G]
    a = s.c_nl[None, :] - lam * s.pmin[None, :]
    H = s.pmax - s.pmin
    out = np.zeros((T, G))
    for t in range(T):
        for g in range(G):
            opts = [(s.seg_c[g, k] - lam[t, g], s.seg_w[g, k]) for k in range(s.seg_w.shape[1]) if s.seg_w[g, k] > 0]
            opts.append((-mu[t], H[g]))
            cap, val = H[g], 0.0
            for c, w in sorted(opts):
                if c >= 0 or cap <= 0:
                    break
                q = min(w, cap)
                val += c * q
                cap -= q
            out[t, g] = a[t, g] + val
    return out


def unit_margins(sysm, u0, on_cost, g_out=()):
    """Per unit, a forward-backward DP over (on/off, hours in state) with minimum up/down times (every initial unit has
    met them, as in otsl.uc.UCModel), hourly on-cost on_cost [T, G] and start-up cost c_su. Returns (best [G] cheapest
    schedule cost, sched [T, G] a cheapest schedule, margin [T, G] = cost of the cheapest schedule in the other state
    in that hour minus best; 0 where both are cheapest)."""
    s = sysm
    T, G = on_cost.shape
    best = np.zeros(G)
    sched = np.zeros((T, G), np.int8)
    margin = np.zeros((T, G))
    out = set(int(g) for g in g_out)
    INF = np.inf
    for g in range(G):
        if g in out:
            best[g], margin[:, g] = 0.0, INF
            continue
        U, D = int(s.min_up[g]), int(s.min_dn[g])
        states = [(1, k) for k in range(1, U + 1)] + [(0, k) for k in range(1, D + 1)]
        idx = {st: i for i, st in enumerate(states)}
        S = len(states)

        def nxt(st):
            on, k = st
            res = [((on, min(k + 1, U if on else D)), 0.0)]
            if on and k >= U:
                res.append(((0, 1), 0.0))
            if not on and k >= D:
                res.append(((1, 1), float(s.c_su[g])))
            return res
        trans = [[(idx[b], c) for b, c in nxt(st)] for st in states]
        hour = lambda t, i: on_cost[t, g] if states[i][0] else 0.0
        F = np.full((T, S), INF)
        init = idx[(1, U)] if u0[g] else idx[(0, D)]
        for j, c in trans[init]:
            F[0, j] = min(F[0, j], c + hour(0, j))
        for t in range(1, T):
            for i in range(S):
                if F[t - 1, i] < INF:
                    for j, c in trans[i]:
                        v = F[t - 1, i] + c + hour(t, j)
                        if v < F[t, j]:
                            F[t, j] = v
        B = np.zeros((T, S))
        for t in range(T - 2, -1, -1):
            for i in range(S):
                B[t, i] = min(c + hour(t + 1, j) + B[t + 1, j] for j, c in trans[i])
        tot = F + B
        on_mask = np.array([st[0] == 1 for st in states])
        b_on = tot[:, on_mask].min(1)
        b_off = tot[:, ~on_mask].min(1)
        best[g] = float(min(b_on.min(), b_off.min()))
        sched[:, g] = (b_on < b_off).astype(np.int8)
        margin[:, g] = np.abs(b_on - b_off)
    return best, sched, margin


def probe_fixings(sysm, sc, lmp, mu, z_lp, C, rel_tol=1e-7):
    """Exact fixings {(t, g): 0/1}: every unit-hour whose margin exceeds C - z_LP (with a small tolerance), fixed to the
    state of the unit's cheapest schedule at the LP prices. Returns (fix, margin, sched)."""
    on = hourly_on_cost(sysm, lmp, mu)
    _, sched, margin = unit_margins(sysm, sc.u0, on, getattr(sc, "g_out", ()) or ())
    thr = (C - z_lp) + rel_tol * abs(C) + 1e-6
    fix = {(int(t), int(g)): int(sched[t, g]) for t, g in np.argwhere(margin > thr)}
    return fix, margin, sched


def margin_feature(sysm, u0, lmp, mu, z_lp, g_out=(), clip=2.0):
    """[T, G] signed probing margin as a model input: the margin in % of z_LP, positive where the unit's cheapest
    schedule at the LP prices is on and negative where it is off, clipped to +-clip (0 where both states tie)"""
    on = hourly_on_cost(sysm, lmp, mu)
    _, sched, margin = unit_margins(sysm, u0, on, g_out)
    m = np.minimum(np.nan_to_num(margin, posinf=clip * abs(z_lp) / 100) / abs(z_lp) * 100, clip)
    return np.where(sched > 0, m, -m).astype(np.float32)
