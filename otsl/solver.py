"""Solver-side alternatives to hard fixing for learned unit commitment (12- and 24-hour RTS-GMLC UC).

Same probabilities / error-cost scores as the fixing rules, used inside the solver instead of (or after) hard fixing:

* solve_uc_hs     the UC MILP of otsl.uc.UCModel through highspy (identical model to UCModel.solve_uc and
                  otsl.b3.solve_milp_hs) with optional hard fixings, extra linear rows (trust region, local
                  branching), a complete MIP start, and an incumbent trace (time, objective, dual bound) that also
                  records whether HiGHS accepted the start.
* trust_region    Predict-and-Search (Han et al., ICLR 2023; ConPaS, Huang et al., ICML 2024): the k0 most confident
                  OFF and k1 most confident ON decisions are not fixed; one row allows at most Delta of them to deviate
                  from the prediction:  sum_{S0} u + sum_{S1} (1 - u) <= Delta.  Optionally combined with hard fixing of
                  a more confident core (hard set H, trust region over S minus H).
* complete_start  a complete primal solution (u, v, w + the dispatch LP of u for every continuous variable) of a
                  schedule; schedule_for_fixings makes a decoded schedule consistent with a set of fixings (forced
                  values, min up/down-feasible rows) so that it is a valid start of the reduced MILP.
* polish          after a reduced MILP: (a) local branching on the full MILP around its incumbent (Hamming radius r,
                  warm start, time limit); (b) gradient release: the dispatch LP's exact derivative d cost / d u at the
                  incumbent ranks the fixings by the saving a flip promises, the m most promising are released and
                  the reduced MILP is re-solved from the incumbent (time limit).

Nothing in the existing modules is changed; the guards come from otsl.combo / otsl.fixpolicy / otsl.hybrid.
"""
from __future__ import annotations

import time

import highspy
import numpy as np
import scipy.sparse as sp

from .fixpolicy import repair_row_forced
from .uc import UCModel, UCScenario, repair_min_updown


# ============================================================================ MILP through highspy
def u_columns(m: UCModel):
    """[T, G] column index of u[t, g]"""
    o, k = m.off["u"]
    return o + np.arange(m.T * k).reshape(m.T, k)


def hamming_row(m: UCModel, ref_u, mask=None):
    """row (cols, coefs, rhs_offset) of sum_{mask, ref=0} u + sum_{mask, ref=1} (1 - u) as  coefs . u + offset"""
    U = u_columns(m)
    ref_u = np.asarray(ref_u).reshape(U.shape)
    mask = np.ones(U.shape, bool) if mask is None else np.asarray(mask, bool)
    cols = U[mask]
    coef = np.where(ref_u[mask] > 0.5, -1.0, 1.0)
    offset = float((ref_u[mask] > 0.5).sum())
    return cols, coef, offset


def solve_uc_hs(m: UCModel, sc: UCScenario, time_limit=60.0, mip_gap=1e-3, z_fix: dict | None = None,
                rows: list | None = None, start_x=None, threads=1, objective_bound=None, log_file=None,
                opts: dict | None = None):
    """Solve the UC MILP with highspy.

    z_fix: {(t, g): 0/1} hard fixings (bounds). rows: extra rows [(cols, coefs, lo, hi)]. start_x: a complete primal
    vector (length m.nv) passed with setSolution (MIP start). Returns dict(status, obj, u, x, time, cpu, gap, bound,
    inc [(s, obj, bound)], nodes, shed, short, start_obj, start_accepted, t_build)."""
    tb = time.time()
    lo, hi, lb, ub = m._rhs_bounds(sc)
    for (t, g), v in (z_fix or {}).items():
        i = m.ix("u", t, g)
        lb[i] = ub[i] = v
    A = m.A
    if rows:
        R, rlo, rhi = [], [], []
        for cols, coefs, l_, h_ in rows:
            r = sp.csr_matrix((np.asarray(coefs, float), (np.zeros(len(cols), int), np.asarray(cols, int))),
                              shape=(1, m.nv))
            R.append(r); rlo.append(l_); rhi.append(h_)
        A = sp.vstack([A] + R, format="csr")
        lo, hi = np.r_[lo, rlo], np.r_[hi, rhi]
    A = A.tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", log_file is not None)
    if log_file is not None:
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("log_file", log_file)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.setOptionValue("threads", int(threads))
    if objective_bound is not None:
        h.setOptionValue("objective_bound", float(objective_bound))
    for k, v in (opts or {}).items():
        h.setOptionValue(k, v)
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = m.c
    lp.col_lower_ = lb
    lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in m.integ]
    h.passModel(lp)
    start_obj = None
    if start_x is not None:
        x0 = np.asarray(start_x, float).copy()
        ii = m.integ > 0
        x0[ii] = np.round(x0[ii])
        start_obj = float(m.c @ x0)
        s = highspy.HighsSolution()
        s.col_value = list(x0)
        s.value_valid = True
        h.setSolution(s)
    inc = []

    def cb(e):
        o = e.data_out
        inc.append((float(o.running_time), float(o.objective_function_value), float(o.mip_dual_bound)))
    h.cbMipImprovingSolution.subscribe(cb)
    t_build = time.time() - tb
    t0, c0 = time.time(), time.process_time()
    h.run()
    dt, cpu = time.time() - t0, time.process_time() - c0
    st = h.getModelStatus()
    info = h.getInfo()
    name = h.modelStatusToString(st)
    accepted = None
    if start_obj is not None:
        # the start is accepted if the first incumbent HiGHS reports has the start's objective (it is reported
        # through the improving-solution callback at time ~0), or if the final solution is the start itself
        accepted = bool(inc and abs(inc[0][1] - start_obj) <= 1e-6 * max(1.0, abs(start_obj)))
    base = dict(time=dt, cpu=cpu, inc=inc, nodes=int(info.mip_node_count), start_obj=start_obj,
                start_accepted=accepted, t_build=t_build, bound=float(info.mip_dual_bound))
    if info.primal_solution_status != 2:
        return dict(base, status="infeasible" if "nfeasible" in name else "no_solution", obj=np.inf, u=None, x=None,
                    gap=np.inf)
    x = np.array(h.getSolution().col_value)
    u, p, r, th, f, shed = m._unpack(x)
    status = "optimal" if name == "Optimal" else ("time_limit" if "Time limit" in name else name)
    return dict(base, status=status, obj=float(info.objective_function_value), u=np.round(u).astype(np.int8), x=x,
                gap=float(info.mip_gap), shed=float(shed), short=float(m._short))


def time_to_target(inc, target, t_offset=0.0):
    """first time (s, + t_offset) at which the incumbent trace reaches obj <= target; nan if never"""
    for t, o, _ in inc:
        if o <= target + 1e-9 * abs(target):
            return t + t_offset
    return np.nan


# ============================================================================ complete MIP starts
def dispatch_x(m: UCModel, sc: UCScenario, u):
    """complete primal vector of a commitment u [T, G]: u, v, w fixed, the dispatch LP for everything else
    (always feasible: shedding, over-generation and reserve shortfall are priced slacks). Returns (x, obj, seconds)."""
    t0 = time.time()
    lo, hi, lb, ub = m._rhs_bounds(sc, u)
    A = m.A.tocsc()
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = m.c
    lp.col_lower_ = lb
    lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    h.run()
    if h.getInfo().primal_solution_status != 2:
        return None, np.inf, time.time() - t0
    x = np.array(h.getSolution().col_value)
    return x, float(h.getInfo().objective_function_value), time.time() - t0


def schedule_for_fixings(u, fix, sysm, u0):
    """u [T, G] made consistent with the fixings {(t, g): v}: fixed entries forced, every row whose forced or
    original entries break min up/down replaced by the Hamming-nearest feasible row through the forced values
    (otsl.fixpolicy.repair_row_forced); other rows min up/down-repaired. None if a row cannot satisfy its fixings."""
    u = np.array(u, np.int8, copy=True)
    T, G = u.shape
    rows = {}
    for (t, g), v in fix.items():
        rows.setdefault(g, {})[t] = int(v)
    out = repair_min_updown(u, u0, sysm.min_up, sysm.min_dn)
    for g, forced in rows.items():
        if all(out[t, g] == v for t, v in forced.items()):
            continue
        r = repair_row_forced(u[:, g], int(u0[g]), int(sysm.min_up[g]), int(sysm.min_dn[g]), forced)
        if r is None:
            return None
        out[:, g] = r
    return out


# ============================================================================ trust region (Predict-and-Search)
def pick_confident(score, yhat, q0, q1):
    """(S0, S1) boolean masks: the q0 share of predicted-OFF decisions and the q1 share of predicted-ON decisions with
    the lowest score (score = expected error cost or error probability; lower = more confident)."""
    S0, S1 = np.zeros(yhat.shape, bool), np.zeros(yhat.shape, bool)
    for val, q, S in ((0, q0, S0), (1, q1, S1)):
        idx = np.flatnonzero(yhat.reshape(-1) == val)
        k = int(round(q * len(idx)))
        if k <= 0:
            continue
        o = idx[np.argsort(score.reshape(-1)[idx], kind="stable")[:k]]
        S.reshape(-1)[o] = True
    return S0, S1


def trust_region_row(m: UCModel, S0, S1, delta):
    """the Predict-and-Search row sum_{S0} u + sum_{S1} (1 - u) <= delta as (cols, coefs, lo, hi)"""
    U = u_columns(m)
    cols = np.r_[U[S0], U[S1]]
    coefs = np.r_[np.ones(int(S0.sum())), -np.ones(int(S1.sum()))]
    return cols, coefs, -np.inf, float(delta) - float(S1.sum())


def masks_from_fix(fix, T, G):
    off, on = np.zeros((T, G), bool), np.zeros((T, G), bool)
    for (t, g), v in fix.items():
        (on if v else off)[t, g] = True
    return off, on


# ============================================================================ polish
def local_branching_row(m: UCModel, u_ref, radius, mask=None):
    """Hamming ball of radius r around u_ref (over mask, default all u): (cols, coefs, lo, hi)"""
    cols, coef, off = hamming_row(m, u_ref, mask)
    return cols, coef, -np.inf, float(radius) - off


def flip_savings(m: UCModel, sc: UCScenario, u):
    """first-order saving of flipping each u[t, g] (positive = the flip is predicted to lower the cost), from the
    dispatch LP's exact derivative d cost / d u at u (otsl.uc.UCModel.dispatch_gradient). Returns (saving, seconds)."""
    t0 = time.time()
    _, grad = m.dispatch_gradient(sc, u)
    sav = np.where(np.asarray(u) > 0.5, grad, -grad)        # ON: lowering u saves grad; OFF: raising u saves -grad
    return sav, time.time() - t0


def gradient_release(fix, saving, n_release):
    """release the n_release fixings with the largest positive predicted saving; returns (fix, released list)"""
    cand = sorted(((saving[t, g], (t, g)) for (t, g) in fix if saving[t, g] > 0), reverse=True)[:int(n_release)]
    fix = dict(fix)
    rel = [tg for _, tg in cand]
    for tg in rel:
        del fix[tg]
    return fix, rel
