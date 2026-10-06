"""Learning to Fix (Fritz, Makrides, Fetanat & Pinson 2026, arXiv 2609.39396), implemented from the full text.

Components (section / equation numbers of the paper):

* kNN classifier (IV-B): k = 50, Euclidean distance on the Table II features (Appendix B), on-probabilities by
  inverse-distance weighting of the neighbours' optimal schedules (eq. 13).
* Fixing rule (eq. 5): generator-specific thresholds [lo_g, hi_g] shared across hours; pi > hi fixes ON, pi < lo
  fixes OFF, otherwise the variable is left to the solver.
* Constant thresholds (eq. 6) and worst-case-misprediction thresholds (eq. 7).
* Suboptimality-constrained thresholds (III-B3): the tightest thresholds, by the probability mass left free (eq. 10,
  quantile approximation of Appendix A, Q = 20), such that *every* validation instance's reduced UC, with all its
  fixings applied jointly, has a solution within eps of C* (8e). Solved with Algorithm 1 (logic-based Benders
  decomposition: master over the thresholds, the validation instances as subproblems) and Algorithm 2 (ADD CUTS:
  up to K_max minimum-cardinality release sets per infeasible instance, no-good constraints between them, combined
  by a logical OR in the master with auxiliary binaries).

Implementation choices where the paper is silent are listed in docs/methods/ltfx.md (section 2); in code they are
marked "[choice]". The UC model is otsl.uc.UCModel (its matrices are reused; the check, relaxation and master
problems are solved with HiGHS through highspy).
"""
from __future__ import annotations

import time

import highspy
import numpy as np
import scipy.sparse as sp

from .uc import UCScenario
from .ucml import canonical_labels

FIX_TOL = 1e-8          # [choice] pi < lo - FIX_TOL fixes OFF, pi > hi + FIX_TOL fixes ON (master solutions sit on cuts)
WIDTH_FLOOR = 1e-6      # Appendix A: 1 / max(width, 1e-6); [choice] the same floor bounds the bin variable
TIE_W = 1e-3            # [choice] tie-break weight pulling flat thresholds towards 0.5 (standard rounding)


# ============================================================================ features (Table II) and kNN (eq. 13)
def table2_features(sysm, d):
    """Instance features z of Appendix B / Table II for 12-hour RTS-GMLC instances. [n, F]

    raw: aggregate demand L_t, wind W_t, solar S_t (PV + rooftop PV), hydro H_t; RES_t = W_t + S_t + H_t
    [choice: hydro is a curtailable profile here, as wind and solar]; net load N_t = L_t - RES_t; deltas of L, RES, N
    (0 in the first hour: the hour before the window is not part of an instance); 3-step moving average and rolling
    maximum of N (over the available steps at the start); N normalised by its mean / max over the horizon and over
    its day (a window lies within one calendar day, so these equal the horizon versions, kept as in Table II);
    sin / cos of 2 pi h / 24 with h the hour of day; initial condition of every unit (signed: +1 on, -1 off; the
    data assume no carry-over of min up / down times, so the number of hours in state is not available)."""
    n, T, _ = d["load"].shape
    L = d["load"].sum(2)
    rt = np.asarray(sysm.r_type)
    av = d["avail"]
    W = av[:, :, rt == "WIND"].sum(2)
    S = av[:, :, np.isin(rt, ["PV", "RTPV"])].sum(2)
    H = av[:, :, np.isin(rt, ["HYDRO", "ROR"])].sum(2)
    RES = W + S + H
    N = L - RES
    delta = lambda V: np.c_[np.zeros((n, 1)), np.diff(V, axis=1)]
    ma3 = np.stack([N[:, max(0, t - 2):t + 1].mean(1) for t in range(T)], 1)
    mx3 = np.stack([N[:, max(0, t - 2):t + 1].max(1) for t in range(T)], 1)
    nmean = N / N.mean(1, keepdims=True)
    nmax = N / N.max(1, keepdims=True)
    hour = np.asarray(d["start"])[:, None] + np.arange(T)[None, :]
    sin, cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    eta = np.where(np.asarray(d["u0"]) > 0, 1.0, -1.0)
    blocks = [L, W, S, H, RES, N, delta(L), delta(RES), delta(N), ma3, mx3, nmean, nmax, nmean, nmax, sin, cos, eta]
    return np.concatenate(blocks, 1)


class KNNProb:
    """kNN on-probabilities (eq. 13): pi = sum_j u*_j / rho_j / sum_j 1 / rho_j over the k nearest training
    instances, rho = Euclidean distance between feature vectors. [choice] features are z-scored on the training
    set (the paper mixes MW, ratios, harmonics and hours; it does not state a scaling); labels are the optimal
    schedules canonicalised inside groups of identical units (an equally optimal schedule; without it the
    labels of identical copies are arbitrary)."""

    def __init__(self, sysm, train, k=50):
        self.s, self.k = sysm, k
        Z = table2_features(sysm, train)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        self.X = (Z - self.mu) / self.sd
        self.Y = canonical_labels(sysm, train["u"], train["u0"]).astype(np.float64)

    def predict(self, d):
        Z = (table2_features(self.s, d) - self.mu) / self.sd
        D2 = (Z ** 2).sum(1)[:, None] + (self.X ** 2).sum(1)[None] - 2 * Z @ self.X.T
        idx = np.argsort(D2, 1, kind="stable")[:, :self.k]
        rho = np.sqrt(np.maximum(np.take_along_axis(D2, idx, 1), 0))
        w = 1.0 / np.maximum(rho, 1e-9)
        w /= w.sum(1, keepdims=True)
        return np.einsum("nk,nktg->ntg", w, self.Y[idx])


# ============================================================================ fixing rule and simple thresholds
def fix_masks(pi, lo, hi, tol=FIX_TOL):
    """eq. (5): (off, on) boolean masks [.., T, G] of the decisions fixed to 0 / 1"""
    return pi < (np.asarray(lo) - tol), pi > (np.asarray(hi) + tol)


def fix_dict(pi, lo, hi):
    off, on = fix_masks(pi, lo, hi)
    out = {(int(t), int(g)): 0 for t, g in zip(*np.where(off))}
    out.update({(int(t), int(g)): 1 for t, g in zip(*np.where(on))})
    return out


def constant_thresholds(G, r):
    """eq. (6): [r, 1 - r] for every generator"""
    return np.full(G, r), np.full(G, 1 - r)


def worst_case_thresholds(pi, u):
    """eq. (7): lo_g = smallest probability of an optimal ON decision, hi_g = largest probability of an optimal OFF
    decision (validation set); if lo_g > hi_g both are set to their midpoint. A generator never ON (OFF) in the
    labels gets lo_g = 1 (hi_g = 0) before the midpoint rule [choice: min / max over an empty set]."""
    G = pi.shape[-1]
    p = pi.reshape(-1, G)
    y = u.reshape(-1, G).astype(bool)
    lo = np.where(y.any(0), np.where(y, p, np.inf).min(0), 1.0)
    hi = np.where((~y).any(0), np.where(~y, p, -np.inf).max(0), 0.0)
    mid = (lo + hi) / 2
    swap = lo > hi
    return np.where(swap, mid, lo), np.where(swap, mid, hi)


# ============================================================================ HiGHS helper
def highs_solve(c, A, lo, hi, lb, ub, integ=None, time_limit=60.0, gap=1e-3, start=None, opts=None):
    """Solve min c x, lo <= A x <= hi, lb <= x <= ub (integ: 0/1 mask of integer columns) with HiGHS.
    Returns dict(status, x, obj, bound, time, gap)."""
    A = sp.csc_matrix(A)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(c, float)
    lp.col_lower_, lp.col_upper_ = np.asarray(lb, float), np.asarray(ub, float)
    lp.row_lower_, lp.row_upper_ = np.asarray(lo, float), np.asarray(hi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    if integ is not None and np.any(integ):
        K, C = highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous
        lp.integrality_ = [K if v else C for v in integ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(gap))
    for k, v in (opts or {}).items():
        h.setOptionValue(k, v)
    h.passModel(lp)
    if start is not None:
        s = highspy.HighsSolution()
        s.col_value = list(np.asarray(start, float))
        s.value_valid = True
        h.setSolution(s)
    t0 = time.time()
    h.run()
    dt = time.time() - t0
    info = h.getInfo()
    st = h.modelStatusToString(h.getModelStatus())
    feas = info.primal_solution_status == 2
    x = np.array(h.getSolution().col_value) if feas else None
    return dict(status=st, x=x, obj=float(info.objective_function_value) if feas else np.inf,
                bound=float(getattr(info, "mip_dual_bound", np.nan)), time=dt,
                gap=float(getattr(info, "mip_gap", np.nan)))


# ============================================================================ subproblems on one UC instance
class InstanceProblems:
    """The check (Algorithm 1, step 5) and the relaxation MILP (Algorithm 2, step 4) of one UC instance, built
    from UCModel's matrices."""

    def __init__(self, m, sc: UCScenario, c_star, eps):
        self.m, self.sc, self.eps = m, sc, eps
        self.c_star = float(c_star)
        self.cap = (1.0 + eps) * self.c_star          # (8e)
        self.lo, self.hi, self.lb, self.ub = m._rhs_bounds(sc)
        self.T, self.G = m.T, m.dims["G"]
        o, _ = m.off["u"]
        self.uix = o + np.arange(self.T * self.G).reshape(self.T, self.G)     # column of u[t, g]

    def _u(self, x):
        return np.round(x[self.uix.reshape(-1)]).reshape(self.T, self.G).astype(np.int8)

    def check(self, off, on, time_limit=60.0):
        """Is there a solution of the reduced UC (fixings off / on) with cost <= (1 + eps) C*? Solved as the reduced
        MILP with the tolerance as objective cut-off, stopped at the first solution under it. Returns
        (ok, info) with info = dict(status, obj, time, u)."""
        lb, ub = self.lb.copy(), self.ub.copy()
        ub[self.uix[off]] = 0.0
        lb[self.uix[on]] = 1.0
        r = highs_solve(self.m.c, self.m.A, self.lo, self.hi, lb, ub, self.m.integ, time_limit, 1e-3,
                        opts={"objective_bound": self.cap * (1 + 1e-9), "mip_max_improving_sols": 1})
        ok = r["x"] is not None and r["obj"] <= self.cap * (1 + 1e-7)
        return ok, dict(status=r["status"], obj=r["obj"], time=r["time"], u=self._u(r["x"]) if r["x"] is not None else None)

    def label_start(self, u_label):
        """full primal solution for a given commitment (the dispatch LP), used as MIP start"""
        lo, hi, lb, ub = self.m._rhs_bounds(self.sc, u_label)
        r = highs_solve(self.m.c, self.m.A, lo, hi, lb, ub, None, 60.0)
        return r["x"], r["obj"]

    def relaxation(self, off, on, nogoods, time_limit=60.0, start_x=None):
        """Algorithm 2, step 4: min sum nu over the fixed decisions s.t. u <= nu (fixed OFF), 1 - u <= nu (fixed ON),
        no-good constraints sum_{R_l} nu <= |R_l| - 1, the UC constraints and the cost tolerance (8e)-(8g).
        nogoods: list of release sets, each a list of (t, g). Returns (R, info): R = list of (t, g) whose fixing
        must be released (nu = 1 and u differs from the fixed value), or None if no solution was found."""
        m = self.m
        F = [(int(t), int(g), 0) for t, g in zip(*np.where(off))] + [(int(t), int(g), 1) for t, g in zip(*np.where(on))]
        pos = {(t, g): j for j, (t, g, _) in enumerate(F)}
        nf, nv = len(F), m.nv
        rows, cols, vals, rlo, rhi = [], [], [], [], []
        for j, (t, g, v) in enumerate(F):
            rows += [j, j]
            cols += [int(self.uix[t, g]), nv + j]
            if v == 0:                       # u - nu <= 0
                vals += [1.0, -1.0]
                rlo.append(-np.inf); rhi.append(0.0)
            else:                            # u + nu >= 1
                vals += [1.0, 1.0]
                rlo.append(1.0); rhi.append(np.inf)
        r0 = nf
        cidx = np.nonzero(m.c)[0]            # cost tolerance row
        rows += [r0] * len(cidx); cols += list(cidx); vals += list(m.c[cidx])
        rlo.append(-np.inf); rhi.append(self.cap)
        r0 += 1
        for R in nogoods:
            idx = [pos[tg] for tg in R if tg in pos]
            rows += [r0] * len(idx); cols += [nv + j for j in idx]; vals += [1.0] * len(idx)
            rlo.append(-np.inf); rhi.append(len(idx) - 1.0)
            r0 += 1
        extra = sp.csr_matrix((vals, (rows, cols)), shape=(r0, nv + nf))
        A = sp.vstack([sp.hstack([m.A, sp.csr_matrix((m.A.shape[0], nf))]), extra], format="csc")
        c = np.r_[np.zeros(nv), np.ones(nf)]
        lo, hi = np.r_[self.lo, rlo], np.r_[self.hi, rhi]
        lb, ub = np.r_[self.lb, np.zeros(nf)], np.r_[self.ub, np.ones(nf)]
        integ = np.r_[m.integ, np.ones(nf)]
        start = None
        if start_x is not None:
            u = self._u(start_x)
            nu = np.array([float(u[t, g] != v) for t, g, v in F])
            start = np.r_[start_x, nu]
        r = highs_solve(c, A, lo, hi, lb, ub, integ, time_limit, 1e-6, start=start, opts={"mip_abs_gap": 0.999})
        if r["x"] is None:
            return None, dict(status=r["status"], time=r["time"], n_nu=None, bound=r["bound"])
        x = r["x"]
        u = self._u(x[:nv])
        nu = x[nv:] > 0.5
        R = [(t, g) for j, (t, g, v) in enumerate(F) if nu[j] and u[t, g] != v]   # [choice] drop nu = 1 that changed nothing
        cost = float(m.c @ x[:nv])
        return R, dict(status=r["status"], time=r["time"], n_nu=int(nu.sum()), size=len(R), bound=r["bound"], cost=cost,
                       u=u)


# ============================================================================ Appendix A: quantile approximation of eq. (10)
def quantile_bins(pi_val, Q=20):
    """Per (t, g): quantile edges of the validation probabilities, widths, and the split index q05 (bins
    1..q05 count for the lower threshold, q05+1..Q for the upper one).
    Returns widths [T, G, Q] (floored at WIDTH_FLOOR) and is_low [T, G, Q]."""
    n, T, G = pi_val.shape
    edges = np.quantile(pi_val, np.linspace(0, 1, Q + 1), axis=0)          # [Q+1, T, G]
    widths = np.diff(edges, axis=0)                                         # [Q, T, G]
    # q05: bins above it contain only probabilities > 0.5, i.e. their lower edge exceeds 0.5
    is_low = edges[:-1] <= 0.5                                              # [Q, T, G]
    return np.maximum(widths, WIDTH_FLOOR).transpose(1, 2, 0), is_low.transpose(1, 2, 0)


class Master:
    """Master problem of Algorithm 1: maximise the approximate probability mass outside [lo_g, hi_g] (Appendix A,
    eq. 16) subject to 0 <= lo_g <= hi_g <= 1 and the accumulated OR-cuts Phi."""

    def __init__(self, pi_val, Q=20, tie_w=TIE_W):
        n, T, G = pi_val.shape
        self.T, self.G, self.Q = T, G, Q
        W, low = quantile_bins(pi_val, Q)
        self.W, self.low = W, low
        # columns: lo[G], hi[G], slo[G], shi[G], bins (one per (t, g, q)), then cut binaries
        self.nb = T * G * Q
        self.n0 = 4 * G + self.nb
        c = np.zeros(self.n0)
        c[2 * G:4 * G] = -tie_w
        wflat = W.reshape(-1)
        c[4 * G:] = -1.0 / wflat
        lb = np.zeros(self.n0)
        ub = np.r_[np.ones(2 * G), np.full(2 * G, 0.5), wflat]
        rows, cols, vals, rlo, rhi = [], [], [], [], []
        r = 0
        for g in range(G):                    # lo_g - hi_g <= 0
            rows += [r, r]; cols += [g, G + g]; vals += [1.0, -1.0]; rlo.append(-np.inf); rhi.append(0.0); r += 1
            rows += [r, r]; cols += [2 * G + g, g]; vals += [1.0, -1.0]; rlo.append(-np.inf); rhi.append(0.0); r += 1  # slo <= lo
            rows += [r, r]; cols += [3 * G + g, G + g]; vals += [1.0, 1.0]; rlo.append(-np.inf); rhi.append(1.0); r += 1  # shi <= 1 - hi
        bidx = (4 * G + np.arange(self.nb)).reshape(T, G, Q)
        for t in range(T):
            for g in range(G):
                ql = np.where(low[t, g])[0]
                qh = np.where(~low[t, g])[0]
                # lo_g - sum_q dlo_q >= 0
                rows += [r] * (1 + len(ql)); cols += [g] + list(bidx[t, g, ql]); vals += [1.0] + [-1.0] * len(ql)
                rlo.append(0.0); rhi.append(np.inf); r += 1
                # hi_g + sum_q dhi_q <= 1
                rows += [r] * (1 + len(qh)); cols += [G + g] + list(bidx[t, g, qh]); vals += [1.0] * (1 + len(qh))
                rlo.append(-np.inf); rhi.append(1.0); r += 1
        self.A0 = sp.csr_matrix((vals, (rows, cols)), shape=(r, self.n0))
        self.c0, self.lb0, self.ub0 = c, lb, ub
        self.rlo0, self.rhi0 = np.array(rlo), np.array(rhi)
        self.cuts = []                        # list of cut sets; a cut set = list of conjunctions (a_lo, b_hi)

    def add_cut(self, conj):
        """conj: list over k of (amax_lo {g: a}, bmin_hi {g: b}); the OR over k of AND_g (lo_g <= a, hi_g >= b)"""
        self.cuts.append(conj)

    def solve(self, time_limit=120.0):
        G = self.G
        nbin = sum(len(cs) for cs in self.cuts)
        nv = self.n0 + nbin
        rows, cols, vals, rlo, rhi = [], [], [], [], []
        r, j = 0, self.n0
        for cs in self.cuts:
            ys = list(range(j, j + len(cs)))
            rows += [r] * len(ys); cols += ys; vals += [1.0] * len(ys); rlo.append(1.0); rhi.append(np.inf); r += 1
            for y, (alo, bhi) in zip(ys, cs):
                for g, a in alo.items():      # lo_g + y <= a + 1
                    rows += [r, r]; cols += [g, y]; vals += [1.0, 1.0]; rlo.append(-np.inf); rhi.append(a + 1.0); r += 1
                for g, b in bhi.items():      # hi_g - y >= b - 1
                    rows += [r, r]; cols += [G + g, y]; vals += [1.0, -1.0]; rlo.append(b - 1.0); rhi.append(np.inf); r += 1
            j += len(cs)
        A = sp.vstack([sp.hstack([self.A0, sp.csr_matrix((self.A0.shape[0], nbin))]),
                       sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))], format="csc")
        c = np.r_[self.c0, np.zeros(nbin)]
        lb, ub = np.r_[self.lb0, np.zeros(nbin)], np.r_[self.ub0, np.ones(nbin)]
        integ = np.r_[np.zeros(self.n0), np.ones(nbin)]
        res = highs_solve(c, A, np.r_[self.rlo0, rlo], np.r_[self.rhi0, rhi], lb, ub, integ, time_limit, 1e-5,
                          opts={"primal_feasibility_tolerance": 1e-9, "mip_feasibility_tolerance": 1e-9})
        if res["x"] is None:
            raise RuntimeError(f"master problem: {res['status']}")
        x = res["x"]
        lo, hi = x[:G].copy(), x[G:2 * G].copy()
        mass_out = float((x[4 * G:self.n0] / self.W.reshape(-1)).sum()) / self.Q      # expected fixed decisions / instance
        return lo, hi, dict(time=res["time"], status=res["status"], obj=res["obj"], gap=res["gap"],
                            approx_fixed_share=mass_out / (self.T * self.G), n_bin=nbin)


# ============================================================================ Algorithm 1 + 2
class LtFTuner:
    """Decomposed tau-tuning (Algorithm 1) with ADD CUTS (Algorithm 2).

    m: UCModel; d: validation data dict (load, avail, u0, sr, obj, u); pi: [n, T, G] validation probabilities.
    Exact additions that only save solves: (i) check results are cached per instance and reused by monotonicity
    (a subset of a passed fixing set passes with the same witness; a superset of a failed one fails);
    (ii) the first relaxation MILP of an instance is warm-started with the instance's optimal schedule (aligned to
    the canonical labels), which releases every fixing it disagrees with and is always feasible."""

    def __init__(self, m, sysm, d, pi, eps, K_max=10, Q=20, check_tl=60.0, relax_tl=60.0, master_tl=120.0,
                 log=print, max_iter=10 ** 6, time_budget=np.inf):
        self.m, self.s, self.d, self.pi, self.eps = m, sysm, d, pi, eps
        self.K_max, self.check_tl, self.relax_tl, self.master_tl = K_max, check_tl, relax_tl, master_tl
        self.log, self.max_iter, self.time_budget = log, max_iter, time_budget
        self.n = len(pi)
        self.master = Master(pi, Q)
        self.u_lab = canonical_labels(sysm, d["u"], d["u0"])
        self.probs = {}
        self.passed = {i: [] for i in range(self.n)}       # (off, on, witness u)
        self.failed = {i: [] for i in range(self.n)}       # (off, on)
        self.stats = dict(n_check_solves=0, n_check_cached=0, t_check=0.0, n_relax=0, t_relax=0.0, n_master=0,
                          t_master=0.0, n_addcuts=0)
        self.history = []

    def prob(self, i):
        if i not in self.probs:
            sc = UCScenario(load=self.d["load"][i], avail=self.d["avail"][i], u0=self.d["u0"][i], sr=self.d["sr"][i])
            self.probs[i] = InstanceProblems(self.m, sc, self.d["obj"][i], self.eps)
        return self.probs[i]

    def check(self, i, off, on):
        for po, pn, w in self.passed[i]:
            if not (off & ~po).any() and not (on & ~pn).any():
                self.stats["n_check_cached"] += 1
                return True, dict(cached=True, u=w)
        for fo, fn in self.failed[i]:
            if not (fo & ~off).any() and not (fn & ~on).any():
                self.stats["n_check_cached"] += 1
                return False, dict(cached=True)
        ok, info = self.prob(i).check(off, on, self.check_tl)
        self.stats["n_check_solves"] += 1
        self.stats["t_check"] += info["time"]
        if ok:
            self.passed[i].append((off.copy(), on.copy(), info["u"]))
        else:
            self.failed[i].append((off.copy(), on.copy()))
        return ok, info

    def add_cuts(self, i, off, on):
        """Algorithm 2. Returns the list of conjunctions [(alo, bhi), ...] and the release sets."""
        P = self.prob(i)
        pi = self.pi[i]
        from .fixpolicy import align_to_prediction
        yhat = np.where(on, 1, np.where(off, 0, (pi > 0.5).astype(np.int8)))
        ua = align_to_prediction(self.s, self.d["u"][i], yhat, self.d["u0"][i])
        x0, c0 = P.label_start(ua)
        start = x0 if (x0 is not None and c0 <= P.cap) else None
        sets, conj, infos = [], [], []
        for k in range(self.K_max):
            st = start
            if st is not None and k > 0:       # the label start is reusable if it violates no no-good constraint
                u0 = P._u(st)
                mism = {(t, g) for t, g in zip(*np.where((off & (u0 == 1)) | (on & (u0 == 0))))}
                if any(set(R_) <= mism for R_ in sets):
                    st = None
            R, info = P.relaxation(off, on, sets, self.relax_tl, start_x=st)
            self.stats["n_relax"] += 1
            self.stats["t_relax"] += info["time"]
            infos.append({kk: v for kk, v in info.items() if kk != "u"})
            if R is None or len(R) == 0:
                break
            sets.append(R)
            alo, bhi = {}, {}
            for t, g in R:
                if off[t, g]:
                    alo[g] = min(alo.get(g, 1.0), float(pi[t, g]))
                else:
                    bhi[g] = max(bhi.get(g, 0.0), float(pi[t, g]))
            conj.append((alo, bhi))
            if info.get("u") is not None:        # the relaxation's solution is a witness for any fixing set that keeps
                ow = off.copy(); nw = on.copy()  # all fixings outside R: record it to save later checks
                for t, g in R:
                    ow[t, g] = False; nw[t, g] = False
                self.passed[i].append((ow, nw, info["u"]))
        self.stats["n_addcuts"] += 1
        return conj, sets, infos

    def run(self):
        t0 = time.time()
        it = 0
        G = self.s.G
        while True:
            lo, hi, minfo = self.master.solve(self.master_tl)
            self.stats["n_master"] += 1
            self.stats["t_master"] += minfo["time"]
            off_all, on_all = fix_masks(self.pi, lo, hi)
            share = float((off_all | on_all).mean())
            bad = None
            for i in range(self.n):
                ok, info = self.check(i, off_all[i], on_all[i])
                if not ok:
                    bad = (i, info)
                    break
            rec = dict(it=it, master_s=minfo["time"], master_status=minfo["status"], master_gap=minfo["gap"],
                       n_cut_sets=len(self.master.cuts), n_bin=minfo["n_bin"], approx_fixed=minfo["approx_fixed_share"],
                       val_fixed=share, wall=time.time() - t0)
            if bad is None:
                rec["result"] = "all instances feasible"
                self.history.append(rec)
                self.log(f"[it {it}] master {minfo['time']:.1f}s val fixed {share * 100:.2f}% -> ALL {self.n} PASS "
                         f"(wall {time.time() - t0:.0f}s)")
                break
            i, info = bad
            conj, sets, infos = self.add_cuts(i, off_all[i], on_all[i])
            self.master.add_cut(conj)
            rec.update(failed=i, check_status=info.get("status", "cached"), check_obj=info.get("obj"),
                       n_fixed=int(off_all[i].sum() + on_all[i].sum()), release_sizes=[len(s) for s in sets],
                       relax=infos)
            self.history.append(rec)
            self.log(f"[it {it}] master {minfo['time']:.1f}s ({minfo['n_bin']} bin) val fixed {share * 100:.2f}% | "
                     f"inst {i} fails ({info.get('status', 'cached')}) -> {len(sets)} release sets, sizes "
                     f"{[len(s) for s in sets]}, relax {sum(x['time'] for x in infos):.0f}s | checks "
                     f"{self.stats['n_check_solves']} solved / {self.stats['n_check_cached']} cached | wall {time.time() - t0:.0f}s")
            if not sets:
                self.log(f"  WARNING: no release set found for instance {i}; stopping")
                break
            it += 1
            if it >= self.max_iter or time.time() - t0 > self.time_budget:
                lo, hi, minfo = self.master.solve(self.master_tl)
                off_all, on_all = fix_masks(self.pi, lo, hi)
                fails = [i for i in range(self.n) if not self.check(i, off_all[i], on_all[i])[0]]
                self.history.append(dict(it=it, result="budget stop", val_fixed=float((off_all | on_all).mean()),
                                         n_fail_final=len(fails), fails=fails, wall=time.time() - t0))
                self.log(f"  stopped: iteration / time budget; final thresholds violate {len(fails)} of {self.n} instances")
                break
        self.lo, self.hi = lo, hi
        self.stats["wall_s"] = time.time() - t0
        return lo, hi

    def verify(self, lo, hi):
        """Independent check of (8e) with the returned thresholds: for every validation instance, a stored witness
        commitment that respects all fixings is priced by the exact dispatch LP (its min up/down rows included).
        Returns per-instance relative cost increase over C* (inf if no witness)."""
        off_all, on_all = fix_masks(self.pi, lo, hi)
        out = []
        for i in range(self.n):
            off, on = off_all[i], on_all[i]
            w = None
            for po, pn, u in self.passed[i]:
                if not (off & ~po).any() and not (on & ~pn).any() and u is not None:
                    w = u
                    break
            if w is None or (w[off] != 0).any() or (w[on] != 1).any():
                out.append(np.inf)
                continue
            P = self.prob(i)
            c = self.m.solve_dispatch(P.sc, w).obj
            out.append((c - P.c_star) / P.c_star)
        return np.array(out)
