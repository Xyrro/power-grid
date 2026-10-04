"""DC-OPF (LP, fixed topology) and DC-OTS (MILP, big-M) solved with HiGHS.

DC-OTS (Fisher, O'Neill & Ferris 2008):

    min  sum_g c_g p_g
    s.t. Cg p - A^T f = pd                               (KCL)
         |f_l - b_l (theta_i - theta_j)| <= M_l (1 - z_l)  (Ohm's law, relaxed when open)
         |f_l| <= fmax_l z_l                              (thermal + angle-difference limit)
         pmin <= p <= pmax,  theta_ref = 0,  |theta| <= THETA
         sum_l (1 - z_l) <= K                             (switching budget, optional)
         z_l = 1 for lines that may not be switched        (optional)

Generator costs are linear (PGLib) or piecewise-linearised quadratics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .case import DCCase

THETA = np.pi / 3  # bound on |theta_i| (rad); big-M for line l is M_l = 2*THETA*b_l


def disjoint_path_bound(case: DCCase, K: int, fixed_closed=None) -> np.ndarray:
    """Valid bound on |theta_i - theta_j| across an OPEN line l when at most K lines are open.

    Any closed line k forces |theta_u - theta_v| = |f_k|/b_k <= fmax_k/b_k. If K edge-disjoint
    i-j paths avoiding l exist, at most K-1 of them can contain another open line, so one path is
    fully closed and bounds the angle difference by its length. Lines that can never be switched
    (fixed_closed) may be shared by the paths (they can't be opened). Returns the max path length
    among a min-cost set of K disjoint paths (inf when fewer exist).
    """
    import networkx as nx

    w = case.fmax / case.b                       # angle needed to saturate each line
    fixed = np.zeros(case.n_line, bool) if fixed_closed is None else fixed_closed
    bound = np.full(case.n_line, np.inf)
    scale = 1e6
    for l in range(case.n_line):
        if fixed[l]:
            continue
        g = nx.DiGraph()
        for k in range(case.n_line):
            if k == l:
                continue
            u, v = int(case.f_bus[k]), int(case.t_bus[k])
            cap = K if fixed[k] else 1
            # model each (possibly parallel) line as its own node pair to keep edge-disjointness exact
            mid = ("e", k)
            for a, b_ in ((u, v), (v, u)):
                g.add_edge(a, (mid, a), capacity=cap, weight=int(w[k] * scale))
                g.add_edge((mid, a), b_, capacity=cap, weight=0)
        src, dst = int(case.f_bus[l]), int(case.t_bus[l])
        g.add_node(src, demand=-K)
        g.add_node(dst, demand=K)
        try:
            flow = nx.min_cost_flow(g)
        except nx.NetworkXUnfeasible:
            continue
        # decompose the unit flows into K paths and take the longest
        flow = {u: {v: f for v, f in d.items() if f > 0} for u, d in flow.items()}
        longest = 0.0
        for _ in range(K):
            node, length, seen = src, 0.0, 0
            while node != dst and seen < 10 * case.n_bus:
                v = next(iter(flow[node]))
                flow[node][v] -= 1
                if flow[node][v] == 0:
                    del flow[node][v]
                if isinstance(v, tuple):
                    length += w[v[0][1]]
                node, seen = v, seen + 1
            longest = max(longest, length)
        bound[l] = longest
    return bound


@dataclass
class Solution:
    status: str          # "optimal", "infeasible", "time_limit", ...
    obj: float
    pg: np.ndarray       # [G]
    va: np.ndarray       # [N]
    flow: np.ndarray     # [L]
    z: np.ndarray        # [L]
    lmp: np.ndarray | None = None   # [N] KCL duals (LP only)
    mu: np.ndarray | None = None    # [L] flow-limit duals (LP only), >= 0 when binding
    gap: float = 0.0
    time: float = 0.0
    nodes: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "time_limit_feasible")


def _pwl(case: DCCase, n_seg: int):
    """Segment widths and slopes for each generator: p = pmin + sum_s delta_s."""
    G = case.n_gen
    S = n_seg if np.any(case.cost_q > 0) else 1
    width = np.zeros((G, S))
    slope = np.zeros((G, S))
    for g in range(G):
        bp = np.linspace(case.pmin[g], case.pmax[g], S + 1)
        c = case.cost_q[g] * bp ** 2 + case.cost[g] * bp
        width[g] = np.diff(bp)
        slope[g] = np.diff(c) / np.maximum(width[g], 1e-12)
    const = case.cost_q * case.pmin ** 2 + case.cost * case.pmin
    return S, width, slope, const


class DCModel:
    """Pre-builds the sparse structure for a case so that each solve only changes the RHS."""

    def __init__(self, case: DCCase, budget: int | None = None, fixed_closed: np.ndarray | None = None,
                 n_seg: int = 4, theta_max: float = THETA, switch_cost: float = 0.0):
        self.case = case
        self.theta_max = theta_max
        self.switch_cost = switch_cost   # $/h charged per opened line (canonicalises degenerate optima)
        self.budget = budget
        self.fixed_closed = np.zeros(case.n_line, bool) if fixed_closed is None else fixed_closed
        N, G, L = case.n_bus, case.n_gen, case.n_line
        self.S, self.width, self.slope, self.const = _pwl(case, n_seg)
        S = self.S
        self.nd = G * S
        A = sp.csr_matrix(case.A)
        self.A_sp = A
        # generator segment -> bus incidence  [N, G*S]
        rows = np.repeat(case.gen_bus, S)
        self.Cseg = sp.csr_matrix((np.ones(G * S), (rows, np.arange(G * S))), shape=(N, G * S))
        self.pmin_bus = case.Cg @ case.pmin
        self.bA = sp.diags(case.b) @ A          # [L, N]
        self.M = 2 * theta_max * case.b
        if budget is not None:
            self.M = np.minimum(self.M, case.b * disjoint_path_bound(case, budget, self.fixed_closed))
        self.c_seg = self.slope.reshape(-1)

    # ------------------------------------------------------------------ utils
    def pg_from_delta(self, delta):
        return self.case.pmin + delta.reshape(self.case.n_gen, self.S).sum(1)

    def cost(self, pg):
        """True (non-PWL) cost of a dispatch, [..., G] -> [...]"""
        return (pg * self.case.cost).sum(-1) + (pg ** 2 * self.case.cost_q).sum(-1)

    def _theta_bounds(self):
        lo = np.full(self.case.n_bus, -self.theta_max)
        hi = np.full(self.case.n_bus, self.theta_max)
        lo[self.case.ref] = hi[self.case.ref] = 0.0
        return lo, hi

    # ---------------------------------------------------------------- DC-OTS
    def solve_ots(self, pd, time_limit: float = 60.0, mip_gap: float = 1e-4, z_fix: dict | None = None,
                  z_start=None, nogood: list | None = None) -> Solution:
        """Solve the DC-OTS MILP for load vector pd [N].

        z_fix: {line: 0/1} partial fixing (used by ML-guided "neural diving").
        nogood: list of 0/1 vectors to exclude (to enumerate alternative optima).
        """
        case = self.case
        N, L = case.n_bus, case.n_line
        nd = self.nd
        nv = nd + N + 2 * L
        iD, iT, iF, iZ = 0, nd, nd + N, nd + N + L
        I_L = sp.identity(L, format="csr")
        Z = lambda r, c: sp.csr_matrix((r, c))
        # KCL: Cseg d - A^T f = pd - Cg pmin
        kcl = sp.hstack([self.Cseg, Z(N, N), -self.A_sp.T, Z(N, L)])
        ohm_u = sp.hstack([Z(L, nd), -self.bA, I_L, sp.diags(self.M)])
        ohm_l = sp.hstack([Z(L, nd), self.bA, -I_L, sp.diags(self.M)])
        fl_u = sp.hstack([Z(L, nd), Z(L, N), I_L, -sp.diags(case.fmax)])
        fl_l = sp.hstack([Z(L, nd), Z(L, N), -I_L, -sp.diags(case.fmax)])
        blocks = [kcl, ohm_u, ohm_l, fl_u, fl_l]
        rhs_eq = pd - self.pmin_bus
        lo = [rhs_eq, np.full(2 * L, -np.inf), np.full(2 * L, -np.inf)]
        hi = [rhs_eq, np.r_[self.M, self.M], np.zeros(2 * L)]
        if self.budget is not None:
            row = sp.csr_matrix(np.r_[np.zeros(nd + N + L), -np.ones(L)][None, :])
            blocks.append(row)
            lo.append([-np.inf])
            hi.append([self.budget - L])
        for ng in nogood or []:
            ng = np.asarray(ng)
            # sum_{l: ng=1} (1 - z_l) + sum_{l: ng=0} z_l >= 1
            coef = np.where(ng > 0.5, -1.0, 1.0)
            row = sp.csr_matrix(np.r_[np.zeros(nd + N + L), coef][None, :])
            blocks.append(row)
            lo.append([1.0 - ng.sum()])
            hi.append([np.inf])
        Amat = sp.vstack(blocks, format="csr")
        cons = LinearConstraint(Amat, np.concatenate(lo), np.concatenate(hi))
        tlo, thi = self._theta_bounds()
        zlo, zhi = np.zeros(L), np.ones(L)
        zlo[self.fixed_closed] = 1.0
        for l, v in (z_fix or {}).items():
            zlo[l] = zhi[l] = v
        lb = np.r_[np.zeros(nd), tlo, -case.fmax, zlo]
        ub = np.r_[self.width.reshape(-1), thi, case.fmax, zhi]
        c = np.r_[self.c_seg, np.zeros(N + L), np.full(L, -self.switch_cost)]
        integ = np.r_[np.zeros(nd + N + L), np.ones(L)]
        opts = {"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False}
        t0 = time.time()
        res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub), options=opts)
        dt = time.time() - t0
        if res.x is None:
            return Solution("infeasible" if res.status == 2 else f"fail{res.status}", np.inf,
                            None, None, None, None, time=dt)
        x = res.x
        status = "optimal" if res.status == 0 else ("time_limit_feasible" if res.status == 1 else f"s{res.status}")
        pg = self.pg_from_delta(x[iD:iT])
        z = np.round(x[iZ:]).astype(np.int8)
        gap = getattr(res, "mip_gap", 0.0) or 0.0
        # obj = generation cost of the returned dispatch (switching cost reported via z)
        return Solution(status, float(self.cost(pg)), pg, x[iT:iF], x[iF:iZ], z,
                        gap=gap, time=dt, nodes=int(getattr(res, "mip_node_count", 0) or 0))

    # ---------------------------------------------------------------- DC-OPF
    def solve_lp(self, pd, z=None, shed_penalty: float | None = None) -> Solution:
        """DC-OPF with fixed topology z (default: all lines closed).

        With ``shed_penalty`` the LP gets load-shedding / spillage slacks so it is
        always feasible; the slack total is a measure of how infeasible z is.
        """
        case = self.case
        N, L = case.n_bus, case.n_line
        z = np.ones(L, dtype=np.int8) if z is None else np.asarray(z).round().astype(np.int8)
        nd = self.nd
        closed = np.where(z > 0)[0]
        nc = len(closed)
        # variables [delta(nd), theta(N), f(L), (shed(N), spill(N))]
        soft = shed_penalty is not None
        ns = 2 * N if soft else 0
        kcl = sp.hstack([self.Cseg, sp.csr_matrix((N, N)), -self.A_sp.T] +
                        ([sp.identity(N), -sp.identity(N)] if soft else []))
        ohm = sp.hstack([sp.csr_matrix((nc, nd)), -self.bA[closed], sp.identity(L, format="csr")[closed]] +
                        ([sp.csr_matrix((nc, ns))] if soft else []))
        A_eq = sp.vstack([kcl, ohm], format="csr")
        b_eq = np.r_[pd - self.pmin_bus, np.zeros(nc)]
        tlo, thi = self._theta_bounds()
        fcap = case.fmax * z
        bounds = np.c_[np.r_[np.zeros(nd), tlo, -fcap, np.zeros(ns)],
                       np.r_[self.width.reshape(-1), thi, fcap, np.full(ns, np.inf)]]
        c = np.r_[self.c_seg, np.zeros(N + L), np.full(ns, shed_penalty or 0.0)]
        t0 = time.time()
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        dt = time.time() - t0
        if res.status != 0:
            return Solution("infeasible", np.inf, None, None, None, z, time=dt)
        x = res.x
        pg = self.pg_from_delta(x[:nd])
        lmp = res.eqlin.marginals[:N]
        # Ohm's-law duals gamma_l = dC/d(rhs of f_l - b_l A_l theta = 0); first-order estimate of the cost
        # change from opening closed line l (b_l -> 0) is  dC_l ~= -gamma_l * f_l  (cf. Fuller et al. 2012)
        gamma = np.zeros(L)
        gamma[closed] = res.eqlin.marginals[N:]
        # flow-limit duals live on the variable bounds of f
        mu = res.upper.marginals[nd + N:nd + N + L] - res.lower.marginals[nd + N:nd + N + L]
        sol = Solution("optimal", float(self.cost(pg)), pg, x[nd:nd + N], x[nd + N:nd + N + L], z,
                       lmp=lmp, mu=-mu, time=dt)
        sol.gamma = gamma
        if soft:
            sol.shed = float(x[nd + N + L:].sum())
            sol.obj_noslack = float(self.cost(pg))
        return sol
