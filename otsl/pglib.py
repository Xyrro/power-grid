"""PGLib-UC (Knueven, Ostrowski & Watson 2018) unit commitment as a matrix MILP for HiGHS.

Formulation: MODEL.pdf / uc_model.py of github.com/power-grid-lib/pglib-uc (equations (1)-(24) there):

    min  sum_g sum_t [ CP_g^1 u + sum_l (CP_g^l - CP_g^1) lambda^l + sum_s CS_g^s delta^s ]
         + VOLL (shed + spill) + RES_PEN shortfall                            (soft balance, see below)
    s.t. (2)  sum_g (p + Pmin u) + sum_w p_w + shed - spill = D(t)
         (3)  sum_g r + shortfall >= R(t)
         (4)-(5) initial up / down requirement, (6) initial logic, (7) start-up categories at t = 1,
         (8)-(10) initial ramp-up / ramp-down / shut-down capability
         (11) must-run, (12) logic, (13)-(14) min up / down, (15)-(16) off-time-dependent start-up categories,
         (17)-(18) start-up / shut-down capability, (19)-(20) ramping (reserve counts against ramp-up),
         (21)-(23) piecewise production cost (convex combination of the curve's points), (24) renewables.

Deviations / implementation choices (none changes the optimum of a feasible instance):

* Must-run units (U_g = 1) have no binary variables: u = 1 is substituted, so their v = w = delta = 0 and their
  no-load cost CP^1 is a constant, carried by one column fixed at 1 ("one"). Only the free units' u, v, w, delta
  are variables. All learning / fixing code sees the free units only (PGSystem.G = number of free units).
* delta (start-up category) is continuous in [0, 1]: for integral (u, v, w) the per-(g, t) constraints
  sum_s delta^s = v, delta^s <= (window sum of w), 0 <= delta <= 1 form an interval matrix, so delta is integral at
  every vertex - the MILP optimum is the reference one. Checked against delta binary (docs/methods/pglib.md).
* Ramp rows that cannot bind (RU, RD >= Pmax - Pmin) are omitted (implied by (17) and p >= 0).
* Soft balance: load shedding / over-generation (VOLL) and reserve shortfall (RES_PEN) are priced slacks, as in
  otsl.uc.UCModel, so that every commitment that satisfies min up/down has a feasible dispatch (needed by the
  fixed-commitment dispatch LP used as critic and for label-free training). At the optimum of every instance
  checked they are zero, i.e. the reference (hard) model has the same optimum.

The class exposes the interface of otsl.uc.UCModel that the reused methods rely on (A, c, integ, nv, T, off,
ix, dims["G"], _rhs_bounds(sc, u_fix), solve_dispatch(sc, u)), so otsl.fixpolicy / otsl.ltfx / otsl.combo work on it.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

PGLIB = "/home/user/power-grid-lib/pglib-uc"
VOLL = 10_000.0          # $/MWh of shed load or over-generation
RES_PEN = 5_000.0        # $/MWh of spinning-reserve shortfall


# ============================================================================ data
@dataclass
class PGSystem:
    """Thermal fleet of one PGLib-UC case. Arrays over ALL units carry the suffix _all; the unsuffixed
    UCSystem-compatible attributes (G, pmax, pmin, seg_w, seg_c, c_nl, c_su, min_up, min_dn, ramp, utype, gbus,
    identical_groups) describe the FREE (not must-run) units only, in the order of the model's u block."""
    names_all: np.ndarray
    must_run_all: np.ndarray
    pmax_all: np.ndarray
    pmin_all: np.ndarray
    ru_all: np.ndarray
    rd_all: np.ndarray
    su_all: np.ndarray
    sd_all: np.ndarray
    ut_all: np.ndarray
    dt_all: np.ndarray
    pw_mw: list                      # per unit: array of curve points (MW)
    pw_cost: list                    # per unit: array of curve costs ($/h)
    st_cost: list                    # per unit: start-up category costs (hot -> cold)
    st_lag: list                     # per unit: start-up category lags (h)
    u0_all: np.ndarray               # initial status
    up0_all: np.ndarray              # hours up before t = 1
    dn0_all: np.ndarray              # hours down before t = 1
    p0_all: np.ndarray               # output before t = 1 (MW)
    ren_names: list = field(default_factory=list)
    free: np.ndarray = None          # indices (into _all) of the free units

    def __post_init__(self):
        self.free = np.where(self.must_run_all == 0)[0]
        self.mr = np.where(self.must_run_all == 1)[0]
        F = self.free
        self.pmax, self.pmin = self.pmax_all[F], self.pmin_all[F]
        self.min_up = np.maximum(1, self.ut_all[F]).astype(int)
        self.min_dn = np.maximum(1, self.dt_all[F]).astype(int)
        self.ramp = self.ru_all[F]
        S = max(len(self.pw_mw[g]) - 1 for g in F)
        S = max(S, 1)
        self.seg_w, self.seg_c = np.zeros((len(F), S)), np.zeros((len(F), S))
        for k, g in enumerate(F):
            mw, c = self.pw_mw[g], self.pw_cost[g]
            for j in range(len(mw) - 1):
                w = mw[j + 1] - mw[j]
                self.seg_w[k, j] = w
                self.seg_c[k, j] = (c[j + 1] - c[j]) / w if w > 1e-12 else 0.0
        self.c_nl = np.array([self.pw_cost[g][0] for g in F])
        self.c_su = np.array([self.st_cost[g][-1] for g in F])          # cold start (upper bound of a start)
        self.c_su_hot = np.array([self.st_cost[g][0] for g in F])
        self.utype = np.array(["CT"] * len(F))                          # PGLib has no unit types
        self.gbus = np.zeros(len(F), int)
        self.n_bus = 1
        self.u0 = self.u0_all[F].astype(np.int8)

    @property
    def G(self):
        return len(self.free)

    @property
    def G_all(self):
        return len(self.pmax_all)

    def identical_groups(self):
        key = {}
        for k, g in enumerate(self.free):
            kk = (round(float(self.pmax_all[g]), 6), round(float(self.pmin_all[g]), 6),
                  tuple(np.round(self.pw_mw[g], 6)), tuple(np.round(self.pw_cost[g], 6)),
                  tuple(np.round(self.st_cost[g], 6)), tuple(self.st_lag[g]), int(self.ut_all[g]), int(self.dt_all[g]),
                  round(float(self.ru_all[g]), 6), round(float(self.rd_all[g]), 6), round(float(self.su_all[g]), 6),
                  round(float(self.sd_all[g]), 6), int(self.u0_all[g]), int(self.up0_all[g]), int(self.dn0_all[g]),
                  round(float(self.p0_all[g]), 6))
            key.setdefault(kk, []).append(k)
        return [v for v in key.values() if len(v) > 1]

    def avg_cost(self):
        """$/MWh at full output of each free unit (merit order)"""
        return (self.c_nl + (self.seg_c * self.seg_w).sum(1)) / np.maximum(self.pmax, 1e-9)


@dataclass
class PGScenario:
    """One instance. load [T, 1] (system demand), avail / avail_min [T, W] (renewable max / min), sr [T] reserve,
    u0 [G free] initial status (the full initial condition is the system's)."""
    load: np.ndarray
    avail: np.ndarray
    sr: np.ndarray
    u0: np.ndarray
    avail_min: np.ndarray = None
    name: str = ""


def load_case(path, T=None):
    """PGLib-UC json -> (PGSystem, PGScenario, raw dict). T < time_periods truncates the horizon."""
    d = json.load(open(path))
    th = d["thermal_generators"]
    names = sorted(th.keys())
    gs = [th[k] for k in names]
    a = lambda k: np.array([g[k] for g in gs], float)
    sysm = PGSystem(
        names_all=np.array(names), must_run_all=a("must_run").astype(int), pmax_all=a("power_output_maximum"),
        pmin_all=a("power_output_minimum"), ru_all=a("ramp_up_limit"), rd_all=a("ramp_down_limit"),
        su_all=a("ramp_startup_limit"), sd_all=a("ramp_shutdown_limit"), ut_all=a("time_up_minimum").astype(int),
        dt_all=a("time_down_minimum").astype(int),
        pw_mw=[np.array([p["mw"] for p in g["piecewise_production"]], float) for g in gs],
        pw_cost=[np.array([p["cost"] for p in g["piecewise_production"]], float) for g in gs],
        st_cost=[np.array([s["cost"] for s in g["startup"]], float) for g in gs],
        st_lag=[[int(s["lag"]) for s in g["startup"]] for g in gs],
        u0_all=a("unit_on_t0").astype(int), up0_all=a("time_up_t0").astype(int), dn0_all=a("time_down_t0").astype(int),
        p0_all=a("power_output_t0"), ren_names=sorted(d["renewable_generators"].keys()))
    TT = int(d["time_periods"]) if T is None else int(T)
    rn = d["renewable_generators"]
    W = len(sysm.ren_names)
    av = np.zeros((TT, W)); avm = np.zeros((TT, W))
    for j, k in enumerate(sysm.ren_names):
        av[:, j] = np.array(rn[k]["power_output_maximum"][:TT])
        avm[:, j] = np.array(rn[k]["power_output_minimum"][:TT])
    sc = PGScenario(load=np.array(d["demand"][:TT], float)[:, None], avail=av, avail_min=avm,
                    sr=np.array(d["reserves"][:TT], float), u0=sysm.u0.copy(), name=os.path.basename(path))
    return sysm, sc, d


def ca_files():
    return sorted(glob.glob(os.path.join(PGLIB, "ca", "*.json")))


def scenario_from(d, i):
    return PGScenario(load=d["load"][i], avail=d["avail"][i], avail_min=d["avail_min"][i], sr=d["sr"][i],
                      u0=d["u0"][i])


# ============================================================================ HiGHS
def highs_run(c, A, lo, hi, lb, ub, integ=None, time_limit=600.0, gap=1e-3, threads=1, incumbents=False,
              duals=False, opts=None, start=None):
    """min c x s.t. lo <= A x <= hi, lb <= x <= ub; integ: 0/1 mask. Returns dict(status, x, obj, bound, gap, time,
    inc [(s, obj, bound)], row_dual, nodes)."""
    import highspy
    A = sp.csc_matrix(A)
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(c, float)
    lp.col_lower_ = np.where(np.isfinite(lb), lb, -inf)
    lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    mip = integ is not None and np.any(integ)
    if mip:
        K, C = highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous
        lp.integrality_ = [K if v else C for v in integ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("time_limit", float(time_limit))
    if mip:
        h.setOptionValue("mip_rel_gap", float(gap))
    for k, v in (opts or {}).items():
        h.setOptionValue(k, v)
    h.passModel(lp)
    if start is not None:
        s = highspy.HighsSolution()
        s.col_value = list(np.asarray(start, float))
        s.value_valid = True
        h.setSolution(s)
    inc = []
    if mip and incumbents:
        def cb(e):
            o = e.data_out
            inc.append((float(o.running_time), float(o.objective_function_value), float(o.mip_dual_bound)))
        h.cbMipImprovingSolution.subscribe(cb)
    t0 = time.time()
    h.run()
    dt = time.time() - t0
    info = h.getInfo()
    st = h.modelStatusToString(h.getModelStatus())
    feas = info.primal_solution_status == 2
    sol = h.getSolution()
    out = dict(status=st, x=np.array(sol.col_value) if feas else None,
               obj=float(info.objective_function_value) if feas else np.inf, time=dt, inc=inc,
               bound=float(info.mip_dual_bound) if mip else (float(info.objective_function_value) if feas else np.nan),
               gap=float(info.mip_gap) if mip else 0.0, nodes=int(info.mip_node_count) if mip else 0)
    if duals and feas and not mip:
        out["row_dual"] = np.array(sol.row_dual)
    return out


# ============================================================================ model
class PGModel:
    """Matrix form of the PGLib-UC MILP for horizon T (constraint matrix built once; scenarios change bounds /
    right-hand sides only)."""

    def __init__(self, sysm: PGSystem, T=48, delta_binary=False):
        self.s, self.T = sysm, T
        self.delta_binary = delta_binary
        s = sysm
        G, GA, W = s.G, s.G_all, len(s.ren_names)
        self.fpos = -np.ones(GA, int)
        self.fpos[s.free] = np.arange(G)
        self.nL = np.array([len(m) for m in s.pw_mw])                  # curve points per unit (all units)
        self.nS = np.array([len(s.st_cost[g]) for g in s.free])         # start-up categories (free units)
        self.lam0 = np.r_[0, np.cumsum(self.nL)[:-1]]
        self.dl0 = np.r_[0, np.cumsum(self.nS)[:-1]]
        NL, NS = int(self.nL.sum()), int(self.nS.sum())
        self.dims = dict(G=G, GA=GA, W=W, NL=NL, NS=NS, S=1, N=1, L=0, R=W)
        sizes = [("u", G), ("v", G), ("w", G), ("dl", NS), ("p", GA), ("res", GA), ("lam", NL), ("r", W),
                 ("shed", 1), ("spill", 1), ("short", 1)]
        self.off, o = {}, 0
        for name, k in sizes:
            self.off[name] = (o, k)
            o += k * T
        self.col_one = o
        self.nv = o + 1
        self._build()

    def ix(self, name, t, j=0):
        o, k = self.off[name]
        return o + t * k + j

    def blk(self, name):
        """column indices [T, k] of a block"""
        o, k = self.off[name]
        return o + np.arange(self.T * k).reshape(self.T, k)

    def _build(self):
        s, T = self.s, self.T
        G, GA = s.G, s.G_all
        U, V, Wc, P, R = (self.blk(n) for n in ("u", "v", "w", "p", "res"))
        DL, LAM = self.blk("dl"), self.blk("lam")
        rows, cols, vals, lo, hi = [], [], [], [], []
        self.nr = 0
        self.tags = {}

        def add(rc, vv, l, h, tag=None):
            """rc: list of column-index arrays (one per term, each [n]); vv: coefficients (scalars or [n])."""
            n = len(np.atleast_1d(l)) if np.ndim(l) else len(rc[0])
            r = self.nr + np.arange(n)
            for ci, cv in zip(rc, vv):
                ci = np.asarray(ci).reshape(-1)
                rows.append(r); cols.append(ci); vals.append(np.broadcast_to(np.asarray(cv, float), ci.shape).copy())
            lo.append(np.broadcast_to(np.asarray(l, float), (n,)).copy())
            hi.append(np.broadcast_to(np.asarray(h, float), (n,)).copy())
            if tag:
                self.tags[tag] = r
            self.nr += n
            return r

        F = s.free
        u0f = s.u0_all[F]
        # (6) / (12) logic
        add([U[0], V[0], Wc[0]], [1, -1, 1], np.zeros(G), np.zeros(G), tag="logic0")
        if T > 1:
            add([U[1:], U[:-1], V[1:], Wc[1:]], [1, -1, -1, 1], np.zeros((T - 1) * G), np.zeros((T - 1) * G))
        # (13) / (14) min up / down (free units), per unit since windows differ
        for k in range(G):
            UT, DT = min(int(s.min_up[k]), T), min(int(s.min_dn[k]), T)
            ts = np.arange(UT - 1, T)                        # rows for t = UT-1 .. T-1 (0-indexed)
            if len(ts):
                terms = [V[ts - j, k] for j in range(UT)] + [U[ts, k]]
                add(terms, [1.0] * UT + [-1.0], np.full(len(ts), -np.inf), np.zeros(len(ts)))
            ts = np.arange(DT - 1, T)
            if len(ts):
                terms = [Wc[ts - j, k] for j in range(DT)] + [U[ts, k]]
                add(terms, [1.0] * DT + [1.0], np.full(len(ts), -np.inf), np.ones(len(ts)))
        # (15) start-up category windows, (16) v = sum delta
        for k, g in enumerate(F):
            lags = s.st_lag[g]
            nS = len(lags)
            for si in range(nS - 1):
                a, b = lags[si], lags[si + 1]
                ts = np.arange(b - 1, T)                     # 1-indexed t >= TS^{s+1}
                if len(ts) == 0:
                    continue
                terms = [DL[ts, self.dl0[k] + si]] + [Wc[ts - i, k] for i in range(a, b)]
                add(terms, [1.0] + [-1.0] * (b - a), np.full(len(ts), -np.inf), np.zeros(len(ts)))
            terms = [V[:, k]] + [DL[:, self.dl0[k] + si] for si in range(nS)]
            add(terms, [1.0] + [-1.0] * nS, np.zeros(T), np.zeros(T))
        # (10) initial shut-down capability: max(Pmax - SD, 0) w(1) <= (Pmax - Pmin) U0 - U0 (P0 - Pmin)
        csd = np.maximum(s.pmax_all[F] - s.sd_all[F], 0)
        rhs10 = (s.pmax_all[F] - s.pmin_all[F]) * u0f - u0f * (s.p0_all[F] - s.pmin_all[F])
        add([Wc[0]], [csd], np.full(G, -np.inf), rhs10, tag="init_sd")
        # (17) / (18) capacity with start-up / shut-down capability (all units; u = 1 constant for must-run)
        rng_all = s.pmax_all - s.pmin_all
        csu_all = np.maximum(s.pmax_all - s.su_all, 0)
        csd_all = np.maximum(s.pmax_all - s.sd_all, 0)
        fr = F
        add([P[:, fr], R[:, fr], U, V], [1.0, 1.0, np.tile(-rng_all[fr], T), np.tile(csu_all[fr], T)],
            np.full(T * G, -np.inf), np.zeros(T * G))
        if T > 1:
            add([P[:-1, fr], R[:-1, fr], U[:-1], Wc[1:]], [1.0, 1.0, np.tile(-rng_all[fr], T - 1), np.tile(csd_all[fr], T - 1)],
                np.full((T - 1) * G, -np.inf), np.zeros((T - 1) * G))
        mr = s.mr
        if len(mr):
            add([P[:, mr], R[:, mr]], [1.0, 1.0], np.full(T * len(mr), -np.inf), np.tile(rng_all[mr], T))
        # (19)/(20) ramping, (8)/(9) at t = 1; omitted where they cannot bind
        bu = np.where(s.ru_all < rng_all - 1e-9)[0]
        bd = np.where(s.rd_all < rng_all - 1e-9)[0]
        init_part = s.u0_all * (s.p0_all - s.pmin_all)
        if len(bu):
            add([P[0, bu], R[0, bu]], [1.0, 1.0], np.full(len(bu), -np.inf), s.ru_all[bu] + init_part[bu], tag="ramp_up0")
            if T > 1:
                add([P[1:, bu], R[1:, bu], P[:-1, bu]], [1.0, 1.0, -1.0], np.full((T - 1) * len(bu), -np.inf),
                    np.tile(s.ru_all[bu], T - 1))
        need_rd0 = np.where(s.rd_all - init_part < -1e-9)[0]
        if len(need_rd0):
            add([P[0, need_rd0]], [-1.0], np.full(len(need_rd0), -np.inf), s.rd_all[need_rd0] - init_part[need_rd0])
        if len(bd) and T > 1:
            add([P[:-1, bd], P[1:, bd]], [1.0, -1.0], np.full((T - 1) * len(bd), -np.inf), np.tile(s.rd_all[bd], T - 1))
        # (21) p = sum_l (P^l - P^1) lambda^l ; (23) sum_l lambda^l = u (1 for must-run)
        for g in range(GA):
            mw = s.pw_mw[g]
            terms = [P[:, g]] + [LAM[:, self.lam0[g] + l] for l in range(len(mw))]
            add(terms, [1.0] + [-(mw[l] - mw[0]) for l in range(len(mw))], np.zeros(T), np.zeros(T))
            terms = [LAM[:, self.lam0[g] + l] for l in range(len(mw))]
            if self.fpos[g] >= 0:
                add(terms + [U[:, self.fpos[g]]], [1.0] * len(mw) + [-1.0], np.zeros(T), np.zeros(T))
            else:
                add(terms, [1.0] * len(mw), np.ones(T), np.ones(T))
        # (2) demand (rhs filled per scenario), (3) reserve
        Rw = self.blk("r")
        terms = [P[:, g] for g in range(GA)] + [U[:, k] for k in range(G)] + [Rw[:, j] for j in range(Rw.shape[1])]
        terms += [self.blk("shed")[:, 0], self.blk("spill")[:, 0]]
        coefs = [1.0] * GA + list(s.pmin_all[F]) + [1.0] * Rw.shape[1] + [1.0, -1.0]
        add(terms, coefs, np.zeros(T), np.zeros(T), tag="demand")
        terms = [R[:, g] for g in range(GA)] + [self.blk("short")[:, 0]]
        add(terms, [1.0] * (GA + 1), np.zeros(T), np.full(T, np.inf), tag="reserve")
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(self.nr, self.nv))
        self.lo0, self.hi0 = np.concatenate(lo), np.concatenate(hi)
        # objective
        c = np.zeros(self.nv)
        c[U.reshape(-1)] = np.tile(s.c_nl, T)
        for k, g in enumerate(F):
            for si in range(len(s.st_cost[g])):
                c[DL[:, self.dl0[k] + si]] = s.st_cost[g][si]
        for g in range(GA):
            cc = s.pw_cost[g]
            for l in range(len(cc)):
                c[LAM[:, self.lam0[g] + l]] = cc[l] - cc[0]
        c[self.blk("shed")] = VOLL
        c[self.blk("spill")] = VOLL
        c[self.blk("short")] = RES_PEN
        c[self.col_one] = T * sum(s.pw_cost[g][0] for g in s.mr)
        self.c = c
        self.integ = np.zeros(self.nv)
        for name in ("u", "v", "w") + (("dl",) if self.delta_binary else ()):
            self.integ[self.blk(name).reshape(-1)] = 1
        self.mr_pmin = float(s.pmin_all[s.mr].sum())

    # ------------------------------------------------------------------------ bounds per scenario
    def _rhs_bounds(self, sc, u_fix=None):
        s, T = self.s, self.T
        G = s.G
        F = s.free
        lo, hi = self.lo0.copy(), self.hi0.copy()
        u0 = np.asarray(sc.u0, float)
        r0 = self.tags["logic0"]
        lo[r0] = hi[r0] = u0
        rd = self.tags["demand"]
        lo[rd] = hi[rd] = np.asarray(sc.load).sum(1) - self.mr_pmin
        lo[self.tags["reserve"]] = sc.sr
        lb, ub = np.zeros(self.nv), np.full(self.nv, np.inf)
        for name in ("u", "v", "w", "dl", "lam"):
            ub[self.blk(name).reshape(-1)] = 1.0
        rng_all = s.pmax_all - s.pmin_all
        ub[self.blk("p")] = np.broadcast_to(rng_all, (T, s.G_all))
        ub[self.blk("res")] = np.broadcast_to(rng_all, (T, s.G_all))
        Rw = self.blk("r")
        if Rw.size:                      # the data dict's avail may carry the must-run capacity as an extra column
            W = Rw.shape[1]
            ub[Rw] = np.asarray(sc.avail)[:, :W]
            amin = getattr(sc, "avail_min", None)
            lb[Rw] = np.asarray(amin)[:, :W] if amin is not None else 0.0
        lb[self.col_one] = ub[self.col_one] = 1.0
        # (4) / (5) initial up / down requirement (the system's initial condition)
        U = self.blk("u")
        for k, g in enumerate(F):
            if s.u0_all[g] == 1:
                n = min(int(s.ut_all[g]) - int(s.up0_all[g]), T)
                if n >= 1:
                    lb[U[:n, k]] = 1.0
            else:
                n = min(int(s.dt_all[g]) - int(s.dn0_all[g]), T)
                if n >= 1:
                    ub[U[:n, k]] = 0.0
            # (7) start-up categories that the initial off time rules out
            DL = self.blk("dl")
            lags = s.st_lag[g]
            for si in range(len(lags) - 1):
                a1 = max(1, lags[si + 1] - int(s.dn0_all[g]) + 1)
                b1 = min(lags[si + 1] - 1, T)
                if a1 <= b1:
                    ub[DL[a1 - 1:b1, self.dl0[k] + si]] = 0.0
        if u_fix is not None:
            u = np.asarray(u_fix, float).reshape(T, G)
            prev = np.vstack([u0[None, :], u[:-1]])
            v, w = np.maximum(u - prev, 0), np.maximum(prev - u, 0)
            for name, val in (("u", u), ("v", v), ("w", w)):
                b = self.blk(name).reshape(-1)
                lb[b] = ub[b] = val.reshape(-1)
        return lo, hi, lb, ub

    # ------------------------------------------------------------------------ solution helpers
    def get(self, x, name):
        o, k = self.off[name]
        return x[o:o + k * self.T].reshape(self.T, k)

    def _unpack(self, x):
        u = self.get(x, "u")
        p = self.get(x, "p")
        self._short = float(self.get(x, "short").sum())
        return u, p, self.get(x, "r"), None, None, float(self.get(x, "shed").sum() + self.get(x, "spill").sum())

    def bounds_fixed(self, sc, fix):
        lo, hi, lb, ub = self._rhs_bounds(sc)
        if fix:
            U = self.blk("u")
            tg = np.array(list(fix.keys()), int).reshape(-1, 2)
            vv = np.array(list(fix.values()), float)
            cols = U[tg[:, 0], tg[:, 1]]
            lb[cols] = np.maximum(lb[cols], vv)
            ub[cols] = np.minimum(ub[cols], vv)
        return lo, hi, lb, ub

    def solve_milp(self, sc, time_limit=600.0, mip_gap=1e-3, z_fix=None, threads=1, incumbents=True, opts=None):
        """full (z_fix None) or reduced MILP through highspy with an incumbent log"""
        lo, hi, lb, ub = self.bounds_fixed(sc, z_fix)
        r = highs_run(self.c, self.A, lo, hi, lb, ub, self.integ, time_limit, mip_gap, threads, incumbents, opts=opts)
        out = dict(status=r["status"], obj=r["obj"], time=r["time"], gap=r["gap"], bound=r["bound"], inc=r["inc"],
                   nodes=r["nodes"], u=None, shed=np.nan, short=np.nan)
        if r["x"] is not None:
            u, p, _, _, _, shed = self._unpack(r["x"])
            out.update(u=np.round(u).astype(np.int8), shed=shed, short=self._short, x=r["x"])
        return out

    def solve_dispatch(self, sc, u, relax=False, duals=False):
        """dispatch LP with fixed commitment u [T, G] (always feasible if u respects min up/down), or the LP
        relaxation (relax=True). Returns an object with obj, u, p, shed, short, time, lmp [T, 1] (demand dual)."""
        lo, hi, lb, ub = self._rhs_bounds(sc, None if relax else u)
        r = highs_run(self.c, self.A, lo, hi, lb, ub, None, 600.0, duals=True)
        return _Sol(self, r)

    def relaxed_reduced(self, sc, fix):
        """LP relaxation of the reduced MILP: (cost, per-hour penalised slack [T], seconds) - the fast
        equivalent of otsl.fixpolicy.relaxed_reduced"""
        lo, hi, lb, ub = self.bounds_fixed(sc, fix)
        r = highs_run(self.c, self.A, lo, hi, lb, ub, None, 600.0)
        if r["x"] is None:
            return np.inf, np.full(self.T, np.inf), r["time"]
        x = r["x"]
        slack = self.get(x, "shed").sum(1) + self.get(x, "spill").sum(1) + self.get(x, "short").sum(1)
        return r["obj"], slack, r["time"]


class _Sol:
    def __init__(self, m, r):
        self.status, self.obj, self.time = r["status"], r["obj"], r["time"]
        if r["x"] is None:
            self.u = self.p = None
            self.shed = self.short = np.inf
            self.lmp = None
            self.x = None
            return
        x = r["x"]
        self.x = x
        self.u, self.p, _, _, _, self.shed = m._unpack(x)
        self.short = m._short
        rd = r.get("row_dual")
        self.lmp = rd[m.tags["demand"]][:, None] if rd is not None else None
        self.res_price = rd[m.tags["reserve"]] if rd is not None else None

    @property
    def ok(self):
        return self.u is not None


# ============================================================================ repairs (no carry-over at t = 0 assumed by
# the reused otsl.uc.repair_min_updown; valid here because every PGLib-CA unit starts on with its minimum up time
# already served - asserted in check_no_carryover)
def check_no_carryover(sysm: PGSystem):
    F = sysm.free
    on = sysm.u0_all[F] == 1
    ok_on = np.all(sysm.up0_all[F][on] >= sysm.ut_all[F][on])
    ok_off = np.all(sysm.dn0_all[F][~on] >= sysm.dt_all[F][~on])
    return bool(ok_on and ok_off)


# ============================================================================ scenario set (perturbed CA instances)
BASES = ["2014-09-01", "2014-12-01", "2015-03-01", "2015-06-01", "Scenario400"]


def load_bases(T=48):
    """the five California load profiles (reserves_0 files: reserve requirement drawn per instance instead).
    Returns (PGSystem, list of dict(demand [T], wind_max [T, W], wind_min [T, W]))."""
    out = []
    sysm, _, _ = load_case(os.path.join(PGLIB, "ca", "Scenario400_reserves_0.json"), T=T)   # the fleet + its wind unit
    W = len(sysm.ren_names)
    for b in BASES:
        s, sc, _ = load_case(os.path.join(PGLIB, "ca", f"{b}_reserves_0.json"), T=T)
        assert np.array_equal(s.names_all, sysm.names_all) and np.allclose(s.pmax_all, sysm.pmax_all)
        wmax = sc.avail if sc.avail.shape[1] == W else np.zeros((T, W))     # profiles without wind: wind = 0
        wmin = sc.avail_min if sc.avail_min.shape[1] == W else np.zeros((T, W))
        out.append(dict(name=b, demand=sc.load[:, 0], wmax=wmax, wmin=wmin))
    return sysm, out


def _ar1(rng, T, sd, rho=0.7):
    e = np.zeros(T)
    e[0] = rng.normal(0, sd)
    for t in range(1, T):
        e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sd)
    return e


def make_instance(sysm, bases, rng, W=1, scale=(0.92, 1.08), noise=0.015, res=(0.0, 0.05), wind=(0.7, 1.3),
                  wind_noise=0.05, cap_margin=0.97):
    """One perturbed instance: a base profile (uniform over the 5), demand x U(scale) x (1 + AR(1) noise), reserve
    requirement U(res) x demand, wind (Scenario400 only) x U(wind) x (1 + AR(1) noise) with the base min/max ratio.
    The demand scale is capped so that peak net load + reserve <= cap_margin x thermal capacity."""
    b = int(rng.integers(len(bases)))
    B = bases[b]
    T = len(B["demand"])
    sc = rng.uniform(*scale)
    dem = B["demand"] * sc * (1 + _ar1(rng, T, noise))
    rf = rng.uniform(*res)
    ws = 1.0
    wmax = np.zeros((T, W)); wmin = np.zeros((T, W))
    if B["wmax"].sum() > 0:
        ws = rng.uniform(*wind)
        f = ws * np.clip(1 + _ar1(rng, T, wind_noise), 0.5, 1.5)
        wmax = B["wmax"] * f[:, None]
        ratio = np.where(B["wmax"] > 1e-9, B["wmin"] / np.maximum(B["wmax"], 1e-9), 0.0)
        wmin = wmax * ratio
    cap = sysm.pmax_all.sum()
    peak = np.max(dem * (1 + rf) - wmax.sum(1))
    if peak > cap_margin * cap:
        dem = dem * (cap_margin * cap + wmax.sum(1).min()) / (dem * (1 + rf)).max()
    mr_cap = float(sysm.pmax_all[sysm.mr].sum())
    return dict(load=dem[:, None], avail=np.c_[wmax, np.full(T, mr_cap)], avail_min=wmin, sr=rf * dem,
                u0=sysm.u0.copy(), base=b, scale=sc, res_frac=rf, wind_scale=ws)


def instance_scenario(inst):
    return PGScenario(load=inst["load"], avail=inst["avail"], avail_min=inst["avail_min"], sr=inst["sr"], u0=inst["u0"])


# ============================================================================ repairs and labels (free units; the data
# dict's "avail" carries the must-run capacity as its last column, so otsl.uc / otsl.constrained repairs count it)
def rep_minud(u, sysm, u0=None):
    from .uc import repair_min_updown
    return repair_min_updown(np.asarray(u, np.int8), sysm.u0 if u0 is None else u0, sysm.min_up, sysm.min_dn)


def rep_adequacy(u, inst, sysm):
    from .uc import adequacy_repair
    v = adequacy_repair(np.asarray(u, np.int8), inst["load"], inst["avail"], inst["sr"], sysm)
    return rep_minud(v, sysm, inst["u0"])


def rep_block(u, inst, sysm, margin=0.0):
    from .constrained import adequacy_repair_blocks
    v = adequacy_repair_blocks(np.asarray(u, np.int8), inst["u0"], inst["load"], inst["avail"], inst["sr"], sysm,
                               margin=margin)
    return rep_minud(v, sysm, inst["u0"])


# ============================================================================ generation (spawn pool)
K_INC = 64
LF_THRESHOLDS = (0.5, 0.2, 0.05, 1e-3)
_W = {}


def _init(cfg):
    sysm, bases = load_bases(cfg["T"])
    _W.update(s=sysm, bases=bases, m=PGModel(sysm, T=cfg["T"]), cfg=cfg)


def gen_instance(job, cfg=None, s=None, m=None):
    """job = (seed,) -> instance with LP relaxation, repaired relaxation labels and (mode 'milp' / 'label') the full
    MILP with its incumbent log"""
    cfg, s, m = cfg or _W["cfg"], s or _W["s"], m or _W["m"]
    seed = job[0] if isinstance(job, (tuple, list)) else job
    rng = np.random.default_rng(seed)
    for n_rej in range(20):       # keep only instances whose LP relaxation needs no shedding / reserve shortfall
        inst = make_instance(s, _W["bases"], rng, W=len(s.ren_names))
        sc = instance_scenario(inst)
        t0 = time.time()
        rel = m.solve_dispatch(sc, None, relax=True)
        if rel.shed + rel.short < 1e-6:
            break
    inst["seed"], inst["n_reject"] = seed, n_rej
    inst.update(u_rel=rel.u, lmp_rel=rel.lmp, res_rel=rel.res_price, c_rel=rel.obj, t_rel=rel.time)
    # label-free label: the LP relaxation rounded at a few thresholds, adequacy + min up/down repaired, the cheapest
    # by the exact dispatch LP (no MILP)
    if "fd" not in _W:
        from .pglib_ml import FastDispatch
        _W["fd"] = FastDispatch(m)
    best = None
    for th in LF_THRESHOLDS:
        lab = rep_adequacy((rel.u > th).astype(np.int8), inst, s)
        c, sh, so, _ = _W["fd"].solve(sc, lab, key=("gen", seed))
        if best is None or c < best[1] - 1e-9:
            best = (lab, c, sh, so, th)
    inst.update(y_lf=best[0], c_lf=best[1], shed_lf=best[2], short_lf=best[3], th_lf=best[4])
    inst["t_label"] = time.time() - t0
    mode = cfg.get("mode", "lp")
    T, G = m.T, s.G
    if mode in ("milp", "label"):
        sol = m.solve_milp(sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], incumbents=True)
        ok = sol["u"] is not None
        inst.update(u=sol["u"] if ok else np.zeros((T, G), np.int8), obj=sol["obj"], time=sol["time"], gap=sol["gap"],
                    bound=sol["bound"], opt=sol["status"] == "Optimal", nodes=sol["nodes"], shed=sol["shed"],
                    short=sol["short"], inc=sol["inc"], loadavg=os.getloadavg()[0])
        inst["obj_lp"] = m.solve_dispatch(sc, sol["u"]).obj if ok else np.nan
    else:
        inst.update(u=np.full((T, G), -1, np.int8), obj=np.nan, time=np.nan, gap=np.nan, bound=np.nan, opt=False,
                    nodes=-1, shed=np.nan, short=np.nan, inc=[], obj_lp=np.nan, loadavg=os.getloadavg()[0])
    return inst


def _gen_job(job):
    return gen_instance(job)


GEN_KEYS = ["load", "avail", "avail_min", "sr", "u0", "base", "scale", "res_frac", "wind_scale", "seed", "u_rel",
            "lmp_rel", "res_rel", "c_rel", "t_rel", "y_lf", "c_lf", "th_lf", "n_reject", "shed_lf", "short_lf", "t_label", "u", "obj", "time",
            "gap", "bound", "opt", "nodes", "shed", "short", "obj_lp", "loadavg"]


def pack_inc(incs, k=K_INC):
    n = len(incs)
    tt, oo, bb = np.full((n, k), np.nan), np.full((n, k), np.nan), np.full((n, k), np.nan)
    for i, inc in enumerate(incs):
        if len(inc) > k:
            inc = list(inc[:k - 1]) + [inc[-1]]
        for j, (t, o, b) in enumerate(inc):
            tt[i, j], oo[i, j], bb[i, j] = t, o, b
    return tt, oo, bb


def to_arrays(res):
    data = {k: np.array([r[k] for r in res]) for k in GEN_KEYS}
    data["inc_t"], data["inc_obj"], data["inc_bound"] = pack_inc([r["inc"] for r in res])
    data["n_inc"] = np.array([len(r["inc"]) for r in res])
    return data


def save_npz(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def load_npz(path):
    return dict(np.load(path, allow_pickle=False))


def subset(d, idx):
    n = len(d["load"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}


def concat(ds):
    keys = set.intersection(*[set(d) for d in ds])
    return {k: np.concatenate([d[k] for d in ds]) for k in keys}


def generate(cfg, seeds, workers=2, log=print, partial_path=None):
    import multiprocessing as mp
    res, t0 = [], time.time()
    n = len(seeds)
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap(_gen_job, [(int(x),) for x in seeds], chunksize=1)):
            res.append(r)
            if cfg.get("mode", "lp") != "lp":
                log(f"[gen] {i + 1}/{n} base {r['base']} MILP {r['time']:.1f}s gap {r['gap'] * 100:.3f}% obj {r['obj']:.2f} "
                    f"bound {r['bound']:.2f} lp-check {r['obj_lp']:.2f} c_rel {r['c_rel']:.2f} c_lf {r['c_lf']:.2f} "
                    f"inc {len(r['inc'])} load {r['loadavg']:.1f} ({time.time() - t0:.0f}s)")
            elif (i + 1) % max(1, n // 20) == 0:
                log(f"[gen] {i + 1}/{n} ({time.time() - t0:.0f}s) t_label {r['t_label']:.1f}s c_lf/c_rel "
                    f"{r['c_lf'] / r['c_rel']:.4f}")
            if partial_path and (i + 1) % 5 == 0:
                save_npz(partial_path, to_arrays(res))
    return to_arrays(res)


def time_to_reach(inc_t, inc_obj, target, rel_tol=1e-6):
    """first time at which an incumbent log reaches obj <= target (1 + rel_tol); nan if never"""
    for t, o in zip(inc_t, inc_obj):
        if np.isfinite(t) and o <= target * (1 + rel_tol) + 1e-9:
            return float(t)
    return np.nan
