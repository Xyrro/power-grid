"""Network-constrained unit commitment (UC) on RTS-GMLC.

Day-ahead UC over T hourly periods (T = 24 by default, T = 1 gives the single-snapshot variant):

    min  sum_t sum_g [ c_nl u + sum_s c_s p_s + c_su v ] + VOLL * shed + c_res * reserve_shortfall
    s.t. u_t - u_{t-1} = v_t - w_t                     (u_0 = initial status)
         min up / min down times (Rajan-Takriti)
         p = pmin u + sum_s p_s,  0 <= p_s <= width_s u    (piecewise-linear heat-rate cost)
         ramping  p_t - p_{t-1} <= RU + pmax v_t,  p_{t-1} - p_t <= RD + pmax w_t
         spinning reserve  r_g <= pmax u - p,  sum_g r_g + shortfall >= SR_t
         DC network: KCL per bus, f = b (theta_i - theta_j), |f| <= rating
         renewables (PV, RTPV, wind, hydro) curtailable up to their profile; load shedding and
         over-generation allowed at VOLL (soft balance, as in MISO's RAC / RACLearn), so every
         commitment has a feasible dispatch.

Given the commitment u, the problem is an LP ("LP solver" box of the framework): solve_dispatch().
Units are in p.u. on a 100 MVA base; costs in $/h.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rts_gmlc")
BASE = 100.0
VOLL = 10_000.0 * BASE        # $/p.u.h  (= $10,000/MWh)
RES_SHORT = 1_000.0 * BASE    # $/p.u.h of spinning-reserve shortfall
THERMAL = ["CT", "CC", "STEAM", "NUCLEAR"]
RENEW = ["PV", "RTPV", "WIND", "HYDRO", "ROR"]


@dataclass
class UCSystem:
    n_bus: int
    ref: int
    bus_ids: np.ndarray
    bus_area: np.ndarray
    load_share: np.ndarray          # [N] share of its area's load
    f_bus: np.ndarray
    t_bus: np.ndarray
    b: np.ndarray                   # [L] susceptance (p.u.)
    fmax: np.ndarray                # [L] continuous rating (p.u.)
    # thermal units
    uid: np.ndarray
    utype: np.ndarray
    gbus: np.ndarray
    pmax: np.ndarray
    pmin: np.ndarray
    seg_w: np.ndarray               # [G, S] segment widths (p.u.)
    seg_c: np.ndarray               # [G, S] segment marginal costs ($/p.u.h)
    c_nl: np.ndarray                # [G] cost of running at pmin ($/h)
    c_su: np.ndarray                # [G] start-up cost ($)
    min_up: np.ndarray              # [G] hours
    min_dn: np.ndarray
    ramp: np.ndarray                # [G] p.u./h
    # renewables
    r_uid: np.ndarray
    r_type: np.ndarray
    r_bus: np.ndarray
    r_pmax: np.ndarray
    # time series (hourly, 2020): [days, 24, ...]
    area_load: np.ndarray           # [D, 24, 3] p.u.
    r_avail: np.ndarray             # [D, 24, R] p.u.
    dates: np.ndarray

    @property
    def G(self): return len(self.pmax)

    @property
    def L(self): return len(self.b)

    @property
    def A(self):
        A = np.zeros((self.L, self.n_bus))
        A[np.arange(self.L), self.f_bus] = 1
        A[np.arange(self.L), self.t_bus] = -1
        return A

    def identical_groups(self):
        """Groups of thermal units that are exact copies (same bus, same parameters): their schedules
        can be swapped without changing cost or flows -> symmetric (tied) MILP optima."""
        key = [(int(self.gbus[g]), round(float(self.pmax[g]), 6), round(float(self.c_nl[g]), 3),
                round(float(self.c_su[g]), 3), int(self.min_up[g]), int(self.min_dn[g])) for g in range(self.G)]
        groups = {}
        for g, k in enumerate(key):
            groups.setdefault(k, []).append(g)
        return [v for v in groups.values() if len(v) > 1]


def load_rts_gmlc(rating="Cont Rating", line_scale=1.0) -> UCSystem:
    d = DATA
    bus = pd.read_csv(os.path.join(d, "bus.csv"))
    br = pd.read_csv(os.path.join(d, "branch.csv"))
    gen = pd.read_csv(os.path.join(d, "gen.csv"))
    bid = bus["Bus ID"].to_numpy()
    idx = {b: i for i, b in enumerate(bid)}
    area = bus["Area"].to_numpy()
    share = np.zeros(len(bus))
    for a in np.unique(area):
        m = area == a
        share[m] = bus["MW Load"].to_numpy()[m] / bus["MW Load"].to_numpy()[m].sum()
    ref = int(np.where(bus["Bus Type"].astype(str).str.upper() == "REF")[0][0]) if "REF" in set(bus["Bus Type"].astype(str).str.upper()) else 0
    tap = br["Tr Ratio"].replace(0, 1.0).fillna(1.0).to_numpy()
    b = 1.0 / (br["X"].to_numpy() * tap)
    fmax = br[rating].to_numpy() / BASE * line_scale

    th = gen[gen["Unit Type"].isin(THERMAL)].reset_index(drop=True)
    G = len(th)
    pmax = th["PMax MW"].to_numpy() / BASE
    pmin = th["PMin MW"].to_numpy() / BASE
    fuel = th["Fuel Price $/MMBTU"].to_numpy()
    vom = th["VOM"].fillna(0).to_numpy()
    S = 3
    seg_w = np.zeros((G, S))
    seg_c = np.zeros((G, S))
    for g in range(G):
        pct = [th.loc[g, f"Output_pct_{i}"] for i in range(5)]
        inc = [th.loc[g, f"HR_incr_{i}"] for i in range(1, 5)]
        pts = [p for p in pct if pd.notna(p)]
        incs = [h for h in inc if pd.notna(h)]
        k = 0
        for i in range(1, len(pts)):
            if k >= S:
                break
            w = (pts[i] - pts[i - 1]) * pmax[g]
            seg_w[g, k] = w
            seg_c[g, k] = (incs[i - 1] / 1000.0 * fuel[g] + vom[g]) * BASE
            k += 1
        # remaining capacity above the last breakpoint (if pmin + widths < pmax)
        rem = pmax[g] - pmin[g] - seg_w[g].sum()
        if rem > 1e-9:
            seg_w[g, max(k - 1, 0)] += rem
    c_nl = (th["HR_avg_0"].to_numpy() / 1000.0 * fuel + vom) * pmin * BASE
    c_su = th["Start Heat Hot MBTU"].fillna(0).to_numpy() * fuel + th["Non Fuel Start Cost $"].fillna(0).to_numpy()
    min_up = np.maximum(1, np.ceil(th["Min Up Time Hr"].to_numpy())).astype(int)
    min_dn = np.maximum(1, np.ceil(th["Min Down Time Hr"].to_numpy())).astype(int)
    ramp = th["Ramp Rate MW/Min"].to_numpy() * 60 / BASE

    rn = gen[gen["Unit Type"].isin(RENEW)].reset_index(drop=True)
    load = pd.read_csv(os.path.join(d, "DAY_AHEAD_regional_Load.csv"))
    D = len(load) // 24
    area_load = load[["1", "2", "3"]].to_numpy().reshape(D, 24, 3) / BASE
    av = np.zeros((len(load), len(rn)))
    for f in ["DAY_AHEAD_pv.csv", "DAY_AHEAD_rtpv.csv", "DAY_AHEAD_wind.csv", "DAY_AHEAD_hydro.csv"]:
        ts = pd.read_csv(os.path.join(d, f))
        for j, u in enumerate(rn["GEN UID"]):
            if u in ts.columns:
                av[:, j] = ts[u].to_numpy()
    r_pmax = rn["PMax MW"].to_numpy() / BASE
    av = np.minimum(av / BASE, r_pmax[None, :])
    dates = load[["Year", "Month", "Day"]].to_numpy()[::24]
    return UCSystem(
        n_bus=len(bus), ref=ref, bus_ids=bid, bus_area=area, load_share=share,
        f_bus=np.array([idx[x] for x in br["From Bus"]]), t_bus=np.array([idx[x] for x in br["To Bus"]]),
        b=b, fmax=fmax, uid=th["GEN UID"].to_numpy(), utype=th["Unit Type"].to_numpy(),
        gbus=np.array([idx[x] for x in th["Bus ID"]]), pmax=pmax, pmin=pmin, seg_w=seg_w, seg_c=seg_c,
        c_nl=c_nl, c_su=c_su, min_up=min_up, min_dn=min_dn, ramp=ramp,
        r_uid=rn["GEN UID"].to_numpy(), r_type=rn["Unit Type"].to_numpy(),
        r_bus=np.array([idx[x] for x in rn["Bus ID"]]), r_pmax=r_pmax,
        area_load=area_load, r_avail=av.reshape(D, 24, -1), dates=dates)


@dataclass
class UCScenario:
    load: np.ndarray        # [T, N] p.u.
    avail: np.ndarray       # [T, R] p.u.
    u0: np.ndarray          # [G] initial status
    sr: np.ndarray          # [T] spinning reserve requirement (p.u.)
    day: int = -1


def make_scenario(sysm: UCSystem, day: int, rng=None, T=24, start=0, load_noise=(0.92, 1.08),
                  bus_noise=(0.97, 1.03), ren_noise=(0.8, 1.2), reserve_frac=0.03):
    """One instance: a calendar day's profiles (hours start .. start+T-1) with random area/bus load and
    per-unit renewable scaling. Initial status u0: priority-list commitment covering the net load +
    reserve of the hour *before* the window (with its own noise) - i.e. the units that were running."""
    rng = rng or np.random.default_rng(0)
    hrs = slice(start, start + T)
    lf = rng.uniform(*load_noise, size=(1, 3))
    bn = rng.uniform(*bus_noise, size=(1, sysm.n_bus))
    rf = rng.uniform(*ren_noise, size=(1, len(sysm.r_pmax)))
    al = sysm.area_load[day, hrs] * lf                                               # [T, 3]
    load = al[:, sysm.bus_area - 1] * sysm.load_share[None, :] * bn
    avail = np.minimum(sysm.r_avail[day, hrs] * rf, sysm.r_pmax[None, :])
    sr = reserve_frac * load.sum(1)
    h0 = max(start - 1, 0)
    n0 = rng.uniform(0.97, 1.03)
    net = (sysm.area_load[day, h0] * lf[0]).sum() * n0 - np.minimum(sysm.r_avail[day, h0] * rf[0], sysm.r_pmax).sum() \
        + reserve_frac * (sysm.area_load[day, h0] * lf[0]).sum()
    avg_cost = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    u0 = np.zeros(sysm.G, np.int8)
    cap = 0.0
    for g in np.argsort(avg_cost):
        if cap >= net * 1.05:
            break
        u0[g] = 1
        cap += sysm.pmax[g]
    return UCScenario(load=load, avail=avail, u0=u0, sr=sr, day=day)


@dataclass
class UCSolution:
    status: str
    obj: float
    u: np.ndarray | None        # [T, G]
    p: np.ndarray | None        # [T, G]
    r: np.ndarray | None        # [T, R] renewable dispatch
    va: np.ndarray | None       # [T, N]
    flow: np.ndarray | None     # [T, L]
    shed: float = 0.0
    short: float = 0.0
    lmp: np.ndarray | None = None
    time: float = 0.0
    gap: float = 0.0

    @property
    def ok(self):
        return self.u is not None


class UCModel:
    """Builds the constraint matrix once for (system, T); scenarios only change bounds / RHS."""

    def __init__(self, sysm: UCSystem, T=24, network=True):
        self.s, self.T, self.network = sysm, T, network
        s = sysm
        G, S, N, L, R = s.G, s.seg_w.shape[1], s.n_bus, s.L, len(s.r_pmax)
        self.dims = dict(G=G, S=S, N=N, L=L, R=R)
        # variable layout (all [T, ...] blocks flattened t-major)
        sizes = [("u", G), ("v", G), ("w", G), ("ps", G * S), ("res", G), ("r", R), ("th", N), ("f", L),
                 ("shed", N), ("spill", N), ("short", 1)]
        self.off, o = {}, 0
        for name, k in sizes:
            self.off[name] = (o, k)
            o += k * T
        self.nv = o
        self._build()

    def ix(self, name, t, j=0):
        o, k = self.off[name]
        return o + t * k + j

    def _build(self):
        s, T = self.s, self.T
        G, S, N, L, R = (self.dims[k] for k in "GSNLR")
        rows, cols, vals, lo, hi = [], [], [], [], []
        self.row_tag = {}
        r = 0

        def add(entries, l, h, tag=None):
            nonlocal r
            for c, v in entries:
                rows.append(r); cols.append(c); vals.append(v)
            lo.append(l); hi.append(h)
            if tag is not None:
                self.row_tag.setdefault(tag, []).append(r)
            r += 1

        A = s.A
        for t in range(T):
            for g in range(G):
                # logic u_t - u_{t-1} - v_t + w_t = 0   (t = 0: rhs = u0, filled per scenario)
                e = [(self.ix("u", t, g), 1.0), (self.ix("v", t, g), -1.0), (self.ix("w", t, g), 1.0)]
                if t > 0:
                    e.append((self.ix("u", t - 1, g), -1.0))
                add(e, 0.0, 0.0, tag=("logic0", g) if t == 0 else None)
                # min up / down
                if s.min_up[g] > 1:
                    e = [(self.ix("v", tau, g), 1.0) for tau in range(max(0, t - s.min_up[g] + 1), t + 1)]
                    add(e + [(self.ix("u", t, g), -1.0)], -np.inf, 0.0)
                if s.min_dn[g] > 1:
                    e = [(self.ix("w", tau, g), 1.0) for tau in range(max(0, t - s.min_dn[g] + 1), t + 1)]
                    add(e + [(self.ix("u", t, g), 1.0)], -np.inf, 1.0)
                for k in range(S):
                    if s.seg_w[g, k] > 0:
                        add([(self.ix("ps", t, g * S + k), 1.0), (self.ix("u", t, g), -s.seg_w[g, k])], -np.inf, 0.0)
                # reserve headroom: res + pmin u + sum ps - pmax u <= 0
                e = [(self.ix("res", t, g), 1.0), (self.ix("u", t, g), s.pmin[g] - s.pmax[g])]
                e += [(self.ix("ps", t, g * S + k), 1.0) for k in range(S)]
                add(e, -np.inf, 0.0)
                # ramping
                if t > 0 and s.ramp[g] < s.pmax[g] - s.pmin[g]:
                    pt = [(self.ix("u", t, g), s.pmin[g])] + [(self.ix("ps", t, g * S + k), 1.0) for k in range(S)]
                    pp = [(self.ix("u", t - 1, g), -s.pmin[g])] + [(self.ix("ps", t - 1, g * S + k), -1.0) for k in range(S)]
                    add(pt + pp + [(self.ix("v", t, g), -s.pmax[g])], -np.inf, s.ramp[g])
                    neg = lambda es: [(c, -v) for c, v in es]
                    add(neg(pt) + neg(pp) + [(self.ix("w", t, g), -s.pmax[g])], -np.inf, s.ramp[g])
            # reserve requirement
            e = [(self.ix("res", t, g), 1.0) for g in range(G)] + [(self.ix("short", t), 1.0)]
            add(e, 0.0, np.inf, tag=("sr", t))
            if self.network:
                for n in range(N):
                    e = []
                    for g in np.where(s.gbus == n)[0]:
                        e.append((self.ix("u", t, g), s.pmin[g]))
                        e += [(self.ix("ps", t, g * S + k), 1.0) for k in range(S)]
                    e += [(self.ix("r", t, j), 1.0) for j in np.where(s.r_bus == n)[0]]
                    e += [(self.ix("f", t, l), -A[l, n]) for l in np.where(A[:, n] != 0)[0]]
                    e.append((self.ix("shed", t, n), 1.0))
                    e.append((self.ix("spill", t, n), -1.0))
                    add(e, 0.0, 0.0, tag=("kcl", t, n))
                for l in range(L):
                    add([(self.ix("f", t, l), 1.0), (self.ix("th", t, s.f_bus[l]), -s.b[l]),
                         (self.ix("th", t, s.t_bus[l]), s.b[l])], 0.0, 0.0)
            else:  # copper plate
                e = []
                for g in range(G):
                    e.append((self.ix("u", t, g), s.pmin[g]))
                    e += [(self.ix("ps", t, g * S + k), 1.0) for k in range(S)]
                e += [(self.ix("r", t, j), 1.0) for j in range(R)] + [(self.ix("shed", t, n), 1.0) for n in range(N)]
                e += [(self.ix("spill", t, n), -1.0) for n in range(N)]
                add(e, 0.0, 0.0, tag=("bal", t))
        self.A = sp.csr_matrix((vals, (rows, cols)), shape=(r, self.nv))
        self.lo0, self.hi0 = np.array(lo, float), np.array(hi, float)
        # objective
        c = np.zeros(self.nv)
        for t in range(T):
            for g in range(G):
                c[self.ix("u", t, g)] = s.c_nl[g]
                c[self.ix("v", t, g)] = s.c_su[g]
                for k in range(S):
                    c[self.ix("ps", t, g * S + k)] = s.seg_c[g, k]
            for n in range(N):
                c[self.ix("shed", t, n)] = VOLL
                c[self.ix("spill", t, n)] = VOLL      # over-generation (min outputs above demand)
            c[self.ix("short", t)] = RES_SHORT
        self.c = c
        self.integ = np.zeros(self.nv)
        for name in ("u", "v", "w"):
            o, k = self.off[name]
            self.integ[o:o + k * T] = 1

    def _rhs_bounds(self, sc: UCScenario, u_fix=None, relax=False):
        s, T = self.s, self.T
        G, S, N, L, R = (self.dims[k] for k in "GSNLR")
        lo, hi = self.lo0.copy(), self.hi0.copy()
        for g in range(G):
            r = self.row_tag[("logic0", g)][0]
            lo[r] = hi[r] = sc.u0[g]
        for t in range(T):
            r = self.row_tag[("sr", t)][0]
            lo[r] = sc.sr[t]
            if self.network:
                for n in range(N):
                    r = self.row_tag[("kcl", t, n)][0]
                    lo[r] = hi[r] = sc.load[t, n]
            else:
                r = self.row_tag[("bal", t)][0]
                lo[r] = hi[r] = sc.load[t].sum()
        lb, ub = np.zeros(self.nv), np.full(self.nv, np.inf)
        for name in ("u", "v", "w"):
            o, k = self.off[name]
            ub[o:o + k * T] = 1.0
        for t in range(T):
            for g in range(G):
                for k in range(S):
                    ub[self.ix("ps", t, g * S + k)] = s.seg_w[g, k]
            o = self.ix("r", t)
            ub[o:o + R] = sc.avail[t]
            o = self.ix("th", t)
            lb[o:o + N], ub[o:o + N] = -np.pi, np.pi
            lb[o + s.ref] = ub[o + s.ref] = 0.0
            o = self.ix("f", t)
            lb[o:o + L], ub[o:o + L] = -s.fmax, s.fmax
            o = self.ix("shed", t)
            ub[o:o + N] = sc.load[t]
        if u_fix is not None:   # fixed commitment -> derive v, w and fix all three
            u = np.asarray(u_fix, float).reshape(T, G)
            prev = np.vstack([sc.u0[None, :], u[:-1]])
            v = np.maximum(u - prev, 0)
            w = np.maximum(prev - u, 0)
            for name, val in (("u", u), ("v", v), ("w", w)):
                o, k = self.off[name]
                lb[o:o + k * T] = ub[o:o + k * T] = val.reshape(-1)
        return lo, hi, lb, ub

    def _unpack(self, x):
        s, T = self.s, self.T
        G, S, N, L, R = (self.dims[k] for k in "GSNLR")
        get = lambda name: x[self.off[name][0]:self.off[name][0] + self.off[name][1] * T].reshape(T, -1)
        u = get("u")
        p = u * s.pmin[None, :] + get("ps").reshape(T, G, S).sum(2)
        self._short = float(get("short").sum())
        return u, p, get("r"), get("th"), get("f"), float(get("shed").sum() + get("spill").sum())

    def solve_uc(self, sc: UCScenario, time_limit=60.0, mip_gap=1e-3, z_fix: dict | None = None,
                 nogood: list | None = None) -> UCSolution:
        """MILP. z_fix: {(t, g): 0/1} partial fixing of commitments (predict-and-fix).
        nogood: list of [T, G] commitments to exclude (enumerates alternative optima)."""
        lo, hi, lb, ub = self._rhs_bounds(sc)
        for (t, g), v in (z_fix or {}).items():
            i = self.ix("u", t, g)
            lb[i] = ub[i] = v
        A = self.A
        if nogood:
            o, k = self.off["u"]
            rows = []
            for ng in nogood:
                ng = np.asarray(ng).reshape(-1)
                coef = np.zeros(self.nv)
                coef[o:o + k * self.T] = np.where(ng > 0.5, -1.0, 1.0)
                rows.append(coef)
                lo = np.r_[lo, 1.0 - ng.sum()]
                hi = np.r_[hi, np.inf]
            A = sp.vstack([A, sp.csr_matrix(np.array(rows))], format="csr")
        t0 = time.time()
        res = milp(self.c, constraints=LinearConstraint(A, lo, hi), integrality=self.integ,
                   bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False})
        dt = time.time() - t0
        if res.x is None:
            return UCSolution("infeasible", np.inf, None, None, None, None, None, time=dt)
        u, p, r, th, f, shed = self._unpack(res.x)
        st = "optimal" if res.status == 0 else "time_limit"
        return UCSolution(st, float(res.fun), np.round(u).astype(np.int8), p, r, th, f, shed=shed, short=self._short, time=dt,
                          gap=float(getattr(res, "mip_gap", 0) or 0))

    def solve_dispatch(self, sc: UCScenario, u, relax=False) -> UCSolution:
        """LP with a fixed commitment (the framework's 'LP solver'), or the LP relaxation (relax=True,
        u ignored). Always feasible thanks to load shedding / reserve shortfall slacks."""
        lo, hi, lb, ub = self._rhs_bounds(sc, None if relax else u)
        t0 = time.time()
        A_eq_mask = lo == hi
        res = linprog(self.c, A_ub=sp.vstack([self.A[~A_eq_mask & np.isfinite(hi)], -self.A[~A_eq_mask & np.isfinite(lo)]]),
                      b_ub=np.r_[hi[~A_eq_mask & np.isfinite(hi)], -lo[~A_eq_mask & np.isfinite(lo)]],
                      A_eq=self.A[A_eq_mask], b_eq=lo[A_eq_mask], bounds=np.c_[lb, ub], method="highs")
        dt = time.time() - t0
        if res.status != 0:
            return UCSolution("infeasible", np.inf, None, None, None, None, None, time=dt)
        uu, p, r, th, f, shed = self._unpack(res.x)
        sol = UCSolution("optimal", float(res.fun), uu, p, r, th, f, shed=shed, short=self._short, time=dt)
        if self.network:
            eq_rows = np.where(A_eq_mask)[0]
            pos = {row: i for i, row in enumerate(eq_rows)}
            T, N = self.T, self.dims["N"]
            lmp = np.zeros((T, N))
            for t in range(T):
                for n in range(N):
                    lmp[t, n] = res.eqlin.marginals[pos[self.row_tag[("kcl", t, n)][0]]]
            sol.lmp = lmp
        return sol


def repair_min_updown(u, u0, min_up, min_dn):
    """RACLearn-style repair: nearest schedule (Hamming) per unit that satisfies min up/down times,
    by dynamic programming over (state, time-in-state). u: [T, G] predicted 0/1. Initial units are
    assumed to have satisfied their minimum times (no carry-over), matching UCModel."""
    T, G = u.shape
    out = np.zeros_like(u)
    for g in range(G):
        U, D = int(min_up[g]), int(min_dn[g])
        # state: (s, k) with s in {0,1}, k = hours in state capped at U (on) or D (off)
        INF = 10 ** 9
        cost = {(int(u0[g]), U if u0[g] else D): 0}
        back = []
        for t in range(T):
            nxt, bp = {}, {}
            for (s, k), c in cost.items():
                for s2 in (0, 1):
                    if s2 == s:
                        k2 = min(k + 1, U if s else D)
                    else:
                        if (s == 1 and k < U) or (s == 0 and k < D):
                            continue
                        k2 = 1
                    c2 = c + (s2 != u[t, g])
                    if c2 < nxt.get((s2, k2), INF):
                        nxt[(s2, k2)] = c2
                        bp[(s2, k2)] = (s, k)
            back.append(bp)
            cost = nxt
        st = min(cost, key=cost.get)
        for t in range(T - 1, -1, -1):
            out[t, g] = st[0]
            st = back[t][st]
    return out


def violates_min_updown(u, u0, min_up, min_dn):
    """Number of units whose schedule violates min up/down (with no carry-over at t = 0)."""
    T, G = u.shape
    bad = 0
    for g in range(G):
        prev, run, ok = int(u0[g]), max(min_up[g], min_dn[g]), True
        for t in range(T):
            s = int(u[t, g])
            if s != prev:
                need = min_up[g] if prev == 1 else min_dn[g]
                if run < need:
                    ok = False
                    break
                run = 1
            else:
                run += 1
            prev = s
        bad += not ok
    return bad
