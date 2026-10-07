"""Robustness under distribution shift for the 12-hour UC (B2): shifted scenarios, an availability- and
topology-aware UC model, and the overrides that let the already-trained methods run on shifted instances.

Shifts (applied on top of the unchanged scenario generator of otsl.uc.make_scenario; test calendar days only):

* load_hi / load_lo   area loads x 1.15 / x 0.85 (on top of the generator's U[0.92, 1.08] area noise); the
                      reserve requirement (3 % of load) and the priority-list initial status follow the scaled load.
* ren_hi              wind, PV and rooftop-PV fleet x 1.5 (installed capacity and availability; hydro unchanged).
* gen_out             2-3 thermal units that are running at the start of the window (u0 = 1) trip and are
                      unavailable for the whole horizon (u = 0 forced in every model, LP and MILP).
* line_out            1-2 transmission lines out for the horizon (flow forced to 0, angle equation dropped), drawn at
                      random from the 4 lines whose single outage raises the instance's LP relaxation cost most (N-1
                      screening of its 20 most loaded lines on the intact network), never islanding a bus
                      (`with_loaded_line_outage`).
* night               12-hour windows starting at 13:00-23:00 and crossing midnight (training windows start at
                      00:00-12:00 only); profiles of the next calendar day are appended.

Models that do not see availability or topology are handled as follows (all in this module, nothing else changes):
an unavailable unit's on-probability is overridden to 0 (OFF) before any rule or decoder, its fixings are forced to
OFF after the guards, and it never counts towards the fixed share. The solver-free guards and repairs (adequacy
guard, block adequacy repair) are written against system data; they get the available fleet (`avail_view`).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from .ltfx import highs_solve
from .uc import UCModel, UCScenario

SHIFT_NAMES = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
VRE = ("WIND", "PV", "RTPV")


@dataclass
class OODScenario(UCScenario):
    g_out: tuple = field(default_factory=tuple)     # unavailable thermal units (whole horizon)
    l_out: tuple = field(default_factory=tuple)     # lines out of service (whole horizon)
    start: int = 0


# ============================================================================ scenario generation
def _profiles(sysm, day, start, T):
    """area loads [T, 3] and renewable availability [T, R] for hours start .. start + T - 1 of `day`, continuing
    into the next calendar day when the window crosses midnight"""
    if start + T <= 24:
        return sysm.area_load[day, start:start + T], sysm.r_avail[day, start:start + T], \
            sysm.area_load[day, max(start - 1, 0)], sysm.r_avail[day, max(start - 1, 0)]
    al = np.concatenate([sysm.area_load[day], sysm.area_load[day + 1]])
    ra = np.concatenate([sysm.r_avail[day], sysm.r_avail[day + 1]])
    return al[start:start + T], ra[start:start + T], al[start - 1], ra[start - 1]


def make_ood_scenario(sysm, day, rng, T=12, start=0, load_mult=1.0, vre_mult=1.0, n_gen_out=0, n_line_out=0,
                      line_pool=None, load_noise=(0.92, 1.08), bus_noise=(0.97, 1.03), ren_noise=(0.8, 1.2),
                      reserve_frac=0.03):
    """otsl.uc.make_scenario with the shifts of this module. With the defaults (no shift, start + T <= 24) it draws
    the same random numbers in the same order and returns exactly make_scenario's instance (tested). Outages are
    drawn after the noise, from the same generator."""
    lf = rng.uniform(*load_noise, size=(1, 3)) * load_mult
    bn = rng.uniform(*bus_noise, size=(1, sysm.n_bus))
    rf = rng.uniform(*ren_noise, size=(1, len(sysm.r_pmax)))
    al_w, ra_w, al_0, ra_0 = _profiles(sysm, day, start, T)
    vre = np.isin(np.asarray(sysm.r_type), VRE)
    kmul = np.where(vre, vre_mult, 1.0)                   # capacity and availability of the VRE fleet scaled
    al = al_w * lf
    load = al[:, sysm.bus_area - 1] * sysm.load_share[None, :] * bn
    avail = np.minimum(ra_w * rf, sysm.r_pmax[None, :]) * kmul[None, :]
    sr = reserve_frac * load.sum(1)
    n0 = rng.uniform(0.97, 1.03)
    net = (al_0 * lf[0]).sum() * n0 - (np.minimum(ra_0 * rf[0], sysm.r_pmax) * kmul).sum() + reserve_frac * (al_0 * lf[0]).sum()
    avg_cost = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    u0 = np.zeros(sysm.G, np.int8)
    cap = 0.0
    for g in np.argsort(avg_cost):
        if cap >= net * 1.05:
            break
        u0[g] = 1
        cap += sysm.pmax[g]
    g_out, l_out = (), ()
    if n_gen_out:
        k = int(rng.integers(n_gen_out[0], n_gen_out[1] + 1))
        on = np.where(u0 == 1)[0]
        pick = list(rng.choice(on, min(k, len(on)), replace=False)) if len(on) else []
        if len(pick) < k:                                 # fewer running units than outages: fill from the rest
            rest = np.setdiff1d(np.arange(sysm.G), pick)
            pick += list(rng.choice(rest, k - len(pick), replace=False))
        g_out = tuple(sorted(int(g) for g in pick))
    if n_line_out:
        k = int(rng.integers(n_line_out[0], n_line_out[1] + 1))
        pool = np.asarray(line_pool if line_pool is not None else np.arange(sysm.L))
        for _ in range(1000):
            pick = rng.choice(pool, k, replace=False)
            if connected_without(sysm, pick):
                break
        l_out = tuple(sorted(int(x) for x in pick))
    return OODScenario(load=load, avail=avail, u0=u0, sr=sr, day=day, g_out=g_out, l_out=l_out, start=start)


def with_loaded_line_outage(sysm, m, sc, rng, n_range=(1, 2), top=4, screen=20):
    """N-1 screening on the LP relaxation: of the `screen` non-bridge lines with the highest peak loading |flow| / rating
    in the instance's LP relaxation on the intact network, the `top` whose single outage raises the relaxation cost most;
    1-2 of them (equal odds) are drawn without replacement and set as sc.l_out, redrawn until no bus is islanded.
    Returns (scenario, intact relaxation cost, relaxation cost increase % of each chosen line alone)."""
    intact = OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, g_out=sc.g_out)
    rel = m.solve_dispatch(intact, None, relax=True)
    loading = (np.abs(rel.flow) / sysm.fmax[None, :]).max(0)
    cand = [int(l) for l in np.argsort(-loading, kind="stable") if connected_without(sysm, [l])][:screen]
    inc = {}
    for l in cand:
        c = m.solve_dispatch(OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, g_out=sc.g_out, l_out=(l,)),
                             None, relax=True).obj
        inc[l] = (c - rel.obj) / rel.obj * 100
    best = sorted(cand, key=lambda l: -inc[l])[:top]
    k = int(rng.integers(n_range[0], n_range[1] + 1))
    for _ in range(1000):
        pick = rng.choice(best, k, replace=False)
        if connected_without(sysm, pick):
            break
    sc.l_out = tuple(sorted(int(x) for x in pick))
    return sc, float(rel.obj), [float(inc[l]) for l in sc.l_out]


def build_shift_scenario(sysm, m, kw, job, line_pool=None):
    """the instance of one job of a shift specification (scripts/uc_ood_gen.py): noise and unit outages from the job
    seed; loaded-line outages ('line_top' in kw) from a second generator seeded with job seed + 7919"""
    kw = dict(kw)
    top = kw.pop("line_top", None)
    screen = kw.pop("line_screen", 20)
    n_line = kw.pop("n_line_out", None)
    if n_line is not None and top is None:
        kw.update(n_line_out=n_line, line_pool=line_pool)
    sc = make_ood_scenario(sysm, job["day"], np.random.default_rng(job["seed"]), T=12, start=job["start"], **kw)
    if top is not None:
        sc, _, _ = with_loaded_line_outage(sysm, m, sc, np.random.default_rng(job["seed"] + 7919), tuple(n_line), top, screen)
    return sc


def connected_without(sysm, lines):
    keep = np.setdiff1d(np.arange(sysm.L), np.asarray(lines, int))
    A = sp.coo_matrix((np.ones(len(keep)), (sysm.f_bus[keep], sysm.t_bus[keep])), shape=(sysm.n_bus, sysm.n_bus))
    return connected_components(A, directed=False)[0] == 1


def scenario_to_dict(sc: OODScenario, extra=None):
    out = dict(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, day=sc.day, start=sc.start,
               g_out=np.array(sc.g_out, int), l_out=np.array(sc.l_out, int))
    out.update(extra or {})
    return out


# ============================================================================ the UC model with outages
class OODModel(UCModel):
    """UCModel whose bounds also apply the scenario's unit and line outages (fields g_out / l_out of an
    OODScenario; plain UCScenario objects are unaffected). Every LP / MILP built from _rhs_bounds - the full and
    reduced MILPs, the LP relaxation, the dispatch LP, the LP-relaxation guard - therefore sees the outages.
    Unit out: u = v = 0 in every period (a given commitment u_fix is overridden to 0 for that unit).
    Line out: its flow is fixed to 0 and its angle equation f = b (theta_i - theta_j) is dropped (row made free)."""

    def __init__(self, sysm, T=12, network=True):
        super().__init__(sysm, T, network)
        self.ang_row = np.full((T, sysm.L), -1, int)
        if network:
            A = self.A.tocsc()
            th0, nth = self.off["th"]
            for t in range(T):
                for l in range(sysm.L):
                    col = self.ix("f", t, l)
                    rows = A.indices[A.indptr[col]:A.indptr[col + 1]]
                    for r in rows:
                        cols = self.A.indices[self.A.indptr[r]:self.A.indptr[r + 1]]
                        if np.any((cols >= th0) & (cols < th0 + nth * T)):
                            self.ang_row[t, l] = r
            assert (self.ang_row >= 0).all()

    def _rhs_bounds(self, sc, u_fix=None, relax=False):
        g_out = tuple(getattr(sc, "g_out", ()) or ())
        l_out = tuple(getattr(sc, "l_out", ()) or ())
        if u_fix is not None and g_out:
            u_fix = np.array(u_fix, float).reshape(self.T, -1).copy()
            u_fix[:, list(g_out)] = 0.0
        lo, hi, lb, ub = super()._rhs_bounds(sc, u_fix, relax)
        for g in g_out:
            for t in range(self.T):
                for name in ("u", "v"):
                    i = self.ix(name, t, g)
                    lb[i] = ub[i] = 0.0
        for l in l_out:
            for t in range(self.T):
                i = self.ix("f", t, l)
                lb[i] = ub[i] = 0.0
                r = self.ang_row[t, l]
                lo[r], hi[r] = -np.inf, np.inf
        return lo, hi, lb, ub


def solve_milp(m: UCModel, sc, fix=None, time_limit=60.0, gap=1e-3):
    """(reduced) MILP with HiGHS through highspy, one thread. fix: {(t, g): 0/1}. Times the whole call (model
    passing included) the same way for the full and the reduced problems."""
    lo, hi, lb, ub = m._rhs_bounds(sc)
    for (t, g), v in (fix or {}).items():
        i = m.ix("u", t, g)
        lb[i] = ub[i] = v
    t0 = time.time()
    r = highs_solve(m.c, m.A, lo, hi, lb, ub, m.integ, time_limit, gap)
    dt = time.time() - t0
    out = dict(feasible=r["x"] is not None, obj=float(r["obj"]), bound=float(r["bound"]), time=float(dt),
               status=r["status"], mip_gap=float(r["gap"]) if np.isfinite(r["gap"]) else None)
    if r["x"] is not None:
        x, T = r["x"], m.T
        get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T]
        out.update(shed=float(get("shed").sum()), spill=float(get("spill").sum()), short=float(get("short").sum()),
                   u=np.round(get("u")).reshape(T, -1).astype(np.int8))
    return out


# ============================================================================ overrides for methods that do not see the shift
def avail_view(sysm, g_out):
    """System data with unavailable units reduced to (numerically) zero capacity and placed last in every merit
    order: for the solver-free guards and repairs (otsl.combo.adequacy_guard, otsl.constrained.adequacy_repair_blocks),
    which then count only the available fleet. The original object is not modified."""
    if not g_out:
        return sysm
    s = copy.copy(sysm)
    s.pmax = sysm.pmax.copy()
    s.pmin = sysm.pmin.copy()
    s.pmax[list(g_out)] = 1e-9
    s.pmin[list(g_out)] = 0.0
    return s


def override_probs(p, g_out):
    """on-probabilities of unavailable units set to 0 (the learned models do not see availability)"""
    if not g_out:
        return p
    p = np.array(p, copy=True)
    p[..., list(g_out)] = 0.0
    return p


def force_out(fix, g_out, T):
    """fixings of unavailable units forced to OFF (after the guards, which may have released them)"""
    fix = dict(fix)
    for g in g_out:
        for t in range(T):
            fix[(t, int(g))] = 0
    return fix


def fixed_share(fix, g_out, T, G):
    """share of the decisions of available units that are fixed (outaged units are excluded from both counts)"""
    n = sum(1 for (t, g) in fix if g not in set(g_out))
    return n / (T * (G - len(g_out)))


# ============================================================================ GNN with the actual topology
@torch.no_grad()
def predict_gated(m1, d, gate):
    """CommitGNN probabilities with an edge gate [B, L] (0 = line out): messages over out-of-service lines are
    switched off (EdgeGNN supports the gate; the trained models were run with all gates = 1)."""
    net = m1.net
    net.eval()
    xb, xg, xe = m1.feat(d)
    hg = net.gen_enc(xg)
    agg = torch.zeros(xb.shape[0], net.N, hg.shape[-1]).index_add_(1, net.gbus, hg)
    h, _ = net.gnn(torch.cat([xb, agg], -1), xe, edge_gate=torch.as_tensor(gate, dtype=torch.float32))
    lg = net.dec(torch.cat([h[:, net.gbus], hg], -1)).transpose(1, 2)
    return torch.sigmoid(lg).numpy()


# ============================================================================ which lines matter
def critical_lines(sysm, m: OODModel, scenarios, thr=0.1):
    """Lines whose single outage raises the LP relaxation cost by more than thr % on at least one of the given
    (training) scenarios, excluding bridges. Returns (pool, per-line max increase %, base costs)."""
    cand = [l for l in range(sysm.L) if connected_without(sysm, [l])]
    inc = np.zeros((len(scenarios), sysm.L))
    base = []
    for k, sc in enumerate(scenarios):
        c0 = m.solve_dispatch(sc, None, relax=True).obj
        base.append(c0)
        for l in cand:
            sc_l = OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, l_out=(l,))
            inc[k, l] = (m.solve_dispatch(sc_l, None, relax=True).obj - c0) / c0 * 100
    mx = inc.max(0)
    pool = [l for l in cand if mx[l] > thr]
    return pool, inc, base


# ============================================================================ follow-up: LP-relaxation veto on OFF fixings
def rarely_on_units(train_u, freq=0.01):
    """units that are on in less than `freq` of the training unit-hours (MILP labels) - a training statistic"""
    return np.where(np.asarray(train_u).mean((0, 1)) < freq)[0]


def lp_off_veto(fix, u_rel, tol=1e-3, units=None, unit_level=False):
    """Release OFF fixings that the instance's own LP relaxation contradicts (a cheap post-hoc fix for thresholds that
    collapsed on units the validation set never needed).
    unit_level=False ("unit-hour" variant): drop the OFF fixing of (t, g) whenever u_rel[t, g] > tol.
    unit_level=True  ("unit" variant): drop every OFF fixing of unit g if u_rel[t, g] > tol in any hour of the horizon.
    units: restrict the veto to these units (e.g. rarely_on_units); None = all units. Returns (fix, released)."""
    u = np.asarray(u_rel, float)
    allowed = None if units is None else set(int(g) for g in units)
    hot = u > tol
    unit_hot = hot.any(0)
    out, rel = {}, 0
    for (t, g), v in fix.items():
        if v == 0 and (allowed is None or g in allowed) and (unit_hot[g] if unit_level else hot[t, g]):
            rel += 1
            continue
        out[(t, g)] = v
    return out, rel
