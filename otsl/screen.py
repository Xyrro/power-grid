"""Screened network and certified solves for network-constrained UC (docs/methods/screen.md).

Screened network. On RTS-GMLC only a handful of the T x L line limits bind at the optimum (0-25 of 1,440 on 12-hour
validation instances), yet the B-theta model (otsl.uc.UCModel) carries every bus balance, angle and flow in every LP
the MILP solver touches. ScreenedUC solves the same UC on a copper plate (UCModel(network=False): the same columns,
one balance row per hour) plus PTDF rows for a monitored set of (hour, line) pairs only, computes every line's flow of
the solution, adds the violated pairs and re-solves (warm-started), until no line is overloaded. The final schedule
is feasible for the full network; the solve is a relaxation of the full problem, so its dual bound is a valid bound for
the full problem and its relative gap holds for it too. Nothing is approximated: the screen only decides how many
rounds are needed. The monitored set starts from the lines the LP relaxation of the full network loads to >= 99 % of
their rating (monitored_from_lp). Xavier, Qiu & Ahmed (2021, INFORMS J. Comput.) use the same lazy transmission
constraints, with a learned initial set.

Certified solve. certified() takes a candidate schedule (e.g. a reduced MILP's) and returns a schedule whose cost is
proven within eps of the optimum: if the LP relaxation's bound already certifies the candidate, (C - z_LP) / C <= eps,
it is returned as is; otherwise the screened full MILP at gap eps, warm-started from it, decides.

PTDF: f = PTDF @ inj with inj_n = sum of thermal output, renewables, shed - spill - load at bus n (the B-theta model's
bus balance, otsl.uc.UCModel._build), the reference bus' column zero. Lines out of service (OODScenario.l_out) are
removed from the PTDF (never islanding: otsl.ood draws outages that keep the network connected); units out
(g_out) are handled by the copper model's bounds (otsl.ood.OODModel).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from .ltfx import highs_solve
from .ood import OODModel


class ScreenedUC:
    """UC MILP on a copper plate plus lazily added PTDF line limits; exact for the network model."""

    def __init__(self, sysm, T=12):
        self.s, self.T = sysm, T
        self.mc = OODModel(sysm, T=T, network=False)
        s, mc = sysm, self.mc
        G, S, R, N = s.G, s.seg_w.shape[1], len(s.r_pmax), s.n_bus
        self.inj = []                       # per hour: [N, nv] map from the model's columns to bus injections
        for t in range(T):
            rows, cols, vals = [], [], []
            for g in range(G):
                rows.append(s.gbus[g]); cols.append(mc.ix("u", t, g)); vals.append(s.pmin[g])
                for k in range(S):
                    rows.append(s.gbus[g]); cols.append(mc.ix("ps", t, g * S + k)); vals.append(1.0)
            for j in range(R):
                rows.append(s.r_bus[j]); cols.append(mc.ix("r", t, j)); vals.append(1.0)
            for n in range(N):
                rows += [n, n]; cols += [mc.ix("shed", t, n), mc.ix("spill", t, n)]; vals += [1.0, -1.0]
            self.inj.append(sp.csr_matrix((vals, (rows, cols)), shape=(N, mc.nv)))
        self._ptdf = {}

    def ptdf(self, l_out=()):
        key = tuple(sorted(int(l) for l in l_out))
        if key not in self._ptdf:
            s = self.s
            b = s.b.copy()
            b[list(key)] = 0.0
            A = s.A
            Bf = b[:, None] * A
            Bbus = A.T @ Bf
            nr = [n for n in range(s.n_bus) if n != s.ref]
            X = np.zeros((s.n_bus, s.n_bus))
            X[np.ix_(nr, nr)] = np.linalg.inv(Bbus[np.ix_(nr, nr)])
            self._ptdf[key] = Bf @ X
        return self._ptdf[key]

    def flows(self, x, sc):
        """[T, L] line flows of a solution x"""
        H = self.ptdf(getattr(sc, "l_out", ()) or ())
        return np.stack([H @ (self.inj[t] @ x - sc.load[t]) for t in range(self.T)])

    def _rows(self, sc, mon):
        H = self.ptdf(getattr(sc, "l_out", ()) or ())
        if not mon:
            return sp.csr_matrix((0, self.mc.nv)), np.zeros(0), np.zeros(0)
        R = sp.vstack([sp.csr_matrix(H[l][None, :]) @ self.inj[t] for t, l in mon], format="csr")
        off = np.array([H[l] @ sc.load[t] for t, l in mon])
        fm = self.s.fmax[[l for _, l in mon]]
        return R, off - fm, off + fm

    def violated(self, x, sc, tol=1e-6):
        f = self.flows(x, sc)
        f[:, list(getattr(sc, "l_out", ()) or ())] = 0.0
        return {(int(t), int(l)) for t, l in np.argwhere(np.abs(f) > self.s.fmax[None, :] * (1 + tol) + 1e-7)}

    def solve(self, sc, mon, fix=None, gap=5e-3, time_limit=60.0, start=None, max_rounds=50):
        """Reduced (fix: {(t, g): 0/1}) or full MILP with lazy line limits. Returns dict(x, u, obj, bound, time,
        rounds, n_mon, mon, ok); time covers every round, row building and flow checks included."""
        t0 = time.perf_counter()
        sc_c = replace(sc, l_out=()) if getattr(sc, "l_out", None) else sc     # outaged lines: in the PTDF only
        lo, hi, lb, ub = self.mc._rhs_bounds(sc_c)
        uo = self.mc.off["u"][0]
        for (t, g), v in (fix or {}).items():
            i = uo + t * self.s.G + g
            lb[i] = ub[i] = v
        mon, rounds, x0, r = set(mon), 0, start, None
        while True:
            R, rlo, rhi = self._rows(sc, sorted(mon))
            left = max(time_limit - (time.perf_counter() - t0), 1.0)
            r = highs_solve(self.mc.c, sp.vstack([self.mc.A, R], format="csr"), np.r_[lo, rlo], np.r_[hi, rhi], lb, ub,
                            self.mc.integ, left, gap, start=x0)
            rounds += 1
            if r["x"] is None:
                return dict(x=None, u=None, obj=np.inf, bound=r["bound"], time=time.perf_counter() - t0, rounds=rounds,
                            n_mon=len(mon), mon=mon, ok=False, status=r["status"])
            new = self.violated(r["x"], sc)
            if not new or rounds >= max_rounds:
                break
            mon |= new
            x0 = r["x"]
        return dict(x=r["x"], u=self.u_of(r["x"]), obj=float(r["obj"]), bound=float(r["bound"]),
                    time=time.perf_counter() - t0, rounds=rounds, n_mon=len(mon), mon=mon, ok=not new, status=r["status"])

    def u_of(self, x):
        o = self.mc.off["u"][0]
        return np.round(x[o:o + self.T * self.s.G]).reshape(self.T, self.s.G).astype(np.int8)


def monitored_from_lp(flow_rel, fmax, alpha=0.99):
    """(hour, line) pairs the LP relaxation of the full network loads to >= alpha of the rating"""
    return {(int(t), int(l)) for t, l in np.argwhere(np.abs(flow_rel) >= alpha * np.asarray(fmax)[None, :])}


def certified(scr: ScreenedUC, sc, cand, z_lp, eps=5e-3, eps_lp=None, time_limit=60.0, mon=()):
    """Certify or improve a candidate (a dict of ScreenedUC.solve with x and obj) to a proven gap.
    LP certificate: (C - z_LP) / C <= eps_lp (default eps) returns the candidate; otherwise the screened full MILP at
    eps, warm-started from it, keeps the better schedule. Returns (x, info)."""
    t0 = time.perf_counter()
    C = cand["obj"]
    lp_gap = (C - z_lp) / C if np.isfinite(C) else np.inf
    if lp_gap <= (eps if eps_lp is None else eps_lp):
        return cand["x"], dict(cert="lp", lp_gap=float(lp_gap), cert_s=time.perf_counter() - t0)
    z = scr.solve(sc, set(mon) | set(cand.get("mon", ())), fix=None, gap=eps, time_limit=time_limit,
                  start=cand["x"] if cand.get("x") is not None else None)
    x = z["x"] if z["x"] is not None and z["obj"] <= C else cand["x"]
    return x, dict(cert="milp", lp_gap=float(lp_gap), cert_s=time.perf_counter() - t0, cert_obj=z["obj"],
                   cert_bound=z["bound"], cert_rounds=z["rounds"], cert_status=z.get("status"))


def certified_pipeline(scr: ScreenedUC, sc, p, rel, delta, eps=5e-3, eps_lp=None, time_limit=60.0, fix=None):
    """The certified pipeline: confidence fixings of p (one threshold delta) -> reduced MILP on the screened network
    -> certificate (LP bound within eps_lp, default eps; else the screened full MILP at eps, warm-started). rel: the
    full network's LP relaxation (UCSolution with obj and flow). fix: fixings to use instead of the confidence rule.
    Returns (u, info); info has the reduced solve's schedule (u_red), cost, rounds and time, and the certificate's kind
    and time."""
    t0 = time.perf_counter()
    if fix is None:
        fix = confidence_fixings(p, scr.s, sc.u0, delta)
    fix = dict(fix)
    for g in getattr(sc, "g_out", ()) or ():                      # unavailable units: off (the bounds say so too)
        for t in range(scr.T):
            fix[(t, int(g))] = 0
    mon = monitored_from_lp(rel.flow, scr.s.fmax)
    cand = scr.solve(sc, mon, fix=fix, gap=eps, time_limit=time_limit)
    info = dict(n_fixed=len(fix), red_obj=cand["obj"], red_rounds=cand["rounds"], red_s=time.perf_counter() - t0,
                n_mon=cand["n_mon"], u_red=cand["u"])
    if cand["x"] is None:                                          # cannot happen with soft balance; kept for safety
        cand = scr.solve(sc, mon, gap=eps, time_limit=time_limit)
        info["fallback"] = True
    x, ci = certified(scr, sc, cand, rel.obj, eps=eps, eps_lp=eps_lp, time_limit=time_limit, mon=mon)
    info.update(ci)
    return scr.u_of(x), info


def confidence_fixings(p, sysm, u0, delta):
    """round(p) wherever min(p, 1 - p) <= delta, minus the rows that min up/down times make infeasible
    (otsl.fixpolicy.release_conflicting_rows). One threshold for every unit and hour."""
    from .fixpolicy import release_conflicting_rows
    p = np.asarray(p, float)
    fix = {(int(t), int(g)): int(p[t, g] > 0.5) for t, g in np.argwhere(np.minimum(p, 1 - p) <= delta)}
    fix, _ = release_conflicting_rows(fix, sysm, u0)
    return fix
