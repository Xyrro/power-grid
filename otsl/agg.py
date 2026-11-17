"""Identical units aggregated: the screened UC with one integer count per group of identical units (docs/methods/agg.md).

On RTS-GMLC 56 of the 73 thermal units form 24 groups of exact copies (same bus, same costs, limits and minimum
times: UCSystem.identical_groups). Their schedules can be permuted without changing cost or flows, so the MILP has
many symmetric optima and branch-and-bound has to raise the bound in every permuted copy of a subtree, and a learned
model trained on one of the tied optima sees arbitrary labels (docs/RESEARCH.md U1, V1).

AggScreenedUC solves the same problem as otsl.screen.ScreenedUC (copper plate + lazily added PTDF line limits) with,
for every group of n identical units, one set of columns (U_t, V_t, W_t in {0, ..., n}, the group's segment outputs and
reserve) instead of n. Each group row is the sum of its members' rows (logic U_t - U_{t-1} = V_t - W_t, min up
sum V <= U, min down sum W <= n - U, segments, reserve headroom); system rows (balance, reserve requirement, PTDF line
rows: the members share a bus) are unchanged. Exact (Knueven, Ostrowski & Watson 2018, IEEE TPS 33(4)): summing the
members' rows makes it a relaxation, so its bound is valid for the per-unit problem, and every integer solution
disaggregates into member schedules of the same cost (disaggregate: stop the units on longest, start the units off
longest; equal split of output and reserve), so nothing is lost either. Initial states need no grouping: the model
assumes every initial unit has met its minimum times (otsl.uc.UCModel), so the count of units on at t = 0 is enough.
Units out of service (OODScenario.g_out) are taken out of their groups and kept as single units (bounds 0).

Fixings are given per unit, as in the per-unit pipeline, and become bounds on counts: in hour t a group's count lies
between the members fixed on and n minus the members fixed off. count_fixings adds the rule that symmetry makes
possible: a group's count is fixed to round(sum of its members' probabilities) when that sum is within delta of an
integer, even when the members' own probabilities are near 1/2 (the model cannot tell copies apart, but it knows how
many are needed).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from .ltfx import highs_solve
from .ood import OODModel

UNIT_NAMES = ("u", "v", "w", "ps", "res")


class AggScreenedUC:
    """Screened UC with identical units aggregated into integer counts; same interface as otsl.screen.ScreenedUC
    (solve returns per-unit schedules)."""

    def __init__(self, sysm, T=12, groups=None):
        self.s, self.T = sysm, T
        self.mc = OODModel(sysm, T=T, network=False)
        self.base_groups = [list(g) for g in (sysm.identical_groups() if groups is None else groups)]
        mc, s = self.mc, sysm
        G, S, R, N = s.G, s.seg_w.shape[1], len(s.r_pmax), s.n_bus
        owner = np.full(mc.nv, -1, int)            # unit owning each column (-1: system column)
        kind = np.full(mc.nv, "", object)
        for name in UNIT_NAMES:
            o, k = mc.off[name]
            for t in range(T):
                for j in range(k):
                    owner[o + t * k + j] = j // S if name == "ps" else j
                    kind[o + t * k + j] = name
        self.owner, self.kind = owner, kind
        rows, cols, vals = [], [], []
        inj = []
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
            inj.append(sp.csr_matrix((vals, (rows, cols)), shape=(N, mc.nv)))
        self.inj_full = inj
        A = mc.A.tocsr()
        self.row_units = []                        # units touched by each row
        for r in range(A.shape[0]):
            cs = A.indices[A.indptr[r]:A.indptr[r + 1]]
            self.row_units.append(set(int(owner[c]) for c in cs if owner[c] >= 0))
        wcols = np.zeros(mc.nv, bool)
        o, k = mc.off["w"]
        wcols[o:o + k * T] = True
        self.mindn_row = np.array([len(self.row_units[r]) == 1 and mc.hi0[r] == 1.0 and
                                   wcols[A.indices[A.indptr[r]:A.indptr[r + 1]]].any() for r in range(A.shape[0])])
        self._built = {}
        from .screen import ScreenedUC
        self._ptdf_src = ScreenedUC.__new__(ScreenedUC)   # reuse its PTDF cache only
        self._ptdf_src.s, self._ptdf_src._ptdf = sysm, {}

    # ------------------------------------------------------------------ model for a set of units out
    def groups_for(self, g_out=()):
        out = set(int(g) for g in g_out)
        grp = []
        for gr in self.base_groups:
            keep = [g for g in gr if g not in out]
            if len(keep) > 1:
                grp.append(keep)
        return grp

    def model(self, g_out=()):
        key = tuple(sorted(int(g) for g in g_out))
        if key in self._built:
            return self._built[key]
        mc, s, T = self.mc, self.s, self.T
        groups = self.groups_for(key)
        rep = np.arange(s.G)                       # representative of each unit
        size = np.ones(s.G, int)
        for gr in groups:
            rep[gr] = gr[0]
            size[gr[0]] = len(gr)
        nonrep = np.array([rep[g] != g for g in range(s.G)])
        keep_col = ~np.array([self.owner[c] >= 0 and nonrep[self.owner[c]] for c in range(mc.nv)])
        A = mc.A.tocsr()
        keep_row = np.array([not (len(u) == 1 and nonrep[next(iter(u))]) for u in self.row_units])
        Ak = A[keep_row][:, keep_col].tocsr()
        new_of = -np.ones(mc.nv, int)
        new_of[keep_col] = np.arange(keep_col.sum())
        # columns of the per-unit model that each aggregated column sums
        members = {g: [g] for g in range(s.G)}
        for gr in groups:
            members[gr[0]] = list(gr)
        agg = sp.lil_matrix((int(keep_col.sum()), mc.nv))
        for c in np.where(keep_col)[0]:
            g = self.owner[c]
            if g < 0:
                agg[new_of[c], c] = 1.0
                continue
            name = self.kind[c]
            o, k = mc.off[name]
            t, j = divmod(c - o, k)
            for mbr in members[g]:
                jj = mbr * s.seg_w.shape[1] + j % s.seg_w.shape[1] if name == "ps" else mbr
                agg[new_of[c], o + t * k + jj] = 1.0
        integ = mc.integ[keep_col]
        hi_scale = np.ones(A.shape[0])
        for r in np.where(self.mindn_row)[0]:
            g = next(iter(self.row_units[r]))
            hi_scale[r] = size[g]
        b = dict(groups=groups, rep=rep, size=size, members=members, keep_col=keep_col, keep_row=keep_row, A=Ak,
                 c=mc.c[keep_col], integ=integ, new_of=new_of, agg=agg.tocsr(), mindn_hi=hi_scale[keep_row],
                 logic0_rows={g: int(np.cumsum(keep_row)[mc.row_tag[("logic0", g)][0]] - 1)
                              for g in range(s.G) if not nonrep[g]},
                 inj=[m[:, keep_col].tocsr() for m in self.inj_full])
        self._built[key] = b
        return b

    # ------------------------------------------------------------------ bounds, fixings, mapping
    def bounds(self, sc, b, fix=None):
        """row and column bounds of the aggregated model for a scenario; fix {(t, g): 0/1} per unit -> count bounds"""
        mc, s, T = self.mc, self.s, self.T
        sc_c = replace(sc, l_out=()) if getattr(sc, "l_out", None) else sc
        lo, hi, lb, ub = mc._rhs_bounds(sc_c)
        lo, hi = lo[b["keep_row"]], hi[b["keep_row"]]
        hi = np.where(np.isfinite(hi) & (hi == 1.0) & (b["mindn_hi"] > 1), b["mindn_hi"], hi)
        lb_a = b["agg"] @ lb
        ub_a = b["agg"] @ ub
        for g, r in b["logic0_rows"].items():
            lo[r] = hi[r] = float(sum(int(sc.u0[m]) for m in b["members"][g]))
        if fix:
            on = {}
            off = {}
            for (t, g), v in fix.items():
                d = on if v else off
                d[(t, int(b["rep"][g]))] = d.get((t, int(b["rep"][g])), 0) + 1
            uo = self.mc.off["u"][0]
            for key in set(on) | set(off):
                t, g = key
                i = b["new_of"][uo + t * s.G + g]
                lb_a[i] = max(lb_a[i], on.get(key, 0))
                ub_a[i] = min(ub_a[i], b["size"][g] - off.get(key, 0))
        return lo, hi, lb_a, ub_a

    def count_bounds(self, sc, b, cfix):
        """cfix {(t, rep): count} -> column bound overrides"""
        uo, G = self.mc.off["u"][0], self.s.G
        return {int(b["new_of"][uo + t * G + g]): float(k) for (t, g), k in cfix.items()}

    def to_agg(self, x, b):
        return b["agg"] @ x

    def ptdf(self, l_out=()):
        from .screen import ScreenedUC
        return ScreenedUC.ptdf(self._ptdf_src, l_out)

    def flows(self, xa, sc, b):
        H = self.ptdf(getattr(sc, "l_out", ()) or ())
        return np.stack([H @ (b["inj"][t] @ xa - sc.load[t]) for t in range(self.T)])

    def _rows(self, sc, mon, b):
        H = self.ptdf(getattr(sc, "l_out", ()) or ())
        if not mon:
            return sp.csr_matrix((0, b["A"].shape[1])), np.zeros(0), np.zeros(0)
        R = sp.vstack([sp.csr_matrix(H[l][None, :]) @ b["inj"][t] for t, l in mon], format="csr")
        off = np.array([H[l] @ sc.load[t] for t, l in mon])
        fm = self.s.fmax[[l for _, l in mon]]
        return R, off - fm, off + fm

    def violated(self, xa, sc, b, tol=1e-6):
        f = self.flows(xa, sc, b)
        f[:, list(getattr(sc, "l_out", ()) or ())] = 0.0
        return {(int(t), int(l)) for t, l in np.argwhere(np.abs(f) > self.s.fmax[None, :] * (1 + tol) + 1e-7)}

    def counts(self, xa, b):
        """[T, G] integer count per representative (0 for non-representatives)"""
        o = self.mc.off["u"][0]
        G, T = self.s.G, self.T
        U = np.zeros((T, G), int)
        for g in range(G):
            if b["rep"][g] == g:
                for t in range(T):
                    U[t, g] = int(round(xa[b["new_of"][o + t * G + g]]))
        return U

    def disaggregate(self, xa, sc, b):
        """per-unit schedule [T, G] of an aggregated solution: in every hour a group stops its units that have been on
        longest and starts those off longest (minimum times then hold whenever the counts satisfy the group rows)"""
        s, T = self.s, self.T
        U = self.counts(xa, b)
        u = np.zeros((T, s.G), np.int8)
        for g in range(s.G):
            if b["rep"][g] != g:
                continue
            mem = b["members"][g]
            state = {m: int(sc.u0[m]) for m in mem}
            age = {m: 10 ** 6 for m in mem}              # every initial unit has met its minimum times
            for t in range(T):
                on = [m for m in mem if state[m]]
                off = [m for m in mem if not state[m]]
                d = U[t, g] - len(on)
                if d > 0:
                    for m in sorted(off, key=lambda m: -age[m])[:d]:
                        state[m], age[m] = 1, 0
                elif d < 0:
                    for m in sorted(on, key=lambda m: -age[m])[:-d]:
                        state[m], age[m] = 0, 0
                for m in mem:
                    age[m] += 1
                    u[t, m] = state[m]
        return u

    def relax_mask(self, sc, b, fix):
        """columns of the aggregated model whose integrality relax() drops: U, V, W of every (hour, group) whose members
        are all fixed (per-unit fixings {(t, g): 0/1})"""
        mc, G = self.mc, self.s.G
        mask = np.zeros(b["A"].shape[1], bool)
        for g in range(G):
            if b["rep"][g] != g:
                continue
            for t in range(self.T):
                if all((t, m) in fix for m in b["members"][g]):
                    for name in ("u", "v", "w"):
                        mask[b["new_of"][mc.ix(name, t, g)]] = True
        return mask

    def solve(self, sc, mon, fix=None, gap=5e-3, time_limit=60.0, start=None, max_rounds=50, cfix=None, opts=None,
              relax=None, target=None):
        """Reduced (fix: per-unit {(t, g): 0/1}; cfix: per-group counts {(t, rep): k}) or full MILP with lazy line
        limits on the aggregated model. start: an aggregated solution vector. relax: boolean mask of aggregated columns
        made continuous (a relaxation: its bound is valid for the full problem; one round, no line check, x is None
        unless every column is integral). target: stop as soon as the dual bound reaches it. Returns dict(x
        (aggregated), u (per unit), obj, bound, time, rounds, n_mon, mon, ok, status)."""
        t0 = time.perf_counter()
        b = self.model(getattr(sc, "g_out", ()) or ())
        lo, hi, lb, ub = self.bounds(sc, b, fix)
        for i, k in (self.count_bounds(sc, b, cfix) if cfix else {}).items():
            k = min(max(k, lb[i]), ub[i])
            lb[i] = ub[i] = k
        integ = b["integ"] if relax is None else np.where(relax, 0.0, b["integ"])
        hook = None
        if target is not None:
            def hook(h, target=target):
                def cb(e):
                    if e.data_out.mip_dual_bound >= target:
                        e.data_in.user_interrupt = True
                h.cbMipInterrupt.subscribe(cb)
        mon, rounds, x0, r = set(mon), 0, start, None
        while True:
            R, rlo, rhi = self._rows(sc, sorted(mon), b)
            left = max(time_limit - (time.perf_counter() - t0), 1.0)
            r = highs_solve(b["c"], sp.vstack([b["A"], R], format="csr"), np.r_[lo, rlo], np.r_[hi, rhi], lb, ub,
                            integ, left, gap, start=x0, opts=opts, hook=hook)
            if relax is not None:
                rounds += 1
                xi = r["x"]
                if xi is not None and np.any(np.abs(xi[b["integ"] > 0] - np.round(xi[b["integ"] > 0])) > 1e-6):
                    xi = None
                return dict(x=xi, u=None, obj=float(r["obj"]), bound=float(r["bound"]), time=time.perf_counter() - t0,
                            rounds=rounds, n_mon=len(mon), mon=mon, ok=False, status=r["status"])
            rounds += 1
            if r["x"] is None:
                return dict(x=None, u=None, obj=np.inf, bound=r["bound"], time=time.perf_counter() - t0, rounds=rounds,
                            n_mon=len(mon), mon=mon, ok=False, status=r["status"])
            new = self.violated(r["x"], sc, b)
            if not new or rounds >= max_rounds:
                break
            mon |= new
            x0 = r["x"]
        return dict(x=r["x"], u=self.disaggregate(r["x"], sc, b), obj=float(r["obj"]), bound=float(r["bound"]),
                    time=time.perf_counter() - t0, rounds=rounds, n_mon=len(mon), mon=mon, ok=not new, status=r["status"])


def count_fixings(p, agg: AggScreenedUC, sc, delta, delta_c=None, fix=None):
    """Per-group count fixings {(t, rep): k} from per-unit probabilities p [T, G]: in hour t a group's count is fixed to
    round(sum of p over its members) when the sum is within delta_c (default delta) of that integer. Hours where every
    member is already fixed by the per-unit rule (fix) are skipped."""
    delta_c = delta if delta_c is None else delta_c
    b = agg.model(getattr(sc, "g_out", ()) or ())
    out = {}
    for gr in b["groups"]:
        for t in range(agg.T):
            if fix is not None and all((t, m) in fix for m in gr):
                continue
            e = float(np.sum(p[t, gr]))
            k = int(round(e))
            if abs(e - k) <= delta_c:
                out[(t, gr[0])] = k
    return out


def agg_certified_pipeline(agg: AggScreenedUC, sc, p, rel, delta, delta_c=None, eps=5e-3, eps_lp=None,
                           time_limit=60.0, fix=None, counts=True, opts=None, probe=False, proof_opts=None,
                           red_eps=None, ladder=None):
    """The certified pipeline (otsl.screen.certified_pipeline) on the aggregated model: per-unit confidence fixings
    (one threshold delta, rows min up/down cannot complete released) become count bounds; with counts=True every
    (hour, group) not wholly fixed is also fixed to round(sum p) when that sum is within delta_c of an integer
    (count_fixings). Reduced aggregated MILP on the screened network -> certificate: the LP relaxation's bound within
    eps_lp (default eps), else the aggregated screened full MILP at eps warm-started from the reduced solution, keeping
    the better. If the count fixings make the reduced MILP infeasible (minimum times), they are dropped and it is
    re-solved. probe=True: the proof step also fixes every decision Lagrangian probing at the LP relaxation's prices
    (rel.lmp, rel.mu; otsl.probe) shows no schedule cheaper than the candidate can change (exact). proof_opts: HiGHS
    options of the proof step only (default opts). red_eps: the reduced MILP's gap (default eps). ladder: a second,
    looser threshold; when the LP bound does not certify the first reduced schedule, the reduced MILP is solved again
    with only the decisions min(p, 1 - p) <= ladder fixed (warm-started; the first schedule satisfies them) before
    the proof step. Returns (u per unit, info)."""
    from .screen import confidence_fixings, monitored_from_lp
    t0 = time.perf_counter()
    if fix is None:
        fix = confidence_fixings(p, agg.s, sc.u0, delta)
    fix = dict(fix)
    for g in getattr(sc, "g_out", ()) or ():
        for t in range(agg.T):
            fix[(t, int(g))] = 0
    cfix = count_fixings(p, agg, sc, delta, delta_c, fix) if counts else {}
    mon = monitored_from_lp(rel.flow, agg.s.fmax)
    reps = eps if red_eps is None else red_eps
    cand = agg.solve(sc, mon, fix=fix, gap=reps, time_limit=time_limit, cfix=cfix, opts=opts)
    info = dict(n_fixed=len(fix), n_cfix=len(cfix), cfix_dropped=False)
    if cand["x"] is None and cfix:
        cand = agg.solve(sc, mon, fix=fix, gap=reps, time_limit=time_limit, opts=opts)
        info["cfix_dropped"] = True
    if cand["x"] is None:
        cand = agg.solve(sc, mon, gap=eps, time_limit=time_limit, opts=opts)
        info["fallback"] = True
    info.update(red_obj=cand["obj"], red_rounds=cand["rounds"], red_s=time.perf_counter() - t0, n_mon=cand["n_mon"],
                u_red=cand["u"])
    C = cand["obj"]
    lp_gap = (C - rel.obj) / C
    t1 = time.perf_counter()
    if lp_gap <= (eps if eps_lp is None else eps_lp):
        info.update(cert="lp", lp_gap=float(lp_gap), cert_s=time.perf_counter() - t1)
        return cand["u"], info
    if ladder is not None:                       # second rung: fewer fixings, warm-started from the first schedule
        fix2 = confidence_fixings(p, agg.s, sc.u0, ladder)
        for g in getattr(sc, "g_out", ()) or ():
            for t in range(agg.T):
                fix2[(t, int(g))] = 0
        c2 = agg.solve(sc, set(mon) | set(cand["mon"]), fix=fix2, gap=reps, time_limit=time_limit, start=cand["x"],
                       opts=opts)
        info.update(n_fixed2=len(fix2), red2_obj=c2["obj"], red2_s=time.perf_counter() - t1)
        if c2["x"] is not None and c2["obj"] < C:
            cand, C = c2, c2["obj"]
            lp_gap = (C - rel.obj) / C
            if lp_gap <= (eps if eps_lp is None else eps_lp):
                info.update(cert="lp2", lp_gap=float(lp_gap), cert_s=time.perf_counter() - t1)
                return cand["u"], info
    exact = None
    if probe:
        from .probe import probe_fixings
        exact, _, _ = probe_fixings(agg.s, sc, rel.lmp, rel.mu, rel.obj, C)
        info["n_exact"] = len(exact)
    z = agg.solve(sc, set(mon) | set(cand["mon"]), fix=exact, gap=eps, time_limit=time_limit, start=cand["x"],
                  opts=opts if proof_opts is None else proof_opts)
    u = z["u"] if z["x"] is not None and z["obj"] <= C else cand["u"]
    info.update(cert="milp", lp_gap=float(lp_gap), cert_s=time.perf_counter() - t1, cert_obj=z["obj"],
                cert_bound=z["bound"], cert_rounds=z["rounds"], cert_status=z.get("status"))
    return u, info


def sandwich_pipeline(agg: AggScreenedUC, sc, p, rel, delta, eps=5e-3, time_limit=60.0, fix=None, keep_lp_frac=False,
                      opts=None):
    """Restrict the confident decisions for a schedule, relax them for a bound. The per-unit confidence fixings F
    (one threshold delta) give the reduced aggregated MILP R(F) (a restriction: its schedule's cost C is an upper bound)
    and the aggregated MILP with the integrality of F's (hour, group) counts dropped, Q(F) (a relaxation: integer only
    where the model is unsure, so its branching is confined to those decisions; its bound is valid for the full problem).
    Certificate, cheapest first: the LP relaxation's bound within eps of C; else Q(F)'s bound, solved until it reaches
    C (1 - eps); else the aggregated screened full MILP at eps warm-started from R(F)'s schedule, keeping the better.
    keep_lp_frac: also keep integrality where the LP relaxation is fractional. Returns (u per unit, info)."""
    from .screen import confidence_fixings, monitored_from_lp
    t0 = time.perf_counter()
    if fix is None:
        fix = confidence_fixings(p, agg.s, sc.u0, delta)
    fix = dict(fix)
    for g in getattr(sc, "g_out", ()) or ():
        for t in range(agg.T):
            fix[(t, int(g))] = 0
    mon = monitored_from_lp(rel.flow, agg.s.fmax)
    cand = agg.solve(sc, mon, fix=fix, gap=eps, time_limit=time_limit, opts=opts)
    info = dict(n_fixed=len(fix))
    if cand["x"] is None:
        cand = agg.solve(sc, mon, gap=eps, time_limit=time_limit, opts=opts)
        info["fallback"] = True
    info.update(red_obj=cand["obj"], red_rounds=cand["rounds"], red_s=time.perf_counter() - t0, n_mon=cand["n_mon"],
                u_red=cand["u"])
    C = cand["obj"]
    lp_gap = (C - rel.obj) / C
    t1 = time.perf_counter()
    info["lp_gap"] = float(lp_gap)
    if lp_gap <= eps:
        info.update(cert="lp", cert_s=time.perf_counter() - t1)
        return cand["u"], info
    b = agg.model(getattr(sc, "g_out", ()) or ())
    qfix = fix
    if keep_lp_frac:
        frac = np.abs(rel.u - np.round(rel.u)) > 1e-6
        qfix = {k: v for k, v in fix.items() if not frac[k]}
    mask = agg.relax_mask(sc, b, qfix)
    mon2 = set(mon) | set(cand["mon"])
    target = C * (1 - eps)
    q = agg.solve(sc, mon2, gap=1e-4, time_limit=time_limit, start=cand["x"], opts=opts, relax=mask, target=target)
    info.update(q_s=time.perf_counter() - t1, q_bound=q["bound"], q_obj=q["obj"], q_status=q["status"],
                n_relaxed=int(mask.sum()))
    if q["bound"] >= target * (1 - 1e-9):
        info.update(cert="sandwich", cert_s=time.perf_counter() - t1)
        return cand["u"], info
    z = agg.solve(sc, mon2, gap=eps, time_limit=max(time_limit - (time.perf_counter() - t1), 1.0), start=cand["x"],
                  opts=opts)
    u = z["u"] if z["x"] is not None and z["obj"] <= C else cand["u"]
    info.update(cert="milp", cert_s=time.perf_counter() - t1, cert_obj=z["obj"], cert_bound=z["bound"],
                cert_rounds=z["rounds"], cert_status=z.get("status"))
    return u, info
