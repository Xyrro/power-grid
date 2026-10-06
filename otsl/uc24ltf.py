"""uc24ltf: faithful Learning to Fix (otsl.ltfx) on the 24-hour benchmark uc24, and a guard-aware variant.

* GuardedLtFTuner  Algorithm 1 + 2 of the paper (otsl.ltfx.LtFTuner) with our guards inside the tuning check:
                   the fixings that a threshold pair produces on an instance are first passed through the same guard
                   chain that is applied at test time -
                     adequacy guard (5 % margin, cheapest OFF fixes released first; scripts/uc_fixing.py),
                     LP-relaxation guard (lp_guard_soft = otsl.fixpolicy.lp_guard, but a no-op when the relaxed
                     reduced problem is infeasible, i.e. on a min up/down conflict: see its docstring) -
                   and the check, the relaxation MILP (ADD CUTS) and the verification all act on the guarded fixing
                   set. Min up/down conflicts are left to the LtF cuts (as in the plain method). The cuts are still conditions on the thresholds (a guarded fixing in a release set R must be
                   released by the thresholds themselves), so the master problem is unchanged.
                   Because the guards only ever release fixings, the guarded set of a threshold pair is a subset of
                   the plain one; the tuning can therefore accept tighter thresholds wherever the guards already
                   remove the harmful fixings.
* StartLtFTuner    otsl.ltfx.LtFTuner with better warm starts for the relaxation MILP of ADD CUTS (Algorithm 2).
                   On uc24 an 8 s relaxation MILP cannot improve on the existing start (the instance's optimal schedule
                   aligned to the fixings, which releases every fixing it disagrees with: 17-154 decisions in a first
                   run), so the cuts were far from minimal. Extra start candidates, each a reduced-MILP solution under the
                   cost cap (found like a check): the fixings with min up/down-conflicting generator rows released, and
                   additionally with the adequacy / soft LP-relaxation guards applied. The feasible start that
                   disagrees with the fewest fixings is used. Starts only change where HiGHS begins; the relaxation
                   MILP (min sum nu, UC constraints, cost cap, no-goods) is the paper's.
* guard_fix        the guard chain on a fixing dict (also used by the test evaluation).
* adequacy_guard   identical copy of scripts/uc_fixing.adequacy_guard (scripts are not importable from otsl).
"""
from __future__ import annotations

import time

import numpy as np

from .fixpolicy import relaxed_reduced, release_conflicting_rows
from .ltfx import LtFTuner, fix_masks
from .uc import UCScenario


def adequacy_guard(fix, load_, avail, sr, sysm, margin=0.05):
    """unfix OFF-fixed units (cheapest first) until, every hour, the units not fixed off cover
    net load + reserve with a margin. Returns the guarded fixings and the number of units released.
    (Identical to scripts/uc_fixing.adequacy_guard.)"""
    avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    order = np.argsort(avg)
    released = 0
    for t in range(load_.shape[0]):
        need = (1 + margin) * (load_[t].sum() - avail[t].sum() + sr[t])
        off = {g for (tt, g), v in fix.items() if tt == t and v == 0}
        cap = sum(sysm.pmax[g] for g in range(sysm.G) if g not in off)
        for g in order:
            if cap >= need:
                break
            if g in off:
                del fix[(t, int(g))]
                cap += sysm.pmax[g]
                released += 1
    return fix, released


def lp_guard_soft(m, sysm, sc: UCScenario, fix, max_iter=4, tol=1e-6):
    """otsl.fixpolicy.lp_guard (same release logic, copied), except when the LP relaxation of the reduced problem is
    infeasible from the start: then the fixings are returned unchanged. An infeasible relaxed reduced problem means
    a min up/down conflict among the fixings (shedding, spill and reserve shortfall are priced slacks); lp_guard
    would then mark every hour bad and release every OFF fixing (and later every fixing), which in the guard-aware
    tuning turns any over-fixed threshold pair into a "passing" one with a tiny fixed share. Leaving the conflict in
    place lets the Learning-to-Fix check fail on it and cut it off through the thresholds instead.
    Returns (fix, released, LP seconds, infeasible_at_start)."""
    fix = dict(fix)
    n0, secs = len(fix), 0.0
    for it in range(max_iter + 1):
        _, slack, dt = relaxed_reduced(m, sc, fix)
        secs += dt
        if it == 0 and not np.all(np.isfinite(slack)):
            return fix, 0, secs, True
        bad = np.where(slack > tol)[0]
        if len(bad) == 0:
            break
        if it == max_iter:
            for t in bad:
                for tt in (t - 1, t, t + 1):
                    for g in range(sysm.G):
                        fix.pop((int(tt), g), None)
            break
        for t in bad:
            for (tt, g), v in list(fix.items()):
                if v == 0 and t - sysm.min_dn[g] <= tt <= t:
                    del fix[(tt, g)]
        if it >= 1:                     # still bad after releasing OFF fixes: also ON fixes (over-generation)
            for t in bad:
                for (tt, g), v in list(fix.items()):
                    if v == 1 and t - sysm.min_up[g] <= tt <= t:
                        del fix[(tt, g)]
    return fix, n0 - len(fix), secs, False


def masks_to_fix(off, on):
    out = {(int(t), int(g)): 0 for t, g in zip(*np.where(off))}
    out.update({(int(t), int(g)): 1 for t, g in zip(*np.where(on))})
    return out


def fix_to_masks(fix, T, G):
    off, on = np.zeros((T, G), bool), np.zeros((T, G), bool)
    for (t, g), v in fix.items():
        (on if v else off)[t, g] = True
    return off, on


def guard_fix(m, sysm, sc: UCScenario, fix, use_adequacy=True, use_lp=True, use_conflict=False, conflict_first=False):
    """guard chain of the guard-aware Learning to Fix: adequacy guard -> LP-relaxation guard (lp_guard_soft)
    [-> min up/down conflict release, off by default: in the tuning it would release whole generator rows of an
    over-fixed threshold pair and let it pass with a small fixed share; conflicts are left to the LtF cuts, as in
    the plain method]. Returns (fix, info)."""
    fix = dict(fix)
    t0 = time.time()
    r_ad = r_lp = r_c = 0
    lp_s, lp_inf = 0.0, False
    if conflict_first and fix:                     # test-time variant: conflicts released first, so that the LP
        fix, r_c = release_conflicting_rows(fix, sysm, sc.u0)    # guard always sees a feasible relaxation
    if use_adequacy and fix:
        fix, r_ad = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
    if use_lp and fix:
        fix, r_lp, lp_s, lp_inf = lp_guard_soft(m, sysm, sc, fix)
    if use_conflict and not conflict_first and fix:
        fix, r_c = release_conflicting_rows(fix, sysm, sc.u0)
    return fix, dict(released_adequacy=r_ad, released_lp=r_lp, released_conflict=r_c, guard_s=time.time() - t0,
                     guard_lp_s=lp_s, lp_infeasible=lp_inf)


class StartLtFTuner(LtFTuner):
    """LtFTuner with extra warm-start candidates for the relaxation MILP (see module docstring)."""

    def __init__(self, *args, start_tl=15.0, **kw):
        super().__init__(*args, **kw)
        self.start_tl = start_tl
        self.stats.update(n_start_cand=0, t_start_cand=0.0, start_used={})

    def _start_candidates(self, i, off, on):
        """[(x, mismatches, tag)] feasible full solutions with cost <= cap"""
        from .fixpolicy import align_to_prediction
        P = self.prob(i)
        pi = self.pi[i]
        T, G = off.shape
        out = []
        yhat = np.where(on, 1, np.where(off, 0, (pi > 0.5).astype(np.int8)))
        ua = align_to_prediction(self.s, self.d["u"][i], yhat, self.d["u0"][i])
        x0, c0 = P.label_start(ua)
        if x0 is not None and c0 <= P.cap:
            out.append((x0, P._u(x0), "label"))
        t0 = time.time()
        fix = masks_to_fix(off, on)
        sc = P.sc
        fx_c, n_c = release_conflicting_rows(fix, self.s, sc.u0)
        cands = []
        if n_c > 0:
            cands.append(("conflict-release", fx_c))
        fx_g, _ = guard_fix(self.m, self.s, sc, fx_c, use_adequacy=True, use_lp=True, use_conflict=False)
        if len(fx_g) < len(fx_c):
            cands.append(("conflict-release+guards", fx_g))
        for tag, fx in cands:
            o2, n2 = fix_to_masks(fx, T, G)
            ok, info = P.check(o2, n2, self.start_tl)          # a check with a short limit; a pass is a witness
            self.stats["n_start_cand"] += 1
            if ok and info.get("u") is not None:
                self.passed[i].append((o2.copy(), n2.copy(), info["u"]))
                x, c = P.label_start(info["u"])
                if x is not None and c <= P.cap:
                    out.append((x, P._u(x), tag))
                    break                                   # the first (less released) candidate that passes
        self.stats["t_start_cand"] += time.time() - t0
        mism = lambda u: int((off & (u == 1)).sum() + (on & (u == 0)).sum())
        return sorted([(x, mism(u), tag) for x, u, tag in out], key=lambda z: z[1])

    def add_cuts(self, i, off, on):
        """Algorithm 2 (as LtFTuner.add_cuts) with the start candidates; for k >= 1 the best candidate that violates
        no no-good constraint is used."""
        P = self.prob(i)
        pi = self.pi[i]
        starts = self._start_candidates(i, off, on)
        sets, conj, infos = [], [], []
        for k in range(self.K_max):
            st, tag = None, None
            for x, _, tg in starts:
                u0 = P._u(x)
                mm = {(t, g) for t, g in zip(*np.where((off & (u0 == 1)) | (on & (u0 == 0))))}
                if not any(set(R_) <= mm for R_ in sets):
                    st, tag = x, tg
                    break
            R, info = P.relaxation(off, on, sets, self.relax_tl, start_x=st)
            self.stats["n_relax"] += 1
            self.stats["t_relax"] += info["time"]
            info = dict(info, start=tag)
            self.stats["start_used"][str(tag)] = self.stats["start_used"].get(str(tag), 0) + 1
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
            if info.get("u") is not None:
                ow = off.copy(); nw = on.copy()
                for t, g in R:
                    ow[t, g] = False; nw[t, g] = False
                self.passed[i].append((ow, nw, info["u"]))
        self.stats["n_addcuts"] += 1
        return conj, sets, infos


class GuardedLtFTuner(StartLtFTuner):
    """LtFTuner whose check / ADD CUTS / verification see the guarded fixing set (see module docstring).
    Guard results are cached per (instance, plain fixing set)."""

    def __init__(self, *args, use_lp=True, **kw):
        super().__init__(*args, **kw)
        self.use_lp = use_lp
        self._gcache = {}
        self.stats.update(n_guard=0, t_guard=0.0, n_guard_cached=0)

    def sc(self, i):
        return UCScenario(load=self.d["load"][i], avail=self.d["avail"][i], u0=self.d["u0"][i], sr=self.d["sr"][i])

    def guarded(self, i, off, on):
        key = (i, np.packbits(off).tobytes(), np.packbits(on).tobytes())
        if key in self._gcache:
            self.stats["n_guard_cached"] += 1
            return self._gcache[key]
        t0 = time.time()
        fix, _ = guard_fix(self.m, self.s, self.sc(i), masks_to_fix(off, on), use_lp=self.use_lp)
        res = fix_to_masks(fix, *off.shape)
        self.stats["n_guard"] += 1
        self.stats["t_guard"] += time.time() - t0
        if len(self._gcache) > 20000:
            self._gcache.clear()
        self._gcache[key] = res
        return res

    def check(self, i, off, on):
        g_off, g_on = self.guarded(i, off, on)
        return super().check(i, g_off, g_on)

    def add_cuts(self, i, off, on):
        g_off, g_on = self.guarded(i, off, on)
        return super().add_cuts(i, g_off, g_on)

    def guarded_masks(self, lo, hi):
        off_all, on_all = fix_masks(self.pi, lo, hi)
        G_off, G_on = np.zeros_like(off_all), np.zeros_like(on_all)
        for i in range(self.n):
            G_off[i], G_on[i] = self.guarded(i, off_all[i], on_all[i])
        return G_off, G_on

    def verify(self, lo, hi):
        """as LtFTuner.verify, on the guarded fixing sets"""
        off_all, on_all = self.guarded_masks(lo, hi)
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


def logloss(p, y, clip=1e-3):
    p = np.clip(p, clip, 1 - clip)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
