"""m24: the 12-hour recipe of otsl.m1x (temporal-GNN ensemble on polished labels, guard-aware Learning to Fix tuning,
LP-relaxation guard, reduced MILP at a loose gap) brought to the 24-hour benchmark uc24 (otsl.b3).

* release_min_conflicts  minimal min up/down conflict release. On 24 hours, rounding a unit's probabilities often gives
                         a row that no min up/down-feasible schedule completes; the row release of the 12-hour hybrid
                         (otsl.fixpolicy.release_conflicting_rows) then frees the whole row, and inside Learning to Fix's
                         tuning check that makes the master collapse every interval (uc24ltf.md: 100 % fixed before
                         the guards, 42 % after). Here, per conflicting row, a dynamic program finds a min up/down-
                         feasible schedule that disagrees with the fewest fixed entries (free entries cost nothing), and
                         only the fixings it disagrees with are released. The remaining fixings are completable, so the
                         reduced MILP is feasible.
* guard_chain            adequacy guard (5 % margin) -> minimal conflict release [-> LP-relaxation guard]: the guards of
                         the tuning check and of the test-time rule.
* M24Tuner               uc24ltf.GuardedLtFTuner (Algorithm 1 + 2 with better relaxation starts, guards inside the
                         check, cuts and verification) with guard_chain as the guard.
* polish                 teacher-polished label: a reduced MILP around the teacher's 90 % most confident decisions
                         (otsl.m1x.polish, highspy path of otsl.b3, guards as above), kept if the dispatch LP prices it
                         below the label-free (block-repaired) label.
* build_gnnt             otsl.m1x.CommitGNNT with dilations 1, 2, 4, 8 (every hour of a 24-hour row sees every other).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import models as _models  # noqa: F401  (keeps torch single-threaded)
from .combo import adequacy_guard
from .fixpolicy import fix_from_ranking, lp_guard
from .m1x import CommitGNNT
from .ucml import UCModel1
from .uc24ltf import GuardedLtFTuner, fix_to_masks, masks_to_fix


# ============================================================================ guards
def nearest_row(forced, u0g, U, D, T):
    """min up/down-feasible 0/1 row (no carry-over at t = 0, as in UCModel) minimising the number of entries of
    `forced` {t: v} it disagrees with; returns the row"""
    INF = 10 ** 9
    cost = {(int(u0g), U if u0g else D): 0}
    back = []
    for t in range(T):
        nxt, bp = {}, {}
        f = forced.get(t)
        for (s, k), c in cost.items():
            for s2 in (0, 1):
                if s2 == s:
                    k2 = min(k + 1, U if s else D)
                else:
                    if (s == 1 and k < U) or (s == 0 and k < D):
                        continue
                    k2 = 1
                c2 = c + (0 if f is None else int(s2 != f))
                if c2 < nxt.get((s2, k2), INF):
                    nxt[(s2, k2)] = c2
                    bp[(s2, k2)] = (s, k)
        back.append(bp)
        cost = nxt
    st = min(cost, key=cost.get)
    row = np.zeros(T, np.int8)
    for t in range(T - 1, -1, -1):
        row[t] = st[0]
        st = back[t][st]
    return row


def release_min_conflicts(fix, sysm, u0, T):
    """release, per unit, only the fixings that the nearest min up/down-feasible row (otherwise free) contradicts.
    Returns (fix, released)."""
    rows = {}
    for (t, g), v in fix.items():
        rows.setdefault(g, {})[t] = v
    out = dict(fix)
    released = 0
    for g, forced in rows.items():
        row = nearest_row(forced, int(u0[g]), int(sysm.min_up[g]), int(sysm.min_dn[g]), T)
        for t, v in forced.items():
            if row[t] != v:
                del out[(t, g)]
                released += 1
    return out, released


def guard_chain(m, sysm, sc, fix, lp=False):
    """adequacy guard -> minimal conflict release [-> LP-relaxation guard]; returns (fix, info)"""
    t0 = time.time()
    fix, r_a = adequacy_guard(dict(fix), sc.load, sc.avail, sc.sr, sysm)
    fix, r_c = release_min_conflicts(fix, sysm, sc.u0, sc.load.shape[0])
    info = dict(released_adeq=r_a, released_rows=r_c)
    if lp and fix:
        fix, r_l, lp_s = lp_guard(m, sysm, sc, fix)
        info.update(released_lp=r_l, lp_guard_s=lp_s)
    info["guard_s"] = time.time() - t0
    return fix, info


class M24Tuner(GuardedLtFTuner):
    """guard-aware Learning to Fix (uc24ltf.GuardedLtFTuner) with the guard chain adequacy + minimal conflict release"""

    def guarded(self, i, off, on):
        key = (i, np.packbits(off).tobytes(), np.packbits(on).tobytes())
        if key in self._gcache:
            self.stats["n_guard_cached"] += 1
            return self._gcache[key]
        t0 = time.time()
        fix, _ = guard_chain(self.m, self.s, self.sc(i), masks_to_fix(off, on), lp=False)
        res = fix_to_masks(fix, *off.shape)
        self.stats["n_guard"] += 1
        self.stats["t_guard"] += time.time() - t0
        if len(self._gcache) > 20000:
            self._gcache.clear()
        self._gcache[key] = res
        return res


# ============================================================================ polished labels
def teacher_fixings(p, m, sysm, sc, ratio=0.9, lp=True):
    """the ratio share of the teacher's most confident decisions fixed to round(p), then the guard chain"""
    fix = fix_from_ranking(np.minimum(p, 1 - p), (p > 0.5).astype(int), ratio)
    return guard_chain(m, sysm, sc, fix, lp=lp)


def polish(m, sysm, sc, p, y_ref, c_ref, ratio=0.9, time_limit=10.0, gap=1e-3, lp=True):
    """reduced MILP around the teacher's confident decisions (highspy, one thread); the label is its schedule if the
    dispatch LP prices it below the reference label y_ref (cost c_ref), else y_ref. Returns (label, record)."""
    from .b3 import solve_milp_hs
    t0 = time.time()
    fix, info = teacher_fixings(p, m, sysm, sc, ratio, lp)
    sol = solve_milp_hs(m, sc, time_limit=time_limit, mip_gap=gap, z_fix=fix or None, threads=1, incumbents=False)
    rec = dict(n_fixed=len(fix), milp_s=sol["time"], status=sol["status"], **{k: v for k, v in info.items()})
    if sol["u"] is None:
        rec.update(cost=np.inf, used="ref", total_s=time.time() - t0)
        return np.asarray(y_ref, np.int8), rec
    lp_sol = m.solve_dispatch(sc, sol["u"])
    rec.update(cost=float(lp_sol.obj), shed=float(lp_sol.shed), short=float(lp_sol.short), milp_obj=float(sol["obj"]))
    if lp_sol.obj < c_ref:
        rec.update(used="milp", total_s=time.time() - t0)
        return sol["u"].astype(np.int8), rec
    rec.update(used="ref", total_s=time.time() - t0)
    return np.asarray(y_ref, np.int8), rec


# ============================================================================ model
def build_gnnt(sysm, feat, T, seed=0, dilations=(1, 2, 4, 8)):
    torch.manual_seed(seed)
    return UCModel1(CommitGNNT(sysm, feat, T, dilations=dilations), feat, "gnn")
