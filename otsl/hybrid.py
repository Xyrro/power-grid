"""Hybrid fixing rules: the joint, eps-guaranteed threshold calibration of Learning to Fix (otsl.ltfx; Fritz,
Makrides, Fetanat & Pinson 2026) combined with our probability models, error-cost scores and guards.

A hybrid rule is   score s[t, g] -> generator-specific grey zone [lo_g, hi_g] (eq. 5 of the paper) -> fixings
                   -> guards (optional) -> reduced MILP.

(a) Guard-aware tuning. The check of Algorithm 1 is applied to the fixings *after* the guards that the rule applies
    at test time, in the same order and with the same code: adequacy guard (otsl.combo.adequacy_guard), min up/down
    row release (otsl.fixpolicy.release_conflicting_rows), LP-relaxation guard (otsl.fixpolicy.lp_guard). The release
    sets of Algorithm 2 are computed on the guarded fixings, so a cut only asks the thresholds to release what the
    guards do not already release. At termination every validation instance's *guarded* reduced UC - all fixings
    jointly - has a solution within eps of C*, exactly the property the paper guarantees for its unguarded rule.
(b) Error-cost scores. The thresholded score is the learned expected cost of a wrong fix h[t, g] (otsl.fixpolicy harm
    ensemble) instead of the probability. It is mapped to a pseudo-probability that keeps the rounding direction:
        s = 0.5 * q(h)        if p <= 0.5 (fix OFF candidates),
        s = 1 - 0.5 * q(h)    if p > 0.5  (fix ON candidates),       q(h) = sigmoid((log h - mu) / sd),
    mu, sd = mean / std of log h over the validation decisions. q is strictly increasing, so s < lo_g fixes the OFF
    predictions of unit g whose error cost is below a unit-specific level, s > hi_g the ON predictions likewise; lo_g =
    hi_g = 0.5 fixes everything by rounding, as for probabilities. The fixing rule, Algorithm 1 + 2 and the Appendix A
    objective (quantile bins of s per (t, g)) are unchanged.
(c) Any probability source (kNN, MILP-label BCE GNN, self-trained GNN, REINFORCE GNN; several training seeds).

The tuner subclasses otsl.ltfx.LtFTuner (nothing there is changed): only the check, the cut generation and the
verification see the guarded fixings. Exact additions that only save work: guard outputs are memoised per instance
and pre-guard fixing set; cuts are appended to a file as they are found, so a run can be resumed or extended to more
validation instances (warm start: the cuts of instances already checked stay valid, they are derived from one
instance each).
"""
from __future__ import annotations

import json
import time

import numpy as np

from .combo import adequacy_guard
from .fixpolicy import lp_guard, release_conflicting_rows
from .ltfx import LtFTuner, fix_masks
from .uc import UCScenario

ALL_GUARDS = ("adeq", "rows", "lp")


# ============================================================================ scores
def harm_to_score(h, p, mu, sd):
    """pseudo-probability of (b): 0.5 q(h) for predicted OFF, 1 - 0.5 q(h) for predicted ON"""
    z = (np.log(np.maximum(h, 1e-30)) - mu) / sd
    q = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    q = np.clip(q, 1e-12, 1 - 1e-12)                 # keep s strictly inside (0, 0.5) / (0.5, 1)
    return np.where(p > 0.5, 1.0 - 0.5 * q, 0.5 * q)


def harm_norm(h_val):
    """mu, sd of log h over all validation decisions (fixed before tuning)"""
    z = np.log(np.maximum(np.asarray(h_val, float), 1e-30)).reshape(-1)
    return float(z.mean()), float(z.std() + 1e-12)


# ============================================================================ guards
def masks_to_fix(off, on):
    out = {(int(t), int(g)): 0 for t, g in zip(*np.where(off))}
    out.update({(int(t), int(g)): 1 for t, g in zip(*np.where(on))})
    return out


def fix_to_masks(fix, T, G):
    off, on = np.zeros((T, G), bool), np.zeros((T, G), bool)
    for (t, g), v in fix.items():
        (on if v else off)[t, g] = True
    return off, on


def apply_guards(m, sysm, sc: UCScenario, fix, guards=ALL_GUARDS):
    """the test-time guards, in order: adequacy (release OFF fixes in merit order until the units not fixed off
    cover net load + reserve + 5 % every hour), min up/down row release, LP-relaxation guard. Returns
    (fix, info) with the released counts and the wall time of all guards (info['guard_s'])."""
    t0 = time.time()
    fix = dict(fix)
    info = {}
    if "adeq" in guards:
        fix, info["released_adeq"] = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
    if "rows" in guards:
        fix, info["released_rows"] = release_conflicting_rows(fix, sysm, sc.u0)
    if "lp" in guards:
        fix, info["released_lp"], info["lp_guard_s"] = lp_guard(m, sysm, sc, fix)
    info["guard_s"] = time.time() - t0
    return fix, info


# ============================================================================ tuner
class HybridTuner(LtFTuner):
    """Algorithm 1 + 2 of Learning to Fix (otsl.ltfx.LtFTuner) with the check and the cuts applied to the guarded
    fixings (guards = () gives the faithful tuner). warm_cuts: cut sets to start the master with (from an earlier
    run on a prefix of the same validation instances, same scores, eps and guards). cut_log: file to which every
    new cut set is appended (one JSON line), for warm starts and resumption."""

    def __init__(self, m, sysm, d, pi, eps, guards=ALL_GUARDS, warm_cuts=None, cut_log=None, **kw):
        super().__init__(m, sysm, d, pi, eps, **kw)
        self.guards = tuple(guards)
        self.gmemo = {i: {} for i in range(self.n)}
        self.stats.update(n_guard=0, n_guard_cached=0, t_guard=0.0, n_warm_cuts=0)
        self.cut_log = cut_log
        for cs in warm_cuts or []:
            self.master.add_cut(cs)
            self.stats["n_warm_cuts"] += 1
        if cut_log is not None:
            add = self.master.add_cut

            def add_and_log(conj):
                add(conj)
                with open(cut_log, "a") as f:
                    f.write(json.dumps(cut_to_json(conj)) + "\n")
            self.master.add_cut = add_and_log

    def post(self, i, off, on):
        """guarded fixing masks of instance i (memoised on the pre-guard masks)"""
        if not self.guards:
            return off, on
        key = (np.packbits(off).tobytes(), np.packbits(on).tobytes())
        memo = self.gmemo[i]
        if key in memo:
            self.stats["n_guard_cached"] += 1
            return memo[key]
        fix, info = apply_guards(self.m, self.s, self.prob(i).sc, masks_to_fix(off, on), self.guards)
        self.stats["n_guard"] += 1
        self.stats["t_guard"] += info["guard_s"]
        res = fix_to_masks(fix, off.shape[0], off.shape[1])
        memo[key] = res
        return res

    def check(self, i, off, on):
        po, pn = self.post(i, off, on)
        return super().check(i, po, pn)

    def add_cuts(self, i, off, on):
        po, pn = self.post(i, off, on)
        return super().add_cuts(i, po, pn)

    def guarded_masks(self, lo, hi):
        off_all, on_all = fix_masks(self.pi, lo, hi)
        po, pn = np.zeros_like(off_all), np.zeros_like(on_all)
        for i in range(self.n):
            po[i], pn[i] = self.post(i, off_all[i], on_all[i])
        return po, pn

    def verify(self, lo, hi):
        """(8e) for the guarded fixings: a stored witness schedule that respects every guarded fixing of instance i,
        priced by the exact dispatch LP. Returns the relative cost increase over C* per instance (inf: no witness)."""
        po_all, pn_all = self.guarded_masks(lo, hi)
        out = []
        for i in range(self.n):
            off, on = po_all[i], pn_all[i]
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


def cut_to_json(conj):
    return [[{str(g): float(a) for g, a in alo.items()}, {str(g): float(b) for g, b in bhi.items()}] for alo, bhi in conj]


def cut_from_json(obj):
    return [({int(g): float(a) for g, a in alo.items()}, {int(g): float(b) for g, b in bhi.items()}) for alo, bhi in obj]


def load_cuts(path):
    with open(path) as f:
        return [cut_from_json(json.loads(line)) for line in f if line.strip()]


# ============================================================================ test-time rule
def hybrid_fixings(score, lo, hi, m=None, sysm=None, sc=None, guards=ALL_GUARDS):
    """fixings of one instance: eq. (5) on the score, then the guards. Returns (fix, info)."""
    off, on = fix_masks(score, lo, hi)
    fix = masks_to_fix(off, on)
    n_pre = len(fix)
    if guards:
        fix, info = apply_guards(m, sysm, sc, fix, guards)
    else:
        info = {"guard_s": 0.0}
    info["n_fixed_pre_guard"] = n_pre
    return fix, info
