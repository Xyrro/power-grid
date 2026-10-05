"""Learning to Fix (Fritz, Makrides, Fetanat & Pinson 2026, arXiv 2609.39396), reconstructed from the public
abstract and search snippets only (the full text was not reachable). See docs/methods/ltf.md.

What is known: a classifier (best: k-nearest neighbours) predicts each generator's hourly on/off status; a
decomposition algorithm sets generator-specific confidence thresholds subject to a prescribed cost tolerance on
validation instances (1 % in the headline result), based on the impact of fixing errors on UC cost and
feasibility; decisions whose confidence passes their generator's threshold are fixed and the reduced MILP is
solved.

Reconstruction (assumptions marked A1..A7 in docs/methods/ltf.md):

* kNN classifier over instance features (hourly load, renewable availability, initial status); the on-probability
  of (t, g) is the (optionally distance-weighted) mean of the k nearest training instances' MILP schedules,
  canonicalised inside groups of identical units (A1).
* Impact of fixing errors, per generator g and threshold theta, on each calibration instance: force every decision
  of g with confidence >= theta to its predicted value in the MILP optimum (aligned to the prediction inside
  identical-unit groups), repair g's row for min up/down (infeasible -> impact = inf), price with the exact
  dispatch LP; optionally let other units compensate (merit-order replacement of lost capacity, or dropping a
  unit made redundant by a wrong ON fix), each a feasible point of the single-generator reduced MILP. The result
  is an upper bound on the cost increase of the single-generator reduced MILP (A3).
* Decomposition (A4): generator-wise impact curves are combined either
  - "budget": maximise the expected fixed share subject to sum_g mean impact_g(theta_g) <= tau (Lagrangian
    decomposition: for a multiplier mu every generator independently maximises share_g - mu * impact_g), or
  - "each": every generator's own mean impact <= tau at its threshold and at every higher one.
* Joint check (JointCalibrator, tested on validation only): the joint fixings of all generators are priced with the
  same kind of LP upper bound; either a sequential generator-by-generator pass ("seq") or the budget proposal
  ("budgetJ") is accepted only if the joint bound on the calibration instances is <= tau (bisection of the internal
  tolerance).
"""
from __future__ import annotations

from itertools import groupby

import numpy as np

from .fixpolicy import align_to_prediction, repair_row_forced
from .uc import UCScenario
from .ucml import canonical_labels

RTYPES = ["PV", "RTPV", "WIND", "HYDRO"]


# ----------------------------------------------------------------------------- kNN classifier
class KNNCommit:
    """k-nearest-neighbour commitment classifier.

    kind="sys": per-hour area loads (3 x T), per-hour renewable availability by type (PV, RTPV, wind,
                hydro / run-of-river: 4 x T), initial status (G);
    kind="bus": per-hour bus loads (N x T), per-unit renewable availability (R x T), initial status (G)
                (the features of scripts/uc_model1.py: knn_candidates).
    Columns are standardised on the training set; each block (load / renewables / initial status) is scaled
    to unit total variance and then by its weight, so the u0 block weight w_u0 sets its relative influence."""

    def __init__(self, sysm, train, k=21, kind="sys", w_u0=1.0, w_ren=1.0, weighted=False, canonical=True):
        self.s, self.k, self.kind, self.weighted = sysm, k, kind, weighted
        self.w = dict(load=1.0, ren=w_ren, u0=w_u0)
        blocks = self._blocks(train)
        self.stats = {b: (x.mean(0), x.std(0) + 1e-9) for b, x in blocks.items()}
        self.X = self._embed(blocks)
        self.Y = (canonical_labels(sysm, train["u"], train["u0"]) if canonical else train["u"]).astype(np.float32)
        self.day = np.asarray(train["day"])

    def _blocks(self, d):
        s = self.s
        n, T, N = d["load"].shape
        if self.kind == "bus":
            load = d["load"].reshape(n, -1)
            ren = d["avail"].reshape(n, -1)
        else:
            A = np.stack([(s.bus_area == a) for a in (1, 2, 3)], 1).astype(float)          # [N, 3]
            load = (d["load"] @ A).reshape(n, -1)
            rt = np.array([("HYDRO" if t in ("HYDRO", "ROR") else t) for t in s.r_type])
            R = np.stack([(rt == t) for t in RTYPES], 1).astype(float)                      # [R, 4]
            ren = (d["avail"] @ R).reshape(n, -1)
        return dict(load=load, ren=ren, u0=d["u0"].astype(float))

    def _embed(self, blocks):
        out = []
        for b, x in blocks.items():
            mu, sd = self.stats[b]
            z = (x - mu) / sd
            keep = self.stats[b][1] > 1e-6
            out.append(z[:, keep] * self.w[b] / np.sqrt(max(keep.sum(), 1)))
        return np.concatenate(out, 1)

    def neighbours(self, d, exclude_same_day=False):
        Z = self._embed(self._blocks(d))
        dist = (Z ** 2).sum(1)[:, None] + (self.X ** 2).sum(1)[None] - 2 * Z @ self.X.T
        if exclude_same_day:                         # out-of-fold use on training instances
            dist = np.where(np.asarray(d["day"])[:, None] == self.day[None], np.inf, dist)
        idx = np.argsort(dist, 1, kind="stable")[:, :self.k]
        return idx, np.sqrt(np.maximum(np.take_along_axis(dist, idx, 1), 0))

    def predict(self, d, exclude_same_day=False):
        idx, dd = self.neighbours(d, exclude_same_day)
        if self.weighted:
            w = 1.0 / (dd + 1e-6)
        else:
            w = np.ones_like(dd)
        w = w / w.sum(1, keepdims=True)
        return np.einsum("nk,nktg->ntg", w, self.Y[idx]).astype(np.float32)


# ----------------------------------------------------------------------------- impact of fixing errors
def _lp(m, sc, u):
    return m.solve_dispatch(sc, u).obj


def ltf_curves(m, sysm, sc: UCScenario, u_star, p, groups=None, comp=True, eps=1e-4):
    """Per-generator impact curves of one calibration instance.

    Returns dict(base, curves={g: [(theta, impact_nocomp, impact_comp), ...]}, n_lp, n_wrong) with theta running
    over the distinct confidences max(p, 1 - p) of g's wrong decisions (descending). impact_* = relative cost
    increase over the aligned optimum when all of g's decisions with confidence >= theta are fixed to the
    rounded prediction (g's row repaired for min up/down; inf if impossible). impact_comp additionally allows
    other units to compensate: (a) merit-order replacement of the capacity g loses, (b) the cheapest single
    unit at least as large, (c) switching off the most expensive unit that a wrong ON fix makes redundant.
    Every priced schedule is feasible for the reduced MILP in which only g's decisions are fixed, so both
    impacts upper-bound that reduced MILP's cost increase (up to its MIP gap)."""
    T, G = p.shape
    yhat = (p > 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    ua = align_to_prediction(sysm, u_star, yhat, sc.u0, groups)
    base = _lp(m, sc, ua)
    n_lp = 1
    wrong = np.argwhere(yhat != ua)
    avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    order = np.argsort(avg)
    U = lambda k: int(sysm.min_up[k])
    D = lambda k: int(sysm.min_dn[k])
    curves = {}
    for g in (np.unique(wrong[:, 1]) if len(wrong) else []):
        g = int(g)
        levels = sorted({float(conf[t, g]) for t in wrong[wrong[:, 1] == g, 0]}, reverse=True)
        curve = []
        for th in levels:
            forced = {t: int(yhat[t, g]) for t in range(T) if conf[t, g] >= th}
            row = repair_row_forced(ua[:, g], sc.u0[g], U(g), D(g), forced)
            if row is None:
                curve.append((th, np.inf, np.inf))
                continue
            u2 = ua.copy()
            u2[:, g] = row
            c_nc = _lp(m, sc, u2)
            n_lp += 1
            best = c_nc
            if comp and (c_nc - base) / base > eps:
                plans = []
                lost = np.where((ua[:, g] == 1) & (row == 0))[0]
                gained = np.where((ua[:, g] == 0) & (row == 1))[0]
                if len(lost):
                    cands = [k for k in order if k != g and ua[lost, k].sum() == 0]
                    need, plan = sysm.pmax[g], []
                    for k in cands:                                  # (a) merit-order replacement
                        if need <= 1e-9:
                            break
                        plan.append(k)
                        need -= sysm.pmax[k]
                    if plan:
                        plans.append(("on", lost, plan))
                    big = [k for k in cands if sysm.pmax[k] >= sysm.pmax[g] - 1e-9]
                    if big:                                          # (b) one unit at least as large
                        plans.append(("on", lost, [min(big, key=lambda k: avg[k])]))
                if len(gained):                                      # (c) drop a now-redundant unit
                    cands = [k for k in order[::-1] if k != g and ua[gained, k].min() == 1
                             and sysm.pmax[k] <= sysm.pmax[g] + 1e-9]
                    if cands:
                        plans.append(("off", gained, [cands[0]]))
                for kind, hrs, plan in plans:
                    u3, ok = u2.copy(), True
                    for k in plan:
                        r3 = repair_row_forced(u2[:, k], sc.u0[k], U(k), D(k),
                                               {int(t): (1 if kind == "on" else 0) for t in hrs})
                        if r3 is None:
                            ok = False
                            break
                        u3[:, k] = r3
                    if ok:
                        best = min(best, _lp(m, sc, u3))
                        n_lp += 1
            curve.append((th, (c_nc - base) / base, (best - base) / base))
        curves[g] = curve
    return dict(base=base, curves=curves, n_lp=n_lp, n_wrong=len(wrong))


# ----------------------------------------------------------------------------- decomposition
def generator_tables(curves, G, which="comp"):
    """Per generator: distinct confidence levels of its calibration errors (descending) and the mean relative
    impact over the calibration instances once every decision with confidence >= level is fixed.
    curves: list over calibration instances of {g: [(theta, imp_nocomp, imp_comp), ...]}."""
    n = len(curves)
    col = 2 if which == "comp" else 1
    out = []
    for g in range(G):
        ev = sorted(((c[0], i, c[col]) for i, cv in enumerate(curves) for c in cv.get(g, [])), key=lambda e: -e[0])
        cur = np.zeros(n)
        levels, means = [], []
        for th, grp in groupby(ev, key=lambda e: e[0]):
            for _, i, imp in grp:
                cur[i] = imp
            levels.append(th)
            means.append(float(cur.mean()) if np.all(np.isfinite(cur)) else np.inf)
        out.append((np.array(levels), np.array(means)))
    return out


def generator_options(tables, conf_cal):
    """Candidate thresholds per generator with their calibration impact and fixed share.
    conf_cal: [n, T, G] confidences on the calibration instances. Option list per g: [(theta, impact, share)]
    from 'fix only decisions more confident than every error' down to 'fix all' (theta = 0.5)."""
    opts = []
    for g, (lv, ms) in enumerate(tables):
        c = conf_cal[:, :, g].reshape(-1)
        share = lambda th: float((c >= th).mean())
        o = []
        if len(lv) == 0:
            o.append((0.5, 0.0, 1.0))
        else:
            th0 = lv[0] + 1e-9
            o.append((th0, 0.0, share(th0)))
            for j in range(len(lv) - 1):
                th = lv[j + 1] + 1e-9
                o.append((th, ms[j], share(th)))
            o.append((0.5, ms[-1], 1.0))
        opts.append(o)
    return opts


def thresholds_each(opts, tau):
    """'each': the lowest threshold such that the generator's own mean impact is <= tau there and at every
    higher threshold."""
    theta = np.zeros(len(opts))
    for g, o in enumerate(opts):
        theta[g] = o[0][0]
        for th, imp, _ in o[1:]:
            if not imp <= tau:
                break
            theta[g] = th
    return theta


def thresholds_budget(opts, tau, n_mu=600):
    """'budget' (Lagrangian decomposition): maximise the mean fixed share over generators subject to
    sum_g impact_g(theta_g) <= tau. For each multiplier mu every generator picks argmax share - mu * impact
    independently; the feasible solution with the largest share over the mu grid is returned, then greedily
    improved by single-generator upgrades that still fit the budget."""
    imp = [np.array([x[1] for x in o]) for o in opts]
    shr = [np.array([x[2] for x in o]) for o in opts]
    best, best_sel = -1.0, None
    for mu in np.r_[np.inf, np.logspace(6, -3, n_mu)]:
        sel = []
        for I, S in zip(imp, shr):
            ok = np.isfinite(I)
            val = np.where(ok, (S - mu * I) if np.isfinite(mu) else -I + 1e-9 * S, -np.inf)
            sel.append(int(np.argmax(val)))
        tot = sum(I[j] for I, j in zip(imp, sel))
        sh = float(np.mean([S[j] for S, j in zip(shr, sel)]))
        if tot <= tau + 1e-12 and sh > best:
            best, best_sel = sh, sel
    sel = list(best_sel)
    used = sum(I[j] for I, j in zip(imp, sel))
    while True:                                   # greedy fill of the remaining budget
        cand = None
        for g, (I, S) in enumerate(zip(imp, shr)):
            for j in range(len(I)):
                d_s, d_i = S[j] - S[sel[g]], I[j] - I[sel[g]]
                if d_s > 1e-12 and np.isfinite(I[j]) and used + d_i <= tau + 1e-12:
                    score = d_s / max(d_i, 1e-12)
                    if cand is None or score > cand[0]:
                        cand = (score, g, j, d_i)
        if cand is None:
            break
        _, g, j, d_i = cand
        sel[g] = j
        used += d_i
    theta = np.array([o[j][0] for o, j in zip(opts, sel)])
    return theta, float(used), float(np.mean([S[j] for S, j in zip(shr, sel)]))


def fix_from_thresholds(p, theta):
    """{(t, g): rounded prediction} for every decision with confidence max(p, 1 - p) >= theta[g]"""
    conf = np.maximum(p, 1 - p)
    yhat = (p > 0.5).astype(int)
    tt, gg = np.where(conf >= theta[None, :])
    return {(int(t), int(g)): int(yhat[t, g]) for t, g in zip(tt, gg)}


# ----------------------------------------------------------------------------- sequential (joint) decomposition
_JW = {}


def _jinit():
    from .uc import UCModel, load_rts_gmlc
    s = load_rts_gmlc()
    _JW.update(s=s, m=UCModel(s, T=12, network=True))


def compensate_capacity(sysm, u, ua, fixed, u0, down=False):
    """Free units switched on (merit order, rows repaired with their fixed entries kept) in every hour in which the
    committed capacity of u falls below that of the aligned optimum ua; with down=True, free units switched off
    (most expensive first) in hours with more capacity than ua while it stays at least ua's. Returns the new
    commitment (a feasible completion of the same fixings)."""
    T, G = u.shape
    avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    order = np.argsort(avg)
    u2 = u.copy()
    cap = lambda t: (ua[t] * sysm.pmax).sum() - (u2[t] * sysm.pmax).sum()
    for t in range(T):
        if down:
            for k in order[::-1]:
                ex = -cap(t)
                if ex <= 1e-9:
                    break
                if not u2[t, k] or fixed[t, k] or sysm.pmax[k] > ex + 1e-9:
                    continue
                forced = {tt: int(u2[tt, k]) for tt in range(T) if fixed[tt, k]}
                forced[t] = 0
                row = repair_row_forced(u2[:, k], u0[k], int(sysm.min_up[k]), int(sysm.min_dn[k]), forced)
                if row is not None:
                    u2[:, k] = row
            continue
        for k in order:
            if cap(t) <= 1e-9:
                break
            if u2[t, k] or fixed[t, k]:
                continue
            forced = {tt: int(u2[tt, k]) for tt in range(T) if fixed[tt, k]}
            forced[t] = 1
            row = repair_row_forced(u2[:, k], u0[k], int(sysm.min_up[k]), int(sysm.min_dn[k]), forced)
            if row is not None:
                u2[:, k] = row
    return u2


def _jcost(args):
    """dispatch-LP cost of a forced commitment, and of its capacity-compensated version (min of both)"""
    sc, u, fixed, ua, comp = args
    s, m = _JW["s"], _JW["m"]
    c = m.solve_dispatch(sc, u).obj
    if comp:
        d = ((ua - u) * s.pmax[None]).sum(1)
        if d.max() > 1e-9:                                   # lost capacity: switch free units on
            c = min(c, m.solve_dispatch(sc, compensate_capacity(s, u, ua, fixed, sc.u0)).obj)
        if d.min() < -1e-9:                                  # surplus capacity: switch free units off
            u3 = compensate_capacity(s, u, ua, fixed, sc.u0, down=True)
            if not np.array_equal(u3, u):
                c = min(c, m.solve_dispatch(sc, u3).obj)
    return c


class JointCalibrator:
    """Sequential decomposition of the threshold choice: generators are visited one at a time (order given); for
    generator g the threshold is lowered level by level (the confidences of its calibration errors) as long as the
    mean *joint* relative cost increase over the calibration instances - all thresholds chosen so far applied
    together, the rest free - stays <= tau. The joint cost of a set of fixings on an instance is the dispatch-LP
    cost of the aligned optimum with every fixed decision forced to its prediction (rows repaired for min
    up/down), optionally with free units compensating lost capacity: a feasible completion of the reduced MILP,
    hence an upper bound of its cost. A final check prices the final joint fixings exactly (the visit used
    compensation by units fixed later); if it exceeds tau the internal tolerance is reduced and the pass rerun."""

    def __init__(self, sysm, d, p, pool, comp=True, groups=None):
        self.s, self.pool, self.comp = sysm, pool, comp
        n, T, G = p.shape
        self.n, self.T, self.G = n, T, G
        self.sc = [UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i]) for i in range(n)]
        self.yhat = (p > 0.5).astype(np.int8)
        self.conf = np.maximum(p, 1 - p)
        self.ua = np.array([align_to_prediction(sysm, d["u"][i], self.yhat[i], d["u0"][i], groups) for i in range(n)])
        self.memo, self.n_lp = {}, 0
        self.base = np.array(self._costs([(i, self.ua[i], np.zeros((T, G), bool)) for i in range(n)], comp=False))

    def _costs(self, items, comp=None):
        comp = self.comp if comp is None else comp
        keys = [(i, u.tobytes(), fixed.tobytes() if comp else b"") for i, u, fixed in items]
        todo = {}
        for k, (i, u, fixed) in zip(keys, items):
            if k not in self.memo and k not in todo:
                todo[k] = (self.sc[i], u, fixed, self.ua[i], comp)
        if todo:
            res = self.pool.map(_jcost, list(todo.values()), chunksize=1)
            self.n_lp += len(todo)
            self.memo.update(zip(todo.keys(), res))
        return [self.memo[k] for k in keys]

    def _row(self, i, g, th):
        forced = {t: int(self.yhat[i, t, g]) for t in range(self.T) if self.conf[i, t, g] >= th}
        return repair_row_forced(self.ua[i, :, g], self.sc[i].u0[g], int(self.s.min_up[g]), int(self.s.min_dn[g]), forced)

    def joint(self, theta, idx=None):
        """exact joint impact (upper bound) of thresholds theta on the calibration instances"""
        idx = range(self.n) if idx is None else idx
        items, bad = [], []
        for i in idx:
            u = self.ua[i].copy()
            ok = True
            for g in range(self.G):
                if np.isfinite(theta[g]) and (self.conf[i, :, g] >= theta[g]).any():
                    row = self._row(i, g, theta[g])
                    if row is None:
                        ok = False
                        break
                    u[:, g] = row
            if ok:
                items.append((i, u, self.conf[i] >= theta[None, :]))
            else:
                bad.append(i)
        c = dict(zip([it[0] for it in items], self._costs(items)))
        return np.array([((c[i] - self.base[i]) / self.base[i]) if i in c else np.inf for i in idx])

    def run(self, tau, order, w=None, log=None):
        n, T, G = self.n, self.T, self.G
        w = np.ones(n) / n if w is None else np.asarray(w, float) / np.sum(w)
        theta = np.full(G, np.inf)
        U = self.ua.copy()
        imp = np.zeros(n)
        wrong = self.yhat != self.ua                                            # [n, T, G]
        for g in order:
            errs = [(float(self.conf[i, t, g]), i) for i, t in zip(*np.where(wrong[:, :, g]))]
            levels = sorted({e[0] for e in errs}, reverse=True)
            if not levels:
                theta[g] = 0.5
                continue
            # candidate thresholds: just above the next lower error level (all errors >= levels[j] fixed, and the
            # correct decisions in between too), or 0.5 below the last one
            cands = [levels[j + 1] + 1e-9 if j + 1 < len(levels) else 0.5 for j in range(len(levels))]
            acc = None
            for j, th_c in enumerate(cands):
                aff = sorted({i for c, i in errs if c >= levels[j]})
                th_try = theta.copy()
                th_try[g] = th_c
                items, newU, inf_ = [], {}, False
                for i in aff:
                    row = self._row(i, g, th_c)
                    if row is None:
                        inf_ = True
                        break
                    if np.array_equal(row, U[i][:, g]):
                        continue
                    u = U[i].copy()
                    u[:, g] = row
                    newU[i] = u
                    items.append((i, u, self.conf[i] >= th_try[None, :]))
                if inf_:
                    break
                cs = self._costs(items)
                trial = imp.copy()
                for (i, _, _), c in zip(items, cs):
                    trial[i] = (c - self.base[i]) / self.base[i]
                if not (w @ trial <= tau):
                    break
                imp = trial
                for i, u in newU.items():
                    U[i] = u
                acc = j
            theta[g] = levels[0] + 1e-9 if acc is None else cands[acc]
            if log:
                log(f"    g={g:2d} levels {len(levels):3d} accepted {(-1 if acc is None else acc) + 1:3d}  theta {theta[g]:.4f}  "
                    f"impact {w @ imp * 100:.3f}%  LPs {self.n_lp}")
        return theta, float(w @ imp)

    def calibrate(self, tau, order=None, w=None, max_rerun=3, log=None, propose=None):
        """Thresholds from a proposer (default: the sequential pass run(); or any tau -> (theta, estimate) map, e.g.
        the separable budget decomposition), then the exact joint check on the calibration instances; the internal
        tolerance is bisected (at most max_rerun reruns) and the feasible thresholds (joint impact <= tau) with the
        largest calibration share are kept"""
        w_ = np.ones(self.n) / self.n if w is None else np.asarray(w, float) / np.sum(w)
        share = lambda th: float((self.conf >= th[None, None, :]).mean())
        propose = propose or (lambda t: self.run(t, order, w))
        lo, hi, t_int, hist, best, last = 0.0, None, tau, [], None, None
        for r in range(max_rerun + 1):
            theta, est = propose(t_int)
            final = self.joint(theta)
            fin = float(w_ @ final) if np.all(np.isfinite(final[w_ > 0])) else np.inf
            hist.append(dict(tau_internal=t_int, est=est, final=fin, share=share(theta)))
            if log:
                log(f"  tau {tau:g}: internal {t_int:.5g} est {est * 100:.3f}% final joint {fin * 100:.3f}% "
                    f"share {share(theta) * 100:.1f}%  LPs {self.n_lp}")
            last = (theta, fin)
            if fin <= tau * 1.001:
                if best is None or share(theta) > share(best[0]):
                    best = (theta, fin)
                lo = t_int
                if hi is None or not self.comp:
                    break
                t_int = (lo + hi) / 2
            else:
                hi = t_int
                t_int = (lo + hi) / 2 if lo > 0 else t_int * min(0.5, tau / fin if np.isfinite(fin) else 0.5)
        theta, fin = best if best is not None else last
        return theta, fin, hist
