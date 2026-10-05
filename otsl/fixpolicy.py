"""Which commitment decisions to fix before the MILP: a fixing policy learned from solver outcomes.

A probability model (Model 1) gives p[t, g] = P(u[t, g] = 1). Partial fixing sets u[t, g] := 1[p > 0.5] for a
subset S of the (unit, hour) decisions and solves the reduced MILP. The question is which S.

Published rules rank decisions by confidence (RACLearn: |p - 0.5| or MC-dropout) or cut them with
generator-specific confidence thresholds calibrated on the cost impact of fixing errors (Learning to Fix,
Fritz et al. 2026). Here the ranking is a model of the *expected cost of fixing* each decision,

    r[t, g] = P(fixing u[t, g] := yhat is harmful | x) * E[harm | harmful, x],

trained on counterfactual solver outcomes: for every decision the probability model gets wrong on
held-out instances, the exact dispatch LP prices the optimal commitment with that one decision forced to
the predicted value (the unit's schedule minimally repaired for min up/down times around it). Decisions
the model gets right have harm 0. Labels are measured against the MILP optimum aligned to the prediction
inside groups of identical units, so swapped copies are not counted as errors.

Contents: symmetry alignment, constrained min up/down repair, harm labelling (single-decision harms and
Learning-to-Fix per-generator impact curves), features, the two-head harm model, and the fixing rules
(RACLearn, Learning to Fix, asymmetric + adequacy guard, harm-ranked policy).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from . import models as _models  # noqa: F401  (keeps torch single-threaded)
from .uc import UCScenario

TYPES = ["CT", "CC", "STEAM", "NUCLEAR"]
HARM_EPS = 1e-4          # a fix is "harmful" if forcing it raises the cost by more than 0.01 %


# ----------------------------------------------------------------------------- labels
def align_to_prediction(sysm, u_star, yhat, u0, groups=None):
    """Permute the schedules of identical units (same parameters, same bus, same initial status) in the
    MILP solution u_star [T, G] so they agree best with the prediction yhat. Cost and flows are unchanged."""
    u = u_star.copy()
    for grp in (groups if groups is not None else sysm.identical_groups()):
        for val in (0, 1):
            mem = [g for g in grp if u0[g] == val]
            if len(mem) < 2:
                continue
            C = np.array([[np.sum(u_star[:, a] != yhat[:, b]) for b in mem] for a in mem])
            ra, cb = linear_sum_assignment(C)
            for a, b in zip(ra, cb):
                u[:, mem[b]] = u_star[:, mem[a]]
    return u


def repair_row_forced(row, u0g, U, D, forced):
    """Nearest (Hamming) 0/1 schedule of one unit that satisfies min up U / min down D (no carry-over at
    t = 0, as in UCModel) and takes the values in `forced` {t: v}. None if no schedule satisfies both."""
    T = len(row)
    INF = 10 ** 9
    cost = {(int(u0g), U if u0g else D): 0}
    back = []
    for t in range(T):
        nxt, bp = {}, {}
        allowed = (forced[t],) if t in forced else (0, 1)
        for (s, k), c in cost.items():
            for s2 in allowed:
                if s2 == s:
                    k2 = min(k + 1, U if s else D)
                else:
                    if (s == 1 and k < U) or (s == 0 and k < D):
                        continue
                    k2 = 1
                c2 = c + (s2 != row[t])
                if c2 < nxt.get((s2, k2), INF):
                    nxt[(s2, k2)] = c2
                    bp[(s2, k2)] = (s, k)
        if not nxt:
            return None
        back.append(bp)
        cost = nxt
    st = min(cost, key=cost.get)
    out = np.zeros(T, np.int8)
    for t in range(T - 1, -1, -1):
        out[t] = st[0]
        st = back[t][st]
    return out


def harm_labels(m, sysm, sc: UCScenario, u_star, p, groups=None, ltf=True):
    """Solver-measured cost of fixing errors for one instance.

    single: {(t, g): relative cost increase of the aligned optimum with u[t, g] forced to yhat} for every
            decision with yhat != aligned optimum (all other decisions have harm 0).
    ltf:    {g: [(theta, impact), ...]} Learning-to-Fix impact curve: relative cost increase when unit g's
            decisions with confidence >= theta are fixed to yhat (all other units at the optimum); theta runs
            over the confidences of g's wrong decisions (impact = inf if the fixings violate min up/down)."""
    T, G = p.shape
    yhat = (p > 0.5).astype(np.int8)
    ua = align_to_prediction(sysm, u_star, yhat, sc.u0, groups)
    base = m.solve_dispatch(sc, ua).obj
    wrong = np.argwhere(yhat != ua)
    single, curves, n_lp = {}, {}, 1
    for t, g in wrong:
        u2 = ua.copy()
        u2[:, g] = repair_row_forced(ua[:, g], sc.u0[g], int(sysm.min_up[g]), int(sysm.min_dn[g]), {int(t): int(yhat[t, g])})
        single[(int(t), int(g))] = (m.solve_dispatch(sc, u2).obj - base) / base
        n_lp += 1
    if ltf:
        conf = np.maximum(p, 1 - p)
        for g in np.unique(wrong[:, 1]) if len(wrong) else []:
            ts = wrong[wrong[:, 1] == g, 0]
            curve = []
            for th in sorted({float(conf[t, g]) for t in ts}, reverse=True):
                forced = {int(t): int(yhat[t, g]) for t in range(T) if conf[t, g] >= th}
                row = repair_row_forced(ua[:, g], sc.u0[g], int(sysm.min_up[g]), int(sysm.min_dn[g]), forced)
                if row is None:
                    curve.append((th, np.inf))
                    continue
                u2 = ua.copy()
                u2[:, g] = row
                curve.append((th, (m.solve_dispatch(sc, u2).obj - base) / base))
                n_lp += 1
            curves[int(g)] = curve
    return dict(base=base, single=single, ltf=curves, n_wrong=len(wrong), n_lp=n_lp)


def compensated_harms(m, sysm, sc: UCScenario, u_star, p, groups=None, uncomp=None, base=None):
    """Harm of each wrong OFF fix when other units may step in (the reduced MILP can switch on free units):
    after forcing u[t, g] := 0 (row g repaired), replace the capacity g loses in each hour by units that are
    off in the optimum - (a) greedily in merit order, (b) the cheapest single unit at least as large - with
    their rows repaired for min up/down; the dispatch LP prices each, and the smallest of the uncompensated
    and compensated harms is returned. {(t, g): harm} for wrong OFF fixes only."""
    T, G = p.shape
    yhat = (p > 0.5).astype(np.int8)
    ua = align_to_prediction(sysm, u_star, yhat, sc.u0, groups)
    n_lp = 0
    if base is None:
        base = m.solve_dispatch(sc, ua).obj
        n_lp += 1
    avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    order = np.argsort(avg)
    out = {}
    for t, g in np.argwhere((yhat != ua) & (yhat == 0)):
        u2 = ua.copy()
        u2[:, g] = repair_row_forced(ua[:, g], sc.u0[g], int(sysm.min_up[g]), int(sysm.min_dn[g]), {int(t): 0})
        if uncomp is not None and (int(t), int(g)) in uncomp:
            best = base * (1 + uncomp[(int(t), int(g))])
        else:
            best = m.solve_dispatch(sc, u2).obj
            n_lp += 1
        if (best - base) / base < 2e-3:                  # already cheap without compensation
            out[(int(t), int(g))] = (best - base) / base
            continue
        lost = np.where((ua[:, g] == 1) & (u2[:, g] == 0))[0]
        if len(lost) == 0:
            out[(int(t), int(g))] = (best - base) / base
            continue
        cands = [k for k in order if k != g and ua[lost, k].sum() == 0]
        plans = []
        need, plan = sysm.pmax[g], []
        for k in cands:                                   # (a) merit-order greedy
            if need <= 1e-9:
                break
            plan.append(k)
            need -= sysm.pmax[k]
        plans.append(plan)
        big = [k for k in cands if sysm.pmax[k] >= sysm.pmax[g] - 1e-9]
        if big:                                           # (b) cheapest single unit at least as large
            plans.append([min(big, key=lambda k: avg[k])])
        for plan in plans:
            u3 = u2.copy()
            ok = True
            for k in plan:
                row = repair_row_forced(u2[:, k], sc.u0[k], int(sysm.min_up[k]), int(sysm.min_dn[k]),
                                        {int(tt): 1 for tt in lost})
                if row is None:
                    ok = False
                    break
                u3[:, k] = row
            if ok and plan:
                best = min(best, m.solve_dispatch(sc, u3).obj)
                n_lp += 1
        out[(int(t), int(g))] = (best - base) / base
    return out, n_lp


# ----------------------------------------------------------------------------- Learning to Fix
def ltf_sweeps(curves, G):
    """Per generator, the mean (over calibration instances) relative cost impact of its fixing errors as the
    confidence threshold is lowered: (thetas descending, mean impact once all wrong decisions with
    confidence >= theta are fixed). curves: list over instances of {g: [(theta, impact), ...]}."""
    n = len(curves)
    out = []
    for g in range(G):
        ev = sorted(((th, i, im) for i, c in enumerate(curves) for th, im in c.get(g, [])), key=lambda e: -e[0])
        cur, tot, ths, ms = np.zeros(n), 0.0, [], []
        for th, i, im in ev:
            tot = np.inf if (not np.isfinite(im) or not np.isfinite(tot)) else tot + im - cur[i]
            cur[i] = im
            ths.append(th)
            ms.append(tot / n)
        out.append((np.array(ths), np.array(ms)))
    return out


def ltf_thresholds(sweeps, tau):
    """Generator-specific confidence thresholds (Fritz et al. 2026, per-generator decomposition): for each
    unit g the lowest threshold theta_g such that, for every threshold >= theta_g, the mean relative cost
    impact of g's fixing errors on the calibration instances stays <= tau. A unit never wrong in calibration
    gets theta_g = 0.5 (all its decisions fixed)."""
    theta = np.full(len(sweeps), 0.5)
    for g, (ths, ms) in enumerate(sweeps):
        bad = np.where(~(ms <= tau))[0]          # inf / nan (min up/down conflicts) count as violations
        if len(bad):
            theta[g] = ths[bad[0]] + 1e-12       # fix only decisions strictly more confident than that error
    return theta


# ----------------------------------------------------------------------------- features
class FixFeaturizer:
    """Per-decision features for the fixing policy: prediction, LP relaxation, unit data, hour, the unit's
    predicted schedule around the hour, system adequacy and relaxed prices."""

    names = None

    def __init__(self, sysm, T):
        s = sysm
        self.s, self.T = s, T
        self.avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
        self.mc0 = np.maximum(s.seg_c[:, 0], 1.0)
        onehot = np.array([[t == k for k in TYPES] for t in s.utype], float)
        size = np.ones(s.G)
        for grp in s.identical_groups():
            size[grp] = len(grp)
        self.static = np.c_[s.pmax, s.pmin / s.pmax, self.avg / 1e4, s.c_nl / 1e4, s.c_su / 1e4, s.min_up / T,
                            s.min_dn / T, np.minimum(s.ramp / s.pmax, 5), onehot, size]
        self.static_names = ["pmax", "pmin_ratio", "avg_cost", "c_nl", "c_su", "min_up", "min_dn", "ramp"] + \
                            [f"type_{k}" for k in TYPES] + ["n_identical"]

    def __call__(self, p, d, i):
        """features [T, G, F] for instance i of dataset d with probabilities p [T, G]"""
        s, T = self.s, self.T
        G = s.G
        y = (p > 0.5).astype(float)
        conf = np.abs(p - 0.5) * 2
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        ur = d["u_rel"][i]
        u0 = d["u0"][i].astype(float)
        prev = np.vstack([u0[None], y[:-1]])
        nxt = np.vstack([y[1:], y[-1:]])
        chg = (np.vstack([u0[None], y])[1:] != np.vstack([u0[None], y])[:-1])            # [T, G] switch at t
        idx = np.arange(T)[:, None]
        dist = np.full((T, G), T, float)
        for g in range(G):
            st = np.where(chg[:, g])[0]
            if len(st):
                dist[:, g] = np.min(np.abs(idx - st[None, :]), 1)
        n_sw = chg.sum(0)[None].repeat(T, 0)
        cprev = np.vstack([conf[:1], conf[:-1]])
        cnext = np.vstack([conf[1:], conf[-1:]])
        load = d["load"][i].sum(1)
        net = load - d["avail"][i].sum(1)
        sr = d["sr"][i]
        cap = (y * s.pmax).sum(1)
        floor = (y * s.pmin).sum(1)
        tot = s.pmax.sum()
        slack = (cap - net - sr) / tot
        slack_wo = slack[:, None] - y * s.pmax[None] / tot         # if this unit were switched off
        slack_w = slack[:, None] + (1 - y) * s.pmax[None] / tot     # if this unit were switched on
        floor_sl = (load - floor) / tot
        n_unc = (conf < 0.9).sum(1)
        lmp = d["lmp_rel"][i][:, s.gbus] if "lmp_rel" in d else np.zeros((T, G))
        prof = (lmp - self.avg[None]) / self.avg[None]
        prof0 = (lmp - self.mc0[None]) / self.mc0[None]
        hour = np.broadcast_to((np.arange(T) / (T - 1))[:, None], (T, G))
        B = lambda a: np.broadcast_to(np.asarray(a, float)[:, None], (T, G))
        cols = [p, conf, np.clip(logit, -12, 12) / 12, y, ur, np.abs(ur - y), np.minimum(ur, 1 - ur),
                hour, (idx == 0) + 0 * p, (idx == T - 1) + 0 * p, B(np.zeros(T)) + u0[None], prev, nxt,
                np.minimum(dist, 6) / 6, n_sw / 4, conf.min(0)[None].repeat(T, 0), conf.mean(0)[None].repeat(T, 0),
                cprev, cnext,
                B(net / tot), B(sr / tot), B(slack), slack_wo, slack_w, B(floor_sl), B(n_unc / G),
                np.clip(prof, -2, 5), np.clip(prof0, -2, 5), lmp / 1e4]
        names = ["p", "conf", "logit", "yhat", "u_rel", "u_rel_disagree", "u_rel_frac", "hour", "first_hour",
                 "last_hour", "u0", "yhat_prev", "yhat_next", "dist_switch", "n_switch", "row_conf_min",
                 "row_conf_mean", "conf_prev", "conf_next", "net_load", "reserve", "cap_slack", "slack_if_off",
                 "slack_if_on", "floor_slack", "n_uncertain_hour", "lmp_margin_avg", "lmp_margin_mc", "lmp"]
        X = np.concatenate([np.stack(cols, -1), np.broadcast_to(self.static[None], (T, G, self.static.shape[1]))], -1)
        FixFeaturizer.names = names + self.static_names
        return X.astype(np.float32)


# ----------------------------------------------------------------------------- harm model
def error_logodds(p, floor=1e-9):
    """log-odds that the rounded prediction is wrong according to the probability model itself"""
    e = np.clip(np.minimum(p, 1 - p), floor, 0.5)
    return np.log(e / (1 - e))


class HarmNet(nn.Module):
    """Two heads on per-decision features.
    cls: logit P(harmful) = alpha * logit(min(p, 1-p)) + beta + c(x): a learned correction of the probability
         model's own error odds (c starts at 0, i.e. at the RACLearn ranking), so the ranking keeps the
         probability model's resolution among very confident decisions.
    reg: E[log harm | harmful, x]."""

    def __init__(self, d_in, hidden=64):
        super().__init__()
        body = lambda: nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.body_c, self.body_r = body(), body()
        self.corr = nn.Linear(hidden, 1)
        nn.init.zeros_(self.corr.weight); nn.init.zeros_(self.corr.bias)
        self.reg = nn.Linear(hidden, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, base):
        cls = self.alpha * base + self.beta + self.corr(self.body_c(x)).squeeze(-1)
        return cls, self.reg(self.body_r(x)).squeeze(-1)


class HarmModel:
    """Expected relative cost of fixing each decision to its rounded prediction. X[..., 0] must be p."""

    def __init__(self, d_in, hidden=64, seed=0):
        torch.manual_seed(seed)
        self.net = HarmNet(d_in, hidden)
        self.mu = self.sd = None

    def _x(self, X):
        X = X.reshape(-1, X.shape[-1])
        return (torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32),
                torch.as_tensor(error_logodds(X[:, 0]), dtype=torch.float32))

    def fit(self, X, h, Xv=None, hv=None, epochs=30, lr=2e-3, bs=4096, wd=1e-4, seed=0, w_reg=0.2,
            val_ratios=(0.8, 0.9, 0.95, 0.97), log=print):
        """X [n, F]; h [n] measured harm (0 for correct decisions). BCE on h > HARM_EPS + MSE of log h on the
        harmful ones. Early stopping on the measured harm of the decisions the policy would fix on val."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        rng = np.random.default_rng(seed)
        x, base = self._x(X)
        yc = torch.as_tensor(h > HARM_EPS, dtype=torch.float32)
        lh = torch.as_tensor(np.log(np.maximum(h, HARM_EPS)), dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        best, state, hist = np.inf, None, []
        if Xv is not None:
            best = self.val_loss(Xv, hv, val_ratios)
            state = {k: t.clone() for k, t in self.net.state_dict().items()}
            if log:
                log(f"  [harm] init (= confidence ranking x mean harm) val harm-in-fixed-set {best:.5f}")
        for ep in range(epochs):
            self.net.train()
            perm = rng.permutation(len(x))
            for k in range(0, len(x), bs):
                b = perm[k:k + bs]
                lc, lr_ = self.net(x[b], base[b])
                loss = F.binary_cross_entropy_with_logits(lc, yc[b])
                m = yc[b] > 0
                if m.any():
                    loss = loss + w_reg * F.mse_loss(lr_[m], lh[b][m]) * m.float().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if Xv is not None:
                v = self.val_loss(Xv, hv, val_ratios)
                hist.append(v)
                if v < best:
                    best, state = v, {k: t.clone() for k, t in self.net.state_dict().items()}
                if log and (ep % 5 == 0 or ep == epochs - 1):
                    log(f"  [harm] ep {ep:3d} train {loss.item():.5f} alpha {self.net.alpha.item():.3f} "
                        f"val harm-in-fixed-set {v:.5f}")
        if state is not None:
            self.net.load_state_dict(state)
        self.best_val, self.hist = best, hist
        return self

    @torch.no_grad()
    def score(self, X):
        """expected relative cost of fixing: P(harmful) * exp(E[log harm | harmful])"""
        self.net.eval()
        lc, lr_ = self.net(*self._x(X))
        return (torch.sigmoid(lc) * torch.exp(lr_.clamp(max=3))).numpy().reshape(X.shape[:-1])

    def val_loss(self, Xv, hv, ratios=(0.8, 0.9, 0.95, 0.97)):
        return fixed_harm(self.score(Xv), hv, ratios)


class HarmEnsemble:
    """Geometric mean of the expected-harm scores of several seeds (reduces seed variance of the ranking)."""

    def __init__(self, members):
        self.members = members

    @classmethod
    def from_states(cls, d_in, hidden, states):
        ms = []
        for st in states:
            hm = HarmModel(d_in, hidden)
            hm.net.load_state_dict(st["state"]); hm.mu, hm.sd = st["mu"], st["sd"]
            ms.append(hm)
        return cls(ms)

    def score(self, X):
        return np.exp(np.mean([np.log(np.maximum(m.score(X), 1e-30)) for m in self.members], 0))


def fixed_harm(score, H, ratios=(0.8, 0.9, 0.95, 0.97)):
    """mean over ratios of the per-instance summed measured harm of the decisions a ranking would fix"""
    n = len(score)
    out = 0.0
    for q in ratios:
        k = int(round(q * score[0].size))
        o = np.argsort(score.reshape(n, -1), 1, kind="stable")[:, :k]
        out += np.take_along_axis(H.reshape(n, -1), o, 1).sum(1).mean()
    return out / len(ratios)


# ----------------------------------------------------------------------------- fixing rules
def fix_from_ranking(score, yhat, ratio):
    """{(t, g): yhat} for the ratio share of decisions with the lowest score"""
    flat = np.argsort(score.reshape(-1), kind="stable")[:int(round(ratio * score.size))]
    tt, gg = np.unravel_index(flat, score.shape)
    return {(int(t), int(g)): int(yhat[t, g]) for t, g in zip(tt, gg)}


def fix_from_thresholds(p, theta):
    """Learning to Fix: fix u[t, g] := yhat where max(p, 1 - p) >= theta[g]"""
    conf = np.maximum(p, 1 - p)
    yhat = (p > 0.5).astype(int)
    tt, gg = np.where(conf >= theta[None, :])
    return {(int(t), int(g)): int(yhat[t, g]) for t, g in zip(tt, gg)}


# ----------------------------------------------------------------------------- LP-relaxation guard
def relaxed_reduced(m, sc: UCScenario, fix):
    """LP relaxation of the reduced MILP (fixed decisions at their values, the rest relaxed to [0, 1]).
    Returns (cost, per-hour penalised slack [T] = shedding + over-generation + reserve shortfall, seconds).
    A lower bound: if the relaxation already sheds or misses reserve in an hour, the reduced MILP will too."""
    import time as _time
    import scipy.sparse as sp
    from scipy.optimize import linprog
    T = m.T
    t0 = _time.time()
    lo, hi, lb, ub = m._rhs_bounds(sc)
    for (t, g), v in fix.items():
        i = m.ix("u", t, g)
        lb[i] = ub[i] = v
    eq = lo == hi
    res = linprog(m.c, A_ub=sp.vstack([m.A[~eq & np.isfinite(hi)], -m.A[~eq & np.isfinite(lo)]]),
                  b_ub=np.r_[hi[~eq & np.isfinite(hi)], -lo[~eq & np.isfinite(lo)]],
                  A_eq=m.A[eq], b_eq=lo[eq], bounds=np.c_[lb, ub], method="highs")
    if res.status != 0:
        return np.inf, np.full(T, np.inf), _time.time() - t0
    x = res.x
    get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T].reshape(T, -1)
    slack = get("shed").sum(1) + get("spill").sum(1) + get("short").sum(1)
    return float(res.fun), slack, _time.time() - t0


def lp_guard(m, sysm, sc: UCScenario, fix, max_iter=4, tol=1e-6):
    """Release fixings that make the LP relaxation of the reduced problem shed load, over-generate or miss
    reserve. In each bad hour t: OFF fixes of every unit in [t - min_dn, t] (a unit fixed off earlier may be
    unable to restart) and, if the relaxation over-generates, ON fixes in [t - min_up, t]. Repeats until the
    relaxation has no penalised slack or max_iter; then releases every fixing in the remaining bad hours +-1.
    Returns (fix, released, LP seconds)."""
    fix = dict(fix)
    n0, secs = len(fix), 0.0
    for it in range(max_iter + 1):
        _, slack, dt = relaxed_reduced(m, sc, fix)
        secs += dt
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
    return fix, n0 - len(fix), secs


def release_conflicting_rows(fix, sysm, u0):
    """Release all fixings of a unit whose fixed entries cannot be completed to a schedule that satisfies
    min up/down times (the reduced MILP would be infeasible). Returns (fix, released)."""
    fix = dict(fix)
    rows = {}
    for (t, g), v in fix.items():
        rows.setdefault(g, {})[t] = v
    T = 1 + max(t for t, _ in fix) if fix else 0
    released = 0
    for g, forced in rows.items():
        if repair_row_forced(np.zeros(T, np.int8), u0[g], int(sysm.min_up[g]), int(sysm.min_dn[g]), forced) is None:
            for t in forced:
                del fix[(t, g)]
                released += 1
    return fix, released
