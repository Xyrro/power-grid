"""rloop: recursive unit-commitment model with the exact dispatch LP in the loop (arm B of docs/recursive/PLAN.md).

A small recurrent network over (unit, hour) tokens and the bus graph keeps an answer state y and a latent z (TRM).
After every outer step its answer is decoded to a schedule that satisfies min up/down by construction, repaired for
adequacy and priced by the exact dispatch LP; the LP's signals are inputs of the next step:

    step 0:  z, y <- recursion(x, no feedback)            -> scores_0 -> s_0 = decode(scores_0) -> LP(s_0)
    step k:  z, y <- recursion(x, feedback(LP(s_{k-1})))  -> scores_k -> s_k -> LP(s_k)
    output:  the cheapest priced schedule s_k (penalties included): every step is a priced candidate.

Identical units (56 of 73 units in 24 groups, each at one bus) are handled by count heads: per group and hour the
network gives ordinal logits l_j = logit P(count > j), j = 0..m-1, from the group's pooled state. The units of a group
are stacked in a fixed order (initially-on units first): unit at position j is on iff count > j, so l_j is that unit's
logit and its target is [count* > j] (the label's count). The exact symmetry is removed and no unit index is needed.
Singletons are groups of size 1. (Option counts=False: per-unit logits, labels aligned to the prediction inside groups.)

Decoding: per-unit Viterbi over the on/off state machine with min up/down and the initial status, on the network's
log-probabilities (most likely feasible schedule), then the block (min up/down-aware) adequacy repair and the min
up/down repair as a safety net (no-op). Option decode='round': threshold 0.5, then the same repairs.

Feedback (per outer step, from the LP of s): per unit-hour the priced schedule, what decoding and repair changed
relative to the rounding, dispatch relative to limits, reserve, the LMP at the unit's bus and the exact LP sensitivity
d cost / d u (bound duals, with the start-up / shut-down variables, as UCModel.dispatch_gradient); per bus-hour LMP,
shed, spill; per line-hour loading, flow-limit dual, binding flag; per hour energy and reserve shortfall, reserve
price, capacity and minimum-output margins; per instance the cost of s and of the best schedule so far relative to the
LP relaxation (the relaxation is a lower bound, so this is a certified gap), served and is-best flags.

Inputs carry physics, not indices: units by their parameters, time by relative mixing (dilated convolutions) and
distance to the horizon's ends, buses by a PTDF signature (projection of how an injection at the bus loads every line,
recomputed for lines out of service) and the bus graph with edge gates.

Components:
  FastLP         dispatch LP / LP relaxation through highspy on the matrices of otsl.ood.OODModel (outages supported),
                 warm-started across calls on one solver object; same model as UCModel.solve_dispatch.
  GroupInfo      identical-unit groups, stacking positions, stacked (count) targets.
  viterbi_minud  min up/down-feasible maximum-likelihood decoding, vectorised over units with equal (U, D).
  StaticFeats    length-agnostic static inputs (unit-hour, bus-hour, line-hour), normalised on training data.
  RLoopNet       the recurrent network (recall, TRM answer/latent recursion, temporal + unit-pool + bus-graph mixing,
                 count heads, halting head).
  LPPool         decoding + repair + dispatch LP + feedback in CPU worker processes (or in process).
  train_supervised / train_reinforce   deep supervision with LP feedback of the model's own schedule; cost-aware
                 fine-tuning with the exact LP as critic (REINFORCE on each step's improvement, RLOO baseline).
  Inference      timed single-instance inference with K outer steps, keeping the best schedule.

Everything runs on CPU or CUDA (`device`); the LPs always run on CPU (in the pool's worker processes).
"""
from __future__ import annotations

import collections
import contextlib
import copy
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constrained import adequacy_repair_blocks
from .fixpolicy import align_to_prediction
from .ood import OODModel, OODScenario
from .uc import load_rts_gmlc, repair_min_updown

# ============================================================================ presets
PRESETS = {
    # CPU feasibility study (this container: one core per process, torch single-threaded)
    "cpu": dict(hidden=64, n_latent=2, n_cycles=1, mp_layers=1, dilations=(1, 2, 4), k_train=3, batch=32, epochs=10,
                lr=1e-3, ema=0.995, lp_workers=0, inflight=1, k_eval=8, counts=True, decode="viterbi", p_ext=0.25,
                rl_updates=150, rl_batch=16, rl_samples=4, rl_k=3, rl_lr=1e-4, bce_w=0.5, halt_w=0.2),
    # larger GPU configuration (one GPU + 16-32 CPU cores for the LP workers in the same job)
    "gpu": dict(hidden=160, n_latent=3, n_cycles=2, mp_layers=2, dilations=(1, 2, 4, 8), k_train=6, batch=64, epochs=60,
                lr=7e-4, ema=0.999, lp_workers=16, inflight=4, k_eval=16, counts=True, decode="viterbi", p_ext=0.25,
                rl_updates=1500, rl_batch=32, rl_samples=8, rl_k=6, rl_lr=1e-4, bce_w=0.5, halt_w=0.2),
}


def slog(x, s=1e3):
    """signed log scale for prices and sensitivities ($/p.u.h): sign(x) log(1 + |x| / s)"""
    return np.sign(x) * np.log1p(np.abs(x) / s)


# ============================================================================ the exact dispatch LP (highspy)
class FastLP:
    """Dispatch LP (fixed commitment) and LP relaxation of otsl.uc.UCModel through highspy, with the outages of
    otsl.ood.OODModel. One solver object is kept and re-used (bounds changed, basis kept: warm start); reset() drops it
    (next solve is cold). Returns primal and dual quantities needed for the feedback features."""

    def __init__(self, sysm, T):
        import highspy
        self.hs = highspy
        self.s, self.T = sysm, T
        self.m = OODModel(sysm, T)
        m = self.m
        A = m.A.tocsc()
        self.nrow, self.ncol = A.shape
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.ncol, self.nrow
        lp.col_cost_ = m.c
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        self.lp = lp
        self.h = None
        self.cidx = np.arange(self.ncol, dtype=np.int32)
        self.ridx = np.arange(self.nrow, dtype=np.int32)
        N = sysm.n_bus
        self.kcl = np.array([[m.row_tag[("kcl", t, n)][0] for n in range(N)] for t in range(T)])
        self.srr = np.array([m.row_tag[("sr", t)][0] for t in range(T)])
        self.S = sysm.seg_w.shape[1]

    def reset(self):
        self.h = None

    def _get(self, name, v):
        o, k = self.m.off[name]
        return v[o:o + k * self.T].reshape(self.T, -1)

    def solve(self, sc, u=None, relax=False):
        hs = self.hs
        inf = hs.kHighsInf
        lo, hi, lb, ub = self.m._rhs_bounds(sc, None if relax else u)
        lo = np.where(np.isfinite(lo), lo, -inf)
        hi = np.where(np.isfinite(hi), hi, inf)
        ub = np.where(np.isfinite(ub), ub, inf)
        t0 = time.time()
        if self.h is None:
            h = hs.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("threads", 1)
            self.lp.col_lower_, self.lp.col_upper_ = lb, ub
            self.lp.row_lower_, self.lp.row_upper_ = lo, hi
            h.passModel(self.lp)
            self.h = h
        else:
            self.h.changeColsBounds(self.ncol, self.cidx, lb, ub)
            self.h.changeRowsBounds(self.nrow, self.ridx, lo, hi)
        self.h.run()
        dt = time.time() - t0
        info = self.h.getInfo()
        sol = self.h.getSolution()
        x, cd, rd = np.array(sol.col_value), np.array(sol.col_dual), np.array(sol.row_dual)
        s, T, G = self.s, self.T, self.s.G
        uu = self._get("u", x)
        ps = self._get("ps", x).reshape(T, G, self.S).sum(2)
        out = dict(obj=float(info.objective_function_value), lp_s=dt, u=uu, p=uu * s.pmin[None] + ps,
                   res=self._get("res", x), flow=self._get("f", x), shed=self._get("shed", x),
                   spill=self._get("spill", x), short=self._get("short", x)[:, 0], lmp=rd[self.kcl], rprice=rd[self.srr],
                   fdual=self._get("f", cd))
        if not relax:
            uf = np.asarray(u, float).reshape(T, G)
            mu, mv, mw = self._get("u", cd), self._get("v", cd), self._get("w", cd)
            prev = np.vstack([np.asarray(sc.u0, float)[None], uf[:-1]])
            up, dn = (uf > prev).astype(float), (uf < prev).astype(float)
            grad = mu + up * mv - dn * mw
            grad[:-1] += -up[1:] * mv[1:] + dn[1:] * mw[1:]
            out["grad"] = grad
        return out


# ============================================================================ instances, groups, decoding
def outs(inst, key):
    return [int(x) for x in np.atleast_1d(inst.get(key, ())) if x >= 0]


def scenario_of(inst):
    """OODScenario from an instance dict (load, avail, u0, sr [, l_out, g_out])"""
    return OODScenario(load=inst["load"], avail=inst["avail"], u0=inst["u0"], sr=inst["sr"],
                       l_out=tuple(outs(inst, "l_out")), g_out=tuple(outs(inst, "g_out")))


class GroupInfo:
    """Groups of identical units (same bus and parameters) plus singletons as groups of size 1. Within a group the
    units are stacked: initially-on units first, then by index; the unit at position j is on iff the group's count > j."""

    def __init__(self, sysm):
        grp = [sorted(g) for g in sysm.identical_groups()]
        inG = {g for gr in grp for g in gr}
        grp += [[g] for g in range(sysm.G) if g not in inG]
        grp.sort(key=lambda gr: gr[0])
        self.groups = grp
        self.NG = len(grp)
        self.gid = np.zeros(sysm.G, np.int64)
        for k, gr in enumerate(grp):
            self.gid[gr] = k
        self.size = np.array([len(g) for g in grp])
        self.mmax = int(self.size.max())

    def positions(self, u0):
        pos = np.zeros(len(u0), np.int64)
        for gr in self.groups:
            for j, g in enumerate(sorted(gr, key=lambda g: (-int(u0[g]), g))):
                pos[g] = j
        return pos

    def counts(self, y):
        """[..., T, G] -> [..., T, NG]"""
        out = np.zeros(y.shape[:-1] + (self.NG,), np.int64)
        for k, gr in enumerate(self.groups):
            out[..., k] = y[..., gr].sum(-1)
        return out

    def stacked(self, y, u0):
        """stacked (count) targets of a schedule y [T, G]: unit at position j is on iff the group's count > j"""
        n = self.counts(y)
        return (n[:, self.gid] > self.positions(u0)[None]).astype(np.float32)


_VCLASS = {}


def _vclasses(min_up, min_dn):
    key = (tuple(int(x) for x in min_up), tuple(int(x) for x in min_dn))
    if key not in _VCLASS:
        cl = collections.defaultdict(list)
        for g, (U, D) in enumerate(zip(*key)):
            cl[(U, D)].append(g)
        _VCLASS[key] = {k: np.array(v) for k, v in cl.items()}
    return _VCLASS[key]


def viterbi_minud(p, u0, min_up, min_dn, eps=1e-4):
    """most likely schedule [T, G] under independent on-probabilities p [T, G] subject to min up / min down times and
    the initial status (no carry-over at t = 0, as UCModel). States per unit: on for k = 1..U hours (U = capped) or off
    for k = 1..D hours; switching off needs k = U on-hours, switching on k = D off-hours. Vectorised over units with the
    same (U, D). With 0/1 input it is the Hamming-nearest feasible schedule (otsl.uc.repair_min_updown)."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    T, G = p.shape
    c_on, c_off = -np.log(p), -np.log1p(-p)
    out = np.zeros((T, G), np.int8)
    for (U, D), gs in _vclasses(min_up, min_dn).items():
        n = len(gs)
        S = U + D                       # states 0..U-1: on k = 1..U; U..U+D-1: off k = 1..D
        V = np.full((n, S), np.inf)
        on0 = np.asarray(u0)[gs] > 0
        V[on0, U - 1] = 0.0
        V[~on0, S - 1] = 0.0
        back = np.zeros((T, n, S), np.int16)
        ar = np.arange(S)
        for t in range(T):
            W = np.full((n, S), np.inf)
            B = np.tile(ar, (n, 1))
            # on 1 <- off D (switch on); also on 1 <- on 1 when U == 1 (stay, capped)
            W[:, 0], B[:, 0] = V[:, S - 1], S - 1
            if U == 1:
                m = V[:, 0] < W[:, 0]
                W[m, 0], B[m, 0] = V[m, 0], 0
            else:
                W[:, 1:U], B[:, 1:U] = V[:, 0:U - 1], ar[0:U - 1]
                m = V[:, U - 1] < W[:, U - 1]                              # on U <- on U (stay, capped)
                W[m, U - 1], B[m, U - 1] = V[m, U - 1], U - 1
            # off 1 <- on U (switch off); also off 1 <- off 1 when D == 1
            W[:, U], B[:, U] = V[:, U - 1], U - 1
            if D == 1:
                m = V[:, U] < W[:, U]
                W[m, U], B[m, U] = V[m, U], U
            else:
                W[:, U + 1:S], B[:, U + 1:S] = V[:, U:S - 1], ar[U:S - 1]
                m = V[:, S - 1] < W[:, S - 1]
                W[m, S - 1], B[m, S - 1] = V[m, S - 1], S - 1
            W[:, :U] += c_on[t, gs][:, None]
            W[:, U:] += c_off[t, gs][:, None]
            V, back[t] = W, B
        st = V.argmin(1)
        rows = np.arange(n)
        for t in range(T - 1, -1, -1):
            out[t, gs] = st < U
            st = back[t, rows, st]
    return out


def decode(scores, inst, sysm, mode="viterbi"):
    """scores: on-probabilities [T, G] (float) or a 0/1 sample (int8) -> min up/down-feasible schedule (Viterbi on the
    probabilities) or the rounding at 0.5 -> block adequacy repair -> min up/down repair (Hamming-nearest).
    Units out of service (g_out) are forced off. Returns (rounding r, schedule s)."""
    sc = np.asarray(scores)
    r = (sc > 0.5).astype(np.int8)
    g_out = outs(inst, "g_out")
    if g_out:
        r[:, g_out] = 0
    if mode == "viterbi" and sc.dtype.kind == "f":
        p = sc.astype(float).copy()
        if g_out:
            p[:, g_out] = 0.0
        v = viterbi_minud(p, inst["u0"], sysm.min_up, sysm.min_dn)
    else:                                   # as otsl.base.lp_round_candidates: round -> block repair -> min up/down
        v = r
    v = adequacy_repair_blocks(v, inst["u0"], inst["load"], inst["avail"], inst["sr"], sysm)
    s = repair_min_updown(v, inst["u0"], sysm.min_up, sysm.min_dn).astype(np.int8)
    if g_out:
        s[:, g_out] = 0
    return r, s


def repair(r, inst, sysm):
    """rounded schedule -> block adequacy repair -> min up/down repair (otsl.b3.rep_block)"""
    return decode(np.asarray(r, np.int8), inst, sysm, "round")[1]


# ============================================================================ features
F_U, F_B, F_L, F_H, F_I = 9, 3, 3, 6, 4
FB_UNIT = F_U + F_B + F_H + F_I + 1          # unit-token feedback width (+ "feedback present" flag)


def feedback_arrays(sol, s_sched, r, inst, sysm, fmax):
    """LP solution of schedule s_sched (decoded / repaired from the rounding r) -> feedback arrays (float32):
    fu [T, G, F_U], fb [T, N, F_B], fl [T, L, F_L], fh [T, F_H]"""
    s = sysm
    sf = s_sched.astype(np.float32)
    p, res = sol["p"], sol["res"]
    span = np.maximum(s.pmax - s.pmin, 1e-6)
    pos = np.where(sf > 0, (p - s.pmin[None] * sf) / span[None], 0.0)
    lmp_u = sol["lmp"][:, s.gbus]
    at_max = (sf > 0) & (p >= s.pmax[None] - 1e-5)
    at_min = (sf > 0) & (p <= s.pmin[None] + 1e-5)
    fu = np.stack([sf, sf - np.asarray(r, np.float32), p / s.pmax[None], pos, res / s.pmax[None], slog(lmp_u),
                   slog(sol["grad"]), at_max, at_min], -1)
    fb = np.stack([slog(sol["lmp"]), sol["shed"], sol["spill"]], -1)
    load = inst["load"].sum(1)
    ld = np.where(fmax[None] > 0, np.abs(sol["flow"]) / np.maximum(fmax[None], 1e-9), 0.0)
    fl = np.stack([ld, slog(sol["fdual"]), ld >= 1 - 1e-6], -1)
    net = load - inst["avail"].sum(1)
    capm = ((sf * s.pmax[None]).sum(1) - net - inst["sr"]) / load
    ming = (load - (sf * s.pmin[None]).sum(1)) / load
    fh = np.stack([sol["shed"].sum(1), sol["spill"].sum(1), sol["short"], slog(sol["rprice"]), capm, ming], -1)
    return [a.astype(np.float32) for a in (fu, fb, fl, fh)]


def instance_feats(cost, best, c_rel, served, is_best):
    """per-instance feedback: certified gaps (log cost ratio to the LP relaxation, in units of 10 %) of the current and
    the best schedule, served and is-best flags"""
    g = lambda c: min(100 * math.log(max(c, 1.0) / c_rel), 50.0) / 10
    return np.array([g(cost), g(best), float(served), float(is_best)], np.float32)


def ptdf(sysm, l_out=()):
    """PTDF [L, N]: flow on each line per unit injection at each bus, withdrawn at the reference bus; lines out of
    service have zero susceptance and zero rows"""
    b = np.array(sysm.b, float).copy()
    b[list(l_out)] = 0.0
    A = sysm.A
    Bb = A.T @ (b[:, None] * A)
    keep = [i for i in range(sysm.n_bus) if i != sysm.ref]
    X = np.zeros((sysm.n_bus, sysm.n_bus))
    X[np.ix_(keep, keep)] = np.linalg.inv(Bb[np.ix_(keep, keep)])
    return (b[:, None] * A) @ X


class StaticFeats:
    """Length-agnostic static inputs. unit-hour [T, G, DX]: unit data (limits, costs, min up/down, ramp, type), initial
    status, identical-group size, hourly system data (load, net load, reserve, renewables), distance to the horizon's
    ends (clipped at 12 h), LP-relaxation commitment and price at the unit's bus, load / renewables at the unit's bus
    [+ symmetry rank if sym_rank]. bus-hour [T, N, DB]: load, renewables, capacity, relaxed price, PTDF signature
    (projection of the bus's PTDF column on the n_ptdf leading singular vectors of the intact network's PTDF).
    line-hour [T, L, DL]: log susceptance, rating, relaxed loading, in service. Normalised with training statistics."""

    def __init__(self, sysm, train_insts, n_stat=400, n_ptdf=16, sym_rank=False):
        from .ucml import TYPES
        s = sysm
        self.s, self.sym_rank, self.n_ptdf = s, sym_rank, n_ptdf
        onehot = np.array([[t == k for k in TYPES] for t in s.utype], float)
        avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
        self.g_static = np.c_[s.pmax, s.pmin, avg / 1e4, s.seg_c / 1e4, s.c_su / 1e4, s.min_up / 24, s.min_dn / 24,
                              np.minimum(s.ramp / s.pmax, 5), onehot].astype(np.float32)
        self.Rb = np.zeros((s.n_bus, len(s.r_pmax)))
        self.Rb[s.r_bus, np.arange(len(s.r_pmax))] = 1
        self.capbus = np.bincount(s.gbus, weights=s.pmax, minlength=s.n_bus)
        self.groups = s.identical_groups()
        self.gsize = np.ones(s.G)
        for gr in self.groups:
            self.gsize[gr] = len(gr)
        P = ptdf(s)
        U_, _, _ = np.linalg.svd(P, full_matrices=False)
        self.pbasis = U_[:, :n_ptdf]                                        # [L, k]
        self._penc = {}
        xs = [self.raw(inst, s.fmax) for inst in train_insts[:n_stat]]
        self.stats = []
        for j in range(3):
            a = np.concatenate([x[j].reshape(-1, x[j].shape[-1]) for x in xs])
            mu, sd = a.mean(0), a.std(0) + 1e-6
            self.stats.append((mu.astype(np.float32), sd.astype(np.float32)))
        self.dx, self.db, self.dl = (x.shape[-1] for x in xs[0])

    def penc(self, l_out):
        key = tuple(sorted(l_out))
        if key not in self._penc:
            self._penc[key] = (self.pbasis.T @ ptdf(self.s, key)).T.astype(np.float32)     # [N, k]
        return self._penc[key]

    def rank(self, u0):
        rank = np.zeros(self.s.G)
        for grp in self.groups:
            for val in (0, 1):
                for r, g in enumerate([g for g in grp if u0[g] == val]):
                    rank[g] = r
        return rank

    def raw(self, inst, fmax):
        s = self.s
        load, avail, u0 = inst["load"], inst["avail"], inst["u0"]
        T, G = load.shape[0], s.G
        ren = avail @ self.Rb.T                                            # [T, N]
        hour = np.stack([load.sum(1), load.sum(1) - avail.sum(1), inst["sr"], avail.sum(1)], -1)   # [T, 4]
        t = np.arange(T)
        edge = np.stack([np.minimum(t, 12) / 12, np.minimum(T - 1 - t, 12) / 12], -1)
        rel_p = slog(inst["lmp_rel"])
        per_unit = [u0, self.gsize] + ([self.rank(u0)] if self.sym_rank else [])
        ux = np.concatenate([
            np.broadcast_to(self.g_static[None], (T, G, self.g_static.shape[1])),
            np.broadcast_to(np.stack(per_unit, -1)[None], (T, G, len(per_unit))),
            np.broadcast_to(hour[:, None], (T, G, 4)), np.broadcast_to(edge[:, None], (T, G, 2)),
            inst["u_rel"][..., None], rel_p[:, s.gbus][..., None], load[:, s.gbus][..., None], ren[:, s.gbus][..., None]],
            -1)
        l_out = outs(inst, "l_out")
        pe = self.penc(l_out)
        bx = np.concatenate([np.stack([load, ren, np.broadcast_to(self.capbus[None], load.shape), rel_p], -1),
                             np.broadcast_to(pe[None], (T,) + pe.shape)], -1)
        lo = np.zeros(s.L)
        lo[l_out] = 1
        ld = np.abs(inst["flow_rel"]) / np.maximum(fmax[None], 1e-9)
        lx = np.stack([np.broadcast_to(np.log(s.b)[None], ld.shape), np.broadcast_to(fmax[None], ld.shape), ld,
                       np.broadcast_to(1 - lo[None], ld.shape)], -1)
        return ux.astype(np.float32), bx.astype(np.float32), lx.astype(np.float32)

    def __call__(self, inst, fmax=None):
        fmax = self.s.fmax if fmax is None else fmax
        out = []
        for a, (mu, sd) in zip(self.raw(inst, fmax), self.stats):
            out.append((a - mu) / sd)
        gate = np.ones(self.s.L, np.float32)
        gate[outs(inst, "l_out")] = 0
        return out + [gate]


# ============================================================================ the network
def _mlp(i, h, o, ln=True):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h) if ln else nn.Identity(), nn.SiLU(), nn.Linear(h, o))


class Core(nn.Module):
    """One shared recursion block on tokens h [B, T, G, H]: temporal mixing per unit (residual dilated 1-D
    convolutions over hours, length-agnostic), mixing across units in the same hour (DeepSets mean pool), message
    passing over the bus graph per hour (units averaged into their buses, interaction-network layers over the lines
    with edge gates for lines out of service), and a token MLP."""

    def __init__(self, sysm, H, mp_layers=1, dilations=(1, 2, 4)):
        super().__init__()
        self.register_buffer("gbus", torch.as_tensor(sysm.gbus, dtype=torch.long))
        self.register_buffer("f", torch.as_tensor(sysm.f_bus, dtype=torch.long))
        self.register_buffer("t", torch.as_tensor(sysm.t_bus, dtype=torch.long))
        cnt = np.bincount(sysm.gbus, minlength=sysm.n_bus).astype(np.float32)
        self.register_buffer("inv_cnt", torch.as_tensor(1.0 / np.maximum(cnt, 1)))
        self.N = sysm.n_bus
        self.convs = nn.ModuleList([nn.Conv1d(H, H, 3, padding=d, dilation=d) for d in dilations])
        self.cnorms = nn.ModuleList([nn.LayerNorm(H) for _ in dilations])
        self.pool = _mlp(2 * H, H, H)
        self.edge = nn.ModuleList([_mlp(3 * H, H, H) for _ in range(mp_layers)])
        self.node = nn.ModuleList([_mlp(3 * H, H, H) for _ in range(mp_layers)])
        self.back = _mlp(2 * H, H, H)
        self.tok = _mlp(H, 2 * H, H)
        self.out_norm = nn.LayerNorm(H)

    def forward(self, h, bus_e, line_e, gate):
        B, T, G, H = h.shape
        x = h.permute(0, 2, 1, 3).reshape(B * G, T, H)
        for conv, ln in zip(self.convs, self.cnorms):
            x = x + conv(F.silu(ln(x)).transpose(1, 2)).transpose(1, 2)
        h = x.reshape(B, G, T, H).permute(0, 2, 1, 3)
        h = h + self.pool(torch.cat([h, h.mean(2, keepdim=True).expand(-1, -1, G, -1)], -1))
        hb = torch.zeros(B, T, self.N, H, device=h.device, dtype=h.dtype).index_add_(2, self.gbus, h)
        hb = hb * self.inv_cnt[:, None] + bus_e
        g = gate[:, None, :, None]                                          # [B, 1, L, 1]
        for eu, nu in zip(self.edge, self.node):
            m = eu(torch.cat([hb[:, :, self.f], hb[:, :, self.t], line_e], -1)) * g
            out = torch.zeros_like(hb).index_add_(2, self.f, m)
            inn = torch.zeros_like(hb).index_add_(2, self.t, m)
            hb = hb + nu(torch.cat([hb, out, inn], -1))
        h = h + self.back(torch.cat([h, hb[:, :, self.gbus]], -1))
        h = h + self.tok(h)
        return self.out_norm(h)


class RLoopNet(nn.Module):
    """Recursive commitment model. State (y, z) per (unit, hour) token; one outer step = n_cycles x (n_latent latent
    updates z <- core(z + inj) then one answer update y <- y + upd(y, z)), with inj = x-embedding (recall) +
    feedback embedding + y. Logits from y: count heads per identical group (ordinal, monotone in the position) or
    per-unit heads. A halting head predicts, per instance, whether this step's schedule is within 0.5 % of the best
    known cost. recall=False: the x-embedding only initialises z. feedback=False: LP signals are zeroed (the priced
    schedule and what the repair changed are kept). carry_z=False: z is re-initialised every outer step."""

    def __init__(self, sysm, dx, db, dl, hidden=64, n_latent=2, n_cycles=1, mp_layers=1, dilations=(1, 2, 4),
                 recall=True, feedback=True, carry_z=True, counts=True, anchor=False, anchor_init=0.0, **_):
        super().__init__()
        H = hidden
        self.H, self.n_latent, self.n_cycles = H, n_latent, n_cycles
        self.recall, self.feedback, self.carry_z, self.counts = recall, feedback, carry_z, counts
        gi = GroupInfo(sysm)
        self.NG, self.mmax = gi.NG, gi.mmax
        self.register_buffer("gid", torch.as_tensor(gi.gid, dtype=torch.long))
        self.register_buffer("ginv", torch.as_tensor(1.0 / gi.size, dtype=torch.float32))
        self.ex = _mlp(dx, H, H)
        self.efb = _mlp(FB_UNIT, H, H)
        self.eb = _mlp(db + F_B + 1, H, H)
        self.el = _mlp(dl + F_L + 1, H, H)
        self.ey = nn.Linear(H, H)
        self.core = Core(sysm, H, mp_layers, dilations)
        self.yupd = _mlp(2 * H, H, H)
        self.head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 1))
        if counts:
            self.ghead = nn.Sequential(nn.LayerNorm(H), _mlp(H, H, self.mmax))
        self.halt = nn.Sequential(nn.LayerNorm(2 * H), _mlp(2 * H, H, 1))
        self.y0 = nn.Parameter(torch.zeros(H))
        self.z0 = nn.Parameter(torch.zeros(H))
        if anchor:      # skip connection to the priced schedule: logit += a (2 s - 1) when feedback is present
            self.anchor = nn.Parameter(torch.full((1,), float(anchor_init)))

    def embed_static(self, ux):
        return self.ex(ux)

    def init_state(self, xe):
        B, T, G, H = xe.shape
        y = self.y0.expand(B, T, G, H)
        z = xe if not self.recall else self.z0.expand(B, T, G, H)
        return y.contiguous(), z.contiguous()

    def logits(self, y, pos):
        """per-unit logits [B, T, G]; pos [B, G] stacking positions (count heads)"""
        if not self.counts:
            return self.head(y).squeeze(-1)
        B, T, G, H = y.shape
        yg = torch.zeros(B, T, self.NG, H, device=y.device, dtype=y.dtype).index_add_(2, self.gid, y)
        yg = yg * self.ginv[:, None]
        raw = self.ghead(yg)                                                # [B, T, NG, mmax]
        # monotone ordinal logits: l_0 = a, l_j = l_{j-1} - softplus(d_j)
        lo = torch.cat([raw[..., :1], raw[..., :1] - torch.cumsum(F.softplus(raw[..., 1:]), -1)], -1)
        per = lo[:, :, self.gid]                                            # [B, T, G, mmax]
        idx = pos[:, None, :, None].expand(B, T, G, 1)
        return per.gather(-1, idx).squeeze(-1)

    def step(self, xe, bx, lx, gate, pos, fbu, fbb, fbl, state, grad_cycles=1):
        """one outer step. xe [B,T,G,H] embedded static unit-hour input; bx [B,T,N,DB]; lx [B,T,L,DL]; gate [B,L];
        pos [B,G]; fbu [B,T,G,FB_UNIT], fbb [B,T,N,F_B+1], fbl [B,T,L,F_L+1] feedback (zeros at step 0).
        Returns logits [B,T,G], halting features [B,2H] (halt_logit) and the new (y, z). Only the last grad_cycles
        cycles carry gradients."""
        y, z = state
        if not self.carry_z:
            z = self.init_state(xe)[1]
        if not self.feedback:                                               # keep schedule bits, drop LP signals
            fbu = torch.cat([fbu[..., :2], torch.zeros_like(fbu[..., 2:-1]), fbu[..., -1:]], -1)
            fbb = torch.cat([torch.zeros_like(fbb[..., :-1]), fbb[..., -1:]], -1)
            fbl = torch.cat([torch.zeros_like(fbl[..., :-1]), fbl[..., -1:]], -1)
        fe = self.efb(fbu)
        be = self.eb(torch.cat([bx, fbb], -1))
        le = self.el(torch.cat([lx, fbl], -1))
        stat = xe if self.recall else 0.0
        for c in range(self.n_cycles):
            # the last grad_cycles cycles keep the caller's grad mode (no_grad at inference), earlier ones never track
            ctx = contextlib.nullcontext() if c >= self.n_cycles - grad_cycles else torch.no_grad()
            with ctx:
                for _ in range(self.n_latent):
                    z = self.core(z + stat + fe + self.ey(y), be, le, gate)
                y = y + self.yupd(torch.cat([y, z], -1))
        hf = torch.cat([y.mean((1, 2)), y.amax((1, 2))], -1)              # halting features [B, 2H]
        lg = self.logits(y, pos)
        if hasattr(self, "anchor"):
            lg = lg + self.anchor * (2 * fbu[..., 0] - 1) * fbu[..., -1]
        return lg, hf, (y, z)

    def halt_logit(self, hf):
        return self.halt(hf).squeeze(-1)


def build_net(sysm, feats, cfg):
    return RLoopNet(sysm, feats.dx, feats.db, feats.dl, **cfg)


# ============================================================================ LP feedback in worker processes
_WK = {}


def _wk_init(cfg):
    torch.set_num_threads(1)
    cores = cfg.get("cores")
    if cores:
        try:
            os.sched_setaffinity(0, set(cores))
        except (AttributeError, OSError):
            pass
    _WK.update(cfg=cfg, sys={}, lp={}, cache=collections.OrderedDict())


def _wk_lp(line_scale, T):
    key = (round(float(line_scale), 4), int(T))
    if key not in _WK["lp"]:
        if key[0] not in _WK["sys"]:
            _WK["sys"][key[0]] = load_rts_gmlc(line_scale=key[0])
        _WK["lp"][key] = FastLP(_WK["sys"][key[0]], key[1])
    return _WK["lp"][key]


def lp_feedback_job(job):
    """(key, inst, scores, line_scale, mode) -> dict(r, s, cost, shed, short, fb arrays as float16, timings).
    scores: on-probabilities [T, G] (float16/32) or a 0/1 sample (int8); decoded with `mode` (viterbi / round)."""
    key, inst, scores, line_scale, mode = job
    lp = _wk_lp(line_scale, inst["load"].shape[0])
    t0 = time.time()
    r, s_sched = decode(scores, inst, lp.s, mode)
    t1 = time.time()
    ck = (key, s_sched.tobytes())
    cache = _WK["cache"]
    if ck in cache:
        base = cache[ck]
        cache.move_to_end(ck)
        fu = base["fu"].copy()
        fu[..., 1] = (s_sched.astype(np.float32) - r).astype(np.float16)
        return dict(base, r=r, fu=fu, t_repair=t1 - t0, t_lp=0.0, hit=True)
    sol = lp.solve(scenario_of(inst), s_sched)
    fmax = lp.s.fmax.copy()
    fmax[outs(inst, "l_out")] = 0
    fu, fb, fl, fh = feedback_arrays(sol, s_sched, r, inst, lp.s, fmax)
    out = dict(r=r, s=s_sched, cost=sol["obj"], shed=float(sol["shed"].sum() + sol["spill"].sum()),
               short=float(sol["short"].sum()), fu=fu.astype(np.float16), fb=fb.astype(np.float16),
               fl=fl.astype(np.float16), fh=fh.astype(np.float16), t_repair=t1 - t0, t_lp=sol["lp_s"], hit=False)
    cache[ck] = out
    while len(cache) > _WK["cfg"].get("cache", 3000):
        cache.popitem(last=False)
    return out


class LPPool:
    """Decoding + repair + dispatch LP + feedback for many (instance, scores) pairs: in `workers` spawn processes
    pinned to `cores` (CPU), or in process (workers = 0). Each worker memoises its LPs by (instance, schedule)."""

    def __init__(self, workers=0, cores=None, cache=3000):
        self.workers = workers
        cfg = dict(cores=cores, cache=cache)
        if workers > 0:
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_wk_init, initargs=(cfg,))
        else:
            self.pool = None
            _wk_init(dict(cores=None, cache=cache))
        self.n_lp, self.n_hit, self.t_lp = 0, 0, 0.0

    def submit(self, jobs):
        """jobs: [(key, inst, scores, line_scale, mode)]"""
        if self.pool is not None:
            # instances with the same key go to the same chunk where possible (worker-side cache)
            return dict(ar=self.pool.map_async(lp_feedback_job, jobs, chunksize=max(1, len(jobs) // (2 * self.workers))),
                        jobs=None)
        return dict(ar=None, jobs=jobs)

    def ready(self, hd):
        return hd["ar"] is None or hd["ar"].ready()

    def get(self, hd):
        res = hd["ar"].get() if hd["ar"] is not None else [lp_feedback_job(j) for j in hd["jobs"]]
        for r in res:
            self.n_lp += not r["hit"]
            self.n_hit += r["hit"]
            self.t_lp += r["t_lp"]
        return res

    def run(self, jobs):
        return self.get(self.submit(jobs))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# ============================================================================ batching
class Batch:
    """static inputs of a list of instances as tensors on `device` (all instances with the same horizon)"""

    def __init__(self, feats, insts, device, fmax=None, gi=None):
        xs = [feats(inst, fmax) for inst in insts]
        f = lambda j: torch.as_tensor(np.stack([x[j] for x in xs]), device=device)
        self.ux, self.bx, self.lx, self.gate = f(0), f(1), f(2), f(3)
        gi = gi or GroupInfo(feats.s)
        self.pos = torch.as_tensor(np.stack([gi.positions(inst["u0"]) for inst in insts]), device=device)
        self.insts = insts
        self.B, self.T, self.G = self.ux.shape[0], self.ux.shape[1], self.ux.shape[2]
        self.N, self.L = self.bx.shape[2], self.lx.shape[2]
        self.device = device


def empty_feedback(b: Batch):
    z = lambda *sh: torch.zeros(*sh, device=b.device)
    return z(b.B, b.T, b.G, FB_UNIT), z(b.B, b.T, b.N, F_B + 1), z(b.B, b.T, b.L, F_L + 1)


def stack_feedback(b: Batch, res, best, gbus):
    """feedback tensors from LP results (one per instance) and the best cost before this result (per instance)"""
    fu, fbb, fbl = [], [], []
    for inst, r, cb in zip(b.insts, res, best):
        served = r["shed"] < 1e-6 and r["short"] < 1e-6
        fi = instance_feats(r["cost"], min(cb, r["cost"]), float(inst["c_rel"]), served, r["cost"] <= cb + 1e-6)
        u = r["fu"].astype(np.float32)
        bb = r["fb"].astype(np.float32)
        T, G = u.shape[:2]
        fu.append(np.concatenate([u, bb[:, gbus], np.broadcast_to(r["fh"].astype(np.float32)[:, None], (T, G, F_H)),
                                  np.broadcast_to(fi[None, None], (T, G, F_I)), np.ones((T, G, 1), np.float32)], -1))
        fbb.append(np.concatenate([bb, np.ones(bb.shape[:2] + (1,), np.float32)], -1))
        ll = r["fl"].astype(np.float32)
        fbl.append(np.concatenate([ll, np.ones(ll.shape[:2] + (1,), np.float32)], -1))
    f = lambda a: torch.as_tensor(np.stack(a), device=b.device)
    return f(fu), f(fbb), f(fbl)


def targets_for(net, gi, sysm, labels, r, u0, stacked=None):
    """training targets: stacked count targets (count heads; precomputed `stacked` if given) or labels aligned to the
    rounding r inside groups of identical units (per-unit heads)"""
    if net.counts:
        return stacked if stacked is not None else np.stack([gi.stacked(labels[i], u0[i]) for i in range(len(labels))])
    return np.stack([align_to_prediction(sysm, labels[i], r[i], u0[i], gi_groups(sysm)) for i in range(len(labels))]
                    ).astype(np.float32)


_GG = {}


def gi_groups(sysm):
    if id(sysm) not in _GG:
        _GG[id(sysm)] = sysm.identical_groups()
    return _GG[id(sysm)]


# ============================================================================ starting schedules (progressive training)
def perturb_label(y, rng, k_mean=3.0):
    """a corrupted schedule near the label: k ~ Poisson units switched off for a random block of hours and k units
    switched on for a random block (then decoded and repaired by the LP worker)"""
    u = np.array(y, np.int8, copy=True)
    T, G = u.shape
    for val in (0, 1):
        for _ in range(rng.poisson(k_mean)):
            g, a = rng.integers(G), rng.integers(T)
            u[a:min(T, a + rng.integers(1, T + 1)), g] = val
    return u


def external_start(inst, y, rng):
    """an arbitrary starting schedule: the LP relaxation rounded at a random threshold, or a perturbed label"""
    if rng.random() < 0.5:
        return (inst["u_rel"] > rng.choice([0.05, 0.2, 0.5, 0.8, 0.95])).astype(np.int8)
    return perturb_label(y, rng)


# ============================================================================ training
class EMA:
    def __init__(self, net, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(net).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        for a, b in zip(self.shadow.parameters(), net.parameters()):
            a.mul_(self.decay).add_(b.detach(), alpha=1 - self.decay)
        for a, b in zip(self.shadow.buffers(), net.buffers()):
            a.copy_(b)


class Episode:
    """one in-flight training batch: instances, static batch, carried state, step index, pending LP handle"""

    def __init__(self, idx, batch):
        self.idx, self.b, self.k, self.state = idx, batch, 0, None
        self.pending, self.fb, self.halt = None, None, None
        self.best = np.full(len(idx), np.inf)
        self.hacc = []


def train_supervised(net, feats, sysm, train_insts, labels, best_known, cfg, pool, device, log=print, val_fn=None,
                     ckpt=None, start_epoch=0, opt_state=None, ema_state=None, seed=0):
    """Deep supervision with LP feedback of the model's own schedule (on-policy). Per batch: K = cfg['k_train'] outer
    steps; step k's loss is BCE to the stacked count targets (or, per-unit heads, to the label aligned to step k's
    rounding inside identical-unit groups) + cfg['halt_w'] x BCE of the halting head (target: the step's schedule
    priced within 0.5 % of the best known cost; on steps whose LP is computed). With deep supervision (cfg['deep_sup'],
    default) the optimizer steps after every outer step and (y, z) are detached (TRM); without it, the loss of the last
    step only is back-propagated through all steps. With probability cfg['p_ext'] a batch starts from an arbitrary
    schedule (LP relaxation rounded at a random threshold, or a perturbed label) and its LP feedback instead of the
    model's own step 0 (progressive training: improve from any starting point). cfg['inflight'] batches are kept in
    flight so the network trains while the LP workers solve."""
    rng = np.random.default_rng(seed + 1000 * start_epoch)
    torch.manual_seed(seed + 1000 * start_epoch)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    accum = cfg.get("accum", 1)         # gradient accumulation over episodes (no deep supervision only)
    total = cfg["epochs"] * math.ceil(len(train_insts) / (cfg["batch"] * accum)) * cfg["k_train"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    if opt_state is not None:
        opt.load_state_dict(opt_state["opt"])
        sched.load_state_dict(opt_state["sched"])
    ema = EMA(net, cfg["ema"])
    if ema_state is not None:
        ema.shadow.load_state_dict(ema_state)
    gi = GroupInfo(sysm)
    u0_all = np.stack([x["u0"] for x in train_insts])
    stacked_all = np.stack([gi.stacked(labels[i], u0_all[i]) for i in range(len(labels))]) if net.counts else None
    K = cfg["k_train"]
    deep_sup = cfg.get("deep_sup", True)
    ls, mode = cfg.get("line_scale", 1.0), cfg.get("decode", "viterbi")
    hw = cfg.get("halt_w", 0.0)
    for ep in range(start_epoch, cfg["epochs"]):
        net.train()
        t0 = time.time()
        perm = rng.permutation(len(train_insts))
        queue = collections.deque([perm[i:i + cfg["batch"]] for i in range(0, len(perm), cfg["batch"])])
        live, hist, n_opt, n_acc = [], collections.defaultdict(list), 0, 0
        opt.zero_grad()

        def start(idx):
            e = Episode(idx, Batch(feats, [train_insts[i] for i in idx], device, gi=gi))
            if rng.random() < cfg.get("p_ext", 0.0):
                e.pending = pool.submit([(("tr", int(i)), train_insts[i], external_start(train_insts[i], labels[i], rng),
                                          ls, mode) for i in idx])
            return e

        while queue or live:
            while queue and len(live) < cfg.get("inflight", 1):
                live.append(start(queue.popleft()))
            e = next((x for x in live if x.pending is None or pool.ready(x.pending)), live[0])
            b = e.b
            hloss = 0.0
            if e.pending is not None:
                res = pool.get(e.pending)
                e.pending = None
                if e.halt is not None and hw > 0:     # halting target of the previous step (its schedule is now priced)
                    c = np.array([x["cost"] for x in res])
                    ht = torch.as_tensor((c <= 1.005 * best_known[e.idx]).astype(np.float32), device=device)
                    hlg = net.halt_logit(e.halt)
                    hloss = hw * F.binary_cross_entropy_with_logits(hlg, ht)
                    hist["halt"].append(hloss.item() / hw)
                    hist["halt_acc"].append(float(((hlg.detach() > 0).float() == ht).float().mean()))
                    hist["halt_pos"].append(float(ht.mean()))
                e.fb = stack_feedback(b, res, e.best, sysm.gbus)
                for j, x in enumerate(res):
                    e.best[j] = min(e.best[j], x["cost"])
                hist["lpgap"].append(float(np.mean([math.log(x["cost"] / train_insts[i]["c_rel"]) * 100
                                                    for x, i in zip(res, e.idx)])))
            xe = net.embed_static(b.ux)
            if e.state is None:
                e.state = net.init_state(xe)
            fb = e.fb if e.fb is not None else empty_feedback(b)
            logits, hf, state = net.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, e.state)
            r = (logits.detach() > 0).cpu().numpy().astype(np.int8)
            tgt = targets_for(net, gi, sysm, labels[e.idx], r, u0_all[e.idx],
                              stacked_all[e.idx] if stacked_all is not None else None)
            tgt_t = torch.as_tensor(tgt, device=device)
            w = None
            if (cfg.get("w_wrong", 0.0) or cfg.get("w_grad", 0.0)) and e.fb is not None:
                # cost-weighted deep supervision: decisions the priced schedule got wrong, weighted by the LP's
                # sensitivity |d cost / d u| (signed log, feedback channel 6)
                wrong = (e.fb[0][..., 0] != tgt_t).float()
                w = 1.0 + wrong * (cfg.get("w_wrong", 0.0) + cfg.get("w_grad", 0.0) * e.fb[0][..., 6].abs())
                w = w / w.mean()
            loss = F.binary_cross_entropy_with_logits(logits, tgt_t, weight=w)
            hist[f"bce{e.k}"].append(loss.item())
            hist[f"err{e.k}"].append(float((r != tgt).sum((1, 2)).mean()))
            # the halting loss needs the LP cost of this step's schedule: the head (a probe on detached pooled
            # features) is trained one step later, when that cost is known
            if deep_sup:
                opt.zero_grad()
                (loss + hloss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                ema.update(net)
                n_opt += 1
                e.state = (state[0].detach(), state[1].detach())
            else:
                e.state = state                       # no detach: the last step's loss reaches every step
                if torch.is_tensor(hloss):
                    e.hacc.append(hloss)
                if e.k == K - 1:
                    ((loss + sum(e.hacc)) / accum).backward()
                    n_acc += 1
                    if n_acc % accum == 0 or (not queue and len(live) == 1):
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
                        opt.zero_grad()
                        for _ in range(K):
                            sched.step()
                        ema.update(net)
                        n_opt += 1
            e.halt = hf.detach()
            e.k += 1
            if e.k >= K:
                live.remove(e)
                continue
            p = torch.sigmoid(logits.detach()).cpu().numpy().astype(np.float16)
            e.pending = pool.submit([(("tr", int(i)), train_insts[i], p[j], ls, mode) for j, i in enumerate(e.idx)])
        msg = " ".join(f"{k} {np.mean(v):.4f}" for k, v in sorted(hist.items()))
        log(f"[rloop-sup] ep {ep} {msg} | opt {n_opt} lr {sched.get_last_lr()[0]:.2e} LPs {pool.n_lp} hits {pool.n_hit} "
            f"lp_s {pool.t_lp:.0f} ({time.time() - t0:.0f}s)")
        rec = dict(epoch=ep, time=time.time() - t0, **{k: float(np.mean(v)) for k, v in hist.items()})
        if val_fn is not None:
            rec["val"] = val_fn(ema.shadow)
            log(f"[rloop-sup] ep {ep} val {rec['val']}")
        if ckpt is not None:
            ckpt(ep, net, ema, opt, sched, rec)
    return net, ema


def train_reinforce(net, feats, sysm, train_insts, labels, best_known, cfg, pool, device, log=print, val_fn=None,
                    ckpt=None, start_update=0, opt_state=None, ema_state=None, seed=0):
    """Cost-aware fine-tuning with the exact LP as critic. Per batch the trajectory follows the deterministic decoding
    (as at inference); at every outer step S Bernoulli samples of the step's probabilities are decoded and priced;
    reward = log(c_best_before / c_sample), the step's improvement over the best schedule so far (what keep-best
    inference gains), clipped below at -cfg['clip'] (cfg['reward'] = 'improve'), or -log(c_sample / c_rel) ('abs');
    step 0 always uses 'abs'. RLOO baseline over the samples; + cfg['bce_w'] x the supervised loss (anchor) +
    cfg['halt_w'] x the halting loss (target: the greedy schedule within 0.5 % of the best known cost). One optimizer
    step per outer step."""
    rng = np.random.default_rng(seed + 7 + start_update)
    gen = torch.Generator(device="cpu").manual_seed(seed + 7 + start_update)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["rl_lr"])
    if opt_state is not None:
        opt.load_state_dict(opt_state["opt"])
    ema = EMA(net, cfg["ema"])
    if ema_state is not None:
        ema.shadow.load_state_dict(ema_state)
    gi = GroupInfo(sysm)
    S, K, B = cfg["rl_samples"], cfg["rl_k"], cfg["rl_batch"]
    ls, mode = cfg.get("line_scale", 1.0), cfg.get("decode", "viterbi")
    hw = cfg.get("halt_w", 0.0)
    t0 = time.time()
    hist = collections.defaultdict(list)
    for upd in range(start_update, cfg["rl_updates"]):
        net.train()
        idx = rng.choice(len(train_insts), B, replace=False)
        insts = [train_insts[i] for i in idx]
        b = Batch(feats, insts, device, gi=gi)
        state, fb = None, empty_feedback(b)
        best = np.full(B, np.inf)
        c_rel = np.array([float(x["c_rel"]) for x in insts])
        u0 = np.stack([x["u0"] for x in insts])
        stk = np.stack([gi.stacked(labels[i], u0[j]) for j, i in enumerate(idx)]) if net.counts else None
        seeded = rng.random() < cfg.get("p_seed", 0.0)
        if seeded:      # start from the LP-rounding baseline's best schedule (5 thresholds, priced): refinement
            ths = (0.001, 0.05, 0.2, 0.5, 0.8)
            jobs = [(("tr", int(i)), insts[j], (insts[j]["u_rel"] > th).astype(np.int8), ls, "round")
                    for th in ths for j, i in enumerate(idx)]
            rs = pool.run(jobs)
            c5 = np.array([x["cost"] for x in rs]).reshape(len(ths), B)
            jb = c5.argmin(0)
            start = [rs[jb[j] * B + j] for j in range(B)]
            fb = stack_feedback(b, start, best, sysm.gbus)
            best = c5.min(0)
            hist["seed_gap"].append(float(np.mean(np.log(best / c_rel)) * 100))
        for k in range(K):
            xe = net.embed_static(b.ux)
            if state is None:
                state = net.init_state(xe)
            logits, hf, st = net.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, state)
            hlog = net.halt_logit(hf.detach())
            tau = cfg.get("rl_temp", 1.0)                       # sampling temperature (< 1: samples nearer greedy)
            p = torch.sigmoid(logits.detach()).cpu()
            smp = (torch.rand((S,) + p.shape, generator=gen) < torch.sigmoid(logits.detach().cpu() / tau))
            pn = p.numpy().astype(np.float16)
            jobs = [(("tr", int(i)), insts[j], pn[j], ls, mode) for j, i in enumerate(idx)]
            jobs += [(("tr", int(idx[j])), insts[j], smp[s_, j].numpy().astype(np.int8), ls, mode)
                     for s_ in range(S) for j in range(B)]
            res = pool.run(jobs)
            c = np.array([x["cost"] for x in res[B:]]).reshape(S, B)
            improve = cfg.get("reward", "improve") == "improve" and (k > 0 or seeded)
            rew = np.log((best if improve else c_rel)[None] / c)
            if improve:
                rew = np.maximum(rew, -cfg.get("clip", 0.05))
            rw = torch.as_tensor(rew, dtype=torch.float32)
            adv = rw - (rw.sum(0, keepdim=True) - rw) / (S - 1)
            adv = adv / (adv.std() + 1e-8)
            sm = smp.to(device)
            lg = logits.unsqueeze(0) / tau
            logp = (sm * F.logsigmoid(lg) + (~sm) * F.logsigmoid(-lg)).sum((-1, -2))   # [S, B]
            pg = -(adv.to(device) * logp).mean() / (b.T * b.G)
            r = (logits.detach() > 0).cpu().numpy().astype(np.int8)
            tgt = torch.as_tensor(targets_for(net, gi, sysm, labels[idx], r, u0, stk), device=device)
            sup = F.binary_cross_entropy_with_logits(logits, tgt)
            greedy = res[:B]
            gc = np.array([x["cost"] for x in greedy])
            ht = torch.as_tensor((gc <= 1.005 * best_known[idx]).astype(np.float32), device=device)
            hl = F.binary_cross_entropy_with_logits(hlog, ht)
            loss = pg + cfg.get("bce_w", 0.5) * sup + hw * hl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema.update(net)
            state = (st[0].detach(), st[1].detach())
            fb = stack_feedback(b, greedy, best, sysm.gbus)
            hist[f"gap{k}"].append(float(np.mean(np.log(gc / c_rel)) * 100))
            hist[f"smp{k}"].append(float(np.mean(np.log(c / c_rel[None])) * 100))
            best = np.minimum(best, gc)
            hist["bce"].append(sup.item())
            hist["halt"].append(hl.item())
        hist["best_seeded" if seeded else "best"].append(float(np.mean(np.log(best / c_rel)) * 100))
        if (upd + 1) % cfg.get("rl_log", 10) == 0:
            msg = " ".join(f"{kk} {np.mean(v):.3f}" for kk, v in sorted(hist.items()))
            log(f"[rloop-rl] upd {upd + 1} {msg} | LPs {pool.n_lp} hits {pool.n_hit} ({time.time() - t0:.0f}s)")
            hist = collections.defaultdict(list)
        if ckpt is not None and (upd + 1) % cfg.get("rl_ckpt", 25) == 0:
            rec = dict(update=upd + 1, time=time.time() - t0)
            if val_fn is not None:
                rec["val"] = val_fn(ema.shadow)
                log(f"[rloop-rl] upd {upd + 1} val {rec['val']}")
            ckpt(upd, net, ema, opt, None, rec)
    return net, ema


# ============================================================================ inference
@torch.no_grad()
def rollout_batch(net, feats, sysm, insts, K, pool, device, keys, line_scale=1.0, fmax=None, mode="viterbi", bias=0.0,
                  starts=None):
    """K outer steps on a batch of instances (validation, through the LP pool). Returns per step a list of per-instance
    (cost, shed, short, halting probability). bias: logit shift before decoding (> 0 commits more). starts: optional
    starting schedules [B, T, G] (0/1), priced first; their LP feedback feeds step 0 and they count as candidates
    (returned as step -1 first in the list)."""
    net.eval()
    b = Batch(feats, insts, device, fmax)
    xe = net.embed_static(b.ux)
    state = net.init_state(xe)
    fb = empty_feedback(b)
    best = np.full(len(insts), np.inf)
    steps = []
    if starts is not None:
        res = pool.run([(key, inst, np.asarray(s0, np.int8), line_scale, "round") for key, inst, s0 in zip(keys, insts, starts)])
        fb = stack_feedback(b, res, best, sysm.gbus)
        steps.append([(x["cost"], x["shed"], x["short"], 0.0) for x in res])
        best = np.minimum(best, [x["cost"] for x in res])
    for k in range(K):
        logits, hf, state = net.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, state)
        p = torch.sigmoid(logits + bias).cpu().numpy().astype(np.float16)
        h = torch.sigmoid(net.halt_logit(hf)).cpu().numpy()
        res = pool.run([(key, inst, pj, line_scale, mode) for key, inst, pj in zip(keys, insts, p)])
        fb = stack_feedback(b, res, best, sysm.gbus)
        steps.append([(x["cost"], x["shed"], x["short"], float(hh)) for x, hh in zip(res, h)])
        best = np.minimum(best, [x["cost"] for x in res])
    return steps


class Inference:
    """single-instance inference for timed evaluation: one process, one solver object per instance (warm-started
    across the instance's own LPs). rollout(inst, K) -> per-step records with cumulative time, and per step the best
    schedule so far with the probabilities it was decoded from."""

    def __init__(self, net, feats, sysm, lp: FastLP, device="cpu", mode="viterbi", bias=0.0):
        self.net, self.feats, self.s, self.lp, self.device, self.mode = net.eval(), feats, sysm, lp, device, mode
        self.bias = bias
        self.gi = GroupInfo(sysm)

    @torch.no_grad()
    def rollout(self, inst, K, t_pre=0.0, fmax=None, start=None):
        """record k: cost of step k's schedule, best so far, halting probability, cumulative time (t_pre, e.g. the LP
        relaxation, + featurisation + every network pass, decoding, LP and feedback build up to step k's LP)"""
        t0 = time.time()
        lp = self.lp
        lp.reset()
        b = Batch(self.feats, [inst], self.device, fmax, gi=self.gi)
        xe = self.net.embed_static(b.ux)
        state = self.net.init_state(xe)
        fb = empty_feedback(b)
        fm = (self.s.fmax if fmax is None else fmax).copy()
        fm[outs(inst, "l_out")] = 0
        sc = scenario_of(inst)
        best, best_u, best_rec, seen, prev, best_p = np.inf, None, None, {}, None, None
        recs, bests = [], []
        best_before = np.inf
        if start is not None:            # an external starting schedule (already decoded / repaired): priced first
            s0 = np.asarray(start, np.int8)
            sol = lp.solve(sc, s0)
            fu, fbb, fl, fh = feedback_arrays(sol, s0, s0, inst, self.s, fm)
            prev = dict(s=s0, cost=sol["obj"], shed=float(sol["shed"].sum() + sol["spill"].sum()),
                        short=float(sol["short"].sum()), fu=fu, fb=fbb, fl=fl, fh=fh)
            seen[s0.tobytes()] = prev
            best, best_u, best_rec, best_p = prev["cost"], s0, prev, s0.astype(float)
        for k in range(K):
            t1 = time.time()
            if prev is not None:
                fb = stack_feedback(b, [prev], np.array([best_before]), self.s.gbus)
            t2 = time.time()
            logits, hf, state = self.net.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, state)
            hl = self.net.halt_logit(hf)
            p = torch.sigmoid(logits[0] + self.bias).cpu().numpy()
            t3 = time.time()
            r, s_sched = decode(p, inst, self.s, self.mode)
            key = s_sched.tobytes()
            t4 = time.time()
            if key in seen:
                res, new = dict(seen[key]), False
                res["fu"] = res["fu"].copy()
                res["fu"][..., 1] = res["s"].astype(np.float32) - r
            else:
                sol = lp.solve(sc, s_sched)
                fu, fbb, fl, fh = feedback_arrays(sol, s_sched, r, inst, self.s, fm)
                res = dict(s=s_sched, cost=sol["obj"], shed=float(sol["shed"].sum() + sol["spill"].sum()),
                           short=float(sol["short"].sum()), fu=fu, fb=fbb, fl=fl, fh=fh)
                seen[key], new = res, True
            best_before = best
            if res["cost"] < best - 1e-9:
                best, best_u, best_rec, best_p = res["cost"], res["s"], res, p
            prev = res
            t5 = time.time()
            recs.append(dict(k=k, cost=res["cost"], shed=res["shed"], short=res["short"], new_lp=new, best=best,
                             best_shed=best_rec["shed"], best_short=best_rec["short"], halt=float(torch.sigmoid(hl)[0]),
                             t_fb=t2 - t1, t_net=t3 - t2, t_dec=t4 - t3, t_lp=t5 - t4, time=t_pre + (t5 - t0)))
            bests.append((best_u, best_p))
        return recs, bests


# ============================================================================ learning-free comparator
def lp_gradient_moves(s, grad, u0, sysm, top=64):
    """block moves on schedule s [T, G] ranked by the first-order cost change predicted by the LP sensitivity grad
    (d cost / d u): switch an off unit on for a block of its minimum up time starting at any hour, switch off a whole
    on-block, or trim one hour from either end of an on-block. Returns [(predicted change, new schedule)] (best
    first); the schedules are then made min up/down feasible by the caller."""
    T, G = s.shape
    moves = []
    for g in range(G):
        col = s[:, g]
        U = int(sysm.min_up[g])
        for t in range(T):
            if col[t] == 0:
                hi = min(T, t + U)
                new = col.copy()
                new[t:hi] = 1
                d = float(grad[t:hi, g][col[t:hi] == 0].sum())
                moves.append((d, g, new))
        t = 0
        while t < T:                                    # on-blocks
            if col[t] == 1:
                a = t
                while t < T and col[t] == 1:
                    t += 1
                b = t                                   # block [a, b)
                new = col.copy()
                new[a:b] = 0
                moves.append((-float(grad[a:b, g].sum()), g, new))
                if b - a > U:
                    for c in (a, b - 1):
                        new = col.copy()
                        new[c] = 0
                        moves.append((-float(grad[c, g]), g, new))
            else:
                t += 1
    moves.sort(key=lambda m: m[0])
    out = []
    for d, g, new in moves[:top]:
        u = s.copy()
        u[:, g] = new
        out.append((d, u))
    return out


def lp_local_search(inst, s0, K, lp: FastLP, sysm, t_budget=None):
    """learning-free refinement with the LP in the loop: from schedule s0, repeatedly try the best untried block move
    by the LP's first-order prediction (made min up/down feasible by the Hamming repair, then the block adequacy
    repair), price it, and accept it if it is cheaper. K = number of LPs after pricing s0. Returns per-LP records
    (cost, shed, short, best so far) and the best schedule."""
    t0 = time.time()
    sc = scenario_of(inst)
    lp.reset()
    sol = lp.solve(sc, s0)
    best, best_s, best_sol = sol["obj"], np.asarray(s0, np.int8), sol
    seen = {best_s.tobytes()}
    cand = lp_gradient_moves(best_s, sol["grad"], inst["u0"], sysm)
    recs = []
    while len(recs) < K and cand:
        d, u = cand.pop(0)
        if d >= 0:
            break
        v = repair_min_updown(u, inst["u0"], sysm.min_up, sysm.min_dn)
        v = adequacy_repair_blocks(v, inst["u0"], inst["load"], inst["avail"], inst["sr"], sysm)
        v = repair_min_updown(v, inst["u0"], sysm.min_up, sysm.min_dn).astype(np.int8)
        if v.tobytes() in seen:
            continue
        seen.add(v.tobytes())
        sol = lp.solve(sc, v)
        if sol["obj"] < best - 1e-9:
            best, best_s, best_sol = sol["obj"], v, sol
            cand = lp_gradient_moves(best_s, sol["grad"], inst["u0"], sysm)
        recs.append(dict(cost=sol["obj"], shed=float(sol["shed"].sum() + sol["spill"].sum()),
                         short=float(sol["short"].sum()), best=best,
                         best_shed=float(best_sol["shed"].sum() + best_sol["spill"].sum()),
                         best_short=float(best_sol["short"].sum()), time=time.time() - t0))
    return recs, best_s
