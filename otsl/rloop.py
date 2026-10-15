"""rloop: recursive unit-commitment model with the exact dispatch LP in the loop (arm B of docs/recursive/PLAN.md).

A small recurrent network over (unit, hour) tokens and the bus graph keeps an answer state y and a latent z (TRM).
After every outer step its answer is rounded, repaired (block adequacy + min up/down) and priced by the exact dispatch
LP; the LP's signals are fed back as inputs of the next step:

    step 0:  z, y <- recursion(x, no feedback)            -> logits_0 -> r_0 = round -> s_0 = repair(r_0) -> LP(s_0)
    step k:  z, y <- recursion(x, feedback(LP(s_{k-1})))  -> logits_k -> r_k -> s_k -> LP(s_k)
    output:  the cheapest priced schedule s_k (penalties included), so every step is a priced candidate.

Feedback (per outer step, from the LP of s): per unit-hour the priced schedule, what the repair changed, dispatch
relative to limits, reserve, the LMP at the unit's bus and the exact LP sensitivity d cost / d u (bound duals, with the
start-up / shut-down variables, as UCModel.dispatch_gradient); per bus-hour LMP, shed, spill; per line-hour loading,
flow-limit dual, binding flag; per hour energy and reserve shortfall, reserve price, capacity and minimum-output
margins; per instance the cost of s and of the best schedule so far relative to the LP relaxation.

Components:
  FastLP         dispatch LP / LP relaxation through highspy on the matrices of otsl.ood.OODModel (outages supported),
                 warm-started across calls on one solver object; exact same model as UCModel.solve_dispatch.
  StaticFeats    length-agnostic static inputs (unit-hour, bus-hour, line-hour), normalised on training data.
  RLoopNet       the recurrent network (recall, TRM answer/latent recursion, temporal + unit-pool + bus-graph mixing).
  LPPool         repair + dispatch LP + feedback in CPU worker processes (or in process), memoised.
  train_supervised / train_reinforce   deep supervision with LP feedback of the model's own rounding; cost-aware
                 fine-tuning with the exact LP as critic (REINFORCE, RLOO baseline).
  rollout        inference with K outer steps on one instance, keeping the best schedule, timed per step.

Everything runs on CPU or CUDA (`device`); the LPs always run on CPU (in the pool's worker processes).
"""
from __future__ import annotations

import collections
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
from .uc import RES_SHORT, VOLL, load_rts_gmlc, repair_min_updown

# ============================================================================ presets
PRESETS = {
    # CPU feasibility study (this container: one core per process, torch single-threaded)
    "cpu": dict(hidden=64, n_latent=2, n_cycles=1, mp_layers=1, dilations=(1, 2, 4), k_train=3, batch=32, epochs=12,
                lr=1e-3, ema=0.995, lp_workers=0, inflight=1, k_eval=8),
    # larger GPU configuration (e.g. one GPU + 16-32 CPU cores for the LP workers in the same job)
    "gpu": dict(hidden=160, n_latent=3, n_cycles=2, mp_layers=2, dilations=(1, 2, 4, 8), k_train=6, batch=64, epochs=60,
                lr=7e-4, ema=0.999, lp_workers=16, inflight=4, k_eval=16),
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
        N, G = sysm.n_bus, sysm.G
        self.kcl = np.array([[m.row_tag[("kcl", t, n)][0] for n in range(N)] for t in range(T)])
        self.srr = np.array([m.row_tag[("sr", t)][0] for t in range(T)])
        self.S = sysm.seg_w.shape[1]

    def reset(self):
        self.h = None

    def _get(self, name, v):
        o, k = self.m.off[name]
        return v[o:o + k * self.T].reshape(self.T, -1)

    def solve(self, sc, u=None, relax=False):
        hs, inf = self.hs, self.hs.kHighsInf
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


# ============================================================================ instances
def scenario_of(inst):
    """OODScenario from an instance dict (load, avail, u0, sr [, l_out, g_out])"""
    return OODScenario(load=inst["load"], avail=inst["avail"], u0=inst["u0"], sr=inst["sr"],
                       l_out=tuple(int(x) for x in np.atleast_1d(inst.get("l_out", ())) if x >= 0),
                       g_out=tuple(int(x) for x in np.atleast_1d(inst.get("g_out", ())) if x >= 0))


def instances_from(d, idx=None, extra=()):
    """list of instance dicts from a stacked dataset dict d (keys load, avail, u0, sr, u_rel, lmp_rel, flow_rel, c_rel)"""
    n = len(d["load"])
    idx = range(n) if idx is None else idx
    keys = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel") + tuple(extra)
    return [{k: d[k][i] for k in keys if k in d} for i in idx]


def repair(r, inst, sysm):
    """rounded schedule -> block (min up/down-aware) adequacy repair -> min up/down repair (otsl.b3.rep_block)"""
    v = adequacy_repair_blocks(np.asarray(r, np.int8), inst["u0"], inst["load"], inst["avail"], inst["sr"], sysm)
    return repair_min_updown(v, inst["u0"], sysm.min_up, sysm.min_dn).astype(np.int8)


# ============================================================================ features
F_U, F_B, F_L, F_H, F_I = 9, 3, 3, 6, 4
FB_UNIT = F_U + F_B + F_H + F_I + 1          # unit-token feedback width (+ "feedback present" flag)


def feedback_arrays(sol, s_sched, r, inst, sysm, fmax):
    """LP solution of schedule s_sched (repaired from the rounding r) -> feedback arrays (float32):
    fu [T, G, F_U], fb [T, N, F_B], fl [T, L, F_L], fh [T, F_H]"""
    s = sysm
    T, G = s_sched.shape
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
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(fmax[None] > 0, np.abs(sol["flow"]) / np.maximum(fmax[None], 1e-9), 0.0)
    fl = np.stack([ld, slog(sol["fdual"]), ld >= 1 - 1e-6], -1)
    net = load - inst["avail"].sum(1)
    capm = ((sf * s.pmax[None]).sum(1) - net - inst["sr"]) / load
    ming = (load - (sf * s.pmin[None]).sum(1)) / load
    fh = np.stack([sol["shed"].sum(1), sol["spill"].sum(1), sol["short"], slog(sol["rprice"]), capm, ming], -1)
    return [a.astype(np.float32) for a in (fu, fb, fl, fh)]


def instance_feats(cost, best, c_rel, served, is_best):
    """per-instance feedback: log cost ratios (in %) to the LP relaxation, served, is-best flags"""
    return np.array([min(100 * math.log(max(cost, 1.0) / c_rel), 50.0) / 10, min(100 * math.log(max(best, 1.0) / c_rel), 50.0) / 10,
                     float(served), float(is_best)], np.float32)


class StaticFeats:
    """Length-agnostic static inputs. unit-hour [T, G, DX]: unit data (limits, costs, min up/down, ramp, type), initial
    status, symmetry rank and group size, hourly system data (load, net load, reserve, renewables), distance to the
    horizon's ends (clipped at 12 h), LP-relaxation commitment and price at the unit's bus, load / renewables at the
    unit's bus. bus-hour [T, N, DB]: load, renewables, capacity, relaxed price. line-hour [T, L, DL]: log susceptance,
    rating, relaxed loading, in service. Normalised with training means / stds (flags excluded)."""

    def __init__(self, sysm, train_insts, n_stat=400):
        from .ucml import TYPES
        s = sysm
        self.s = s
        onehot = np.array([[t == k for k in TYPES] for t in s.utype], float)
        avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
        self.g_static = np.c_[s.pmax, s.pmin, avg / 1e4, s.seg_c / 1e4, s.c_su / 1e4, s.min_up / 24, s.min_dn / 24,
                              np.minimum(s.ramp / s.pmax, 5), onehot].astype(np.float32)
        self.Rb = np.zeros((s.n_bus, len(s.r_pmax)))
        self.Rb[s.r_bus, np.arange(len(s.r_pmax))] = 1
        self.capbus = np.bincount(s.gbus, weights=s.pmax, minlength=s.n_bus)
        self.groups = s.identical_groups()
        xs = [self.raw(inst, s.fmax) for inst in train_insts[:n_stat]]
        self.stats = []
        for j in range(3):
            a = np.concatenate([x[j].reshape(-1, x[j].shape[-1]) for x in xs])
            mu, sd = a.mean(0), a.std(0) + 1e-6
            self.stats.append((mu.astype(np.float32), sd.astype(np.float32)))
        self.dx, self.db, self.dl = (x.shape[-1] for x in xs[0])

    def sym(self, u0):
        G = self.s.G
        rank, size = np.zeros(G), np.ones(G)
        for grp in self.groups:
            for val in (0, 1):
                mem = [g for g in grp if u0[g] == val]
                for r, g in enumerate(mem):
                    rank[g], size[g] = r, len(mem)
        return rank, size

    def raw(self, inst, fmax):
        s = self.s
        load, avail, u0 = inst["load"], inst["avail"], inst["u0"]
        T, G = load.shape[0], s.G
        ren = avail @ self.Rb.T                                            # [T, N]
        rank, size = self.sym(u0)
        hour = np.stack([load.sum(1), load.sum(1) - avail.sum(1), inst["sr"], avail.sum(1)], -1)   # [T, 4]
        t = np.arange(T)
        edge = np.stack([np.minimum(t, 12) / 12, np.minimum(T - 1 - t, 12) / 12], -1)
        rel_p = slog(inst["lmp_rel"])
        ux = np.concatenate([
            np.broadcast_to(self.g_static[None], (T, G, self.g_static.shape[1])),
            np.broadcast_to(np.stack([u0, rank, size], -1)[None], (T, G, 3)),
            np.broadcast_to(hour[:, None], (T, G, 4)), np.broadcast_to(edge[:, None], (T, G, 2)),
            inst["u_rel"][..., None], rel_p[:, s.gbus][..., None], load[:, s.gbus][..., None], ren[:, s.gbus][..., None]],
            -1)
        bx = np.stack([load, ren, np.broadcast_to(self.capbus[None], load.shape), rel_p], -1)
        lo = np.zeros(s.L)
        l_out = [int(x) for x in np.atleast_1d(inst.get("l_out", ())) if x >= 0]
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
        l_out = [int(x) for x in np.atleast_1d(inst.get("l_out", ())) if x >= 0]
        gate = np.ones(self.s.L, np.float32)
        gate[l_out] = 0
        return out + [gate]


# ============================================================================ the network
def _mlp(i, h, o, ln=True):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h) if ln else nn.Identity(), nn.SiLU(), nn.Linear(h, o))


class Core(nn.Module):
    """One shared recursion block on tokens h [B, T, G, H]: temporal mixing per unit (residual dilated 1-D
    convolutions over hours, length-agnostic), mixing across units in the same hour (DeepSets mean pool), message
    passing over the bus graph per hour (units summed into their buses, interaction-network layers over the lines with
    edge gates for lines out of service), and a token MLP."""

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
    feedback embedding + y. Logits from y. recall=False: the x-embedding only initialises z."""

    def __init__(self, sysm, dx, db, dl, hidden=64, n_latent=2, n_cycles=1, mp_layers=1, dilations=(1, 2, 4),
                 recall=True, feedback=True, carry_z=True, **_):
        super().__init__()
        H = hidden
        self.H, self.n_latent, self.n_cycles = H, n_latent, n_cycles
        self.recall, self.feedback, self.carry_z = recall, feedback, carry_z
        self.register_buffer("gbus", torch.as_tensor(sysm.gbus, dtype=torch.long))
        self.ex = _mlp(dx, H, H)
        self.efb = _mlp(FB_UNIT, H, H)
        self.eb = _mlp(db + F_B + 1, H, H)
        self.el = _mlp(dl + F_L + 1, H, H)
        self.ey = nn.Linear(H, H)
        self.core = Core(sysm, H, mp_layers, dilations)
        self.yupd = _mlp(2 * H, H, H)
        self.head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 1))
        self.y0 = nn.Parameter(torch.zeros(H))
        self.z0 = nn.Parameter(torch.zeros(H))

    def embed_static(self, ux):
        return self.ex(ux)

    def init_state(self, xe):
        B, T, G, H = xe.shape
        y = self.y0.expand(B, T, G, H)
        z = xe if not self.recall else self.z0.expand(B, T, G, H)
        return y.contiguous(), z.contiguous()

    def step(self, xe, bx, lx, gate, fbu, fbb, fbl, state, grad_cycles=1):
        """one outer step. xe [B,T,G,H] embedded static unit-hour input; bx [B,T,N,DB]; lx [B,T,L,DL]; gate [B,L];
        fbu [B,T,G,FB_UNIT], fbb [B,T,N,F_B+1], fbl [B,T,L,F_L+1] feedback (zeros + flag 0 at step 0).
        Returns logits [B,T,G] and the new (y, z). Only the last grad_cycles cycles carry gradients (TRM)."""
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
            ctx = torch.enable_grad() if c >= self.n_cycles - grad_cycles else torch.no_grad()
            with ctx:
                for _ in range(self.n_latent):
                    z = self.core(z + stat + fe + self.ey(y), be, le, gate)
                y = y + self.yupd(torch.cat([y, z], -1))
        return self.head(y).squeeze(-1), (y, z)


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
    _WK["cfg"] = cfg
    _WK["sys"] = {}
    _WK["lp"] = {}


def _wk_lp(line_scale, T):
    key = (round(float(line_scale), 4), int(T))
    if key not in _WK["lp"]:
        if key[0] not in _WK["sys"]:
            _WK["sys"][key[0]] = load_rts_gmlc(line_scale=key[0])
        _WK["lp"][key] = FastLP(_WK["sys"][key[0]], key[1])
    return _WK["lp"][key]


def lp_feedback_job(job):
    """(inst, r, line_scale, cold) -> dict(s, cost, shed, short, fb arrays as float16, t_repair, t_lp).
    inst: instance dict (load, avail, u0, sr, l_out, g_out); r: rounded schedule [T, G] (int8)."""
    inst, r, line_scale, cold = job
    lp = _wk_lp(line_scale, inst["load"].shape[0])
    if cold:
        lp.reset()
    t0 = time.time()
    s_sched = repair(r, inst, lp.s)
    g_out = [int(x) for x in np.atleast_1d(inst.get("g_out", ())) if x >= 0]
    if g_out:
        s_sched[:, g_out] = 0
    t1 = time.time()
    sol = lp.solve(scenario_of(inst), s_sched)
    fmax = lp.s.fmax.copy()
    l_out = [int(x) for x in np.atleast_1d(inst.get("l_out", ())) if x >= 0]
    fmax[l_out] = 0
    fu, fb, fl, fh = feedback_arrays(sol, s_sched, r, inst, lp.s, fmax)
    return dict(s=s_sched, cost=sol["obj"], shed=float(sol["shed"].sum() + sol["spill"].sum()),
                short=float(sol["short"].sum()), fu=fu.astype(np.float16), fb=fb.astype(np.float16),
                fl=fl.astype(np.float16), fh=fh.astype(np.float16), t_repair=t1 - t0, t_lp=sol["lp_s"])


class LPPool:
    """Repair + dispatch LP + feedback for many (instance, rounding) pairs: in `workers` spawn processes pinned to
    `cores` (CPU), or in process (workers = 0). Results memoised by (instance key, rounding)."""

    def __init__(self, workers=0, cores=None, cache=6000):
        self.workers = workers
        cfg = dict(cores=cores)
        if workers > 0:
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_wk_init, initargs=(cfg,))
        else:
            self.pool = None
            if not _WK:
                _wk_init(dict(cores=None))
        self.cache = collections.OrderedDict()
        self.cap = cache
        self.n_lp, self.n_hit, self.t_lp = 0, 0, 0.0

    def submit(self, jobs, keys):
        """jobs: [(inst, r, line_scale)], keys: hashable instance keys. Returns a handle for get()."""
        ck = [(k, np.asarray(j[1], np.int8).tobytes()) for k, j in zip(keys, jobs)]
        todo, seen = [], set()
        for c, j in zip(ck, jobs):
            if c not in self.cache and c not in seen:
                seen.add(c)
                todo.append((c, (j[0], np.asarray(j[1], np.int8), j[2], False)))
        self.n_hit += len(ck) - len(todo)
        if self.pool is not None and todo:
            ar = self.pool.map_async(lp_feedback_job, [t[1] for t in todo], chunksize=max(1, len(todo) // (4 * self.workers)))
        else:
            ar = None
        return dict(ck=ck, todo=todo, ar=ar)

    def ready(self, hd):
        return hd["ar"] is None or hd["ar"].ready()

    def get(self, hd):
        if hd["todo"]:
            res = hd["ar"].get() if hd["ar"] is not None else [lp_feedback_job(t[1]) for t in hd["todo"]]
            for (c, _), r in zip(hd["todo"], res):
                self.cache[c] = r
                self.t_lp += r["t_lp"]
            self.n_lp += len(hd["todo"])
        out = []
        for c in hd["ck"]:
            self.cache.move_to_end(c)
            out.append(self.cache[c])
        while len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return out

    def run(self, jobs, keys):
        return self.get(self.submit(jobs, keys))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# ============================================================================ batching
class Batch:
    """static inputs of a list of instances as tensors on `device` (all instances with the same horizon)"""

    def __init__(self, feats, insts, device, fmax=None):
        xs = [feats(inst, fmax) for inst in insts]
        f = lambda j: torch.as_tensor(np.stack([x[j] for x in xs]), device=device)
        self.ux, self.bx, self.lx, self.gate = f(0), f(1), f(2), f(3)
        self.insts = insts
        self.B, self.T, self.G = self.ux.shape[0], self.ux.shape[1], self.ux.shape[2]
        self.N, self.L = self.bx.shape[2], self.lx.shape[2]
        self.device = device


def empty_feedback(b: Batch):
    z = lambda *sh: torch.zeros(*sh, device=b.device)
    return z(b.B, b.T, b.G, FB_UNIT), z(b.B, b.T, b.N, F_B + 1), z(b.B, b.T, b.L, F_L + 1)


def stack_feedback(b: Batch, res, best, gbus):
    """feedback tensors from LP results (one per instance) and the best cost so far (per instance)"""
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


def aligned_targets(sysm, groups, y, r, u0):
    """symmetry-aware targets: labels permuted inside groups of identical units to agree best with the rounding r"""
    return np.stack([align_to_prediction(sysm, y[i], r[i], u0[i], groups) for i in range(len(y))]).astype(np.float32)


# ============================================================================ starting schedules (progressive training)
def perturb_label(y, inst, sysm, rng, k_mean=3.0):
    """a corrupted schedule near the label: k ~ Poisson units switched off for a random block of hours and k units
    switched on for a random block (then repaired by the LP worker)"""
    u = np.array(y, np.int8, copy=True)
    T, G = u.shape
    for _ in range(rng.poisson(k_mean)):
        g = rng.integers(G)
        a = rng.integers(T)
        b = min(T, a + rng.integers(1, T + 1))
        u[a:b, g] = 0
    for _ in range(rng.poisson(k_mean)):
        g = rng.integers(G)
        a = rng.integers(T)
        b = min(T, a + rng.integers(1, T + 1))
        u[a:b, g] = 1
    return u


def external_start(inst, y, sysm, rng):
    """an arbitrary starting rounding: the LP relaxation rounded at a random threshold, or a perturbed label"""
    if rng.random() < 0.5:
        th = rng.choice([0.05, 0.2, 0.5, 0.8, 0.95])
        return (inst["u_rel"] > th).astype(np.int8)
    return perturb_label(y, inst, sysm, rng)


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


def _bce(logits, target, w_on=1.0):
    w = torch.where(target > 0.5, w_on, 1.0) if w_on != 1.0 else None
    return F.binary_cross_entropy_with_logits(logits, target, weight=w)


class Episode:
    """one in-flight training batch: instances, static batch, carried state, step index, pending LP handle"""

    def __init__(self, idx, batch, y, state, k_total, ext):
        self.idx, self.b, self.y, self.state, self.k, self.K = idx, batch, y, state, 0, k_total
        self.ext = ext
        self.pending = None
        self.fb = None
        self.best = np.full(len(idx), np.inf)
        self.losses = []


def train_supervised(net, feats, sysm, train_insts, labels, cfg, pool, device, log=print, val_fn=None, ckpt=None,
                     start_epoch=0, opt_state=None, ema_state=None, seed=0):
    """Deep supervision with LP feedback of the model's own rounding (on-policy). Per batch: K = cfg['k_train'] outer
    steps; step k's loss is BCE to the label aligned (inside identical-unit groups) to step k's rounding; with deep
    supervision (cfg['deep_sup'], default) the optimizer steps after every outer step and (y, z) are detached (TRM);
    without it, the loss of the last step only is back-propagated through all steps. With probability cfg['p_ext'] a
    batch starts from an arbitrary schedule (LP relaxation rounded at a random threshold, or a perturbed label) with its
    LP feedback instead of the model's own step 0 (progressive training: improve from any starting point).
    cfg['inflight'] batches are kept in flight so the network trains while the LP workers solve."""
    rng = np.random.default_rng(seed + 1000 * start_epoch)
    torch.manual_seed(seed + 1000 * start_epoch)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    total_steps = cfg["epochs"] * math.ceil(len(train_insts) / cfg["batch"]) * cfg["k_train"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    if opt_state is not None:
        opt.load_state_dict(opt_state["opt"])
        sched.load_state_dict(opt_state["sched"])
    ema = EMA(net, cfg["ema"])
    if ema_state is not None:
        ema.shadow.load_state_dict(ema_state)
    groups = sysm.identical_groups()
    gbus = sysm.gbus
    K = cfg["k_train"]
    deep_sup = cfg.get("deep_sup", True)
    ls = cfg.get("line_scale", 1.0)
    w_on = cfg.get("w_on", 1.0)
    for ep in range(start_epoch, cfg["epochs"]):
        net.train()
        t0 = time.time()
        perm = rng.permutation(len(train_insts))
        batches = [perm[i:i + cfg["batch"]] for i in range(0, len(perm), cfg["batch"])]
        queue = collections.deque(batches)
        live = []
        hist = collections.defaultdict(list)
        n_opt = 0

        def start(idx):
            insts = [train_insts[i] for i in idx]
            b = Batch(feats, insts, device)
            xe = None
            ext = rng.random() < cfg.get("p_ext", 0.0)
            e = Episode(idx, b, labels[idx], None, K, ext)
            if ext:
                rs = np.stack([external_start(insts[j], labels[i], sysm, rng) for j, i in enumerate(idx)])
                e.pending = pool.submit([(inst, r, ls) for inst, r in zip(insts, rs)], [("tr", int(i)) for i in idx])
            return e

        while queue or live:
            while queue and len(live) < cfg.get("inflight", 1):
                live.append(start(queue.popleft()))
            # pick an episode whose LP results are ready (else the oldest, and wait)
            e = next((x for x in live if x.pending is None or pool.ready(x.pending)), live[0])
            if e.pending is not None:
                res = pool.get(e.pending)
                e.pending = None
                e.fb = stack_feedback(e.b, res, e.best, gbus)
                for j, r in enumerate(res):
                    e.best[j] = min(e.best[j], r["cost"])
            b = e.b
            xe = net.embed_static(b.ux)
            if e.state is None:
                e.state = net.init_state(xe)
            fb = e.fb if e.fb is not None else empty_feedback(b)
            logits, state = net.step(xe, b.bx, b.lx, b.gate, *fb, e.state)
            r = (logits.detach() > 0).cpu().numpy().astype(np.int8)
            tgt = torch.as_tensor(aligned_targets(sysm, groups, e.y, r, np.stack([x["u0"] for x in b.insts])), device=device)
            loss = _bce(logits, tgt, w_on)
            hist[f"bce{e.k}"].append(loss.item())
            hist[f"err{e.k}"].append(float((r != tgt.cpu().numpy()).sum((1, 2)).mean()))
            if deep_sup:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                ema.update(net)
                n_opt += 1
                e.state = (state[0].detach(), state[1].detach())
            else:
                e.state = state
                if e.k == K - 1:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    for _ in range(K):
                        sched.step()
                    ema.update(net)
                    n_opt += 1
            e.k += 1
            if e.k >= K:
                live.remove(e)
                continue
            e.pending = pool.submit([(inst, rr, ls) for inst, rr in zip(b.insts, r)], [("tr", int(i)) for i in e.idx])
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


def train_reinforce(net, feats, sysm, train_insts, labels, cfg, pool, device, log=print, val_fn=None, ckpt=None,
                    start_update=0, opt_state=None, ema_state=None, seed=0):
    """Cost-aware fine-tuning with the exact LP as critic. Per batch the trajectory follows the greedy rounding (as at
    inference); at every outer step S Bernoulli samples of the step's probabilities are repaired and priced, reward =
    log(c_best_before / c_sample) (cfg['reward'] = 'improve': the improvement over the best schedule so far, the
    quantity keep-best inference gains; clipped below at -cfg['clip']) or -log(c_sample / c_rel) ('abs'); RLOO baseline
    over the samples; plus cfg['bce_w'] x the supervised loss (anchor). One optimizer step per outer step."""
    rng = np.random.default_rng(seed + 7 + start_update)
    gen = torch.Generator(device="cpu").manual_seed(seed + 7 + start_update)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["rl_lr"])
    if opt_state is not None:
        opt.load_state_dict(opt_state["opt"])
    ema = EMA(net, cfg["ema"])
    if ema_state is not None:
        ema.shadow.load_state_dict(ema_state)
    groups = sysm.identical_groups()
    S, K, B = cfg["rl_samples"], cfg["rl_k"], cfg["rl_batch"]
    ls = cfg.get("line_scale", 1.0)
    t0 = time.time()
    hist = collections.defaultdict(list)
    for upd in range(start_update, cfg["rl_updates"]):
        net.train()
        idx = rng.choice(len(train_insts), B, replace=False)
        insts = [train_insts[i] for i in idx]
        keys = [("tr", int(i)) for i in idx]
        b = Batch(feats, insts, device)
        state = None
        fb = empty_feedback(b)
        best = np.full(B, np.inf)
        c_rel = np.array([float(x["c_rel"]) for x in insts])
        u0 = np.stack([x["u0"] for x in insts])
        for k in range(K):
            xe = net.embed_static(b.ux)
            if state is None:
                state = net.init_state(xe)
            logits, st = net.step(xe, b.bx, b.lx, b.gate, *fb, state)
            p = torch.sigmoid(logits.detach()).cpu()
            r = (p > 0.5).numpy().astype(np.int8)
            smp = (torch.rand((S,) + p.shape, generator=gen) < p)                # [S, B, T, G]
            jobs = [(inst, rr, ls) for inst, rr in zip(insts, r)]
            jobs += [(insts[j], smp[s_, j].numpy().astype(np.int8), ls) for s_ in range(S) for j in range(B)]
            res = pool.run(jobs, keys + [keys[j] for s_ in range(S) for j in range(B)])
            c = np.array([x["cost"] for x in res[B:]]).reshape(S, B)
            ref = best if (cfg.get("reward", "improve") == "improve" and k > 0) else c_rel
            rew = np.log(ref[None] / c)
            if cfg.get("reward", "improve") == "improve" and k > 0:
                rew = np.maximum(rew, -cfg.get("clip", 0.05))
            rw = torch.as_tensor(rew, dtype=torch.float32)
            adv = rw - (rw.sum(0, keepdim=True) - rw) / (S - 1)
            adv = adv / (adv.std() + 1e-8)
            lg = logits.unsqueeze(0)
            sm = smp.to(device)
            logp = (sm * F.logsigmoid(lg) + (~sm) * F.logsigmoid(-lg)).sum((-1, -2))   # [S, B]
            pg = -(adv.to(device) * logp).mean() / (b.T * b.G)
            tgt = torch.as_tensor(aligned_targets(sysm, groups, labels[idx], r, u0), device=device)
            sup = _bce(logits, tgt)
            loss = pg + cfg.get("bce_w", 0.5) * sup
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema.update(net)
            state = (st[0].detach(), st[1].detach())
            greedy = res[:B]
            fb = stack_feedback(b, greedy, best, sysm.gbus)
            gc = np.array([x["cost"] for x in greedy])
            hist[f"gap{k}"].append(float(np.mean(np.log(gc / c_rel)) * 100))
            best = np.minimum(best, gc)
            hist["best"].append(float(np.mean(np.log(best / c_rel)) * 100)) if k == K - 1 else None
            hist[f"smp{k}"].append(float(np.mean(np.log(c / c_rel[None])) * 100))
            hist["bce"].append(sup.item())
        if (upd + 1) % cfg.get("rl_log", 10) == 0:
            msg = " ".join(f"{kk} {np.mean(v[-cfg.get('rl_log', 10) * 1:]):.3f}" for kk, v in sorted(hist.items()))
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
def rollout_batch(net, feats, sysm, insts, K, pool, device, keys, line_scale=1.0, fmax=None, th=0.0):
    """K outer steps on a batch of instances (validation): returns per instance the list of per-step LP results
    (cost, shed, short) and the best index. Uses the LP pool (memoised)."""
    net.eval()
    b = Batch(feats, insts, device, fmax)
    xe = net.embed_static(b.ux)
    state = net.init_state(xe)
    fb = empty_feedback(b)
    best = np.full(len(insts), np.inf)
    steps = []
    for k in range(K):
        logits, state = net.step(xe, b.bx, b.lx, b.gate, *fb, state)
        r = (logits > th).cpu().numpy().astype(np.int8)
        res = pool.run([(inst, rr, line_scale) for inst, rr in zip(insts, r)], keys)
        fb = stack_feedback(b, res, best, sysm.gbus)
        steps.append([(x["cost"], x["shed"], x["short"]) for x in res])
        best = np.minimum(best, [x["cost"] for x in res])
    return steps


class Inference:
    """single-instance inference for timed evaluation: one process, one solver object per instance (warm-started
    across the instance's own LPs). rollout(inst, K) -> per-step records with cumulative time."""

    def __init__(self, net, feats, sysm, lp: FastLP, device="cpu"):
        self.net, self.feats, self.s, self.lp, self.device = net.eval(), feats, sysm, lp, device

    @torch.no_grad()
    def rollout(self, inst, K, t_pre=0.0, fmax=None, th=0.0):
        """K outer steps; record k has the cost of step k's schedule, the best so far, and the cumulative time (t_pre,
        e.g. the LP relaxation, + featurisation + every network pass, repair, LP and feedback build up to step k's LP)"""
        t0 = time.time()
        lp = self.lp
        lp.reset()
        b = Batch(self.feats, [inst], self.device, fmax)
        xe = self.net.embed_static(b.ux)
        state = self.net.init_state(xe)
        fb = empty_feedback(b)
        fm = (self.s.fmax if fmax is None else fmax).copy()
        l_out = [int(x) for x in np.atleast_1d(inst.get("l_out", ())) if x >= 0]
        fm[l_out] = 0
        g_out = [int(x) for x in np.atleast_1d(inst.get("g_out", ())) if x >= 0]
        sc = scenario_of(inst)
        best, best_u, best_rec, seen, prev = np.inf, None, None, {}, None
        recs = []
        for k in range(K):
            t1 = time.time()
            if prev is not None:
                fb = stack_feedback(b, [prev], np.array([best]), self.s.gbus)
            t2 = time.time()
            logits, state = self.net.step(xe, b.bx, b.lx, b.gate, *fb, state)
            r = (logits[0] > th).cpu().numpy().astype(np.int8)
            if g_out:
                r[:, g_out] = 0
            t3 = time.time()
            s_sched = repair(r, inst, self.s)
            if g_out:
                s_sched[:, g_out] = 0
            key = s_sched.tobytes()
            t4 = time.time()
            if key in seen:
                res, new = dict(seen[key]), False
                res["fu"] = res["fu"].copy()
                res["fu"][..., 1] = res["s"].astype(np.float32) - r          # what the repair changed this time
            else:
                sol = lp.solve(sc, s_sched)
                fu, fbb, fl, fh = feedback_arrays(sol, s_sched, r, inst, self.s, fm)
                res = dict(s=s_sched, cost=sol["obj"], shed=float(sol["shed"].sum() + sol["spill"].sum()),
                           short=float(sol["short"].sum()), fu=fu, fb=fbb, fl=fl, fh=fh)
                seen[key], new = res, True
            if res["cost"] < best - 1e-9:
                best, best_u, best_rec = res["cost"], res["s"], res
            prev = res
            t5 = time.time()
            recs.append(dict(k=k, cost=res["cost"], shed=res["shed"], short=res["short"], new_lp=new, best=best,
                             best_shed=best_rec["shed"], best_short=best_rec["short"], t_fb=t2 - t1, t_net=t3 - t2,
                             t_repair=t4 - t3, t_lp=t5 - t4, time=t_pre + (t5 - t0)))
        return recs, best_u
