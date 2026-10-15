"""Tiny Recursive Model (TRM) for end-to-end unit commitment on the 12-hour RTS-GMLC benchmark (arm A of
docs/recursive/PLAN.md).

Tokens: one per (unit, hour), [B, T, G, D]. One tiny shared network `Net` of `layers` layers; each layer mixes
  * across units within an hour (`UnitMix`: mean and capacity-weighted pools, plus a merit-order prefix pool, i.e. the
    capacity-weighted state of the cheaper units; or multi-head attention for larger runs),
  * across hours for a unit (`TimeMix`: depthwise convolution along the hours, length-agnostic),
  * over the transmission graph through the units' buses (`GraphMix`: unit -> bus aggregation, the bus inputs, `hops`
    steps of the normalised bus adjacency, bus -> unit),
  * across channels (SwiGLU MLP),
each sub-block residual with a post RMS norm (as TRM / HRM). Every operation is equivariant to permuting identical
units (same parameters, bus and initial status), which are told apart only by the symmetry-rank input feature, so
augmenting with permutations of identical units is a no-op by construction (tests/test_trm.py checks it).

Recursion (TRM, Jolicoeur-Martineau 2025): answer embedding y and latent z;
    latent_recursion:  z <- net(x + y + z) n times, then y <- net(y + z)
    deep_recursion:    T - 1 latent recursions without gradient, one with gradient
    deep supervision:  up to n_sup steps per instance, (y, z) carried detached, BCE on every step's answer
                       and a halting head (BCE on "the rounded answer has at most halt_tol wrong decisions"),
                       instances leave the training batch when they halt (ACT with the single-pass halting of TRM)
                       and are replaced by fresh ones; EMA of the weights.
Controls (same blocks): `oneshot` (the same 2-layer net applied once, equal parameters) and `deep` (T (n + 1) separate
2-layer blocks applied once: the compute of one supervision step without weight sharing).

Training signal: canonical labels inside groups of identical units (otsl.ucml.canonical_labels) on the 500 MILP + 1,500
teacher-polished labels of the m1x study; second stage REINFORCE with the exact dispatch LP as critic on the final
answer (after the block adequacy and min up/down repairs used at inference), with a BCE anchor to the labels.

Inference: recursion -> threshold -> block adequacy repair -> min up/down repair -> dispatch LP; or screening of the
schedules of the last supervision steps (or of several thresholds) with <= 5 LPs.

Everything is device-agnostic (`device="cuda"` or `"cpu"`); `PRESETS` holds the CPU feasibility setting and TRM's
paper scale.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constrained import adequacy_repair_blocks
from .uc import repair_min_updown
from .ucml import canonical_labels, symmetry_rank

ROOT = os.path.join("data", "generated", "uc12")
M1X = os.path.join("data", "generated", "uc12_m1x")
VAL_SPLITS = ("val", "val_extra", "val_extra2")
DATA_KEYS = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel")


# ============================================================================ configuration
@dataclass
class TRMConfig:
    # architecture
    hidden: int = 64
    layers: int = 2                 # layers of the shared network (one "net call")
    expansion: float = 2.0          # SwiGLU width / hidden
    unit_mix: str = "pool"          # pool | attn
    heads: int = 4                  # attention heads (unit_mix = attn)
    time_kernel: int = 5            # depthwise convolution along the hours
    hops: int = 2                   # bus-graph propagation steps per layer
    lp_feats: bool = True           # LP-relaxation inputs (commitment, price, line loading)
    # recursion
    mode: str = "trm"               # trm | oneshot | deep
    n: int = 6                      # latent updates per cycle
    T: int = 3                      # cycles per supervision step (T - 1 without gradient)
    n_sup: int = 16                 # deep-supervision steps (training); 1 = deep supervision off
    halt_train: bool = True         # ACT: instances leave the batch when the halting head says so
    halt_tol: int = 2               # halting target: rounded answer has <= halt_tol wrong decisions
    halt_explore: float = 0.1       # prob. of a random minimum number of steps in [2, n_sup] (HRM / TRM)
    halt_weight: float = 0.5
    # optimisation
    batch: int = 16
    steps: int = 4000               # optimiser steps
    lr: float = 1e-3
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup: int = 200
    lr_min_frac: float = 0.1        # cosine decay to lr * lr_min_frac
    ema: float = 0.999
    clip: float = 1.0
    amp: str = "bf16"               # bf16 | none (autocast on CPU and CUDA)
    seed: int = 0
    pos_weight: float = 1.0         # BCE weight of ON labels

    @property
    def calls_per_step(self):
        """net calls (each `layers` deep) per supervision step"""
        if self.mode == "trm":
            return self.T * (self.n + 1)
        if self.mode == "deep":
            return self.T * (self.n + 1)
        return 1

    def to_json(self):
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_json(cls, d):
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)


PRESETS = {
    # CPU feasibility study (this repository's runs): tiny width, scaled-down recursion
    "cpu": dict(hidden=64, layers=2, n=4, T=2, n_sup=8, batch=16, steps=3000, lr=1e-3, weight_decay=0.1,
                warmup=200, ema=0.999, amp="bf16"),
    # TRM's paper scale (Sudoku-Extreme / Maze-Hard setting), for GPUs
    "paper": dict(hidden=512, layers=2, n=6, T=3, n_sup=16, batch=768, steps=60000, lr=1e-4, weight_decay=1.0,
                  betas=(0.9, 0.95), warmup=2000, ema=0.999, amp="bf16", unit_mix="attn", heads=8),
    # a middle ground for a single GPU
    "gpu_small": dict(hidden=128, layers=2, n=6, T=3, n_sup=16, batch=256, steps=30000, lr=3e-4, weight_decay=0.5,
                      warmup=1000, ema=0.999, amp="bf16", unit_mix="attn", heads=4),
}


def make_config(preset="cpu", **over):
    d = dict(PRESETS[preset])
    d.update({k: v for k, v in over.items() if v is not None})
    return TRMConfig(**d)


# ============================================================================ data
def load_npz(path):
    return dict(np.load(path, allow_pickle=False))


def train_data(n=2000, root=ROOT, m1x=M1X):
    """the m1x training set: train.npz (500, MILP labels) + the first (n - 500) / 250 extra chunks with their
    teacher-polished labels (no full MILP). Returns arrays incl. y (raw labels, not canonical)."""
    tr = load_npz(os.path.join(root, "train.npz"))
    parts = [{k: tr[k] for k in DATA_KEYS}]
    ys = [tr["u"]]
    k = 0
    while sum(len(p["load"]) for p in parts) < n:
        d = load_npz(os.path.join(m1x, f"extra_{k}.npz"))
        parts.append({key: d[key] for key in DATA_KEYS})
        ys.append(load_npz(os.path.join(m1x, f"polish_{k}.npz"))["y"])
        k += 1
    out = {key: np.concatenate([p[key] for p in parts])[:n] for key in DATA_KEYS}
    out["y"] = np.concatenate(ys)[:n].astype(np.int8)
    return out


def val_data(root=ROOT, splits=VAL_SPLITS):
    """val + val_extra + val_extra2 (360) with MILP labels u and objective obj"""
    parts = [load_npz(os.path.join(root, f"{s}.npz")) for s in splits]
    keys = DATA_KEYS + ("u", "obj", "gap")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def subset(d, idx):
    n = len(d["load"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}


# ============================================================================ features
class TRMFeatures:
    """Token features [N, T, G, F] and bus features [N, T, Nb, Fb] (any horizon T), standardised with the
    statistics of a training set. Token: unit static data (as otsl.ucml.UCFeaturizer), initial status, symmetry rank
    and group size, hourly system totals (load, renewables, net load, reserve, net load + reserve), position in the
    window (first hour, exp(-t / 4), exp(-(T - 1 - t) / 4): bounded, so longer windows stay in range), the bus's load
    and renewables, and optionally the LP relaxation (u_rel, price at the bus, largest incident line loading).
    Bus: load, renewables, unit capacity, optionally price and largest incident loading."""

    def __init__(self, sysm, lp=True, stats=None):
        self.s, self.lp = sysm, lp
        s = sysm
        TYPES = ["CT", "CC", "STEAM", "NUCLEAR"]
        onehot = np.array([[t == k for k in TYPES] for t in s.utype], float)
        avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
        self.g_static = np.c_[s.pmax, s.pmin, avg / 1e4, s.seg_c / 1e4, s.c_su / 1e4, s.min_up / 24, s.min_dn / 24,
                              np.minimum(s.ramp / s.pmax, 5), onehot]
        self.Rb = np.zeros((s.n_bus, len(s.r_pmax)))
        self.Rb[s.r_bus, np.arange(len(s.r_pmax))] = 1
        self.capbus = np.bincount(s.gbus, weights=s.pmax, minlength=s.n_bus)
        self.inc = np.zeros((s.n_bus, s.L))                      # bus-line incidence
        self.inc[s.f_bus, np.arange(s.L)] = 1
        self.inc[s.t_bus, np.arange(s.L)] = 1
        self.stats = stats

    def raw(self, d):
        s = self.s
        load, avail, u0 = d["load"], d["avail"], d["u0"]
        N, T, Nb = load.shape
        G = s.G
        ren_bus = avail @ self.Rb.T                                          # [N, T, Nb]
        tot, ren = load.sum(2), avail.sum(2)
        net = tot - ren
        hour = np.stack([tot, ren, net, d["sr"], net + d["sr"]], -1)          # [N, T, 5]
        t = np.arange(T, dtype=float)
        pos = np.stack([(t == 0).astype(float), np.exp(-t / 4), np.exp(-(T - 1 - t) / 4)], -1)   # [T, 3]
        rank, size = symmetry_rank(s, u0)
        parts = [np.broadcast_to(self.g_static[None, None], (N, T, G, self.g_static.shape[1])),
                 np.broadcast_to(u0[:, None, :, None].astype(float), (N, T, G, 1)),
                 np.broadcast_to(np.stack([rank, size], -1)[:, None], (N, T, G, 2)),
                 np.broadcast_to(hour[:, :, None, :], (N, T, G, 5)),
                 np.broadcast_to(pos[None, :, None, :], (N, T, G, 3)),
                 load[:, :, s.gbus][..., None], ren_bus[:, :, s.gbus][..., None]]
        bus = [load[..., None], ren_bus[..., None], np.broadcast_to(self.capbus[None, None, :, None], (N, T, Nb, 1))]
        if self.lp:
            lmp = d["lmp_rel"] / 1e4                                         # [N, T, Nb]
            ld = np.abs(d["flow_rel"]) / s.fmax[None, None, :]               # [N, T, L]
            ldb = (ld[:, :, None, :] * self.inc[None, None]).max(-1)        # [N, T, Nb]
            parts += [d["u_rel"][..., None], lmp[:, :, s.gbus][..., None], ldb[:, :, s.gbus][..., None]]
            bus += [lmp[..., None], ldb[..., None]]
        return np.concatenate(parts, -1), np.concatenate(bus, -1)

    def fit(self, d):
        x, b = self.raw(d)
        self.stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6) for a in (x, b)]
        return self

    @property
    def dims(self):
        n_tok = self.g_static.shape[1] + 1 + 2 + 5 + 3 + 2 + (3 if self.lp else 0)
        return n_tok, 3 + (2 if self.lp else 0)

    def __call__(self, d, device="cpu", dtype=torch.float32):
        x, b = self.raw(d)
        (mx, sx), (mb, sb) = self.stats
        return (torch.as_tensor((x - mx) / sx, dtype=dtype, device=device),
                torch.as_tensor((b - mb) / sb, dtype=dtype, device=device))

    def state(self):
        return dict(lp=self.lp, stats=[[m.tolist(), s.tolist()] for m, s in self.stats])

    @classmethod
    def from_state(cls, sysm, st):
        f = cls(sysm, lp=st["lp"])
        f.stats = [(np.array(m), np.array(s)) for m, s in st["stats"]]
        return f


# ============================================================================ model
def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class UnitMix(nn.Module):
    """mixing across the units of an hour. pool: mean and capacity-weighted pools (broadcast) plus a merit-order prefix
    pool (capacity-weighted state of the strictly cheaper units, ties counted half), all permutation-equivariant
    over identical units. attn: multi-head self-attention over the units of the hour."""

    def __init__(self, D, pw, merit, kind="pool", heads=4):
        super().__init__()
        self.kind = kind
        if kind == "pool":
            self.register_buffer("pw", torch.as_tensor(pw, dtype=torch.float32))
            self.register_buffer("merit", torch.as_tensor(merit, dtype=torch.float32))
            self.a = nn.Linear(D, D, bias=False)
            self.hour = nn.Linear(2 * D, D, bias=False)
            self.mer = nn.Linear(D, D, bias=False)
        else:
            self.attn = nn.MultiheadAttention(D, heads, bias=False, batch_first=True)

    def forward(self, h):
        if self.kind == "pool":
            a = self.a(h)                                                    # [B, T, G, D]
            m = a.mean(2, keepdim=True)
            c = torch.einsum("btgd,g->btd", a, self.pw.to(a.dtype)).unsqueeze(2)
            mer = torch.einsum("kg,btgd->btkd", self.merit.to(a.dtype), a * self.pw.to(a.dtype)[:, None])
            return self.hour(torch.cat([m, c], -1)) + self.mer(mer)
        B, T, G, D = h.shape
        x = h.reshape(B * T, G, D)
        return self.attn(x, x, x, need_weights=False)[0].reshape(B, T, G, D)


class TimeMix(nn.Module):
    """mixing across the hours of a unit: depthwise 1-D convolution (zero padded), SiLU, pointwise linear"""

    def __init__(self, D, k=5):
        super().__init__()
        self.dw = nn.Conv1d(D, D, k, padding=k // 2, groups=D, bias=False)
        self.pw = nn.Linear(D, D, bias=False)

    def forward(self, h):
        B, T, G, D = h.shape
        x = h.permute(0, 2, 3, 1).reshape(B * G, D, T)
        x = self.dw(x).reshape(B, G, D, T).permute(0, 3, 1, 2)
        return self.pw(F.silu(x))


class GraphMix(nn.Module):
    """mixing over the transmission network: unit states averaged into their buses, projected, plus the bus inputs
    (when the call sees x), `hops` steps of the symmetric-normalised bus adjacency, a projection of all hops, and back
    to every unit of the bus"""

    def __init__(self, D, Cg, Ahat, gbus, hops=2):
        super().__init__()
        self.hops = hops
        self.register_buffer("Cg", torch.as_tensor(Cg, dtype=torch.float32))
        self.register_buffer("A", torch.as_tensor(Ahat, dtype=torch.float32))
        self.register_buffer("gbus", torch.as_tensor(gbus, dtype=torch.long))
        self.inp = nn.Linear(D, D, bias=False)
        self.out = nn.Linear((hops + 1) * D, D, bias=False)

    def forward(self, h, bus_add=None):
        s = self.inp(torch.einsum("ng,btgd->btnd", self.Cg.to(h.dtype), h))
        if bus_add is not None:
            s = s + bus_add.to(s.dtype)
        s = F.silu(s)
        outs = [s]
        A = self.A.to(s.dtype)
        for _ in range(self.hops):
            s = torch.einsum("nm,btmd->btnd", A, s)
            outs.append(s)
        return self.out(torch.cat(outs, -1))[:, :, self.gbus]


class SwiGLU(nn.Module):
    def __init__(self, D, expansion=2.0):
        super().__init__()
        E = int(round(D * expansion))
        self.wg = nn.Linear(D, E, bias=False)
        self.wu = nn.Linear(D, E, bias=False)
        self.wd = nn.Linear(E, D, bias=False)

    def forward(self, h):
        return self.wd(F.silu(self.wg(h)) * self.wu(h))


class Layer(nn.Module):
    def __init__(self, cfg, geo):
        super().__init__()
        D = cfg.hidden
        self.unit = UnitMix(D, geo["pw"], geo["merit"], cfg.unit_mix, cfg.heads)
        self.time = TimeMix(D, cfg.time_kernel)
        self.graph = GraphMix(D, geo["Cg"], geo["Ahat"], geo["gbus"], cfg.hops)
        self.mlp = SwiGLU(D, cfg.expansion)

    def forward(self, h, bus_add=None):
        h = rms(h + self.unit(h))
        h = rms(h + self.time(h))
        h = rms(h + self.graph(h, bus_add))
        return rms(h + self.mlp(h))


class Net(nn.Module):
    """the tiny shared network: `layers` layers"""

    def __init__(self, cfg, geo):
        super().__init__()
        self.layers = nn.ModuleList([Layer(cfg, geo) for _ in range(cfg.layers)])

    def forward(self, h, bus_add=None):
        for layer in self.layers:
            h = layer(h, bus_add)
        return h


def geometry(sysm):
    """fixed system structure used by the mixers"""
    s = sysm
    G, Nb = s.G, s.n_bus
    avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
    merit = (avg[None, :] < avg[:, None] - 1e-9).astype(float)                # merit[k, g] = 1: g cheaper than k
    tie = np.isclose(avg[None, :], avg[:, None], atol=1e-9, rtol=0) & ~np.eye(G, dtype=bool)
    merit += 0.5 * tie
    pw = s.pmax / s.pmax.sum()
    cnt = np.bincount(s.gbus, minlength=Nb).astype(float)
    Cg = np.zeros((Nb, G))
    Cg[s.gbus, np.arange(G)] = 1.0 / cnt[s.gbus]
    A = np.zeros((Nb, Nb))
    A[s.f_bus, s.t_bus] = 1
    A[s.t_bus, s.f_bus] = 1
    deg = A.sum(1)
    Ahat = A / np.sqrt(np.outer(np.maximum(deg, 1), np.maximum(deg, 1)))
    return dict(merit=merit, pw=pw, Cg=Cg, Ahat=Ahat, gbus=s.gbus)


class TRMUC(nn.Module):
    """Tiny recursive model for UC (mode trm), and its one-shot / deep non-recursive controls"""

    def __init__(self, cfg: TRMConfig, sysm, x_dim, b_dim):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden
        geo = geometry(sysm)
        self.embed = nn.Linear(x_dim, D)
        self.bus_embed = nn.Linear(b_dim, D)
        if cfg.mode == "deep":
            self.nets = nn.ModuleList([Net(cfg, geo) for _ in range(cfg.calls_per_step)])
        else:
            self.net = Net(cfg, geo)
        self.head = nn.Linear(D, 1)
        self.q_head = nn.Linear(D, 1)
        g = torch.Generator().manual_seed(1234)
        self.register_buffer("y_init", torch.nn.init.trunc_normal_(torch.empty(D), std=1.0, a=-2, b=2, generator=g))
        self.register_buffer("z_init", torch.nn.init.trunc_normal_(torch.empty(D), std=1.0, a=-2, b=2, generator=g))
        with torch.no_grad():
            self.q_head.bias.fill_(-3.0)          # start by "continue" (TRM / HRM initialise the Q head this way)

    def inputs(self, xt, xb):
        return self.embed(xt), self.bus_embed(xb)

    def init_state(self, B, T, G):
        D = self.cfg.hidden
        return (self.y_init.view(1, 1, 1, D).expand(B, T, G, D).clone(),
                self.z_init.view(1, 1, 1, D).expand(B, T, G, D).clone())

    def latent_recursion(self, x, bus, y, z):
        for _ in range(self.cfg.n):
            z = self.net(x + y + z, bus)
        y = self.net(y + z, None)
        return y, z

    def readout(self, y):
        return self.head(y).squeeze(-1).float(), self.q_head(y.mean((1, 2))).squeeze(-1).float()

    def step(self, x, bus, y, z, grad=True):
        """one supervision step from (y, z). Returns detached (y, z), answer logits [B, T, G], halting logit [B]"""
        c = self.cfg
        if c.mode == "trm":
            with torch.no_grad():
                for _ in range(c.T - 1):
                    y, z = self.latent_recursion(x, bus, y, z)
            with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
                y, z = self.latent_recursion(x, bus, y, z)
                logit, q = self.readout(y)
            return y.detach(), z.detach(), logit, q
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            h = x + y + z
            if c.mode == "oneshot":
                h = self.net(h, bus)
            else:
                for net in self.nets:
                    h = net(h, bus)
            logit, q = self.readout(h)
        return h.detach(), z, logit, q


def build(cfg, sysm, feats, device="cpu"):
    torch.manual_seed(cfg.seed)
    xd, bd = feats.dims
    return TRMUC(cfg, sysm, xd, bd).to(device)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def autocast(cfg, device):
    dev = torch.device(device).type
    if cfg.amp == "bf16":
        return torch.autocast(dev, dtype=torch.bfloat16)
    return torch.autocast(dev, enabled=False)


# ============================================================================ inference
@torch.no_grad()
def run_steps(model, xt, xb, k_max, bs=32, record=None, cfg=None):
    """probabilities after each supervision step 1..k_max (mode trm; one-shot / deep: a single step), batched.
    Returns P [N, k_max, T, G] (float32 numpy) and halting logits Q [N, k_max]. record: optional list of step counts to
    keep (saves memory)."""
    model.eval()
    cfg = cfg or model.cfg
    N, T, G = xt.shape[:3]
    keep = list(range(1, k_max + 1)) if record is None else sorted(record)
    P = np.zeros((N, len(keep), T, G), np.float32)
    Q = np.zeros((N, k_max), np.float32)
    for i in range(0, N, bs):
        with autocast(cfg, xt.device):
            x, bus = model.inputs(xt[i:i + bs], xb[i:i + bs])
            y, z = model.init_state(len(x), T, G)
            if cfg.mode != "trm":     # non-recursive controls: one pass, the same answer at every "step"
                _, _, logit, q = model.step(x, bus, y, z, grad=False)
                P[i:i + bs] = torch.sigmoid(logit).float().cpu().numpy()[:, None]
                Q[i:i + bs] = q.float().cpu().numpy()[:, None]
                continue
            for k in range(1, k_max + 1):
                y, z, logit, q = model.step(x, bus, y, z, grad=False)
                Q[i:i + bs, k - 1] = q.float().cpu().numpy()
                if k in keep:
                    P[i:i + bs, keep.index(k)] = torch.sigmoid(logit).float().cpu().numpy()
    return P, Q


def decode(p, d, i, sysm, th=0.5):
    """threshold -> block (min up/down-aware) adequacy repair -> min up/down repair (otsl.combo.block_decode)"""
    u = (p > th).astype(np.int8)
    u = adequacy_repair_blocks(u, d["u0"][i], d["load"][i], d["avail"][i], d["sr"][i], sysm)
    return repair_min_updown(u, d["u0"][i], sysm.min_up, sysm.min_dn)


# ============================================================================ training (deep supervision)
def lr_at(cfg, step):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    f = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.lr_min_frac + (1 - cfg.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, f))))


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=tuple(cfg.betas))


class Carry:
    """the training batch of TRM / HRM: B slots, each holding an instance and its carried (y, z); a slot is refilled
    with the next instance of the epoch permutation when its instance halts or reaches n_sup steps"""

    def __init__(self, model, n_data, cfg, rng, T, G):
        self.cfg, self.rng, self.n = cfg, rng, n_data
        B = cfg.batch
        self.perm, self.ptr, self.epoch = rng.permutation(n_data), 0, 0
        self.idx = np.array([self._next() for _ in range(B)])
        self.y, self.z = model.init_state(B, T, G)
        self.steps = np.zeros(B, int)
        self.min_steps = np.array([self._min() for _ in range(B)])
        self.done_steps = []                 # supervision steps used by finished instances

    def _next(self):
        if self.ptr >= self.n:
            self.perm, self.ptr, self.epoch = self.rng.permutation(self.n), 0, self.epoch + 1
        self.ptr += 1
        return int(self.perm[self.ptr - 1])

    def _min(self):
        c = self.cfg
        if c.n_sup > 1 and self.rng.random() < c.halt_explore:
            return int(self.rng.integers(2, c.n_sup + 1))
        return 1

    def advance(self, model, halt_logit):
        c = self.cfg
        self.steps += 1
        halted = self.steps >= c.n_sup
        if c.halt_train and c.n_sup > 1:
            halted |= (halt_logit > 0) & (self.steps >= self.min_steps)
        for b in np.where(halted)[0]:
            self.done_steps.append(int(self.steps[b]))
            self.idx[b] = self._next()
            self.steps[b] = 0
            self.min_steps[b] = self._min()
        if halted.any():
            h = torch.as_tensor(halted, device=self.y.device)
            y0, z0 = model.init_state(int(halted.sum()), self.y.shape[1], self.y.shape[2])
            self.y[h], self.z[h] = y0.to(self.y.dtype), z0.to(self.z.dtype)
        return halted

    def state(self):
        return dict(perm=self.perm, ptr=self.ptr, epoch=self.epoch, idx=self.idx, steps=self.steps,
                    min_steps=self.min_steps, y=self.y.cpu(), z=self.z.cpu())

    def load(self, st, device):
        self.perm, self.ptr, self.epoch = st["perm"], st["ptr"], st["epoch"]
        self.idx, self.steps, self.min_steps = st["idx"], st["steps"], st["min_steps"]
        self.y, self.z = st["y"].to(device), st["z"].to(device)


def bce_loss(logit, target, pos_weight=1.0):
    w = None if pos_weight == 1.0 else torch.where(target > 0.5, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logit, target, weight=w, reduction="none").mean((1, 2))


@torch.no_grad()
def val_loss(model, xt, xb, target, cfg, steps=(1, 4, 16), bs=32):
    """mean BCE against canonical labels after each of `steps` supervision steps; wrong decisions per instance"""
    P, _ = run_steps(model, xt, xb, max(steps), bs=bs, record=steps, cfg=cfg)
    y = target.cpu().numpy()
    out = {}
    for j, k in enumerate(sorted(steps)):
        p = np.clip(P[:, j], 1e-6, 1 - 1e-6)
        out[f"bce@{k}"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out[f"wrong@{k}"] = float(((p > 0.5) != (y > 0.5)).sum((1, 2)).mean())
    return out


def save_ckpt(path, model, ema, opt, step, carry, rng, cfg, feats, log):
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step,
                    carry=carry.state() if carry is not None else None, rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cfg=cfg.to_json(), feats=feats.state(), log=log), tmp)
    os.replace(tmp, path)


def load_model(path, sysm, device="cpu", which="ema"):
    """(model with EMA or raw weights, featurizer, config, checkpoint dict)"""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TRMConfig.from_json(ck["cfg"])
    feats = TRMFeatures.from_state(sysm, ck["feats"])
    model = build(cfg, sysm, feats, device)
    model.load_state_dict(ck[which] if which in ck else ck["model"])
    model.eval()
    return model, feats, cfg, ck


def train(cfg: TRMConfig, sysm, tr, va, ckpt, device="cpu", log_every=50, val_every=500, save_every_s=300,
          logf=print, val_steps=(1, 4, 16)):
    """deep-supervision training with resume from `ckpt` (written every save_every_s seconds and at the end)."""
    feats = TRMFeatures(sysm, lp=cfg.lp_feats).fit(tr)
    xt, xb = feats(tr, device)
    tgt = torch.as_tensor(canonical_labels(sysm, tr["y"], tr["u0"]), dtype=torch.float32, device=device)
    vxt, vxb = feats(va, device)
    vtgt = torch.as_tensor(canonical_labels(sysm, va["u"], va["u0"]), dtype=torch.float32, device=device)
    model = build(cfg, sysm, feats, device)
    ema = EMA(model, cfg.ema)
    opt = make_optimizer(model, cfg)
    rng = np.random.default_rng(cfg.seed)
    T, G = tr["load"].shape[1], sysm.G
    carry = Carry(model, len(tr["load"]), cfg, rng, T, G)
    step, log = 0, dict(train=[], val=[], params=n_params(model), time_s=0.0)
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if ck["carry"] is not None:
            carry.load(ck["carry"], device)
        logf(f"[trm] resumed at step {step}")
    logf(f"[trm] params {n_params(model)}; cfg {json.dumps(cfg.to_json())}")
    t0, t_save, acc = time.time() - log["time_s"], time.time(), []
    while step < cfg.steps:
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(cfg, step)
        model.train()
        idx = torch.as_tensor(carry.idx, device=device)
        with autocast(cfg, device):
            x, bus = model.inputs(xt[idx], xb[idx])
            y, z, logit, q = model.step(x, bus, carry.y, carry.z, grad=True)
        target = tgt[idx]
        l_ans = bce_loss(logit, target, cfg.pos_weight)
        with torch.no_grad():
            wrong = ((logit > 0) != (target > 0.5)).sum((1, 2))
            halt_t = (wrong <= cfg.halt_tol).float()
        l_q = F.binary_cross_entropy_with_logits(q, halt_t, reduction="none")
        loss = (l_ans + cfg.halt_weight * l_q).mean() if cfg.mode == "trm" else l_ans.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        ema.update(model)
        acc.append((float(l_ans.mean()), float(wrong.float().mean()), float(halt_t.mean()),
                    float(((q > 0).float() == halt_t).float().mean()), float(carry.steps.mean() + 1)))
        carry.y, carry.z = y.float(), z.float()
        carry.advance(model, q.detach().cpu().numpy())
        step += 1
        if step % log_every == 0 or step == cfg.steps:
            a = np.array(acc)
            rec = dict(step=step, bce=float(a[:, 0].mean()), wrong=float(a[:, 1].mean()), halt_rate=float(a[:, 2].mean()),
                       q_acc=float(a[:, 3].mean()), slot_step=float(a[:, 4].mean()),
                       steps_per_inst=float(np.mean(carry.done_steps[-200:])) if carry.done_steps else float("nan"),
                       epoch=carry.epoch, lr=lr_at(cfg, step - 1), s=time.time() - t0)
            log["train"].append(rec)
            logf("[trm] " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
            acc = []
        if step % val_every == 0 or step == cfg.steps:
            vs = tuple(sorted({1, min(cfg.n_sup, 4), cfg.n_sup} | set(val_steps))) if cfg.mode == "trm" else (1,)
            v = dict(step=step, **val_loss(ema.shadow, vxt[:60], vxb[:60], vtgt[:60], cfg, vs))
            log["val"].append(v)
            logf("[trm] val(60, EMA) " + json.dumps(v))
        log["time_s"] = time.time() - t0
        if time.time() - t_save > save_every_s or step == cfg.steps:
            save_ckpt(ckpt, model, ema, opt, step, carry, rng, cfg, feats, log)
            t_save = time.time()
    return model, ema, feats, log


# ============================================================================ REINFORCE with the exact LP critic
def train_reinforce(model, cfg, sysm, feats, tr, oracle, ckpt, steps=150, bs=16, n_samples=6, lr=1e-4, k_sup=None,
                    anchor=0.5, device="cpu", seed=0, logf=print, log_every=10, save_every_s=300, ema_decay=0.99):
    """Cost-aware fine-tuning of the final answer: k_sup - 1 supervision steps without gradient, the last one with;
    n_samples Bernoulli schedules per instance, each repaired as at inference (block adequacy + min up/down) and priced
    by the dispatch LP; reward -log(cost / c_rel); leave-one-out baseline; plus `anchor` x BCE to the canonical labels
    (keeps the imitation structure; REINFORCE alone over-commits, RESEARCH.md V4). Resumable from ckpt."""
    k_sup = k_sup or (cfg.n_sup if cfg.mode == "trm" else 1)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    xt, xb = feats(tr, device)
    tgt = torch.as_tensor(canonical_labels(sysm, tr["y"], tr["u0"]), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, ema_decay)
    n, T, G = len(tr["load"]), tr["load"].shape[1], sysm.G
    step, hist = 0, []
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, hist = ck["step"], ck["hist"]
        rng.bit_generator.state = ck["rng"]
        gen.set_state(ck["gen"])
        logf(f"[rl] resumed at step {step}")
    t0, t_save = time.time(), time.time()
    while step < steps:
        idx = rng.choice(n, bs, replace=False)
        ii = torch.as_tensor(idx, device=device)
        model.eval()                       # no dropout anyway; norms are stateless
        with autocast(cfg, device):
            x, bus = model.inputs(xt[ii], xb[ii])
            y, z = model.init_state(bs, T, G)
            with torch.no_grad():
                for _ in range(k_sup - 1):
                    y, z, _, _ = model.step(x, bus, y, z, grad=False)
            x, bus = model.inputs(xt[ii], xb[ii])
            _, _, logit, _ = model.step(x, bus, y, z, grad=True)
        p = torch.sigmoid(logit.detach()).cpu()
        smp = torch.rand((n_samples,) + p.shape, generator=gen) < p                     # [S, B, T, G]
        smp_d = smp.to(device)
        logp = (smp_d * F.logsigmoid(logit) + (~smp_d) * F.logsigmoid(-logit)).sum((-1, -2))
        us = smp.numpy().astype(np.int8)
        rep = np.array([[decode(us[k, b].astype(float), tr, idx[b], sysm, th=0.5) for b in range(bs)]
                        for k in range(n_samples)])
        flat = np.tile(idx, n_samples)
        c, sh, so = oracle.evaluate(tr, flat, rep.reshape((-1, T, G)), flat)
        c = c.reshape(n_samples, bs)
        r = torch.as_tensor(-np.log(c / tr["c_rel"][idx][None]), dtype=torch.float32, device=device)
        adv = r - (r.sum(0, keepdim=True) - r) / (n_samples - 1)
        adv = adv / (adv.std() + 1e-8)
        l_pg = -(adv * logp).mean() / (T * G)
        l_bce = bce_loss(logit, tgt[ii]).mean()
        loss = l_pg + anchor * l_bce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        step += 1
        gap = float(np.mean(c / tr["c_rel"][idx][None] - 1))
        served = float(np.mean((sh < 1e-6) & (so < 1e-6)))
        hist.append(dict(step=step, gap_rel=gap, served=served, bce=float(l_bce), lps=int(oracle.n), s=time.time() - t0))
        if step % log_every == 0 or step == steps:
            h = hist[-log_every:]
            logf(f"[rl] step {step} sampled gap to relaxation {np.mean([x['gap_rel'] for x in h]) * 100:.3f}% "
                 f"served {np.mean([x['served'] for x in h]) * 100:.1f}% bce {np.mean([x['bce'] for x in h]):.4f} "
                 f"LPs {oracle.n} ({time.time() - t0:.0f}s)")
        if time.time() - t_save > save_every_s or step == steps:
            tmp = ckpt + ".tmp"
            torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step,
                            hist=hist, rng=rng.bit_generator.state, gen=gen.get_state()), tmp)
            os.replace(tmp, ckpt)
            t_save = time.time()
    return model, ema, hist
