"""Tiny Recursive Model (TRM) for end-to-end unit commitment on the 12-hour RTS-GMLC benchmark (arm A of
docs/recursive/PLAN.md).

Tokens: one per (unit, hour), [B, T, G, D]. One tiny shared network `Net` of `layers` layers; each layer mixes
  * across units within an hour (`UnitMix`): mean and capacity-weighted pools plus a merit-order prefix pool (the
    capacity-weighted state of the cheaper units), or multi-head attention with an electrical-distance bias;
  * across hours for a unit (`TimeMix`): depthwise convolution along the hours, or attention with a relative
    (ALiBi-style) hour-difference bias; both are length-agnostic;
  * over the transmission network through the units' buses (`GraphMix`): unit -> bus averaging, the bus inputs (loads,
    renewables, PTDF encodings, LP prices / line loadings), `hops` steps of the normalised bus adjacency and one step of
    an electrical-distance kernel (from the effective reactance distance of the DC susceptance matrix), bus -> unit;
  * across channels (SwiGLU MLP);
each sub-block residual with a post RMS norm (TRM / HRM). There are no unit-ID embeddings: a unit is described by its
parameters. Identical units (same parameters, bus and initial status) therefore get identical outputs.

Identical units (sym = "count", default). 56 of the 73 units are in 24 groups of exact copies. The target of a unit is
the mean label of its subgroup (group x initial status) per hour, i.e. the subgroup's on-count divided by its size;
BCE against it is minimised by the count, so the loss is invariant to permuting copies and augmentation by such
permutations is the identity. Decoding turns the expected count of a subgroup into per-copy scores by water-filling
(copy j gets clip(count - j, 0, 1)) and decodes every unit with a Viterbi DP over the on/off state machine with min
up/down and the initial status (feasible by construction), then the block adequacy repair and one dispatch LP.
sym = "rank" is the earlier recipe (symmetry-rank feature + canonical labels) for comparison.

Recursion (TRM, Jolicoeur-Martineau 2025): answer embedding y and latent z;
    latent_recursion:  z <- net(x + y + z) n times, then y <- net(y + z)
    step:              T - 1 latent recursions without gradient, one with gradient
    deep supervision:  up to n_sup steps per instance, (y, z) carried detached, BCE on every step's answer and a
                       halting head (BCE on "the rounded answer has at most halt_tol wrong unit-hours"); instances leave
                       the training batch when they halt and are replaced (ACT with TRM's single-pass halting); EMA.
Controls with the same blocks: `oneshot` (the same net applied once: equal parameters) and `deep` (T (n + 1) separate
nets applied once: the compute of one supervision step without weight sharing).

Second stage: REINFORCE with the exact dispatch LP as critic on the schedule decoded as at inference, at the last or a
random supervision step, leave-one-out baseline, BCE anchor to the labels. A post-hoc halting head is fitted on
LP-priced targets ("the step's schedule is within 0.5 % of the best known cost").

Everything is device-agnostic (`device="cuda"` or `"cpu"`); `PRESETS` holds the CPU feasibility setting and TRM's
paper scale.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

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
    unit_mix: str = "pool"          # pool | attn (electrical-distance bias)
    time_mix: str = "conv"          # conv | attn (relative hour bias)
    heads: int = 4
    time_kernel: int = 5
    hops: int = 2                   # bus-adjacency propagation steps per layer
    elec: bool = True               # + one step of the electrical-distance kernel per layer
    lp_feats: bool = True           # LP-relaxation inputs (commitment, price, line loading)
    sym: str = "count"              # count (subgroup counts + Viterbi) | rank (rank feature + canonical labels)
    y0: str = "fixed"               # fixed | lp (y initialised from the LP-relaxation commitment)
    cost_weight: float = 0.0        # alpha of the error-cost weighting of the BCE (0 = off)
    # recursion
    mode: str = "trm"               # trm | oneshot | deep
    n: int = 6
    T: int = 3
    n_sup: int = 16                 # deep-supervision steps (training); 1 = deep supervision off
    halt_train: bool = True
    halt_tol: int = 2
    halt_explore: float = 0.1
    halt_weight: float = 0.5
    # optimisation
    batch: int = 16
    steps: int = 4000
    lr: float = 1e-3
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup: int = 200
    lr_min_frac: float = 0.1
    ema: float = 0.999
    clip: float = 1.0
    amp: str = "bf16"               # bf16 | none
    seed: int = 0

    @property
    def calls_per_step(self):
        return self.T * (self.n + 1) if self.mode in ("trm", "deep") else 1

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
                  betas=(0.9, 0.95), warmup=2000, ema=0.999, amp="bf16", unit_mix="attn", time_mix="attn", heads=8),
    # a middle ground for one GPU
    "gpu_small": dict(hidden=128, layers=2, n=6, T=3, n_sup=16, batch=256, steps=30000, lr=3e-4, weight_decay=0.5,
                      warmup=1000, ema=0.999, amp="bf16", unit_mix="attn", time_mix="attn", heads=4),
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
    teacher-polished labels (no full MILP). y = raw labels; c_lab = dispatch-LP cost of the label where stored"""
    tr = load_npz(os.path.join(root, "train.npz"))
    parts = [{k: tr[k] for k in DATA_KEYS}]
    ys, cl = [tr["u"]], [tr["obj"]]
    k = 0
    while sum(len(p["load"]) for p in parts) < n:
        d = load_npz(os.path.join(m1x, f"extra_{k}.npz"))
        parts.append({key: d[key] for key in DATA_KEYS})
        pol = load_npz(os.path.join(m1x, f"polish_{k}.npz"))
        ys.append(pol["y"])
        cl.append(np.where(pol["used_milp"], pol["cost"], d["c_lf"]))
        k += 1
    out = {key: np.concatenate([p[key] for p in parts])[:n] for key in DATA_KEYS}
    out["y"] = np.concatenate(ys)[:n].astype(np.int8)
    out["c_lab"] = np.concatenate(cl)[:n]
    return out


def val_data(root=ROOT, splits=VAL_SPLITS):
    """val + val_extra + val_extra2 (360) with MILP labels u and objective obj"""
    parts = [load_npz(os.path.join(root, f"{s}.npz")) for s in splits]
    keys = DATA_KEYS + ("u", "obj", "gap")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def subset(d, idx):
    n = len(d["load"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}


# ============================================================================ system structure
def ptdf_matrix(sysm):
    """DC PTDF [L, N] with the reference bus as slack"""
    s = sysm
    A = s.A
    Bbus = A.T @ np.diag(s.b) @ A
    keep = [i for i in range(s.n_bus) if i != s.ref]
    X = np.zeros((s.n_bus, s.n_bus))
    X[np.ix_(keep, keep)] = np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return np.diag(s.b) @ A @ X


def elec_distance(sysm):
    """effective reactance distance between buses: R_ij = L+_ii + L+_jj - 2 L+_ij, L = A^T diag(b) A"""
    s = sysm
    Lp = np.linalg.pinv(s.A.T @ np.diag(s.b) @ s.A)
    d = np.diag(Lp)
    return np.maximum(d[:, None] + d[None, :] - 2 * Lp, 0.0)


def subgroups(sysm, u0):
    """per instance, lists of units that are exact copies with the same initial status (size >= 2)"""
    groups = sysm.identical_groups()
    out = []
    for i in range(len(u0)):
        sg = []
        for grp in groups:
            for v in (0, 1):
                mem = [g for g in grp if u0[i, g] == v]
                if len(mem) > 1:
                    sg.append(mem)
        out.append(sg)
    return out


def subgroup_mean_matrix(sysm, u0):
    """S [N, G, G]: S[i, g, h] = 1 / size if g, h are copies with the same initial status (incl. g = h), else I"""
    N, G = u0.shape
    S = np.tile(np.eye(G, dtype=np.float32), (N, 1, 1))
    for i, sg in enumerate(subgroups(sysm, u0)):
        for mem in sg:
            S[i][np.ix_(mem, mem)] = 1.0 / len(mem)
    return S


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
    R = elec_distance(s)
    tau = np.median(R[~np.eye(Nb, dtype=bool)])
    K = np.exp(-R / tau)
    K /= K.sum(1, keepdims=True)
    Ru = R[s.gbus][:, s.gbus] / tau                                           # unit-unit electrical distance
    return dict(merit=merit, pw=pw, Cg=Cg, Ahat=Ahat, K=K, Ru=Ru, gbus=s.gbus)


# ============================================================================ features
class TRMFeatures:
    """Token features [N, T, G, F] and bus features [N, T, Nb, Fb] (any horizon T), standardised with the
    statistics of a training set. Token: unit static data (as otsl.ucml.UCFeaturizer), initial status, subgroup size
    (and the symmetry rank if sym = "rank"), hourly totals (load, renewables, net load, reserve, net load + reserve),
    position (first hour, exp(-t / 4), exp(-(T - 1 - t) / 4): bounded, so longer windows stay in range), the bus's load,
    renewables and PTDF encoding (8 principal components of its PTDF column), optionally the LP relaxation (u_rel,
    price at the bus, largest incident line loading). Bus: load, renewables, unit capacity, PTDF encoding, optionally
    price and largest incident loading."""

    N_PTDF = 8

    def __init__(self, sysm, lp=True, sym="count", stats=None):
        self.s, self.lp, self.sym = sysm, lp, sym
        s = sysm
        TYPES = ["CT", "CC", "STEAM", "NUCLEAR"]
        onehot = np.array([[t == k for k in TYPES] for t in s.utype], float)
        avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
        self.g_static = np.c_[s.pmax, s.pmin, avg / 1e4, s.seg_c / 1e4, s.c_su / 1e4, s.min_up / 24, s.min_dn / 24,
                              np.minimum(s.ramp / s.pmax, 5), onehot]
        self.Rb = np.zeros((s.n_bus, len(s.r_pmax)))
        self.Rb[s.r_bus, np.arange(len(s.r_pmax))] = 1
        self.capbus = np.bincount(s.gbus, weights=s.pmax, minlength=s.n_bus)
        self.inc = np.zeros((s.n_bus, s.L))
        self.inc[s.f_bus, np.arange(s.L)] = 1
        self.inc[s.t_bus, np.arange(s.L)] = 1
        P = ptdf_matrix(s).T                                                 # [N, L]
        P = P - P.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(P, full_matrices=False)
        enc = U[:, :self.N_PTDF] * S[:self.N_PTDF]
        enc *= np.sign(enc[np.abs(enc).argmax(0), np.arange(self.N_PTDF)])  # deterministic signs
        self.ptdf_enc = enc
        self.stats = stats
        self.col_urel = None

    def raw(self, d):
        s = self.s
        load, avail, u0 = d["load"], d["avail"], d["u0"]
        N, T, Nb = load.shape
        G = s.G
        ren_bus = avail @ self.Rb.T
        tot, ren = load.sum(2), avail.sum(2)
        net = tot - ren
        hour = np.stack([tot, ren, net, d["sr"], net + d["sr"]], -1)
        t = np.arange(T, dtype=float)
        pos = np.stack([(t == 0).astype(float), np.exp(-t / 4), np.exp(-(T - 1 - t) / 4)], -1)
        rank, size = symmetry_rank(s, u0)
        symf = np.stack([rank, size], -1) if self.sym == "rank" else size[..., None]
        parts = [np.broadcast_to(self.g_static[None, None], (N, T, G, self.g_static.shape[1])),
                 np.broadcast_to(u0[:, None, :, None].astype(float), (N, T, G, 1)),
                 np.broadcast_to(symf[:, None], (N, T, G, symf.shape[-1])),
                 np.broadcast_to(hour[:, :, None, :], (N, T, G, 5)),
                 np.broadcast_to(pos[None, :, None, :], (N, T, G, 3)),
                 load[:, :, s.gbus][..., None], ren_bus[:, :, s.gbus][..., None],
                 np.broadcast_to(self.ptdf_enc[s.gbus][None, None], (N, T, G, self.N_PTDF))]
        bus = [load[..., None], ren_bus[..., None], np.broadcast_to(self.capbus[None, None, :, None], (N, T, Nb, 1)),
               np.broadcast_to(self.ptdf_enc[None, None], (N, T, Nb, self.N_PTDF))]
        if self.lp:
            lmp = d["lmp_rel"] / 1e4
            ld = np.abs(d["flow_rel"]) / s.fmax[None, None, :]
            ldb = (ld[:, :, None, :] * self.inc[None, None]).max(-1)
            self.col_urel = sum(p.shape[-1] for p in parts)
            urel = d["u_rel"]
            if self.sym == "count":     # the relaxation splits identical copies arbitrarily: keep its subgroup mean
                urel = np.einsum("igh,ith->itg", subgroup_mean_matrix(s, u0), urel)
            parts += [urel[..., None], lmp[:, :, s.gbus][..., None], ldb[:, :, s.gbus][..., None]]
            bus += [lmp[..., None], ldb[..., None]]
        return np.concatenate(parts, -1), np.concatenate(bus, -1)

    def fit(self, d):
        x, b = self.raw(d)
        self.stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6) for a in (x, b)]
        return self

    @property
    def dims(self):
        n_tok = self.g_static.shape[1] + 1 + (2 if self.sym == "rank" else 1) + 5 + 3 + 2 + self.N_PTDF + \
            (3 if self.lp else 0)
        return n_tok, 3 + self.N_PTDF + (2 if self.lp else 0)

    def __call__(self, d, device="cpu", dtype=torch.float32):
        x, b = self.raw(d)
        (mx, sx), (mb, sb) = self.stats
        return (torch.as_tensor((x - mx) / sx, dtype=dtype, device=device),
                torch.as_tensor((b - mb) / sb, dtype=dtype, device=device))

    def state(self):
        return dict(lp=self.lp, sym=self.sym, stats=[[m.tolist(), s.tolist()] for m, s in self.stats])

    @classmethod
    def from_state(cls, sysm, st):
        f = cls(sysm, lp=st["lp"], sym=st.get("sym", "count"))
        f.stats = [(np.array(m), np.array(s)) for m, s in st["stats"]]
        return f


def targets(sysm, d, y, sym):
    """training targets [N, T, G]: subgroup-mean labels (counts / size) or canonical labels"""
    if sym == "rank":
        return canonical_labels(sysm, y, d["u0"]).astype(np.float32)
    S = subgroup_mean_matrix(sysm, d["u0"])
    return np.einsum("igh,ith->itg", S, y.astype(np.float32))


# ============================================================================ model
def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class BiasedAttention(nn.Module):
    """multi-head self-attention over the second-to-last axis with a fixed bias matrix scaled per head
    (bias [L, L] buffer, learnable slope per head)"""

    def __init__(self, D, heads, bias=None):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.slope = nn.Parameter(torch.linspace(0.5, 2.0, heads)) if bias is not None else None
        if bias is not None:
            self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, x, bias=None):              # x [M, L, D]
        M, L, D = x.shape
        q, k, v = self.qkv(x).view(M, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = None
        b = self.bias if bias is None and self.slope is not None else bias
        if b is not None:
            mask = (-self.slope.view(1, self.h, 1, 1) * b[:L, :L].view(1, 1, L, L)).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(M, L, D))


class UnitMix(nn.Module):
    """mixing across the units of an hour. pool: mean and capacity-weighted pools (broadcast) plus a merit-order prefix
    pool (capacity-weighted state of the strictly cheaper units, ties counted half). attn: self-attention over the units
    of the hour with an electrical-distance bias (units at the same bus: 0). Both are equivariant over identical units."""

    def __init__(self, D, geo, kind="pool", heads=4):
        super().__init__()
        self.kind = kind
        if kind == "pool":
            self.register_buffer("pw", torch.as_tensor(geo["pw"], dtype=torch.float32))
            self.register_buffer("merit", torch.as_tensor(geo["merit"], dtype=torch.float32))
            self.a = nn.Linear(D, D, bias=False)
            self.hour = nn.Linear(2 * D, D, bias=False)
            self.mer = nn.Linear(D, D, bias=False)
        else:
            self.attn = BiasedAttention(D, heads, geo["Ru"])

    def forward(self, h):
        if self.kind == "pool":
            a = self.a(h)
            pw = self.pw.to(a.dtype)
            m = a.mean(2, keepdim=True)
            c = torch.einsum("btgd,g->btd", a, pw).unsqueeze(2)
            mer = torch.einsum("kg,btgd->btkd", self.merit.to(a.dtype), a * pw[:, None])
            return self.hour(torch.cat([m, c], -1)) + self.mer(mer)
        B, T, G, D = h.shape
        return self.attn(h.reshape(B * T, G, D)).reshape(B, T, G, D)


class TimeMix(nn.Module):
    """mixing across the hours of a unit. conv: depthwise 1-D convolution, SiLU, pointwise linear. attn: self-attention
    over the hours with an ALiBi-style bias proportional to |t - t'| (relative, extrapolates to longer windows)"""

    def __init__(self, D, k=5, kind="conv", heads=4, t_max=96):
        super().__init__()
        self.kind = kind
        if kind == "conv":
            self.dw = nn.Conv1d(D, D, k, padding=k // 2, groups=D, bias=False)
            self.pw = nn.Linear(D, D, bias=False)
        else:
            t = np.arange(t_max)
            self.attn = BiasedAttention(D, heads, np.abs(t[:, None] - t[None, :]) / 4.0)

    def forward(self, h):
        B, T, G, D = h.shape
        if self.kind == "conv":
            x = h.permute(0, 2, 3, 1).reshape(B * G, D, T)
            x = self.dw(x).reshape(B, G, D, T).permute(0, 3, 1, 2)
            return self.pw(F.silu(x))
        x = h.permute(0, 2, 1, 3).reshape(B * G, T, D)
        return self.attn(x).reshape(B, G, T, D).permute(0, 2, 1, 3)


class GraphMix(nn.Module):
    """mixing over the transmission network: unit states averaged into their buses, projected, plus the bus inputs
    (when the call sees x), SiLU, `hops` steps of the normalised bus adjacency and (elec) one step of the
    electrical-distance kernel, a projection of all of them, and back to every unit of the bus"""

    def __init__(self, D, geo, hops=2, elec=True):
        super().__init__()
        self.hops, self.elec = hops, elec
        self.register_buffer("Cg", torch.as_tensor(geo["Cg"], dtype=torch.float32))
        self.register_buffer("A", torch.as_tensor(geo["Ahat"], dtype=torch.float32))
        self.register_buffer("K", torch.as_tensor(geo["K"], dtype=torch.float32))
        self.register_buffer("gbus", torch.as_tensor(geo["gbus"], dtype=torch.long))
        self.inp = nn.Linear(D, D, bias=False)
        self.out = nn.Linear((hops + 1 + int(elec)) * D, D, bias=False)

    def forward(self, h, bus_add=None):
        s = self.inp(torch.einsum("ng,btgd->btnd", self.Cg.to(h.dtype), h))
        if bus_add is not None:
            s = s + bus_add.to(s.dtype)
        s = F.silu(s)
        outs = [s]
        if self.elec:
            outs.append(torch.einsum("nm,btmd->btnd", self.K.to(s.dtype), s))
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
        self.unit = UnitMix(D, geo, cfg.unit_mix, cfg.heads)
        self.time = TimeMix(D, cfg.time_kernel, cfg.time_mix, cfg.heads)
        self.graph = GraphMix(D, geo, cfg.hops, cfg.elec)
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


class TRMUC(nn.Module):
    """Tiny recursive model for UC (mode trm), and its one-shot / deep non-recursive controls"""

    def __init__(self, cfg: TRMConfig, sysm, x_dim, b_dim, col_urel=None):
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
        self.col_urel = col_urel
        if cfg.y0 == "lp":
            assert col_urel is not None, "y0 = lp needs the LP features"
            self.y0_proj = nn.Linear(1, D)
        with torch.no_grad():
            self.q_head.bias.fill_(-3.0)          # start by "continue" (as TRM / HRM initialise the Q head)

    def inputs(self, xt, xb):
        return self.embed(xt), self.bus_embed(xb)

    def init_state(self, B, T, G, xt=None):
        D = self.cfg.hidden
        y = self.y_init.view(1, 1, 1, D).expand(B, T, G, D).clone()
        z = self.z_init.view(1, 1, 1, D).expand(B, T, G, D).clone()
        if self.cfg.y0 == "lp" and xt is not None:
            y = rms(y + self.y0_proj(xt[..., self.col_urel:self.col_urel + 1]).float())
        return y, z

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
        g = grad and torch.is_grad_enabled()
        if c.mode == "trm":
            with torch.no_grad():
                for _ in range(c.T - 1):
                    y, z = self.latent_recursion(x, bus, y, z)
            with torch.set_grad_enabled(g):
                y, z = self.latent_recursion(x, bus, y, z)
                logit, q = self.readout(y)
            return y.detach(), z.detach(), logit, q
        with torch.set_grad_enabled(g):
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
    if feats.col_urel is None and feats.lp:     # sets col_urel
        feats.raw({k: v[:1] for k, v in _dummy(sysm).items()})
    return TRMUC(cfg, sysm, xd, bd, feats.col_urel).to(device)


def _dummy(sysm, T=2):
    G, Nb, R, L = sysm.G, sysm.n_bus, len(sysm.r_pmax), sysm.L
    return dict(load=np.ones((1, T, Nb)), avail=np.zeros((1, T, R)), u0=np.zeros((1, G), np.int8), sr=np.ones((1, T)),
                u_rel=np.zeros((1, T, G)), lmp_rel=np.zeros((1, T, Nb)), flow_rel=np.zeros((1, T, L)))


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
    dev = torch.device(device).type if not isinstance(device, torch.device) else device.type
    if cfg.amp == "bf16":
        return torch.autocast(dev, dtype=torch.bfloat16)
    return torch.autocast(dev, enabled=False)


# ============================================================================ inference
@torch.no_grad()
def run_steps(model, xt, xb, k_max, bs=32, record=None, cfg=None):
    """probabilities after supervision steps 1..k_max (one-shot / deep: one pass, repeated), batched.
    Returns P [N, len(record), T, G] (float32) and halting logits Q [N, k_max]."""
    model.eval()
    cfg = cfg or model.cfg
    N, T, G = xt.shape[:3]
    keep = list(range(1, k_max + 1)) if record is None else sorted(record)
    P = np.zeros((N, len(keep), T, G), np.float32)
    Q = np.zeros((N, k_max), np.float32)
    for i in range(0, N, bs):
        with autocast(cfg, xt.device):
            x, bus = model.inputs(xt[i:i + bs], xb[i:i + bs])
            y, z = model.init_state(len(x), T, G, xt[i:i + bs])
            if cfg.mode != "trm":
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


@torch.no_grad()
def hidden_at_steps(model, xt, xb, k_max, bs=32):
    """mean-pooled answer embedding after every step [N, k_max, D] (input of the halting head)"""
    model.eval()
    cfg = model.cfg
    N, T, G = xt.shape[:3]
    H = np.zeros((N, k_max, cfg.hidden), np.float32)
    for i in range(0, N, bs):
        with autocast(cfg, xt.device):
            x, bus = model.inputs(xt[i:i + bs], xb[i:i + bs])
            y, z = model.init_state(len(x), T, G, xt[i:i + bs])
            for k in range(k_max):
                y, z, _, _ = model.step(x, bus, y, z, grad=False)
                H[i:i + bs, k] = y.float().mean((1, 2)).cpu().numpy()
    return H


def viterbi(w, u0, U, D):
    """argmax_u sum_t w_t u_t over 0/1 schedules with min up U / min down D and initial status u0 (initial units have
    satisfied their minimum times, no carry-over, as in UCModel)"""
    T = len(w)
    NEG = -1e18
    cost = {(int(u0), U if u0 else D): 0.0}
    back = []
    for t in range(T):
        nxt, bp = {}, {}
        for (s, k), c in cost.items():
            for s2 in (0, 1):
                if s2 == s:
                    k2 = min(k + 1, U if s else D)
                else:
                    if (s == 1 and k < U) or (s == 0 and k < D):
                        continue
                    k2 = 1
                c2 = c + (w[t] if s2 else 0.0)
                if c2 > nxt.get((s2, k2), NEG):
                    nxt[(s2, k2)] = c2
                    bp[(s2, k2)] = (s, k)
        back.append(bp)
        cost = nxt
    st = max(cost, key=cost.get)
    out = np.zeros(T, np.int8)
    for t in range(T - 1, -1, -1):
        out[t] = st[0]
        st = back[t][st]
    return out


def allocate(p, sg):
    """water-filling of identical copies: copy j of a subgroup gets clip(sum_g p_g - j, 0, 1) per hour"""
    q = np.array(p, float, copy=True)
    for mem in sg:
        mu = p[:, mem].sum(1)
        for j, g in enumerate(mem):
            q[:, g] = np.clip(mu - j, 0.0, 1.0)
    return q


def decode(p, d, i, sysm, th=0.5, how="viterbi", sg=None):
    """schedule of instance i from probabilities p [T, G]:
       viterbi:   water-filling over identical copies (sg = subgroups of the instance; None = no allocation) -> per-unit
                  Viterbi on log-odds relative to th (min up/down, initial status) -> block adequacy repair -> min up/down
       threshold: p > th -> block adequacy repair -> min up/down repair (otsl.combo.block_decode)"""
    u0 = d["u0"][i]
    if how == "viterbi":
        q = allocate(p, sg) if sg else np.asarray(p, float)
        q = np.clip(q, 1e-4, 1 - 1e-4)
        w = np.log(q / (1 - q)) - math.log(th / (1 - th))
        u = np.stack([viterbi(w[:, g], u0[g], int(sysm.min_up[g]), int(sysm.min_dn[g])) for g in range(sysm.G)], 1)
    else:
        u = (np.asarray(p) > th).astype(np.int8)
    u = adequacy_repair_blocks(u, u0, d["load"][i], d["avail"][i], d["sr"][i], sysm)
    return repair_min_updown(u, u0, sysm.min_up, sysm.min_dn)


# ============================================================================ training (deep supervision)
def lr_at(cfg, step):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    f = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.lr_min_frac + (1 - cfg.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, f))))


def make_optimizer(model, cfg, lr=None):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr or cfg.lr, betas=tuple(cfg.betas))


class Carry:
    """the training batch of TRM / HRM: B slots, each holding an instance and its carried (y, z); a slot is refilled
    with the next instance of the epoch permutation when its instance halts or reaches n_sup steps"""

    def __init__(self, model, xt, cfg, rng):
        self.cfg, self.rng, self.n = cfg, rng, len(xt)
        B = cfg.batch
        self.perm, self.ptr, self.epoch = rng.permutation(self.n), 0, 0
        self.idx = np.array([self._next() for _ in range(B)])
        T, G = xt.shape[1], xt.shape[2]
        self.y, self.z = model.init_state(B, T, G, xt[torch.as_tensor(self.idx, device=xt.device)])
        self.steps = np.zeros(B, int)
        self.min_steps = np.array([self._min() for _ in range(B)])
        self.done_steps = []

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

    @torch.no_grad()
    def advance(self, model, halt_logit, xt):
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
            hb = np.where(halted)[0]
            y0, z0 = model.init_state(len(hb), self.y.shape[1], self.y.shape[2],
                                      xt[torch.as_tensor(self.idx[hb], device=xt.device)])
            h = torch.as_tensor(hb, device=self.y.device)
            self.y[h], self.z[h] = y0.to(self.y.dtype), z0.to(self.z.dtype)
        return halted

    def state(self):
        return dict(perm=self.perm, ptr=self.ptr, epoch=self.epoch, idx=self.idx, steps=self.steps,
                    min_steps=self.min_steps, y=self.y.cpu(), z=self.z.cpu(), done=self.done_steps[-2000:])

    def load(self, st, device):
        self.perm, self.ptr, self.epoch = st["perm"], st["ptr"], st["epoch"]
        self.idx, self.steps, self.min_steps = st["idx"], st["steps"], st["min_steps"]
        self.y, self.z = st["y"].to(device), st["z"].to(device)
        self.done_steps = list(st.get("done", []))


def bce_loss(logit, target, weight=None):
    return F.binary_cross_entropy_with_logits(logit, target, weight=weight, reduction="none").mean((1, 2))


def wrong_count(logit, target, S=None):
    """wrong unit-hours of the rounded answer: per subgroup |round(expected count) - target count| (S = subgroup-mean
    matrices [B, G, G]), per singleton |round(p) - y|"""
    if S is None:
        return ((logit > 0) != (target > 0.5)).sum((1, 2))
    p = torch.sigmoid(logit)
    size = 1.0 / torch.diagonal(S, dim1=1, dim2=2)                       # [B, G]
    cnt = torch.einsum("bgh,bth->btg", S, p) * size[:, None]
    tc = target * size[:, None]
    return ((torch.round(cnt) - torch.round(tc)).abs() / size[:, None]).sum((1, 2))


@torch.no_grad()
def val_loss(model, xt, xb, target, cfg, steps=(1, 4, 16), bs=32, S=None):
    """mean BCE against the targets after each of `steps` supervision steps, and wrong unit-hours per instance"""
    P, _ = run_steps(model, xt, xb, max(steps), bs=bs, record=steps, cfg=cfg)
    y = target.cpu().numpy()
    out = {}
    for j, k in enumerate(sorted(steps)):
        p = np.clip(P[:, j], 1e-6, 1 - 1e-6)
        out[f"bce@{k}"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        lg = torch.logit(torch.as_tensor(p))
        out[f"wrong@{k}"] = float(wrong_count(lg, torch.as_tensor(y), S).float().mean())
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


def cost_weights(grad, y, c_ref, alpha):
    """per-decision BCE weights from the dispatch LP's sensitivities at the label: the first-order cost of flipping
    decision (t, g) is +grad (OFF -> ON) or -grad (ON -> OFF); w = 1 + alpha * log1p(max(cost, 0) / (1e-3 c_ref))"""
    flip = np.where(y > 0.5, -grad, grad)
    e = np.maximum(flip, 0.0) / (1e-3 * c_ref[:, None, None])
    return (1.0 + alpha * np.log1p(e)).astype(np.float32)


def train(cfg: TRMConfig, sysm, tr, va, ckpt, device="cpu", log_every=50, val_every=500, save_every_s=300,
          logf=print, val_steps=(1, 4, 16), grad_lp=None):
    """deep-supervision training with resume from `ckpt` (written every save_every_s seconds and at the end).
    grad_lp: [N, T, G] dispatch-LP sensitivities at the training labels (needed when cfg.cost_weight > 0)."""
    feats = TRMFeatures(sysm, lp=cfg.lp_feats, sym=cfg.sym).fit(tr)
    xt, xb = feats(tr, device)
    tgt = torch.as_tensor(targets(sysm, tr, tr["y"], cfg.sym), device=device)
    W = None
    if cfg.cost_weight > 0:
        w = cost_weights(grad_lp, tr["y"], tr["c_rel"], cfg.cost_weight)
        if cfg.sym == "count":       # copies share their subgroup's weight
            w = np.einsum("igh,ith->itg", subgroup_mean_matrix(sysm, tr["u0"]), w)
        W = torch.as_tensor(w, device=device)
    S_tr = torch.as_tensor(subgroup_mean_matrix(sysm, tr["u0"]), device=device) if cfg.sym == "count" else None
    vxt, vxb = feats(va, device)
    vtgt = torch.as_tensor(targets(sysm, va, va["u"], cfg.sym), device=device)
    S_va = torch.as_tensor(subgroup_mean_matrix(sysm, va["u0"][:60]), device=device) if cfg.sym == "count" else None
    model = build(cfg, sysm, feats, device)
    ema = EMA(model, cfg.ema)
    opt = make_optimizer(model, cfg)
    rng = np.random.default_rng(cfg.seed)
    carry = Carry(model, xt, cfg, rng)
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
        l_ans = bce_loss(logit, target, None if W is None else W[idx])
        with torch.no_grad():
            wrong = wrong_count(logit.detach(), target, None if S_tr is None else S_tr[idx])
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
        carry.advance(model, q.detach().float().cpu().numpy(), xt)
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
            v = dict(step=step, **val_loss(ema.shadow, vxt[:60], vxb[:60], vtgt[:60], cfg, vs, S=S_va))
            log["val"].append(v)
            logf("[trm] val(60, EMA) " + json.dumps(v))
        log["time_s"] = time.time() - t0
        if time.time() - t_save > save_every_s or step == cfg.steps:
            save_ckpt(ckpt, model, ema, opt, step, carry, rng, cfg, feats, log)
            t_save = time.time()
    return model, ema, feats, log


# ============================================================================ REINFORCE with the exact LP critic
def train_reinforce(model, cfg, sysm, feats, tr, oracle, ckpt, steps=150, bs=16, n_samples=6, lr=1e-4, k_sup=None,
                    random_step=False, anchor=0.5, device="cpu", seed=0, logf=print, log_every=10, save_every_s=300,
                    ema_decay=0.98, base_ckpt=None):
    """Cost-aware fine-tuning: (k - 1) supervision steps without gradient, step k with; k = k_sup (default n_sup) or,
    with random_step, uniform in 1..k_sup per batch (every step learns to improve the schedule; under the leave-one-out
    baseline, rewarding step k's cost and rewarding its improvement over step k - 1 are the same). n_samples
    Bernoulli schedules per instance are decoded as at inference (Viterbi on the sample, adequacy repair), priced by the
    dispatch LP; reward -log(cost / c_rel); plus `anchor` x BCE to the targets. Resumable from ckpt."""
    k_sup = k_sup or (cfg.n_sup if cfg.mode == "trm" else 1)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    xt, xb = feats(tr, device)
    tgt = torch.as_tensor(targets(sysm, tr, tr["y"], cfg.sym), device=device)
    sgs = subgroups(sysm, tr["u0"]) if cfg.sym == "count" else None
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
        k = int(rng.integers(1, k_sup + 1)) if random_step else k_sup
        ii = torch.as_tensor(idx, device=device)
        model.eval()
        with autocast(cfg, device):
            with torch.no_grad():
                x, bus = model.inputs(xt[ii], xb[ii])
                y, z = model.init_state(bs, T, G, xt[ii])
                for _ in range(k - 1):
                    y, z, _, _ = model.step(x, bus, y, z, grad=False)
            x, bus = model.inputs(xt[ii], xb[ii])
            _, _, logit, _ = model.step(x, bus, y, z, grad=True)
        p = torch.sigmoid(logit.detach().float()).cpu()
        smp = torch.rand((n_samples,) + p.shape, generator=gen) < p
        smp_d = smp.to(device)
        logp = (smp_d * F.logsigmoid(logit) + (~smp_d) * F.logsigmoid(-logit)).sum((-1, -2))
        us = smp.numpy().astype(np.float64)
        # samples are already symmetric-broken draws: decode each unit on its own sample (no allocation)
        rep = np.array([[decode(us[s_, b], tr, idx[b], sysm, th=0.5, how="viterbi") for b in range(bs)]
                        for s_ in range(n_samples)])
        flat = np.tile(idx, n_samples)
        c, sh, so = oracle.evaluate(tr, flat, rep.reshape((-1, T, G)), flat)
        c, sh, so = c.reshape(n_samples, bs), sh.reshape(n_samples, bs), so.reshape(n_samples, bs)
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
        hist.append(dict(step=step, k=k, gap_rel=float(np.mean(c / tr["c_rel"][idx][None] - 1)),
                         served=float(np.mean((sh < 1e-6) & (so < 1e-6))), bce=float(l_bce), lps=int(oracle.n),
                         s=time.time() - t0))
        if step % log_every == 0 or step == steps:
            h = hist[-log_every:]
            logf(f"[rl] step {step} sampled gap to relaxation {np.mean([x['gap_rel'] for x in h]) * 100:.3f}% "
                 f"served {np.mean([x['served'] for x in h]) * 100:.1f}% bce {np.mean([x['bce'] for x in h]):.4f} "
                 f"LPs {oracle.n} ({time.time() - t0:.0f}s)")
        if time.time() - t_save > save_every_s or step == steps:
            tmp = ckpt + ".tmp"
            torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step,
                            hist=hist, rng=rng.bit_generator.state, gen=gen.get_state(), base=base_ckpt,
                            cfg=cfg.to_json(), feats=feats.state()), tmp)
            os.replace(tmp, ckpt)
            t_save = time.time()
    return model, ema, hist


# ============================================================================ post-hoc halting head on LP-priced targets
def fit_halting(H, good, epochs=300, lr=1e-2, wd=1e-3, seed=0):
    """logistic regression of `good` (step k's schedule within 0.5 % of the best known cost) on the pooled answer
    embedding H [M, D]. Returns (weight [D], bias)."""
    torch.manual_seed(seed)
    X = torch.as_tensor(H, dtype=torch.float32)
    yv = torch.as_tensor(good, dtype=torch.float32)
    lin = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(lin(X).squeeze(-1), yv)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin.weight.detach().numpy()[0], float(lin.bias.detach())
