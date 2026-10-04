"""Training loops for Model 1 (switching) and Model 2 (dispatch)."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import DispatchNet, SwitchGNN, SwitchMLP

torch.set_num_threads(4)


# ----------------------------------------------------------------------------------- Model 1
class Model1:
    """Wraps a switching network + featurizer; predicts p_open [B, L] (0 on non-switchable lines)."""

    def __init__(self, net, feat, switchable, kind="gnn"):
        self.net, self.feat, self.kind = net, feat, kind
        self.sw = torch.as_tensor(switchable)

    def logits(self, d, idx=None):
        if self.kind == "gnn":
            x, e = self.feat(d, idx)
            out = self.net(x, e)
        else:
            out = self.net(self.feat.flat(d, idx))
        return out.masked_fill(~self.sw, -30.0)

    @torch.no_grad()
    def predict(self, d, bs=256):
        self.net.eval()
        n = len(d["pd"])
        return np.concatenate([torch.sigmoid(self.logits(d, np.arange(i, min(i + bs, n)))).numpy()
                               for i in range(0, n, bs)])


def build_model1(case, feat, switchable, kind="gnn", hidden=64, layers=6, seed=0):
    torch.manual_seed(seed)
    if kind == "gnn":
        net = SwitchGNN(case.f_bus, case.t_bus, case.n_bus, feat.node_dim, feat.edge_dim, hidden, layers)
    else:
        net = SwitchMLP(case.n_bus * feat.node_dim + case.n_line * feat.edge_dim, case.n_line, hidden=4 * hidden)
    return Model1(net, feat, switchable, kind)


def train_bce(m1: Model1, train, val, epochs=150, lr=1e-3, bs=64, soft_targets=None, log_every=25, seed=0):
    """Imitation of MILP switching decisions (the framework's 'Model 1 Loss').

    soft_targets: optional [n, L] targets in [0,1] replacing the hard MILP labels (e.g. the average
    over several (near-)optimal topologies)."""
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(1.0 - train["z"], dtype=torch.float32) if soft_targets is None else \
        torch.as_tensor(soft_targets, dtype=torch.float32)
    yv = torch.as_tensor(1.0 - val["z"], dtype=torch.float32)
    sw = m1.sw
    pos = y[:, sw].mean().item()
    pw = torch.tensor(min((1 - pos) / max(pos, 1e-6), 50.0))
    opt = torch.optim.AdamW(m1.net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = np.inf, None
    n = len(y)
    t0 = time.time()
    for ep in range(epochs):
        m1.net.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg = m1.logits(train, idx)
            loss = F.binary_cross_entropy_with_logits(lg[:, sw], y[idx][:, sw], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m1.net.eval()
        with torch.no_grad():
            lv = m1.logits(val)
            vl = F.binary_cross_entropy_with_logits(lv[:, sw], yv[:, sw], pos_weight=pw).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(m1.net.state_dict())
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  [bce] ep {ep:3d} train {loss.item():.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    m1.net.load_state_dict(best_state)
    return m1


def sample_topologies(logits, K, n_samples, gen):
    """Sample switch-off sets from independent Bernoullis; samples exceeding the budget are
    truncated to their K most likely lines. Returns open-masks [n_samples, B, L] (bool) and the
    log-probability of the sampled (pre-truncation) Bernoulli draws."""
    p = torch.sigmoid(logits)
    u = torch.rand((n_samples,) + p.shape, generator=gen)
    o = u < p
    logp = (o * F.logsigmoid(logits) + (~o) * F.logsigmoid(-logits)).sum(-1)
    over = o.sum(-1) > K
    if over.any():
        sc = torch.where(o, logits.expand_as(o), torch.full_like(u, -1e9))
        kth = sc.topk(K, -1).values[..., -1:]
        o = torch.where(over.unsqueeze(-1), o & (sc >= kth), o)
    return o, logp


def train_reinforce(m1: Model1, train, oracle, K, switch_cost, steps=300, bs=32, n_samples=8, lr=3e-4,
                    infeas_penalty=0.02, bce_weight=0.0, log_every=25, seed=0, key_offset=0):
    """Cost-aware fine-tuning: maximise the LP-evaluated saving of sampled topologies.

    reward = (c_allclosed - c(z) - switch_cost*|open|) / c_allclosed   (infeasible -> -penalty)
    Uses a leave-one-out baseline over the n_samples drawn for each scenario (RLOO).
    The exact LP is the critic, so there is no surrogate bias and degenerate labels do not matter.
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m1.net.parameters(), lr=lr)
    y = torch.as_tensor(1.0 - train["z"], dtype=torch.float32)
    sw = m1.sw
    n = len(train["pd"])
    hist = []
    t0 = time.time()
    for step in range(steps):
        m1.net.train()
        idx = rng.choice(n, bs, replace=False)
        lg = m1.logits(train, idx)
        o, logp = sample_topologies(lg.detach(), K, n_samples, gen)
        # recompute log-prob with grad for the (pre-truncation) draws
        logp = (o * F.logsigmoid(lg) + (~o) * F.logsigmoid(-lg))[..., sw].sum(-1)   # [S, B]
        z = (~o).numpy().astype(np.int8)
        pds = np.repeat(train["pd"][idx][None], n_samples, 0).reshape(-1, train["pd"].shape[1])
        keys = np.tile(idx + key_offset, n_samples)
        c = oracle.costs(pds, z.reshape(-1, z.shape[-1]), keys).reshape(n_samples, bs)
        c0 = train["c0"][idx][None]
        r = (c0 - c - switch_cost * (1 - z).sum(-1)) / c0
        r = np.where(np.isfinite(r), r, -infeas_penalty)
        r = torch.as_tensor(r, dtype=torch.float32)
        adv = r - (r.sum(0, keepdim=True) - r) / (n_samples - 1)
        loss = -(adv * logp).mean()
        if bce_weight > 0:
            loss = loss + bce_weight * F.binary_cross_entropy_with_logits(lg[:, sw], y[idx][:, sw])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.net.parameters(), 1.0)
        opt.step()
        hist.append(r.mean().item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  [rl] step {step:4d} mean reward {np.mean(hist[-log_every:]) * 100:.4f}% "
                  f"infeasible {(~np.isfinite(c)).mean() * 100:.1f}%  LPs {oracle.n_solves} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return m1, hist


# ----------------------------------------------------------------------------------- Model 2
def build_model2(case, feat, mode, hidden=64, layers=6, seed=0):
    torch.manual_seed(seed)
    return DispatchNet(case, feat.node_dim, feat.edge_dim, mode=mode, hidden=hidden, layers=layers)


def train_model2(m2: DispatchNet, feat, d, dv, epochs=100, lr=1e-3, bs=64, w_flow=0.0, w_cost=0.0,
                 log_every=20, seed=0):
    """Supervised on fixed-topology LP solutions (the framework's 'Model 2 Loss').

    d must contain pd, zz (topology), lp_pg, lp_va, lp_c. For mode='physics' an optional thermal
    overload penalty (w_flow) and cost term (w_cost) can be added since flows are available."""
    rng = np.random.default_rng(seed)
    P = m2.phys
    pg_scale = torch.as_tensor(np.maximum(d["lp_pg"].std(0), 1e-3))
    va_scale = torch.as_tensor(np.maximum(d["lp_va"].std(0), 1e-3))
    opt = torch.optim.AdamW(m2.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(d["pd"])
    best, best_state = np.inf, None
    t0 = time.time()

    def batch_loss(dd, idx):
        x, e = feat(dd, idx)
        z = torch.as_tensor(dd["zz"][idx], dtype=torch.float64)
        pd = torch.as_tensor(dd["pd"][idx])
        pg, va = m2(x, e, z, pd)
        l_pg = (((pg - torch.as_tensor(dd["lp_pg"][idx])) / pg_scale) ** 2).mean()
        l_va = (((va - torch.as_tensor(dd["lp_va"][idx])) / va_scale) ** 2).mean()
        loss = l_pg + l_va
        if w_flow > 0:
            f = P.flows(va, z)
            loss = loss + w_flow * (F.relu(f.abs() - P.fmax * z) ** 2).sum(-1).mean()
        if w_cost > 0:
            loss = loss + w_cost * (P.gen_cost(pg) / torch.as_tensor(dd["lp_c"][idx]) - 1).mean()
        return loss

    for ep in range(epochs):
        m2.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            loss = batch_loss(d, perm[i:i + bs])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt.step()
        sched.step()
        m2.eval()
        with torch.no_grad():
            vl = np.mean([batch_loss(dv, np.arange(i, min(i + 256, len(dv["pd"])))).item()
                          for i in range(0, len(dv["pd"]), 256)])
        if vl < best:
            best, best_state = vl, copy.deepcopy(m2.state_dict())
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  [m2:{m2.mode}] ep {ep:3d} train {loss.item():.4f} val {vl:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
    m2.load_state_dict(best_state)
    return m2


@torch.no_grad()
def predict_model2(m2, feat, d, bs=256):
    m2.eval()
    pgs, vas = [], []
    for i in range(0, len(d["pd"]), bs):
        idx = np.arange(i, min(i + bs, len(d["pd"])))
        x, e = feat(d, idx)
        pg, va = m2(x, e, torch.as_tensor(d["zz"][idx], dtype=torch.float64), torch.as_tensor(d["pd"][idx]))
        pgs.append(pg.numpy()); vas.append(va.numpy())
    return np.concatenate(pgs), np.concatenate(vas)
