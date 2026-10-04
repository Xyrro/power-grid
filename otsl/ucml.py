"""Learning for unit commitment: features, Model 1 (commitment GNN / MLP), Model 2 (dispatch),
decoding, the dispatch-LP oracle, training loops and metrics."""
from __future__ import annotations

import copy
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import EdgeGNN, mlp  # noqa: F401  (also sets torch threads)
from .uc import UCModel, UCScenario, load_rts_gmlc, repair_min_updown

# ----------------------------------------------------------------------------------- features
TYPES = ["CT", "CC", "STEAM", "NUCLEAR"]


def symmetry_rank(sysm, u0):
    """Rank of each unit inside its group of identical units with the same initial status (0, 1, ...),
    plus the group size. Identical units get identical features, so without this a GNN cannot tell
    them apart (permutation equivariance) and cannot learn which copy to switch."""
    G = sysm.G
    rank = np.zeros((len(u0), G))
    size = np.ones((len(u0), G))
    for grp in sysm.identical_groups():
        for i in range(len(u0)):
            for val in (0, 1):
                mem = [g for g in grp if u0[i, g] == val]
                for r, g in enumerate(mem):
                    rank[i, g] = r
                    size[i, g] = len(mem)
    return rank, size


def canonical_labels(sysm, u, u0):
    """Symmetry-broken labels: inside each identical group (same u0), reorder the schedules so the copy
    with the lowest rank is on most (lexicographically largest schedule). Cost and flows are unchanged."""
    u = u.copy()
    for grp in sysm.identical_groups():
        for i in range(len(u)):
            for val in (0, 1):
                mem = [g for g in grp if u0[i, g] == val]
                if len(mem) < 2:
                    continue
                cols = [tuple(u[i, :, g]) for g in mem]
                order = sorted(range(len(mem)), key=lambda j: cols[j], reverse=True)
                ui = u[i]
                ui[:, mem] = np.stack([np.array(cols[j]) for j in order], axis=1)
    return u


class UCFeaturizer:
    """Bus features (load, renewables per period [+ LP-relaxation prices]), unit features (static data,
    initial status, symmetry rank [+ LP-relaxation commitment]), edge features (line data [+ relaxed
    loading])."""

    def __init__(self, sysm, train, relax=False, sym=False):
        self.s, self.relax, self.sym = sysm, relax, sym
        s = sysm
        G = s.G
        onehot = np.array([[t == k for k in TYPES] for t in s.utype], float)
        avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
        self.g_static = np.c_[s.pmax, s.pmin, avg / 1e4, s.seg_c / 1e4, s.c_su / 1e4, s.min_up / 24, s.min_dn / 24,
                              np.minimum(s.ramp / s.pmax, 5), onehot]
        self.Rb = np.zeros((s.n_bus, len(s.r_pmax)))
        self.Rb[s.r_bus, np.arange(len(s.r_pmax))] = 1
        self.capbus = np.bincount(s.gbus, weights=s.pmax, minlength=s.n_bus)
        self.e_static = np.c_[np.log(s.b), s.fmax]
        xb, xg, xe = self._raw(train)
        self.stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6) for a in (xb, xg, xe)]
        self.bus_dim, self.gen_dim, self.edge_dim = xb.shape[-1], xg.shape[-1], xe.shape[-1]

    def _raw(self, d, idx=None):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        load, avail, u0 = sel(d["load"]), sel(d["avail"]), sel(d["u0"])
        B, T, N = load.shape
        ren = avail @ self.Rb.T                                            # [B, T, N]
        xb = [load.transpose(0, 2, 1), ren.transpose(0, 2, 1), np.broadcast_to(self.capbus[None, :, None], (B, N, 1))]
        xg = [np.broadcast_to(self.g_static, (B,) + self.g_static.shape), u0[..., None].astype(float),
              np.broadcast_to((load.sum(2) - avail.sum(2))[:, None, :], (B, self.s.G, T))]  # net load per period
        xe = [np.broadcast_to(self.e_static, (B,) + self.e_static.shape)]
        if self.sym:
            rank, size = symmetry_rank(self.s, u0)
            xg += [rank[..., None], size[..., None]]
        if self.relax:
            xb.append(sel(d["lmp_rel"]).transpose(0, 2, 1) / 1e4)
            xg.append(sel(d["u_rel"]).transpose(0, 2, 1))
            xe.append(np.abs(sel(d["flow_rel"])).transpose(0, 2, 1) / self.s.fmax[None, :, None])
        return np.concatenate(xb, -1), np.concatenate(xg, -1), np.concatenate(xe, -1)

    def __call__(self, d, idx=None):
        out = []
        for a, (m, s) in zip(self._raw(d, idx), self.stats):
            out.append(torch.as_tensor((a - m) / s, dtype=torch.float32))
        return out

    def flat(self, d, idx=None):
        return torch.cat([x.flatten(1) for x in self(d, idx)], 1)


# ----------------------------------------------------------------------------------- Model 1
class CommitGNN(nn.Module):
    """RACLearn-style: unit encoder -> summed into buses with a bus encoder -> message passing over the
    transmission graph -> per-unit decoder producing one logit per period."""

    def __init__(self, sysm, bus_in, gen_in, edge_in, T, hidden=64, layers=4, dropout=0.0):
        super().__init__()
        self.register_buffer("gbus", torch.as_tensor(sysm.gbus, dtype=torch.long))
        self.N = sysm.n_bus
        self.gen_enc = mlp(gen_in, hidden, hidden)
        self.gnn = EdgeGNN(sysm.f_bus, sysm.t_bus, sysm.n_bus, bus_in + hidden, edge_in, hidden, layers)
        self.dec = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, T))

    def forward(self, xb, xg, xe):
        hg = self.gen_enc(xg)                                              # [B, G, H]
        agg = torch.zeros(xb.shape[0], self.N, hg.shape[-1]).index_add_(1, self.gbus, hg)
        h, _ = self.gnn(torch.cat([xb, agg], -1), xe)
        return self.dec(torch.cat([h[:, self.gbus], hg], -1)).transpose(1, 2)   # [B, T, G]


class CommitMLP(nn.Module):
    def __init__(self, d_in, T, G, hidden=256):
        super().__init__()
        self.T, self.G = T, G
        self.net = mlp(d_in, hidden, T * G, n=3)

    def forward(self, x):
        return self.net(x).view(-1, self.T, self.G)


class UCModel1:
    def __init__(self, net, feat, kind):
        self.net, self.feat, self.kind = net, feat, kind

    def logits(self, d, idx=None):
        if self.kind == "gnn":
            return self.net(*self.feat(d, idx))
        return self.net(self.feat.flat(d, idx))

    @torch.no_grad()
    def predict(self, d, bs=256, mc=0):
        self.net.eval()
        n = len(d["load"])
        if mc:
            for mod in self.net.modules():
                if isinstance(mod, nn.Dropout):
                    mod.train()
        outs, sds = [], []
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            if mc:
                s = torch.stack([torch.sigmoid(self.logits(d, idx)) for _ in range(mc)])
                outs.append(s.mean(0).numpy()); sds.append(s.std(0).numpy())
            else:
                outs.append(torch.sigmoid(self.logits(d, idx)).numpy())
        self.net.eval()
        return (np.concatenate(outs), np.concatenate(sds)) if mc else np.concatenate(outs)


def build_uc_model1(sysm, feat, T, kind="gnn", seed=0, dropout=0.0):
    torch.manual_seed(seed)
    if kind == "gnn":
        net = CommitGNN(sysm, feat.bus_dim, feat.gen_dim, feat.edge_dim, T, dropout=dropout)
    else:
        net = CommitMLP(sysm.n_bus * feat.bus_dim + sysm.G * feat.gen_dim + sysm.L * feat.edge_dim, T, sysm.G)
    return UCModel1(net, feat, kind)


def train_uc_bce(m1, tr, va, targets, epochs=60, lr=1e-3, bs=64, seed=0, log_every=20):
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(targets, dtype=torch.float32)
    yv = torch.as_tensor(va["u_target"], dtype=torch.float32)
    opt = torch.optim.AdamW(m1.net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = np.inf, None
    n = len(y)
    t0 = time.time()
    for ep in range(epochs):
        m1.net.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(m1.logits(tr, idx), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m1.net.eval()
        with torch.no_grad():
            vl = np.mean([F.binary_cross_entropy_with_logits(m1.logits(va, np.arange(i, min(i + 256, len(yv)))),
                                                              yv[i:i + 256]).item() for i in range(0, len(yv), 256)])
        if vl < best:
            best, state = vl, copy.deepcopy(m1.net.state_dict())
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  [uc-bce] ep {ep:3d} train {loss.item():.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    m1.net.load_state_dict(state)
    return m1


# ----------------------------------------------------------------------------------- LP oracle
_W = {}


def _init(cfg):
    s = load_rts_gmlc(line_scale=cfg.get("line_scale", 1.0))
    _W.update(s=s, m=UCModel(s, T=cfg["T"], network=cfg.get("network", True)), cfg=cfg)


def _dispatch(args):
    load, avail, u0, sr, u = args
    sol = _W["m"].solve_dispatch(UCScenario(load=load, avail=avail, u0=u0, sr=sr), u)
    return sol.obj, sol.shed, sol.short


class DispatchOracle:
    """Fixed-commitment dispatch LPs in parallel, memoised by (instance key, commitment)."""

    def __init__(self, cfg, workers=2):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))
        self.cache, self.n = {}, 0

    def evaluate(self, d, idx, us, keys):
        """us: [M, T, G]; idx: instance index per row. Returns cost, slack (shed+spill), reserve shortfall."""
        ck = [(k, np.asarray(u, np.int8).tobytes()) for k, u in zip(keys, us)]
        todo = {}
        for j, c in enumerate(ck):
            if c not in self.cache and c not in todo:
                todo[c] = j
        if todo:
            js = list(todo.values())
            res = self.pool.map(_dispatch, [(d["load"][idx[j]], d["avail"][idx[j]], d["u0"][idx[j]], d["sr"][idx[j]],
                                             us[j]) for j in js], chunksize=max(1, len(js) // 16))
            self.n += len(js)
            for j, r in zip(js, res):
                self.cache[ck[j]] = r
        out = np.array([self.cache[c] for c in ck])
        return out[:, 0], out[:, 1], out[:, 2]

    def close(self):
        self.pool.close()


def train_uc_reinforce(m1, tr, oracle, steps=200, bs=16, n_samples=6, lr=3e-4, seed=0, log_every=25, repair=None):
    """Cost-aware fine-tuning with the exact dispatch LP as the critic (RLOO baseline)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m1.net.parameters(), lr=lr)
    n = len(tr["load"])
    hist, t0 = [], time.time()
    for step in range(steps):
        m1.net.train()
        idx = rng.choice(n, bs, replace=False)
        lg = m1.logits(tr, idx)                                            # [B, T, G]
        p = torch.sigmoid(lg.detach())
        smp = (torch.rand((n_samples,) + p.shape, generator=gen) < p)      # [S, B, T, G]
        logp = (smp * F.logsigmoid(lg) + (~smp) * F.logsigmoid(-lg)).sum((-1, -2))
        us = smp.numpy().astype(np.int8)
        if repair is not None:
            us = np.array([[repair(us[k, b], tr["u0"][idx[b]]) for b in range(bs)] for k in range(n_samples)])
        flat_idx = np.tile(idx, n_samples)
        c, _, _ = oracle.evaluate(tr, flat_idx, us.reshape((-1,) + us.shape[2:]), flat_idx)
        c = c.reshape(n_samples, bs)
        r = torch.as_tensor(-np.log(c / tr["obj"][idx][None]), dtype=torch.float32)   # 0 = MILP cost
        adv = r - (r.sum(0, keepdim=True) - r) / (n_samples - 1)
        adv = adv / (adv.std() + 1e-8)
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.net.parameters(), 1.0)
        opt.step()
        hist.append(float(np.exp(-r.mean()) - 1))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  [uc-rl] step {step:4d} mean sampled gap {np.mean(hist[-log_every:]) * 100:.3f}%  "
                  f"LPs {oracle.n} ({time.time() - t0:.0f}s)", flush=True)
    return m1, hist


# ----------------------------------------------------------------------------------- metrics
def uc_metrics(cost, shed, short, d, u=None, label="", **extra):
    ref = d["obj"]
    gap = (cost - ref) / ref * 100
    hard = (shed < 1e-6) & (short < 1e-6)
    out = {"method": label, "gap_mean_%": float(gap.mean()), "gap_median_%": float(np.median(gap)),
           "gap_p95_%": float(np.percentile(gap, 95)), "no_shed_no_shortfall_%": float(hard.mean() * 100),
           # VOLL-priced shedding dominates means; report the mean over instances served without shedding too
           "gap_mean_served_%": float(gap[hard].mean()) if hard.any() else float("nan"),
           "matches_or_beats_milp_%": float((gap <= 1e-3).mean() * 100)}
    if u is not None:
        out["unit_hour_accuracy_%"] = float((u == d["u"]).mean() * 100)
        out["exact_match_%"] = float((u == d["u"]).all((1, 2)).mean() * 100)
        out["units_on"] = float(u.sum((1, 2)).mean() / u.shape[1])
    out.update(extra)
    return out


# ----------------------------------------------------------------------------------- Model 2 (UC)
class UCPhysics(nn.Module):
    """Differentiable dispatch physics for a given (possibly relaxed) commitment u [B, T, G]."""

    def __init__(self, sysm, reg=1e-6):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
        A = np.zeros((sysm.L, sysm.n_bus)); A[np.arange(sysm.L), sysm.f_bus] = 1; A[np.arange(sysm.L), sysm.t_bus] = -1
        Cg = np.zeros((sysm.n_bus, sysm.G)); Cg[sysm.gbus, np.arange(sysm.G)] = 1
        Cr = np.zeros((sysm.n_bus, len(sysm.r_pmax))); Cr[sysm.r_bus, np.arange(len(sysm.r_pmax))] = 1
        for k, v in dict(A=A, Cg=Cg, Cr=Cr, b=sysm.b, fmax=sysm.fmax, pmin=sysm.pmin, pmax=sysm.pmax,
                         c_nl=sysm.c_nl, c_su=sysm.c_su, ramp=sysm.ramp, seg_w=sysm.seg_w, seg_c=sysm.seg_c).items():
            self.register_buffer(k, f(v))
        Lap = A.T @ np.diag(sysm.b) @ A
        keep = [i for i in range(sysm.n_bus) if i != sysm.ref]
        self.register_buffer("Linv", f(np.linalg.inv(Lap[np.ix_(keep, keep)] + reg * np.eye(len(keep)))))
        self.register_buffer("keep", torch.as_tensor(keep))

    def balance(self, p, r, u, load):
        """Exact power balance per period: raise committed units toward pmax, or curtail renewables then
        lower units toward pmin. Anything left is shed / spilled (the commitment cannot serve the load)."""
        D = load.sum(-1, keepdim=True)
        S = p.sum(-1, keepdim=True) + r.sum(-1, keepdim=True)
        up = (u * self.pmax - p).sum(-1, keepdim=True).clamp_min(1e-9)
        a = ((D - S) / up).clamp(0, 1)
        p = p + a * (u * self.pmax - p)
        S = p.sum(-1, keepdim=True) + r.sum(-1, keepdim=True)
        rs = r.sum(-1, keepdim=True).clamp_min(1e-9)
        b_ = ((S - D) / rs).clamp(0, 1)
        r = r * (1 - b_)
        S = p.sum(-1, keepdim=True) + r.sum(-1, keepdim=True)
        dn = (p - u * self.pmin).sum(-1, keepdim=True).clamp_min(1e-9)
        c = ((S - D) / dn).clamp(0, 1)
        p = p - c * (p - u * self.pmin)
        mismatch = (p.sum(-1) + r.sum(-1) - D.squeeze(-1)).abs()          # shed or spill
        return p, r, mismatch

    def angles(self, p, r, load):
        inj = p @ self.Cg.T + r @ self.Cr.T - load                         # [B, T, N]
        th = torch.einsum("ij,btj->bti", self.Linv, inj[..., self.keep])
        va = torch.zeros_like(inj)
        va[..., self.keep] = th
        return va

    def flows(self, va):
        return self.b * (va @ self.A.T)

    def cost(self, p, u, u0, mismatch):
        """No-load + convex piecewise-linear energy cost + start-ups + VOLL on mismatch ($)."""
        x = (p - u * self.pmin).clamp_min(0)
        e = torch.zeros_like(p)
        cum = torch.zeros_like(p)
        for k in range(self.seg_w.shape[1]):
            take = torch.minimum((x - cum).clamp_min(0), self.seg_w[:, k] * torch.ones_like(x))
            e = e + take * self.seg_c[:, k]
            cum = cum + self.seg_w[:, k]
        prev = torch.cat([u0.unsqueeze(1), u[:, :-1]], 1)
        su = (u - prev).clamp_min(0)
        from .uc import VOLL
        return (u * self.c_nl + e + su * self.c_su).sum((-1, -2)) + VOLL * mismatch.sum(-1)

    def violations(self, p, va, u, load, r):
        f = self.flows(va)
        kcl = (p @ self.Cg.T + r @ self.Cr.T - f @ self.A - load).abs()   # [B, T, N]
        gen = F.relu(u * self.pmin - p) + F.relu(p - u * self.pmax)
        line = F.relu(f.abs() - self.fmax)
        both = u[:, 1:] * u[:, :-1]
        ramp = F.relu((p[:, 1:] - p[:, :-1]).abs() - self.ramp) * both if p.shape[1] > 1 else torch.zeros_like(p[:, :1])
        return {"kcl_max": kcl.amax((-1, -2)), "kcl_sum": kcl.sum((-1, -2)), "gen_max": gen.amax((-1, -2)),
                "line_max": line.amax((-1, -2)), "line_n": (line > 1e-4).sum((-1, -2)).double(),
                "ramp_max": ramp.amax((-1, -2)) if ramp.numel() else torch.zeros(p.shape[0], dtype=p.dtype)}


class UCDispatchNet(nn.Module):
    """Model 2 for UC. mode='direct': regress (PG, VA) (the framework). mode='physics': predict each unit's
    position in [pmin, pmax] -> exact balance repair -> VA from DC power flow."""

    def __init__(self, sysm, feat, T, mode="physics", hidden=64, layers=4):
        super().__init__()
        self.mode, self.T = mode, T
        self.phys = UCPhysics(sysm)
        self.backbone = CommitGNN(sysm, feat.bus_dim, feat.gen_dim + T, feat.edge_dim, T, hidden, layers)
        self.va_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, T))
        self._h = None

        def hook(mod, inp, out):
            self._h = out[0]
        self.backbone.gnn.register_forward_hook(hook)

    def forward(self, xb, xg, xe, u, load, avail, u0):
        uf = u.float()
        out = self.backbone(xb, torch.cat([xg, uf.transpose(1, 2)], -1), xe).double()   # [B, T, G]
        P = self.phys
        if self.mode == "direct":
            pg = P.pmin + (P.pmax - P.pmin) * out                          # unconstrained regression
            va = self.va_head(self._h).transpose(1, 2).double() * 0.3
            return pg, va, avail.clone()
        ud = u.double()
        p = ud * (P.pmin + (P.pmax - P.pmin) * torch.sigmoid(out))
        p, r, _ = P.balance(p, avail.clone(), ud, load)
        va = P.angles(p, r, load)
        return p, va, r
