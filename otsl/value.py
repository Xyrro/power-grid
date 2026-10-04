"""Learned switching values: an alternative to imitating (ambiguous) MILP labels.

For a topology z, the *switching value* of closed line l is the exact cost change of opening it,
    v_l(z) = C(z - e_l) - C(z)          (LP-evaluated; +inf if infeasible).
Unlike the MILP's argmin, v(z) is unique (no degeneracy), dense (one target per line), and
cheap: one LP per line (ms each) instead of one MILP per scenario.

Exhaustive greedy = repeatedly open the line with the best v_l (K * |L| LPs). The learned version
predicts v(z) with a topology-gated GNN from the duals of ONE LP at z, then LP-verifies only the
top-R predictions: K * (1 + R) LPs.
"""
from __future__ import annotations

import copy
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F

from .data import _init, _W
from .models import EdgeGNN, mlp


def _lp_full(args):
    pd, z = args
    s = _W["m"].solve_lp(pd, z)
    if not s.ok:
        return None
    return s.obj, s.lmp, s.mu, s.gamma, s.flow, s.va, s.pg


class GreedyOracle:
    """Exact LP evaluations for greedy search, parallelised over candidate lines."""

    def __init__(self, cfg, workers=4):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))
        self.n_lp = 0

    def solve(self, pds, zs):
        res = self.pool.map(_lp_full, list(zip(pds, zs)), chunksize=max(1, len(zs) // 64))
        self.n_lp += len(zs)
        return res

    def close(self):
        self.pool.close()


def state_record(pd, z, sol):
    obj, lmp, mu, gamma, flow, va, pg = sol
    return {"pd": pd, "z": z.copy(), "c": obj, "lmp0": lmp, "mu0": mu, "gamma0": gamma, "flow0": flow,
            "va0": va, "pg0": pg}


def exhaustive_greedy(oracle, pds, sw, K, switch_cost, record=False):
    """Greedy OTS with exact LP evaluation of every switchable line at every step.
    Returns final z, cost, #LPs per scenario and (optionally) the visited states with exact values."""
    n, L = len(pds), len(sw)
    z = np.ones((n, L), np.int8)
    base = oracle.solve(pds, z)
    cur = [b for b in base]
    active = np.array([b is not None for b in base])
    n_lp = np.ones(n)
    states = []
    swi = np.where(sw)[0]
    for step in range(K):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        jobs_pd, jobs_z, owner, line = [], [], [], []
        for i in idx:
            for l in swi:
                if z[i, l] == 0:
                    continue
                z2 = z[i].copy(); z2[l] = 0
                jobs_pd.append(pds[i]); jobs_z.append(z2); owner.append(i); line.append(l)
        res = oracle.solve(jobs_pd, jobs_z)
        owner, line = np.array(owner), np.array(line)
        for i in idx:
            mk = np.where(owner == i)[0]
            n_lp[i] += len(mk)
            vals = np.full(L, np.inf)
            for j in mk:
                if res[j] is not None:
                    vals[line[j]] = res[j][0] - cur[i][0]
            if record:
                rec = state_record(pds[i], z[i], cur[i])
                rec["v"] = vals
                rec["scen"] = i
                states.append(rec)
            best = int(np.argmin(vals))
            if vals[best] + switch_cost < -1e-9:
                z[i, best] = 0
                cur[i] = res[mk[line[mk] == best][0]]
            else:
                active[i] = False
    cost = np.array([c[0] if c is not None else np.inf for c in cur])
    return z, cost, n_lp, states


# ------------------------------------------------------------------------------- value model
class ValueGNN(torch.nn.Module):
    """Predicts, for every line, P(opening it is infeasible) and its cost change (in % of C(z))."""

    def __init__(self, case, node_in, edge_in, hidden=64, layers=6):
        super().__init__()
        self.gnn = EdgeGNN(case.f_bus, case.t_bus, case.n_bus, node_in, edge_in + 1, hidden, layers)
        self.head = mlp(3 * hidden, hidden, 2)

    def forward(self, x, e, z):
        h, g = self.gnn(x, torch.cat([e, z.float().unsqueeze(-1)], -1), edge_gate=z.float())
        f, t = self.gnn.f, self.gnn.t
        out = self.head(torch.cat([g, h[:, f] + h[:, t], (h[:, f] - h[:, t]).abs()], -1))
        return out[..., 0], out[..., 1]            # infeasibility logit, value (% of cost)


def _targets(st, sw):
    v = np.array([s["v"] for s in st])
    c = np.array([s["c"] for s in st])
    infeas = ~np.isfinite(v)
    pct = np.where(infeas, 0.0, v / c[:, None] * 100)
    mask = sw[None, :] & (np.array([s["z"] for s in st]) > 0)      # closed switchable lines
    return infeas, pct, mask


def stack_states(states):
    keys = ["pd", "z", "c", "lmp0", "mu0", "gamma0", "flow0", "va0", "pg0"]
    return {k: np.array([s[k] for s in states]) for k in keys}


def train_value(model, feat, states, val_states, sw, epochs=100, lr=1e-3, bs=64, seed=0, log_every=20):
    rng = np.random.default_rng(seed)
    d, dv = stack_states(states), stack_states(val_states)
    inf_t, pct_t, mask_t = (torch.as_tensor(a) for a in _targets(states, sw))
    inf_v, pct_v, mask_v = (torch.as_tensor(a) for a in _targets(val_states, sw))
    scale = float(pct_t[mask_t & ~inf_t].abs().std()) + 1e-6
    model.scale = scale
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = np.inf, None

    def loss_fn(dd, idx, inf_, pct_, mask_):
        x, e = feat(dd, idx)
        lg, val = model(x, e, torch.as_tensor(dd["z"][idx]))
        m = mask_[idx]
        l_inf = F.binary_cross_entropy_with_logits(lg[m], inf_[idx][m].float())
        mf = m & ~inf_[idx]
        l_val = F.smooth_l1_loss(val[mf], (pct_[idx][mf] / scale).float())
        return l_inf + l_val

    t0 = time.time()
    n = len(d["pd"])
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            loss = loss_fn(d, perm[i:i + bs], inf_t, pct_t, mask_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(dv, np.arange(len(dv["pd"])), inf_v, pct_v, mask_v).item()
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  [value] ep {ep:3d} train {loss.item():.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def learned_greedy(model, feat, oracle, pds, sw, K, switch_cost, R=2, p_inf_max=0.5):
    """Greedy OTS where a learned value model proposes R lines per step and the LP verifies them."""
    model.eval()
    n, L = len(pds), len(sw)
    z = np.ones((n, L), np.int8)
    cur = oracle.solve(pds, z)
    active = np.array([c is not None for c in cur])
    n_lp = np.ones(n)
    for step in range(K):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        st = stack_states([state_record(pds[i], z[i], cur[i]) for i in idx])
        x, e = feat(st)
        lg, val = model(x, e, torch.as_tensor(st["z"]))
        score = val.numpy() + 1e3 * (torch.sigmoid(lg).numpy() > p_inf_max)
        score = np.where(sw[None, :] & (st["z"] > 0), score, np.inf)
        jobs_pd, jobs_z, owner, line = [], [], [], []
        for r, i in enumerate(idx):
            for l in np.argsort(score[r])[:R]:
                if not np.isfinite(score[r, l]):
                    continue
                z2 = z[i].copy(); z2[l] = 0
                jobs_pd.append(pds[i]); jobs_z.append(z2); owner.append(i); line.append(l)
        res = oracle.solve(jobs_pd, jobs_z) if jobs_z else []
        owner, line = np.array(owner), np.array(line)
        for i in idx:
            mk = np.where(owner == i)[0]
            n_lp[i] += len(mk)
            best_j, best_c = None, cur[i][0] - switch_cost - 1e-9
            for j in mk:
                if res[j] is not None and res[j][0] < best_c:
                    best_j, best_c = j, res[j][0]
            if best_j is None:
                active[i] = False
            else:
                z[i, line[best_j]] = 0
                cur[i] = res[best_j]
            n_lp[i] += 0
    cost = np.array([c[0] if c is not None else np.inf for c in cur])
    return z, cost, n_lp


@torch.no_grad()
def learned_beam(model, feat, oracle, pds, sw, K, switch_cost, B=3, R=3, p_inf_max=0.5):
    """Beam search over switching sets: each of the B kept topologies proposes its R best lines
    (learned values), all children are LP-verified, the B cheapest survive. Returns the best
    topology ever seen (all-closed included), so it is never worse than DC-OPF."""
    model.eval()
    n, L = len(pds), len(sw)
    root = oracle.solve(pds, np.ones((n, L), np.int8))
    beams = [[(np.ones(L, np.int8), root[i])] if root[i] is not None else [] for i in range(n)]
    best = [(np.ones(L, np.int8), root[i]) for i in range(n)]
    n_lp = np.ones(n)
    tot = lambda z, s: s[0] + switch_cost * (1 - z).sum()
    for step in range(K):
        flat = [(i, z, s) for i in range(n) for z, s in beams[i]]
        if not flat:
            break
        st = stack_states([state_record(pds[i], z, s) for i, z, s in flat])
        x, e = feat(st)
        lg, val = model(x, e, torch.as_tensor(st["z"]))
        score = val.numpy() + 1e3 * (torch.sigmoid(lg).numpy() > p_inf_max)
        score = np.where(sw[None, :] & (st["z"] > 0), score, np.inf)
        jobs, meta = [], []
        seen = [set() for _ in range(n)]
        for r, (i, z, s) in enumerate(flat):
            for l in np.argsort(score[r])[:R]:
                if not np.isfinite(score[r, l]):
                    continue
                z2 = z.copy(); z2[l] = 0
                key = z2.tobytes()
                if key in seen[i]:
                    continue
                seen[i].add(key)
                jobs.append((pds[i], z2)); meta.append((i, z2))
        res = oracle.solve([j[0] for j in jobs], [j[1] for j in jobs]) if jobs else []
        children = [[] for _ in range(n)]
        for (i, z2), sres in zip(meta, res):
            n_lp[i] += 1
            if sres is not None:
                children[i].append((z2, sres))
        for i in range(n):
            children[i].sort(key=lambda c: tot(*c))
            beams[i] = children[i][:B]
            if beams[i] and best[i][1] is not None and tot(*beams[i][0]) < tot(*best[i]):
                best[i] = beams[i][0]
    z = np.array([b[0] for b in best])
    cost = np.array([b[1][0] if b[1] is not None else np.inf for b in best])
    return z, cost, n_lp
