"""Constrained (Lagrangian) policy-gradient fine-tuning of the commitment model (Model 1) for the
end-to-end mode (one dispatch LP at inference), and a simplified prediction-and-repair baseline in
the style of He et al. 2026 (IET GTD, doi 10.1049/gtd2.70405).

Plain REINFORCE (otsl.ucml.train_uc_reinforce) maximises -log(total cost / reference), where the
total cost of the dispatch LP contains VOLL-priced shedding / spill and priced reserve shortfall.
The penalty terms dominate the reward scale, so the operating-cost signal is drowned and the policy
buys feasibility with extra units. Here the dispatch result of every sampled commitment is split per
hour into

    operating cost  op_t  = no-load + energy + start-up cost of hour t          (objective)
    penalty         pen_t = VOLL * (shed_t + spill_t) + RES_SHORT * shortfall_t  (constraint)

(sum_t op_t + pen_t = LP objective), and Model 1 is trained on

    min E[sum_t op_t / ref]   s.t.   P(deployed commitment sheds or misses reserve) <= eps

with a Lagrange multiplier lambda updated by dual ascent on the violation rate of the *deployed*
(thresholded, min up/down repaired) commitment, which is evaluated with the LP at every step.
Gradient estimator:
  * per-hour credit assignment: decision (t, g) is credited with the cost of hours t-w .. t+w only;
  * leave-one-out baseline over the samples of an instance plus the deployed (greedy) commitment;
  * the two advantage streams are normalised separately, A = (A_op + lambda A_pen) / (1 + lambda)
    (as in PPO-Lagrangian, Ray et al. 2019), so the cost signal is not drowned by penalty variance;
  * optional MuProp-style control variate (Gu et al. 2016) from the exact LP sensitivities at the
    deployed commitment: the first-order Taylor term of the operating cost is removed from the
    sampled return and its expectation is differentiated analytically (sum_tg dop/du_tg * dp_tg).
    The analytic term is a dense per-unit signal that pushes down units whose no-load cost exceeds
    their dispatch value (the 'unnecessary units').
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

from .uc import RES_SHORT, VOLL, UCModel, UCScenario, load_rts_gmlc

# ----------------------------------------------------------------------------------- LP worker
_W = {}


def _init(cfg):
    s = load_rts_gmlc(line_scale=cfg.get("line_scale", 1.0))
    m = UCModel(s, T=cfg["T"], network=cfg.get("network", True))
    _W.update(s=s, m=m, cfg=cfg, mats=None)


def _mats(m, lo, hi):
    """Row split of the constraint matrix (the equality pattern is the same for every scenario)."""
    if _W.get("mats") is None:
        eq = lo == hi
        iu, il = ~eq & np.isfinite(hi), ~eq & np.isfinite(lo)
        _W["mats"] = (eq, iu, il, sp.vstack([m.A[iu], -m.A[il]]).tocsr(), m.A[eq].tocsr())
    return _W["mats"]


def hourly_dispatch(m: UCModel, sc: UCScenario, u, grad=False):
    """Fixed-commitment dispatch LP (identical to UCModel.solve_dispatch) returning the per-hour split
    of its objective into operating cost and penalties, and optionally the exact sensitivities
    d cost / d u (identical to UCModel.dispatch_gradient)."""
    s, T, G = m.s, m.T, m.dims["G"]
    S = m.dims["S"]
    u = np.asarray(u, float).reshape(T, G)
    lo, hi, lb, ub = m._rhs_bounds(sc, u)
    eq, iu, il, A_ub, A_eq = _mats(m, lo, hi)
    res = linprog(m.c, A_ub=A_ub, b_ub=np.r_[hi[iu], -lo[il]], A_eq=A_eq, b_eq=lo[eq],
                  bounds=np.c_[lb, ub], method="highs")
    if res.status != 0:
        raise RuntimeError(f"dispatch LP failed: {res.message}")
    x = res.x
    get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T].reshape(T, -1)
    uu, v = get("u"), get("v")
    ps = get("ps").reshape(T, G, S)
    op = uu @ s.c_nl + (ps * s.seg_c[None]).sum((1, 2)) + v @ s.c_su
    shed = get("shed").sum(1) + get("spill").sum(1)
    short = get("short")[:, 0]
    pen = VOLL * shed + RES_SHORT * short
    out = {"obj": float(res.fun), "op": op, "pen": pen, "shed": shed, "short": short}
    if grad:
        marg = res.lower.marginals + res.upper.marginals
        gm = lambda name: marg[m.off[name][0]:m.off[name][0] + m.off[name][1] * T].reshape(T, G)
        mu, mv, mw = gm("u"), gm("v"), gm("w")
        prev = np.vstack([sc.u0[None, :], u[:-1]])
        up, dn = (u > prev).astype(float), (u < prev).astype(float)
        g = mu + up * mv - dn * mw
        g[:-1] += -up[1:] * mv[1:] + dn[1:] * mw[1:]
        out["grad"] = g
    return out


def _job(args):
    load, avail, u0, sr, u, grad = args
    return hourly_dispatch(_W["m"], UCScenario(load=load, avail=avail, u0=u0, sr=sr), u, grad)


class HourlyOracle:
    """Dispatch LPs in a spawn pool, memoised by (instance key, commitment[, grad])."""

    def __init__(self, cfg, workers=2):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))
        self.cache, self.n = {}, 0

    def evaluate(self, d, idx, us, keys, grad=False):
        ck = [(int(k), np.asarray(u, np.int8).tobytes(), bool(grad)) for k, u in zip(keys, us)]
        todo = {}
        for j, c in enumerate(ck):
            if c not in self.cache and c not in todo:
                todo[c] = j
        if todo:
            js = list(todo.values())
            res = self.pool.map(_job, [(d["load"][idx[j]], d["avail"][idx[j]], d["u0"][idx[j]], d["sr"][idx[j]],
                                        np.asarray(us[j], np.int8), grad) for j in js],
                                chunksize=max(1, len(js) // 16))
            self.n += len(js)
            for j, r in zip(js, res):
                self.cache[ck[j]] = r
        rs = [self.cache[c] for c in ck]
        out = {k: np.array([r[k] for r in rs]) for k in ("obj", "op", "pen", "shed", "short")}
        if grad:
            out["grad"] = np.array([r["grad"] for r in rs])
        return out

    def scores(self, d, idx, us, keys):
        """(cost, shed, short) totals, the interface of ucml.DispatchOracle."""
        o = self.evaluate(d, idx, us, keys)
        return o["obj"], o["shed"].sum(1), o["short"].sum(1)

    def close(self):
        self.pool.close()


# ----------------------------------------------------------------------------------- training
def _window(x, w):
    """x [..., T] -> sum over hours t-w .. t+w (w None: whole horizon) for every t."""
    T = x.shape[-1]
    if w is None or w >= T:
        return np.repeat(x.sum(-1, keepdims=True), T, -1)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, -1)], -1)
    hi = np.minimum(np.arange(T) + w + 1, T)
    lo = np.maximum(np.arange(T) - w, 0)
    return c[..., hi] - c[..., lo]


def train_constrained(m1, tr, oracle, ref, repair, steps=120, bs=16, n_samples=5, lr=3e-4, seed=0,
                      eps=0.1, lam0=1.0, lam_lr=0.3, window=1, kappa=1e-3, tau=1.0, muprop=False,
                      beta=1.0, log_every=10, key_offset=0, val_fn=None, val_every=0, lam_max=1e3, kl=0.0):
    """Lagrangian policy gradient (see module docstring).

    ref: [N] per-instance cost scale (LP relaxation cost for the label-free variant: no MILP needed).
    repair(u, i): schedule repair applied to every sampled / deployed commitment of training instance i
    (min up/down repair, optionally preceded by the block adequacy repair) - part of the environment.
    eps: target violation rate of the deployed commitment.  lam: multiplier of the penalty stream,
    log-space dual ascent  log lam += lam_lr * (violation rate of deployed commitments - eps).
    window: per-hour credit window w (None = whole horizon, i.e. no credit assignment).
    kappa: penalty stream c_t = log(1 + pen_t / (kappa * ref)) (graded, scale-free).
    tau: sampling temperature.  muprop: Taylor control variate from LP sensitivities, weight beta on
    the analytic term (beta = 1 is the unbiased estimator).
    lam_max: cap on lambda (limits the overshoot after the first, violation-heavy steps).
    kl: weight of a KL anchor sum_tg KL(Bern(p_tg) || Bern(p0_tg)) to the initial (imitation) policy p0:
    the imitation policy is cheap whenever it serves, so the fine-tune should move only the decisions
    the LP critic asks for (as in KL-regularised policy optimisation).
    Every step evaluates bs * (n_samples + 1) LPs (samples + the deployed commitment)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m1.net.parameters(), lr=lr)
    n = len(tr["load"])
    S = n_samples
    lam = float(lam0)
    hist, t0 = [], time.time()
    if kl > 0:
        import copy
        net0 = copy.deepcopy(m1.net).eval()
        m0 = type(m1)(net0, m1.feat, m1.kind)
    for step in range(steps):
        m1.net.train()
        idx = rng.choice(n, bs, replace=False)
        lg = m1.logits(tr, idx)                                            # [B, T, G]
        if kl > 0:
            with torch.no_grad():
                lg0 = m0.logits(tr, idx)
        B, T, G = lg.shape
        lt = lg / tau
        p = torch.sigmoid(lt.detach())
        smp = torch.rand((S,) + p.shape, generator=gen) < p                # [S, B, T, G]
        logp = smp * F.logsigmoid(lt) + (~smp) * F.logsigmoid(-lt)        # [S, B, T, G]
        raw = smp.numpy().astype(np.int8)
        greedy = (lg.detach() > 0).numpy().astype(np.int8)
        us = np.array([[repair(raw[k, b], idx[b]) for b in range(B)] for k in range(S)])
        ug = np.array([repair(greedy[b], idx[b]) for b in range(B)])
        flat_idx = np.tile(idx, S)
        o = oracle.evaluate(tr, flat_idx, us.reshape((-1, T, G)), flat_idx + key_offset)
        og = oracle.evaluate(tr, idx, ug, idx + key_offset, grad=muprop)
        r = ref[idx]
        op = o["op"].reshape(S, B, T) / r[None, :, None]
        pen = o["pen"].reshape(S, B, T)
        cpen = np.log1p(pen / (kappa * r[None, :, None]))
        op_g = og["op"] / r[:, None]
        cpen_g = np.log1p(og["pen"] / (kappa * r[:, None]))
        clean_g = og["pen"] < 1e-6 * r[:, None]                             # [B, T] hours where deployed u is clean
        if muprop:
            gr = og["grad"] / r[:, None, None] * clean_g[..., None]         # d op_t / d u_tg at clean hours
            # linear term in the raw (independent Bernoulli) sample, so that its expectation is sum gr * (p - ug)
            lin = (gr[None] * (raw.astype(float) - ug[None].astype(float))).sum(-1)  # [S, B, T]
            op_res = op - lin
        else:
            op_res = op
        R_op, R_c = _window(op_res, window), _window(cpen, window)          # [S, B, T]
        Rg_op, Rg_c = _window(op_g, window), _window(cpen_g, window)        # [B, T]
        b_op = (R_op.sum(0, keepdims=True) - R_op + Rg_op[None]) / S
        b_c = (R_c.sum(0, keepdims=True) - R_c + Rg_c[None]) / S
        A_op, A_c = -(R_op - b_op), -(R_c - b_c)
        s_op = A_op.std() + 1e-8
        s_c = A_c.std() + 1e-8 if A_c.std() > 1e-6 else 1.0
        A = (A_op / s_op + lam * A_c / s_c) / (1 + lam)
        A_t = torch.as_tensor(A, dtype=torch.float32)
        loss = -(A_t * logp.sum(-1)).sum(-1).mean()
        if muprop and beta > 0:
            # analytic gradient of E[sum_tg dop/du (u - ug)] = sum_tg dop/du * p_tg (descend on cost)
            gr_t = torch.as_tensor(gr / s_op / (1 + lam), dtype=torch.float32)
            loss = loss + beta * (gr_t * torch.sigmoid(lt)).sum((1, 2)).mean()
        if kl > 0:
            pk = torch.sigmoid(lg)
            klv = pk * (F.logsigmoid(lg) - F.logsigmoid(lg0)) + (1 - pk) * (F.logsigmoid(-lg) - F.logsigmoid(-lg0))
            loss = loss + kl * klv.sum((1, 2)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.net.parameters(), 1.0)
        opt.step()
        viol_g = (og["shed"].sum(1) > 1e-6) | (og["short"].sum(1) > 1e-6)
        lam = float(np.clip(lam * np.exp(lam_lr * (viol_g.mean() - eps)), 1e-3, lam_max))
        viol_s = (o["shed"].sum(1) > 1e-6) | (o["short"].sum(1) > 1e-6)
        hist.append({"step": step, "lam": lam, "viol_greedy": float(viol_g.mean()), "viol_sample": float(viol_s.mean()),
                     "op_gap_greedy_%": float(np.mean(og["op"].sum(1) / r - 1) * 100),
                     "cost_gap_greedy_%": float(np.mean(og["obj"] / r - 1) * 100),
                     "op_gap_sample_%": float(np.mean(o["op"].sum(1).reshape(S, B) / r[None] - 1) * 100),
                     "units_on_sample": float(us.sum((2, 3)).mean() / T),
                     "units_on_greedy": float(ug.sum((1, 2)).mean() / T), "LPs": oracle.n, "s": time.time() - t0})
        if log_every and (step % log_every == 0 or step == steps - 1):
            h = hist[-log_every:]
            print(f"  [lag-pg] step {step:4d} lam {lam:7.3f}  viol greedy {np.mean([x['viol_greedy'] for x in h]):.3f} "
                  f"sample {np.mean([x['viol_sample'] for x in h]):.3f}  op gap greedy {np.mean([x['op_gap_greedy_%'] for x in h]):6.2f}% "
                  f"units {np.mean([x['units_on_greedy'] for x in h]):5.2f} (samples {np.mean([x['op_gap_sample_%'] for x in h]):6.2f}%, "
                  f"{np.mean([x['units_on_sample'] for x in h]):5.2f})  LPs {oracle.n} ({time.time() - t0:.0f}s)", flush=True)
        if val_fn is not None and val_every and (step + 1) % val_every == 0:
            hist[-1]["val"] = val_fn(m1)
    return m1, hist


def train_reinforce_plain(m1, tr, oracle, ref, repair, steps=120, bs=16, n_samples=6, lr=3e-4, seed=0,
                          log_every=10, key_offset=0, val_fn=None, val_every=0):
    """Plain REINFORCE of otsl.ucml.train_uc_reinforce (reward -log(total LP cost / ref), RLOO baseline,
    normalised advantages, same sampling and seeds) with an instance-aware repair(u, i) in the loop."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m1.net.parameters(), lr=lr)
    n = len(tr["load"])
    hist, t0 = [], time.time()
    for step in range(steps):
        m1.net.train()
        idx = rng.choice(n, bs, replace=False)
        lg = m1.logits(tr, idx)
        p = torch.sigmoid(lg.detach())
        smp = (torch.rand((n_samples,) + p.shape, generator=gen) < p)
        logp = (smp * F.logsigmoid(lg) + (~smp) * F.logsigmoid(-lg)).sum((-1, -2))
        raw = smp.numpy().astype(np.int8)
        us = np.array([[repair(raw[k, b], idx[b]) for b in range(bs)] for k in range(n_samples)])
        flat_idx = np.tile(idx, n_samples)
        o = oracle.evaluate(tr, flat_idx, us.reshape((-1,) + us.shape[2:]), flat_idx + key_offset)
        c = o["obj"].reshape(n_samples, bs)
        r = torch.as_tensor(-np.log(c / ref[idx][None]), dtype=torch.float32)
        adv = r - (r.sum(0, keepdim=True) - r) / (n_samples - 1)
        adv = adv / (adv.std() + 1e-8)
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.net.parameters(), 1.0)
        opt.step()
        viol = (o["shed"].sum(1) > 1e-6) | (o["short"].sum(1) > 1e-6)
        hist.append({"step": step, "gap_sample_%": float(np.exp(-r.mean()) * 100 - 100), "viol_sample": float(viol.mean()),
                     "units_on_sample": float(us.sum((2, 3)).mean() / us.shape[2]), "LPs": oracle.n, "s": time.time() - t0})
        if log_every and (step % log_every == 0 or step == steps - 1):
            h = hist[-log_every:]
            print(f"  [rl-repair] step {step:4d} sampled gap {np.mean([x['gap_sample_%'] for x in h]):7.2f}%  viol "
                  f"{np.mean([x['viol_sample'] for x in h]):.3f}  units {np.mean([x['units_on_sample'] for x in h]):5.2f}  "
                  f"LPs {oracle.n} ({time.time() - t0:.0f}s)", flush=True)
        if val_fn is not None and val_every and (step + 1) % val_every == 0:
            hist[-1]["val"] = val_fn(m1)
    return m1, hist


# ----------------------------------------------------------------------------------- He et al.-style repair
class RepairPolicy(nn.Module):
    """Per-decision policy over the repairable (unit, hour) decisions: features of the decision, the
    unit and the hour (solver-free context computed from the predictor's schedule) -> logit of
    'unit on'. Shared weights across decisions (permutation-equivariant over the repair set)."""

    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # start from the predictor: the last layer gets a skip from the predictor's logit (feature 0)
        self.skip = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):                                                   # x [..., d]
        return self.net(x).squeeze(-1) + self.skip * x[..., 0]


class RepairEnv:
    """Builds the repair set and the policy state for a batch of instances.

    Reliable / repairable split (He et al.: root-relaxation history + prediction confidence):
    uncertainty score = (1 - |2p - 1|) + h_tg + 1[relaxation fractional in this instance], where h_tg is
    the frequency with which the root LP relaxation of the training instances is fractional for unit g
    at hour t of the horizon (historical root-relaxation records). The k highest-scoring decisions are
    repairable; the rest keep the behaviour-cloning prediction."""

    def __init__(self, sysm, train, k=48):
        self.s, self.k = sysm, k
        fr = (train["u_rel"] > 0.01) & (train["u_rel"] < 0.99)
        self.hist = fr.mean(0)                                              # [T, G]
        avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
        onehot = np.array([[t == k_ for k_ in ["CT", "CC", "STEAM", "NUCLEAR"]] for t in sysm.utype], float)
        self.g_static = np.c_[sysm.pmax, sysm.pmin, avg / 1e4, sysm.c_su / 1e5, sysm.min_up / 24, sysm.min_dn / 24, onehot]

    def build(self, d, idx, p):
        """p: [B, T, G] predictor probabilities. Returns repair-set indices [B, k] (flattened t*G+g),
        features [B, k, d], base schedule [B, T, G]."""
        s = self.s
        B, T, G = p.shape
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        urel = d["u_rel"][idx]
        frac = (urel > 0.01) & (urel < 0.99)
        score = (1 - np.abs(2 * p - 1)) + self.hist[None] + frac
        sel = np.argsort(-score.reshape(B, -1), 1)[:, :self.k]               # [B, k]
        base = (p > 0.5).astype(np.int8)
        load, avail, sr = d["load"][idx], d["avail"][idx], d["sr"][idx]
        net = load.sum(2) - avail.sum(2)                                    # [B, T]
        cap = (base * s.pmax).sum(2)
        margin = (cap - net - sr) / load.sum(2)                             # capacity margin of the BC schedule
        floor = ((base * s.pmin).sum(2) - load.sum(2)) / load.sum(2)
        lmp = d["lmp_rel"][idx][:, :, s.gbus] / 1e4                          # [B, T, G]
        tt, gg = sel // G, sel % G
        bi = np.arange(B)[:, None]
        f = [logit[bi, tt, gg], p[bi, tt, gg], urel[bi, tt, gg], self.hist[tt, gg], frac[bi, tt, gg].astype(float),
             margin[bi, tt], floor[bi, tt], lmp[bi, tt, gg], tt / T, d["u0"][idx][bi, gg].astype(float),
             score.reshape(B, -1)[bi, sel]]
        x = np.concatenate([np.stack(f, -1), self.g_static[gg]], -1)
        return sel, x.astype(np.float32), base

    @staticmethod
    def apply(base, sel, a):
        u = base.copy()
        B, T, G = u.shape
        flat = u.reshape(B, -1)
        flat[np.arange(B)[:, None], sel] = a
        return flat.reshape(B, T, G)


def train_repair_ppo(pol, env, p_bc, tr, oracle, ref, repair, steps=120, bs=16, n_samples=6, lr=1e-3,
                     epochs=4, clip=0.2, seed=0, log_every=10, key_offset=0):
    """PPO-clip on a one-step repair episode (action: on/off of every repairable decision; reward
    -log(dispatch LP cost / ref), the LP cost contains the shedding / reserve penalties, i.e. operating
    cost and feasibility feedback). Advantage: leave-one-out baseline over the samples of an instance."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    n = len(tr["load"])
    S = n_samples
    hist, t0 = [], time.time()
    for step in range(steps):
        idx = rng.choice(n, bs, replace=False)
        sel, x, base = env.build(tr, idx, p_bc[idx])
        xt = torch.as_tensor(x)
        with torch.no_grad():
            lg_old = pol(xt)                                                # [B, k]
            pr = torch.sigmoid(lg_old)
            a = (torch.rand((S,) + pr.shape, generator=gen) < pr)           # [S, B, k]
            lp_old = (a * F.logsigmoid(lg_old) + (~a) * F.logsigmoid(-lg_old)).sum(-1)
        an = a.numpy().astype(np.int8)
        us = np.array([[repair(RepairEnv.apply(base[b:b + 1], sel[b:b + 1], an[k, b:b + 1])[0], tr["u0"][idx[b]])
                        for b in range(bs)] for k in range(S)])
        flat_idx = np.tile(idx, S)
        o = oracle.evaluate(tr, flat_idx, us.reshape((-1,) + us.shape[2:]), flat_idx + key_offset)
        rew = -np.log(o["obj"].reshape(S, bs) / ref[idx][None])
        adv = rew - (rew.sum(0, keepdims=True) - rew) / (S - 1)
        adv = torch.as_tensor(adv / (adv.std() + 1e-8), dtype=torch.float32)
        for _ in range(epochs):
            lg = pol(xt)
            lp = (a * F.logsigmoid(lg) + (~a) * F.logsigmoid(-lg)).sum(-1)  # [S, B]
            ratio = torch.exp(lp - lp_old)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
        viol = (o["shed"].sum(1) > 1e-6) | (o["short"].sum(1) > 1e-6)
        hist.append({"step": step, "gap_%": float(np.mean(np.exp(-rew)) * 100 - 100), "viol_sample": float(viol.mean()),
                     "LPs": oracle.n, "s": time.time() - t0})
        if log_every and (step % log_every == 0 or step == steps - 1):
            h = hist[-log_every:]
            print(f"  [repair-ppo] step {step:4d} sampled gap {np.mean([x['gap_%'] for x in h]):8.2f}%  "
                  f"viol {np.mean([x['viol_sample'] for x in h]):.3f}  LPs {oracle.n} ({time.time() - t0:.0f}s)", flush=True)
    return pol, hist


@torch.no_grad()
def repair_probs(pol, env, d, p_bc, bs=64):
    """Composite probabilities: predictor outside the repair set, repair policy inside it."""
    out = p_bc.copy()
    n = len(p_bc)
    for i in range(0, n, bs):
        idx = np.arange(i, min(i + bs, n))
        sel, x, _ = env.build(d, idx, p_bc[idx])
        pr = torch.sigmoid(pol(torch.as_tensor(x))).numpy()
        B, T, G = p_bc[idx].shape
        flat = out[idx].reshape(B, -1)
        flat[np.arange(B)[:, None], sel] = pr
        out[idx] = flat.reshape(B, T, G)
    return out


# ----------------------------------------------------------------------------------- solver-free repair
def adequacy_repair_blocks(u, u0, load, avail, sr, sysm, margin=0.0):
    """Min up/down-aware adequacy repair of a schedule u [T, G] (solver-free).

    otsl.uc.adequacy_repair switches units on for single hours; the min up/down repair that follows
    (Hamming-nearest) often reverts such one-hour additions of units with long minimum up times, so
    the capacity deficit comes back. Here, for every hour t (in time order) whose committed capacity
    is below net load + reserve (+ margin x load), switch on the off unit with the lowest cost per MW
    of the block it needs: cost = start-up (if it was off before t) + no-load x block length, block =
    hours t .. t + min_up - 1 (or bridging back to its last on-hour if that is within its minimum down
    time, which avoids a start-up). Units are added until the deficit is covered (nothing is
    switched off). Apply repair_min_updown afterwards (it leaves feasible schedules unchanged)."""
    u = np.array(u, dtype=np.int8, copy=True)
    T, G = u.shape
    s = sysm
    avg = (s.c_nl + (s.seg_c * s.seg_w).sum(1)) / s.pmax
    for t in range(T):
        need = load[t].sum() - avail[t].sum() + sr[t] + margin * load[t].sum()
        cap = float((u[t] * s.pmax).sum())
        if cap >= need:
            continue
        best = []
        for g in np.where(u[t] == 0)[0]:
            prev_on = np.where(u[:t, g] == 1)[0]
            last = prev_on[-1] if len(prev_on) else (-1 if u0[g] else None)
            if last is not None and t - last - 1 < s.min_dn[g]:
                lo, su = last + 1, 0.0               # bridge the short off-gap: no start-up
            else:
                lo, su = t, s.c_su[g]
            hi = min(T, t + int(s.min_up[g]))
            L = hi - lo
            score = (su + s.c_nl[g] * L) / (s.pmax[g] * max(1, hi - t)) + avg[g]
            best.append((score, g, lo, hi))
        for score, g, lo, hi in sorted(best):
            if cap >= need:
                break
            u[lo:hi, g] = 1
            cap += s.pmax[g]
    return u
