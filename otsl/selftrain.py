"""Self-training for unit commitment with the solver as teacher (expert iteration without MILP labels).

Round 0 imitates the repaired LP relaxation (one LP per label). Each later round

    predict on training instances -> fix the model's most confident decisions on which it agrees with the
    current label (the current label stays feasible, as in RINS) -> reduced MILP with a short limit ->
    score the schedule with the exact dispatch LP -> replace the label if it is cheaper -> retrain.

No full MILP is solved for training. Worker pools use the spawn start method (HiGHS stalls after fork).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn.functional as F

from .uc import UCModel, UCScenario, load_rts_gmlc

_W = {}


def _init(cfg):
    s = load_rts_gmlc(line_scale=cfg.get("line_scale", 1.0))
    _W.update(s=s, m=UCModel(s, T=cfg["T"], network=cfg.get("network", True)), cfg=cfg)


def _as_fix(arr):
    return {(int(t), int(g)): int(v) for t, g, v in arr}


def _reduced(job):
    """label generation: reduced MILP around the current label, then the exact dispatch LP of its schedule"""
    i, load, avail, u0, sr, fix_arr, time_limit, gap = job
    m = _W["m"]
    sc = UCScenario(load=load, avail=avail, u0=u0, sr=sr)
    sol = m.solve_uc(sc, time_limit=time_limit, mip_gap=gap, z_fix=_as_fix(fix_arr) or None)
    if sol.u is None:
        return dict(i=i, u=None, cost=np.inf, shed=np.inf, short=np.inf, milp_s=sol.time, lp_s=0.0, status=sol.status)
    lp = m.solve_dispatch(sc, sol.u)
    return dict(i=i, u=sol.u, cost=lp.obj, milp_obj=sol.obj, shed=lp.shed, short=lp.short, milp_s=sol.time,
                lp_s=lp.time, status=sol.status, mip_gap=sol.gap)


def _fixeval(job):
    """test-time evaluation: full MILP and every reduced MILP of one instance, back to back in one process"""
    i, load, avail, u0, sr, specs, time_limit, gap = job
    m = _W["m"]
    sc = UCScenario(load=load, avail=avail, u0=u0, sr=sr)
    out = {}
    for name, fix_arr in specs:
        sol = m.solve_uc(sc, time_limit=time_limit, mip_gap=gap, z_fix=_as_fix(fix_arr) or None)
        t, fb = sol.time, 0
        if sol.u is None:                      # fixings conflict with min up/down: solve without them
            fb = 1
            sol = m.solve_uc(sc, time_limit=time_limit, mip_gap=gap)
            t += sol.time
        out[name] = dict(obj=sol.obj, shed=sol.shed, short=sol.short, time=t, fallback=fb, status=sol.status,
                         u=sol.u, n_fixed=len(fix_arr))
    return i, out


class SolverPool:
    def __init__(self, cfg, workers=2):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))

    def run(self, fn, jobs, log_every=0, tag=""):
        res, t0 = [], time.time()
        for k, r in enumerate(self.pool.imap_unordered(fn, jobs, chunksize=1)):
            res.append(r)
            if log_every and (k + 1) % log_every == 0:
                print(f"  [{tag}] {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        return res

    def close(self):
        self.pool.close()
        self.pool.join()


def fix_array(fix):
    return np.array([(t, g, v) for (t, g), v in fix.items()], dtype=np.int64).reshape(-1, 3)


def label_fixings(p, label, ratio, agree=True):
    """the ratio*T*G decisions with the largest confidence |p - 0.5|; with agree=True only decisions on which the
    rounded prediction equals the current label (so the label is a feasible point of the reduced MILP), fixed
    to the label's value"""
    err = np.minimum(p, 1 - p)
    pred = (p > 0.5).astype(np.int8)
    if agree:
        err = np.where(pred == label, err, np.inf)
    flat = np.argsort(err.reshape(-1), kind="stable")[:int(ratio * p.size)]
    flat = flat[np.isfinite(err.reshape(-1)[flat])]
    tt, gg = np.unravel_index(flat, p.shape)
    src = label if agree else pred
    return {(int(t), int(g)): int(src[t, g]) for t, g in zip(tt, gg)}


def train_bce_fixed(m1, tr, targets, epochs=80, lr=1e-3, bs=64, seed=0, log_every=20):
    """BCE on a label pool for a fixed number of epochs (cosine schedule, final weights): no validation labels are
    needed, so the validation set stays free of any solver labels."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(targets, dtype=torch.float32)
    opt = torch.optim.AdamW(m1.net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n, t0 = len(y), time.time()
    for ep in range(epochs):
        m1.net.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(m1.logits(tr, idx), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  [st-bce] ep {ep:3d} train {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    m1.net.eval()
    return m1


def train_reinforce_sil(m1, tr, oracle, labels, label_cost, steps=120, bs=16, n_samples=6, lr=3e-4, seed=0,
                        log_every=25, repair=None):
    """REINFORCE with the exact dispatch LP as critic (RLOO baseline), with the solver label of each instance added
    to its group of samples (self-imitation): the label's log-likelihood is pushed up with its leave-one-out
    advantage clipped at zero, i.e. only when it beats the policy's own samples. Same units as the policy-gradient
    term, so no weight has to be tuned. labels: [N, T, G]; label_cost: [N] dispatch-LP cost of each label."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m1.net.parameters(), lr=lr)
    n = len(tr["load"])
    hist, t0 = [], time.time()
    yl = torch.as_tensor(labels, dtype=torch.float32)
    for step in range(steps):
        m1.net.train()
        idx = rng.choice(n, bs, replace=False)
        lg = m1.logits(tr, idx)                                            # [B, T, G]
        p = torch.sigmoid(lg.detach())
        smp = (torch.rand((n_samples,) + p.shape, generator=gen) < p)      # [S, B, T, G]
        logp = (smp * F.logsigmoid(lg) + (~smp) * F.logsigmoid(-lg)).sum((-1, -2))
        y = yl[idx]
        logp_lab = (y * F.logsigmoid(lg) + (1 - y) * F.logsigmoid(-lg)).sum((-1, -2))   # [B]
        us = smp.numpy().astype(np.int8)
        if repair is not None:
            us = np.array([[repair(us[k, b], tr["u0"][idx[b]]) for b in range(bs)] for k in range(n_samples)])
        flat_idx = np.tile(idx, n_samples)
        c, _, _ = oracle.evaluate(tr, flat_idx, us.reshape((-1,) + us.shape[2:]), flat_idx)
        c = np.vstack([c.reshape(n_samples, bs), label_cost[idx][None]])  # [S+1, B]
        r = torch.as_tensor(-np.log(c / tr["obj"][idx][None]), dtype=torch.float32)
        adv = r - (r.sum(0, keepdim=True) - r) / n_samples
        sd = adv.std() + 1e-8
        adv = adv / sd
        loss = -(adv[:-1] * logp).sum(0).mean() / n_samples - (adv[-1].clamp_min(0) * logp_lab).mean() / n_samples
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.net.parameters(), 1.0)
        opt.step()
        hist.append(float(np.exp(-r[:-1].mean()) - 1))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  [st-sil] step {step:4d} mean sampled gap to LP bound {np.mean(hist[-log_every:]) * 100:.3f}%  "
                  f"label better than samples {float((adv[-1] > 0).float().mean()) * 100:.0f}%  LPs {oracle.n} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return m1, hist


def clone(m1):
    return copy.deepcopy(m1)
