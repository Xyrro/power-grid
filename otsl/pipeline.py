"""Inference-time pipelines that turn predictions into topologies, plus evaluation metrics.

Every pipeline returns, per scenario, a set of candidate topologies; the exact fixed-topology
LP ("LP Solver" box of the framework) scores each one and the cheapest feasible candidate is
kept.  Including the all-closed topology as a candidate guarantees a feasible answer that is
never worse than plain DC-OPF.
"""
from __future__ import annotations

import time
import multiprocessing as mp

import numpy as np

from .data import _init, make_model


def _lp_cost(args):
    from .data import _W
    pd, z = args
    s = _W["m"].solve_lp(pd, z)
    return s.obj if s.ok else np.inf


class LPOracle:
    """Parallel, memoised fixed-topology DC-OPF evaluations."""

    def __init__(self, cfg, workers=4):
        self.cfg = cfg
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))
        self.cache = {}
        self.n_solves = 0

    def costs(self, pds, zs, keys=None):
        """pds [M,N], zs [M,L] -> cost [M] (inf if infeasible). keys: scenario ids for caching."""
        keys = range(len(pds)) if keys is None else keys
        ck = [(k, np.asarray(z, np.int8).tobytes()) for k, z in zip(keys, zs)]
        todo = [i for i, c in enumerate(ck) if c not in self.cache]
        # de-duplicate within the request
        uniq = {}
        for i in todo:
            uniq.setdefault(ck[i], i)
        idx = list(uniq.values())
        if idx:
            res = self.pool.map(_lp_cost, [(pds[i], zs[i]) for i in idx], chunksize=max(1, len(idx) // 32))
            self.n_solves += len(idx)
            for i, r in zip(idx, res):
                self.cache[ck[i]] = r
        return np.array([self.cache[c] for c in ck])

    def close(self):
        self.pool.close()


# ------------------------------------------------------------------------------- candidates
def decode_threshold(p_open, K, switchable, thr=0.5):
    """Open the (at most K) switchable lines with p_open > thr, highest first."""
    z = np.ones(p_open.shape, np.int8)
    p = np.where(switchable, p_open, -1)
    for i in range(len(p)):
        order = np.argsort(-p[i])[:K]
        z[i, order[p[i, order] > thr]] = 0
    return z


def candidates_from_probs(p_open, K, switchable, n_samples=16, top_m=None, rng=None):
    """Candidate topologies per scenario from Model-1 probabilities:
    all-closed (fallback), threshold decode, every subset of <= K lines among the top_m most likely
    lines (local enumeration), and budget-truncated Bernoulli samples."""
    from itertools import combinations
    rng = rng or np.random.default_rng(0)
    B, L = p_open.shape
    out = []
    thr = decode_threshold(p_open, K, switchable)
    for i in range(B):
        p = np.where(switchable, p_open[i], 0.0)
        order = np.argsort(-p)
        cands = [np.ones(L, np.int8), thr[i]]
        top = order[:top_m or (K + 2)]
        for k in range(1, K + 1):
            for sub in combinations(top, k):
                z = np.ones(L, np.int8)
                z[list(sub)] = 0
                cands.append(z)
        for _ in range(n_samples):
            u = rng.random(L) < p
            if u.sum() > K:  # keep the K most likely of the sampled lines
                on = np.where(u)[0]
                u[:] = False
                u[on[np.argsort(-p[on])[:K]]] = True
            z = np.ones(L, np.int8)
            z[u] = 0
            cands.append(z)
        out.append(np.unique(np.array(cands), axis=0))
    return out


def pick_best(oracle, pds, cand_lists, keys, switch_cost=0.0):
    """Score all candidates with the LP (+ switching cost) and return (best z, best generation cost,
    #candidates) per scenario."""
    flat_pd, flat_z, flat_k, owner = [], [], [], []
    for i, cl in enumerate(cand_lists):
        for z in cl:
            flat_pd.append(pds[i]); flat_z.append(z); flat_k.append(keys[i]); owner.append(i)
    c = oracle.costs(np.array(flat_pd), np.array(flat_z), flat_k)
    tot = c + switch_cost * (1 - np.array(flat_z)).sum(1)
    owner = np.array(owner)
    best_z, best_c = [], []
    for i in range(len(cand_lists)):
        m = owner == i
        j = np.argmin(tot[m])
        best_z.append(np.array(flat_z)[m][j]); best_c.append(c[m][j])
    return np.array(best_z), np.array(best_c), np.array([len(cl) for cl in cand_lists])


# ------------------------------------------------------------------------------- heuristics
def dual_greedy(cfg, pd, K, R=5):
    """Non-learning baseline: repeatedly open the line with the best LP-verified saving among the R
    lines with the most negative first-order estimate dC_l = -gamma_l f_l (Fuller et al. 2012)."""
    m = make_model(cfg)
    sw = ~m.fixed_closed
    z = np.ones(m.case.n_line, np.int8)
    s = m.solve_lp(pd, z)
    n_lp = 1
    for _ in range(K):
        est = np.where(sw & (z > 0), -s.gamma * s.flow, np.inf)
        cand = np.argsort(est)[:R]
        best = None
        for l in cand:
            if not np.isfinite(est[l]):
                continue
            z2 = z.copy(); z2[l] = 0
            s2 = m.solve_lp(pd, z2); n_lp += 1
            if s2.ok and s2.obj < (best[1].obj if best else s.obj - 1e-6):
                best = (z2, s2)
        if best is None:
            break
        z, s = best
    return z, s.obj, n_lp


def _dg(args):
    cfg, pd, K, R = args
    return dual_greedy(cfg, pd, K, R)


def run_dual_greedy(cfg, pds, K, R=5, workers=4):
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_dg, [(cfg, pd, K, R) for pd in pds])
    return np.array([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[2] for r in res])


def knn_candidates(train_pd, train_z, test_pd, k=10):
    """Johnson et al. (2020): topologies of the k nearest training scenarios."""
    mu, sd = train_pd.mean(0), train_pd.std(0) + 1e-9
    a, b = (train_pd - mu) / sd, (test_pd - mu) / sd
    d = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * b @ a.T
    nn_idx = np.argsort(d, 1)[:, :k]
    out = []
    for i in range(len(test_pd)):
        c = np.concatenate([np.ones((1, train_z.shape[1]), np.int8), train_z[nn_idx[i]]])
        out.append(np.unique(c, axis=0))
    return out


# ------------------------------------------------------------------------------- metrics
def metrics(cost, test, z=None, label="", **extra):
    """cost: generation cost of the chosen topology z (inf if infeasible). All comparisons use the
    MILP objective  generation cost + switch_cost * #open  (switch_cost = 0 for raw DC-OTS)."""
    sc = float(test.get("switch_cost", 0.0))
    c0 = test["c0"]
    cs = test["c_ots"] + sc * (1 - test["z"]).sum(1)
    feas = np.isfinite(cost)
    n_open = np.zeros(len(cost)) if z is None else (1 - z).sum(1)
    cost_f = np.where(feas, cost + sc * n_open, c0)          # infeasible -> fall back to all-closed
    gap = (cost_f - cs) / cs * 100
    ben = c0 - cs
    m = ben > 1e-6 * c0
    cap = (c0[m] - cost_f[m]) / ben[m] * 100
    out = {"method": label, "feasible_%": feas.mean() * 100, "gap_mean_%": gap.mean(),
           "gap_p95_%": np.percentile(gap, 95), "gap_max_%": gap.max(),
           "benefit_captured_%": cap.mean() if m.any() else np.nan,
           "beats_or_ties_milp_%": (gap <= 1e-4).mean() * 100}
    if z is not None:
        out["z_exact_match_%"] = (z == test["z"]).all(1).mean() * 100
        out["z_hamming"] = (z != test["z"]).sum(1).mean()
        out["n_open"] = n_open.mean()
    out.update(extra)
    return out
