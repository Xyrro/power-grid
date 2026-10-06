"""Learning and fixing on PGLib-UC: adapters from the RTS-GMLC methods to otsl/pglib.py.

* PGFeaturizer / UnitNet   Model 1 without a network graph: a per-unit network with shared weights and a learned
                           unit embedding; input = unit data, the system's hourly context (demand, net load, reserve,
                           wind, relaxed prices) and the unit's LP-relaxation commitment around each hour. It plugs
                           into otsl.ucml.UCModel1 (kind "gnn": net(*feat(d, idx))), so train_uc_bce,
                           train_uc_reinforce, otsl.selftrain.train_bce_fixed work unchanged.
* FastDispatch             fixed-commitment dispatch LP kept loaded in one highspy object; a new commitment only
                           changes column bounds, so HiGHS warm-starts from the previous basis.
* DispatchOracle           spawn pool of FastDispatch workers with the evaluate() interface of
                           otsl.ucml.DispatchOracle (memoised).
* table2_features / KNNProb  Learning to Fix's kNN (eq. 13, k = 50) with the Table II features of a copper-plate
                           system with one wind profile.
* PGLtFTuner               otsl.ltfx.LtFTuner on PGLib scenarios (only the scenario constructor differs).
* lp_guard                 otsl.fixpolicy.lp_guard with the relaxed reduced LP solved through highspy.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import models as _models  # noqa: F401  (keeps torch single-threaded)
from .ltfx import LtFTuner
from .pglib import PGModel, PGScenario, load_bases, rep_minud
from .ucml import UCModel1, canonical_labels


# ============================================================================ Model 1 over units
class PGFeaturizer:
    def __init__(self, sysm, train, relax=True):
        s = sysm
        self.s, self.relax = s, relax
        avg = s.avg_cost()
        rank = np.argsort(np.argsort(avg)) / s.G
        mc_lo = s.seg_c[:, 0]
        self.g_static = np.c_[np.log1p(s.pmax), s.pmin / np.maximum(s.pmax, 1e-9), np.log1p(avg), np.log1p(s.c_nl),
                              np.log1p(s.c_su), s.min_up, s.min_dn, np.minimum(s.ramp / np.maximum(s.pmax, 1e-9), 5),
                              rank, np.log1p(mc_lo)]
        self.mu_g, self.sd_g = self.g_static.mean(0), self.g_static.std(0) + 1e-6
        xs = self._sys(train)
        self.mu_s, self.sd_s = xs.reshape(-1, xs.shape[-1]).mean(0), xs.reshape(-1, xs.shape[-1]).std(0) + 1e-6
        self.avg, self.mc_lo = avg, mc_lo
        self.T = train["load"].shape[1]
        self.sys_dim, self.gen_dim = xs.shape[-1], self.g_static.shape[1]
        self.dyn_dim = 5 if relax else 0

    def _sys(self, d, idx=None):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        load = sel(d["load"]).sum(2)
        av = sel(d["avail"])
        wind = av[..., :-1].sum(2)
        net = load - av.sum(2)
        sr = sel(d["sr"])
        cols = [load / 1e4, net / 1e4, sr / 1e3, wind / 1e4]
        if self.relax:
            cols += [np.log1p(np.maximum(sel(d["lmp_rel"])[..., 0], 0)), np.log1p(np.maximum(sel(d["res_rel"]), 0))]
        T = load.shape[1]
        hour = np.broadcast_to(np.arange(T)[None] % 24 / 24.0, load.shape)
        cols += [np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour)]
        return np.stack(cols, -1)

    def __call__(self, d, idx=None):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        xs = torch.as_tensor((self._sys(d, idx) - self.mu_s) / self.sd_s, dtype=torch.float32)       # [B, T, ks]
        xg = torch.as_tensor((self.g_static - self.mu_g) / self.sd_g, dtype=torch.float32)            # [G, kg]
        if self.relax:
            ur = sel(d["u_rel"]).astype(np.float32)                                                    # [B, T, G]
            prev = np.concatenate([np.ones_like(ur[:, :1]), ur[:, :-1]], 1)
            nxt = np.concatenate([ur[:, 1:], ur[:, -1:]], 1)
            lmp = sel(d["lmp_rel"])[..., 0][..., None]                                                 # [B, T, 1]
            marg = np.clip((lmp - self.mc_lo[None, None]) / (np.abs(self.mc_lo[None, None]) + 1.0), -3, 3)
            marg2 = np.clip((lmp - self.avg[None, None]) / (np.abs(self.avg[None, None]) + 1.0), -3, 3)
            xd = torch.as_tensor(np.stack([ur, prev, nxt, marg, marg2], -1), dtype=torch.float32)       # [B, T, G, 5]
        else:
            B, T = xs.shape[:2]
            xd = torch.zeros(B, T, self.s.G, 0)
        return xs, xg, xd


class UnitNet(nn.Module):
    """logit[b, t, g] = f(unit encoder(static_g, embedding_g), hour encoder(system_t), dynamic_{b,t,g})"""

    def __init__(self, G, sys_dim, gen_dim, dyn_dim, hidden=48, emb=8):
        super().__init__()
        self.emb = nn.Embedding(G, emb)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.genc = nn.Sequential(nn.Linear(gen_dim + emb, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.senc = nn.Sequential(nn.Linear(sys_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.dec = nn.Sequential(nn.Linear(2 * hidden + dyn_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        self.dyn_dim = dyn_dim

    def forward(self, xs, xg, xd):
        G = xg.shape[0]
        hg = self.genc(torch.cat([xg, self.emb.weight], -1))                    # [G, H]
        hs = self.senc(xs)                                                       # [B, T, H]
        B, T, H = hs.shape
        z = torch.cat([hs[:, :, None].expand(B, T, G, H), hg[None, None].expand(B, T, G, H), xd], -1)
        return self.dec(z).squeeze(-1)                                          # [B, T, G]


def build_model1(sysm, feat, seed=0, hidden=48):
    torch.manual_seed(seed)
    net = UnitNet(sysm.G, feat.sys_dim, feat.gen_dim, feat.dyn_dim, hidden=hidden)
    return UCModel1(net, feat, "gnn")


# ============================================================================ dispatch LP, persistent
class FastDispatch:
    """The dispatch LP of PGModel kept loaded in one highspy object. solve(sc, u) changes only bounds (and the
    scenario's row bounds when the scenario changes) and re-runs from the previous basis."""

    def __init__(self, m: PGModel):
        import highspy
        self.m, self.hs = m, highspy
        self.h = None
        self.key = None
        self.ucols = np.r_[m.blk("u").reshape(-1), m.blk("v").reshape(-1), m.blk("w").reshape(-1)].astype(np.int32)

    def _load(self, sc, lo, hi, lb, ub):
        hs = self.hs
        m = self.m
        A = m.A.tocsc()
        inf = hs.kHighsInf
        lp = hs.HighsLp()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_ = m.c
        lp.col_lower_ = np.where(np.isfinite(lb), lb, -inf)
        lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
        lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
        lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data.astype(float)
        h = hs.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.passModel(lp)
        self.h = h

    def solve(self, sc, u, key=None):
        m = self.m
        lo, hi, lb, ub = m._rhs_bounds(sc, u)
        t0 = time.time()
        if self.h is None or key is None or key != self.key:
            self._load(sc, lo, hi, lb, ub)
            self.key = key
        else:
            c = self.ucols
            self.h.changeColsBounds(len(c), c, lb[c], ub[c])
        self.h.run()
        info = self.h.getInfo()
        st = self.h.modelStatusToString(self.h.getModelStatus())
        if info.primal_solution_status != 2:
            self.h = None
            return np.inf, np.inf, np.inf, time.time() - t0
        x = np.array(self.h.getSolution().col_value)
        g = lambda name: m.get(x, name)
        return (float(info.objective_function_value), float(g("shed").sum() + g("spill").sum()), float(g("short").sum()),
                time.time() - t0)


_W = {}


def _init(cfg):
    sysm, _ = load_bases(cfg["T"])
    m = PGModel(sysm, T=cfg["T"])
    _W.update(s=sysm, m=m, fd=FastDispatch(m))


def _dispatch(args):
    key, load, avail, avail_min, sr, u0, u = args
    sc = PGScenario(load=load, avail=avail, avail_min=avail_min, sr=sr, u0=u0)
    c, sh, so, _ = _W["fd"].solve(sc, u, key=key)
    return c, sh, so


class DispatchOracle:
    """otsl.ucml.DispatchOracle for PGLib: fixed-commitment dispatch LPs in a spawn pool, memoised by
    (instance key, commitment). Jobs of the same instance go to the same chunk where possible (warm starts)."""

    def __init__(self, cfg, workers=2):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))
        self.cache, self.n = {}, 0

    def evaluate(self, d, idx, us, keys):
        ck = [(int(k), np.asarray(u, np.int8).tobytes()) for k, u in zip(keys, us)]
        todo = {}
        for j, c in enumerate(ck):
            if c not in self.cache and c not in todo:
                todo[c] = j
        if todo:
            js = sorted(todo.values(), key=lambda j: ck[j][0])
            args = [(int(keys[j]), d["load"][idx[j]], d["avail"][idx[j]], d["avail_min"][idx[j]], d["sr"][idx[j]],
                     d["u0"][idx[j]], us[j]) for j in js]
            res = self.pool.map(_dispatch, args, chunksize=max(1, len(js) // 8))
            self.n += len(js)
            for j, r in zip(js, res):
                self.cache[ck[j]] = r
        out = np.array([self.cache[c] for c in ck])
        return out[:, 0], out[:, 1], out[:, 2]

    def close(self):
        self.pool.close()
        self.pool.join()


# ============================================================================ Learning to Fix: Table II features, kNN
def table2_features(d):
    """Table II of Fritz et al. for a copper-plate system with one aggregate wind profile (Scenario400; zero
    otherwise): aggregate demand L, wind W, solar S (= 0), hydro H (= 0), RES = W, net load N = L - RES; hourly deltas
    (0 in the first hour); 3-step moving average and rolling max of N; N / mean and N / max over the horizon and over
    its calendar day; sin / cos of the hour of day; initial condition of every free unit (signed hours in state; all
    units start on with the same history in PGLib-CA, so this block is constant and drops out after scaling)."""
    n, T, _ = d["load"].shape
    L = d["load"].sum(2)
    Wd = d["avail"][..., :-1].sum(2)
    S = np.zeros_like(L)
    H = np.zeros_like(L)
    RES = Wd + S + H
    N = L - RES
    delta = lambda V: np.c_[np.zeros((n, 1)), np.diff(V, axis=1)]
    ma3 = np.stack([N[:, max(0, t - 2):t + 1].mean(1) for t in range(T)], 1)
    mx3 = np.stack([N[:, max(0, t - 2):t + 1].max(1) for t in range(T)], 1)
    nmean = N / N.mean(1, keepdims=True)
    nmax = N / N.max(1, keepdims=True)
    day = np.arange(T) // 24
    dmean = np.zeros_like(N); dmax = np.zeros_like(N)
    for k in np.unique(day):
        sl = day == k
        dmean[:, sl] = N[:, sl] / N[:, sl].mean(1, keepdims=True)
        dmax[:, sl] = N[:, sl] / N[:, sl].max(1, keepdims=True)
    hour = np.arange(T)[None].repeat(n, 0) % 24
    sin, cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    eta = np.where(np.asarray(d["u0"]) > 0, 1.0, -1.0)
    return np.concatenate([L, Wd, S, H, RES, N, delta(L), delta(RES), delta(N), ma3, mx3, nmean, nmax, dmean, dmax,
                           sin, cos, eta], 1)


class KNNProb:
    """eq. (13): inverse-distance-weighted mean of the k nearest training schedules (canonicalised inside groups of
    identical units); Euclidean distance on z-scored Table II features. As otsl.ltfx.KNNProb."""

    def __init__(self, sysm, train, labels, k=50):
        self.s, self.k = sysm, k
        Z = table2_features(train)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        self.X = (Z - self.mu) / self.sd
        self.Y = canonical_labels(sysm, labels, train["u0"]).astype(np.float64)

    def predict(self, d):
        Z = (table2_features(d) - self.mu) / self.sd
        D2 = (Z ** 2).sum(1)[:, None] + (self.X ** 2).sum(1)[None] - 2 * Z @ self.X.T
        k = min(self.k, len(self.X))
        idx = np.argsort(D2, 1, kind="stable")[:, :k]
        rho = np.sqrt(np.maximum(np.take_along_axis(D2, idx, 1), 0))
        w = 1.0 / np.maximum(rho, 1e-9)
        w /= w.sum(1, keepdims=True)
        return np.einsum("nk,nktg->ntg", w, self.Y[idx])


class PGLtFTuner(LtFTuner):
    """Algorithm 1 + 2 of otsl.ltfx on PGLib instances (the instance problems are built from PGModel's matrices)."""

    def prob(self, i):
        from .ltfx import InstanceProblems
        if i not in self.probs:
            sc = PGScenario(load=self.d["load"][i], avail=self.d["avail"][i], avail_min=self.d["avail_min"][i],
                            sr=self.d["sr"][i], u0=self.d["u0"][i])
            self.probs[i] = InstanceProblems(self.m, sc, self.d["obj"][i], self.eps)
        return self.probs[i]


# ============================================================================ guards
def lp_guard(m: PGModel, sysm, sc, fix, max_iter=4, tol=1e-6):
    """otsl.fixpolicy.lp_guard (same release logic) with the relaxed reduced LP solved by PGModel.relaxed_reduced"""
    fix = dict(fix)
    n0, secs = len(fix), 0.0
    for it in range(max_iter + 1):
        _, slack, dt = m.relaxed_reduced(sc, fix)
        secs += dt
        bad = np.where(slack > tol)[0]
        if len(bad) == 0:
            break
        if it == max_iter:
            for t in bad:
                for tt in (t - 1, t, t + 1):
                    for g in range(sysm.G):
                        fix.pop((int(tt), g), None)
            break
        for t in bad:
            for (tt, g), v in list(fix.items()):
                if v == 0 and t - sysm.min_dn[g] <= tt <= t:
                    del fix[(tt, g)]
        if it >= 1:
            for t in bad:
                for (tt, g), v in list(fix.items()):
                    if v == 1 and t - sysm.min_up[g] <= tt <= t:
                        del fix[(tt, g)]
    return fix, n0 - len(fix), secs


def repair_fn(sysm):
    return lambda u, u0: rep_minud(u, sysm, u0)


# ============================================================================ solver jobs (spawn pool workers)
def _sc(a):
    return PGScenario(load=a["load"], avail=a["avail"], avail_min=a["avail_min"], sr=a["sr"], u0=a["u0"])


def inst_arrays(d, i):
    return {k: d[k][i] for k in ("load", "avail", "avail_min", "sr", "u0")}


def _as_fix(arr):
    return {(int(t), int(g)): int(v) for t, g, v in np.asarray(arr).reshape(-1, 3)}


def fix_array(fix):
    return np.array([(t, g, v) for (t, g), v in fix.items()], dtype=np.int64).reshape(-1, 3)


def reduced_job(job):
    """self-training label: reduced MILP with fixings, then the exact dispatch LP of its schedule"""
    i, a, fix_arr, tl, gap = job
    m, sc = _W["m"], _sc(a)
    sol = m.solve_milp(sc, time_limit=tl, mip_gap=gap, z_fix=_as_fix(fix_arr) or None, incumbents=False)
    if sol["u"] is None:
        return dict(i=i, u=None, cost=np.inf, milp_s=sol["time"], status=sol["status"])
    c, sh, so, dt = _W["fd"].solve(sc, sol["u"], key=("st", i))
    return dict(i=i, u=sol["u"], cost=c, shed=sh, short=so, milp_obj=sol["obj"], milp_s=sol["time"], lp_s=dt,
                status=sol["status"], gap=sol["gap"])


def harm_job(job):
    """otsl.fixpolicy.harm_labels (single-decision harms, no LtF curves) with the persistent dispatch LP: for every
    decision where the rounded prediction disagrees with the (aligned) reference schedule, the relative cost increase
    of the reference with that decision forced to the prediction (row repaired for min up/down)."""
    from .fixpolicy import align_to_prediction, repair_row_forced
    i, a, u_star, p = job
    s, sc, fd = _W["s"], _sc(a), _W["fd"]
    T, G = p.shape
    yhat = (p > 0.5).astype(np.int8)
    ua = align_to_prediction(s, u_star, yhat, a["u0"])
    base = fd.solve(sc, ua, key=("h", i))[0]
    wrong = np.argwhere(yhat != ua)
    out, t0 = {}, time.time()
    for t, g in wrong:
        u2 = ua.copy()
        row = repair_row_forced(ua[:, g], a["u0"][g], int(s.min_up[g]), int(s.min_dn[g]), {int(t): int(yhat[t, g])})
        if row is None:
            out[(int(t), int(g))] = np.inf
            continue
        u2[:, g] = row
        out[(int(t), int(g))] = (fd.solve(sc, u2, key=("h", i))[0] - base) / base
    return dict(i=i, base=base, single=out, n_wrong=len(wrong), secs=time.time() - t0)


def fixeval_job(job):
    """test: (optionally) the full MILP, then every reduced MILP of one instance back to back in one process.
    specs: list of (name, fix array [k, 3], extra seconds already spent on inference / guards)."""
    i, a, specs, tl, gap, full = job
    m, sc = _W["m"], _sc(a)
    out = {}
    if full:
        sol = m.solve_milp(sc, time_limit=tl, mip_gap=gap, incumbents=True)
        out["__full__"] = dict(obj=sol["obj"], bound=sol["bound"], gap=sol["gap"], time=sol["time"],
                               status=sol["status"], inc=sol["inc"], shed=sol["shed"], short=sol["short"])
    for name, fix_arr, extra, lpg in specs:
        fix = _as_fix(fix_arr)
        n_pre, lpg_s, rel_lp = len(fix), 0.0, 0
        if lpg and fix:                       # LP-relaxation guard (its LPs are timed and added to the method)
            t0 = time.time()
            fix, rel_lp, _ = lp_guard(m, _W["s"], sc, fix)
            lpg_s = time.time() - t0
        sol = m.solve_milp(sc, time_limit=tl, mip_gap=gap, z_fix=fix or None, incumbents=False)
        out[name] = dict(obj=sol["obj"], bound=sol["bound"], gap=sol["gap"], time=sol["time"], status=sol["status"],
                         shed=sol["shed"], short=sol["short"], n_fixed_final=int(len(fix)), n_fixed_pre=n_pre,
                         released_lp=int(rel_lp), lpg_s=float(lpg_s), extra_s=float(extra),
                         feasible=sol["u"] is not None)
    out["__loadavg__"] = os.getloadavg()[0]
    return i, out


def dispatch_job(job):
    """end-to-end: dispatch LPs of given commitments of one instance -> [(cost, shed, short, seconds)]"""
    i, a, us = job
    sc, fd = _sc(a), _W["fd"]
    return i, [fd.solve(sc, u, key=("e2e", i)) for u in us]


class SolverPool:
    def __init__(self, cfg, workers=2):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,))

    def run(self, fn, jobs, log=None, every=1):
        res, t0 = [], time.time()
        for k, r in enumerate(self.pool.imap_unordered(fn, jobs, chunksize=1)):
            res.append(r)
            if log and (k + 1) % every == 0:
                log(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
        return res

    def close(self):
        self.pool.close()
        self.pool.join()
