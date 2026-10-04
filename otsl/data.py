"""Scenario sampling and parallel generation of DC-OTS / DC-OPF datasets."""
from __future__ import annotations

import os
import multiprocessing as mp  # always "spawn": HiGHS thread pools do not survive fork()

import numpy as np

from .case import load_case
from .opt import DCModel

_W = {}


def make_model(cfg: dict) -> DCModel:
    case = load_case(cfg["case"], line_limit_scale=cfg.get("line_scale", 1.0))
    fixed = case.bridges() if cfg.get("fix_bridges", True) else None
    budget = cfg.get("budget")
    bound_k = None if budget is None else budget + cfg.get("outages", 0)
    m = DCModel(case, budget=budget, fixed_closed=fixed, bound_k=bound_k)
    if cfg.get("switch_cost_rel", 0.0) > 0:
        # per-line switching cost = rel * (nominal all-closed DC-OPF cost)
        m.switch_cost = cfg["switch_cost_rel"] * m.solve_lp(case.pd * cfg.get("load_factor", 1.0)).obj
    return m


def sample_loads(case, n, rng, load_factor=1.0, glob=(0.85, 1.15), loc=(0.9, 1.1)):
    """PGLearn-style sampling: global scale x independent per-bus noise."""
    g = rng.uniform(*glob, size=(n, 1))
    l = rng.uniform(*loc, size=(n, case.n_bus))
    return case.pd[None, :] * load_factor * g * l


def _init(cfg):
    _W["m"] = make_model(cfg)
    _W["cfg"] = cfg


def _solve_one(args):
    m, cfg = _W["m"], _W["cfg"]
    pd, base = args if isinstance(args, tuple) else (args, None)
    lp = m.solve_lp(pd, base)
    out = {"pd": pd, "lp_ok": lp.ok}
    if base is not None:
        out["base"] = base
    if not lp.ok:
        return out
    out.update(c0=lp.obj, lmp0=lp.lmp, mu0=lp.mu, gamma0=lp.gamma, flow0=lp.flow, pg0=lp.pg, va0=lp.va)
    ots = m.solve_ots(pd, time_limit=cfg.get("time_limit", 60), mip_gap=cfg.get("mip_gap", 1e-4), base=base)
    out.update(ots_status=ots.status, ots_time=ots.time, ots_gap=ots.gap)
    if ots.z is None:
        return out
    out.update(z=ots.z, c_ots=ots.obj, pg=ots.pg, va=ots.va, flow=ots.flow)
    # alternative (near-)optimal topologies via no-good cuts: measures label ambiguity
    alts = []
    for _ in range(cfg.get("n_alt", 0)):
        alt = m.solve_ots(pd, time_limit=cfg.get("time_limit", 60), mip_gap=cfg.get("mip_gap", 1e-4),
                          nogood=[out["z"]] + [a[0] for a in alts], base=base)
        if alt.z is None:
            break
        lp_alt = m.solve_lp(pd, alt.z)  # exact cost of that topology (+ switching cost)
        alts.append((alt.z, lp_alt.obj + m.switch_cost * (alt.z == 0).sum() if lp_alt.ok else np.inf))
    if alts:
        out["alt_z"] = np.array([a[0] for a in alts])
        out["alt_c"] = np.array([a[1] for a in alts])
    return out


def generate(cfg: dict, n: int, seed: int, workers: int = 4) -> dict:
    """Sample n load scenarios and solve all-closed DC-OPF + DC-OTS for each."""
    m = make_model(cfg)
    rng = np.random.default_rng(seed)
    pds = sample_loads(m.case, n, rng, load_factor=cfg.get("load_factor", 1.0))
    jobs = list(pds)
    if cfg.get("outages", 0):
        # base-case topology changes: `outages` random non-bridge lines out of service, with probability
        # outage_prob per scenario (tests generalisation to topologies not seen in training)
        cand = np.where(~m.fixed_closed)[0]
        bases = np.ones((n, m.case.n_line), np.int8)
        for i in range(n):
            if rng.random() < cfg.get("outage_prob", 1.0):
                bases[i, rng.choice(cand, cfg["outages"], replace=False)] = 0
        jobs = list(zip(pds, bases))
    import time
    t0, res = time.time(), []
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap(_solve_one, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % max(1, n // 10) == 0:
                print(f"[generate] {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    keep = [r for r in res if r.get("lp_ok") and "z" in r]
    print(f"[generate] {len(keep)}/{n} scenarios kept "
          f"({sum(not r.get('lp_ok') for r in res)} infeasible OPF, "
          f"{sum(r.get('ots_status') == 'time_limit_feasible' for r in res)} MILP hit time limit)")
    data = {}
    for k in ["pd", "c0", "lmp0", "mu0", "gamma0", "flow0", "pg0", "va0", "z", "c_ots", "pg", "va", "flow",
              "ots_time", "ots_gap"]:
        data[k] = np.array([r[k] for r in keep])
    data["ots_opt"] = np.array([r["ots_status"] == "optimal" for r in keep])
    if cfg.get("outages", 0):
        data["base"] = np.array([r["base"] for r in keep]).astype(np.int8)
    data["switch_cost"] = np.array(m.switch_cost)
    if cfg.get("n_alt", 0):
        na = cfg["n_alt"]
        L = m.case.n_line
        az = np.ones((len(keep), na, L), np.int8)
        ac = np.full((len(keep), na), np.inf)
        for i, r in enumerate(keep):
            if "alt_z" in r:
                k = len(r["alt_z"])
                az[i, :k], ac[i, :k] = r["alt_z"], r["alt_c"]
        data["alt_z"], data["alt_c"] = az, ac
    return data


def _lp_pair(args):
    pd, z = args
    s = _W["m"].solve_lp(pd, z)
    if not s.ok:
        return None
    return s.obj, s.pg, s.va, s.flow


def solve_lps(cfg: dict, pds: np.ndarray, zs: np.ndarray, workers: int = 4):
    """Batch fixed-topology DC-OPF (the 'LP Solver' box): returns cost, pg, va, flow (NaN if infeasible)."""
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_lp_pair, list(zip(pds, zs)), chunksize=8)
    m = make_model(cfg)
    G, N, L = m.case.n_gen, m.case.n_bus, m.case.n_line
    n = len(res)
    cost = np.full(n, np.inf)
    pg, va, fl = np.full((n, G), np.nan), np.full((n, N), np.nan), np.full((n, L), np.nan)
    for i, r in enumerate(res):
        if r is not None:
            cost[i], pg[i], va[i], fl[i] = r
    return cost, pg, va, fl


def save(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: str) -> dict:
    return dict(np.load(path, allow_pickle=False))
