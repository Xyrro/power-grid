"""Parallel generation of unit-commitment datasets (MILP labels, LP relaxation, simple baselines)."""
from __future__ import annotations

import multiprocessing as mp  # spawn: HiGHS thread pools do not survive fork()
import os
import time

import numpy as np

from .uc import UCModel, UCScenario, load_rts_gmlc, make_scenario, repair_min_updown

_W = {}


def _init(cfg):
    s = load_rts_gmlc(line_scale=cfg.get("line_scale", 1.0))
    _W.update(s=s, m=UCModel(s, T=cfg["T"], network=cfg.get("network", True)), cfg=cfg)


def scenario_from(d: dict, i: int) -> UCScenario:
    return UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])


def _solve(job):
    day, start, seed = job
    s, m, cfg = _W["s"], _W["m"], _W["cfg"]
    sc = make_scenario(s, day, np.random.default_rng(seed), T=cfg["T"], start=start)
    out = {"load": sc.load, "avail": sc.avail, "u0": sc.u0, "sr": sc.sr, "day": day, "start": start}
    rel = m.solve_dispatch(sc, None, relax=True)
    out.update(u_rel=rel.u, lmp_rel=rel.lmp, c_rel=rel.obj, flow_rel=rel.flow)
    sol = m.solve_uc(sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"])
    out.update(u=sol.u, p=sol.p, va=sol.va, r=sol.r, obj=sol.obj, time=sol.time, gap=sol.gap,
               opt=sol.status == "optimal", shed=sol.shed)
    # simple baselines: keep the units that were running (persistence); rounded LP relaxation (+ repair)
    allon = m.solve_dispatch(sc, np.repeat(sc.u0[None, :], cfg["T"], 0))
    rnd = (rel.u > 0.5).astype(np.int8)
    if cfg["T"] > 1:
        rnd = repair_min_updown(rnd, sc.u0, s.min_up, s.min_dn)
    rr = m.solve_dispatch(sc, rnd)
    out.update(c_u0=allon.obj, c_round=rr.obj)
    alts = []
    for _ in range(cfg.get("n_alt", 0)):
        a = m.solve_uc(sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], nogood=[sol.u] + [x[0] for x in alts])
        if a.u is None:
            break
        lp = m.solve_dispatch(sc, a.u)          # exact cost of that commitment
        alts.append((a.u, lp.obj))
    if alts:
        out["alt_u"] = np.array([a[0] for a in alts])
        out["alt_c"] = np.array([a[1] for a in alts])
    return out


def generate(cfg: dict, days, n: int, seed: int, workers=2) -> dict:
    rng = np.random.default_rng(seed)
    T = cfg["T"]
    jobs = []
    for i in range(n):
        day = int(rng.choice(days))
        start = int(rng.integers(0, 24 - T + 1))
        jobs.append((day, start, int(rng.integers(1 << 31))))
    res, t0 = [], time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap(_solve, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % max(1, n // 10) == 0:
                print(f"[uc-generate] {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    keep = [r for r in res if r["u"] is not None]
    print(f"[uc-generate] kept {len(keep)}/{n}; MILP optimal {np.mean([r['opt'] for r in keep]) * 100:.1f}% "
          f"mean time {np.mean([r['time'] for r in keep]):.2f}s")
    data = {k: np.array([r[k] for r in keep]) for k in
            ["load", "avail", "u0", "sr", "day", "start", "u_rel", "lmp_rel", "c_rel", "flow_rel", "u", "p", "va",
             "r", "obj", "time", "gap", "opt", "shed", "c_u0", "c_round"]}
    if cfg.get("n_alt", 0):
        na, (Tn, G) = cfg["n_alt"], keep[0]["u"].shape
        az = np.zeros((len(keep), na, Tn, G), np.int8)
        ac = np.full((len(keep), na), np.inf)
        for i, r in enumerate(keep):
            if "alt_u" in r:
                k = len(r["alt_u"])
                az[i, :k], ac[i, :k] = r["alt_u"], r["alt_c"]
        data["alt_u"], data["alt_c"] = az, ac
    return data


def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)


def load(path):
    return dict(np.load(path, allow_pickle=False))
