"""B3: a harder unit-commitment benchmark (24-hour horizon on RTS-GMLC) and MILP-free training data.

* solve_milp_hs   the UC MILP of otsl.uc.UCModel (same matrices, bounds and objective) solved with highspy
                  instead of scipy.optimize.milp, so that every improving incumbent can be logged
                  (time, objective, dual bound) through HiGHS's MIP callback; needed for time-to-quality.
                  The full and the reduced (partially fixed) MILPs of B3 are all solved through this path.
* generate_b3     parallel instance generation (spawn pool):
                    mode "lp"   - training data WITHOUT any MILP: scenario, LP relaxation (commitment, prices,
                                  flows), the repaired relaxation label and its dispatch-LP cost; u / obj = NaN;
                    mode "milp" - the same plus the full MILP with its incumbent trajectory (val / test).
* repairs         min up/down repair, per-hour adequacy repair and block (min up/down-aware) adequacy repair.
"""
from __future__ import annotations

import multiprocessing as mp  # spawn: HiGHS thread pools do not survive fork()
import os
import time

import numpy as np

from .constrained import adequacy_repair_blocks
from .uc import UCModel, UCScenario, adequacy_repair, load_rts_gmlc, make_scenario, repair_min_updown

K_INC = 64          # incumbents kept per MILP (first K_INC - 1 and the last)


# ----------------------------------------------------------------------------------- MILP via highspy
def solve_milp_hs(m: UCModel, sc: UCScenario, time_limit=600.0, mip_gap=1e-3, z_fix: dict | None = None,
                  threads=1, incumbents=True, trace_every=0.0):
    """MILP of UCModel.solve_uc (identical model) through highspy with an incumbent log.
    Returns dict(status, obj, u, time, gap, bound, inc [(seconds, objective, dual bound)], nodes); with
    trace_every > 0 also trace [(seconds, primal bound, dual bound)] sampled every trace_every seconds."""
    import highspy
    lo, hi, lb, ub = m._rhs_bounds(sc)
    for (t, g), v in (z_fix or {}).items():
        i = m.ix("u", t, g)
        lb[i] = ub[i] = v
    A = m.A.tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.setOptionValue("threads", int(threads))
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = m.c
    lp.col_lower_ = lb
    lp.col_upper_ = np.where(np.isfinite(ub), ub, highspy.kHighsInf)
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -highspy.kHighsInf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, highspy.kHighsInf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in m.integ]
    h.passModel(lp)
    inc = []
    if incumbents:
        def cb(e):
            o = e.data_out
            inc.append((float(o.running_time), float(o.objective_function_value), float(o.mip_dual_bound)))
        h.cbMipImprovingSolution.subscribe(cb)
    trace = []
    if trace_every > 0:
        def cb2(e):
            o = e.data_out
            if not trace or o.running_time - trace[-1][0] >= trace_every:
                trace.append((float(o.running_time), float(o.mip_primal_bound), float(o.mip_dual_bound)))
        h.cbMipInterrupt.subscribe(cb2)
    t0, c0 = time.time(), time.process_time()
    h.run()
    dt, cpu = time.time() - t0, time.process_time() - c0
    st = h.getModelStatus()
    info = h.getInfo()
    name = h.modelStatusToString(st)
    if info.primal_solution_status != 2:            # no feasible solution
        return dict(status="infeasible" if "nfeasible" in name else "no_solution", obj=np.inf, u=None, time=dt,
                    gap=np.inf, bound=float(info.mip_dual_bound), inc=inc, nodes=int(info.mip_node_count), trace=trace,
                    cpu=cpu)
    x = np.array(h.getSolution().col_value)
    u, p, r, th, f, shed = m._unpack(x)
    short = m._short
    status = "optimal" if name == "Optimal" else ("time_limit" if "Time limit" in name else name)
    return dict(status=status, obj=float(info.objective_function_value), u=np.round(u).astype(np.int8), p=p,
                time=dt, gap=float(info.mip_gap), bound=float(info.mip_dual_bound), inc=inc,
                nodes=int(info.mip_node_count), shed=float(shed), short=float(short), trace=trace, cpu=cpu)


def time_to_reach(inc, target, rel_tol=1e-6):
    """first time (s) at which an incumbent log [(t, obj, bound)] reaches obj <= target * (1 + rel_tol);
    nan if never"""
    for t, o, _ in inc:
        if o <= target * (1 + rel_tol) + 1e-9:
            return t
    return np.nan


def pack_inc(incs, k=K_INC):
    """list of incumbent logs -> arrays [n, k] (time, objective), NaN padded; keeps the first k-1 and the last"""
    n = len(incs)
    tt, oo = np.full((n, k), np.nan), np.full((n, k), np.nan)
    for i, inc in enumerate(incs):
        if len(inc) > k:
            inc = list(inc[:k - 1]) + [inc[-1]]
        for j, (t, o, _) in enumerate(inc):
            tt[i, j], oo[i, j] = t, o
    return tt, oo


def inc_from_arrays(tt, oo):
    return [(t, o, np.nan) for t, o in zip(tt, oo) if np.isfinite(t)]


# ----------------------------------------------------------------------------------- repairs
def rep_minud(u, u0, sysm):
    return repair_min_updown(np.asarray(u, np.int8), u0, sysm.min_up, sysm.min_dn)


def rep_adequacy(u, d, i, sysm):
    """per-hour adequacy repair then min up/down repair (the repair used for the B2 relaxation labels)"""
    v = adequacy_repair(np.asarray(u, np.int8), d["load"][i], d["avail"][i], d["sr"][i], sysm)
    return rep_minud(v, d["u0"][i], sysm)


def rep_block(u, d, i, sysm, margin=0.0):
    """block (min up/down-aware) adequacy repair then min up/down repair (W1)"""
    v = adequacy_repair_blocks(np.asarray(u, np.int8), d["u0"][i], d["load"][i], d["avail"][i], d["sr"][i], sysm,
                               margin=margin)
    return rep_minud(v, d["u0"][i], sysm)


# ----------------------------------------------------------------------------------- generation
_W = {}


def _init(cfg):
    s = load_rts_gmlc(line_scale=cfg.get("line_scale", 1.0))
    _W.update(s=s, m=UCModel(s, T=cfg["T"], network=cfg.get("network", True)), cfg=cfg)


def instance(job, cfg, s, m):
    """one B3 instance (job = (day, start, seed)); mode lp: no MILP; mode milp: + full MILP with incumbents"""
    day, start, seed = job
    T = cfg["T"]
    t0 = time.time()
    sc = make_scenario(s, day, np.random.default_rng(seed), T=T, start=start)
    out = {"load": sc.load, "avail": sc.avail, "u0": sc.u0, "sr": sc.sr, "day": day, "start": start, "seed": seed}
    rel = m.solve_dispatch(sc, None, relax=True)
    out.update(u_rel=rel.u, lmp_rel=rel.lmp, c_rel=rel.obj, flow_rel=rel.flow, t_rel=rel.time)
    d1 = {k: np.asarray(v)[None] for k, v in out.items() if k in ("load", "avail", "u0", "sr")}
    rnd = (rel.u > 0.5).astype(np.int8)
    lab = rep_adequacy(rnd, d1, 0, s)                  # repaired relaxation label (V4)
    lab_b = rep_block(rnd, d1, 0, s)                    # block-repaired relaxation (W1 heuristic)
    lp_r = m.solve_dispatch(sc, rep_minud(rnd, sc.u0, s))
    lp_l = m.solve_dispatch(sc, lab)
    lp_b = m.solve_dispatch(sc, lab_b)
    out.update(y_lf=lab, c_lf=lp_l.obj, short_lf=lp_l.short, shed_lf=lp_l.shed, y_blk=lab_b, c_blk=lp_b.obj,
               short_blk=lp_b.short, shed_blk=lp_b.shed, c_round=lp_r.obj)
    out["t_label"] = time.time() - t0               # relaxation + repairs + 3 dispatch LPs
    G = s.G
    if cfg.get("mode", "lp") == "milp":
        sol = solve_milp_hs(m, sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], threads=1,
                            trace_every=cfg.get("trace_every", 0.0))
        ok = sol["u"] is not None
        out.update(u=sol["u"] if ok else np.zeros((T, G), np.int8), obj=sol["obj"], time=sol["time"], cpu=sol["cpu"],
                   gap=sol["gap"], bound=sol["bound"], opt=sol["status"] == "optimal", nodes=sol["nodes"],
                   shed=sol.get("shed", np.nan), short=sol.get("short", np.nan), inc=sol["inc"], trace=sol["trace"])
        out["loadavg"] = os.getloadavg()[0]            # 1-min load average at the end of the solve (contention)
        if ok:   # dispatch LP of the MILP schedule (sanity: reproduces the MILP objective)
            out["obj_lp"] = m.solve_dispatch(sc, sol["u"]).obj
        else:
            out["obj_lp"] = np.nan
    else:
        out.update(u=np.full((T, G), -1, np.int8), obj=np.nan, time=np.nan, cpu=np.nan, gap=np.nan, bound=np.nan, opt=False,
                   nodes=-1, shed=np.nan, short=np.nan, inc=[], obj_lp=np.nan, loadavg=os.getloadavg()[0])
    return out


def _job(job):
    return instance(job, _W["cfg"], _W["s"], _W["m"])


ARRAY_KEYS = ["load", "avail", "u0", "sr", "day", "start", "seed", "u_rel", "lmp_rel", "c_rel", "flow_rel", "t_rel",
              "y_lf", "c_lf", "short_lf", "shed_lf", "y_blk", "c_blk", "short_blk", "shed_blk", "c_round", "t_label",
              "u", "obj", "time", "cpu", "gap", "bound", "opt", "nodes", "shed", "short", "obj_lp", "loadavg"]


def make_jobs(days, n, seed, T=24, start=0):
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(n):
        day = int(rng.choice(days))
        st = int(rng.integers(0, 24 - T + 1)) if start is None else int(start)
        jobs.append((day, st, int(rng.integers(1 << 31))))
    return jobs


def generate_b3(cfg, jobs, workers=2, log=print, partial_path=None):
    """runs `instance` for every job in a spawn pool (imap keeps the job order); returns a dict of arrays.
    partial_path: re-saved every 5 results so a long run leaves usable data behind."""
    res, t0 = [], time.time()
    n = len(jobs)
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap(_job, jobs, chunksize=1)):
            res.append(r)
            if cfg.get("mode") == "milp":
                log(f"[b3-gen] {i + 1}/{n} day {r['day']} MILP {r['time']:.1f}s gap {r['gap'] * 100:.3f}% "
                    f"{'opt' if r['opt'] else 'limit'} obj {r['obj']:.1f} lp-check {r['obj_lp']:.1f} "
                    f"incumbents {len(r['inc'])} ({time.time() - t0:.0f}s)")
            elif (i + 1) % max(1, n // 10) == 0:
                log(f"[b3-gen] {i + 1}/{n} ({time.time() - t0:.0f}s)")
            if partial_path and (i + 1) % 5 == 0:
                save_b3(partial_path, to_arrays(res))
    return to_arrays(res)


def to_arrays(res):
    data = {k: np.array([r[k] for r in res]) for k in ARRAY_KEYS}
    data["inc_t"], data["inc_obj"] = pack_inc([r["inc"] for r in res])
    data["n_inc"] = np.array([len(r["inc"]) for r in res])
    return data


def save_b3(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def load_b3(path):
    return dict(np.load(path, allow_pickle=False))


def subset(d, idx):
    n = len(d["load"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}
