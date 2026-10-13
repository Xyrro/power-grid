"""A harder unit-commitment benchmark for partial fixing: screening of candidate settings, the chosen benchmark
(RTS-GMLC, 12-hour windows, line ratings at 60 %), guards, polished labels and the guard-aware tuner used on it
(docs/methods/hard.md).

Screening (scripts/uc_hard_screen.py) measures, per candidate setting, how weak the LP relaxation is and how long
the full MILP takes:

* RTS-GMLC, networked, 24 h (and 48 h): derated line ratings (otsl.uc.load_rts_gmlc(line_scale=...)), higher
  spinning-reserve requirements (reserve_frac);
* PGLib-UC FERC (934 thermal units, copper plate) on the first 24 h of the winter-fleet files.

Statistics of the LP relaxation against the MILP schedule (relax_stats): gap of the LP bound to the MILP objective
and to the MILP's dual bound, share of integral relaxation values, and the number of relaxation values that are
integral but disagree with the MILP schedule ("wrong integral values"; the MILP schedule is first aligned to the
rounded relaxation over identical units, otsl.fixpolicy.align_to_prediction, so symmetric optima do not count).
"""
from __future__ import annotations

import time

import numpy as np

from .uc import UCScenario, UCSystem, make_scenario

INT_TOL = 1e-6


# ============================================================================ RTS-GMLC scenarios
def make_scenario_h(sysm: UCSystem, day: int, rng=None, T=24, start=0, reserve_frac=0.03, load_noise=(0.92, 1.08),
                    bus_noise=(0.97, 1.03), ren_noise=(0.8, 1.2)) -> UCScenario:
    """otsl.uc.make_scenario with a configurable reserve requirement; windows that run past midnight (e.g. a 48 h
    horizon) continue with the next calendar day's profiles. Same random draws, in the same order, as
    make_scenario, so a window inside one day reproduces make_scenario exactly."""
    if start + T <= 24:
        return make_scenario(sysm, day, rng, T=T, start=start, load_noise=load_noise, bus_noise=bus_noise,
                             ren_noise=ren_noise, reserve_frac=reserve_frac)
    rng = rng or np.random.default_rng(0)
    D = sysm.area_load.shape[0]
    if day * 24 + start + T > D * 24:
        raise ValueError("window runs past the end of the year")
    al_all = sysm.area_load.reshape(-1, 3)
    av_all = sysm.r_avail.reshape(-1, sysm.r_avail.shape[2])
    h = day * 24 + start
    lf = rng.uniform(*load_noise, size=(1, 3))
    bn = rng.uniform(*bus_noise, size=(1, sysm.n_bus))
    rf = rng.uniform(*ren_noise, size=(1, len(sysm.r_pmax)))
    al = al_all[h:h + T] * lf
    load = al[:, sysm.bus_area - 1] * sysm.load_share[None, :] * bn
    avail = np.minimum(av_all[h:h + T] * rf, sysm.r_pmax[None, :])
    sr = reserve_frac * load.sum(1)
    h0 = h - 1 if start > 0 else day * 24          # make_scenario uses hour 0 itself when start = 0
    n0 = rng.uniform(0.97, 1.03)
    net = (al_all[h0] * lf[0]).sum() * n0 - np.minimum(av_all[h0] * rf[0], sysm.r_pmax).sum() \
        + reserve_frac * (al_all[h0] * lf[0]).sum()
    avg_cost = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    u0 = np.zeros(sysm.G, np.int8)
    cap = 0.0
    for g in np.argsort(avg_cost):
        if cap >= net * 1.05:
            break
        u0[g] = 1
        cap += sysm.pmax[g]
    return UCScenario(load=load, avail=avail, u0=u0, sr=sr, day=day)


def split_days(sysm: UCSystem, T=24):
    """calendar-day split of otsl / scripts/uc_gen.py: test day % 5 == 0, validation day % 5 == 2, training the
    rest; days whose window would run past the end of the year are dropped"""
    D = sysm.area_load.shape[0]
    nd = int(np.ceil(T / 24))
    days = np.arange(D - nd + 1)
    return dict(train=days[(days % 5 != 0) & (days % 5 != 2)], val=days[days % 5 == 2], test=days[days % 5 == 0])


# ============================================================================ relaxation statistics
def relax_stats(u_rel, u_star, sysm, u0, tol=INT_TOL):
    """LP-relaxation commitment u_rel [T, G] vs a MILP schedule u_star [T, G]: share of integral values, number of
    fractional values, wrong integral values (integral in the relaxation, different in the aligned MILP schedule),
    and Hamming distance of the rounded relaxation to the aligned schedule."""
    from .fixpolicy import align_to_prediction
    u_rel = np.asarray(u_rel, float)
    integ = (u_rel <= tol) | (u_rel >= 1 - tol)
    rnd = (u_rel > 0.5).astype(np.int8)
    ua = align_to_prediction(sysm, np.asarray(u_star, np.int8), rnd, u0)
    wrong = integ & (rnd != ua)
    wrong_raw = integ & (rnd != np.asarray(u_star, np.int8))
    return dict(int_share=float(integ.mean()), n_frac=int((~integ).sum()), n_wrong=int(wrong.sum()),
                n_wrong_raw=int(wrong_raw.sum()), ham_round=int((rnd != ua).sum()),
                n_wrong_one=int((wrong & (rnd == 1)).sum()), n_wrong_zero=int((wrong & (rnd == 0)).sum()))


def trace_gap_times(trace, gaps=(0.01, 0.005, 0.0025, 0.001)):
    """first time each relative gap (primal - dual) / primal is reached in a (seconds, primal, dual) trace"""
    from .base import time_to_gap
    return {f"t_gap_{g * 100:g}": time_to_gap(trace, g) for g in gaps}


# ============================================================================ PGLib MILP with a bound trace
def pg_milp_trace(m, sc, time_limit=600.0, mip_gap=1e-3, z_fix=None, threads=1, trace_every=1.0):
    """PGModel MILP through highspy with an incumbent log [(s, obj, bound)] and a (s, primal, dual) trace sampled
    every trace_every seconds (otsl.pglib.highs_run plus the interrupt callback)."""
    import highspy
    import scipy.sparse as sp
    lo, hi, lb, ub = m.bounds_fixed(sc, z_fix)
    A = sp.csc_matrix(m.A)
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(m.c, float)
    lp.col_lower_ = np.where(np.isfinite(lb), lb, -inf)
    lp.col_upper_ = np.where(np.isfinite(ub), ub, inf)
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    K, C = highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous
    lp.integrality_ = [K if v else C for v in m.integ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.passModel(lp)
    inc, trace = [], []

    def cb(e):
        o = e.data_out
        inc.append((float(o.running_time), float(o.objective_function_value), float(o.mip_dual_bound)))
    h.cbMipImprovingSolution.subscribe(cb)
    if trace_every > 0:
        def cb2(e):
            o = e.data_out
            if not trace or o.running_time - trace[-1][0] >= trace_every:
                trace.append((float(o.running_time), float(o.mip_primal_bound), float(o.mip_dual_bound)))
        h.cbMipInterrupt.subscribe(cb2)
    t0, c0 = time.time(), time.process_time()
    h.run()
    dt, cpu = time.time() - t0, time.process_time() - c0
    info = h.getInfo()
    st = h.modelStatusToString(h.getModelStatus())
    feas = info.primal_solution_status == 2
    out = dict(status="optimal" if st == "Optimal" else ("time_limit" if "Time limit" in st else st),
               obj=float(info.objective_function_value) if feas else np.inf, time=dt, cpu=cpu,
               gap=float(info.mip_gap) if feas else np.inf, bound=float(info.mip_dual_bound),
               nodes=int(info.mip_node_count), inc=inc, trace=trace, u=None, shed=np.nan, short=np.nan)
    if feas:
        x = np.array(h.getSolution().col_value)
        u, p, _, _, _, shed = m._unpack(x)
        out.update(u=np.round(u).astype(np.int8), shed=float(shed), short=float(m._short))
    return out


# ============================================================================ the chosen benchmark
# RTS-GMLC, networked DC model, 12-hour windows (random start within a calendar day, as otsl.ucdata), continuous line
# ratings derated to 60 % (docs/methods/hard.md: the screening that chose it; 24 hours at 60 % gives a similar LP gap
# but full MILPs of 6-9 minutes). Instances: otsl.b3.instance (make_scenario noise, 3 % spinning reserve, LP relaxation,
# label-free labels, optionally the full MILP with an incumbent log and a bound trace).
LINE_SCALE = 0.6
T_H = 12
CFG = {"T": T_H, "network": True, "line_scale": LINE_SCALE, "time_limit": 600.0, "mip_gap": 1e-3, "trace_every": 1.0}
SPLIT_SEED = {"train": 621, "val": 622, "test": 623}
SPLIT_N = {"train": 3000, "val": 60, "test": 60}
DATA_DIR = "data/generated/hard"


def bench_system():
    from .uc import UCModel, load_rts_gmlc
    s = load_rts_gmlc(line_scale=LINE_SCALE)
    return s, UCModel(s, T=T_H, network=True)


def bench_jobs(split):
    """(day, start, seed) per instance of a split: calendar-day split of scripts/uc_gen.py, otsl.b3.make_jobs"""
    from .b3 import make_jobs
    from .uc import load_rts_gmlc
    days = split_days(load_rts_gmlc(), 24)[split]
    return make_jobs(days, SPLIT_N[split], SPLIT_SEED[split], T=T_H, start=None)


def load_split(split, data_dir=DATA_DIR, with_traces=False):
    """concatenate the finished chunks of a split (in chunk order); with_traces also returns the MILP incumbent logs
    and bound traces [(s, primal, dual)] per instance"""
    import glob
    import json
    import os
    from .b3 import load_b3
    fs = sorted(glob.glob(os.path.join(data_dir, f"{split}_[0-9][0-9].npz")))
    parts = [load_b3(f) for f in fs]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    if with_traces:
        inc, tr = [], []
        for f in fs:
            j = json.load(open(f.replace(".npz", "_trace.json")))
            inc += j["inc"]
            tr += j["trace"]
        return d, inc, tr
    return d


# ============================================================================ guards and polished labels (as otsl.m24)
def nearest_row(forced, u0g, U, D, T):
    """min up/down-feasible 0/1 row (no carry-over at t = 0, as in UCModel) minimising the number of entries of
    `forced` {t: v} it disagrees with (otsl.m24.nearest_row)"""
    INF = 10 ** 9
    cost = {(int(u0g), U if u0g else D): 0}
    back = []
    for t in range(T):
        nxt, bp = {}, {}
        f = forced.get(t)
        for (s, k), c in cost.items():
            for s2 in (0, 1):
                if s2 == s:
                    k2 = min(k + 1, U if s else D)
                else:
                    if (s == 1 and k < U) or (s == 0 and k < D):
                        continue
                    k2 = 1
                c2 = c + (0 if f is None else int(s2 != f))
                if c2 < nxt.get((s2, k2), INF):
                    nxt[(s2, k2)] = c2
                    bp[(s2, k2)] = (s, k)
        back.append(bp)
        cost = nxt
    st = min(cost, key=cost.get)
    row = np.zeros(T, np.int8)
    for t in range(T - 1, -1, -1):
        row[t] = st[0]
        st = back[t][st]
    return row


def release_min_conflicts(fix, sysm, u0, T):
    """per unit, release only the fixings that the nearest min up/down-feasible row contradicts (otsl.m24)"""
    rows = {}
    for (t, g), v in fix.items():
        rows.setdefault(g, {})[t] = v
    out = dict(fix)
    released = 0
    for g, forced in rows.items():
        row = nearest_row(forced, int(u0[g]), int(sysm.min_up[g]), int(sysm.min_dn[g]), T)
        for t, v in forced.items():
            if row[t] != v:
                del out[(t, g)]
                released += 1
    return out, released


def guard_chain(m, sysm, sc, fix, lp=False):
    """adequacy guard (5 % margin) -> minimal min up/down conflict release [-> LP-relaxation guard] (otsl.m24);
    returns (fix, info) with the released counts and the guards' wall time"""
    from .combo import adequacy_guard
    from .fixpolicy import lp_guard
    t0 = time.time()
    fix, r_a = adequacy_guard(dict(fix), sc.load, sc.avail, sc.sr, sysm)
    fix, r_c = release_min_conflicts(fix, sysm, sc.u0, sc.load.shape[0])
    info = dict(released_adeq=r_a, released_rows=r_c)
    if lp and fix:
        fix, r_l, lp_s = lp_guard(m, sysm, sc, fix)
        info.update(released_lp=r_l, lp_guard_s=lp_s)
    info["guard_s"] = time.time() - t0
    return fix, info


def teacher_fixings(p, m, sysm, sc, ratio=0.9, lp=True):
    """the ratio share of the teacher's most confident decisions fixed to round(p), then the guard chain"""
    from .fixpolicy import fix_from_ranking
    fix = fix_from_ranking(np.minimum(p, 1 - p), (p > 0.5).astype(int), ratio)
    return guard_chain(m, sysm, sc, fix, lp=lp)


def polish(m, sysm, sc, p, y_ref, c_ref, ratio=0.9, time_limit=10.0, gap=1e-3, lp=True):
    """teacher-polished label (otsl.m24.polish): reduced MILP around the teacher's confident decisions (highspy, one
    thread); the label is its schedule if the dispatch LP prices it below the reference label y_ref (cost c_ref),
    else y_ref. Returns (label, record)."""
    from .b3 import solve_milp_hs
    t0 = time.time()
    fix, info = teacher_fixings(p, m, sysm, sc, ratio, lp)
    sol = solve_milp_hs(m, sc, time_limit=time_limit, mip_gap=gap, z_fix=fix or None, threads=1, incumbents=False)
    rec = dict(n_fixed=len(fix), milp_s=sol["time"], status=sol["status"], **info)
    if sol["u"] is None:
        rec.update(cost=np.inf, used="ref", total_s=time.time() - t0)
        return np.asarray(y_ref, np.int8), rec
    lp_sol = m.solve_dispatch(sc, sol["u"])
    rec.update(cost=float(lp_sol.obj), shed=float(lp_sol.shed), short=float(lp_sol.short), milp_obj=float(sol["obj"]))
    if lp_sol.obj < c_ref:
        rec.update(used="milp", total_s=time.time() - t0)
        return sol["u"].astype(np.int8), rec
    rec.update(used="ref", total_s=time.time() - t0)
    return np.asarray(y_ref, np.int8), rec


def build_gnnt(sysm, feat, T=T_H, seed=0, dilations=(1, 2, 4)):
    """otsl.m1x.CommitGNNT (dilations 1, 2, 4: every hour of a 12-hour row sees every other), as in otsl.m1x"""
    import torch
    from .m1x import CommitGNNT
    from .ucml import UCModel1
    torch.manual_seed(seed)
    return UCModel1(CommitGNNT(sysm, feat, T, dilations=dilations), feat, "gnn")


def sc_of(d, i):
    return UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])


# ============================================================================ tuning (Learning to Fix, Algorithm 1 + 2)
from .uc24ltf import GuardedLtFTuner, fix_to_masks, masks_to_fix  # noqa: E402


class HardTuner(GuardedLtFTuner):
    """guard-aware Learning to Fix (otsl.uc24ltf.GuardedLtFTuner: relaxation warm starts, guards inside the check, the
    cuts and the verification) with the guard chain adequacy + minimal min up/down conflict release (as otsl.m24)"""

    def guarded(self, i, off, on):
        key = (i, np.packbits(off).tobytes(), np.packbits(on).tobytes())
        if key in self._gcache:
            self.stats["n_guard_cached"] += 1
            return self._gcache[key]
        t0 = time.time()
        fix, _ = guard_chain(self.m, self.s, self.sc(i), masks_to_fix(off, on), lp=False)
        res = fix_to_masks(fix, *off.shape)
        self.stats["n_guard"] += 1
        self.stats["t_guard"] += time.time() - t0
        if len(self._gcache) > 20000:
            self._gcache.clear()
        self._gcache[key] = res
        return res
