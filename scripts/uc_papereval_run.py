"""Rows of the Learning to Fix paper's Table I (Fritz et al., arXiv 2609.39396) that our earlier studies did not run,
on the 12-hour fresh test set (B2, uc12, test_fresh).

    python scripts/uc_papereval_run.py --n_fix 60 --n_rank 120 --workers 2

Per instance i < n_fix, in one worker and back to back (60 s limit, 0.1 % gap, as in the dataset):
  1. the full MILP (its MIP gap is recorded, so its dual bound is obj * (1 - gap));
  2. reduced MILPs with constant confidence thresholds (the paper's Sec. III-B1): fix u[t, g] := round(p) where
     p <= tau_lo or p >= 1 - tau_lo, for tau_lo in {0.1, 0.05, 0.01}, and the hard threshold tau = 0.5 (everything
     fixed), on two probability models:
       knn50  kNN classifier with k = 50 and inverse-distance weights (the paper's eq. 13), features of
              otsl.ltf.KNNCommit(kind="sys"): per-hour area loads, per-hour renewable availability by type,
              initial status; labels canonicalised inside groups of identical units (as in our reconstruction);
       bce    the MILP-label BCE GNN (uc_model1_4.pt; RACLearn's predictor), probabilities of ltf_probs.npz;
  3. the cost-ranked kNN (Pineda & Morales 2022, the paper's ref. [28] and its "cost-ranked" baseline): the 50
     nearest training instances by the same distance; each neighbour's MILP schedule (raw, not canonicalised) is
     fixed and priced with the dispatch LP of this instance; a schedule that violates min up/down given this
     instance's initial status makes the LP infeasible and is discarded; the cheapest feasible schedule is kept.
Instances n_fix <= i < n_rank get only step 3 (no full MILP).

No fallback re-solve: a reduced problem without a solution is recorded as status "infeasible" (the paper counts it
as infeasible). Times: solver wall time of each reduced MILP; for the cost-ranked kNN the wall time of every LP
(bounds + solve) and the per-instance kNN inference time (neighbour search, measured in the main process).
Output: results/papereval_fresh_runs.jsonl (one record per solve / rule).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from otsl.ltf import KNNCommit  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

TL, GAP, T = 60.0, 1e-3, 12
TAUS = (0.5, 0.1, 0.05, 0.01)
K_RANK = 50
OUT = os.path.join(ROOT, "results", "papereval_fresh_runs.jsonl")
_W = {}


def _init():
    import scipy.sparse as sp
    s = load_rts_gmlc()
    m = UCModel(s, T=T, network=True)
    lo, hi = m.lo0, m.hi0
    eq = lo == hi
    iu, il = ~eq & np.isfinite(hi), ~eq & np.isfinite(lo)
    _W.update(s=s, m=m, mats=(eq, iu, il, sp.vstack([m.A[iu], -m.A[il]]).tocsr(), m.A[eq].tocsr()))


def dispatch(m, mats, sc, u):
    """fixed-commitment dispatch LP (same model as UCModel.solve_dispatch, matrices cached);
    returns (status, obj, shed, short, seconds incl. building the bounds)"""
    from scipy.optimize import linprog
    t0 = time.perf_counter()
    lo, hi, lb, ub = m._rhs_bounds(sc, u)
    eq, iu, il, A_ub, A_eq = mats
    assert (eq == (lo == hi)).all()
    res = linprog(m.c, A_ub=A_ub, b_ub=np.r_[hi[iu], -lo[il]], A_eq=A_eq, b_eq=lo[eq], bounds=np.c_[lb, ub],
                  method="highs")
    dt = time.perf_counter() - t0
    if res.status != 0:
        return "infeasible", float("inf"), float("nan"), float("nan"), dt
    x = res.x
    get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T]
    return "optimal", float(res.fun), float(get("shed").sum() + get("spill").sum()), float(get("short").sum()), dt


def _solve(sc, fix):
    sol = _W["m"].solve_uc(sc, time_limit=TL, mip_gap=GAP, z_fix=fix or None)
    if sol.u is None:
        return dict(obj=None, shed=None, short=None, time=float(sol.time), status="infeasible", gap=None,
                    units_on=None)
    return dict(obj=float(sol.obj), shed=float(sol.shed), short=float(sol.short), time=float(sol.time),
                status=sol.status, gap=float(sol.gap), units_on=float(sol.u.sum() / T))


def _job(args):
    i, sc, specs, full, cand, knn_s = args
    out = []
    if full:
        out.append(dict(i=i, rule="full", fixed_share=0.0, loadavg=os.getloadavg()[0], **_solve(sc, {})))
    for rule, fix in specs:
        fix = {(int(t), int(g)): int(v) for t, g, v in fix}
        out.append(dict(i=i, rule=rule, fixed_share=len(fix) / (T * len(sc.u0)), loadavg=os.getloadavg()[0],
                        **_solve(sc, fix)))
    if cand is not None:
        res = [dispatch(_W["m"], _W["mats"], sc, u) for u in cand]
        cost = np.array([r[1] for r in res])
        ok = np.isfinite(cost)
        b = int(np.argmin(np.where(ok, cost, np.inf))) if ok.any() else -1
        out.append(dict(i=i, rule="knn_costrank", fixed_share=1.0, loadavg=os.getloadavg()[0],
                        status="optimal" if b >= 0 else "infeasible",
                        obj=float(cost[b]) if b >= 0 else None, shed=res[b][2] if b >= 0 else None,
                        short=res[b][3] if b >= 0 else None, best_rank=b, n_feasible=int(ok.sum()),
                        lp_s=[r[4] for r in res], costs=[float(c) if np.isfinite(c) else None for c in cost],
                        sheds=[r[2] for r in res], shorts=[r[3] for r in res], knn_s=knn_s,
                        time=float(sum(r[4] for r in res) + knn_s)))
    return out


def threshold_fix(p, tau):
    """{(t, g): round(p)} where min(p, 1 - p) <= tau (tau = 0.5: every decision)"""
    yhat = (p > 0.5).astype(int)
    sel = np.minimum(p, 1 - p) <= tau if tau < 0.5 else np.ones_like(p, bool)
    tt, gg = np.where(sel)
    return [(int(t), int(g), int(yhat[t, g])) for t, g in zip(tt, gg)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fix", type=int, default=60)
    ap.add_argument("--n_rank", type=int, default=120)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.chdir(ROOT)
    sysm = load_rts_gmlc()
    tr = load("data/generated/uc12/train.npz")
    d = load("data/generated/uc12/test_fresh.npz")
    knn = KNNCommit(sysm, tr, k=K_RANK, kind="sys", w_u0=1.0, weighted=True, canonical=True)
    p_bce = np.load("results/uc12/ltf_probs.npz")["bce_tf"].astype(np.float64)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(json.loads(line)["i"])
    jobs = []
    n = max(a.n_fix, a.n_rank)
    for i in range(n):
        if i in done:
            continue
        one = {k: v[i:i + 1] for k, v in d.items()}
        t0 = time.perf_counter()
        nb, _ = knn.neighbours(one)                 # kNN inference (neighbour search) for this instance
        p_knn = knn.predict(one)[0].astype(np.float64)
        knn_s = time.perf_counter() - t0
        sc = scenario_from(d, i)
        specs = []
        if i < a.n_fix:
            for tau in TAUS:
                specs.append((f"knn50@{tau}", threshold_fix(p_knn, tau)))
            for tau in TAUS:
                specs.append((f"bce@{tau}", threshold_fix(p_bce[i], tau)))
        cand = tr["u"][nb[0]] if i < a.n_rank else None
        jobs.append((i, sc, specs, i < a.n_fix, cand, knn_s))
    print(f"{len(jobs)} instances to run ({len(done)} done)", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(OUT, "a") as f:
            for k, recs in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                msg = " ".join(f"{r['rule']}:{r['time']:.1f}s" + ("" if r["obj"] is None else
                               f"/{(r['obj'] / d['obj'][r['i']] - 1) * 100:+.2f}%") for r in recs)
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} {msg}", flush=True)
