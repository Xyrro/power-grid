"""Learning to Fix (faithful), step 3: test evaluation, full MILP and every reduced MILP back to back per instance.

    python3 scripts/uc_ltfx_eval.py --split test_fresh --n 60 --workers 2

Rules (thresholds were fixed on validation by scripts/uc_ltfx_tune.py):
  <model> ltf eps=<e>%   suboptimality-constrained thresholds (Algorithm 1 + 2), model in knn / st / rl / bce
  <model> const r        constant thresholds [r, 1 - r] (eq. 6)
  <model> worst          worst-case-misprediction thresholds (eq. 7)
  knn tau=0.5            standard binary classification (everything fixed)
  ours: harm+guard 90 %  learned error-cost ranking + adequacy guard (BCE probabilities, harm model seed 0)
        combo st+harm+lp 95 / 98 %   self-trained probabilities + error-cost ranking + adequacy + LP-relaxation guard
Per instance and rule: reduced MILP (60 s, 0.1 %), with a fallback to the full MILP if the fixings make it infeasible
(recorded separately; the paper's metrics treat it as infeasible). Times: the method's time includes inference (kNN /
GNN forward pass, the LP relaxation for the GNN features, the error-cost model), the guards, and the reduced MILP.
Output: results/uc12/ltfx_eval_<split>.jsonl (one record per instance and rule).
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import harm_from_file, harm_scores, rule_fixings, solver_guards  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
TL, GAP = 60.0, 1e-3
GNNS = {"bce": "uc_model1_4.pt", "st": "selftrain_m3.pt", "rl": "uc_model1_rl.pt"}

_W = {}


def _init():
    torch.set_num_threads(1)
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=12, network=True))


def _solve(sc, fix):
    m = _W["m"]
    sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP, z_fix=fix or None)
    rec = dict(feasible=sol.u is not None, obj=float(sol.obj), time=float(sol.time), status=sol.status,
               mip_gap=float(sol.gap), shed=float(sol.shed), short=float(sol.short), fallback=0)
    if sol.u is None:                       # fixings conflict with min up/down: full MILP as fallback (our convention)
        fb = m.solve_uc(sc, time_limit=TL, mip_gap=GAP)
        rec.update(fallback=1, fb_obj=float(fb.obj), fb_time=float(fb.time), fb_shed=float(fb.shed),
                   fb_short=float(fb.short))
    return rec


def _job(args):
    i, sc, specs = args
    out = []
    full = _solve(sc, {})
    out.append(dict(i=i, rule="full MILP", fixed_share=0.0, pre_s=0.0, relax_s=0.0, guard_s=0.0, **full))
    t0 = time.time()
    _W["m"].solve_dispatch(sc, None, relax=True)          # LP relaxation: input of the GNN / error-cost features
    relax_s = time.time() - t0
    for name, fix, pre_s, gnn, lpg in specs:
        guard_s, extra = 0.0, {}
        if lpg:
            fix, extra, guard_s = solver_guards(_W["m"], _W["s"], sc, fix)
        r = _solve(sc, fix)
        out.append(dict(i=i, rule=name, fixed_share=len(fix) / (sc.load.shape[0] * len(sc.u0)), pre_s=pre_s,
                        relax_s=relax_s if gnn else 0.0, guard_s=guard_s, **extra, **r))
    return out


def load_thresholds():
    """{rule name: (model, lo, hi)} from the tuning outputs"""
    th = {}
    for f in sorted(glob.glob(os.path.join(OUT, "ltfx_tune_*.json"))):
        r = json.load(open(f))
        nm = f"{r['model']} ltf eps={r['eps'] * 100:g}%" + ("" if r["converged"] else " (not converged)")
        th[nm] = (r["model"], np.array(r["lo"]), np.array(r["hi"]))
    cheap = json.load(open(os.path.join(OUT, "ltfx_cheap_thresholds.json")))
    for k, v in cheap.items():
        model, kind = k.split("_", 1)
        th[f"{model} {kind.replace('const', 'const ')}"] = (model, np.array(v["lo"]), np.array(v["hi"]))
    return th


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cheap_models", default="knn,st", help="models for which constant / worst-case rows are run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true", help="build the rules and stop (no MILPs)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    tr = load(os.path.join(ROOT, "train.npz"))
    d = load(os.path.join(ROOT, f"{a.split}.npz"))
    n = min(a.n, len(d["load"]))
    one = lambda i: {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}
    # ---------------- probabilities, per instance, timed (inference cost of each model)
    knn = KNNProb(sysm, tr, k=50)
    trs = strip(tr)
    gnn = {k: load_model1(os.path.join(OUT, f), sysm, trs, 12) for k, f in GNNS.items()}
    P, T_inf = {k: [] for k in ["knn"] + list(GNNS)}, {k: [] for k in ["knn"] + list(GNNS)}
    for i in range(n):
        di = one(i)
        t0 = time.time(); P["knn"].append(knn.predict(di)[0]); T_inf["knn"].append(time.time() - t0)
        for k, m1 in gnn.items():
            t0 = time.time(); P[k].append(m1.predict(di)[0].astype(np.float64)); T_inf[k].append(time.time() - t0)
    P = {k: np.array(v) for k, v in P.items()}
    chk = np.load(os.path.join(OUT, "ltfx_probs.npz"))
    if a.split == "test_fresh":
        for k in P:
            print(f"max |p - ltfx_probs| {k}: {np.abs(P[k] - chk[f'{k}_tf'][:n]).max():.2e}", flush=True)
    th = load_thresholds()
    cheap_models = a.cheap_models.split(",")
    ff = FixFeaturizer(sysm, 12)
    harm = harm_from_file(os.path.join(OUT, "combo_harm_s0.pt"))
    harm_scores(harm, ff, P["bce"][0:1], d, [0])          # warm-up (first-call overhead not billed to instance 0)
    jobs = []
    for i in range(n):
        sc = scenario_from(d, i)
        specs = []
        for name, (model, lo, hi) in th.items():
            if " ltf " not in name and model not in cheap_models:
                continue
            t0 = time.time(); fix = fix_dict(P[model][i], lo, hi); dt = time.time() - t0
            specs.append((name, fix, T_inf[model][i] + dt, model != "knn", False))
        t0 = time.time(); fix = fix_dict(P["knn"][i], np.full(sysm.G, 0.5), np.full(sysm.G, 0.5))
        specs.append(("knn tau=0.5", fix, T_inf["knn"][i] + time.time() - t0, False, False))
        t0 = time.time(); h = harm_scores(harm, ff, P["bce"][i:i + 1], d, [i])[0]; th_b = time.time() - t0
        t0 = time.time(); fix = rule_fixings("harm", 0.90, P["bce"][i], sc, sysm, h)
        specs.append(("ours: error-cost + adequacy guard 90% (BCE)", fix, T_inf["bce"][i] + th_b + time.time() - t0, True, False))
        t0 = time.time(); h = harm_scores(harm, ff, P["st"][i:i + 1], d, [i])[0]; th_s = time.time() - t0
        for q in (0.95, 0.98):
            t0 = time.time(); fix = rule_fixings("harm", q, P["st"][i], sc, sysm, h)
            specs.append((f"ours: combined st+error-cost+adequacy+LP guard {q * 100:g}%", fix,
                          T_inf["st"][i] + th_s + time.time() - t0, True, True))
        jobs.append((i, sc, specs))
    names = [s[0] for s in jobs[0][2]]
    print(f"{len(names)} rules per instance:", names, flush=True)
    if a.dry:
        for nm, fix, pre, g_, l_ in jobs[0][2]:
            print(f"  {nm:55s} fixed {len(fix) / (12 * sysm.G) * 100:5.1f}%  pre {pre * 1000:.1f} ms")
        sys.exit(0)
    path = os.path.join(OUT, f"ltfx_eval_{a.split}{a.tag}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["i"] for line in open(path)}
    jobs = [j for j in jobs if j[0] not in done]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(path, "a") as f:
        for k, recs in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} full {recs[0]['time']:.1f}s, reduced "
                  f"total {sum(r['time'] for r in recs[1:]):.1f}s, infeasible "
                  f"{[r['rule'] for r in recs if not r['feasible']]}", flush=True)
