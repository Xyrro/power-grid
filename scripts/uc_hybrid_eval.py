"""Hybrid study, step 3: test evaluation, the full MILP and every reduced MILP back to back per instance.

    python3 scripts/uc_hybrid_eval.py --start 0 --n 60 --workers 1

Rules (every threshold, score and guard was fixed on validation before this script reads test_fresh):
  baselines   faithful LtF kNN eps=1% / BCE eps=1% / self-trained eps=10% (results/uc12/ltfx_tune_*.json);
              faithful LtF BCE eps=1% + guards at test only (no retuning; ablation);
              ours: combined pipeline 98 % (self-trained seed 0 + error cost + adequacy + rows + LP guard);
              ours: error-cost + adequacy guard 90 % (BCE seed 0)
  hybrids     every results/uc12/hybrid_tune_<tag>.json listed in --tags (default: all converged, non-smoke runs),
              score from results/uc12/hybrid_probs-style inputs recomputed per instance (timed), eq. (5) on the score,
              then the guards the rule was tuned with (in the worker, timed); "+lp" variants (--lp_tags) add the
              LP-relaxation guard at test time.
Per instance: full MILP (60 s, 0.1 %), then every reduced MILP (60 s, 0.1 %); an infeasible reduced MILP falls back
to the full MILP (recorded; the paper's metrics count it as infeasible). Method time = reduced MILP + inference
(GNN forward pass, kNN search, error-cost features and ensemble) + LP relaxation for the GNN / error-cost inputs +
guards. Output: results/uc12/hybrid_eval_test_fresh.jsonl (one record per instance and rule).
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
from otsl.combo import harm_from_file, harm_scores, rule_fixings  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import ALL_GUARDS, apply_guards, harm_to_score  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402
from uc_hybrid_prep import model_paths  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
TL, GAP = 60.0, 1e-3
BASE = {"faithful LtF kNN eps=1%": ("ltfx_tune_knn_1", "knn"),
        "faithful LtF BCE eps=1%": ("ltfx_tune_bce_1", "bce_s0"),
        "faithful LtF self-trained eps=10%": ("ltfx_tune_st_10", "st_s0")}
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
    if sol.u is None:
        fb = m.solve_uc(sc, time_limit=TL, mip_gap=GAP)
        rec.update(fallback=1, fb_obj=float(fb.obj), fb_time=float(fb.time), fb_shed=float(fb.shed),
                   fb_short=float(fb.short))
    return rec


def _job(args):
    i, sc, specs = args
    G, T = len(sc.u0), sc.load.shape[0]
    out = []
    full = _solve(sc, {})
    out.append(dict(i=i, rule="full MILP", fixed_share=0.0, fixed_pre_guard=0.0, pre_s=0.0, relax_s=0.0, guard_s=0.0, **full))
    t0 = time.time()
    _W["m"].solve_dispatch(sc, None, relax=True)          # LP relaxation: input of the GNN / error-cost features
    relax_s = time.time() - t0
    for name, fix, pre_s, gnn, guards in specs:
        n_pre = len(fix)
        extra = {}
        if guards:
            fix, extra = apply_guards(_W["m"], _W["s"], sc, fix, guards)
        r = _solve(sc, fix)
        out.append(dict(i=i, rule=name, fixed_share=len(fix) / (T * G), fixed_pre_guard=n_pre / (T * G), pre_s=pre_s,
                        relax_s=relax_s if gnn else 0.0, guard_s=extra.get("guard_s", 0.0),
                        **{k: v for k, v in extra.items() if k != "guard_s"}, **r))
    return out


def hybrid_rules(tags, lp_tags):
    """{rule name: (score key, lo, hi, guards)}"""
    out = {}
    files = sorted(glob.glob(os.path.join(OUT, "hybrid_tune_*.json")))
    for f in files:
        r = json.load(open(f))
        if tags and r["tag"] not in tags:
            continue
        if not tags and ("smoke" in r["tag"] or not r["converged"]):
            continue
        g = tuple(r["guards"])
        nm = f"hybrid {r['tag']}"
        out[nm] = (r["score"], np.array(r["lo"]), np.array(r["hi"]), g)
        if r["tag"] in lp_tags:
            out[nm + " +lp"] = (r["score"], np.array(r["lo"]), np.array(r["hi"]), ALL_GUARDS)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tags", default="", help="comma-separated hybrid tags (default: all converged runs)")
    ap.add_argument("--lp_tags", default="", help="hybrid tags also evaluated with the LP-relaxation guard at test")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    tr = load(os.path.join(ROOT, "train.npz"))
    d = load(os.path.join(ROOT, f"{a.split}.npz"))
    idx = list(range(a.start, min(a.start + a.n, len(d["load"]))))
    one = lambda i: {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}
    tags = [t for t in a.tags.split(",") if t]
    lp_tags = [t for t in a.lp_tags.split(",") if t]
    hyb = hybrid_rules(tags, lp_tags)
    norms = json.load(open(os.path.join(OUT, "hybrid_probs.json")))["harm_norm_logh_mean_sd"]
    # ---------------- models needed
    srcs = {"bce_s0", "st_s0"} | {k.replace("harm_", "") for k, *_ in hyb.values()}
    paths = model_paths()
    trs = strip(tr)
    gnn = {k: load_model1(os.path.join(OUT, paths[k]), sysm, trs, 12) for k in sorted(srcs)}
    knn = KNNProb(sysm, tr, k=50)
    harms = {k: harm_from_file(os.path.join(OUT, f"combo_harm_s{k.split('_s')[1]}.pt")) for k in sorted(srcs)}
    ff = FixFeaturizer(sysm, 12)
    harm_scores(harms["bce_s0"], ff, gnn["bce_s0"].predict(one(idx[0])), one(idx[0]), [0])   # warm-up
    base = {k: (json.load(open(os.path.join(OUT, f + ".json"))), src) for k, (f, src) in BASE.items()}
    chk = np.load(os.path.join(OUT, "hybrid_probs.npz"))
    jobs = []
    for i in idx:
        di = one(i)
        sc = scenario_from(d, i)
        P, Tinf, H, Th = {}, {}, {}, {}
        t0 = time.time(); P["knn"] = knn.predict(di)[0]; Tinf["knn"] = time.time() - t0
        for k, m1 in gnn.items():
            t0 = time.time(); P[k] = m1.predict(di)[0].astype(np.float64); Tinf[k] = time.time() - t0
            t0 = time.time(); H[k] = harm_scores(harms[k], ff, P[k][None], d, [i])[0]; Th[k] = time.time() - t0
            if a.split == "test_fresh" and f"{k}_tf" in chk.files:
                assert np.abs(P[k] - chk[f"{k}_tf"][i]).max() < 1e-6 and np.abs(H[k] / chk[f"harm_{k}_tf"][i] - 1).max() < 1e-4
        specs = []
        for name, (r, src) in base.items():
            t0 = time.time(); fix = fix_dict(P[src], np.array(r["lo"]), np.array(r["hi"])); dt = time.time() - t0
            specs.append((name, fix, Tinf[src] + dt, src != "knn", ()))
        r = base["faithful LtF BCE eps=1%"][0]
        t0 = time.time(); fix = fix_dict(P["bce_s0"], np.array(r["lo"]), np.array(r["hi"])); dt = time.time() - t0
        specs.append(("faithful LtF BCE eps=1% + guards at test", fix, Tinf["bce_s0"] + dt, True, ALL_GUARDS))
        t0 = time.time(); fix = rule_fixings("harm", 0.98, P["st_s0"], sc, sysm, H["st_s0"]); dt = time.time() - t0
        specs.append(("ours: combined st+error-cost+adequacy+LP guard 98%", fix, Tinf["st_s0"] + Th["st_s0"] + dt, True,
                      ("rows", "lp")))
        t0 = time.time(); fix = rule_fixings("harm", 0.90, P["bce_s0"], sc, sysm, H["bce_s0"]); dt = time.time() - t0
        specs.append(("ours: error-cost + adequacy guard 90% (BCE)", fix, Tinf["bce_s0"] + Th["bce_s0"] + dt, True, ()))
        for name, (key, lo, hi, guards) in hyb.items():
            t0 = time.time()
            if key.startswith("harm_"):
                src = key[5:]
                s_ = harm_to_score(H[src], P[src], *norms[src])
                pre = Tinf[src] + Th[src]
            else:
                src, s_ = key, P[key]
                pre = Tinf[src]
            fix = fix_dict(s_, lo, hi)
            specs.append((name, fix, pre + time.time() - t0, True, guards))
        jobs.append((i, sc, specs))
    names = [s[0] for s in jobs[0][2]]
    print(f"{len(names)} rules per instance:", names, flush=True)
    if a.dry:
        for nm, fix, pre, g_, gu in jobs[0][2]:
            print(f"  {nm:60s} fixed (pre-guard) {len(fix) / (12 * sysm.G) * 100:5.1f}%  pre {pre * 1000:.1f} ms  guards {gu}")
        sys.exit(0)
    path = os.path.join(OUT, f"hybrid_eval_{a.split}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["i"] for line in open(path)}
    jobs = [j for j in jobs if j[0] not in done]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(path, "a") as f:
        for k, recs in enumerate(pool.imap(_job, jobs, chunksize=1)):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} full {recs[0]['time']:.1f}s, reduced "
                  f"total {sum(r['time'] for r in recs[1:]):.1f}s, infeasible "
                  f"{[r['rule'] for r in recs if not r['feasible']]}", flush=True)
