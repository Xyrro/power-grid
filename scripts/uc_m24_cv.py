"""m24 study, validation-only diagnostic: does Learning to Fix's validated eps hold out of sample on 24 hours?

    taskset -c 1 python3 scripts/uc_m24_cv.py [--budget_min 60] [--tl 120]

Two-fold cross-validation on the 40 validation instances with full MILPs (uc24ltf_val.npz; folds: a fixed random
permutation, seed 0). For each fold and each score
  * m24: the 5-member temporal-GNN ensemble (data/generated/uc24_m24/probs/fin_ens_probs.npz), otsl.m24.M24Tuner
    (adequacy guard + minimal conflict release inside the check), as results/uc24/m24_tune_hg_fin.json;
  * lp:  the LP-relaxation values, otsl.uc24ltf.StartLtFTuner without guards, as results/uc24/base_tune_lp_1.json;
the thresholds are tuned (eps = 1 %, same settings) on the 20 instances of the other fold and applied to the 20
held-out instances as at test (m24: + adequacy guard, minimal conflict release and LP-relaxation guard; lp: no guard);
the reduced MILP is solved at 0.1 % (--tl s) and its cost compared with C* (the validation full MILP). In-sample
control: the 40-instance thresholds of both runs on all 40 instances, measured the same way.
Output: results/uc24/m24_cv_<score>_f<k>.json (tuning), results/uc24/m24_cv.jsonl (one line per solve); resume skips
finished tunings and solves.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import load_b3  # noqa: E402
from otsl.base import solve_reduced  # noqa: E402
from otsl.ltfx import fix_dict, fix_masks  # noqa: E402
from otsl.m24 import M24Tuner, guard_chain  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402
from otsl.uc24ltf import StartLtFTuner  # noqa: E402

U24, DATA, RES = "data/generated/uc24", "data/generated/uc24_m24", "results/uc24"
EPS = 0.01
OUT = os.path.join(RES, "m24_cv.jsonl")


def folds(n=40, seed=0):
    p = np.random.default_rng(seed).permutation(n)
    return [np.sort(p[:n // 2]), np.sort(p[n // 2:])]


def tune(score, k, train_idx, m, sysm, d, pi, budget_min):
    out = os.path.join(RES, f"m24_cv_{score}_f{k}.json")
    if os.path.exists(out):
        return json.load(open(out))
    sub = {kk: v[train_idx] for kk, v in d.items()}
    logf = open(out.replace(".json", ".log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    log(f"m24 cv: score {score} fold {k} (tuned on instances {train_idx.tolist()}) eps={EPS}")
    t0 = time.time()
    if score == "m24":
        tu = M24Tuner(m, sysm, sub, pi[train_idx], EPS, K_max=10, Q=20, check_tl=60.0, relax_tl=8.0, master_tl=120.0,
                      log=log, time_budget=budget_min * 60, start_tl=15.0)
    else:
        tu = StartLtFTuner(m, sysm, sub, pi[train_idx], EPS, K_max=10, Q=20, check_tl=60.0, relax_tl=8.0,
                           master_tl=120.0, log=log, time_budget=budget_min * 60)
    lo, hi = tu.run()
    conv = tu.history[-1].get("result") == "all instances feasible"
    off, on = fix_masks(pi[train_idx], lo, hi)
    res = dict(score=score, fold=k, train_idx=train_idx.tolist(), converged=conv, lo=lo.tolist(), hi=hi.tolist(),
               fixed_share_pre_guard=float((off | on).mean()), n_iter=len(tu.history), wall_s=time.time() - t0)
    if score == "m24":
        po, pn = tu.guarded_masks(lo, hi)
        res["fixed_share"] = float((po | pn).mean())
    else:
        res["fixed_share"] = res["fixed_share_pre_guard"]
    json.dump(res, open(out, "w"), indent=1)
    log(f"done: converged={conv} fixed {res['fixed_share'] * 100:.2f}% wall {res['wall_s']:.0f}s")
    print(f"tuned {score} fold {k}: converged={conv} fixed {res['fixed_share'] * 100:.2f}% wall {res['wall_s']:.0f}s",
          flush=True)
    return res


def measure(kind, score, k, idx, lo, hi, m, sysm, d, pi, tl, done):
    for i in idx:
        key = (kind, score, k, int(i))
        if key in done:
            continue
        sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        fx = fix_dict(pi[i], lo, hi)
        n_pre = len(fx)
        info = {}
        if score == "m24":
            fx, info = guard_chain(m, sysm, sc, fx, lp=True)
        r = solve_reduced(m, sc, fx, tl, 1e-3)
        c_star = float(d["obj"][i])
        rec = dict(kind=kind, score=score, fold=k, i=int(i), feasible=r["feasible"], obj=r["obj"], c_star=c_star,
                   rel_inc=(r["obj"] - c_star) / c_star if r["feasible"] else None, n_fixed_pre=n_pre,
                   n_fixed=len(fx), time=r["time"], status=r["status"], mip_gap=r["mip_gap"],
                   **{kk: v for kk, v in info.items() if kk.startswith("released")})
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        done.add(key)
        ri = rec["rel_inc"]
        print(f"{kind} {score} f{k} i={i}: " + (f"{ri * 100:+.3f}%" if ri is not None else "INFEASIBLE") +
              f" fixed {len(fx) / (24 * sysm.G) * 100:.1f}% {r['time']:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_min", type=float, default=60.0)
    ap.add_argument("--tl", type=float, default=120.0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=24, network=True)
    d = load_b3(os.path.join(U24, "uc24ltf_val.npz"))
    PI = {"m24": np.load(os.path.join(DATA, "probs", "fin_ens_probs.npz"))["val"].astype(np.float64),
          "lp": d["u_rel"].astype(np.float64)}
    done = set()
    if os.path.exists(OUT):
        for x in open(OUT):
            r = json.loads(x)
            done.add((r["kind"], r["score"], r["fold"], r["i"]))
    F = folds()
    for k in (0, 1):
        tr, ho = F[1 - k], F[k]
        for score in ("m24", "lp"):
            res = tune(score, k, tr, m, sysm, d, PI[score], a.budget_min)
            measure("heldout", score, k, ho, np.array(res["lo"]), np.array(res["hi"]), m, sysm, d, PI[score], a.tl, done)
    full = {"m24": json.load(open(os.path.join(RES, "m24_tune_hg_fin.json"))),
            "lp": json.load(open(os.path.join(RES, "base_tune_lp_1.json")))}
    for score in ("m24", "lp"):
        measure("insample", score, -1, range(40), np.array(full[score]["lo"]), np.array(full[score]["hi"]), m, sysm, d,
                PI[score], a.tl, done)
    print("done", flush=True)
