"""Learning to Fix (faithful), step 2: suboptimality-constrained thresholds by Algorithm 1 + 2 on the validation
instances (val + val_extra), for the kNN and our GNN probabilities, and the cheap thresholds (constant, worst-case).

    python3 scripts/uc_ltfx_tune.py --jobs knn:0.01,bce:0.01 --workers 2 [--n_val 180] [--relax_tl 10]
    python3 scripts/uc_ltfx_tune.py --plan default      # GNN eps = 1 % runs, kNN eps = 1/5/10 %, then the best GNN
                                                        # (largest val fixed share at eps = 1 %) at eps = 5 / 10 %

Each job runs Algorithm 1 in its own worker process (one core; HiGHS single-threaded). Per job:
results/uc12/ltfx_tune_<model>_<eps>.json (thresholds, iteration log, solve statistics, verification) and
results/uc12/ltfx_tune_<model>_<eps>.log.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.ltfx import LtFTuner, constant_thresholds, fix_masks, worst_case_thresholds  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import canonical_labels  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
FIELDS = ("load", "avail", "u0", "sr", "obj", "u", "day", "start", "gap", "time")


def val_data(n_val):
    va = load(os.path.join(ROOT, "val.npz"))
    parts = [va]
    if os.path.exists(os.path.join(ROOT, "val_extra.npz")):
        parts.append(load(os.path.join(ROOT, "val_extra.npz")))
    d = {k: np.concatenate([p[k] for p in parts])[:n_val] for k in FIELDS}
    return d


def val_probs(name, n_val):
    P = np.load(os.path.join(OUT, "ltfx_probs.npz"))
    parts = [P[f"{name}_va"]] + ([P[f"{name}_vx"]] if f"{name}_vx" in P.files else [])
    return np.concatenate(parts)[:n_val]


def tag(name, eps):
    return f"{name}_{eps * 100:g}"


def run_job(args):
    name, eps, cfg = args
    os.chdir(os.path.dirname(HERE))
    t_cpu0, t0 = time.process_time(), time.time()
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    d = val_data(cfg["n_val"])
    pi = val_probs(name, cfg["n_val"])
    logf = open(os.path.join(OUT, f"ltfx_tune_{tag(name, eps)}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    log(f"LtF tuning {name} eps={eps} n_val={len(pi)} K_max={cfg['K_max']} relax_tl={cfg['relax_tl']} "
        f"check_tl={cfg['check_tl']} Q={cfg['Q']}")
    tu = LtFTuner(m, sysm, d, pi, eps, K_max=cfg["K_max"], Q=cfg["Q"], check_tl=cfg["check_tl"],
                  relax_tl=cfg["relax_tl"], master_tl=cfg["master_tl"], log=log, time_budget=cfg["budget_s"])
    lo, hi = tu.run()
    off, on = fix_masks(pi, lo, hi)
    conv = tu.history[-1].get("result") == "all instances feasible"
    t_ver = time.time()
    ver = tu.verify(lo, hi)
    log(f"verification (witness schedules priced by the dispatch LP): max rel. cost increase {np.max(ver) * 100:.4f}% "
        f"(eps {eps * 100:g}%), instances above eps: {int((ver > eps * (1 + 1e-6)).sum())}, {time.time() - t_ver:.0f}s")
    res = dict(model=name, eps=eps, converged=conv, lo=lo.tolist(), hi=hi.tolist(), n_val=len(pi),
               val_fixed_share=float((off | on).mean()), val_fixed_off=float(off.mean()), val_fixed_on=float(on.mean()),
               n_iter=len(tu.history), n_cut_sets=len(tu.master.cuts),
               verify_rel_increase=ver.tolist(), verify_max=float(np.max(ver)), verify_ok=bool(np.all(ver <= eps * (1 + 1e-6))),
               stats=tu.stats, history=tu.history, cfg=cfg, wall_s=time.time() - t0,
               cpu_s=time.process_time() - t_cpu0)
    with open(os.path.join(OUT, f"ltfx_tune_{tag(name, eps)}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% wall {res['wall_s']:.0f}s "
        f"cpu {res['cpu_s']:.0f}s")
    logf.close()
    return name, eps, res["val_fixed_share"], conv, res["wall_s"]


def cheap_thresholds(n_val):
    """constant (eq. 6) and worst-case (eq. 7) thresholds of every model on the same validation instances"""
    sysm = load_rts_gmlc()
    d = val_data(n_val)
    y = canonical_labels(sysm, d["u"], d["u0"])
    out = {}
    for name in ("knn", "bce", "st", "rl"):
        pi = val_probs(name, n_val)
        lo, hi = worst_case_thresholds(pi, y)
        off, on = fix_masks(pi, lo, hi)
        out[f"{name}_worst"] = dict(lo=lo.tolist(), hi=hi.tolist(), val_fixed_share=float((off | on).mean()))
        for r in (0.1, 0.05, 0.01):
            lo, hi = constant_thresholds(sysm.G, r)
            off, on = fix_masks(pi, lo, hi)
            out[f"{name}_const{r:g}"] = dict(lo=lo.tolist(), hi=hi.tolist(), val_fixed_share=float((off | on).mean()))
    with open(os.path.join(OUT, "ltfx_cheap_thresholds.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=10 ** 6)
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--Q", type=int, default=20)
    ap.add_argument("--relax_tl", type=float, default=10.0)
    ap.add_argument("--check_tl", type=float, default=60.0)
    ap.add_argument("--master_tl", type=float, default=120.0)
    ap.add_argument("--budget_min", type=float, default=150.0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    cfg = dict(n_val=a.n_val, K_max=a.K_max, Q=a.Q, relax_tl=a.relax_tl, check_tl=a.check_tl, master_tl=a.master_tl,
               budget_s=a.budget_min * 60)
    cheap = cheap_thresholds(a.n_val)
    print("cheap thresholds, val fixed share:", {k: round(v["val_fixed_share"], 4) for k, v in cheap.items()}, flush=True)
    t0 = time.time()
    done = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        if a.plan == "default":
            first = [("bce", 0.01), ("st", 0.01), ("rl", 0.01), ("knn", 0.01), ("knn", 0.05), ("knn", 0.10)]
        else:
            first = [(j.split(":")[0], float(j.split(":")[1])) for j in a.jobs.split(",") if j]
        pending = [pool.apply_async(run_job, ((n, e, cfg),)) for n, e in first]
        gnn_best_submitted = a.plan != "default"
        while pending:
            time.sleep(5)
            still = []
            for p in pending:
                if p.ready():
                    r = p.get()
                    done.append(r)
                    print(f"[{time.time() - t0:.0f}s] done {r[0]} eps={r[1]}: val fixed {r[2] * 100:.2f}% "
                          f"converged={r[3]} wall {r[4]:.0f}s", flush=True)
                else:
                    still.append(p)
            pending = still
            if not gnn_best_submitted:
                g1 = {r[0]: r[2] for r in done if r[1] == 0.01 and r[0] in ("bce", "st", "rl") and r[3]}
                if len(g1) == 3 or (len([r for r in done if r[1] == 0.01 and r[0] in ("bce", "st", "rl")]) == 3):
                    best = max(g1, key=g1.get) if g1 else "rl"
                    print(f"best GNN at eps = 1 % (val fixed share, converged runs only): {best} {g1}", flush=True)
                    json.dump({"best_gnn": best, "val_fixed_share_eps1": g1,
                               "rule": "largest validation fixed share at eps = 1 % among converged runs"},
                              open(os.path.join(OUT, "ltfx_best_gnn.json"), "w"), indent=1)
                    pending += [pool.apply_async(run_job, ((best, e, cfg),)) for e in (0.05, 0.10)]
                    gnn_best_submitted = True
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
