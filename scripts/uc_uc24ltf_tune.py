"""uc24ltf, step 3: suboptimality-constrained thresholds (Algorithm 1 + 2 of Learning to Fix) on the uc24
validation instances with full MILPs (data/generated/uc24/uc24ltf_val.npz).

    python3 scripts/uc_uc24ltf_tune.py --jobs gnn:0.05,gnn:0.05:g,knn:0.01 --relax_tl 10 --budget_min 25

A job is <model>:<eps>[:g]; model in knn / bce / rl / gnn (= the GNN chosen in results/uc24/uc24ltf_probs.json);
unguarded jobs use otsl.uc24ltf.StartLtFTuner (better relaxation-MILP warm starts;
--plain_start: otsl.ltfx.LtFTuner as on uc12); ":g" = guard-aware tuning (otsl.uc24ltf.GuardedLtFTuner: adequacy guard + soft LP-relaxation guard applied to the
fixings inside every check, cut and verification). Jobs run one after the other in this
process (one core; HiGHS single-threaded). Per job: results/uc24/uc24ltf_tune_<model>[_g]_<eps%>.{json,log}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import load_b3  # noqa: E402
from otsl.ltfx import LtFTuner, fix_masks  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.uc24ltf import GuardedLtFTuner, StartLtFTuner  # noqa: E402

ROOT, OUT = os.path.join("data", "generated", "uc24"), os.path.join("results", "uc24")
FIELDS = ("load", "avail", "u0", "sr", "obj", "u", "day", "start", "gap", "time")


def tag(name, eps, guarded):
    return f"{name}{'_g' if guarded else ''}_{eps * 100:g}"


def run_job(name, eps, guarded, cfg, sysm, m):
    t_cpu0, t0 = time.process_time(), time.time()
    d0 = load_b3(cfg["val"])
    n = min(cfg["n_val"], len(d0["obj"]))
    d = {k: d0[k][:n] for k in FIELDS}
    P = np.load(os.path.join(OUT, "uc24ltf_probs.npz"))
    pi = P[f"{name}_va"][:n]
    logf = open(os.path.join(OUT, f"uc24ltf_tune_{tag(name, eps, guarded)}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    log(f"LtF tuning (uc24) {name} eps={eps} guarded={guarded} n_val={n} K_max={cfg['K_max']} relax_tl={cfg['relax_tl']} "
        f"check_tl={cfg['check_tl']} Q={cfg['Q']} budget {cfg['budget_s']:.0f}s")
    cls = GuardedLtFTuner if guarded else (LtFTuner if cfg.get("plain_start") else StartLtFTuner)
    tu = cls(m, sysm, d, pi, eps, K_max=cfg["K_max"], Q=cfg["Q"], check_tl=cfg["check_tl"], relax_tl=cfg["relax_tl"],
             master_tl=cfg["master_tl"], log=log, time_budget=cfg["budget_s"])
    lo, hi = tu.run()
    off, on = fix_masks(pi, lo, hi)
    conv = tu.history[-1].get("result") == "all instances feasible"
    t_ver = time.time()
    ver = tu.verify(lo, hi)
    log(f"verification (witness schedules priced by the dispatch LP): max rel. cost increase {np.max(ver) * 100:.4f}% "
        f"(eps {eps * 100:g}%), instances above eps: {int((ver > eps * (1 + 1e-6)).sum())}, {time.time() - t_ver:.0f}s")
    res = dict(model=name, eps=eps, guarded=guarded, converged=conv, lo=lo.tolist(), hi=hi.tolist(), n_val=n,
               val_fixed_share=float((off | on).mean()), val_fixed_off=float(off.mean()), val_fixed_on=float(on.mean()),
               n_iter=len(tu.history), n_cut_sets=len(tu.master.cuts),
               verify_rel_increase=ver.tolist(), verify_max=float(np.max(ver)),
               verify_ok=bool(np.all(ver <= eps * (1 + 1e-6))), n_verify_fail=int((ver > eps * (1 + 1e-6)).sum()),
               stats=tu.stats, history=tu.history, cfg=cfg, wall_s=time.time() - t0, cpu_s=time.process_time() - t_cpu0,
               failed_instances=sorted({h["failed"] for h in tu.history if "failed" in h}))
    if guarded:
        go, gn = tu.guarded_masks(lo, hi)
        res.update(val_fixed_share_guarded=float((go | gn).mean()), val_fixed_off_guarded=float(go.mean()),
                   val_fixed_on_guarded=float(gn.mean()))
    with open(os.path.join(OUT, f"uc24ltf_tune_{tag(name, eps, guarded)}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}%"
        + (f" (after guards {res['val_fixed_share_guarded'] * 100:.2f}%)" if guarded else "")
        + f" wall {res['wall_s']:.0f}s cpu {res['cpu_s']:.0f}s")
    logf.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--val", default=os.path.join(ROOT, "uc24ltf_val.npz"))
    ap.add_argument("--n_val", type=int, default=10 ** 6)
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--Q", type=int, default=20)
    ap.add_argument("--relax_tl", type=float, default=10.0)
    ap.add_argument("--check_tl", type=float, default=60.0)
    ap.add_argument("--master_tl", type=float, default=120.0)
    ap.add_argument("--budget_min", type=float, default=25.0)
    ap.add_argument("--plain_start", action="store_true", help="otsl.ltfx.LtFTuner's label start only")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    cfg = dict(val=a.val, n_val=a.n_val, K_max=a.K_max, Q=a.Q, relax_tl=a.relax_tl, check_tl=a.check_tl,
               master_tl=a.master_tl, budget_s=a.budget_min * 60, plain_start=a.plain_start)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=24, network=True)
    choice = json.load(open(os.path.join(OUT, "uc24ltf_probs.json"))).get("gnn_choice", "bce")
    t0 = time.time()
    for j in a.jobs.split(","):
        parts = j.split(":")
        name = choice if parts[0] == "gnn" else parts[0]
        eps, guarded = float(parts[1]), len(parts) > 2 and parts[2] == "g"
        r = run_job(name, eps, guarded, cfg, sysm, m)
        print(f"[{time.time() - t0:.0f}s] done {tag(name, eps, guarded)}: converged={r['converged']} val fixed "
              f"{r['val_fixed_share'] * 100:.2f}%" + (f" guarded {r['val_fixed_share_guarded'] * 100:.2f}%" if guarded else "")
              + f" iters {r['n_iter']} wall {r['wall_s']:.0f}s verify_fail {r['n_verify_fail']}", flush=True)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
