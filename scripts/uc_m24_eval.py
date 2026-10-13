"""m24 study, test: every run of a uc24 test instance back to back in one process on core 1 (paired timing).

    taskset -c 1 python3 scripts/uc_m24_eval.py --run hg_fin --src fin [--idx 0-39]

Per instance (data/generated/uc24/test.npz):
  1. LP relaxation (timed; input of every LP-based rule and of the GNNs);
  2. full MILP at the reference settings (300 s, 0.1 %; highspy, one thread) with its incumbent log: dual bound DB,
     speed-up denominator and the time-to-quality reference;
  3. full MILP at mip_rel_gap 0.5 % (same limit);
  4. learning-free Learning to Fix on the LP relaxation (results/uc24/base_tune_lp_1.json, eps = 1 %, tuned on the same
     40 validation instances), faithful (no guards), reduced MILP at 0.5 %;
  5. our earlier guarded 95 % rule (label-free imitation GNN, stored probabilities of results/uc24/uc24ltf_probs.npz and
     their stored inference time; adequacy guard, LP-relaxation guard, min up/down row release; scripts/uc_b3_fix.py),
     reduced MILP at 0.1 % (its published setting) and 0.5 %;
  6. m24: the 5-member temporal-GNN ensemble (data/generated/uc24_m24/models/<src>_s*.pt, every forward pass timed),
     eq. (5) with the thresholds of results/uc24/m24_tune_<run>.json, adequacy guard + minimal conflict release + LP-
     relaxation guard (otsl.m24.guard_chain), reduced MILP at 0.1 % and at 0.5 % (the pipeline).
No fallback: a reduced problem without a solution is infeasible. Method time = inference + LP relaxation + fixing +
guards + reduced MILP. Output: results/uc24/m24_eval_test.jsonl (one line per instance; resume skips finished ones).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import load_b3  # noqa: E402
from otsl.base import solve_reduced  # noqa: E402
from otsl.fixpolicy import lp_guard, release_conflicting_rows  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.m24 import guard_chain  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402
from uc_b3_fix import make_specs  # noqa: E402
from uc_m24_train import KEYS, featurizer, load_member  # noqa: E402

U24, DATA, RES = "data/generated/uc24", "data/generated/uc24_m24", "results/uc24"
TL, REF, LOOSE = 300.0, 1e-3, 5e-3


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0


def parse(s, n):
    if "-" in s:
        x, y = s.split("-")
        return list(range(int(x), int(y) + 1))
    return [int(x) for x in s.split(",") if x] or list(range(n))


def rec_of(sol, **kw):
    r = {k: v for k, v in sol.items() if k != "trace"}
    r.update(kw)
    r["t_method"] = r["time"] + r.get("pre_s", 0.0) + r.get("t_rel", 0.0) + r.get("t_guard", 0.0)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="m24 tuning run (results/uc24/m24_tune_<run>.json)")
    ap.add_argument("--src", required=True, help="model name: data/generated/uc24_m24/models/<src>_s*.pt")
    ap.add_argument("--idx", default="0-39")
    ap.add_argument("--out", default=os.path.join(RES, "m24_eval_test.jsonl"))
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    import highspy
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=24, network=True)
    d = load_b3(os.path.join(U24, "test.npz"))
    feat = featurizer(sysm)
    members = [load_member(p, sysm, feat) for p in sorted(glob.glob(os.path.join(DATA, "models", f"{a.src}_s*.pt")))]
    tu = json.load(open(os.path.join(RES, f"m24_tune_{a.run}.json")))
    lo, hi = np.array(tu["lo"]), np.array(tu["hi"])
    lp = json.load(open(os.path.join(RES, "base_tune_lp_1.json")))
    lo_lp, hi_lp = np.array(lp["lo"]), np.array(lp["hi"])
    P = np.load(os.path.join(RES, "uc24ltf_probs.npz"))
    stored = {json.loads(x)["i"]: json.loads(x) for x in open(os.path.join(RES, "uc24ltf_eval_test.jsonl"))}
    idx = parse(a.idx, 40)
    done = {json.loads(x)["i"] for x in open(a.out)} if os.path.exists(a.out) else set()
    idx = [i for i in idx if i not in done]
    one = {k: d[k][:1] for k in ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel")}
    for m1 in members:                                    # warm-up
        m1.predict(one)
    print(f"highspy {highspy.Highs().version()}; {len(members)} members; tuned run {a.run} (converged {tu['converged']}, "
          f"val fixed {tu['val_fixed_share'] * 100:.1f}%); {len(idx)} instances", flush=True)
    t_start = time.time()
    for k, i in enumerate(idx):
        sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        rec = dict(i=i, loadavg0=os.getloadavg()[0], cpu=sorted(os.sched_getaffinity(0)))
        rel, t_rel = timed(m.solve_dispatch, sc, None, relax=True)
        rec.update(t_rel=t_rel, c_rel=float(rel.obj), u_rel_maxdiff=float(np.abs(rel.u - d["u_rel"][i]).max()))
        runs = {}
        runs["full"] = rec_of(solve_reduced(m, sc, None, TL, REF), mip_rel_gap=REF)
        runs["full gap=0.5%"] = rec_of(solve_reduced(m, sc, None, TL, LOOSE), mip_rel_gap=LOOSE)
        # learning-free LtF on the LP relaxation, faithful, 0.5 %
        fx, dt = timed(fix_dict, rel.u, lo_lp, hi_lp)
        runs["LtF-LP gap=0.5%"] = rec_of(solve_reduced(m, sc, fx, TL, LOOSE), mip_rel_gap=LOOSE, pre_s=dt, t_rel=t_rel,
                                         t_guard=0.0, n_fixed_pre=len(fx))
        # earlier guarded 95 % rule (stored probabilities and inference time)
        s0 = stored[i]["old: 95%|best=bce_g"]
        (_, arr, g), = make_specs(i, d, {"bce": P["bce_te"].astype(np.float64), "rl": P["rl_te"].astype(np.float64)},
                                  sysm, ["bce_g"], [0.95])
        fx = {(int(t), int(gg)): int(v) for t, gg, v in arr}
        t0 = time.time()
        fx2, r_lp, _ = lp_guard(m, sysm, sc, fx)
        fx2, r_c = release_conflicting_rows(fx2, sysm, sc.u0)
        tg = time.time() - t0
        for gap, nm in ((REF, "guarded 95%"), (LOOSE, "guarded 95% gap=0.5%")):
            runs[nm] = rec_of(solve_reduced(m, sc, fx2, TL, gap), mip_rel_gap=gap, pre_s=float(s0.get("pre_s", 0.0)),
                              t_rel=t_rel, t_guard=tg, n_fixed_pre=len(fx), released_lp=r_lp, released_rows=r_c,
                              stored_obj=s0.get("obj"))
        # m24: ensemble -> tuned thresholds -> adequacy + minimal conflict release + LP guard
        di = {kk: d[kk][i:i + 1] for kk in ("load", "avail", "u0", "sr")}
        di.update(u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
        p, t_inf = timed(lambda: np.mean([m1.predict(di)[0].astype(np.float64) for m1 in members], 0))
        fx, dt = timed(fix_dict, p, lo, hi)
        fx2, info = guard_chain(m, sysm, sc, fx, lp=True)
        for gap, nm in ((REF, "m24 + LP guard"), (LOOSE, "m24 pipeline (LP guard, gap=0.5%)")):
            runs[nm] = rec_of(solve_reduced(m, sc, fx2, TL, gap), mip_rel_gap=gap, pre_s=t_inf + dt, t_rel=t_rel,
                              t_guard=info["guard_s"], n_fixed_pre=len(fx),
                              **{kk: v for kk, v in info.items() if kk.startswith("released")})
        for nm, r in runs.items():
            if nm.startswith("full"):
                r.update(pre_s=0.0, t_rel=0.0, t_guard=0.0, t_method=r["time"])
        rec["runs"] = runs
        rec["loadavg1"] = os.getloadavg()[0]
        with open(a.out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        full = runs["full"]
        msg = " | ".join(f"{nm[:22]}: " + (f"{(r['obj'] - full['obj']) / full['obj'] * 100:+.2f}% {r['t_method']:.1f}s"
                                           if r["feasible"] else "INF") for nm, r in runs.items() if nm != "full")
        print(f"[{k + 1}/{len(idx)} {time.time() - t_start:.0f}s] i={i} full {full['time']:.1f}s | {msg}", flush=True)
    print("done", flush=True)
