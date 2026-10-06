"""Hybrid study, step 2: Learning to Fix's threshold tuning (Algorithm 1 + 2, otsl.ltfx) on hybrid scores, with the
test-time guards inside the check (otsl.hybrid.HybridTuner).

    python3 scripts/uc_hybrid_tune.py --job TAG:SCORE:EPS:GUARDS:N_VAL[:WARM_TAG] [--job ...] [--budget_min 60]

  SCORE   <src>_s<seed> (probabilities, src in bce / st) or harm_<src>_s<seed> (error-cost scores mapped by
          otsl.hybrid.harm_to_score)
  GUARDS  all (adequacy + min up/down rows + LP relaxation), none, or a comma-free list like adeq+rows+lp
  N_VAL   number of validation instances, in the order val (60), val_extra (120), val_extra2 (180)
  WARM    tag of an earlier run on a prefix of the same instances (same score, eps, guards): its cuts start the master,
          and its final thresholds are first checked on the new instances (held-out violation rate of the smaller
          validation set; these check results also stay in the tuner's cache)

Jobs run one after another in this process (one core; HiGHS single-threaded). Per job: results/uc12/hybrid_tune_<TAG>
.json (thresholds, guarded / unguarded validation fixed shares, iteration log, statistics, verification, held-out
checks), .log, and _cuts.jsonl (every cut set, appended as found).

    python3 scripts/uc_hybrid_tune.py --holdout ltfx_tune_bce_1:bce_s0:none:180
        held-out check only: thresholds of an existing tuning file applied to validation instances [N:360)
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
from otsl.hybrid import ALL_GUARDS, HybridTuner, harm_to_score, load_cuts  # noqa: E402
from otsl.ltfx import fix_masks  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
FIELDS = ("load", "avail", "u0", "sr", "obj", "u", "day", "start", "gap", "time")
VAL = (("va", "val"), ("vx", "val_extra"), ("vx2", "val_extra2"))


def val_data(n_val):
    parts = [load(os.path.join(ROOT, f"{v}.npz")) for _, v in VAL if os.path.exists(os.path.join(ROOT, f"{v}.npz"))]
    return {k: np.concatenate([p[k] for p in parts])[:n_val] for k in FIELDS}


def val_scores(key, n_val, splits=None):
    P = np.load(os.path.join(OUT, "hybrid_probs.npz"))
    splits = splits or [k for k, _ in VAL]
    if key.startswith("harm_"):
        src = key[5:]
        norm = json.load(open(os.path.join(OUT, "hybrid_probs.json")))["harm_norm_logh_mean_sd"][src]
        parts = [harm_to_score(P[f"{key}_{k}"], P[f"{src}_{k}"], *norm) for k in splits if f"{key}_{k}" in P.files]
    else:
        parts = [P[f"{key}_{k}"] for k in splits if f"{key}_{k}" in P.files]
    return np.concatenate(parts)[:n_val]


def parse_guards(g):
    return () if g == "none" else ALL_GUARDS if g == "all" else tuple(g.split("+"))


def run_job(spec, budget_s, relax_tl, check_tl, K_max, Q):
    parts = spec.split(":")
    tag, key, eps, gname, n_val = parts[0], parts[1], float(parts[2]), parts[3], int(parts[4])
    warm = parts[5] if len(parts) > 5 else None
    guards = parse_guards(gname)
    t_cpu0, t0 = time.process_time(), time.time()
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    d = val_data(n_val)
    pi = val_scores(key, n_val)
    assert len(pi) == n_val == len(d["obj"]), (len(pi), n_val, len(d["obj"]))
    logf = open(os.path.join(OUT, f"hybrid_tune_{tag}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    cut_log = os.path.join(OUT, f"hybrid_tune_{tag}_cuts.jsonl")
    warm_cuts, warm_res = None, None
    if warm:
        warm_cuts = load_cuts(os.path.join(OUT, f"hybrid_tune_{warm}_cuts.jsonl"))
        warm_res = json.load(open(os.path.join(OUT, f"hybrid_tune_{warm}.json")))
        assert warm_res["score"] == key and warm_res["eps"] == eps and tuple(warm_res["guards"]) == guards
    open(cut_log, "w").close()
    log(f"hybrid tuning {tag}: score {key} eps={eps} guards={guards} n_val={n_val} K_max={K_max} relax_tl={relax_tl} "
        f"check_tl={check_tl} Q={Q} warm={warm} ({len(warm_cuts or [])} cuts)")
    tu = HybridTuner(m, sysm, d, pi, eps, guards=guards, warm_cuts=warm_cuts, cut_log=cut_log, K_max=K_max, Q=Q,
                     check_tl=check_tl, relax_tl=relax_tl, master_tl=120.0, log=log, time_budget=budget_s)
    if warm_cuts:                          # the warm-start cuts are part of this run's master: log them as well
        from otsl.hybrid import cut_to_json
        with open(cut_log, "a") as f:
            for c in warm_cuts:
                f.write(json.dumps(cut_to_json(c)) + "\n")
    holdout = None
    if warm_res is not None:               # held-out check of the smaller validation set's thresholds
        lo0, hi0 = np.array(warm_res["lo"]), np.array(warm_res["hi"])
        n0 = warm_res["n_val"]
        off0, on0 = fix_masks(pi, lo0, hi0)
        th = time.time()
        res = []
        for i in range(n0, n_val):
            ok, info = tu.check(i, off0[i], on0[i])
            res.append(dict(i=i, ok=bool(ok), status=info.get("status", "cached"),
                            obj_ratio=float(info["obj"] / d["obj"][i]) if np.isfinite(info.get("obj", np.inf)) else None))
        nf = sum(not r["ok"] for r in res)
        holdout = dict(thresholds_from=warm, n_train=n0, n_holdout=len(res), n_fail=nf, fail_rate=nf / max(1, len(res)),
                       fails=[r for r in res if not r["ok"]], wall_s=time.time() - th)
        log(f"held-out check of {warm}'s thresholds on instances {n0}..{n_val - 1}: {nf} / {len(res)} fail "
            f"({time.time() - th:.0f}s)")
    lo, hi = tu.run()
    conv = tu.history[-1].get("result") == "all instances feasible"
    t_ver = time.time()
    ver = tu.verify(lo, hi)
    log(f"verification (witness schedules of the guarded fixings priced by the dispatch LP): max rel. cost increase "
        f"{np.max(ver) * 100:.4f}% (eps {eps * 100:g}%), instances above eps: {int((ver > eps * (1 + 1e-6)).sum())}, "
        f"{time.time() - t_ver:.0f}s")
    off, on = fix_masks(pi, lo, hi)
    po, pn = tu.guarded_masks(lo, hi)
    res = dict(tag=tag, score=key, eps=eps, guards=list(guards), converged=conv, lo=lo.tolist(), hi=hi.tolist(),
               n_val=n_val, warm_from=warm, n_warm_cuts=len(warm_cuts or []),
               val_fixed_share_pre_guard=float((off | on).mean()), val_fixed_share=float((po | pn).mean()),
               val_fixed_off=float(po.mean()), val_fixed_on=float(pn.mean()),
               n_iter=len(tu.history), n_cut_sets=len(tu.master.cuts), holdout=holdout,
               verify_rel_increase=ver.tolist(), verify_max=float(np.max(ver)),
               verify_ok=bool(np.all(ver <= eps * (1 + 1e-6))), stats=tu.stats, history=tu.history,
               cfg=dict(K_max=K_max, Q=Q, relax_tl=relax_tl, check_tl=check_tl, budget_s=budget_s),
               wall_s=time.time() - t0, cpu_s=time.process_time() - t_cpu0)
    with open(os.path.join(OUT, f"hybrid_tune_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% after guards "
        f"({res['val_fixed_share_pre_guard'] * 100:.2f}% before) wall {res['wall_s']:.0f}s cpu {res['cpu_s']:.0f}s")
    logf.close()
    return tag, res["val_fixed_share"], conv, res["wall_s"]


def holdout_only(spec, check_tl):
    """thresholds of an existing tuning file (FILE:SCORE:GUARDS:N0) checked on validation instances [N0:360)"""
    fname, key, gname, n0 = spec.split(":")
    guards = parse_guards(gname)
    r = json.load(open(os.path.join(OUT, f"{fname}.json")))
    eps = r["eps"]
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12, network=True)
    d = val_data(10 ** 6)
    pi = val_scores(key, 10 ** 6)
    n0 = int(n0)
    tu = HybridTuner(m, sysm, d, pi, eps, guards=guards, check_tl=check_tl)
    off, on = fix_masks(pi, np.array(r["lo"]), np.array(r["hi"]))
    t0 = time.time()
    res = []
    for i in range(n0, len(pi)):
        ok, info = tu.check(i, off[i], on[i])
        res.append(dict(i=i, ok=bool(ok), status=info.get("status"), obj_ratio=float(info["obj"] / d["obj"][i])
                        if np.isfinite(info.get("obj", np.inf)) else None))
    nf = sum(not x["ok"] for x in res)
    out = dict(thresholds=fname, score=key, guards=list(guards), eps=eps, n_train=n0, n_holdout=len(res), n_fail=nf,
               fail_rate=nf / max(1, len(res)), fails=[x for x in res if not x["ok"]], wall_s=time.time() - t0,
               stats=tu.stats)
    with open(os.path.join(OUT, f"hybrid_holdout_{fname}_{gname}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"held-out {fname} ({gname}): {nf} / {len(res)} fail ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--holdout", action="append", default=[])
    ap.add_argument("--budget_min", type=float, default=60.0)
    ap.add_argument("--relax_tl", type=float, default=6.0)
    ap.add_argument("--check_tl", type=float, default=60.0)
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--Q", type=int, default=20)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    t0 = time.time()
    for spec in a.job:
        r = run_job(spec, a.budget_min * 60, a.relax_tl, a.check_tl, a.K_max, a.Q)
        print(f"[{time.time() - t0:.0f}s] done {r[0]}: val fixed {r[1] * 100:.2f}% converged={r[2]} wall {r[3]:.0f}s", flush=True)
    for spec in a.holdout:
        holdout_only(spec, a.check_tl)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)
