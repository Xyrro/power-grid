"""hard benchmark, test evaluation: every rule of one test instance back to back in one single-threaded process.

    taskset -c N python scripts/uc_hard_eval.py --shard i --nshards 2          # one process per core
    python scripts/uc_hard_eval.py --dry                                       # print the rule list

Per test instance (data/generated/hard/test_<kk>.npz; nothing is tuned here):
  full MILP at mip_rel_gap 0.1 % (re-timed here: speed-up denominator T_ref, dual bound DB, bound trace for
  time-to-quality), 0.5 % and 1 %;
  LP relaxation (timed; part of every rule that uses it);
  LtF-LP     Learning to Fix on the LP-relaxation values, faithful thresholds of results/hard/hard_tune_lp.json;
  LP-int+g   every integral LP-relaxation value fixed, then adequacy guard + minimal min up/down conflict release +
             LP-relaxation guard (otsl.hard.guard_chain, lp=True);
  LP pipeline  the learned pipeline without learning: guard-aware thresholds on the LP-relaxation values
             (hard_tune_lp_g.json) -> adequacy + minimal conflict release -> LP-relaxation guard;
  LtF-kNN    Learning to Fix's kNN (eq. 13, k = 50, training scenarios with polished labels), faithful thresholds of
             hard_tune_knn.json (no LP relaxation needed);
  pipeline   temporal-GNN ensemble (data/generated/hard/models/<ens>_s*.pt) -> guard-aware thresholds of
             hard_tune_<hg>.json -> adequacy + minimal conflict release -> LP-relaxation guard;
each fixing rule's reduced MILP at 0.1 % and 0.5 % (time limit 600 s). Method time = reduced MILP + inference /
thresholding + LP relaxation (if used) + guards. Records: results/hard/hard_eval_s<i>.jsonl (resume skips finished
instances).
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
from otsl.base import lp_integral_fixings, solve_reduced  # noqa: E402
from otsl.hard import DATA_DIR, bench_system, guard_chain, load_split, sc_of  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402

RES = "results/hard"
GAPS_FULL = (1e-3, 5e-3, 1e-2)
GAPS_RULE = (1e-3, 5e-3)
TL = 600.0
KEYS = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel", "c_rel", "start")


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


def done_any(out, own_only=False):
    """instances finished in any records file of the evaluation (shards and helpers); own_only: in `out` only (a run
    with --out elsewhere, e.g. step 2 of docs/unified_metrics.md, must not skip what the original study finished)"""
    done = set()
    for f in [out] if own_only else set(glob.glob(os.path.join(RES, "hard_eval_s*.jsonl")) + [out]):
        if os.path.exists(f):
            done |= {json.loads(x)["i"] for x in open(f) if x.strip()}
    return done


def thresholds(run):
    j = json.load(open(os.path.join(RES, f"hard_tune_{run}.json")))
    return np.array(j["lo"]), np.array(j["hi"]), bool(j["converged"])


def knn_model(sysm):
    from otsl.b3 import load_b3
    parts, ys = [], []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "train_[0-9][0-9].npz"))):
        pf = f.replace("train_", "polish_")
        if os.path.exists(pf):
            d = load_b3(f)
            parts.append({k: d[k] for k in ("load", "avail", "u0", "start")})
            ys.append(np.load(pf)["y"])
    tr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tr["u"] = np.concatenate(ys)
    return KNNProb(sysm, tr, k=50)


def main(a):
    torch.set_num_threads(1)
    sysm, m = bench_system()
    te = load_split("test")
    n = len(te["obj"])
    rules = []
    if os.path.exists(os.path.join(RES, "hard_tune_lp.json")):
        rules.append("LtF-LP")
    rules.append("LP-int+g")
    if os.path.exists(os.path.join(RES, "hard_tune_lp_g.json")):
        rules.append("LP pipeline")
    if os.path.exists(os.path.join(RES, "hard_tune_knn.json")) and glob.glob(os.path.join(DATA_DIR, "train_[0-9][0-9].npz")):
        rules.append("LtF-kNN")                  # needs the training scenarios and their polished labels
    members = sorted(glob.glob(os.path.join(DATA_DIR, "models", f"{a.ens}_s*.pt")))
    if members and os.path.exists(os.path.join(RES, f"hard_tune_{a.hg}.json")):
        rules.append("pipeline")
    print(f"rules: {rules}; ensemble {a.ens} ({len(members)} members), thresholds hard_tune_{a.hg}.json", flush=True)
    if a.dry:
        return
    th = {r: thresholds(x) for r, x in (("LtF-LP", "lp"), ("LP pipeline", "lp_g"), ("LtF-kNN", "knn"), ("pipeline", a.hg))
          if r in rules}
    knn = knn_model(sysm) if "LtF-kNN" in rules else None
    nets = []
    if "pipeline" in rules:
        from uc_hard_train import featurizer, load_member
        feat = featurizer(sysm)
        nets = [load_member(p, sysm, feat) for p in members]
    out = a.out or os.path.join(RES, f"hard_eval_s{a.shard}.jsonl")
    done = {r["i"] for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    todo = [i for i in range(n) if i % a.nshards == a.shard and i not in done]
    if a.only:
        todo = [i for i in todo if i in {int(x) for x in a.only.split(",")}]
    # warm-up (first-call overheads of the LP solver and of the networks)
    d0 = {k: te[k][:1] for k in KEYS}
    m.solve_dispatch(sc_of(te, 0), None, relax=True)
    for net in nets:
        net.predict(d0)
    if knn is not None:
        knn.predict(d0)
    print(f"shard {a.shard}/{a.nshards} on CPU {sorted(os.sched_getaffinity(0))}: {len(todo)} instances", flush=True)
    t_start = time.time()
    if a.reverse:
        todo = todo[::-1]
    for j, i in enumerate(todo):
        if i in done_any(out, own_only=bool(a.out)):                 # finished meanwhile by another process (helper shard)
            continue
        sc = sc_of(te, i)
        base = dict(i=i, day=int(te["day"][i]), seed=int(te["seed"][i]), shard=a.shard, cpu=sorted(os.sched_getaffinity(0)))
        recs = []
        for gap in GAPS_FULL:
            r = solve_reduced(m, sc, {}, TL, gap, trace_every=1.0 if gap == GAPS_FULL[0] else 0.0)
            recs.append(dict(base, rule="full MILP", mip_rel_gap=gap, fixed_share=0.0, pre_s=0.0, relax_s=0.0, guard_s=0.0,
                             **r))
        rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
        di = {k: te[k][i:i + 1] for k in KEYS}
        di.update(u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None], c_rel=np.array([rel.obj]))
        specs = []
        if "LtF-LP" in rules:
            fx, s = timed(fix_dict, rel.u, *th["LtF-LP"][:2])
            specs.append(("LtF-LP", fx, s, relax_s, False))
        fx, s = timed(lp_integral_fixings, rel.u)
        specs.append(("LP-int+g", fx, s, relax_s, True))
        if "LP pipeline" in rules:
            fx, s = timed(fix_dict, rel.u, *th["LP pipeline"][:2])
            specs.append(("LP pipeline", fx, s, relax_s, True))
        if "LtF-kNN" in rules:
            p, s1 = timed(lambda: knn.predict(di)[0])
            fx, s2 = timed(fix_dict, p, *th["LtF-kNN"][:2])
            specs.append(("LtF-kNN", fx, s1 + s2, 0.0, False))
        if "pipeline" in rules:
            p, s1 = timed(lambda: np.mean([net.predict(di)[0].astype(np.float64) for net in nets], 0))
            fx, s2 = timed(fix_dict, p, *th["pipeline"][:2])
            specs.append(("pipeline", fx, s1 + s2, relax_s, True))
        G, T = sysm.G, sc.load.shape[0]
        for name, fix_pre, pre_s, rs, guards in specs:
            if guards:
                fix, info = guard_chain(m, sysm, sc, fix_pre, lp=True)
                guard_s = info.pop("guard_s")
            else:
                fix, info, guard_s = fix_pre, {}, 0.0
            for gap in GAPS_RULE:
                r = solve_reduced(m, sc, fix, TL, gap)
                recs.append(dict(base, rule=name, mip_rel_gap=gap, fixed_share=len(fix) / (T * G),
                                 fixed_pre_guard=len(fix_pre) / (T * G), pre_s=pre_s, relax_s=rs, guard_s=guard_s,
                                 **info, **r))
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        print(f"[{j + 1}/{len(todo)} {time.time() - t_start:.0f}s] i={i} " +
              ", ".join(f"{r['rule']}@{r['mip_rel_gap'] * 100:g}% {r['time']:.1f}s{'' if r['feasible'] else ' INF'}"
                        for r in recs), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--ens", default="fin")
    ap.add_argument("--hg", default="hg_fin")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="process the shard's instances from the end (helper)")
    ap.add_argument("--out", default="", help="records file (default results/hard/hard_eval_s<shard>.jsonl)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    main(a)
