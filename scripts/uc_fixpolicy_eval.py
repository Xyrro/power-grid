"""Reduced-MILP evaluation of fixing rules on B2 (uc12): full MILP and every reduced MILP back-to-back per
instance in the same worker process (same load conditions).

    python scripts/uc_fixpolicy_eval.py --split test --n 60 --ratios 0.8,0.9,0.95,0.97 --full \
        --methods rac,ltf,asym,rl,harm_c --tag test

Rules (all on the same probability model unless stated):
  rac         RACLearn: fix the share of decisions with the largest |p - 0.5| (the saved net has no
              dropout, so the MC-dropout confidence reduces to the probability margin)
  ltf         Learning to Fix: generator-specific thresholds max(p, 1-p) >= theta_g from the measured cost
              impact of each generator's fixing errors (tolerance tau calibrated to the target share)
  asym        rank by min(p, 1-p), OFF decisions x10, then the adequacy guard (our previous best at 90 %)
  rl          rank by the REINFORCE fine-tuned model's probabilities (fix values from that model too)
  harm_u      proposed: rank by the learned expected cost of fixing (otsl/fixpolicy.py), uncompensated labels
  harm_c      proposed, labels with compensated OFF errors (replacement units allowed)
  <harm_x>_guard  the same + adequacy guard
  rac_lp, harm_c_lp   post-hoc extension: + row feasibility check (min up/down) + LP-relaxation guard (the LP
              of the reduced problem must not shed / miss reserve; its solve time is added to the rule's time);
              harm_c_lp also keeps the adequacy guard
Output: results/uc12/fixpolicy_eval_<tag>.jsonl (one record per solve) and a summary on stdout.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otsl.fixpolicy import FixFeaturizer, HarmEnsemble, fix_from_ranking, fix_from_thresholds, lp_guard, release_conflicting_rows  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_fixing import adequacy_guard  # noqa: E402

TL, GAP = 60.0, 1e-3
_W = {}


def _init():
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=12, network=True))


def _solve(sc, fix):
    m = _W["m"]
    sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP, z_fix=fix or None)
    t_, fb = sol.time, 0
    if sol.u is None:                      # fixings conflict with min up/down: solve without them
        fb = 1
        sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP)
        t_ += sol.time
    return dict(obj=float(sol.obj), shed=float(sol.shed), short=float(sol.short), time=float(t_), status=sol.status,
                fallback=fb)


def _job(args):
    i, sc, specs, full = args
    out = []
    if full:
        out.append(dict(i=i, method="full", ratio=0.0, fixed_share=0.0, **_solve(sc, {})))
    for method, ratio, fix, extra in specs:
        lp_s = 0.0
        if method.endswith("_lp"):              # solver-aware guard, timed as part of the rule
            fix, rel_rows = release_conflicting_rows(fix, _W["s"], sc.u0)
            fix, rel_lp, lp_s = lp_guard(_W["m"], _W["s"], sc, fix)
            extra = dict(extra, released_rows=rel_rows, released_lp=rel_lp, lp_guard_s=lp_s)
        r = _solve(sc, fix)
        r["time"] += lp_s
        out.append(dict(i=i, method=method, ratio=ratio, fixed_share=len(fix) / (sc.load.shape[0] * len(sc.u0)),
                        **extra, **r))
    return out


def ltf_for_share(tab, ratio):
    """threshold vector whose calibration (val) fixed share is closest to the target"""
    j = int(np.argmin([abs(s - ratio) for _, _, s in tab]))
    return np.array(tab[j][1]), tab[j][0]


def summarize(recs, ref_obj, label_full="full"):
    by = {}
    for r in recs:
        by.setdefault((r["method"], r["ratio"]), {})[r["i"]] = r
    full = by.get((label_full, 0.0), {})
    common = sorted(set.intersection(*[set(v) for v in by.values()])) if by else []
    t_full = np.mean([full[i]["time"] for i in common]) if full else np.nan
    rows = []
    for (meth, ratio), d in sorted(by.items(), key=lambda kv: (kv[0][0] != label_full, kv[0][0], kv[0][1])):
        rr = [d[i] for i in common]
        cost = np.array([r["obj"] for r in rr])
        ref = ref_obj[common]
        gap = (cost - ref) / ref * 100
        served = np.array([(r["shed"] < 1e-6) and (r["short"] < 1e-6) for r in rr])
        tm = np.mean([r["time"] for r in rr])
        rows.append(dict(method=meth, target=ratio, fixed_share=float(np.mean([r["fixed_share"] for r in rr]) * 100),
                         gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_p90=float(np.percentile(gap, 90)),
                         gap_max=float(gap.max()), served=float(served.mean() * 100),
                         match=float((gap <= 1e-3).mean() * 100), time_s=float(tm),
                         speedup=float(t_full / tm) if full else float("nan"),
                         fallback=float(np.mean([r["fallback"] for r in rr]) * 100), n=len(rr)))
    return rows, common


def fmt(rows):
    cols = ["method", "target", "fixed_share", "gap_mean", "gap_median", "gap_p90", "served", "match", "time_s", "speedup", "fallback"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ratios", default="0.9,0.95")
    ap.add_argument("--methods", default="rac,ltf,asym,rl,harm_c")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ltf_calib", default="ltf_pool")
    ap.add_argument("--skip", default="", help="comma-separated rule@share pairs not to solve, e.g. ltf@0.97")
    a = ap.parse_args()
    root, out_dir, T = "data/generated/uc12", "results/uc12", 12
    d = load(os.path.join(root, f"{a.split}.npz"))
    sysm = load_rts_gmlc()
    P = np.load(os.path.join(out_dir, "fixpolicy_probs.npz"))
    key = {"val": "va", "test": "te", "train": "tr"}[a.split]
    p_bce, p_rl = P[f"bce_{key}"], P[f"rl_{key}"]
    ratios = [float(x) for x in a.ratios.split(",")]
    methods = a.methods.split(",")
    pols, hms = {}, {}
    for v in ("uncomp", "comp"):
        fp = os.path.join(out_dir, f"fixpolicy_policy_{v}.pt")
        if os.path.exists(fp):
            pols[v] = torch.load(fp, weights_only=False)
            hms[v] = HarmEnsemble.from_states(pols[v]["d_in"], pols[v]["hidden"], pols[v]["members"])
    pol = pols["uncomp"]                      # Learning-to-Fix tables (uncompensated impact curves)
    ff = FixFeaturizer(sysm, T)
    idx = list(range(a.start, min(a.start + a.n, len(d["load"]))))
    path = os.path.join(out_dir, f"fixpolicy_eval_{a.tag or a.split}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                done.add(json.loads(line)["i"])
    jobs = []
    for i in idx:
        if i in done:
            continue
        sc = scenario_from(d, i)
        p = p_bce[i]
        yhat = (p > 0.5).astype(int)
        err = np.minimum(p, 1 - p)
        X_i = ff(p, d, i)[None]
        r_harm = {"u": hms["uncomp"].score(X_i)[0] if "uncomp" in hms else None,
                  "c": hms["comp"].score(X_i)[0] if "comp" in hms else None}
        specs = []
        skip = {(x.split("@")[0], float(x.split("@")[1])) for x in a.skip.split(",") if x}
        for ratio in ratios:
            for meth in methods:
                if (meth, ratio) in skip:
                    continue
                extra = {}
                if meth in ("rac", "rac_lp"):
                    fix = fix_from_ranking(err, yhat, ratio)
                elif meth == "ltf":
                    th, tau = ltf_for_share(pol[a.ltf_calib], ratio)
                    fix = fix_from_thresholds(p, th)
                    extra["tau"] = tau
                elif meth == "asym":
                    fix, rel = adequacy_guard(fix_from_ranking(err * np.where(p < 0.5, 10.0, 1.0), yhat, ratio),
                                              sc.load, sc.avail, sc.sr, sysm)
                    extra["released"] = rel
                elif meth == "rl":
                    q = p_rl[i]
                    fix = fix_from_ranking(np.minimum(q, 1 - q), (q > 0.5).astype(int), ratio)
                elif meth.startswith("harm_"):
                    fix = fix_from_ranking(r_harm[meth.split("_")[1]], yhat, ratio)
                    if meth.endswith("_guard") or meth.endswith("_lp"):
                        fix, rel = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
                        extra["released"] = rel
                else:
                    raise ValueError(meth)
                specs.append((meth, ratio, fix, extra))
        jobs.append((i, sc, specs, a.full))
    print(f"{a.split}: {len(jobs)} instances to solve ({len(done)} already done), {len(ratios)} ratios x "
          f"{len(methods)} rules{' + full MILP' if a.full else ''}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(path, "a") as f:
            for k, recs in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                msg = "  ".join(f"{r['method']}@{r['ratio']:.2f}:{r['time']:.1f}s/{(r['obj'] / d['obj'][r['i']] - 1) * 100:+.2f}%"
                                for r in recs)
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} {msg}", flush=True)
    recs = [json.loads(line) for line in open(path)]
    recs = [r for r in recs if r["i"] in set(idx)]
    rows, common = summarize(recs, d["obj"])
    print(f"\n{len(common)} instances\n" + fmt(rows), flush=True)
