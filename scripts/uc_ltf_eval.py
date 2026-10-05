"""Learning to Fix reconstruction, step 4: reduced-MILP evaluation of fixing rules on B2 (uc12). The full MILP and
every reduced MILP of an instance are solved back to back in the same worker process (60 s, 0.1 % gap).

    python scripts/uc_ltf_eval.py --split test_fresh --n 60 --full --rules <r1,r2,...> --tag fresh

Rules
  ltf:<model>:<variant>:<impact>:<tau>   Learning to Fix: fix u[t,g] := round(p) where max(p, 1-p) >= theta_g
                                          (thresholds from results/uc12/ltf_thresholds{,_seq}.json); model in
                                          knn | bce | st | rl | knn_mix<seed> | knn_boot<seed>; variant in
                                          seq (sequential joint) | budget | each (separable)
  rac@<share>     RACLearn: the share of decisions with the largest |p - 0.5| (MILP-label BCE GNN)
  asym@<share>    BCE GNN, OFF decisions x10, + adequacy guard (ours, V3)
  harm@<share>    learned error-cost ranking (compensated labels) + adequacy guard (ours, W3)
  st@<share>      self-trained GNN (round 3), OFF x10 + adequacy guard (ours, W2)
  rl@<share>      REINFORCE GNN probabilities, |p - 0.5| (ours, U4 / V3)
  <rule>+lp       + row release (min up/down) and the LP-relaxation guard (otsl/fixpolicy.py), timed with the rule
Output: results/uc12/ltf_eval_<tag>.jsonl (one record per solve).
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
from otsl.fixpolicy import FixFeaturizer, HarmEnsemble, fix_from_ranking, lp_guard, release_conflicting_rows  # noqa: E402
from otsl.ltf import fix_from_thresholds  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_fixing import adequacy_guard  # noqa: E402

TL, GAP, T = 60.0, 1e-3, 12
KEY = {"val": "va", "test": "te", "test_fresh": "tf"}
_W = {}


def _init():
    torch.set_num_threads(1)
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=T, network=True))


def _solve(sc, fix):
    m = _W["m"]
    sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP, z_fix=fix or None)
    t_, fb = sol.time, 0
    if sol.u is None:                      # fixings conflict with min up/down / ramping: solve without them
        fb = 1
        sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP)
        t_ += sol.time
    return dict(obj=float(sol.obj), shed=float(sol.shed), short=float(sol.short), time=float(t_), status=sol.status,
                fallback=fb, units_on=float(sol.u.sum() / T))


def _job(args):
    i, sc, specs, full = args
    out = []
    if full:
        out.append(dict(i=i, rule="full", fixed_share=0.0, **_solve(sc, {})))
    for rule, fix, extra in specs:
        fix = {(int(t), int(g)): int(v) for t, g, v in fix}
        lp_s = 0.0
        if rule.endswith("+lp"):
            fix, rel_rows = release_conflicting_rows(fix, _W["s"], sc.u0)
            fix, rel_lp, lp_s = lp_guard(_W["m"], _W["s"], sc, fix)
            extra = dict(extra, released_rows=rel_rows, released_lp=rel_lp, lp_guard_s=lp_s)
        r = _solve(sc, fix)
        r["time"] += lp_s
        out.append(dict(i=i, rule=rule, fixed_share=len(fix) / (T * len(sc.u0)), **extra, **r))
    return out


class RuleMaker:
    def __init__(self, split, d, sysm):
        self.d, self.s, self.k = d, sysm, KEY[split]
        self.P = np.load("results/uc12/ltf_probs.npz")
        self.th = {}
        for fn in ("ltf_thresholds.json", "ltf_thresholds_seq.json"):     # separable and sequential decompositions
            fp = os.path.join("results/uc12", fn)
            if os.path.exists(fp):
                for model, dct in json.load(open(fp)).items():
                    self.th.setdefault(model, {}).update(dct)
        pol = torch.load("results/uc12/fixpolicy_policy_comp.pt", weights_only=False)
        self.harm = HarmEnsemble.from_states(pol["d_in"], pol["hidden"], pol["members"])
        self.ff = FixFeaturizer(sysm, T)

    def p(self, model, i):
        return self.P[f"{model}_{self.k}"][i]

    def __call__(self, rule, i, sc):
        base = rule[:-3] if rule.endswith("+lp") else rule
        extra = {}
        if base.startswith("ltf:"):
            _, model, variant, impact, tau = base.split(":")
            th = np.array(self.th[model][f"{variant}:{impact}:{tau}"]["theta"])
            fix = fix_from_thresholds(self.p("knn" if model.startswith("knn") else model, i), th)
        else:
            name, share = base.split("@")
            share = float(share)
            p = self.p({"rac": "bce", "asym": "bce", "harm": "bce", "st": "st", "stsym": "st", "rl": "rl"}[name], i)
            yhat = (p > 0.5).astype(int)
            err = np.minimum(p, 1 - p)
            if name in ("rac", "stsym", "rl"):
                fix = fix_from_ranking(err, yhat, share)
            elif name in ("asym", "st"):
                fix, extra["released"] = adequacy_guard(fix_from_ranking(err * np.where(p < 0.5, 10.0, 1.0), yhat, share),
                                                        sc.load, sc.avail, sc.sr, self.s)
            elif name == "harm":
                score = self.harm.score(self.ff(p, self.d, i)[None])[0]
                fix, extra["released"] = adequacy_guard(fix_from_ranking(score, yhat, share), sc.load, sc.avail, sc.sr, self.s)
            else:
                raise ValueError(rule)
        return [(t, g, v) for (t, g), v in fix.items()], extra


def summarize(recs, ref_obj, ref_time=None):
    """one row per rule over the instances on which every rule (and the full MILP) was solved; without a full-MILP
    record, speed-ups are relative to ref_time (the dataset's MILP times, not solved back to back)"""
    by = {}
    for r in recs:
        by.setdefault(r["rule"], {})[r["i"]] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()])) if by else []
    full = by.get("full", {})
    t_full = np.array([full[i]["time"] for i in common]) if full else (ref_time[common] if ref_time is not None else None)
    rows = []
    for rule, dct in by.items():
        rr = [dct[i] for i in common]
        cost = np.array([r["obj"] for r in rr])
        ref = ref_obj[common]
        gap = (cost - ref) / ref * 100
        served = np.array([(r["shed"] < 1e-6) and (r["short"] < 1e-6) for r in rr])
        tm = np.array([r["time"] for r in rr])
        rows.append(dict(rule=rule, n=len(rr), fixed_share=float(np.mean([r["fixed_share"] for r in rr]) * 100),
                         gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_p90=float(np.percentile(gap, 90)),
                         gap_max=float(gap.max()), served=float(served.mean() * 100), match=float((gap <= 1e-3).mean() * 100),
                         gt1=int((gap > 1).sum()), gt10=int((gap > 10).sum()),
                         feasible=float(100 - np.mean([r["fallback"] for r in rr]) * 100),
                         time_s=float(tm.mean()), speedup=float(t_full.mean() / tm.mean()) if t_full is not None else float("nan"),
                         speedup_median=float(np.median(t_full / tm)) if t_full is not None else float("nan"),
                         units_on=float(np.mean([r.get("units_on", np.nan) for r in rr]))))
    return rows, common


def fmt(rows):
    cols = ["rule", "fixed_share", "gap_mean", "gap_median", "gap_p90", "gap_max", "served", "feasible", "time_s", "speedup",
            "speedup_median"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = load(os.path.join("data/generated/uc12", f"{a.split}.npz"))
    sysm = load_rts_gmlc()
    rules = [r for r in a.rules.split(",") if r]
    mk = RuleMaker(a.split, d, sysm)
    path = os.path.join("results/uc12", f"ltf_eval_{a.tag or a.split}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            done.add(json.loads(line)["i"])
    idx = list(range(a.start, min(a.start + a.n, len(d["load"]))))
    jobs = []
    for i in idx:
        if i in done:
            continue
        sc = scenario_from(d, i)
        specs = []
        for rule in rules:
            fix, extra = mk(rule, i, sc)
            specs.append((rule, fix, extra))
        jobs.append((i, sc, specs, a.full))
    print(f"{a.split}: {len(jobs)} instances to solve ({len(done)} done), {len(rules)} rules"
          f"{' + full MILP' if a.full else ''}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(path, "a") as f:
            for k, recs in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                msg = "  ".join(f"{r['rule']}:{r['time']:.1f}s/{(r['obj'] / d['obj'][r['i']] - 1) * 100:+.2f}%" for r in recs)
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} {msg}", flush=True)
    recs = [json.loads(line) for line in open(path)]
    recs = [r for r in recs if r["i"] in set(idx)]
    rows, common = summarize(recs, d["obj"], d["time"])
    print(f"\n{len(common)} instances\n" + fmt(rows), flush=True)
