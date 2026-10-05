"""Learning to Fix reconstruction, step 5: tables, equal-speed-up (Pareto) comparison, seed spread.

    python scripts/uc_ltf_report.py

Reads results/uc12/ltf_eval_{fresh,fresh_b,test,val}.jsonl (whichever exist) and results/uc12/ltf_thresholds.json;
writes results/uc12/ltf_results.{md,json}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otsl.ucdata import load  # noqa: E402
from uc_ltf_eval import summarize  # noqa: E402

OUT = "results/uc12"
SPEEDS = [2, 3, 5, 10, 20]
FAM = {"rac": "RACLearn (BCE confidence)", "asym": "asymmetric + adequacy guard (ours)",
       "harm": "learned error-cost + adequacy guard (ours)", "st": "self-trained, asym + guard (ours)",
       "rl": "REINFORCE probabilities (ours)", "stsym": "self-trained, confidence (ours)"}
MOD = {"knn": "kNN", "bce": "BCE GNN", "st": "self-trained GNN", "rl": "REINFORCE GNN"}


def family(rule):
    lp = rule.endswith("+lp")
    base = rule[:-3] if lp else rule
    if base == "full":
        return "full MILP"
    if base.startswith("ltf:"):
        _, model, variant, impact, _ = base.split(":")
        if model.startswith("knn_mix"):
            f = f"Learning to Fix, kNN, val+train calibration ({variant}, {impact})"
        elif model.startswith("knn_boot") or model.startswith("knn_bs"):
            f = f"Learning to Fix, kNN, bootstrap calibration seed {model[-1]} ({variant}, {impact})"
        else:
            f = f"Learning to Fix, {MOD[model]} ({variant}, {impact})"
    else:
        f = FAM[base.split("@")[0]]
    return f + (" + LP guard" if lp else "")


def point_label(rule):
    base = rule[:-3] if rule.endswith("+lp") else rule
    if base.startswith("ltf:"):
        return "tau " + f"{float(base.split(':')[-1]) * 100:g} %"
    return f"{float(base.split('@')[1]) * 100:g} % target" if "@" in base else ""


def seed_merge(rows):
    """knn_mix<s> / knn_boot<s> rules: mean and spread (min..max) over the seeds, per tolerance"""
    grp = {}
    for r in rows:
        for pre in ("knn_mix", "knn_boot", "knn_bs"):
            if r["rule"].startswith("ltf:" + pre):
                parts = r["rule"].split(":")
                grp.setdefault(":".join([parts[0], pre] + parts[2:]), []).append(r)
    out = []
    for k, rs in grp.items():
        m = {c: float(np.mean([r[c] for r in rs])) for c in ("fixed_share", "gap_mean", "gap_median", "served", "feasible", "speedup")}
        sp = {c + "_range": [float(min(r[c] for r in rs)), float(max(r[c] for r in rs))] for c in m}
        out.append(dict(rule=k, seeds=len(rs), **m, **sp))
    return out


def pareto(rows, key="gap_mean"):
    fams = {}
    for r in rows:
        if r["rule"] == "full" or r["rule"].startswith("ltf:knn_mix") or r["rule"].startswith("ltf:knn_boot") \
                or r["rule"].startswith("ltf:knn_bs"):
            continue
        fams.setdefault(family(r["rule"]), []).append(r)
        # union rows: the best any variant of a group attains
        if r["rule"].startswith("ltf:"):
            fams.setdefault("[union] Learning to Fix, any model / tolerance / guard", []).append(r)
            if r["rule"].startswith("ltf:knn:") and not r["rule"].endswith("+lp"):
                fams.setdefault("[union] Learning to Fix, kNN as published (no guard)", []).append(r)
        elif r["rule"].startswith("rac@"):
            fams.setdefault("[union] RACLearn or Learning to Fix (any model)", []).append(r)
        else:
            fams.setdefault("[union] ours, any ranking / guard", []).append(r)
    for r in rows:
        if r["rule"].startswith("ltf:") and not r["rule"].startswith("ltf:knn_"):
            fams.setdefault("[union] RACLearn or Learning to Fix (any model)", []).append(r)
    tab = {}
    for f, rs in fams.items():
        tab[f] = {}
        for s in SPEEDS:
            ok = [r for r in rs if r["speedup"] >= s]
            tab[f][s] = min(ok, key=lambda r: r[key]) if ok else None
    return tab


def md_rows(rows):
    L = ["| rule | point | fixed % | mean gap % | median gap % | p90 gap % | max gap % | # > 1 % | # > 10 % | served % | reduced MILP feasible % | time s | speed-up | median speed-up |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["rule"] != "full", family(r["rule"]), -r["fixed_share"])):
        L.append(f"| {family(r['rule'])} | {point_label(r['rule'])} | {r['fixed_share']:.1f} | {r['gap_mean']:.3f} | "
                 f"{r['gap_median']:.3f} | {r['gap_p90']:.3f} | {r['gap_max']:.2f} | {r['gt1']} | {r['gt10']} | {r['served']:.1f} | {r['feasible']:.1f} | "
                 f"{r['time_s']:.2f} | {r['speedup']:.1f}x | {r['speedup_median']:.1f}x |")
    return L


def md_pareto(tab, key="gap_mean"):
    L = ["| rule family | " + " | ".join(f">= {s}x" for s in SPEEDS) + " |", "|---|" + "---|" * len(SPEEDS)]
    for f in sorted(tab):
        cells = []
        for s in SPEEDS:
            r = tab[f][s]
            cells.append("–" if r is None else f"{r[key]:.2f} ({point_label(r['rule'])}, {r['speedup']:.1f}x, served {r['served']:.0f} %)")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    return L


def paired(recs, rows, ref_obj, n_boot=2000, seed=0):
    """Each Learning-to-Fix point vs the point of our rules with the closest speed-up (log scale): paired difference
    of the per-instance gaps (ours - LtF) on the common instances, bootstrap 95 % CI of the mean difference."""
    by = {}
    for r in recs:
        by.setdefault(r["rule"], {})[r["i"]] = r
    sp = {r["rule"]: r["speedup"] for r in rows}
    ours = [r for r in sp if not r.startswith("ltf:") and not r.startswith("rac@") and r != "full"]
    rng = np.random.default_rng(seed)
    out = []
    for a in sorted(r for r in sp if r.startswith("ltf:") and not r.startswith("ltf:knn_")):
        if not ours:
            break
        b = min(ours, key=lambda o: abs(np.log(sp[o] / sp[a])))
        common = sorted(set(by[a]) & set(by[b]))
        ga = np.array([(by[a][i]["obj"] - ref_obj[i]) / ref_obj[i] * 100 for i in common])
        gb = np.array([(by[b][i]["obj"] - ref_obj[i]) / ref_obj[i] * 100 for i in common])
        dlt = gb - ga
        bs = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(n_boot)])
        out.append(dict(ltf=a, ours=b, speed_ltf=sp[a], speed_ours=sp[b], n=len(common), mean_diff=float(dlt.mean()),
                        ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], median_diff=float(np.median(dlt)),
                        ours_better=int((dlt < -1e-3).sum()), ltf_better=int((dlt > 1e-3).sum())))
    return out


def load_pass(tag, d, n):
    path = os.path.join(OUT, f"ltf_eval_{tag}.jsonl")
    if not os.path.exists(path):
        return None
    recs = [json.loads(line) for line in open(path)]
    recs = [r for r in recs if r["i"] < n]
    rows, common = summarize(recs, d["obj"], d["time"])
    recs = [r for r in recs if r["i"] in set(common)]
    return dict(rows=rows, n=len(common), seeds=seed_merge(rows), pareto=pareto(rows), pareto_median=pareto(rows, "gap_median"),
                paired=paired(recs, rows, d["obj"]) if tag != "val" else [])


if __name__ == "__main__":
    res, md = {}, ["# Learning to Fix (reconstructed) vs our fixing rules on B2 (12-hour UC, RTS-GMLC)", "",
                   "Generated by `scripts/uc_ltf_report.py`; method and caveats: `docs/methods/ltf.md`. Gaps are to the "
                   "dataset's full-MILP objective (60 s, 0.1 %); speed-up = mean full-MILP time / mean rule time, with the "
                   "full MILP re-solved back to back in the same worker (ratio of means; the median of per-instance ratios is "
                   "also given). Served = no load shedding and no reserve shortfall. Feasible = the reduced MILP had a "
                   "solution (otherwise the full MILP was solved and its time added).", ""]
    passes = [("fresh", "test_fresh", 60, "Main result: first 60 instances of the fresh test set (test_fresh, seed 23)"),
              ("fresh_b", "test_fresh", 60, "Fresh test set, second pass (seeded calibration variants; own full MILP)"),
              ("test", "test", 60, "Original test set, first 60 instances"),
              ("val", "val", 60, "Validation (selection only; no back-to-back full MILP: speed-ups vs the dataset's MILP times)")]
    for tag, split, n, title in passes:
        if not os.path.exists(os.path.join("data/generated/uc12", f"{split}.npz")):
            continue
        d = load(os.path.join("data/generated/uc12", f"{split}.npz"))
        r = load_pass(tag, d, n)
        if r is None:
            continue
        res[tag] = r
        md += [f"## {title} ({r['n']} instances)", ""] + md_rows(r["rows"]) + [""]
        if r["seeds"]:
            md += ["Seeded calibration (3 seeds; mean, range in brackets):", "",
                   "| rule | seeds | fixed % | mean gap % | median gap % | served % | speed-up |", "|---|---|---|---|---|---|---|"]
            for s in r["seeds"]:
                md.append(f"| {s['rule']} | {s['seeds']} | {s['fixed_share']:.1f} [{s['fixed_share_range'][0]:.1f}, {s['fixed_share_range'][1]:.1f}] | "
                          f"{s['gap_mean']:.3f} [{s['gap_mean_range'][0]:.3f}, {s['gap_mean_range'][1]:.3f}] | "
                          f"{s['gap_median']:.3f} [{s['gap_median_range'][0]:.3f}, {s['gap_median_range'][1]:.3f}] | "
                          f"{s['served']:.1f} [{s['served_range'][0]:.1f}, {s['served_range'][1]:.1f}] | "
                          f"{s['speedup']:.1f} [{s['speedup_range'][0]:.1f}, {s['speedup_range'][1]:.1f}] |")
            md.append("")
        if tag != "val":
            md += ["Best **mean** gap (%) a rule family attains with a point at least that fast:", ""] + md_pareto(r["pareto"]) + [""]
            md += ["Best **median** gap (%) at equal speed-up:", ""] + md_pareto(r["pareto_median"], "gap_median") + [""]
            if r["paired"]:
                md += ["Paired comparison: every Learning-to-Fix point against the rule of ours with the closest speed-up "
                       "(difference of per-instance gaps, ours minus LtF, percentage points; bootstrap 95 % CI over instances):", "",
                       "| Learning to Fix | speed-up | ours | speed-up | mean diff | 95 % CI | median diff | ours better / LtF better |",
                       "|---|---|---|---|---|---|---|---|"]
                for q in r["paired"]:
                    md.append(f"| {family(q['ltf'])}, {point_label(q['ltf'])} | {q['speed_ltf']:.1f}x | {family(q['ours'])}, "
                              f"{point_label(q['ours'])} | {q['speed_ours']:.1f}x | {q['mean_diff']:+.3f} | "
                              f"[{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}] | {q['median_diff']:+.3f} | {q['ours_better']} / {q['ltf_better']} |")
                md.append("")
    th = json.load(open(os.path.join(OUT, "ltf_thresholds.json")))
    if os.path.exists(os.path.join(OUT, "ltf_thresholds_seq.json")):
        for model, dct in json.load(open(os.path.join(OUT, "ltf_thresholds_seq.json"))).items():
            th.setdefault(model, {}).update(dct)
    md += ["## Calibration (val): tolerance -> predicted impact and fixed share", "",
           "| model | decomposition:impact:tau | calibration impact % | fixed share val % | fixed share test_fresh % |", "|---|---|---|---|---|"]
    for model, rr in th.items():
        for k, v in rr.items():
            if ":comp:" in k and (k.startswith("seq") or k.startswith("budget") or float(k.split(":")[-1]) <= 0.01):
                md.append(f"| {model} | {k} | {v['impact_cal'] * 100:.3f} | {v['share_va'] * 100:.1f} | "
                          f"{v.get('share_tf', float('nan')) * 100:.1f} |")
    md.append("")
    with open(os.path.join(OUT, "ltf_results.md"), "w") as f:
        f.write("\n".join(md))

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    with open(os.path.join(OUT, "ltf_results.json"), "w") as f:
        json.dump(clean(res), f, indent=1, default=float)
    print("\n".join(md))
