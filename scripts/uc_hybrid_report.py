"""Hybrid study, step 4: tables for results/uc12/hybrid_results.md / .json.

    python3 scripts/uc_hybrid_report.py

Paper metrics (Learning to Fix, Sec. IV-C): feasibility rate of the reduced MILP; on feasible instances the gap
(C - DB) / C to the back-to-back full MILP's dual bound DB = obj * (1 - MIP gap), runtime and per-instance speed-up
T_MILP / T_method (inference, LP relaxation for model inputs and guards included), fixed share (after guards).
Our earlier convention: gap to the dataset MILP objective over all instances (an infeasible reduced MILP falls back to
the full MILP and pays both times), served share (no shedding / spill / reserve shortfall), speed-up as a ratio of
mean times. Seeded hybrids: statistics per seed, then mean and [min, max] over seeds. Paired comparisons: per-instance
differences against the faithful Learning to Fix rules (kNN and BCE GNN, eps = 1 %), instance bootstrap 95 % CI
(2,000 resamples); for seeded hybrids the per-instance values are averaged over seeds first.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.ucdata import load  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"
REFS = ["faithful LtF kNN eps=1%", "faithful LtF BCE eps=1%"]
PARETO = (3, 5, 8, 12, 20)
TAG = re.compile(r"hybrid (?P<fam>h[a-z])_(?P<src>[a-z]+)_s(?P<seed>\d)_(?P<e>e\d+)_n(?P<n>\d+)(?P<lp> \+lp)?$")


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def per_instance(path, d):
    recs = [json.loads(line) for line in open(path)]
    by = {}
    for r in recs:
        by.setdefault(r["rule"], {})[r["i"]] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    full = by["full MILP"]
    DB = np.array([full[i]["obj"] * (1 - full[i]["mip_gap"]) for i in common])
    t_full = np.array([full[i]["time"] for i in common])
    ref = d["obj"][common]
    out = {}
    for rule, rr in by.items():
        R = [rr[i] for i in common]
        feas = np.array([r["feasible"] for r in R])
        cost = np.array([r["obj"] for r in R])
        t_m = np.array([r["time"] + r["pre_s"] + r["relax_s"] + r["guard_s"] for r in R])
        with np.errstate(invalid="ignore", divide="ignore"):
            gap_db = np.where(feas, (cost - DB) / cost * 100, np.nan)
        sp = np.where(feas, t_full / t_m, np.nan)
        c_fb = np.array([r["obj"] if r["feasible"] else r["fb_obj"] for r in R])
        t_fb = np.array([tm + (0 if r["feasible"] else r["fb_time"]) for tm, r in zip(t_m, R)])
        shed = np.array([(r["shed"] if r["feasible"] else r["fb_shed"]) for r in R])
        short = np.array([(r["short"] if r["feasible"] else r["fb_short"]) for r in R])
        out[rule] = dict(feas=feas, gap_db=gap_db, sp=sp, t_m=t_m, g_ref=(c_fb - ref) / ref * 100, t_fb=t_fb,
                         served=(shed < 1e-6) & (short < 1e-6), fixed=np.array([r["fixed_share"] for r in R]) * 100,
                         fixed_pre=np.array([r.get("fixed_pre_guard", r["fixed_share"]) for r in R]) * 100,
                         guard_s=np.array([r["guard_s"] for r in R]), served_feas=np.where(feas, (shed < 1e-6) & (short < 1e-6), False))
    return out, common, t_full


def stats(x, t_full):
    f = x["feas"]
    g, sp = x["gap_db"][f], x["sp"][f]
    return dict(n=int(len(f)), feasible=float(f.mean() * 100), n_infeasible=int((~f).sum()),
                gap_mean=float(g.mean()), gap_max=float(g.max()), gap_median=float(np.median(g)), gap_mean_ci=boot(g),
                runtime_mean=float(x["t_m"][f].mean()), runtime_max=float(x["t_m"][f].max()),
                speedup_mean=float(sp.mean()), speedup_max=float(sp.max()), speedup_median=float(np.median(sp)),
                speedup_mean_ci=boot(sp), fixed_mean=float(x["fixed"][f].mean()), fixed_max=float(x["fixed"][f].max()),
                fixed_pre_guard_mean=float(x["fixed_pre"].mean()), guard_s_mean=float(x["guard_s"].mean()),
                served_feasible=float(x["served"][f].mean() * 100),
                # our convention
                gap_ref_mean=float(x["g_ref"].mean()), gap_ref_ci=boot(x["g_ref"]), gap_ref_median=float(np.median(x["g_ref"])),
                gap_ref_max=float(x["g_ref"].max()), n_gt1=int((x["g_ref"] > 1).sum()), n_gt10=int((x["g_ref"] > 10).sum()),
                served=float(x["served"].mean() * 100), speedup_ratio_of_means=float(t_full.mean() / x["t_fb"].mean()))


def family(rule):
    m = TAG.match(rule)
    if not m:
        return None, None
    return f"{m['fam']} {m['src']} {m['e']} n{m['n']}{' +lp' if m['lp'] else ''}", int(m["seed"])


def paired(a, b, metric):
    """mean of per-instance (a - b) and bootstrap CI over instances where both are defined"""
    x = a - b
    x = x[np.isfinite(x)]
    return dict(diff=float(x.mean()) if len(x) else np.nan, ci=boot(x), n=int(len(x)))


def fmt(x, p=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{p}f}"


def rng_fmt(v, lo, hi, p=2):
    return f"{fmt(v, p)} [{fmt(lo, p)}, {fmt(hi, p)}]"


NAMES = {"hg": "guard-aware LtF on BCE probabilities", "he": "guard-aware LtF on error-cost scores (BCE)",
         "hn": "LtF with all guards (incl. LP) in the check, BCE"}


def fam_label(fk):
    fam, src, e, n = fk.split()[:4]
    lp = " + LP guard at test" if fk.endswith("+lp") else ""
    return f"{NAMES.get(fam, fam)}, eps = {e[1:]} %, {n[1:]} val{lp}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(OUT, "hybrid_eval_test_fresh.jsonl"))
    ap.add_argument("--out", default=os.path.join(OUT, "hybrid_results"))
    args = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    d = load(os.path.join(ROOT, "test_fresh.npz"))
    path = args.eval
    X, common, t_full = per_instance(path, d)
    rows = {k: stats(v, t_full) for k, v in X.items()}
    # ------------------------------------------------------------------ seed families
    fams = {}
    for k in X:
        fk, s = family(k)
        if fk:
            fams.setdefault(fk, {})[s] = k
    famrows = {}
    for fk, seeds in fams.items():
        rs = [rows[seeds[s]] for s in sorted(seeds)]
        agg = {}
        for key in ("feasible", "gap_mean", "gap_max", "speedup_mean", "speedup_max", "fixed_mean", "runtime_mean",
                    "gap_ref_mean", "served", "speedup_ratio_of_means", "n_gt1", "n_gt10", "n_infeasible", "served_feasible"):
            v = [r[key] for r in rs]
            agg[key] = [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
        agg["seeds"] = sorted(seeds)
        famrows[fk] = agg
    # ------------------------------------------------------------------ paired comparisons
    def seed_mean(fk, key):
        arr = np.array([X[fams[fk][s]][key] for s in sorted(fams[fk])], float)
        return np.mean(arr, 0)          # nan if any seed infeasible (paper metrics)
    pairs = {}
    targets = {k: ("rule", k) for k in X if k not in ("full MILP",) + tuple(REFS)}
    targets.update({f"family: {fk}": ("fam", fk) for fk in fams if len(fams[fk]) > 1})
    for name, (kind, key) in targets.items():
        for ref in REFS:
            if ref not in X:
                continue
            R = X[ref]
            if kind == "rule":
                A = X[key]
                gdb, sp, gref = A["gap_db"], A["sp"], A["g_ref"]
            else:
                gdb, sp, gref = seed_mean(key, "gap_db"), seed_mean(key, "sp"), seed_mean(key, "g_ref")
            pairs[f"{name} vs {ref}"] = dict(
                gap_db=paired(gdb, R["gap_db"], "gap_db"), speedup=paired(sp, R["sp"], "sp"),
                log_speedup=paired(np.log(sp), np.log(R["sp"]), "lsp"), gap_ref=paired(gref, R["g_ref"], "g_ref"))
    # ------------------------------------------------------------------ Pareto (best mean gap at speed-up >= x)
    pts = [(k, r["speedup_mean"], r["gap_mean"], r["feasible"]) for k, r in rows.items() if k != "full MILP"]
    pts += [(f"family: {fk} (seed mean)", a["speedup_mean"][0], a["gap_mean"][0], a["feasible"][0]) for fk, a in famrows.items()]
    pareto = {}
    for x in PARETO:
        c = [p for p in pts if p[1] >= x]
        if c:
            b = min(c, key=lambda p: p[2])
            pareto[x] = dict(rule=b[0], speedup=b[1], gap=b[2], feasible=b[3])
        hy = [p for p in c if "hybrid" in p[0] or "family" in p[0]]
        base = [p for p in c if not ("hybrid" in p[0] or "family" in p[0])]
        pareto[f"{x}_hybrid"] = dict(zip(("rule", "speedup", "gap", "feasible"), min(hy, key=lambda p: p[2]))) if hy else None
        pareto[f"{x}_baseline"] = dict(zip(("rule", "speedup", "gap", "feasible"), min(base, key=lambda p: p[2]))) if base else None
    # ------------------------------------------------------------------ tuning tables
    tune = {}
    for f in sorted(glob.glob(os.path.join(OUT, "hybrid_tune_*.json"))):
        r = json.load(open(f))
        if "smoke" in r["tag"]:
            continue
        st = r["stats"]
        tune[r["tag"]] = dict(score=r["score"], eps=r["eps"], guards=r["guards"], n_val=r["n_val"], converged=r["converged"],
                              val_fixed=r["val_fixed_share"] * 100, val_fixed_pre=r["val_fixed_share_pre_guard"] * 100,
                              val_off=r["val_fixed_off"] * 100, val_on=r["val_fixed_on"] * 100, n_iter=r["n_iter"],
                              n_cut_sets=r["n_cut_sets"], n_warm=r["n_warm_cuts"], checks=st["n_check_solves"],
                              cached=st["n_check_cached"], relax=st["n_relax"], t_relax=st["t_relax"], guards_n=st["n_guard"],
                              t_guard=st["t_guard"], verify_max=r["verify_max"] * 100, wall_h=r["wall_s"] / 3600,
                              cpu_h=r["cpu_s"] / 3600, holdout=r.get("holdout"),
                              collapsed=int((np.array(r["hi"]) - np.array(r["lo"]) < 1e-6).sum()),
                              lo_zero=int((np.array(r["lo"]) <= 1e-9).sum()), hi_one=int((np.array(r["hi"]) >= 1 - 1e-9).sum()))
    faithful = {}
    for f in sorted(glob.glob(os.path.join(OUT, "ltfx_tune_*.json"))):
        r = json.load(open(f))
        faithful[f"{r['model']} eps={r['eps'] * 100:g}%"] = dict(val_fixed=r["val_fixed_share"] * 100, n_val=r["n_val"],
                                                                 wall_h=r["wall_s"] / 3600)
    holdouts = {}
    for f in sorted(glob.glob(os.path.join(OUT, "hybrid_holdout_*.json"))):
        r = json.load(open(f))
        holdouts[os.path.basename(f)[15:-5]] = {k: r[k] for k in ("thresholds", "score", "guards", "eps", "n_train",
                                                                   "n_holdout", "n_fail", "fail_rate", "wall_s")}
    for tag, t in tune.items():
        if t["holdout"]:
            h = t["holdout"]
            holdouts[f"{h['thresholds_from']} (in {tag})"] = {k: h[k] for k in ("thresholds_from", "n_train", "n_holdout",
                                                                                "n_fail", "fail_rate", "wall_s")}
    # ------------------------------------------------------------------ 180 vs 360
    vs = {}
    for k in X:
        m = TAG.match(k)
        if m and m["n"] == "360" and not m["lp"]:
            k180 = k.replace("_n360", "_n180")
            if k180 in X:
                a, b = X[k], X[k180]
                eps = int(m["e"][1:])
                vs[k] = dict(r360=rows[k], r180=rows[k180],
                             above_eps_360=int((a["g_ref"] > eps).sum()), above_eps_180=int((b["g_ref"] > eps).sum()),
                             gap_ref_diff_360_minus_180=paired(a["g_ref"], b["g_ref"], ""),
                             gap_db_diff_360_minus_180=paired(a["gap_db"], b["gap_db"], ""),
                             log_speedup_diff_360_minus_180=paired(np.log(a["sp"]), np.log(b["sp"]), ""))
    sel = json.load(open(os.path.join(OUT, "hybrid_selection.json"))) if os.path.exists(os.path.join(OUT, "hybrid_selection.json")) else None
    res = dict(n=len(common), instances=[int(i) for i in common], full_milp_time_mean=float(t_full.mean()), rows=rows,
               families=famrows, paired=pairs, pareto={str(k): v for k, v in pareto.items()}, tuning=tune,
               faithful_tuning=faithful, holdouts=holdouts, val180_vs_360=vs, selection=sel)
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    # ------------------------------------------------------------------ markdown
    order = ["full MILP"] + REFS + ["faithful LtF self-trained eps=10%", "faithful LtF BCE eps=1% + guards at test",
                                    "ours: error-cost + adequacy guard 90% (BCE)", "ours: combined st+error-cost+adequacy+LP guard 98%"]
    keys = [k for k in order if k in rows] + sorted(k for k in rows if k not in order)
    L = [f"# Hybrid fixing rules (Learning to Fix calibration + our models and guards): results", "",
         f"test_fresh, instances {common[0]}–{common[-1]} (n = {len(common)}); full MILP back to back {t_full.mean():.1f} s mean. "
         "Generated by `scripts/uc_hybrid_report.py`; method: [`docs/methods/hybrid.md`](../../docs/methods/hybrid.md).", ""]
    if sel:
        L += [f"Validation selection ({sel['rule']}): {sel['val_fixed_share']} → **{sel['chosen']}**.", ""]
    L += ["## Validation: tuning runs", "",
          "| run | score | eps | guards in check | n val | conv. | val fixed % after / before guards (OFF / ON) | iter. | cut sets (warm) | check MILPs / cached | relax. MILPs (s) | guard evals (s) | collapsed / lo=0 / hi=1 | witness max % | wall h |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, t in tune.items():
        L.append(f"| {tag} | {t['score']} | {t['eps'] * 100:g} % | {'+'.join(t['guards']) or 'none'} | {t['n_val']} | {t['converged']} | "
                 f"{t['val_fixed']:.2f} / {t['val_fixed_pre']:.2f} ({t['val_off']:.1f} / {t['val_on']:.1f}) | {t['n_iter']} | "
                 f"{t['n_cut_sets']} ({t['n_warm']}) | {t['checks']} / {t['cached']} | {t['relax']} ({t['t_relax']:.0f}) | "
                 f"{t['guards_n']} ({t['t_guard']:.0f}) | {t['collapsed']} / {t['lo_zero']} / {t['hi_one']} | {t['verify_max']:.3f} | {t['wall_h']:.2f} |")
    L += ["", "Faithful Learning to Fix runs (180 validation instances, no guards; `ltfx_tune_*.json`): " +
          ", ".join(f"{k} {v['val_fixed']:.1f} %" for k, v in faithful.items()), ""]
    if holdouts:
        L += ["**Held-out check** (thresholds tuned on the first 180 validation instances, checked on the 180 new ones: "
              "does the guarded reduced UC of each new instance keep a solution within eps of C*?)", "",
              "| thresholds | held-out instances | fail | fail rate |", "|---|---|---|---|"]
        for k, h in holdouts.items():
            L.append(f"| {k} | {h['n_holdout']} | {h['n_fail']} | {h['fail_rate'] * 100:.1f} % |")
        L.append("")
    L += ["## Test: paper metrics (gap to the back-to-back full MILP's dual bound; statistics over feasible instances)", "",
          "| rule | feasible % | gap mean % [95 % CI] | gap max % | runtime mean s | speed-up mean [95 % CI] | speed-up max | fixed mean % (pre-guard) | served % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = rows[k]
        L.append(f"| {k} | {r['feasible']:.1f} | {fmt(r['gap_mean'])} [{fmt(r['gap_mean_ci'][0])}, {fmt(r['gap_mean_ci'][1])}] | "
                 f"{fmt(r['gap_max'])} | {fmt(r['runtime_mean'], 1)} | {fmt(r['speedup_mean'], 1)} [{fmt(r['speedup_mean_ci'][0], 1)}, "
                 f"{fmt(r['speedup_mean_ci'][1], 1)}] | {fmt(r['speedup_max'], 1)} | {fmt(r['fixed_mean'], 1)} ({fmt(r['fixed_pre_guard_mean'], 1)}) | "
                 f"{fmt(r['served_feasible'], 1)} |")
    if famrows:
        L += ["", "## Test: seed families (mean [min, max] over seeds)", "",
              "| family | seeds | feasible % | gap mean % | gap max % | speed-up mean | fixed % | gap to reference % (our convention) | served % | # > 1 % | speed-up (ratio of means) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for fk, a in sorted(famrows.items()):
            L.append(f"| {fam_label(fk)} | {a['seeds']} | {rng_fmt(*a['feasible'], 1)} | {rng_fmt(*a['gap_mean'])} | {rng_fmt(*a['gap_max'])} | "
                     f"{rng_fmt(*a['speedup_mean'], 1)} | {rng_fmt(*a['fixed_mean'], 1)} | {rng_fmt(*a['gap_ref_mean'])} | "
                     f"{rng_fmt(*a['served'], 1)} | {rng_fmt(*a['n_gt1'], 1)} | {rng_fmt(*a['speedup_ratio_of_means'])} |")
    L += ["", "## Test: our earlier convention (gap to the dataset MILP objective, all instances, infeasible → full-MILP fallback)", "",
          "| rule | mean gap % [95 % CI] | median % | max % | # > 1 % | # > 10 % | served % | speed-up (ratio of means) | infeasible |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = rows[k]
        L.append(f"| {k} | {fmt(r['gap_ref_mean'])} [{fmt(r['gap_ref_ci'][0])}, {fmt(r['gap_ref_ci'][1])}] | {fmt(r['gap_ref_median'], 3)} | "
                 f"{fmt(r['gap_ref_max'])} | {r['n_gt1']} | {r['n_gt10']} | {fmt(r['served'], 1)} | {fmt(r['speedup_ratio_of_means'])} | {r['n_infeasible']} |")
    L += ["", "## Paired comparisons against faithful Learning to Fix (rule − reference; bootstrap 95 % CI over instances)", "",
          "Gap to DB and speed-up: instances feasible for both (for a seed family, for every seed). Gap to reference: all instances, fallback included.", "",
          "| rule | reference | Δ gap to DB, pp [CI] | Δ mean speed-up [CI] | Δ log speed-up [CI] | Δ gap to reference, pp [CI] | n |",
          "|---|---|---|---|---|---|---|"]
    for k, p in pairs.items():
        a, ref = k.rsplit(" vs ", 1)
        g, s, ls, gr = p["gap_db"], p["speedup"], p["log_speedup"], p["gap_ref"]
        L.append(f"| {a} | {ref} | {g['diff']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] | {s['diff']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] | "
                 f"{ls['diff']:+.3f} [{ls['ci'][0]:+.3f}, {ls['ci'][1]:+.3f}] | {gr['diff']:+.3f} [{gr['ci'][0]:+.3f}, {gr['ci'][1]:+.3f}] | {g['n']} |")
    L += ["", "## Pareto: lowest mean gap to DB among rules with mean per-instance speed-up ≥ x", "",
          "| speed-up ≥ | best overall | best hybrid | best baseline |", "|---|---|---|---|"]
    for x in PARETO:
        cell = lambda p: f"{p['gap']:.2f} % at {p['speedup']:.1f}× ({p['rule']}; {p['feasible']:.0f} % feasible)" if p else "–"
        L.append(f"| {x}× | {cell(pareto.get(x))} | {cell(pareto.get(f'{x}_hybrid'))} | {cell(pareto.get(f'{x}_baseline'))} |")
    if vs:
        L += ["", "## 180 vs 360 validation instances (same rule, seed and eps)", "",
              "| rule (360) | gap DB mean % 180 → 360 | max % | feasible % | speed-up | test instances above eps (ref.) | Δ gap to ref. 360 − 180, pp [CI] | Δ log speed-up [CI] |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in vs.items():
            a, b = v["r360"], v["r180"]
            g, ls = v["gap_ref_diff_360_minus_180"], v["log_speedup_diff_360_minus_180"]
            L.append(f"| {k} | {fmt(b['gap_mean'])} → {fmt(a['gap_mean'])} | {fmt(b['gap_max'])} → {fmt(a['gap_max'])} | "
                     f"{b['feasible']:.1f} → {a['feasible']:.1f} | {fmt(b['speedup_mean'], 1)} → {fmt(a['speedup_mean'], 1)} | "
                     f"{v['above_eps_180']} → {v['above_eps_360']} | {g['diff']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] | "
                     f"{ls['diff']:+.3f} [{ls['ci'][0]:+.3f}, {ls['ci'][1]:+.3f}] |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
