"""Learning to Fix (faithful), step 4: tables for results/uc12/ltfx_results.md / .json.

    python3 scripts/uc_ltfx_report.py [--split test_fresh]

Paper metrics (Section IV-C): feasibility rate of the reduced MILP; on feasible instances the optimality gap
(C - DB) / C against the back-to-back full MILP's dual bound DB = obj * (1 - MIP gap); runtime and per-instance
speed-up T_MILP / T_method (mean and max) with inference and guard time included; fixed share. Our earlier
convention: gap to the dataset MILP objective over all instances (an infeasible reduced MILP falls back to the full
MILP and pays both times), served share (no shedding / spill / reserve shortfall), median gap, # gaps > 10 %,
speed-up as a ratio of mean times.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.ucdata import load  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize(path, d):
    recs = [json.loads(line) for line in open(path)]
    by = {}
    for r in recs:
        by.setdefault(r["rule"], {})[r["i"]] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    full = by["full MILP"]
    DB = np.array([full[i]["obj"] * (1 - full[i]["mip_gap"]) for i in common])
    t_full = np.array([full[i]["time"] for i in common])
    ref = d["obj"][common]
    rows = {}
    for rule, rr in by.items():
        R = [rr[i] for i in common]
        feas = np.array([r["feasible"] for r in R])
        cost = np.array([r["obj"] for r in R])
        t_m = np.array([r["time"] + r["pre_s"] + r["relax_s"] + r["guard_s"] for r in R])
        with np.errstate(invalid="ignore", divide="ignore"):
            gap_db = (cost - DB) / cost * 100
        sp = t_full / t_m
        # our convention (fallback to the full MILP when infeasible)
        c_fb = np.array([r["obj"] if r["feasible"] else r["fb_obj"] for r in R])
        t_fb = np.array([tm + (0 if r["feasible"] else r["fb_time"]) for tm, r in zip(t_m, R)])
        shed = np.array([(r["shed"] if r["feasible"] else r["fb_shed"]) for r in R])
        short = np.array([(r["short"] if r["feasible"] else r["fb_short"]) for r in R])
        g_ref = (c_fb - ref) / ref * 100
        served = (shed < 1e-6) & (short < 1e-6)
        f = feas
        rows[rule] = dict(
            n=len(R), feasible=float(f.mean() * 100), n_infeasible=int((~f).sum()), infeasible_idx=[int(i) for i, x in zip(common, f) if not x],
            gap_db_mean=float(gap_db[f].mean()) if f.any() else np.nan, gap_db_max=float(gap_db[f].max()) if f.any() else np.nan,
            gap_db_median=float(np.median(gap_db[f])) if f.any() else np.nan,
            runtime_mean=float(t_m[f].mean()) if f.any() else np.nan, runtime_max=float(t_m[f].max()) if f.any() else np.nan,
            speedup_mean=float(sp[f].mean()) if f.any() else np.nan, speedup_max=float(sp[f].max()) if f.any() else np.nan,
            speedup_median=float(np.median(sp[f])) if f.any() else np.nan,
            fixed_mean=float(np.mean([r["fixed_share"] for r in R]) * 100), fixed_max=float(np.max([r["fixed_share"] for r in R]) * 100),
            gap_ref_mean=float(g_ref.mean()), gap_ref_mean_ci=boot_ci(g_ref), gap_ref_median=float(np.median(g_ref)),
            gap_ref_max=float(g_ref.max()), n_gt10=int((g_ref > 10).sum()), n_gt1=int((g_ref > 1).sum()),
            served=float(served.mean() * 100), speedup_ratio_of_means=float(t_full.mean() / t_fb.mean()),
            time_mean_incl_fallback=float(t_fb.mean()), milp_only_time_mean=float(np.mean([r["time"] for r in R])),
            pre_s_mean=float(np.mean([r["pre_s"] + r["relax_s"] + r["guard_s"] for r in R])),
            gap_ref_per_instance=[float(x) for x in g_ref])
    return rows, common, t_full


def fmt(x, p=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def tuning_details():
    """per-run tuning statistics for docs/methods/ltfx.md (iterations, failing instances, relaxation outcomes)"""
    import collections
    rows = []
    P = np.load(os.path.join(OUT, "ltfx_probs.npz"))
    from otsl.ltfx import fix_masks
    for f in sorted(glob.glob(os.path.join(OUT, "ltfx_tune_*.json"))):
        r = json.load(open(f))
        h = r["history"]
        lo, hi = np.array(r["lo"]), np.array(r["hi"])
        off, on = fix_masks(P[f"{r['model']}_tf"][:60], lo, hi)
        rel = [y for x in h for y in x.get("relax", [])]
        st = collections.Counter(y["status"] for y in rel)
        chk = collections.Counter(x.get("check_status") for x in h if "failed" in x)
        rows.append(dict(run=f"{r['model']} eps={r['eps'] * 100:g}%", converged=r["converged"], iters=r["n_iter"],
                         failing_instances=len({x["failed"] for x in h if "failed" in x}),
                         check_fail_status=dict(chk), relax_status=dict(st), alternatives_per_cut=float(np.mean([len(x["release_sizes"]) for x in h if "release_sizes" in x])),
                         release_size_mean=float(np.mean([s for x in h for s in x.get("release_sizes", [])])),
                         val_fixed=r["val_fixed_share"] * 100, test_fixed=float((off | on).mean() * 100),
                         test_fixed_off=float(off.mean() * 100), test_fixed_on=float(on.mean() * 100),
                         collapsed=int(((hi - lo) < 1e-6).sum()), width_median=float(np.median(hi - lo)),
                         wall_h=r["wall_s"] / 3600, cpu_h=r["cpu_s"] / 3600, verify_max_pct=r["verify_max"] * 100,
                         t_check_h=r["stats"]["t_check"] / 3600, t_relax_h=r["stats"]["t_relax"] / 3600,
                         t_master_s=r["stats"]["t_master"], n_check=r["stats"]["n_check_solves"], n_cached=r["stats"]["n_check_cached"],
                         n_relax=r["stats"]["n_relax"]))
    return rows


ORDER = ["full MILP", "knn tau=0.5", "knn const 0.1", "knn const 0.05", "knn const 0.01", "knn worst",
         "knn ltf eps=10%", "knn ltf eps=5%", "knn ltf eps=1%", "st const 0.1", "st const 0.05", "st const 0.01", "st worst",
         "st ltf eps=10%", "st ltf eps=5%", "st ltf eps=1%", "rl ltf eps=1%", "bce ltf eps=1%",
         "ours: error-cost + adequacy guard 90% (BCE)", "ours: combined st+error-cost+adequacy+LP guard 95%",
         "ours: combined st+error-cost+adequacy+LP guard 98%"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--tag", default="")
    ap.add_argument("--details", action="store_true", help="print per-run tuning statistics only")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.details:
        print(json.dumps(tuning_details(), indent=1))
        sys.exit(0)
    d = load(os.path.join(ROOT, f"{a.split}.npz"))
    rows, common, t_full = summarize(os.path.join(OUT, f"ltfx_eval_{a.split}{a.tag}.jsonl"), d)
    tune = {}
    for f in sorted(glob.glob(os.path.join(OUT, "ltfx_tune_*.json"))):
        r = json.load(open(f))
        tune[f"{r['model']} eps={r['eps'] * 100:g}%"] = {k: r[k] for k in ("converged", "val_fixed_share", "n_iter", "n_cut_sets",
                                                                       "verify_max", "verify_ok", "wall_s", "cpu_s", "n_val")} | {
            "stats": r["stats"]}
    out = {"split": a.split, "n": len(common), "full_milp_time_mean": float(t_full.mean()), "rows": rows, "tuning": tune}
    with open(os.path.join(OUT, f"ltfx_results{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    keys = [k for k in ORDER if k in rows] + sorted(k for k in rows if k not in ORDER)
    L = [f"# Learning to Fix, faithful implementation: results ({a.split}, first {len(common)} instances)", "",
         f"Full MILP back to back: {t_full.mean():.1f} s mean. Generated by `scripts/uc_ltfx_report.py`.", "",
         "## Paper metrics (gap to the full MILP's dual bound; statistics over feasible instances)", "",
         "| rule | feasible % | gap mean % | gap max % | runtime mean s | runtime max s | speed-up mean | speed-up max | fixed mean % | fixed max % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = rows[k]
        L.append(f"| {k} | {fmt(r['feasible'], 1)} | {fmt(r['gap_db_mean'])} | {fmt(r['gap_db_max'])} | {fmt(r['runtime_mean'], 1)} | "
                 f"{fmt(r['runtime_max'], 1)} | {fmt(r['speedup_mean'], 1)} | {fmt(r['speedup_max'], 1)} | {fmt(r['fixed_mean'], 1)} | {fmt(r['fixed_max'], 1)} |")
    L += ["", "## Our earlier convention (gap to the dataset MILP objective, all instances, infeasible -> full-MILP fallback)", "",
          "| rule | mean gap % [95 % CI] | median gap % | max gap % | # > 1 % | # > 10 % | served % | speed-up (ratio of means) | infeasible |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = rows[k]
        L.append(f"| {k} | {fmt(r['gap_ref_mean'])} [{fmt(r['gap_ref_mean_ci'][0])}, {fmt(r['gap_ref_mean_ci'][1])}] | "
                 f"{fmt(r['gap_ref_median'], 3)} | {fmt(r['gap_ref_max'])} | {r['n_gt1']} | {r['n_gt10']} | {fmt(r['served'], 1)} | "
                 f"{fmt(r['speedup_ratio_of_means'], 2)} | {r['n_infeasible']} |")
    L += ["", "## Tuning (validation)", "",
          "| model, eps | converged | val fixed % | iterations | cut sets | check solves / cached | relaxation MILPs (s) | master solves (s) | witness max increase % | wall h | CPU h |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in tune.items():
        s = t["stats"]
        L.append(f"| {k} | {t['converged']} | {t['val_fixed_share'] * 100:.2f} | {t['n_iter']} | {t['n_cut_sets']} | "
                 f"{s['n_check_solves']} / {s['n_check_cached']} | {s['n_relax']} ({s['t_relax']:.0f}) | {s['n_master']} ({s['t_master']:.0f}) | "
                 f"{t['verify_max'] * 100:.3f} | {t['wall_s'] / 3600:.2f} | {t['cpu_s'] / 3600:.2f} |")
    # paired comparison at matched speed: every LtF / threshold rule against our rule with the closest speed-up
    ours = [k for k in keys if k.startswith("ours")]
    L += ["", "## Paired comparison with our rules (gap to the dataset MILP objective, per-instance difference, bootstrap 95 % CI)", "",
          "| rule (speed-up) | our closest-speed rule (speed-up) | rule mean gap % | ours mean gap % | difference ours - rule, pp [95 % CI] |",
          "|---|---|---|---|---|"]
    pairs = {}
    for k in keys:
        if k.startswith("ours") or k == "full MILP" or not ours:
            continue
        r = rows[k]
        o = min(ours, key=lambda x: abs(np.log(rows[x]["speedup_ratio_of_means"] / r["speedup_ratio_of_means"])))
        diff = np.array(rows[o]["gap_ref_per_instance"]) - np.array(r["gap_ref_per_instance"])
        ci = boot_ci(diff)
        pairs[k] = dict(ours=o, diff_mean=float(diff.mean()), ci=ci)
        L.append(f"| {k} ({r['speedup_ratio_of_means']:.1f}x) | {o} ({rows[o]['speedup_ratio_of_means']:.1f}x) | {r['gap_ref_mean']:.2f} | "
                 f"{rows[o]['gap_ref_mean']:.2f} | {diff.mean():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    out["paired"] = pairs
    with open(os.path.join(OUT, f"ltfx_results{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open(os.path.join(OUT, f"ltfx_results{a.tag}.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

