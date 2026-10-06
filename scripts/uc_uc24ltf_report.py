"""uc24ltf, step 5: report. Paper metrics (Learning to Fix, Sec. IV-C), our earlier convention, time-to-quality from
the stored incumbent logs, served share, back-to-back timing checks, paired comparisons, tuning and compute costs.

    python3 scripts/uc_uc24ltf_report.py

Inputs: data/generated/uc24/test.npz (full MILP: objective, dual bound, time, incumbent log),
results/uc24/uc24ltf_eval_test.jsonl (this study's runs), results/uc24/b3_fix_test{,_extra}.jsonl (stored records of
the earlier uc24 rules), results/uc24/uc24ltf_tune_*.json, uc24ltf_probs.json, data/generated/uc24/uc24ltf_*.json.
Output: results/uc24/uc24ltf_results.md, results/uc24/uc24ltf_results.json.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import inc_from_arrays, time_to_reach  # noqa: E402
from uc_papereval_score import stats  # noqa: E402

ROOT, OUT = os.path.join("data", "generated", "uc24"), os.path.join("results", "uc24")
TG = 24 * 73
SKIP = {"i", "loadavg0", "full_b2b", "t_rel"}


def first_below(inc, target):
    for t, o, *_ in inc:
        if o <= target * (1 + 1e-9):
            return t
    return np.nan


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, len(x), (B, len(x)))
    return [float(v) for v in np.percentile(x[bi].mean(1), [2.5, 97.5])]


def score(name, R, d, idx, incs, t_full, fac=None, extra_ov=None, stored=False):
    """R: list of per-instance records (aligned with idx). Returns the row dict and per-instance arrays."""
    n = len(idx)
    db, ref = d["bound"][idx], d["obj"][idx]
    recs, tm_all, feas = [], np.zeros(n), np.zeros(n, bool)
    for k, r in enumerate(R):
        if stored:
            ok = int(r["fallback"]) == 0 and r["obj"] is not None and np.isfinite(r["obj"])
            tm = r["t_milp"] + r["t_guard"] + d["t_rel"][idx[k]] + (extra_ov[k] if extra_ov is not None else 0.0)
            fixed = r["n_fixed"] / TG
            obj, shed, short = r["obj"], r["shed"], r["short"]
        else:
            ok = bool(r["feasible"])
            tm = r["pre_s"] + r["t_guard"] + r["t_milp"] + r.get("_t_rel", 0.0)
            fixed = r["n_fixed"] / TG
            obj, shed, short = r["obj"], r["shed"], r["short"]
        feas[k], tm_all[k] = ok, tm
        recs.append(dict(obj=obj if ok else None, shed=shed if ok else 1.0, short=short if ok else 1.0, time=tm,
                         feasible=ok, fixed=fixed,
                         obj_any=(obj if ok else ref[k]),                           # infeasible -> full MILP fallback
                         time_any=(tm if ok else tm + t_full[k])))
    row = stats(recs, db, t_full, np.zeros(n), ref, name, d["opt"][idx])
    row["n_infeasible"] = int((~feas).sum())
    f = np.where(feas)[0]
    obj = np.array([recs[k]["obj"] for k in f], float)
    # time-to-quality: first time the full MILP's incumbent log reaches the method's cost (censored at its end)
    if len(f):
        ttq = np.array([time_to_reach(incs[k], obj[j]) for j, k in enumerate(f)])
        reached = np.isfinite(ttq)
        ttq_c = np.where(reached, ttq, t_full[f])
        row.update(ttq_reached=float(reached.mean() * 100), ttq_speedup_median=float(np.median(ttq_c / tm_all[f])),
                   ttq_speedup_mean=float(np.mean(ttq_c / tm_all[f])),
                   ttq_speedup_ratio_of_means=float(ttq_c.mean() / tm_all[f].mean()))
    # anytime: time to a solution within q of the full MILP's final cost
    for q in (0.005, 0.001):
        tf_q = np.array([time_to_reach(incs[k], ref[k] * (1 + q)) for k in range(n)])
        tmq = np.full(n, np.nan)
        if not stored or "inc" in R[0] and isinstance(R[0]["inc"], list):
            for k, r in enumerate(R):
                if feas[k]:
                    ov = tm_all[k] - r["t_milp"]
                    tmq[k] = ov + first_below(r["inc"], ref[k] * (1 + q))
        both = np.isfinite(tf_q) & np.isfinite(tmq)
        row[f"reach_{q * 100:g}%"] = float(np.isfinite(tmq).mean() * 100)
        row[f"reach_{q * 100:g}%_speedup_median_both"] = float(np.median(tf_q[both] / tmq[both])) if both.any() else np.nan
        row[f"reach_{q * 100:g}%_time_median"] = float(np.nanmedian(tmq)) if np.isfinite(tmq).any() else np.nan
    if fac is not None:
        row["speedup_mean_loadcorr"] = float(np.mean(t_full[f] * fac / tm_all[f]))
    row["gap_any_ref_ci"] = boot_ci([(recs[k]["obj_any"] - ref[k]) / ref[k] * 100 for k in range(n)])
    per = dict(feas=feas, time=tm_all,
               gap_db=np.array([(recs[k]["obj"] - db[k]) / recs[k]["obj"] * 100 if feas[k] else np.nan for k in range(n)]),
               gap_any=np.array([(recs[k]["obj_any"] - ref[k]) / ref[k] * 100 for k in range(n)]),
               served=np.array([feas[k] and recs[k]["shed"] < 1e-6 and recs[k]["short"] < 1e-6 for k in range(n)]))
    return row, per


def ff(v, nd=1, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fmt_rows(rows, cols):
    head = "| rule | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    out = head
    for r in rows:
        cells = []
        for key, _, nd in cols:
            v = r.get(key, "")
            if isinstance(v, float):
                cells.append("–" if not np.isfinite(v) else f"{v:.{nd}f}")
            else:
                cells.append(str(v))
        out += f"| {r['method']} | " + " | ".join(cells) + " |\n"
    return out


PAPER = [("feasible", "feas. %", 1), ("gap_mean", "gap to DB mean %", 2), ("gap_median", "median %", 2),
         ("gap_max", "max %", 2), ("time_mean", "time mean s", 1), ("speedup_mean", "speed-up mean", 1),
         ("speedup_median", "median", 1), ("speedup_max", "max", 1), ("fixed_mean", "fixed mean %", 1),
         ("fixed_max", "fixed max %", 1), ("served", "served %", 1)]
OURS = [("old_gap_mean", "gap to MILP obj, all inst. %", 2), ("old_gap_median", "median %", 3),
        ("speedup_ratio_of_means", "speed-up, ratio of means (feasible)", 2), ("old_speedup", "ratio of means, fallback incl.", 2),
        ("n_infeasible", "infeasible", 0), ("speedup_mean_loadcorr", "speed-up mean, load-corrected", 1)]
TTQ = [("ttq_reached", "full MILP reaches the cost %", 1), ("ttq_speedup_median", "TTQ speed-up median", 2),
       ("ttq_speedup_ratio_of_means", "TTQ ratio of means", 2), ("reach_0.5%", "reaches ≤ 0.5 % %", 1),
       ("reach_0.5%_time_median", "time to ≤ 0.5 % median s", 1), ("reach_0.5%_speedup_median_both", "speed-up to ≤ 0.5 % (both reach)", 2),
       ("reach_0.1%", "reaches ≤ 0.1 % %", 1), ("reach_0.1%_speedup_median_both", "speed-up to ≤ 0.1 %", 2)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="evaluation file suffix (uc24ltf_eval_test<tag>.jsonl) and output suffix")
    ap.add_argument("--extra", default="", help="further evaluation jsonl files (same instances, other rules) to merge")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    d = dict(np.load(os.path.join(ROOT, "test.npz")))
    by_i = {}
    for k_f, fn in enumerate([f"uc24ltf_eval_test{a.tag}.jsonl"] + [f for f in a.extra.split(",") if f]):
        for line in open(os.path.join(OUT, fn)):
            r = json.loads(line)
            for nm in [k for k in r if k not in SKIP]:      # GNN rules pay the LP relaxation timed in their own run
                r[nm]["_t_rel"] = r["t_rel"] if "knn" not in nm.lower() else 0.0
            if k_f == 0:
                by_i[r["i"]] = r
            elif r["i"] in by_i:
                by_i[r["i"]].update({k: v for k, v in r.items() if k not in SKIP})
    ev = [by_i[i] for i in sorted(by_i)]
    if a.extra:                                             # keep instances present in every file
        need = set().union(*[set(k for k in r if k not in SKIP) for r in ev])
        ev = [r for r in ev if need <= set(r)]
    idx = np.array([r["i"] for r in ev])
    n = len(idx)
    incs = [inc_from_arrays(d["inc_t"][i], d["inc_obj"][i]) for i in idx]
    t_full = d["time"][idx]
    names = [k for k in ev[0] if k not in SKIP]
    res = {"n": int(n)}
    # ---------------- back-to-back check of the full MILP
    b2b = [r for r in ev if "full_b2b" in r]
    fac = None
    if b2b:
        bi = np.array([r["i"] for r in b2b])
        tb = np.array([r["full_b2b"]["t_milp"] for r in b2b])
        ratio = tb / d["time"][bi]
        fac = float(np.median(ratio))
        res["b2b"] = dict(n=len(b2b), i=bi.tolist(), t_gen=d["time"][bi].tolist(), t_b2b=tb.tolist(), ratio=ratio.tolist(),
                          load_factor_median=fac,
                          obj_diff_pct=[float((r["full_b2b"]["obj"] - d["obj"][r["i"]]) / d["obj"][r["i"]] * 100) for r in b2b],
                          loadavg_gen=d["loadavg"][bi].tolist(), loadavg_b2b=[r["full_b2b"]["loadavg"] for r in b2b])
    # ---------------- this study's runs
    rows, per = [], {}
    full_row = dict(method="full MILP (dataset run, 0.1 %, 300 s)", n=n, feasible=100.0, gap_mean=float(((d["obj"] - d["bound"]) / d["obj"])[idx].mean() * 100),
                    gap_median=float(np.median(((d["obj"] - d["bound"]) / d["obj"])[idx]) * 100),
                    gap_max=float(((d["obj"] - d["bound"]) / d["obj"])[idx].max() * 100), time_mean=float(t_full.mean()),
                    speedup_mean=1.0, speedup_median=1.0, speedup_max=1.0, fixed_mean=0.0, fixed_max=0.0,
                    served=float(((d["shed"] < 1e-6) & (d["short"] < 1e-6))[idx].mean() * 100))
    rows.append(full_row)
    for nm in names:
        R = []
        for r in ev:
            R.append(dict(r[nm]))
        row, p = score(nm, R, d, idx, incs, t_full, fac=fac)
        row["source"] = "this run"
        rows.append(row)
        per[nm] = p
    # ---------------- stored records of the earlier rules (timed under load ~6.7; overheads as in papereval)
    ovh = json.load(open(os.path.join("results", "papereval_overhead.json")))["uc24"]
    lf = ovh["load_factor"]
    t_gnn, t_adeq = np.array(ovh["t_gnn"]) * lf, np.array(ovh["t_adeq"]) * lf
    stored = {}
    for fn in ("b3_fix_test.jsonl", "b3_fix_test_extra.jsonl"):
        for line in open(os.path.join(OUT, fn)):
            r = json.loads(line)
            for k, v in r.items():
                if "|" in k:
                    stored.setdefault(k, {})[r["i"]] = v
    srows = []
    for key in sorted(stored, key=lambda k: (k.split("|")[1], k)):
        if not all(i in stored[key] for i in idx):
            continue
        R = [stored[key][i] for i in idx]
        ov = t_gnn[idx] + (t_adeq[idx] if key.endswith("_g") else 0)
        row, p = score(f"stored: {key}", R, d, idx, incs, t_full, extra_ov=ov, stored=True)
        row["source"] = "stored b3 record (load ~6.7)"
        srows.append(row)
        per[f"stored: {key}"] = p
    # re-run vs stored (same rule, same instances): timing drift
    drift = {}
    for nm in names:
        if nm.startswith("old: "):
            key = nm[5:]
            if key in stored and f"stored: {key}" in per:
                drift[key] = dict(time_ratio_rerun_over_stored_median=float(np.median(per[nm]["time"] / per[f"stored: {key}"]["time"])),
                                  same_obj_share=float(np.mean([abs((ev[k][nm]["obj"] or np.inf) - (stored[key][i]["obj"] or 0))
                                                               <= 1e-6 * abs(stored[key][i]["obj"] or 1) for k, i in enumerate(idx)])))
    res["rerun_vs_stored"] = drift
    # ---------------- paired comparisons (gap to DB, instances feasible for both)
    ltf = [nm for nm in names if nm.startswith("LtF")]
    ours = [nm for nm in names if not nm.startswith("LtF")] + [r["method"] for r in srows]
    pairs = []
    allrows = {r["method"]: r for r in rows + srows}
    for a_ in ltf:
        for b_ in ltf + ours:
            if a_ == b_:
                continue
            both = per[a_]["feas"] & per[b_]["feas"]
            diff = per[b_]["gap_db"][both] - per[a_]["gap_db"][both]
            pairs.append(dict(ltf=a_, other=b_, n=int(both.sum()), diff_mean=float(diff.mean()) if both.any() else np.nan,
                              diff_ci=boot_ci(diff), sp_ltf=allrows[a_].get("speedup_mean"), sp_other=allrows[b_].get("speedup_mean"),
                              gap_ltf=allrows[a_].get("gap_mean"), gap_other=allrows[b_].get("gap_mean")))
    res["pairs"] = pairs
    # ---------------- tuning summary
    tun = []
    for f in sorted(glob.glob(os.path.join(OUT, "uc24ltf_tune_*.json"))):
        r = json.load(open(f))
        relax = [x for h in r["history"] for x in h.get("relax", [])]
        alts = [len(h.get("release_sizes", [])) for h in r["history"] if "failed" in h]
        sizes = [s for h in r["history"] for s in h.get("release_sizes", [])]
        nm = f"LtF{'+guards' if r['guarded'] else ''} {r['model']} eps={r['eps'] * 100:g}%"
        test_fixed = allrows.get(nm, allrows.get(nm + " (not converged)", {})).get("fixed_mean")
        tun.append(dict(rule=nm, converged=r["converged"], n_val=r["n_val"], iterations=r["n_iter"],
                        failed_instances=len(r.get("failed_instances", [])), alternatives_mean=float(np.mean(alts)) if alts else 0.0,
                        release_size_mean=float(np.mean(sizes)) if sizes else 0.0, n_relax=len(relax),
                        n_relax_limit=int(sum(1 for x in relax if "imit" in str(x.get("status", "")))),
                        checks_solved=r["stats"]["n_check_solves"], checks_cached=r["stats"]["n_check_cached"],
                        val_fixed=r["val_fixed_share"] * 100, val_fixed_guarded=r.get("val_fixed_share_guarded", np.nan) * 100
                        if r.get("val_fixed_share_guarded") is not None else np.nan,
                        test_fixed=test_fixed, verify_max=r["verify_max"] * 100, n_verify_fail=r.get("n_verify_fail", 0),
                        wall_h=r["wall_s"] / 3600, cpu_h=r["cpu_s"] / 3600,
                        collapsed=int(np.sum(np.isclose(r["lo"], r["hi"]))), lo_zero=int(np.sum(np.array(r["lo"]) <= 1e-9)),
                        hi_one=int(np.sum(np.array(r["hi"]) >= 1 - 1e-9))))
    res["tuning"] = tun
    # ---------------- compute (solver core-hours)
    cost = {}
    for nm in ("uc24ltf_val", "uc24ltf_train_lab"):
        p = os.path.join(ROOT, f"{nm}.json")
        if os.path.exists(p):
            g = json.load(open(p))
            cost[nm] = dict(n=g["n"], milp_wall_h=g["milp_time_sum_s"] / 3600, milp_cpu_h=g["milp_cpu_sum_s"] / 3600,
                            run_wall_h=g["wall_s"] / 3600, time_mean=g["milp_time_mean"], time_median=g["milp_time_median"],
                            opt_frac=g["milp_opt_frac"], gap_mean=g["milp_gap_mean_%"], gap_max=g["milp_gap_max_%"],
                            cfg=g["cfg"], loadavg=g["loadavg_mean"])
    cost["tuning"] = dict(wall_h=float(sum(t["wall_h"] for t in tun)), cpu_h=float(sum(t["cpu_h"] for t in tun)))
    ev_milp = sum(r[nm]["t_milp"] for r in ev for nm in names) + sum(r["full_b2b"]["t_milp"] for r in b2b)
    cost["test_eval"] = dict(milp_wall_h=ev_milp / 3600)
    res["cost"] = cost
    res["rows"] = rows
    res["stored_rows"] = srows
    pj = json.load(open(os.path.join(OUT, "uc24ltf_probs.json")))
    res["probs"] = pj
    # ---------------- markdown
    md = [f"# uc24ltf: Learning to Fix on the 24-hour benchmark (test, {n} instances)\n"]
    md.append("Generated by `scripts/uc_uc24ltf_report.py`. Method write-up: `docs/methods/uc24ltf.md`. Evaluation files: "
              + ", ".join(f"`{f}`" for f in [f"uc24ltf_eval_test{a.tag}.jsonl"] + [f for f in a.extra.split(",") if f])
              + ". \"(not converged)\": tuning stopped at its budget, thresholds violate some validation instances.\n")
    md.append("## Paper metrics (gap to the full MILP's dual bound; mean of per-instance speed-ups vs the dataset full-MILP "
              "time; method time includes inference, LP relaxation for GNN features, guards; statistics over feasible "
              "instances)\n")
    md.append(fmt_rows(rows + srows, PAPER))
    md.append("\n`stored:` rows are the earlier uc24 runs (`b3_fix_test*.jsonl`), timed in the earlier study under a heavier load "
              "(1-min load average ~6.7, two solver workers); `old:` rows are the same rules re-run in this study's worker.\n")
    md.append("\n## Our earlier convention, load correction\n")
    md.append("Gap to the full MILP's objective over all instances, an infeasible reduced problem charged with the full MILP "
              "(cost and time added); speed-up as a ratio of mean times. Load-corrected speed-up: full-MILP time × the "
              "median back-to-back / dataset time ratio" + (f" ({fac:.2f})" if fac else "") + ".\n")
    md.append(fmt_rows(rows[1:] + srows, OURS))
    md.append("\n## Time to quality (full MILP's stored incumbent log)\n")
    md.append(fmt_rows(rows[1:] + srows, TTQ))
    if b2b:
        md.append(f"\n## Back-to-back check ({len(b2b)} instances, full MILP re-solved in the evaluation worker)\n")
        md.append("| i | dataset time s | back-to-back s | ratio | objective diff % |\n|---|---|---|---|---|\n")
        for k in range(len(b2b)):
            md.append(f"| {res['b2b']['i'][k]} | {res['b2b']['t_gen'][k]:.1f} | {res['b2b']['t_b2b'][k]:.1f} | "
                      f"{res['b2b']['ratio'][k]:.2f} | {res['b2b']['obj_diff_pct'][k]:+.4f} |\n")
        md.append(f"\nMedian ratio {fac:.2f}. Re-run vs stored records (same rule, median method-time ratio): "
                  + ", ".join(f"{k}: {v['time_ratio_rerun_over_stored_median']:.2f} (same cost on {v['same_obj_share'] * 100:.0f} %)"
                              for k, v in drift.items()) + "\n")
    md.append("\n## Paired comparisons (gap to DB, other − LtF, instances feasible for both; bootstrap 95 % CI)\n")
    md.append("Shown: pairs whose mean speed-ups are within a factor 2 of each other, and every pair with a re-run "
              "(`old:`) rule; all pairs are in the JSON.\n\n")
    md.append("| LtF rule | speed-up | other rule | speed-up | n | gap diff pp [95 % CI] |\n|---|---|---|---|---|---|\n")
    for p in pairs:
        near = p["sp_ltf"] and p["sp_other"] and 0.5 <= p["sp_other"] / p["sp_ltf"] <= 2.0
        if not (near or p["other"].startswith("old:")):
            continue
        md.append(f"| {p['ltf']} | {ff(p['sp_ltf'])} | {p['other']} | {ff(p['sp_other'])} | {p['n']} | "
                  f"{ff(p['diff_mean'], 2, True)} [{ff(p['diff_ci'][0], 2, True)}, {ff(p['diff_ci'][1], 2, True)}] |\n")
    md.append("\n## Tuning (validation)\n")
    tc = [("converged", "conv."), ("n_val", "n val"), ("iterations", "iter."), ("failed_instances", "inst. failed"),
          ("alternatives_mean", "alt./cut"), ("release_size_mean", "release size"), ("n_relax", "relax MILPs"),
          ("n_relax_limit", "at limit"), ("checks_solved", "checks solved"), ("checks_cached", "cached"),
          ("val_fixed", "val fixed %"), ("val_fixed_guarded", "val fixed after guards %"), ("test_fixed", "test fixed %"),
          ("collapsed", "collapsed"), ("lo_zero", "lo = 0"), ("verify_max", "witness max %"), ("n_verify_fail", "verify fails"),
          ("cpu_h", "CPU h")]
    md.append("| rule | " + " | ".join(c[1] for c in tc) + " |\n|---|" + "---|" * len(tc) + "\n")
    for t in tun:
        md.append(f"| {t['rule']} | " + " | ".join(
            (f"{t[k]:.2f}" if isinstance(t[k], float) and np.isfinite(t[k]) else ("–" if isinstance(t[k], float) else str(t[k])))
            for k, _ in tc) + " |\n")
    md.append("\n## Compute\n")
    md.append("```\n" + json.dumps(cost, indent=1, default=float) + "\n```\n")
    md.append("\n## Probabilities (validation)\n")
    md.append("```\n" + json.dumps(pj, indent=1, default=float) + "\n```\n")
    lines, prev = [], ""
    for ln in "".join(md).split("\n"):                    # blank line before every table and heading
        if (ln.startswith("|") and not prev.startswith("|") and prev.strip()) or (ln.startswith("#") and prev.strip()):
            lines.append("")
        lines.append(ln)
        prev = ln
    md = ["\n".join(lines)]
    open(os.path.join(OUT, f"uc24ltf_results{a.tag}.md"), "w").write("".join(md))
    json.dump(res, open(os.path.join(OUT, f"uc24ltf_results{a.tag}.json"), "w"), indent=1,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (bool(o) if isinstance(o, np.bool_) else float(o)))
    print("".join(md))
