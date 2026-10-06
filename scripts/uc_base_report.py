"""No-learning baselines and fair solver budgets: tables for results/<bench>/base_results.{md,json}.

    python3 scripts/uc_base_report.py --bench uc12
    python3 scripts/uc_base_report.py --bench uc24

Inputs: results/<bench>/base_eval_test.jsonl (scripts/uc_base_eval.py: every run of an instance back to back on one
core), base_val.jsonl / base_val_select.json (scripts/uc_base_val.py), base_tune_lp_1.json (scripts/uc_base_tune.py),
the stored reference records (uc12: hybrid_eval_test_fresh.jsonl; uc24: uc24ltf_eval_test.jsonl) and the datasets.

Paper metrics (Learning to Fix, Sec. IV-C): feasibility rate; over feasible instances the gap (C - DB) / C to the
reference dual bound DB (uc12: obj * (1 - mip_gap) of the stored back-to-back full MILP of the hybrid study; uc24:
the dataset run's bound; the same DB for every row), the per-instance speed-up T_MILP / T_method with T_MILP the
full MILP re-timed in the same process (mean, median, ratio of mean times), fixed share after guards, served share.
Same budget: the full MILP's best incumbent after exactly the method's per-instance time (its incumbent log), and
time-to-quality (when the full MILP first reaches the method's cost). Paired comparisons: per-instance differences,
instance bootstrap 95 % CIs (2,000 resamples, seed 0).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import load_b3, time_to_reach  # noqa: E402
from otsl.base import boot_ci, incumbent_at, paired, time_to_gap  # noqa: E402

CFG = {"uc12": dict(T=12, test="data/generated/uc12/test_fresh.npz", val=["data/generated/uc12/val.npz",
                                                                            "data/generated/uc12/val_extra.npz"],
                    refs=["hybrid (he_bce_s0_e1_n360)", "LtF kNN eps=1%"], tl=60.0),
       "uc24": dict(T=24, test="data/generated/uc24/test.npz", val=["data/generated/uc24/uc24ltf_val.npz"],
                    refs=["LtF kNN eps=1%", "ours: guarded 95% (imitation GNN)"], tl=300.0)}
G = 73
GAPS = ("full gap=0.25%", "full gap=0.5%", "full gap=1%")


def jl(p):
    return [json.loads(x) for x in open(p)]


def ref_bounds(bench, d, idx):
    """reference dual bound and objective (the ones the reference tables used), stored full-MILP time"""
    if bench == "uc12":
        full = {r["i"]: r for r in jl("results/uc12/hybrid_eval_test_fresh.jsonl") if r["rule"] == "full MILP"}
        db = np.array([full[i]["obj"] * (1 - full[i]["mip_gap"]) for i in idx])
        t_st = np.array([full[i]["time"] for i in idx])
        return db, d["obj"][idx], t_st
    return d["bound"][idx], d["obj"][idx], d["time"][idx]


def lp_gap_stats(d, idx, db=None):
    """LP relaxation vs the MILP: (C* - LP) / C*, (DB - LP) / DB, integral share, rounding agreement (aligned)"""
    from otsl.fixpolicy import align_to_prediction
    from otsl.uc import load_rts_gmlc
    sysm = load_rts_gmlc()
    c, lp = d["obj"][idx], d["c_rel"][idx]
    g = (c - lp) / c * 100
    out = dict(n=len(idx), gap_to_opt_mean=float(g.mean()), gap_to_opt_median=float(np.median(g)),
               gap_to_opt_max=float(g.max()), gap_to_opt_p90=float(np.percentile(g, 90)))
    if db is not None:
        gd = (db - lp) / db * 100
        out.update(gap_to_db_mean=float(gd.mean()), gap_to_db_median=float(np.median(gd)), gap_to_db_max=float(gd.max()),
                   db_below_lp=int((gd < -1e-9).sum()))
    u = d["u_rel"][idx]
    dist = np.minimum(u, 1 - u)
    integ = dist <= 1e-6
    agree, agree_int, wrong_int = [], [], []
    for k, i in enumerate(idx):
        r = (u[k] > 0.5).astype(np.int8)
        ua = align_to_prediction(sysm, d["u"][i], r, d["u0"][i])
        eq = ua == r
        agree.append(eq.mean())
        agree_int.append(eq[integ[k]].mean())
        wrong_int.append(int((~eq & integ[k]).sum()))
    out.update(integral_share=float(integ.mean() * 100), integral_share_0p05=float((dist <= 0.05).mean() * 100),
               rounded_agrees=float(np.mean(agree) * 100), integral_agrees=float(np.mean(agree_int) * 100),
               wrong_integral_per_instance=float(np.mean(wrong_int)))
    return out


def per_run(recs, name, db, TG):
    """per-instance arrays of one run name"""
    n = len(recs)
    feas, cost, tm, fixed, served, inc = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), \
        np.zeros(n, bool), []
    for k, r in enumerate(recs):
        x = r["runs"][name]
        feas[k] = bool(x["feasible"])
        tm[k] = x.get("t_method", x.get("time"))
        if feas[k]:
            cost[k] = x["obj"]
            served[k] = (x.get("shed") or 0) < 1e-6 and (x.get("short") or 0) < 1e-6
            fixed[k] = x.get("n_fixed", 0) / TG
        inc.append(x.get("inc", []))
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(feas, (cost - db) / cost * 100, np.nan)
    return dict(feas=feas, cost=cost, tm=tm, gap=gap, fixed=fixed, served=served, inc=inc)


def row_stats(name, P, t_full):
    f = P["feas"]
    out = dict(method=name, n=int(len(f)), feasible=float(f.mean() * 100))
    if not f.any():
        return out
    g, tm, tf = P["gap"][f], P["tm"][f], t_full[f]
    sp = tf / tm
    out.update(gap_mean=float(g.mean()), gap_ci=boot_ci(g), gap_median=float(np.median(g)), gap_max=float(g.max()),
               time_mean=float(tm.mean()), speedup_mean=float(sp.mean()), speedup_ci=boot_ci(sp),
               speedup_median=float(np.median(sp)), speedup_ratio_of_means=float(tf.mean() / tm.mean()),
               fixed_mean=float(np.nanmean(P["fixed"][f]) * 100), served=float(P["served"][f].mean() * 100))
    return out


def same_budget(P, full_inc, full_t, db):
    """the full MILP's best incumbent after the method's time; time-to-quality"""
    n = len(P["feas"])
    g_full = np.full(n, np.nan)
    ttq = np.full(n, np.nan)
    for k in range(n):
        c = incumbent_at(full_inc[k], P["tm"][k])
        if np.isfinite(c):
            g_full[k] = (c - db[k]) / c * 100
        if P["feas"][k]:
            t = time_to_reach(full_inc[k], P["cost"][k])
            ttq[k] = t if np.isfinite(t) else full_t[k]           # censored at the full MILP's end
    f = P["feas"]
    both = f & np.isfinite(g_full)
    out = dict(full_has_solution=float(np.isfinite(g_full).mean() * 100),
               full_gap_mean_where_solution=float(np.nanmean(g_full)) if np.isfinite(g_full).any() else np.nan,
               method_gap_mean=float(np.nanmean(P["gap"][f])) if f.any() else np.nan,
               paired_method_minus_full=paired(P["gap"], g_full),
               method_better=int((P["gap"][both] < g_full[both] - 1e-9).sum()),
               full_better=int((g_full[both] < P["gap"][both] - 1e-9).sum()),
               full_none_method_ok=int((f & ~np.isfinite(g_full)).sum()), n=int(n))
    if f.any():
        out.update(ttq_speedup_median=float(np.median(ttq[f] / P["tm"][f])),
                   ttq_reached=float(np.mean([np.isfinite(time_to_reach(full_inc[k], P["cost"][k])) for k in np.where(f)[0]]) * 100))
    return out, g_full


def fmt(v, nd=2, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(c, nd=2, sign=False):
    return f"[{fmt(c[0], nd, sign)}, {fmt(c[1], nd, sign)}]"


def main(bench, ev="", sel_path="", tune_path="", out_dir=""):
    cfg = CFG[bench]
    R = os.path.join("results", bench)
    recs = sorted(jl(ev or os.path.join(R, "base_eval_test.jsonl")), key=lambda r: r["i"])
    d = load_b3(cfg["test"])
    idx = np.array([r["i"] for r in recs])
    n = len(idx)
    TG = cfg["T"] * G
    db, c_ref, t_stored = ref_bounds(bench, d, idx)
    full = per_run(recs, "full", db, TG)
    t_full = full["tm"]
    names = [k for k in recs[0]["runs"] if k != "full"]
    names = [k for k in names if all(k in r["runs"] for r in recs)]
    P = {nm: per_run(recs, nm, db, TG) for nm in names}
    rows = [dict(row_stats("full MILP (0.1 %, re-timed)", full, t_full), kind="full")]
    for nm in names:
        kind = recs[0]["runs"][nm].get("kind", "full")
        rows.append(dict(row_stats(nm, P[nm], t_full), kind=kind, key=nm))
    res = dict(bench=bench, n=n, instances=idx.tolist())
    # ------------------------------------------------ timing and reproducibility checks
    own_db = np.array([r["runs"]["full"]["bound"] for r in recs])
    res["timing"] = dict(full_time_mean=float(t_full.mean()), full_time_median=float(np.median(t_full)),
                         full_at_limit=float(np.mean([r["runs"]["full"]["status"] != "optimal" for r in recs]) * 100),
                         stored_full_time_mean=float(t_stored.mean()),
                         ratio_retimed_over_stored_median=float(np.median(t_full / t_stored)),
                         same_full_obj_share=float(np.mean(np.abs(full["cost"] - c_ref) <= 1e-6 * c_ref) * 100),
                         loadavg_mean=float(np.mean([r["loadavg0"] for r in recs])),
                         own_db_minus_ref_db_mean_pct=float(np.mean((own_db - db) / db * 100)),
                         u_rel_maxdiff=float(max(r["u_rel_maxdiff"] for r in recs)), t_rel_mean=float(np.mean([r["t_rel"] for r in recs])))
    chk = {}
    for nm in names:
        x0 = recs[0]["runs"][nm]
        if "stored_obj" not in x0:
            continue
        so = np.array([r["runs"][nm]["stored_obj"] if r["runs"][nm]["stored_obj"] is not None else np.nan for r in recs], float)
        st = np.array([r["runs"][nm]["stored_time"] for r in recs], float)
        mine = np.array([r["runs"][nm]["time"] for r in recs], float)
        same = [(np.isnan(a) and not r["runs"][nm]["feasible"]) or (r["runs"][nm]["feasible"] and abs(r["runs"][nm]["obj"] - a) <= 1e-6 * abs(a))
                for a, r in zip(so, recs)]
        nfx = [r["runs"][nm]["n_fixed_pre"] == r["runs"][nm]["stored_n_fixed"] or r["runs"][nm]["n_fixed"] == r["runs"][nm]["stored_n_fixed"]
               for r in recs]
        chk[nm] = dict(same_obj_share=float(np.mean(same) * 100), same_n_fixed_share=float(np.mean(nfx) * 100),
                       solver_time_ratio_median=float(np.median(mine / st)))
    res["reference_rerun_vs_stored"] = chk
    # ------------------------------------------------ LP relaxation gap (test instances used, and validation)
    res["lp_relaxation"] = dict(test=lp_gap_stats(d, idx, db))
    parts = [load_b3(p) for p in cfg["val"]]
    dv = {k: np.concatenate([p[k] for p in parts]) for k in ("obj", "c_rel", "u_rel", "u", "u0")}
    res["lp_relaxation"]["val"] = lp_gap_stats(dv, np.arange(len(dv["obj"])))
    # ------------------------------------------------ paired comparisons and same budget
    pairs = []
    for ref in cfg["refs"]:
        if ref not in P:
            continue
        for nm in names:
            if nm == ref:
                continue
            a, b = P[nm], P[ref]
            both = a["feas"] & b["feas"]
            lg = np.where(both, np.log(b["tm"] / a["tm"]), np.nan)       # log speed-up difference (same T_MILP)
            pairs.append(dict(method=nm, ref=ref, n_both=int(both.sum()), d_gap=paired(np.where(both, a["gap"], np.nan), b["gap"]),
                              d_log_speedup=paired(lg, np.zeros(n))))
    res["paired"] = pairs
    sb = {}
    gfull = {}
    for nm in names:
        if recs[0]["runs"][nm].get("kind") in ("ref", "base", "e2e"):
            sb[nm], gfull[nm] = same_budget(P[nm], full["inc"], t_full, db)
    res["same_budget"] = sb
    # loose gaps: when would the 0.1 % run's own trace have reached them (sanity check of the runs)
    tr = [r["runs"]["full"].get("trace", []) for r in recs]
    res["loose_gap_from_trace"] = {g: float(np.nanmedian([time_to_gap(t, float(g.split("=")[1][:-1]) / 100) for t in tr]))
                                   for g in GAPS if g in P}
    # ------------------------------------------------ validation: selection and tuning
    sel = json.load(open(sel_path or os.path.join(R, "base_val_select.json")))
    tune = json.load(open(tune_path or os.path.join(R, "base_tune_lp_1.json")))
    res["val_select"] = sel
    res["ltf_lp_tuning"] = {k: tune[k] for k in ("converged", "n_val", "val_fixed_share", "val_fixed_off", "val_fixed_on",
                                                 "n_iter", "n_cut_sets", "verify_max", "n_verify_fail", "wall_s", "cpu_s")}
    res["ltf_lp_tuning"].update(lo_zero=int(np.sum(np.array(tune["lo"]) <= 1e-9)), hi_one=int(np.sum(np.array(tune["hi"]) >= 1 - 1e-9)),
                                collapsed=int(np.sum(np.isclose(tune["lo"], tune["hi"]))))
    res["rows"] = rows
    res["per_instance"] = {nm: dict(gap=P[nm]["gap"].tolist(), time=P[nm]["tm"].tolist(), feasible=P[nm]["feas"].tolist(),
                                    full_gap_same_budget=gfull[nm].tolist() if nm in gfull else None) for nm in names}
    res["per_instance"]["full"] = dict(gap=full["gap"].tolist(), time=t_full.tolist(), db_ref=db.tolist())
    out_dir = out_dir or R
    json.dump(res, open(os.path.join(out_dir, "base_results.json"), "w"), indent=1, default=float)
    write_md(bench, res, out_dir)
    return res


def write_md(bench, res, out_dir):
    cfg = CFG[bench]
    L = [f"# No-learning baselines and fair solver budgets ({bench})", "",
         f"Generated by `scripts/uc_base_report.py --bench {bench}`; method: [`docs/methods/base.md`](../../docs/methods/base.md). "
         f"{res['n']} test instances ({'first 60 of test_fresh' if bench == 'uc12' else 'test.npz, instances ' + str(res['instances'][0]) + '–' + str(res['instances'][-1])}), "
         f"every run back to back on one core.", ""]
    t = res["timing"]
    L += [f"Full MILP re-timed: mean {t['full_time_mean']:.1f} s (median {t['full_time_median']:.1f} s, {t['full_at_limit']:.0f} % at the "
          f"{cfg['tl']:.0f} s limit); stored reference run {t['stored_full_time_mean']:.1f} s (median ratio re-timed / stored "
          f"{t['ratio_retimed_over_stored_median']:.2f}); same objective as the stored run on {t['same_full_obj_share']:.0f} % of the "
          f"instances; 1-min load average {t['loadavg_mean']:.1f}. LP relaxation {t['t_rel_mean']:.2f} s (identical to the stored one: "
          f"max |Δu| {t['u_rel_maxdiff']:.1e}).", ""]
    if res["reference_rerun_vs_stored"]:
        L += ["Reference rules re-run from their saved thresholds / probabilities (reduced MILP identical to the stored run?):", "",
              "| rule | same objective % | same fixed count % | solver time re-run / stored (median) |", "|---|---|---|---|"]
        for nm, v in res["reference_rerun_vs_stored"].items():
            L.append(f"| {nm} | {v['same_obj_share']:.0f} | {v['same_n_fixed_share']:.0f} | {v['solver_time_ratio_median']:.2f} |")
        L.append("")
    lp = res["lp_relaxation"]
    L += ["## LP relaxation", "", "| instances | n | LP below MILP objective: mean / median / p90 / max % | LP below the reference dual bound: mean / median % | "
          "integral values (1e-6) % | rounded LP agrees with the MILP schedule (aligned) % | on integral values % | wrong integral values / instance |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in lp.items():
        L.append(f"| {k} | {v['n']} | {v['gap_to_opt_mean']:.3f} / {v['gap_to_opt_median']:.3f} / {v['gap_to_opt_p90']:.3f} / {v['gap_to_opt_max']:.3f} | "
                 + (f"{v['gap_to_db_mean']:.3f} / {v['gap_to_db_median']:.3f}" if "gap_to_db_mean" in v else "–")
                 + f" | {v['integral_share']:.1f} | {v['rounded_agrees']:.2f} | {v['integral_agrees']:.2f} | {v['wrong_integral_per_instance']:.1f} |")
    L += ["", "## Paper metrics (gap to the reference dual bound; feasible instances; speed-up against the re-timed full MILP)", "",
          "| method | feasible % | gap mean % [95 % CI] | median | max | speed-up mean [95 % CI] | median | ratio of means | fixed % | served % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if "gap_mean" not in r:
            L.append(f"| {r['method']} | {r['feasible']:.1f} | – | – | – | – | – | – | – | – |")
            continue
        L.append(f"| {r['method']} | {r['feasible']:.1f} | {fmt(r['gap_mean'])} {ci(r['gap_ci'])} | {fmt(r['gap_median'])} | {fmt(r['gap_max'])} | "
                 f"{fmt(r['speedup_mean'], 1)} {ci(r['speedup_ci'], 1)} | {fmt(r['speedup_median'], 1)} | {fmt(r['speedup_ratio_of_means'], 1)} | "
                 f"{fmt(r.get('fixed_mean'), 1)} | {fmt(r.get('served'), 1)} |")
    L += ["", "## Paired comparisons (method − reference; instances feasible for both; bootstrap 95 % CI)", "",
          "| method | reference | n | Δ gap, pp [CI] | Δ log speed-up [CI] (> 0: method faster) |", "|---|---|---|---|---|"]
    for p in res["paired"]:
        L.append(f"| {p['method']} | {p['ref']} | {p['n_both']} | {fmt(p['d_gap']['diff'], 2, True)} {ci(p['d_gap']['ci'], 2, True)} | "
                 f"{fmt(p['d_log_speedup']['diff'], 2, True)} {ci(p['d_log_speedup']['ci'], 2, True)} |")
    L += ["", "## Same budget: the full MILP stopped at each method's per-instance time (its incumbent log)", "",
          "| method | full MILP has a solution % | method gap % | full MILP gap at the same time % (where it has one) | Δ method − full, pp [CI] (n) | method better / full better / full has none | time-to-quality speed-up (median) |",
          "|---|---|---|---|---|---|---|"]
    for nm, v in res["same_budget"].items():
        p = v["paired_method_minus_full"]
        L.append(f"| {nm} | {v['full_has_solution']:.0f} | {fmt(v['method_gap_mean'])} | {fmt(v['full_gap_mean_where_solution'])} | "
                 f"{fmt(p['diff'], 2, True)} {ci(p['ci'], 2, True)} ({p['n']}) | {v['method_better']} / {v['full_better']} / {v['full_none_method_ok']} | "
                 f"{fmt(v.get('ttq_speedup_median'))} |")
    L += ["", "Loose-gap runs vs the 0.1 % run's own trace (median time at which the trace first shows the gap): "
          + ", ".join(f"{k}: {v:.1f} s" for k, v in res["loose_gap_from_trace"].items()), ""]
    s = res["val_select"]
    L += ["## Validation", "", f"LP-integral tolerance ({s['n_val']} validation instances; rule: {s['rule']}):", "",
          "| guards | tol | infeasible | gap to C* mean / max % | fixed % | reduced MILP + guards, mean s |", "|---|---|---|---|---|---|"]
    for gd in ("none", "guards"):
        for r in s[gd]["rows"]:
            mark = " **(selected)**" if r["tol"] == s[gd]["selected_tol"] else ""
            L.append(f"| {gd} | {r['tol']:g}{mark} | {r['n_infeasible']} | {r['gap_mean']:.3f} / {r['gap_max']:.2f} | {r['fixed']:.1f} | {r['time_mean']:.1f} |")
    L.append(f"\nLP rounding + repair + LPs on validation: gap to C* mean {s['lpround']['gap_mean']:.2f} %, median "
             f"{s['lpround']['gap_median']:.2f} %, served {s['lpround']['served']:.1f} %.\n")
    t = res["ltf_lp_tuning"]
    L.append(f"Learning to Fix on the LP-relaxation values (eps = 1 %, {t['n_val']} validation instances): converged {t['converged']}, "
             f"{t['n_iter']} iterations, {t['n_cut_sets']} cuts, validation fixed {t['val_fixed_share'] * 100:.1f} % (OFF "
             f"{t['val_fixed_off'] * 100:.1f} / ON {t['val_fixed_on'] * 100:.1f}), {t['lo_zero']} generators never fixed OFF, "
             f"{t['hi_one']} never fixed ON, {t['collapsed']} collapsed intervals; witness max {t['verify_max'] * 100:.3f} %; "
             f"{t['wall_s'] / 60:.0f} min wall.")
    open(os.path.join(out_dir, "base_results.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=["uc12", "uc24"])
    ap.add_argument("--eval", default="")
    ap.add_argument("--select", default="")
    ap.add_argument("--tune", default="")
    ap.add_argument("--out_dir", default="")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    main(a.bench, a.eval, a.select, a.tune, a.out_dir)
