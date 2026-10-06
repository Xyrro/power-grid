"""Robustness study, step 3: tables from results/uc12/ood_eval.jsonl (and the instance files of
data/generated/uc12_ood/<shift>/).

    python3 scripts/uc_ood_report.py

Paper metrics of Learning to Fix (gap = (C - DB) / C to the back-to-back full MILP's dual bound; speed-up = per-instance
T_MILP / T_method with every overhead; statistics over feasible instances), plus the median speed-up, the served
share (no shedding, over-generation or reserve shortfall), and our convention (gap to the full MILP's objective over
all instances; an infeasible reduced MILP falls back to the full MILP and pays both times).
Bootstrap (2,000 resamples of instances): mean-gap CIs; paired differences (rule - LtF kNN) on the instances both
solve; degradation (shift - in-distribution) with independent resampling of the two instance sets; and the
difference in degradation (rule - LtF kNN, shift - in-distribution), paired within each set.
Output: results/uc12/ood_results.json, results/uc12/ood_results.md.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.fixpolicy import align_to_prediction  # noqa: E402
from otsl.ltfx import table2_features  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402

ROOT, OUTD, OUT = "data/generated/uc12", "data/generated/uc12_ood", "results/uc12"
ORDER = ("id", "load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
SHIFT_LABEL = {"id": "in-distribution (test_fresh 0-39)", "load_hi": "load +15 %", "load_lo": "load -15 %",
               "ren_hi": "wind + solar x 1.5", "gen_out": "2-3 thermal units out", "line_out": "1-2 lines out",
               "night": "windows across midnight"}
RULES = ["full MILP", "LtF kNN eps=1%", "LtF BCE eps=1%", "hybrid eps=1%", "guarded error-cost 90%",
         "guarded error-cost 95%", "combined 98%", "no learning: LP-integral fixings + guards",
         "hybrid eps=1% (GNN sees topology)", "guarded error-cost 90% (GNN sees topology)",
         "e2e RL 1 LP", "e2e RL screen", "e2e lag 1 LP", "e2e lag screen"]
LABEL = {"full MILP": "full MILP", "LtF kNN eps=1%": "LtF, kNN, eps = 1 % (paper's setting)",
         "LtF BCE eps=1%": "LtF on our BCE GNN, eps = 1 %", "hybrid eps=1%": "hybrid (guard-aware LtF on error cost)",
         "guarded error-cost 90%": "guarded error-cost rule, 90 %", "guarded error-cost 95%": "guarded error-cost rule, 95 % (+ LP guard)",
         "combined 98%": "combined pipeline, 98 %", "no learning: LP-integral fixings + guards": "no learning: fix LP-integral unit-hours + guards",
         "hybrid eps=1% (GNN sees topology)": "hybrid, GNN sees topology", "guarded error-cost 90% (GNN sees topology)": "guarded 90 %, GNN sees topology",
         "e2e RL 1 LP": "end-to-end REINFORCE, 1 LP (no MILP)", "e2e RL screen": "end-to-end REINFORCE, 7-threshold screening",
         "e2e lag 1 LP": "end-to-end combined (Lagrangian), 1 LP", "e2e lag screen": "end-to-end combined, screening"}
REF = "LtF kNN eps=1%"
OURS = ["hybrid eps=1%", "guarded error-cost 90%", "guarded error-cost 95%", "LtF BCE eps=1%", "combined 98%",
        "no learning: LP-integral fixings + guards", "e2e lag screen"]
NB = 2000


def boot_mean(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    b = x[np.random.default_rng(seed).integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def per_rule(recs_by_k, full_by_k, ks):
    """per-instance arrays of one rule"""
    R = [recs_by_k[k] for k in ks]
    F = [full_by_k[k] for k in ks]
    feas = np.array([bool(r["feasible"]) for r in R])
    cost = np.array([r["obj"] for r in R], float)
    DB = np.array([f["bound"] if np.isfinite(f["bound"]) else f["obj"] * (1 - (f["mip_gap"] or 0)) for f in F])
    C_full = np.array([f["obj"] for f in F])
    t_full = np.array([f["time"] for f in F])
    t_m = np.array([r["time"] + r["pre_s"] + r["relax_s"] + r["guard_s"] for r in R])
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(feas, (cost - DB) / cost * 100, np.nan)
        sp = np.where(feas, t_full / t_m, np.nan)
    c_fb = np.where(feas, cost, C_full)
    t_fb = np.where(feas, t_m, t_m + t_full)
    slack = np.array([(r.get("shed", 0) or 0) + (r.get("spill", 0) or 0) + (r.get("short", 0) or 0) for r in R])
    slack_f = np.array([(f.get("shed", 0) or 0) + (f.get("spill", 0) or 0) + (f.get("short", 0) or 0) for f in F])
    served = np.where(feas, slack < 1e-6, slack_f < 1e-6)
    return dict(feas=feas, gap=gap, sp=sp, t_m=t_m, t_full=t_full, g_ref=(c_fb - C_full) / C_full * 100, t_fb=t_fb,
                served=served, fixed=np.array([r.get("fixed_share", 0) for r in R]) * 100,
                fixed_pre=np.array([r.get("fixed_pre_guard", r.get("fixed_share", 0)) for r in R]) * 100,
                n_lp=np.array([r.get("n_lp", 0) for r in R]), ks=np.array(ks))


def stats(x):
    f = x["feas"]
    g, sp = x["gap"][f], x["sp"][f]
    if not f.any():
        return dict(n=int(len(f)), feasible=0.0)
    return dict(n=int(len(f)), feasible=float(f.mean() * 100), n_infeasible=int((~f).sum()),
                gap_mean=float(g.mean()), gap_mean_ci=boot_mean(g), gap_median=float(np.median(g)), gap_max=float(g.max()),
                speedup_mean=float(sp.mean()), speedup_mean_ci=boot_mean(sp), speedup_median=float(np.median(sp)),
                runtime_mean=float(x["t_m"][f].mean()), fixed_mean=float(x["fixed"][f].mean()),
                fixed_pre_mean=float(x["fixed_pre"][f].mean()), served=float(x["served"].mean() * 100),
                gap_ref_mean=float(x["g_ref"].mean()), gap_ref_ci=boot_mean(x["g_ref"]),
                gap_ref_median=float(np.median(x["g_ref"])), n_gt1=int((x["g_ref"] > 1).sum()),
                n_gt10=int((x["g_ref"] > 10).sum()), speedup_ratio_of_means=float(x["t_full"].mean() / x["t_fb"].mean()),
                lps_mean=float(x["n_lp"].mean()))


def paired(a, b, seed=0):
    """rule a - rule b over the instances both solve: gap (pp) and log speed-up"""
    both = a["feas"] & b["feas"]
    dg = (a["gap"] - b["gap"])[both]
    dl = (np.log(a["sp"]) - np.log(b["sp"]))[both]
    return dict(n=int(both.sum()), d_gap=float(dg.mean()) if len(dg) else np.nan, d_gap_ci=boot_mean(dg, seed),
                d_logsp=float(dl.mean()) if len(dl) else np.nan, d_logsp_ci=boot_mean(dl, seed),
                feas_a=float(a["feas"].mean() * 100), feas_b=float(b["feas"].mean() * 100))


def degradation(x_sh, x_id, seed=0):
    """shift - in-distribution, independent instance resampling: mean gap (feasible), feasibility, mean log speed-up"""
    rng = np.random.default_rng(seed)
    gs, gi = x_sh["gap"][x_sh["feas"]], x_id["gap"][x_id["feas"]]
    ls, li = np.log(x_sh["sp"][x_sh["feas"]]), np.log(x_id["sp"][x_id["feas"]])
    out = {}
    for key, a, b in (("d_gap", gs, gi), ("d_logsp", ls, li), ("d_feas", x_sh["feas"] * 100.0, x_id["feas"] * 100.0),
                      ("d_gap_ref", x_sh["g_ref"], x_id["g_ref"])):
        if len(a) == 0 or len(b) == 0:
            out[key], out[key + "_ci"] = np.nan, [np.nan, np.nan]
            continue
        ba = a[rng.integers(0, len(a), (NB, len(a)))].mean(1)
        bb = b[rng.integers(0, len(b), (NB, len(b)))].mean(1)
        out[key] = float(a.mean() - b.mean())
        out[key + "_ci"] = [float(np.percentile(ba - bb, 2.5)), float(np.percentile(ba - bb, 97.5))]
    return out


def did(a_sh, b_sh, a_id, b_id, seed=0):
    """difference in degradation of the mean gap (pp): [mean(a - b) on shift] - [mean(a - b) in distribution], each on
    the instances both rules solve; resampled within each set"""
    rng = np.random.default_rng(seed)
    ds = (a_sh["gap"] - b_sh["gap"])[a_sh["feas"] & b_sh["feas"]]
    di = (a_id["gap"] - b_id["gap"])[a_id["feas"] & b_id["feas"]]
    if len(ds) == 0 or len(di) == 0:
        return dict(did=np.nan, ci=[np.nan, np.nan])
    bs = ds[rng.integers(0, len(ds), (NB, len(ds)))].mean(1)
    bi = di[rng.integers(0, len(di), (NB, len(di)))].mean(1)
    return dict(did=float(ds.mean() - di.mean()), ci=[float(np.percentile(bs - bi, 2.5)), float(np.percentile(bs - bi, 97.5))])


def ood_diagnostics(sysm, tr, shift, ks, Xtr, mu, sd):
    """how far each shift is from the training data, and how much the predictions degrade"""
    L_tr = tr["load"].sum(2)
    N_tr = L_tr - tr["avail"].sum(2)
    inst = [dict(np.load(os.path.join(OUTD, shift, f"{k:03d}.npz"))) for k in ks]
    d = {key: np.stack([x[key] for x in inst]) for key in ("load", "avail", "u0", "sr")}
    d["start"] = np.array([int(x["start"]) for x in inst])
    L = d["load"].sum(2)
    N = L - d["avail"].sum(2)
    Z = (table2_features(sysm, d) - mu) / sd
    D2 = (Z ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None] - 2 * Z @ Xtr.T
    nn = np.sqrt(np.maximum(D2.min(1), 0))
    out = dict(n=len(ks), load_above_train_max=float((L > L_tr.max()).mean() * 100),
               load_below_train_min=float((L < L_tr.min()).mean() * 100),
               net_outside_train=float(((N > N_tr.max()) | (N < N_tr.min())).mean() * 100),
               load_MW=[float(L.min() * 100), float(L.max() * 100)], net_MW=[float(N.min() * 100), float(N.max() * 100)],
               knn_nn_dist_median=float(np.median(nn)), knn_nn_dist=nn.tolist())
    # prediction error vs the full MILP's schedule (aligned inside identical-unit groups), unit-hours per instance;
    # unavailable units are excluded (their predictions are overridden)
    for key in ("knn", "bce", "st"):
        w, wo, wn = [], [], []
        for x in inst:
            p = x[f"p_{key}"]
            y = (p > 0.5).astype(np.int8)
            ua = align_to_prediction(sysm, x["u_full"], y, x["u0"])
            keep = np.setdiff1d(np.arange(sysm.G), x["g_out"])
            diff = (y != ua)[:, keep]
            w.append(int(diff.sum()))
            wo.append(int((diff & (y[:, keep] == 0)).sum()))          # predicted OFF, MILP ON (under-commitment)
            wn.append(int((diff & (y[:, keep] == 1)).sum()))
        out[f"wrong_{key}"] = float(np.mean(w))
        out[f"wrong_off_{key}"] = float(np.mean(wo))
        out[f"wrong_on_{key}"] = float(np.mean(wn))
    out["milp_units_on"] = float(np.mean([x["u_full"].sum() / 12 for x in inst]))
    out["milp_time_mean"] = float(np.mean([float(x["time_full"]) for x in inst]))
    out["relax_gap_pct"] = float(np.mean([(float(x["obj_full"]) - float(x["c_rel"])) / float(x["obj_full"]) * 100 for x in inst]))
    out["u_rel_integral"] = float(np.mean([((np.abs(x["u_rel"]) < 1e-6) | (np.abs(x["u_rel"] - 1) < 1e-6)).mean() for x in inst]) * 100)
    return out


def fmt(x, p=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{p}f}"


def ci(v, c, p=2):
    return f"{fmt(v, p)} [{fmt(c[0], p)}, {fmt(c[1], p)}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(OUT, "ood_eval.jsonl"))
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    recs = [json.loads(line) for line in open(a.path)]
    by = {}
    for r in recs:
        by.setdefault(r["shift"], {}).setdefault(r["rule"], {})[r["k"]] = r
    sysm = load_rts_gmlc()
    tr = load(os.path.join(ROOT, "train.npz"))
    Ztr = table2_features(sysm, tr)
    mu, sd = Ztr.mean(0), Ztr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr = (Ztr - mu) / sd
    specs = json.load(open(os.path.join(OUTD, "specs.json")))
    cov = json.load(open(os.path.join(OUTD, "coverage_train.json")))
    res = {"n_per_shift": {}, "stats": {}, "paired_vs_ltf_knn": {}, "degradation": {}, "did_vs_ltf_knn": {}, "ood": {},
           "specs": {k: {kk: v for kk, v in s.items() if kk != "jobs"} for k, s in specs["shifts"].items()},
           "line_pool": specs["line_pool"], "coverage_train": cov["train"]}
    X = {}
    shifts = [s for s in ORDER if s in by]
    for sh in shifts:
        rr = by[sh]
        full = rr["full MILP"]
        ks = sorted(k for k in full if all(k in rr[ru] for ru in rr if not ru.endswith("(GNN sees topology)")))
        res["n_per_shift"][sh] = len(ks)
        X[sh] = {ru: per_rule(rr[ru], full, ks) for ru in RULES if ru in rr and all(k in rr[ru] for k in ks)}
        res["stats"][sh] = {ru: stats(x) for ru, x in X[sh].items()}
        res["paired_vs_ltf_knn"][sh] = {ru: paired(X[sh][ru], X[sh][REF]) for ru in OURS if ru in X[sh]}
        if sh == "line_out":
            for ru in ("hybrid eps=1% (GNN sees topology)", "guarded error-cost 90% (GNN sees topology)"):
                if ru in X[sh]:
                    res["paired_vs_ltf_knn"][sh][ru] = paired(X[sh][ru], X[sh][REF])
                    base = ru.replace(" (GNN sees topology)", "")
                    res["paired_vs_ltf_knn"][sh][ru + " vs static graph"] = paired(X[sh][ru], X[sh][base])
        res["ood"][sh] = ood_diagnostics(sysm, tr, sh, ks, Xtr, mu, sd)
        extra = [rr["full MILP"][k].get("c_rel_intact") for k in ks]
        if extra and extra[0] is not None:
            inc = [(float(x["c_rel"]) - ci_) / ci_ * 100 for x, ci_ in
                   zip([dict(np.load(os.path.join(OUTD, sh, f"{k:03d}.npz"))) for k in ks], extra)]
            res["ood"][sh]["outage_relax_cost_increase_pct"] = dict(mean=float(np.mean(inc)), median=float(np.median(inc)),
                                                                   share_gt_0_1=float((np.array(inc) > 0.1).mean() * 100))
    if "id" in X:
        for sh in shifts:
            if sh == "id":
                continue
            res["degradation"][sh] = {ru: degradation(X[sh][ru], X["id"][ru]) for ru in X[sh] if ru in X["id"]}
            res["did_vs_ltf_knn"][sh] = {ru: did(X[sh][ru], X[sh][REF], X["id"][ru], X["id"][REF])
                                         for ru in OURS if ru in X[sh] and ru in X["id"]}
    # ---------------- recovery test (scripts/uc_ood_finetune.py): same shifted instances, models given 50 labelled
    # shifted instances; "before" = the main run's records of the same rules and instances
    ftp = os.path.join(OUT, "ood_ft_eval.jsonl")
    if os.path.exists(ftp):
        fby = {}
        for line in open(ftp):
            r = json.loads(line)
            fby.setdefault(r["shift"], {}).setdefault(r["rule"], {})[r["k"]] = r
        res["recovery"] = {}
        for sh, rr in fby.items():
            ks = sorted(k for k in rr["full MILP"] if all(k in v for v in rr.values()) and k in by[sh]["full MILP"])
            out = {"n": len(ks)}
            info = os.path.join(OUT, f"ood_ft_bce_{sh}.json")
            if os.path.exists(info):
                out["bce_finetune"] = json.load(open(info))
            for ru in rr:
                if ru == "full MILP":
                    continue
                after = per_rule(rr[ru], rr["full MILP"], ks)
                before = per_rule(by[sh][ru], by[sh]["full MILP"], ks)
                out[ru] = {"before": stats(before), "after": stats(after), "after_minus_before": paired(after, before)}
            res["recovery"][sh] = out
    json.dump(res, open(os.path.join(OUT, "ood_results.json"), "w"), indent=1, default=float)

    # ---------------------------------------------------------------- markdown
    L = ["# Robustness under distribution shift (12-hour UC): results", "",
         "Generated by `scripts/uc_ood_report.py` from `results/uc12/ood_eval.jsonl`; method and verdict: "
         "[`docs/methods/ood.md`](../../docs/methods/ood.md). Instances per shift: "
         + ", ".join(f"{sh} {res['n_per_shift'][sh]}" for sh in shifts) + ".", "",
         "Paper metrics: gap = (C − DB) / C to the back-to-back full MILP's dual bound, feasible instances only; speed-up = "
         "per-instance T_full MILP / T_method, all overheads included (LP relaxation for GNN inputs, inference, guards, LPs). "
         "*Served* = no shedding, over-generation or reserve shortfall (all instances; an infeasible reduced MILP counts with "
         "the full MILP's schedule). End-to-end rows solve no MILP; they are always feasible (soft constraints).", ""]
    KEY = ["LtF kNN eps=1%", "LtF BCE eps=1%", "hybrid eps=1%", "guarded error-cost 90%", "guarded error-cost 95%",
           "combined 98%", "no learning: LP-integral fixings + guards", "e2e RL screen", "e2e lag screen"]
    L += ["## Overview: mean gap % [95 % CI], speed-up mean / median, feasible % (n per set as above)", "",
          "| rule | " + " | ".join(SHIFT_LABEL[sh] for sh in shifts) + " |", "|---|" + "---|" * len(shifts)]
    for ru in KEY:
        cells = []
        for sh in shifts:
            st_ = res["stats"][sh].get(ru)
            if not st_ or st_.get("feasible", 0) == 0:
                cells.append("–")
                continue
            cells.append(f"{ci(st_['gap_mean'], st_['gap_mean_ci'])}, {fmt(st_['speedup_mean'], 1)}× / "
                         f"{fmt(st_['speedup_median'], 1)}×, {fmt(st_['feasible'], 0)} %")
        L.append(f"| {LABEL[ru]} | " + " | ".join(cells) + " |")
    L += ["", "Paired against LtF-kNN (Δ mean gap pp [95 % CI], rule − LtF-kNN, instances both solve):", "",
          "| rule | " + " | ".join(SHIFT_LABEL[sh] for sh in shifts) + " |", "|---|" + "---|" * len(shifts)]
    for ru in OURS:
        cells = []
        for sh in shifts:
            pv = res["paired_vs_ltf_knn"][sh].get(ru)
            cells.append(ci(pv["d_gap"], pv["d_gap_ci"]) if pv else "–")
        L.append(f"| {LABEL[ru]} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["## What each shift does to the inputs", "",
          "| shift | load MW (min–max) | hours above train max / below train min load | net load outside train range | "
          "kNN distance to nearest training instance (median) | wrong unit-hours / instance: kNN, BCE GNN, self-trained (OFF-wrong) "
          "| MILP units on | MILP time s | relaxation gap % | LP-integral unit-hours % | outage raises relaxation cost |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sh in shifts:
        o = res["ood"][sh]
        oc = o.get("outage_relax_cost_increase_pct")
        L.append(f"| {SHIFT_LABEL[sh]} | {o['load_MW'][0]:.0f}–{o['load_MW'][1]:.0f} | {o['load_above_train_max']:.1f} % / "
                 f"{o['load_below_train_min']:.1f} % | {o['net_outside_train']:.1f} % | {o['knn_nn_dist_median']:.2f} | "
                 f"{o['wrong_knn']:.1f}, {o['wrong_bce']:.1f}, {o['wrong_st']:.1f} ({o['wrong_off_knn']:.1f}, "
                 f"{o['wrong_off_bce']:.1f}, {o['wrong_off_st']:.1f}) | {o['milp_units_on']:.1f} | {o['milp_time_mean']:.1f} | "
                 f"{o['relax_gap_pct']:.2f} | {o['u_rel_integral']:.1f} | "
                 + (f"{oc['mean']:.2f} % mean, {oc['share_gt_0_1']:.0f} % of instances > 0.1 %" if oc else "–") + " |")
    L += ["", "Training data: " + ", ".join(f"{k} {v}" for k, v in cov["train"].items() if k in
                                           ("n", "n_days", "months", "window_start", "load_MW", "net_load_MW", "u0_units_on")), ""]
    for sh in shifts:
        L += [f"## {SHIFT_LABEL[sh]} (n = {res['n_per_shift'][sh]})", "",
              "| rule | feasible % | gap mean % [95 % CI] | median | max | speed-up mean [95 % CI] | median | fixed % | served % | "
              "gap to MILP obj. % (our conv.) | # > 1 % |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for ru in RULES:
            if ru not in res["stats"][sh]:
                continue
            s = res["stats"][sh][ru]
            if s.get("feasible", 0) == 0:
                L.append(f"| {LABEL[ru]} | 0 | – | – | – | – | – | – | – | – | – |")
                continue
            fx = "–" if ru.startswith("e2e") or ru == "full MILP" else fmt(s["fixed_mean"], 1)
            L.append(f"| {LABEL[ru]} | {fmt(s['feasible'], 1)} | {ci(s['gap_mean'], s['gap_mean_ci'])} | {fmt(s['gap_median'])} | "
                     f"{fmt(s['gap_max'])} | {ci(s['speedup_mean'], s['speedup_mean_ci'], 1)} | {fmt(s['speedup_median'], 1)} | "
                     f"{fx} | {fmt(s['served'], 1)} | {fmt(s['gap_ref_mean'])} | {s['n_gt1']} |")
        L.append("")
        pv = res["paired_vs_ltf_knn"][sh]
        if pv:
            L += ["Paired against LtF-kNN (rule − LtF-kNN on instances both solve; bootstrap 95 % CI):", "",
                  "| rule | n | Δ gap pp | Δ log speed-up |", "|---|---|---|---|"]
            for ru, p in pv.items():
                nm = LABEL.get(ru, ru.replace(" vs static graph", "") + " − same rule on the static graph")
                if ru.endswith(" vs static graph"):
                    nm = LABEL[ru.replace(" vs static graph", "")] + " − same on the static graph"
                L.append(f"| {nm} | {p['n']} | {ci(p['d_gap'], p['d_gap_ci'])} | {ci(p['d_logsp'], p['d_logsp_ci'])} |")
            L.append("")
    if res["degradation"]:
        L += ["## Degradation against in-distribution", "",
              "Δ = shift − in-distribution (independent bootstrap of the two instance sets). Mean gap over feasible instances "
              "(pp), feasibility (pp), mean log speed-up (negative = slower relative to the full MILP).", "",
              "| shift | rule | Δ mean gap pp [CI] | Δ feasible pp | Δ log speed-up [CI] | Δ gap, our conv. pp [CI] |",
              "|---|---|---|---|---|---|"]
        for sh, dd in res["degradation"].items():
            for ru in RULES:
                if ru not in dd or ru == "full MILP":
                    continue
                g = dd[ru]
                L.append(f"| {SHIFT_LABEL[sh]} | {LABEL[ru]} | {ci(g['d_gap'], g['d_gap_ci'])} | {fmt(g['d_feas'], 1)} | "
                         f"{ci(g['d_logsp'], g['d_logsp_ci'])} | {ci(g['d_gap_ref'], g['d_gap_ref_ci'])} |")
        L += ["", "## Does a rule degrade less than LtF-kNN? (difference in degradation of the mean gap, pp)", "",
              "[mean(rule − LtF-kNN) on the shift] − [the same in distribution]; negative = the rule loses less than LtF-kNN.", "",
              "| shift | " + " | ".join(LABEL[r] for r in OURS) + " |", "|---|" + "---|" * len(OURS)]
        for sh, dd in res["did_vs_ltf_knn"].items():
            L.append(f"| {SHIFT_LABEL[sh]} | " + " | ".join(ci(dd[r]["did"], dd[r]["ci"]) if r in dd else "–" for r in OURS) + " |")
        L.append("")
    for sh, out in res.get("recovery", {}).items():
        L += [f"## Recovery test: {SHIFT_LABEL[sh]} after 50 labelled shifted instances (n = {out['n']})", "",
              "kNN: the 50 instances added to its pool; BCE GNN: fine-tuned (`scripts/uc_ood_finetune.py`); thresholds, error-cost "
              "model and guards unchanged. *Before* = the main run on the same instances; *after* = a new back-to-back run "
              "(its own full MILP). Δ = after − before on instances both solve.", "",
              "| rule | before: gap mean % [CI], max, served %, feasible % | after: gap mean % [CI], max, served %, feasible %, "
              "speed-up mean / median | Δ gap pp [CI] | instances > 10 % (our conv.) before → after |",
              "|---|---|---|---|---|"]
        for ru, v in out.items():
            if ru in ("n", "bce_finetune"):
                continue
            b, a_, d = v["before"], v["after"], v["after_minus_before"]
            L.append(f"| {LABEL[ru]} | {ci(b['gap_mean'], b['gap_mean_ci'])}, {fmt(b['gap_max'], 1)}, {fmt(b['served'], 1)}, "
                     f"{fmt(b['feasible'], 1)} | {ci(a_['gap_mean'], a_['gap_mean_ci'])}, {fmt(a_['gap_max'], 1)}, "
                     f"{fmt(a_['served'], 1)}, {fmt(a_['feasible'], 1)}, {fmt(a_['speedup_mean'], 1)} / "
                     f"{fmt(a_['speedup_median'], 1)} | {ci(d['d_gap'], d['d_gap_ci'])} | {b['n_gt10']} → {a_['n_gt10']} |")
        if "bce_finetune" in out:
            ft = out["bce_finetune"]
            L += ["", f"BCE fine-tuning: validation log-loss on 10 held-out shifted instances {ft['val_logloss_before']:.4f} → "
                      f"{ft['best']:.4f} ({ft['n_shifted_train']} shifted + {ft['n_orig']} original training instances)."]
        L.append("")
    open(os.path.join(OUT, "ood_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
