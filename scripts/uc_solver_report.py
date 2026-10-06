"""Scoring of the solver-side variants (scripts/uc_solver_eval.py records) under the paper's metrics.

    python3 scripts/uc_solver_report.py --bench uc12 --runs results/uc12/solver_test_fresh.jsonl --tag test_fresh

Per instance and variant (instances with a full-MILP record only):
  method time  = pre-solve work (inference, LP relaxation for the GNN features, rule, guards, decoding, start LP; for
                 polish variants also the whole base pipeline) + model build + solver run
  gap          = (C - DB) / C, DB = the dual bound of the full MILP solved cold in the same process (paper metric)
  speed-up     = T_full / T_method per instance; mean and median over feasible instances
  time limit tau (anytime variants: pas, ftp, lb, grad, full): the run is cut at tau seconds of solver time; the
                 incumbent at tau is the last improving solution logged before tau (the MIP start, or the base
                 solution for polish variants, counts from time 0). HiGHS is deterministic, so this equals a run with
                 time limit tau up to timing noise.
  TTQ q        = time until the method holds a schedule within q of the full MILP's final cost (full MILP: its own
                 incumbent trace).
Paired bootstrap (10,000 resamples of instances) for gap differences against chosen comparators, on instances feasible
for both.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANYTIME = ("pas", "ftp", "lb", "grad", "full", "warmfull", "warmred", "core", "hard", "ref")


def load_runs(paths):
    recs = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                recs.setdefault(r["i"], {}).update(r)
    return [recs[i] for i in sorted(recs)]


def base_obj(r):
    """objective available at time 0 of a run: the MIP start / polish base (None if none)"""
    if r.get("base_obj") is not None:
        return r["base_obj"]
    if r.get("start_obj") is not None and r.get("start_accepted"):
        return r["start_obj"]
    return None


def at_tau(r, tau=None):
    """(feasible, objective, solver seconds) of a run cut at tau seconds (None: as run)"""
    if not r or r.get("t_run") is None:
        return False, np.inf, np.nan
    if tau is None or r["t_run"] <= tau:
        ok = r.get("feasible", False)
        obj = r["obj"] if ok else np.inf
        b = base_obj(r)
        if b is not None:
            obj = min(obj, b)
            ok = True
        return ok, obj, r["t_run"]
    best = base_obj(r)
    best = np.inf if best is None else best
    for t, o, _ in r["inc"]:
        if t <= tau:
            best = min(best, o)
    return np.isfinite(best), best, tau


def method_time(r, t_solver):
    return r.get("pre_s", 0.0) + r.get("t_build", 0.0) + t_solver


def ttq(r, target, full=False):
    """time (s, method clock) at which the run first holds a schedule with cost <= target; nan if never"""
    pre = r.get("pre_s", 0.0) + r.get("t_build", 0.0)
    b = base_obj(r)
    if b is not None and b <= target * (1 + 1e-9):
        return pre
    for t, o, _ in r.get("inc", []):
        if o <= target * (1 + 1e-9):
            return pre + t
    return np.nan


def per_instance(recs, v, tau=None):
    """arrays over instances: feasible, gap %, speed-up, method time, full time, fixed share, ttq1, ttq05"""
    out = {k: [] for k in ("i", "feas", "gap", "sp", "t", "tfull", "fixed", "ttq1", "ttq05", "ttq1_full",
                           "ttq05_full", "accepted")}
    for R in recs:
        if "full" not in R or v not in R:
            continue
        F = R["full"]
        DB = F["bound"]
        T_full = method_time(F, F["t_run"])
        Cf = F["obj"]
        r = R[v]
        ok, obj, ts = at_tau(r, tau)
        t = method_time(r, ts) if ok or r.get("t_run") is not None else np.nan
        out["i"].append(R["i"])
        out["feas"].append(ok)
        out["gap"].append((obj - DB) / obj * 100 if ok else np.nan)
        out["sp"].append(T_full / t if ok else np.nan)
        out["t"].append(t)
        out["tfull"].append(T_full)
        G = R.get("_G", 73)
        T = R.get("_T", None)
        nf = r.get("n_fixed")
        out["fixed"].append(nf / (G * T) if (nf is not None and T) else np.nan)
        for q, k in ((0.01, "ttq1"), (0.005, "ttq05")):
            out[k].append(ttq(r, Cf * (1 + q)))
            out[k + "_full"].append(ttq(F, Cf * (1 + q)))
        out["accepted"].append(r.get("start_accepted"))
    return {k: np.array(v_, dtype=float if k != "accepted" else object) for k, v_ in out.items()}


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(recs, v, tau=None, T=12):
    for R in recs:
        R["_T"] = T
    a = per_instance(recs, v, tau)
    f = a["feas"] > 0
    n = len(f)
    if n == 0:
        return None
    g, ci_lo, ci_hi = boot_ci(a["gap"][f])
    ttq_ok1 = np.isfinite(a["ttq1"]) & np.isfinite(a["ttq1_full"])
    ttq_ok05 = np.isfinite(a["ttq05"]) & np.isfinite(a["ttq05_full"])
    acc = [x for x in a["accepted"] if x is not None]
    return {"variant": v, "tau": tau, "n": n, "feasible_%": float(f.mean() * 100),
            "gap_mean_%": g, "gap_ci": [ci_lo, ci_hi],
            "gap_median_%": float(np.median(a["gap"][f])) if f.any() else np.nan,
            "gap_max_%": float(np.max(a["gap"][f])) if f.any() else np.nan,
            "speedup_mean": float(np.mean(a["sp"][f])) if f.any() else np.nan,
            "speedup_median": float(np.median(a["sp"][f])) if f.any() else np.nan,
            "ratio_of_mean_times": float(a["tfull"][f].mean() / a["t"][f].mean()) if f.any() else np.nan,
            "time_mean_s": float(np.mean(a["t"][f])) if f.any() else np.nan,
            "fixed_%": float(np.nanmean(a["fixed"][f]) * 100) if f.any() and np.isfinite(a["fixed"][f]).any() else np.nan,
            "ttq1_reached_%": float(np.isfinite(a["ttq1"]).mean() * 100),
            "ttq05_reached_%": float(np.isfinite(a["ttq05"]).mean() * 100),
            "ttq1_speedup_median": float(np.median(a["ttq1_full"][ttq_ok1] / a["ttq1"][ttq_ok1])) if ttq_ok1.any() else np.nan,
            "ttq05_speedup_median": float(np.median(a["ttq05_full"][ttq_ok05] / a["ttq05"][ttq_ok05])) if ttq_ok05.any() else np.nan,
            "ttq1_median_s": float(np.nanmedian(a["ttq1"])) if np.isfinite(a["ttq1"]).any() else np.nan,
            "ttq05_median_s": float(np.nanmedian(a["ttq05"])) if np.isfinite(a["ttq05"]).any() else np.nan,
            "start_accepted_%": float(np.mean(acc) * 100) if acc else None}


def paired(recs, v, w, tau_v=None, tau_w=None, T=12):
    """mean gap difference v - w (pp) on instances feasible for both, with a bootstrap CI; and log speed-up diff"""
    for R in recs:
        R["_T"] = T
    a, b = per_instance(recs, v, tau_v), per_instance(recs, w, tau_w)
    common = sorted(set(a["i"][a["feas"] > 0]) & set(b["i"][b["feas"] > 0]))
    ia = {i: k for k, i in enumerate(a["i"])}
    ib = {i: k for k, i in enumerate(b["i"])}
    dg = [a["gap"][ia[i]] - b["gap"][ib[i]] for i in common]
    dl = [np.log(a["sp"][ia[i]]) - np.log(b["sp"][ib[i]]) for i in common]
    m, lo, hi = boot_ci(dg)
    ml, lol, hil = boot_ci(dl)
    return {"v": v, "w": w, "tau_v": tau_v, "tau_w": tau_w, "n_common": len(common), "gap_diff_pp": m,
            "gap_diff_ci": [lo, hi], "log_speedup_diff": ml, "log_speedup_diff_ci": [lol, hil]}


def fmt_row(s):
    t = "" if s["tau"] is None else f" @ τ = {s['tau']:g} s"
    return (f"| {s['variant']}{t} | {s['feasible_%']:.0f} | {s['gap_mean_%']:.3f} [{s['gap_ci'][0]:.2f}, {s['gap_ci'][1]:.2f}] "
            f"| {s['gap_median_%']:.3f} | {s['gap_max_%']:.2f} | {s['speedup_mean']:.2f} | {s['speedup_median']:.2f} "
            f"| {s['ratio_of_mean_times']:.2f} | {s['time_mean_s']:.1f} | "
            + (f"{s['fixed_%']:.0f}" if np.isfinite(s['fixed_%']) else "–") + " |")


HEADER = ("| variant | feasible % | gap to DB mean % [95 % CI] | median % | max % | speed-up mean | median | "
          "ratio of mean times | time mean s | fixed % |\n|---|---|---|---|---|---|---|---|---|---|")



# ============================================================================ full report
def warm_table(recs, pairs, T):
    """warm start vs cold: time to proof (solver run to termination incl. pre-solve work), TTQ 1 % / 0.5 % of the
    full MILP's final cost; pairs = [(label, warm variant, cold variant)]"""
    rows = []
    for label, w, c in pairs:
        for R in recs:
            R["_T"] = T
        aw, ac = per_instance(recs, w), per_instance(recs, c)
        iw = {i: k for k, i in enumerate(aw["i"])}
        common = [i for i in ac["i"] if i in iw]
        jc = {i: k for k, i in enumerate(ac["i"])}
        tw = np.array([aw["t"][iw[i]] for i in common]); tc = np.array([ac["t"][jc[i]] for i in common])
        out = {"label": label, "warm": w, "cold": c, "n": len(common),
               "proof_time_mean_s": [float(tc.mean()), float(tw.mean())],
               "proof_time_median_s": [float(np.median(tc)), float(np.median(tw))],
               "proof_speedup_median": float(np.median(tc / tw)), "proof_speedup_ratio_of_means": float(tc.mean() / tw.mean()),
               "log_ratio_ci": list(boot_ci(np.log(tc / tw))[1:])}
        for q in ("ttq1", "ttq05"):
            xc = np.array([ac[q][jc[i]] for i in common]); xw = np.array([aw[q][iw[i]] for i in common])
            both = np.isfinite(xc) & np.isfinite(xw)
            out[q] = {"reached_%": [float(np.isfinite(xc).mean() * 100), float(np.isfinite(xw).mean() * 100)],
                      "median_s": [float(np.nanmedian(xc)) if np.isfinite(xc).any() else None,
                                   float(np.nanmedian(xw)) if np.isfinite(xw).any() else None],
                      "speedup_median_both": float(np.median(xc[both] / xw[both])) if both.any() else None}
        acc = [aw["accepted"][iw[i]] for i in common if aw["accepted"][iw[i]] is not None]
        out["start_accepted_%"] = float(np.mean(acc) * 100) if acc else None
        rows.append(out)
    return rows


def pareto_plot(points, curves, path, title):
    """points: [(label, speed-up, gap, family)], curves: [(label, [(sp, gap)], family)]; family in ref/new/full"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"ref": "#2a78d6", "new": "#eb6834", "full": "#1baf7a"}
    mk = {"ref": "s", "new": "o", "full": "^"}
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    for lab, pts, fam in curves:
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col[fam], lw=2, alpha=0.9, zorder=2)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk[fam], color=col[fam], ms=4, zorder=3)
        ax.annotate(lab, (pts[-1][0], pts[-1][1]), textcoords="offset points", xytext=(4, 2), fontsize=7, color="#52514e")
    for lab, sp, g, fam in points:
        ax.plot(sp, g, mk[fam], color=col[fam], ms=8, mec="white", mew=1.5, zorder=4)
        ax.annotate(lab, (sp, g), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#0b0b0b")
    ax.set_xscale("log")
    ax.set_xlabel("speed-up over the full MILP (mean of per-instance ratios, log scale)", color="#52514e")
    ax.set_ylabel("gap to the full MILP's dual bound, mean %", color="#52514e")
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(True, which="both", color="#e8e7e2", lw=0.6)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=mk[f], color=col[f], ls="-" if f != "ref" else "", label=l)
                       for f, l in (("ref", "hard fixing (references)"), ("new", "solver-side variants"),
                                    ("full", "full MILP cut at time τ"))], fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)



def build_report(spec, out_json, out_md, out_png, title):
    """spec (dict, results/<bench>/solver_selection.json): bench, runs, refs {label: variant}, families
    [{name, rows: [{label, variant, tau}]}], warm [[label, warm, cold]], polish [{label, base, variant, taus,
    tau_sel}], comparators [variant], curves [{label, variant, taus, family}], points_extra"""
    T = 12 if spec["bench"] == "uc12" else 24
    recs = load_runs(spec["runs"])
    n_full = sum(1 for R in recs if "full" in R)
    res = {"bench": spec["bench"], "n_instances": n_full, "spec": spec, "rows": [], "paired": [], "warm": [],
           "polish": [], "curves": []}
    md = [f"# {title}", "",
          f"{n_full} instances; gap = (C − DB) / C with DB the dual bound of the full MILP solved cold in the same process; "
          "speed-up = mean (median) of per-instance T_full / T_method over feasible instances; method time includes "
          "inference, the LP relaxation fed to the GNN, guards, decoding / start LPs and, for polish variants, the whole base "
          "pipeline. 95 % CIs: instance bootstrap.", ""]
    full = summary(recs, "full", None, T)
    md += [f"Full MILP (cold, this process): mean {full['time_mean_s']:.1f} s, gap to its own bound "
           f"{full['gap_mean_%']:.3f} % (max {full['gap_max_%']:.2f} %).", ""]
    res["full"] = full

    def add_rows(rows, header):
        md.append(header)
        md.append("")
        md.append(HEADER)
        for r in rows:
            sm = summary(recs, r["variant"], r.get("tau"), T)
            if sm is None:
                continue
            sm["label"] = r["label"]
            res["rows"].append(sm)
            line = fmt_row(sm)
            md.append(line.replace(f"| {sm['variant']}", f"| {r['label']} (`{sm['variant']}`)", 1))
        md.append("")

    add_rows([{"label": k, "variant": v} for k, v in spec["refs"].items()],
             "## Hard-fixing references, re-run in the same process")
    for fam in spec["families"]:
        add_rows(fam["rows"], f"## {fam['name']}")
    # paired comparisons
    md += ["## Paired differences (variant − comparator; gap in pp, log speed-up; instances feasible for both)", "",
           "| variant | comparator | n | gap diff pp [95 % CI] | log speed-up diff [95 % CI] |", "|---|---|---|---|---|"]
    allrows = [dict(label=k, variant=v, tau=None) for k, v in spec["refs"].items()] + \
              [r for fam in spec["families"] for r in fam["rows"]]
    for r in allrows:
        for c in spec["comparators"]:
            if r["variant"] == c:
                continue
            pr = paired(recs, r["variant"], c, r.get("tau"), None, T)
            pr["label"] = r["label"]
            res["paired"].append(pr)
            md.append(f"| {r['label']} | `{c}` | {pr['n_common']} | {pr['gap_diff_pp']:+.3f} [{pr['gap_diff_ci'][0]:+.3f}, "
                      f"{pr['gap_diff_ci'][1]:+.3f}] | {pr['log_speedup_diff']:+.2f} [{pr['log_speedup_diff_ci'][0]:+.2f}, "
                      f"{pr['log_speedup_diff_ci'][1]:+.2f}] |")
    md.append("")
    # warm starts
    if spec.get("warm"):
        W = warm_table(recs, spec["warm"], T)
        res["warm"] = W
        md += ["## Warm start vs cold start (same instances)", "",
               "| comparison | n | start accepted % | time to proof mean s cold / warm | median cold / warm | median ratio "
               "cold/warm [95 % CI of mean log ratio] | TTQ 1 %: reached % cold / warm, median s cold / warm | TTQ 0.5 %: "
               "reached %, median s |", "|---|---|---|---|---|---|---|---|"]
        for w in W:
            ci = np.exp(w["log_ratio_ci"])
            f1, f5 = w["ttq1"], w["ttq05"]
            fm = lambda x: "–" if x is None else f"{x:.1f}"
            md.append(f"| {w['label']} | {w['n']} | {w['start_accepted_%'] if w['start_accepted_%'] is not None else '–'} | "
                      f"{w['proof_time_mean_s'][0]:.1f} / {w['proof_time_mean_s'][1]:.1f} | {w['proof_time_median_s'][0]:.1f} / "
                      f"{w['proof_time_median_s'][1]:.1f} | {w['proof_speedup_median']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
                      f"{f1['reached_%'][0]:.0f} / {f1['reached_%'][1]:.0f}, {fm(f1['median_s'][0])} / {fm(f1['median_s'][1])} | "
                      f"{f5['reached_%'][0]:.0f} / {f5['reached_%'][1]:.0f}, {fm(f5['median_s'][0])} / {fm(f5['median_s'][1])} |")
        md.append("")
    # polish curves: gap reduction vs added time
    if spec.get("polish"):
        md += ["## Polish: gap reduction against added time", "",
               "| polish (base) | τ s | gap base → polished, mean % | Δ gap pp [95 % CI] | added time mean s | speed-up base → polished |",
               "|---|---|---|---|---|---|"]
        for pz in spec["polish"]:
            sb = summary(recs, pz["base"], None, T)
            for tau in pz["taus"]:
                sp_ = summary(recs, pz["variant"], tau, T)
                pr = paired(recs, pz["variant"], pz["base"], tau, None, T)
                add = sp_["time_mean_s"] - sb["time_mean_s"]
                sel = " **(selected on val)**" if tau == pz.get("tau_sel") else ""
                res["polish"].append(dict(label=pz["label"], base=pz["base"], variant=pz["variant"], tau=tau,
                                          gap_base=sb["gap_mean_%"], gap=sp_["gap_mean_%"], diff=pr["gap_diff_pp"],
                                          diff_ci=pr["gap_diff_ci"], added_s=add, sp_base=sb["speedup_mean"],
                                          sp=sp_["speedup_mean"], selected=bool(sel)))
                md.append(f"| {pz['label']}{sel} | {tau:g} | {sb['gap_mean_%']:.3f} → {sp_['gap_mean_%']:.3f} | "
                          f"{pr['gap_diff_pp']:+.3f} [{pr['gap_diff_ci'][0]:+.3f}, {pr['gap_diff_ci'][1]:+.3f}] | {add:.2f} | "
                          f"{sb['speedup_mean']:.2f} → {sp_['speedup_mean']:.2f} |")
        md.append("")
    # Pareto
    points, curves = [], []
    for k, v in spec["refs"].items():
        sm = summary(recs, v, None, T)
        points.append((k, sm["speedup_mean"], sm["gap_mean_%"], "ref"))
    for fam in spec["families"]:
        for r in fam["rows"]:
            if r.get("plot", True):
                sm = summary(recs, r["variant"], r.get("tau"), T)
                points.append((r.get("short", r["label"]), sm["speedup_mean"], sm["gap_mean_%"], "new"))
    for c in spec.get("curves", []):
        pts = []
        for tau in c["taus"]:
            sm = summary(recs, c["variant"], tau, T)
            if sm and sm["feasible_%"] == 100:
                pts.append((sm["speedup_mean"], sm["gap_mean_%"]))
        res["curves"].append(dict(label=c["label"], variant=c["variant"], taus=c["taus"], pts=pts))
        if pts:
            curves.append((c["label"], pts, c.get("family", "new")))
    md += ["## Pareto front (mean gap vs mean speed-up)", "", f"![Pareto]({os.path.basename(out_png)})", "",
           "| point | speed-up mean | gap mean % |", "|---|---|---|"]
    for lab, sp_, g, fam in sorted(points, key=lambda x: -x[1]):
        md.append(f"| {lab} | {sp_:.2f} | {g:.3f} |")
    md.append("")
    pareto_plot(points, curves, out_png, title)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12")
    ap.add_argument("--runs", default="")
    ap.add_argument("--taus", default="")
    ap.add_argument("--spec", default="", help="selection spec JSON: build the full report")
    a = ap.parse_args()
    if a.spec:
        os.chdir(os.path.dirname(HERE))
        spec = json.load(open(a.spec))
        o = spec["out_prefix"]
        build_report(spec, o + ".json", o + ".md", o + "_pareto.png", spec["title"])
        raise SystemExit(0)
    T = 12 if a.bench == "uc12" else 24
    recs = load_runs(a.runs.split(","))
    vs = []
    for R in recs:
        for k in R:
            if ":" in k or k == "full":
                if k not in vs:
                    vs.append(k)
    taus = [float(x) for x in a.taus.split(",") if x]
    print(f"{len(recs)} instances")
    print(HEADER)
    for v in vs:
        s = summary(recs, v, None, T)
        if s:
            print(fmt_row(s))
        if v.split(":")[0] in ("pas", "ftp", "lb", "grad", "full", "warmfull"):
            for tau in taus:
                s = summary(recs, v, tau, T)
                if s:
                    print(fmt_row(s))
