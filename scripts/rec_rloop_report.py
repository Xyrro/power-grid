"""rloop (arm B) report: test tables with paired bootstrap CIs, quality against K and against wall time, extrapolation.

    python3 scripts/rec_rloop_report.py        # -> results/recursive/rloop_results.{md,json}

Reads results/recursive/rloop_eval_<bench>_s*.jsonl (scripts/rec_rloop_eval.py), rloop_val_summary.json and
rloop_select.json. Gap = (C - DB) / C with DB the dual bound of the full MILP at 0.1 % solved in the same process; C is
the dispatch-LP cost of the method's schedule including priced shedding / reserve shortfall (so a schedule that sheds
has a large gap); served = no shedding and no reserve shortfall. Speed-up = T_full(0.1 %) / T_method per instance
(mean, median, geometric mean); every method's time includes the LP relaxation and all inference, decoding and LPs.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rec_rloop_common as C  # noqa: E402
from otsl.base import boot_ci, paired  # noqa: E402

BENCH_NAMES = {"uc12": "uc12 test_fresh (120, 12 h)", "uc24": "uc24 test (40, 24 h, zero-shot)",
               "hard": "congested benchmark (60, line ratings x 0.6, zero-shot)",
               "line_out": "line-outage shift (30, 1-2 critical lines out, zero-shot)"}


def load(bench):
    recs = []
    for f in sorted(os.listdir(C.RES)):
        if f.startswith(f"rloop_eval_{bench}_") and f.endswith(".jsonl"):
            recs += C.read_jsonl(os.path.join(C.RES, f))
    seen, out = set(), []
    for r in sorted(recs, key=lambda r: r["k"]):
        if r["k"] not in seen:
            seen.add(r["k"])
            out.append(r)
    return out


def ref_full(r):
    """the reference full MILP (0.1 %; in smoke runs the first gap solved)"""
    return r["full"].get("0.001") or next(iter(r["full"].values()))


def best_of(c):
    """c: [[cost, shed, short, cum_time], ...] -> (cost, served, time of the last LP)"""
    j = int(np.argmin([x[0] for x in c]))
    return c[j][0], c[j][1] < 1e-6 and c[j][2] < 1e-6, c[-1][3]


def methods(r, k_sel, thetas=(0.5, 1.0, 2.0, 4.0)):
    """per-method (cost, served, time) of one instance record"""
    out = {}
    f = r["full"]
    for g, name in (("0.001", "full MILP 0.1 %"), ("0.005", "full MILP 0.5 %")):
        if g in f and f[g]["feasible"]:
            out[name] = (f[g]["obj"], f[g]["shed"] < 1e-6 and f[g]["short"] < 1e-6, f[g]["time"])
    c, s, t = best_of(r["lpround"]["c"])
    out["LP rounding + repair, 5 LPs"] = (c, s, r["lpround"]["t_pre"] + t)
    for kk in ("1", "3", "5"):
        if f"m1x_{kk}" in r:
            c, s, t = best_of(r[f"m1x_{kk}"]["c"])
            out[f"m1x ensemble + repair, {kk} LP{'s' if kk != '1' else ''}"] = (c, s, r[f"m1x_{kk}"]["t_pre"] + t)
    if "m1x_5" in r:          # equal-compute screening: the union of both candidate sets (10 LPs)
        c1, s1, t1 = best_of(r["lpround"]["c"])
        c2, s2, t2 = best_of(r["m1x_5"]["c"])
        tt = r["m1x_5"]["t_pre"] + t2 + t1                  # relaxation + m1x inference + both sets of repairs / LPs
        out["LP rounding (5) + m1x (5), 10 LPs"] = (c1, s1, tt) if c1 <= c2 else (c2, s2, tt)
    if "m1x_vit" in r:
        c, s, t = best_of(r["m1x_vit"]["c"])
        out["m1x ensemble, Viterbi + repair, 1 LP"] = (c, s, r["m1x_vit"]["t_pre"] + t)
    R = r["rloop"]
    for K in sorted({1, 2, 4, k_sel, 2 * k_sel, len(R)}):
        if K <= len(R):
            x = R[K - 1]
            out[f"rloop K={K}"] = (x["best"], x["best_shed"] < 1e-6 and x["best_short"] < 1e-6, x["time"])
    for key, name in (("lpls", "LP rounding + local search, LP-gradient ranking (no learning)"),
                      ("m1x_guided", "LP rounding + local search guided by m1x (static)"),
                      ("rloop_guided", "LP rounding + local search guided by rloop (recursive, LP feedback)")):
        if key not in r:
            continue
        L = r[key]
        c1, s1, t1 = best_of(r["lpround"]["c"])
        for K in (4, 8, 16, 32):
            x = L["recs"][min(K, len(L["recs"])) - 1] if L["recs"] else None
            if x is None or x["best"] >= c1:
                out[f"{name}, +{K} LPs"] = (c1, s1, L["t_pre"] + (x["time"] if x else 0))
            else:
                out[f"{name}, +{K} LPs"] = (x["best"], x["best_shed"] < 1e-6 and x["best_short"] < 1e-6, L["t_pre"] + x["time"])
    if "rloop_seeded" in r:
        x = r["rloop_seeded"][-1]
        out[f"rloop seeded with LP rounding, K={len(r['rloop_seeded'])}"] = (
            x["best"], x["best_shed"] < 1e-6 and x["best_short"] < 1e-6, x["time"])
    # halting head: stop at the first step whose halting probability exceeds 0.5 (at most 2 k_sel steps)
    kh = next((i + 1 for i, x in enumerate(R[:2 * k_sel]) if x["halt"] > 0.5), min(len(R), 2 * k_sel))
    x = R[kh - 1]
    out["rloop, learned halting"] = (x["best"], x["best_shed"] < 1e-6 and x["best_short"] < 1e-6, x["time"])
    # safety net on the main pipeline for each threshold theta (records hold the reduced MILP for every instance
    # flagged at the smallest threshold; larger thresholds flag a subset)
    if r.get("main_pipe") == "rloop_guided":
        base = out["LP rounding + local search guided by rloop (recursive, LP feedback), +16 LPs"]
        mname = "rloop-guided search (+16 LPs)"
    else:
        x = R[k_sel - 1]
        base = (x["best"], x["best_shed"] < 1e-6 and x["best_short"] < 1e-6, x["time"])
        mname = f"rloop K={k_sel}"
    for th in thetas:
        flag = (not base[1]) or r["cert_gap"] > th
        if flag and "safety" in r and r["safety"]["feasible"]:
            sf = r["safety"]
            c = min(base[0], sf["obj"])
            s = (sf["shed"] < 1e-6 and sf["short"] < 1e-6) if sf["obj"] <= base[0] else base[1]
            out[f"{mname} + safety net (theta {th:g} %)"] = (c, s, base[2] + sf["t_fix"] + sf["time"], True)
        else:
            out[f"{mname} + safety net (theta {th:g} %)"] = (base[0], base[1], base[2], flag)
    return out


def table(recs, k_sel):
    per = [methods(r, k_sel) for r in recs]
    db = np.array([ref_full(r)["bound"] for r in recs])
    tf = np.array([ref_full(r)["time"] for r in recs])
    names = list(per[0].keys())
    rows, raw = {}, {}
    for nm in names:
        c = np.array([p[nm][0] for p in per])
        s = np.array([p[nm][1] for p in per])
        t = np.array([p[nm][2] for p in per])
        g = (c - db) / c * 100
        sp = tf / t
        row = dict(served=float(s.mean() * 100), gap_mean=float(g.mean()), gap_ci=boot_ci(g),
                   gap_median=float(np.median(g)), gap_max=float(g.max()), time_mean=float(t.mean()),
                   speedup_mean=float(sp.mean()), speedup_median=float(np.median(sp)),
                   speedup_geo=float(np.exp(np.log(sp).mean())), n=int(len(g)))
        if "safety net" in nm:
            row["fallback_rate"] = float(np.mean([p[nm][3] for p in per]) * 100)
        rows[nm] = row
        raw[nm] = (g, np.log(sp), s)
    return rows, raw


def paired_rows(raw, ref_names, main):
    out = {}
    for ref in ref_names:
        if ref not in raw or main not in raw:
            continue
        g, ls, s = raw[main]
        g0, ls0, s0 = raw[ref]
        out[ref] = dict(d_gap=paired(g, g0), d_log_speedup=paired(ls, ls0), d_served=paired(s * 100.0, s0 * 100.0))
    return out


def curves(recs):
    """mean gap of the best schedule against K, and against mean cumulative time"""
    db = np.array([ref_full(r)["bound"] for r in recs])
    K = min(len(r["rloop"]) for r in recs)
    out = []
    for k in range(K):
        c = np.array([r["rloop"][k]["best"] for r in recs])
        g = (c - db) / c * 100
        sv = np.array([r["rloop"][k]["best_shed"] < 1e-6 and r["rloop"][k]["best_short"] < 1e-6 for r in recs])
        t = np.array([r["rloop"][k]["time"] for r in recs])
        nl = np.array([sum(x["new_lp"] for x in r["rloop"][:k + 1]) for r in recs])
        out.append(dict(K=k + 1, gap_mean=float(g.mean()), gap_median=float(np.median(g)), served=float(sv.mean() * 100),
                        time_mean=float(t.mean()), lps_mean=float(nl.mean()),
                        speedup_mean=float(np.mean([ref_full(r)["time"] for r in recs] / t))))
    return out


def fmt_ci(ci):
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def main():
    os.chdir(C.ROOT)
    sel = json.load(open(os.path.join(C.RES, "rloop_select.json"))) if os.path.exists(os.path.join(C.RES, "rloop_select.json")) else {}
    out, md = {"select": sel}, ["# rloop (arm B) results: recursion with the dispatch LP in the loop", "",
                                "*Generated by `scripts/rec_rloop_report.py`; method and discussion in "
                                "[`docs/recursive/rloop.md`](../../docs/recursive/rloop.md).*", ""]
    for bench in ("uc12", "uc24", "hard", "line_out"):
        recs = load(bench)
        if not recs:
            continue
        k_sel = sel.get("K", {}).get(bench, 8)
        rows, raw = table(recs, k_sel)
        main_nm = f"rloop K={k_sel}" if sel.get("main", "rloop") == "rloop" else \
            "LP rounding + local search guided by rloop (recursive, LP feedback), +16 LPs"
        refs = [n for n in rows if (n.startswith("LP rounding") or n.startswith("m1x") or n.startswith("full MILP 0.5"))
                and n != main_nm]
        pr = paired_rows(raw, refs, main_nm)
        sn = next((n for n in rows if n.endswith(f"safety net (theta {sel.get('theta', 1.0):g} %)")), "")
        pr_s = paired_rows(raw, refs, sn) if sn in raw else {}
        cv = curves(recs)
        out[bench] = dict(n=len(recs), k_sel=k_sel, rows=rows, paired=pr, paired_safety=pr_s, curve=cv)
        md += [f"## {BENCH_NAMES[bench]}", "",
               f"{len(recs)} instances; full MILP at 0.1 %: mean {np.mean([ref_full(r)['time'] for r in recs]):.1f} s "
               f"(re-solved in the same process). K = {k_sel} chosen on validation.", "",
               "| method | served | gap mean [95 % CI] | median | max | time (s) | speed-up mean / median / geo |",
               "|---|---|---|---|---|---|---|"]
        for nm, v in rows.items():
            fb = f" (fallback {v['fallback_rate']:.0f} %)" if "fallback_rate" in v else ""
            md.append(f"| {nm}{fb} | {v['served']:.1f} % | {v['gap_mean']:.2f} % [{v['gap_ci'][0]:.2f}, {v['gap_ci'][1]:.2f}] | "
                      f"{v['gap_median']:.2f} % | {v['gap_max']:.2f} % | {v['time_mean']:.2f} | "
                      f"{v['speedup_mean']:.1f}× / {v['speedup_median']:.1f}× / {v['speedup_geo']:.1f}× |")
        for title, P in ((main_nm, pr), (sn, pr_s)):
            if not P:
                continue
            md += ["", f"Paired differences, **{title}** minus each reference (Δ gap < 0: {title} more accurate; "
                       f"Δ log speed-up > 0: {title} faster):", "",
                   "| reference | Δ gap, pp [95 % CI] | Δ log speed-up [CI] | Δ served, pp [CI] |", "|---|---|---|---|"]
            for ref, v in P.items():
                md.append(f"| {ref} | {v['d_gap']['diff']:+.2f} {fmt_ci(v['d_gap']['ci'])} | "
                          f"{v['d_log_speedup']['diff']:+.2f} {fmt_ci(v['d_log_speedup']['ci'])} | "
                          f"{v['d_served']['diff']:+.1f} {fmt_ci(v['d_served']['ci'])} |")
        md += ["", "Quality against the number of outer steps K (best schedule so far; each step one network pass and at "
               "most one LP):", "", "| K | gap mean | median | served | LPs | time (s) | speed-up mean |",
               "|---|---|---|---|---|---|---|"]
        for c in cv:
            md.append(f"| {c['K']} | {c['gap_mean']:.3f} % | {c['gap_median']:.3f} % | {c['served']:.1f} % | "
                      f"{c['lps_mean']:.1f} | {c['time_mean']:.2f} | {c['speedup_mean']:.1f}× |")
        md.append("")
    vs = os.path.join(C.RES, "rloop_val_summary.json")
    if os.path.exists(vs):
        V = json.load(open(vs))["rows"]
        out["validation"] = V
        md += ["## Validation (360 instances, gap to the stored MILP dual bound)", "",
               "| method | served | gap mean | median | max |", "|---|---|---|---|---|"]
        for nm, v in V.items():
            md.append(f"| {nm} | {v['served']:.1f} % | {v['gap_mean']:.3f} % | {v['gap_median']:.3f} % | {v['gap_max']:.2f} % |")
        md.append("")
    json.dump(out, open(os.path.join(C.RES, "rloop_results.json"), "w"), indent=1, default=float)
    open(os.path.join(C.RES, "rloop_results.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
