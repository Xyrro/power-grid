"""Stacking follow-up: do partial fixing and a loose solver gap add up? Tables for the "Stacking" section of
results/<bench>/base_results.md (and the "stacking" key of base_results.json).

    python3 scripts/uc_base_stack_report.py --bench uc12
    python3 scripts/uc_base_stack_report.py --bench uc24

Inputs: results/<bench>/base_eval_test.jsonl (first pass: full MILP at 0.1 % = T_ref and at 0.25 / 0.5 / 1 %, every rule
at 0.1 %) and results/<bench>/base_stack_test.jsonl (stacking pass, one process per benchmark, every run of an instance
back to back: the full MILP at 0.5 / 1 % re-run, every rule's reduced MILP at 0.5 / 1 %, the 12-hour fused rules also at
0.1 %). Paper metrics against the reference dual bound; speed-up = T_ref / T_method (T_ref from the first pass; the
re-run loose-gap full MILPs measure the timing drift between the passes); paired comparisons against the full MILP
at the same mip_rel_gap from the same pass (Δ gap pp, Δ log speed-up; bootstrap 95 % CI). Pareto view (descriptive,
on test): a (rule, gap) point is dominated if another point has a lower-or-equal mean gap and a higher-or-equal
speed-up, one of them strictly.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import load_b3  # noqa: E402
from otsl.base import boot_ci, paired  # noqa: E402
from uc_base_report import CFG, G, fmt, jl, ref_bounds  # noqa: E402

GAPS = (0.001, 0.005, 0.01)
LBL = {0.001: "0.1 %", 0.005: "0.5 %", 0.01: "1 %"}


def arr(recs, name, db, TG, key_t="t_method"):
    n = len(recs)
    feas, cost, tm, fixed = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    served = np.zeros(n, bool)
    for k, r in enumerate(recs):
        x = r["runs"].get(name)
        if x is None:
            continue
        feas[k] = bool(x["feasible"])
        tm[k] = x.get(key_t, x["time"])
        if feas[k]:
            cost[k] = x["obj"]
            fixed[k] = x.get("n_fixed", 0) / TG
            served[k] = (x.get("shed") or 0) < 1e-6 and (x.get("short") or 0) < 1e-6
    with np.errstate(invalid="ignore"):
        gap = np.where(feas, (cost - db) / cost * 100, np.nan)
    return dict(feas=feas, gap=gap, tm=tm, fixed=fixed, served=served)


def stats(P, t_ref):
    f = P["feas"]
    g, sp = P["gap"][f], t_ref[f] / P["tm"][f]
    return dict(feasible=float(f.mean() * 100), gap_mean=float(g.mean()), gap_ci=boot_ci(g), gap_median=float(np.median(g)),
                speedup_mean=float(sp.mean()), speedup_ci=boot_ci(sp), speedup_median=float(np.median(sp)),
                speedup_geomean=float(np.exp(np.log(sp).mean())), fixed=float(np.nanmean(P["fixed"][f]) * 100),
                served=float(P["served"][f].mean() * 100))


def main(bench):
    cfg = CFG[bench]
    R = os.path.join("results", bench)
    first = {r["i"]: r for r in jl(os.path.join(R, "base_eval_test.jsonl"))}
    stack = {r["i"]: r for r in jl(os.path.join(R, "base_stack_test.jsonl"))}
    idx = np.array(sorted(set(first) & set(stack)))
    A, B = [first[i] for i in idx], [stack[i] for i in idx]
    d = load_b3(cfg["test"])
    db, _, _ = ref_bounds(bench, d, idx)
    TG = cfg["T"] * G
    t_ref = np.array([r["runs"]["full"]["time"] for r in A])
    # ---------------- timing drift between the passes (same full-MILP settings re-run)
    drift = {}
    for g in (0.005, 0.01):
        nm = f"full gap={g * 100:g}%"
        t1 = np.array([r["runs"][nm]["time"] for r in A])
        t2 = np.array([r["runs"][nm]["time"] for r in B])
        same = np.mean([abs(a["runs"][nm]["obj"] - b["runs"][nm]["obj"]) <= 1e-6 * abs(a["runs"][nm]["obj"]) for a, b in zip(A, B)])
        drift[nm] = dict(ratio_median=float(np.median(t2 / t1)), within_10pct=float(np.mean(np.abs(t2 / t1 - 1) <= 0.1) * 100),
                         ratio_of_means=float(t2.mean() / t1.mean()), same_obj=float(same * 100))
    # ---------------- every (rule, gap) point
    rules = sorted({x.get("rule", nm) for r in B for nm, x in r["runs"].items() if not nm.startswith("full")})
    pts, src_of = {}, {}
    full = {0.001: arr(A, "full", db, TG, "time"), 0.005: arr(B, "full gap=0.5%", db, TG, "time"),
            0.01: arr(B, "full gap=1%", db, TG, "time")}
    for g, P in full.items():
        pts[("full MILP", g)] = P
        src_of[("full MILP", g)] = "first pass" if g == 0.001 else "stack pass"
    for ru in rules:
        for g in GAPS:
            nm = ru if g == 0.001 else f"{ru} @gap={g * 100:g}%"
            src = B if all(nm in r["runs"] for r in B) else (A if all(nm in r["runs"] for r in A) else None)
            if src is not None:
                pts[(ru, g)] = arr(src, nm, db, TG)
                src_of[(ru, g)] = "stack pass" if src is B else "first pass"
    rows = []
    for (ru, g), P in pts.items():
        st = stats(P, t_ref)
        F = full[g]
        both = P["feas"] & F["feas"]
        if ru != "full MILP":
            st["vs_full_same_gap"] = dict(
                d_gap=paired(np.where(both, P["gap"], np.nan), F["gap"]),
                d_log_speedup=paired(np.where(both, np.log(F["tm"] / P["tm"]), np.nan), np.zeros(len(idx))))
        st.update(rule=ru, gap=g, source=src_of[(ru, g)])
        rows.append(st)
    # ---------------- Pareto view (mean gap vs mean speed-up, and vs geometric mean)
    for key in ("speedup_mean", "speedup_geomean"):
        for r in rows:
            r[f"dominated_{key}"] = any((o["gap_mean"] <= r["gap_mean"] and o[key] >= r[key]
                                         and (o["gap_mean"] < r["gap_mean"] or o[key] > r[key])) for o in rows if o is not r)
    res = dict(n=int(len(idx)), drift=drift, rows=rows)
    json.dump(res, open(os.path.join(R, "base_stack.json"), "w"), indent=1, default=float)
    J = json.load(open(os.path.join(R, "base_results.json")))
    J["stacking"] = res
    json.dump(J, open(os.path.join(R, "base_results.json"), "w"), indent=1, default=float)
    write_md(bench, res, R)
    return res


def ci(c, nd=2, sign=False):
    return f"[{fmt(c[0], nd, sign)}, {fmt(c[1], nd, sign)}]"


def write_md(bench, res, R):
    L = ["## Stacking: partial fixing on top of a loose solver gap", "",
         f"Generated by `scripts/uc_base_stack_report.py --bench {bench}`. {res['n']} instances; every rule's reduced MILP "
         "at mip_rel_gap 0.5 % and 1 %, with the full MILP at the same gaps re-run back to back in the same pass. Gap to "
         "the reference dual bound; speed-up against the first pass's 0.1 % full MILP (T_ref). Timing drift between the "
         "passes (loose-gap full MILP re-run / first pass, median): "
         + ", ".join(f"{k}: {v['ratio_median']:.2f} ({v['within_10pct']:.0f} % within 10 %, same objective {v['same_obj']:.0f} %)"
                     for k, v in res["drift"].items()) + ".", "",
         "| rule | mip_rel_gap | feasible % | gap mean % [CI] | speed-up mean [CI] | median | geo. mean | fixed % | vs full MILP at the same gap: Δ gap pp [CI] | Δ log speed-up [CI] | Pareto (mean / geo.) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["full MILP"] + sorted({r["rule"] for r in res["rows"]} - {"full MILP"})
    for ru in order:
        for r in sorted([x for x in res["rows"] if x["rule"] == ru], key=lambda x: x["gap"]):
            v = r.get("vs_full_same_gap")
            par = ("frontier" if not r["dominated_speedup_mean"] else "–") + " / " + ("frontier" if not r["dominated_speedup_geomean"] else "–")
            L.append(f"| {ru} | {LBL[r['gap']]} | {r['feasible']:.1f} | {fmt(r['gap_mean'])} {ci(r['gap_ci'])} | "
                     f"{fmt(r['speedup_mean'], 1)} {ci(r['speedup_ci'], 1)} | {fmt(r['speedup_median'], 1)} | {fmt(r['speedup_geomean'], 1)} | "
                     f"{fmt(r['fixed'], 1)} | "
                     + (f"{fmt(v['d_gap']['diff'], 2, True)} {ci(v['d_gap']['ci'], 2, True)} | "
                        f"{fmt(v['d_log_speedup']['diff'], 2, True)} {ci(v['d_log_speedup']['ci'], 2, True)}" if v else "– | –")
                     + f" | {par} |")
    sec = "\n".join(L) + "\n"
    p = os.path.join(R, "base_results.md")
    md = open(p).read()
    if "## Stacking" in md:
        a = md.index("## Stacking")
        b = md.find("\n## ", a + 5)
        md = md[:a] + sec + (md[b + 1:] if b >= 0 else "")
    else:
        md = md.rstrip("\n") + "\n\n" + sec
    open(p, "w").write(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=["uc12", "uc24"])
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    main(a.bench)
