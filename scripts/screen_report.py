"""Screened network and the certified pipeline (docs/methods/screen.md), report: the validation choice and the test.

    python3 scripts/screen_report.py --select        # validation records -> results/screen/select.json (the choice)
    python3 scripts/screen_report.py                 # test records -> results/screen/report.{md,json}

Scoring as step 2 (scripts/unified_step2.py, whose records are loaded first so that every gap is taken against the
tightest bound known for the instance): gap (C - DB) / C; served: no shed, over-generation or reserve shortfall in any
hour; speed-up: the full MILP at 0.1 % on the B-theta network solved in the same process, divided by the method's time.
The screened MILPs' dual bounds are valid bounds of the full problem (their model is a relaxation of it) and enter DB.

Derived from each certified-pipeline record (same run, nothing re-solved): '(no certificate)' delivers the reduced
schedule (time: LP relaxation + inference + reduced solve); '(LP certificate 1 %)' delivers it when its cost is within
1 % of the LP relaxation's bound and the certified result otherwise.

Validation choice (fixed before the validation run): among the certified pipelines cert:<model>:<delta> (every one
proven within 0.5 %), the highest geometric-mean speed-up over the 360 validation instances; a single network (s0)
is preferred to the ensemble when its speed-up is within 5 % of the best.
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unified_rescore as U  # noqa: E402
import unified_step2 as S2  # noqa: E402

REG, TOL = U.REG, U.TOL
RES = "results/screen"
EPS = 5e-3


def served(v, pre=""):
    return (v.get(pre + "shed") or 0) < TOL and (v.get(pre + "short") or 0) < TOL


def records(folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        if os.path.basename(p).startswith("manifest"):
            continue
        for r in U.jl(p):
            out[(r["bench"], r["set"], r["k"])] = r
    return list(out.values())


def derived(v, eps_lp):
    """(cost, served, time) of a certified-pipeline record without the certificate (eps_lp None) or with the LP
    certificate at eps_lp"""
    t_red = v["parts"].get("lp_relaxation", 0) + v["parts"].get("inference", 0) + v["red_s"]
    if eps_lp is None or v["lp_gap"] <= eps_lp:
        return v["red_cost"], (v.get("red_shed") or 0) < TOL and (v.get("red_short") or 0) < TOL, t_red
    return v["cost"], served(v), v["time"]


# ============================================================================ validation
def select(res):
    R = records(os.path.join(res, "val"))
    if not R:
        sys.exit("no validation records")
    db = {r["k"]: max([r["c_rel"]] + [v["bound"] for n, v in r["runs"].items()
                                      if n.startswith(("full@", "scr:full@")) and U.finite(v.get("bound"))]) for r in R}
    names = sorted({n for r in R for n in r["runs"]})
    rows = []
    for n in names:
        rr = [r for r in R if n in r["runs"] and "full@0.1%" in r["runs"]]
        g = [(r["runs"][n]["cost"] - db[r["k"]]) / r["runs"][n]["cost"] * 100 for r in rr]
        sp = [r["runs"]["full@0.1%"]["time"] / r["runs"][n]["time"] for r in rr]
        rows.append(dict(method=n, n=len(rr), gap_mean=sum(g) / len(g), gap_max=max(g), served=100 * sum(served(r["runs"][n]) for r in rr) / len(rr),
                         sp_geo=math.exp(sum(map(math.log, sp)) / len(sp)), time_mean=sum(r["runs"][n]["time"] for r in rr) / len(rr)))
        if n.startswith("cert:"):
            for lab, e in (("no certificate", None), ("LP certificate 1 %", 1e-2)):
                d = [derived(r["runs"][n], e) for r in rr]
                g = [(c - db[r["k"]]) / c * 100 for (c, s, t), r in zip(d, rr)]
                sp = [r["runs"]["full@0.1%"]["time"] / t for (c, s, t), r in zip(d, rr)]
                rows.append(dict(method=f"{n} ({lab})", n=len(rr), gap_mean=sum(g) / len(g), gap_max=max(g),
                                 served=100 * sum(s for c, s, t in d) / len(d), sp_geo=math.exp(sum(map(math.log, sp)) / len(sp)),
                                 time_mean=sum(t for c, s, t in d) / len(d)))
    certs = [x for x in rows if x["method"].startswith("cert:") and "(" not in x["method"] and not x["method"].startswith("cert:lp")]
    best = max(certs, key=lambda x: x["sp_geo"])
    single = [x for x in certs if x["method"].startswith("cert:s0:")]
    s_best = max(single, key=lambda x: x["sp_geo"]) if single else None
    choice = s_best if s_best and s_best["sp_geo"] >= 0.95 * best["sp_geo"] else best
    out = dict(n=len(R), choice=choice["method"], rule="max geometric-mean speed-up among cert:<model>:<delta>; s0 if "
               "within 5 % of the best", rows=sorted(rows, key=lambda x: -x["sp_geo"]))
    os.makedirs(res, exist_ok=True)
    json.dump(out, open(os.path.join(res, "select.json"), "w"), indent=1)
    print(f"{len(R)} validation instances; choice: {choice['method']}")
    for x in out["rows"]:
        print(f"  {x['method']:42s} gap {x['gap_mean']:6.3f} % (max {x['gap_max']:5.2f})  served {x['served']:5.1f} %  "
              f"time {x['time_mean']:6.2f} s  speed-up {x['sp_geo']:5.2f}x")


# ============================================================================ test
def load_test(res):
    for r in records(os.path.join(res, "test")):
        x = S2.inst_key(r["bench"], r["set"], r["k"])
        src = "screen/test"
        REG.lp(x, r["c_rel"], src)
        tref = r["runs"]["full@0.1%"]["time"]
        for name, v in r["runs"].items():
            if name.startswith(("full@", "scr:full@")) and U.finite(v.get("bound")):
                REG.bound(x, v["bound"], src)
            mt = U.meta(src, "screen", "complete", "S3", cpu=r.get("cpu_model"), highs=r.get("highs"))
            REG.put(f"S3 | {name}", x, v["cost"], served(v), v["time"], tref, mt, fellback=bool(v.get("fallback")))
            if name.startswith("cert:"):
                for lab, e in (("no certificate", None), ("LP certificate 1 %", 1e-2)):
                    c, s, t = derived(v, e)
                    REG.put(f"S3 | {name} ({lab})", x, c, s, t, tref, mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--partial", action="store_true", help="tables over the instances present")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.select:
        select(a.res)
        return
    sel = json.load(open(os.path.join(a.res, "select.json")))["choice"]
    S2.ARGS = argparse.Namespace(partial=False)
    U.load_all()                                       # step 1 and step 2, as scripts/unified_step2.py (tightest bounds)
    S2.load_fix(S2.RES)
    for tag in ("fixtune2", "fixtune3", "fixorig"):
        S2.load_fix_addon(S2.RES, tag)
    S2.load_arm_a(S2.RES)
    S2.load_arm_b(S2.RES, "uc12", "armB", "S2·B")
    S2.load_arm_b(S2.RES, "uc12", "armB0", "S2·B0")
    for sd in (1, 2, 3):
        S2.load_arm_b(S2.RES, "uc12", f"armBs{sd}", f"S2·Bs{sd}")
    for b in ("uc24", "hard", "line_out"):
        S2.load_arm_b(S2.RES, b, "armB", "S2·B")
    S2.load_hard(S2.RES)
    S2.load_longmilp(S2.RES)
    load_test(a.res)
    ref = REG.methods.get("S3 | full@0.1%", {"data": {}})["data"]
    S2.ref_times = lambda insts: {x: [ref[x][2]] for x in insts if x in ref and ref[x][2]}   # 'common ref': this run's
    present ={x for n, M in REG.methods.items() if n.startswith("S3") for x in M["data"]}
    C = f"S3 | {sel}"
    rows12 = [("Full MILP, 0.1 % gap (reference)", "S3 | full@0.1%"), ("Full MILP, 0.5 % gap", "S3 | full@0.5%"),
              ("Screened full MILP, 0.1 % gap (no learning)", "S3 | scr:full@0.1%"),
              ("Screened full MILP, 0.5 % gap (no learning)", "S3 | scr:full@0.5%"),
              ("Recommended pipeline (step 2)", "S3 | pipe@0.5%"), ("Pipeline + bound check (step 2)", "S3 | pipe+bc"),
              ("Its fixings on the screened network", "S3 | scr:pipe"),
              ("Certified pipeline (proven within 0.5 %)", C),
              ("Certified pipeline, LP certificate at 1 % (proven within 1 %)", f"{C} (LP certificate 1 %)"),
              ("Certified pipeline without the certificate", f"{C} (no certificate)"),
              ("Certified pipeline on the LP values (no learning)", "S3 | cert:lp:0.05"),
              ("… LP certificate at 1 %", "S3 | cert:lp:0.05 (LP certificate 1 %)"),
              ("… without the certificate", "S3 | cert:lp:0.05 (no certificate)")]
    rows_nl = [r for r in rows12 if "pipe" not in r[1] and sel not in r[1]]
    TAB = {"12 h, test_fresh": (("tf", 120), rows12), "24 hours (uc24)": (("u24", 40), rows_nl),
           "Congested lines (ratings × 0.6)": (("hard", 60), rows_nl)}
    R = dict(choice=sel, tables={})
    L = ["# Screened network and the certified pipeline: test", "",
         f"Generated by `scripts/screen_report.py` from `results/screen/test/`; scoring as step 2 (`docs/unified_metrics.md`), "
         f"gaps against the tightest bound known per instance. Certified pipeline: `{sel}` (validation choice, "
         "`results/screen/select.json`). Every method of an instance ran back to back in one single-threaded process on "
         "one core; speed-ups are paired with that process's full MILP at 0.1 %.", ""]
    for title, ((key, n), rows) in TAB.items():
        insts = [(key, k) for k in range(n)]
        if a.partial:
            insts = [x for x in insts if x in present]
        if insts:
            S2.table(title, insts, rows, R, L)
    # shifted sets
    shift_rows = [r for r in rows12 if r[1] in ("S3 | full@0.1%", "S3 | full@0.5%", "S3 | scr:full@0.5%", "S3 | pipe@0.5%",
                                                 "S3 | pipe+bc", "S3 | scr:pipe", C, f"{C} (LP certificate 1 %)",
                                                 f"{C} (no certificate)")]
    L += ["## Shifted 12-hour sets: mean gap · served · speed-up (geometric mean)", "",
          "| method | " + " | ".join(U.SHIFTS) + " |", "|---|" + "---|" * len(U.SHIFTS)]
    for lab, nme in shift_rows:
        cells = []
        for sft in U.SHIFTS:
            insts = [(sft, k) for k in range(30)]
            if a.partial:
                insts = [x for x in insts if nme in REG.methods and x in REG.methods[nme]["data"]]
            if insts and nme in REG.methods and U.complete_on(nme, insts):
                s = U.stats(nme, insts)
                R["tables"].setdefault("shifted", {}).setdefault(lab, {})[sft] = s
                cells.append(f"{s['gap_mean']:.2f} % · {s['served']:.0f} % · {s['sp_geo']:.1f}×")
            else:
                cells.append("–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    # paired comparisons
    tf = [("tf", k) for k in range(120)]
    sh = [(sft, k) for sft in U.SHIFTS for k in range(30)]
    if a.partial:
        tf = [x for x in tf if x in present]
        sh = [x for x in sh if x in present]
    P = [(C, "S3 | full@0.5%", tf), (C, "S3 | pipe@0.5%", tf), (C, "S3 | pipe+bc", tf), (C, "S3 | scr:pipe", tf),
         (f"{C} (no certificate)", "S3 | scr:pipe", tf), (f"{C} (no certificate)", "S3 | pipe@0.5%", tf),
         (f"{C} (LP certificate 1 %)", "S3 | pipe@0.5%", tf), (C, "S3 | cert:lp:0.05", tf),
         ("S3 | scr:full@0.5%", "S3 | full@0.5%", tf), ("S3 | scr:full@0.1%", "S3 | full@0.1%", tf),
         ("S3 | scr:pipe", "S3 | pipe@0.5%", tf), (C, "S3 | pipe+bc", sh), (C, "S3 | full@0.5%", sh)]
    for bench, key, n in (("uc24", "u24", 40), ("hard", "hard", 60)):
        ins = [(key, k) for k in range(n)]
        if a.partial:
            ins = [x for x in ins if x in present]
        P += [("S3 | scr:full@0.5%", "S3 | full@0.5%", ins), ("S3 | scr:full@0.1%", "S3 | full@0.1%", ins)]
    L += ["## Paired comparisons", "",
          "Per instance: A's gap minus B's (negative: A is closer to the bound), and A's paired speed-up over B's "
          "(geometric mean; above 1: A is faster). 95 % bootstrap intervals over instances.", "",
          "| A | B | instances | gap A | gap B | A − B, pp [95 % CI] | A better / B better | speed-up A over B [95 % CI] |",
          "|---|---|---|---|---|---|---|---|"]
    R["paired"] = []
    for na, nb, ins in P:
        if not ins or not all(n in REG.methods and U.complete_on(n, ins) for n in (na, nb)):
            continue
        p = S2.paired(na, nb, ins, False)
        p["set"] = "shifted" if ins[0][0] in U.SHIFTS else ins[0][0]
        R["paired"].append(p)
        sp = f"{p['speed']:.2f}× [{p['speed_ci'][0]:.2f}, {p['speed_ci'][1]:.2f}]" if "speed" in p else "–"
        L.append(f"| {na[5:]} | {nb[5:]} ({p['set']}) | {p['n']} | {p['gap_a']:.3f} | {p['gap_b']:.3f} | {p['diff']:+.3f} "
                 f"[{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}] | {p['a_better']} / {p['b_better']} | {sp} |")
    L.append("")
    # anatomy of the certified pipelines: fixed share, rounds, how each answer was certified, where the time goes
    L += ["## Anatomy of the certified pipelines", "",
          "| pipeline | set | instances | fixed | lines monitored (start) | screen rounds | certified by the LP bound | "
          "by the MILP | time: LP relaxation + inference / reduced MILP / certificate (s, mean) |",
          "|---|---|---|---|---|---|---|---|---|"]
    recs_t = records(os.path.join(a.res, "test"))
    for name in sorted({n for r in recs_t for n in r["runs"] if n.startswith("cert:")}):
        for title, keep in (("12 h test_fresh", lambda r: r["bench"] == "uc12" and r["set"] == "tf"),
                            ("shifted 12 h", lambda r: r["bench"] == "uc12" and r["set"] != "tf"),
                            ("24 h", lambda r: r["bench"] == "uc24"), ("congested", lambda r: r["bench"] == "hard")):
            vs = [(r, r["runs"][name]) for r in recs_t if keep(r) and name in r["runs"]]
            if not vs:
                continue
            nT = [12 * 73 if r["bench"] != "uc24" else 24 * 73 for r, _ in vs]
            fx = sum(v["n_fixed"] / n for (r, v), n in zip(vs, nT)) / len(vs)
            mean = lambda f: sum(f(v) for _, v in vs) / len(vs)
            n_lp = sum(v["cert"] == "lp" for _, v in vs)
            L.append(f"| {name} | {title} | {len(vs)} | {100 * fx:.1f} % | {mean(lambda v: v['n_mon']):.1f} | "
                     f"{mean(lambda v: v['red_rounds']):.2f} | {n_lp} | {len(vs) - n_lp} | "
                     f"{mean(lambda v: v['parts'].get('lp_relaxation', 0) + v['parts'].get('inference', 0)):.2f} / "
                     f"{mean(lambda v: v['red_s']):.2f} / {mean(lambda v: v['cert_s']):.2f} |")
    L.append("")
    # checks: certificates hold; same references as step 2
    L += ["## Checks", ""]
    bad, n_cert = [], 0
    for r in records(os.path.join(a.res, "test")):
        for name, v in r["runs"].items():
            if not name.startswith("cert:"):
                continue
            n_cert += 1
            b = r["c_rel"] if v["cert"] == "lp" else max(v.get("cert_bound", -math.inf), r["c_rel"])
            obj = min(v["red_obj"], v.get("cert_obj", math.inf))
            if (obj - b) / obj > EPS + 1e-9:
                bad.append((r["bench"], r["set"], r["k"], name, v.get("cert_status")))
    L.append(f"* Certificates: {n_cert - len(bad)} of {n_cert} certified-pipeline results proven within 0.5 % by their "
             f"own bound (the LP relaxation's or the screened MILP's); not proven: {len(bad)} {bad[:5]} "
             "(a solve that hit the time limit).")
    lines_ok = [v.get("lines_ok", True) for r in records(os.path.join(a.res, "test")) for n, v in r["runs"].items()
                if n.startswith("scr:")]
    L.append(f"* Screened solves ending with every line within its rating: {sum(lines_ok)} of {len(lines_ok)}.")
    same = []
    for nm in ("full@0.1%", "full@0.5%", "pipe@0.5%"):
        a2, a3 = f"S2·fix | {nm}", f"S3 | {nm}"
        if a2 in REG.methods and a3 in REG.methods:
            xs = [x for x in REG.methods[a3]["data"] if x in REG.methods[a2]["data"]]
            eq = sum(abs(REG.methods[a2]["data"][x][0] - REG.methods[a3]["data"][x][0]) <= 1e-3 * REG.methods[a3]["data"][x][0] for x in xs)
            same.append(f"{nm} {eq}/{len(xs)} within 0.1 % of step 2's cost")
    L.append("* Same methods as step 2: " + "; ".join(same) + ".")
    below = [x for n, M in REG.methods.items() if n.startswith("S3") for x, d in M["data"].items()
             if U.db_of(x) is not None and d[0] < U.db_of(x) * (1 - 1e-7)]
    L.append(f"* Delivered costs below the tightest bound: {len(below)}.")
    hosts = sorted({(r.get('cpu_model'), r.get('highs')) for r in records(os.path.join(a.res, 'test'))})
    L.append(f"* Hardware: {hosts}.")
    R["checks"] = dict(not_proven=bad, n_cert=n_cert, lines_ok=[sum(lines_ok), len(lines_ok)], below=len(below), hosts=hosts)
    os.makedirs(a.res, exist_ok=True)
    open(os.path.join(a.res, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump(R, open(os.path.join(a.res, "report.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
