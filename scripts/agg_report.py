"""Round 8 (docs/methods/agg.md), report: the validation choices and the test.

    python3 scripts/agg_report.py --select        # validation records -> results/agg/select.json (the choices)
    python3 scripts/agg_report.py                 # test records -> results/agg/report.{md,json}

Scoring as step 2 and round 7 (scripts/unified_step2.py, scripts/screen_report.py, whose records are loaded first so
that every gap is taken against the tightest bound known for the instance): gap (C - DB) / C; served: no shed,
over-generation or reserve shortfall; speed-up: the full MILP at 0.1 % on the B-theta network solved in the same
process, divided by the method's time. The screened and aggregated MILPs' bounds are valid for the full problem.

Derived from each certified-pipeline record (same run, nothing re-solved): '(no certificate)' delivers the reduced
schedule (time: LP relaxation + inference + reduced solve; confidence fixing alone, RACLearn's rule); '(LP certificate
1 %)' delivers it when its cost is within 1 % of the LP relaxation's bound and the certified result otherwise.

Validation choices (fixed before the validation run, docs/methods/agg.md): (1) learned: the aggregated certified
pipeline with a learned predictor (s0, ens5, knn) with the highest geometric-mean speed-up over the 360 instances, s0
preferred when within 5 % of the best; (2) learning-free: the lag / lp variant with the highest geometric-mean
speed-up.
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import screen_report as SR  # noqa: E402
import unified_rescore as U  # noqa: E402
import unified_step2 as S2  # noqa: E402

REG, TOL = U.REG, U.TOL
RES = "results/agg"
EPS = 5e-3
LEARNED = ("s0", "ens5", "knn")
served, records = SR.served, SR.records


def derived(v, eps_lp):
    """round 7's derived operating points; an answer proven at the rounding step (cert 'round') has no reduced
    schedule: it is its own uncertified and LP-certified result"""
    if v.get("cert") == "round":
        return v["cost"], served(v), v["time"]
    return SR.derived(v, eps_lp)


def is_cert(n):
    return n.startswith(("cert:", "acert:"))


def pred_of(n):
    return n.split(":")[1].split("+")[0]


# ============================================================================ validation
def select(res):
    """validation passes: val (the grid) and val2 (cheapest-first candidates, added after the exploration); each method's
    speed-up is paired with the reference solved in its own process; a method in both passes is reported per pass"""
    passes = [(f, records(os.path.join(res, f))) for f in ("val", "val2") if os.path.isdir(os.path.join(res, f))]
    passes = [(f, R) for f, R in passes if R]
    if not passes:
        sys.exit("no validation records")
    db = {}
    for _, R in passes:
        for r in R:
            b = [r["c_rel"]] + [v[x] for n, v in r["runs"].items() for x in ("bound", "cert_bound") if U.finite(v.get(x))]
            db[r["k"]] = max(db.get(r["k"], -math.inf), max(b))
    rows = []
    for f, R in passes:
        rows += pass_rows(R, db, "" if f == "val" else f" [{f}]")
    finish_select(res, rows, sum(len(R) for _, R in passes), max(len(R) for _, R in passes))


def pass_rows(R, db, suffix):
    names = sorted({n for r in R for n in r["runs"]})
    rows = []
    for n in names:
        rr = [r for r in R if n in r["runs"] and "full@0.1%" in r["runs"]]
        variants = [(n, None)] + ([(f"{n} (no certificate)", "none"), (f"{n} (LP certificate 1 %)", 1e-2)] if is_cert(n) else [])
        for lab, e in variants:
            if e is None:
                d = [(r["runs"][n]["cost"], served(r["runs"][n]), r["runs"][n]["time"]) for r in rr]
            else:
                d = [derived(r["runs"][n], None if e == "none" else e) for r in rr]
            g = [(c - db[r["k"]]) / c * 100 for (c, s, t), r in zip(d, rr)]
            sp = [r["runs"]["full@0.1%"]["time"] / t for (c, s, t), r in zip(d, rr)]
            x = dict(method=lab + suffix, n=len(rr), gap_mean=sum(g) / len(g), gap_max=max(g), served=100 * sum(s for c, s, t in d) / len(d),
                     sp_geo=math.exp(sum(map(math.log, sp)) / len(sp)), time_mean=sum(t for c, s, t in d) / len(d))
            if e is None and is_cert(n):
                x["by_lp"] = sum(r["runs"][n]["cert"] in ("lp", "round") for r in rr)
            rows.append(x)
    return rows


def finish_select(res, rows, n_rec, n_inst):
    acert = [x for x in rows if x["method"].startswith("acert:") and "(" not in x["method"]]
    learned = [x for x in acert if pred_of(x["method"]) in LEARNED]
    free = [x for x in acert if pred_of(x["method"]) not in LEARNED]
    best = max(learned, key=lambda x: x["sp_geo"])
    single = [x for x in learned if pred_of(x["method"]) == "s0"]
    s_best = max(single, key=lambda x: x["sp_geo"]) if single else None
    choice = s_best if s_best and s_best["sp_geo"] >= 0.95 * best["sp_geo"] else best
    choice_free = max(free, key=lambda x: x["sp_geo"])
    strip = lambda m: m.split(" [")[0]
    out = dict(n=n_inst, records=n_rec, choice=strip(choice["method"]), choice_free=strip(choice_free["method"]),
               rule="learned: max geometric-mean speed-up among acert with s0 / ens5 / knn, s0 if within 5 % of the best; "
                    "learning-free: max geometric-mean speed-up among acert with lag / lp",
               rows=sorted(rows, key=lambda x: -x["sp_geo"]))
    os.makedirs(res, exist_ok=True)
    json.dump(out, open(os.path.join(res, "select.json"), "w"), indent=1)
    print(f"{n_inst} validation instances; learned choice: {out['choice']}; learning-free choice: {out['choice_free']}")
    for x in out["rows"]:
        print(f"  {x['method']:46s} gap {x['gap_mean']:6.3f} % (max {x['gap_max']:5.2f})  served {x['served']:5.1f} %  "
              f"time {x['time_mean']:6.2f} s  speed-up {x['sp_geo']:5.2f}x" + (f"  LP-certified {x['by_lp']}" if "by_lp" in x else ""))


def select9(res):
    """round 9 (rule fixed before its validation pass): the margin networks (m0 / mens5) against round 8's learned
    choice in the same processes (results/agg/val3); the best m-variant by geometric-mean speed-up replaces it only if
    the paired speed-up over it is above 1"""
    import numpy as np
    R = records(os.path.join(res, "val3"))
    if not R:
        sys.exit("no round 9 validation records")
    base = json.load(open(os.path.join(res, "select.json")))["choice"]
    db = {r["k"]: max([r["c_rel"]] + [v[x] for v in r["runs"].values() for x in ("bound", "cert_bound") if U.finite(v.get(x))])
          for r in R}
    rows = pass_rows(R, db, "")
    cand = [x for x in rows if x["method"].startswith("acert:m") and "(" not in x["method"]]
    best = max(cand, key=lambda x: x["sp_geo"])
    rng = np.random.default_rng(0)
    pairs = [(r["runs"][base]["time"] / r["runs"][best["method"]]["time"],
              (r["runs"][best["method"]]["cost"] - db[r["k"]]) / r["runs"][best["method"]]["cost"]
              - (r["runs"][base]["cost"] - db[r["k"]]) / r["runs"][base]["cost"]) for r in R
             if base in r["runs"] and best["method"] in r["runs"]]
    ls, dg = np.log([p[0] for p in pairs]), np.array([p[1] for p in pairs]) * 100
    bs = [math.exp(rng.choice(ls, len(ls)).mean()) for _ in range(2000)]
    bd = [rng.choice(dg, len(dg)).mean() for _ in range(2000)]
    sp = math.exp(ls.mean())
    out = dict(n=len(R), base=base, best=best["method"], speed_vs_base=sp, speed_ci=[np.percentile(bs, 2.5), np.percentile(bs, 97.5)],
               gap_diff_pp=float(dg.mean()), gap_ci=[np.percentile(bd, 2.5), np.percentile(bd, 97.5)], adopt=bool(sp > 1.0),
               rule="best margin-network variant by geometric-mean speed-up; adopted if its paired speed-up over round 8's "
                    "learned choice (same processes) is above 1", rows=sorted(rows, key=lambda x: -x["sp_geo"]))
    json.dump(out, open(os.path.join(res, "select9.json"), "w"), indent=1, default=float)
    print(f"{len(R)} instances; best {best['method']} vs {base}: {sp:.3f}x [{out['speed_ci'][0]:.3f}, {out['speed_ci'][1]:.3f}], "
          f"gap {out['gap_diff_pp']:+.3f} pp -> adopt {out['adopt']}")
    for x in out["rows"]:
        if "(" not in x["method"]:
            print(f"  {x['method']:34s} gap {x['gap_mean']:6.3f} % (max {x['gap_max']:5.2f})  time {x['time_mean']:6.2f} s  "
                  f"speed-up {x['sp_geo']:5.2f}x" + (f"  LP-certified {x['by_lp']}" if "by_lp" in x else ""))


# ============================================================================ test
def load_test(res, tag="S4"):
    for r in records(os.path.join(res, "test")):
        x = S2.inst_key(r["bench"], r["set"], r["k"])
        src = "agg/test"
        REG.lp(x, r["c_rel"], src)
        tref = r["runs"]["full@0.1%"]["time"]
        for name, v in r["runs"].items():
            for b in ("bound", "cert_bound"):
                if (name.startswith(("full@", "scr:full@", "agg:full@")) or b == "cert_bound") and U.finite(v.get(b)):
                    REG.bound(x, v[b], src)
            mt = U.meta(src, "agg", "complete", tag, cpu=r.get("cpu_model"), highs=r.get("highs"))
            REG.put(f"{tag} | {name}", x, v["cost"], served(v), v["time"], tref, mt, fellback=bool(v.get("fallback")))
            if is_cert(name):
                for lab, e in (("no certificate", None), ("LP certificate 1 %", 1e-2)):
                    c, s, t = derived(v, e)
                    REG.put(f"{tag} | {name} ({lab})", x, c, s, t, tref, mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--select9", action="store_true", help="round 9: margin networks against round 8's learned choice")
    ap.add_argument("--partial", action="store_true", help="tables over the instances present")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.select:
        select(a.res)
        return
    if a.select9:
        select9(a.res)
        return
    sj = json.load(open(os.path.join(a.res, "select.json")))
    sel, self_ = sj["choice"], sj["choice_free"]
    S2.ARGS = argparse.Namespace(partial=False)
    U.load_all()                                       # step 1 and step 2 (tightest bounds), as round 7's report
    S2.load_fix(S2.RES)
    for tag in ("fixtune2", "fixtune3", "fixorig"):
        S2.load_fix_addon(S2.RES, tag)
    S2.load_arm_a(S2.RES)
    S2.load_arm_b(S2.RES, "uc12", "armB", "S2·B")
    for b in ("uc24", "hard", "line_out"):
        S2.load_arm_b(S2.RES, b, "armB", "S2·B")
    S2.load_hard(S2.RES)
    S2.load_longmilp(S2.RES)
    SR.load_test(SR.RES)                               # round 7's records: their bounds count too
    load_test(a.res)
    ref = REG.methods.get("S4 | full@0.1%", {"data": {}})["data"]
    S2.ref_times = lambda insts: {x: [ref[x][2]] for x in insts if x in ref and ref[x][2]}
    present = {x for n, M in REG.methods.items() if n.startswith("S4") for x in M["data"]}
    C, F = f"S4 | {sel}", f"S4 | {self_}"
    abl = "S4 | acert:s0:0.02"
    rows12 = [("Full MILP, 0.1 % gap (reference)", "S4 | full@0.1%"), ("Full MILP, 0.5 % gap", "S4 | full@0.5%"),
              ("Learning to Fix (kNN, ε = 1 %, as published)", "S4 | ltf_knn_1"),
              ("Round 7: screened full MILP, 0.5 %", "S4 | scr:full@0.5%"),
              ("Aggregated screened full MILP, 0.1 % (no learning)", "S4 | agg:full@0.1%"),
              ("Aggregated screened full MILP, 0.5 % (no learning)", "S4 | agg:full@0.5%"),
              ("Round 7: certified pipeline", "S4 | cert:s0:0.02"),
              ("Round 7: certified pipeline on the LP values (no learning)", "S4 | cert:lp:0.05"),
              (f"Aggregated certified pipeline, learned ({sel[6:]})", C),
              ("Learned pipeline, LP certificate at 1 % (proven within 1 %)", f"{C} (LP certificate 1 %)"),
              ("Learned pipeline without the certificate (confidence fixing alone, RACLearn's rule)", f"{C} (no certificate)"),
              ("Aggregated certified pipeline, one GNN, no probing (ablation)", abl),
              ("Aggregated certified pipeline, one GNN + probing (ablation)", "S4 | acert:s0+x:0.02"),
              (f"Aggregated certified pipeline, no learning ({self_[6:]})", F),
              ("Learning-free pipeline, LP certificate at 1 %", f"{F} (LP certificate 1 %)"),
              ("Learning-free pipeline without the certificate", f"{F} (no certificate)")]
    rows12 = list(dict.fromkeys(rows12))
    rows_nl = [r for r in rows12 if r[1] not in ("S4 | ltf_knn_1", "S4 | cert:s0:0.02", C, f"{C} (LP certificate 1 %)",
                                                 f"{C} (no certificate)", abl, "S4 | acert:s0+x:0.02")]
    TAB = {"12 h, test_fresh": (("tf", 120), rows12), "24 hours (uc24)": (("u24", 40), rows_nl),
           "Congested lines (ratings × 0.6)": (("hard", 60), rows_nl)}
    R = dict(choice=sel, choice_free=self_, tables={})
    L = ["# Round 8: identical units aggregated, Lagrangian probing, certified pipelines — test", "",
         f"Generated by `scripts/agg_report.py` from `results/agg/test/`; scoring as step 2 (`docs/unified_metrics.md`), "
         f"gaps against the tightest bound known per instance. Learned pipeline `{sel}`, learning-free pipeline `{self_}` "
         "(validation choices, `results/agg/select.json`). Every method of an instance ran back to back in one "
         "single-threaded process on one core; speed-ups are paired with that process's full MILP at 0.1 %.", ""]
    for title, ((key, n), rows) in TAB.items():
        insts = [(key, k) for k in range(n)]
        if a.partial:
            insts = [x for x in insts if x in present]
        rows = [r for r in rows if r[1] in REG.methods]
        if insts and rows:
            S2.table(title, insts, rows, R, L)
    shift_rows = [r for r in rows12 if r[1] in ("S4 | full@0.1%", "S4 | full@0.5%", "S4 | ltf_knn_1", "S4 | agg:full@0.5%",
                                                 "S4 | cert:s0:0.02", C, f"{C} (no certificate)", F)]
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
    tf = [("tf", k) for k in range(120)]
    sh = [(sft, k) for sft in U.SHIFTS for k in range(30)]
    if a.partial:
        tf = [x for x in tf if x in present]
        sh = [x for x in sh if x in present]
    P = [(C, "S4 | ltf_knn_1", tf), (F, "S4 | ltf_knn_1", tf), (C, "S4 | cert:s0:0.02", tf), (C, "S4 | full@0.5%", tf),
         (C, abl, tf), (abl, "S4 | cert:s0:0.02", tf), (F, "S4 | cert:lp:0.05", tf), (C, F, tf),
         (f"{C} (no certificate)", "S4 | ltf_knn_1", tf), ("S4 | agg:full@0.5%", "S4 | scr:full@0.5%", tf),
         ("S4 | agg:full@0.5%", "S4 | full@0.5%", tf), ("S4 | agg:full@0.1%", "S4 | full@0.1%", tf),
         (C, "S4 | ltf_knn_1", sh), (C, "S4 | cert:s0:0.02", sh), (C, "S4 | full@0.5%", sh), (F, "S4 | full@0.5%", sh)]
    for key, n in (("u24", 40), ("hard", 60)):
        ins = [(key, k) for k in range(n)]
        if a.partial:
            ins = [x for x in ins if x in present]
        P += [("S4 | agg:full@0.5%", "S4 | scr:full@0.5%", ins), ("S4 | agg:full@0.1%", "S4 | full@0.1%", ins),
              (F, "S4 | cert:lp:0.05", ins), (F, "S4 | full@0.5%", ins)]
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
        p.update(a=na[5:], b=nb[5:])
        R["paired"].append(p)
        sp = f"{p['speed']:.2f}× [{p['speed_ci'][0]:.2f}, {p['speed_ci'][1]:.2f}]" if "speed" in p else "–"
        L.append(f"| {na[5:]} | {nb[5:]} ({p['set']}) | {p['n']} | {p['gap_a']:.3f} | {p['gap_b']:.3f} | {p['diff']:+.3f} "
                 f"[{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}] | {p['a_better']} / {p['b_better']} | {sp} |")
    L.append("")
    L += ["## Anatomy of the certified pipelines", "",
          "| pipeline | set | instances | fixed | exact fixings (proof step) | certified by the LP bound | by the MILP | "
          "time: LP relaxation + inference / reduced MILP / certificate (s, mean) |", "|---|---|---|---|---|---|---|---|"]
    recs_t = records(os.path.join(a.res, "test"))
    for name in sorted({n for r in recs_t for n in r["runs"] if is_cert(n)}):
        for title, keep in (("12 h test_fresh", lambda r: r["bench"] == "uc12" and r["set"] == "tf"),
                            ("shifted 12 h", lambda r: r["bench"] == "uc12" and r["set"] != "tf"),
                            ("24 h", lambda r: r["bench"] == "uc24"), ("congested", lambda r: r["bench"] == "hard")):
            vs = [(r, r["runs"][name]) for r in recs_t if keep(r) and name in r["runs"]]
            if not vs:
                continue
            nT = [12 * 73 if r["bench"] != "uc24" else 24 * 73 for r, _ in vs]
            fx = sum(v["n_fixed"] / n for (r, v), n in zip(vs, nT)) / len(vs)
            mean = lambda f: sum(f(v) for _, v in vs) / len(vs)
            n_lp = sum(v["cert"] in ("lp", "lp2", "round") for _, v in vs)
            milp = [v for _, v in vs if v["cert"] == "milp"]
            ex = f"{sum(v.get('n_exact', 0) for v in milp) / max(len(milp), 1):.0f}" if any("n_exact" in v for v in milp) else "–"
            L.append(f"| {name} | {title} | {len(vs)} | {100 * fx:.1f} % | {ex} | {n_lp} | {len(vs) - n_lp} | "
                     f"{mean(lambda v: v['parts'].get('lp_relaxation', 0) + v['parts'].get('inference', 0)):.2f} / "
                     f"{mean(lambda v: v['red_s']):.2f} / {mean(lambda v: v['cert_s']):.2f} |")
    L.append("")
    L += ["## Checks", ""]
    bad, n_cert = [], 0
    for r in recs_t:
        for name, v in r["runs"].items():
            if not is_cert(name):
                continue
            n_cert += 1
            b = r["c_rel"] if v["cert"] in ("lp", "lp2", "round") else max(v.get("cert_bound", -math.inf), r["c_rel"])
            obj = v["cost"] if v["cert"] == "round" else min(v["red_obj"], v.get("red2_obj", math.inf), v.get("cert_obj", math.inf))
            if (obj - b) / obj > EPS + 1e-9:
                bad.append((r["bench"], r["set"], r["k"], name, v.get("cert_status")))
    L.append(f"* Certificates: {n_cert - len(bad)} of {n_cert} certified-pipeline results proven within 0.5 % by their "
             f"own bound; not proven: {len(bad)} (the proof step hit its time limit): {bad[:8]}")
    lines_ok = [v.get("lines_ok", True) for r in recs_t for n, v in r["runs"].items() if n.startswith(("scr:", "agg:"))]
    L.append(f"* Screened / aggregated full solves ending with every line within its rating: {sum(lines_ok)} of {len(lines_ok)}.")
    same = []
    for nm in ("full@0.1%", "full@0.5%", "cert:s0:0.02", "cert:lp:0.05", "scr:full@0.5%"):
        a3, a4 = f"S3 | {nm}", f"S4 | {nm}"
        if a3 in REG.methods and a4 in REG.methods:
            xs = [x for x in REG.methods[a4]["data"] if x in REG.methods[a3]["data"]]
            eq = sum(abs(REG.methods[a3]["data"][x][0] - REG.methods[a4]["data"][x][0]) <= 1e-3 * REG.methods[a4]["data"][x][0] for x in xs)
            same.append(f"{nm} {eq}/{len(xs)} within 0.1 % of round 7's cost")
    L.append("* Same methods as round 7: " + "; ".join(same) + ".")
    below = [x for n, M in REG.methods.items() if n.startswith("S4") for x, d in M["data"].items()
             if U.db_of(x) is not None and d[0] < U.db_of(x) * (1 - 1e-7)]
    L.append(f"* Delivered costs below the tightest bound: {len(below)}.")
    hosts = sorted({(r.get('cpu_model'), r.get('highs')) for r in recs_t})
    L.append(f"* Hardware: {hosts}.")
    R["checks"] = dict(not_proven=bad, n_cert=n_cert, lines_ok=[sum(lines_ok), len(lines_ok)], below=len(below), hosts=hosts)
    os.makedirs(a.res, exist_ok=True)
    open(os.path.join(a.res, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump(R, open(os.path.join(a.res, "report.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
