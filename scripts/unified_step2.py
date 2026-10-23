"""Step 2 of docs/unified_metrics.md, report: the re-timing on one machine (scripts/unified_retime.py) scored under the
protocol of scripts/unified_rescore.py, with the checks that say whether the re-run is on track.

    python3 scripts/unified_step2.py                      # results/unified/step2.{md,json}
    python3 scripts/unified_step2.py --res DIR --partial  # smoke runs: tables over the instances present

Everything of step 1 is loaded first (so the bound DB stays the tightest over every run, now including step 2's full
MILPs and the long bound solves of results/unified/retime/longmilp_*.jsonl); step 2's records are added under the run
names S2-fix, S2-A, S2-B, S2-B0, S2-hard. Speed-ups are paired within each process (T of that process's own full MILP at
0.1 % / T of the method); 'common ref' divides instead by the median of every process's full-MILP time on the
instance (all processes of an instance ran back to back on the same core).

Checks:
  identity     the LP relaxation cost of every instance agrees with step 1's records (exact up to 1e-6, relative)
  hardware     CPU model and HiGHS build of every step-2 record (one of each expected)
  same models  arm A and arm B use the checkpoints of the GPU round: their schedules' costs should equal the saved
               records'; learning-free rules (LP rounding, LP-integral fixing, Learning to Fix on the LP relaxation)
               should equal step 1's up to the MILP tolerance
  timing noise per instance, the spread (max / min) of the full MILP's time over the processes that re-solved it
  bounds       no delivered cost below the tightest bound; how many instances are still above 0.1 % unresolved
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

REG, TOL = U.REG, U.TOL
RES = "results/unified/retime"


def recs(pattern, key):
    """records of a glob pattern, the last one per key (a re-run of a failed process may append twice)"""
    out = {}
    for p in sorted(glob.glob(pattern)):
        for r in U.jl(p):
            out[key(r)] = r
    return list(out.values())


def inst_key(bench, s, k):
    if bench == "uc24":
        return ("u24", k)
    if bench == "hard":
        return ("hard", k)
    return ("tf", k) if s == "tf" else (s, k)


def family(name):
    if name.startswith("full"):
        return "solver"
    if name.startswith(("lpint", "ltf_lp", "lpround")):
        return "no learning"
    if name.startswith(("e2e", "knn")):
        return "one-shot"
    return "fixing"


# ============================================================================ loaders
def load_fix(res):
    for bench in ("uc12", "uc24"):
        for r in recs(os.path.join(res, f"{bench}_fix_[ts]*.jsonl"), lambda r: (r["bench"], r["set"], r["k"])):
            x = inst_key(r["bench"], r["set"], r["k"])
            src = f"unified/retime/{bench}_fix"
            REG.lp(x, r["c_rel"], src)
            ref = r["runs"].get("full@0.1%")
            tref = ref["time"] if ref else None
            for name, v in r["runs"].items():
                if name.startswith("full") and U.finite(v.get("bound")):
                    REG.bound(x, v["bound"], src)
                if not U.finite(v["cost"]):
                    REG.notes.append(f"S2: {name} at {x} delivered no schedule")
                    continue
                REG.put(f"S2·fix | {name}", x, v["cost"], v["shed"] < TOL and v["short"] < TOL, v["time"], tref,
                        U.meta(src, family(name), "complete", "S2-fix", cpu=r.get("cpu_model"), highs=r.get("highs")),
                        fellback=bool(v.get("fallback")))


def load_fix_addon(res, name):
    """an add-on pass of unified_fix_eval.py (a few methods re-run later, e.g. after a re-tuning), records
    {bench}_{name}_*.jsonl: its methods paired with its own full MILP at 0.1 %; methods the main pass already has for
    the instance are not loaded again"""
    for bench in ("uc12", "uc24"):
        for r in recs(os.path.join(res, f"{bench}_{name}_[ts]*.jsonl"), lambda r: (r["bench"], r["set"], r["k"])):
            x = inst_key(r["bench"], r["set"], r["k"])
            src = f"unified/retime/{bench}_{name}"
            REG.lp(x, r["c_rel"], src)
            ref = r["runs"].get("full@0.1%")
            if ref and U.finite(ref.get("bound")):
                REG.bound(x, ref["bound"], src)
            for m, v in r["runs"].items():
                if m.startswith("full") or x in REG.methods.get(f"S2·fix | {m}", {}).get("data", {}):
                    continue
                if not U.finite(v["cost"]):
                    REG.notes.append(f"S2 add-on {name}: {m} at {x} delivered no schedule")
                    continue
                REG.put(f"S2·fix | {m}", x, v["cost"], v["shed"] < TOL and v["short"] < TOL, v["time"],
                        ref["time"] if ref else None,
                        U.meta(src, family(m), "complete", f"S2-fix:{name}", cpu=r.get("cpu_model")),
                        fellback=bool(v.get("fallback")))


def load_arm_a(res):
    src = "unified/retime/uc12_armA"
    for r in recs(os.path.join(res, "uc12_armA_[ts]*.jsonl"), lambda r: r["i"]):
        x = ("tf", r["i"])
        REG.bound(x, r["milp01"]["bound"], src)
        REG.bound(x, r["milp05"]["bound"], src)
        REG.lp(x, r["relax"]["cost"], src)
        tref = r["milp01"]["time"]
        for k, v in r.items():
            if k == "relax" or not (isinstance(v, dict) and U.finite(v.get("cost")) and "time" in v):
                continue
            REG.put(f"S2·A | {k}", x, v["cost"], (v.get("shed") or 0) + (v.get("short") or 0) < TOL, v["time"], tref,
                    U.meta(src, "arm A" if not k.startswith(("milp", "lpround", "m1x", "rl_")) else "other", "complete", "S2-A"))


def load_arm_b(res, bench, kind, label):
    """rec_rloop_eval records (as scripts/unified_rescore.py load_arm_b_gpu; the 0.5 % full MILP is optional)"""
    src, run = f"unified/retime/{bench}_{kind}", f"S2-{kind}"
    key = {"uc12": "tf", "uc24": "u24", "hard": "hard", "line_out": "line_out"}[bench]
    for r in recs(os.path.join(res, f"{bench}_{kind}_[ts]*.jsonl"), lambda r: r["k"]):
        x = (key, r["k"])
        f1, f5 = r["full"]["0.001"], r["full"].get("0.005")
        for f in (f1, f5):
            if f:
                REG.bound(x, f["bound"], src)
        REG.lp(x, r["c_rel"], src)
        tref = f1["time"]
        mt = lambda fam: U.meta(src, fam, "complete", run)
        REG.put(f"{label} | full MILP 0.1 %", x, f1["obj"], U.slack(f1) < TOL, f1["time"], tref, mt("solver"))
        if f5:
            REG.put(f"{label} | full MILP 0.5 %", x, f5["obj"], U.slack(f5) < TOL, f5["time"], tref, mt("solver"))
        c, s, t = U.best_of(r["lpround"]["c"])
        REG.put(f"{label} | LP rounding + repair, 5 LPs", x, c, s, r["lpround"]["t_pre"] + t, tref, mt("no learning"))
        ls = r["lpls"]["recs"][-1]
        REG.put(f"{label} | LP-checked local search", x, ls["best"], ls["best_shed"] < TOL and ls["best_short"] < TOL,
                r["lpls"]["t_pre"] + ls["time"], tref, mt("no learning"))
        rl, ks = r["rloop"], r["k_sel"]
        y = rl[ks - 1]
        srv = y["best_shed"] < TOL and y["best_short"] < TOL
        REG.put(f"{label} | @{ks} steps", x, y["best"], srv, y["time"], tref, mt("arm B"))
        h = rl[U.halt_step(rl)]
        REG.put(f"{label} | halting head", x, h["best"], h["best_shed"] < TOL and h["best_short"] < TOL, h["time"], tref,
                mt("arm B"))
        cost, t = y["best"], y["time"]
        s = r.get("safety")
        if s:
            t += s["time"] + s.get("t_fix", 0.0)
            if s["feasible"] and U.slack(s) < TOL and (not srv or s["obj"] < cost):
                cost, srv = s["obj"], True
        REG.put(f"{label} | @{ks} + safety net", x, cost, srv, t, tref, mt("arm B"))


def load_hard(res):
    src = "unified/retime/hard_fix"
    R = recs(os.path.join(res, "hard_fix_[ts]*.jsonl"), lambda r: (r["i"], r["rule"], r["mip_rel_gap"]))
    full = {}
    for r in R:
        if r["rule"] == "full MILP":
            x = ("hard", r["i"])
            REG.bound(x, r["bound"] if U.finite(r["bound"]) else r["obj"] * (1 - (r["mip_gap"] or 0)), src)
            full[(x, r["mip_rel_gap"])] = r
    for r in R:
        x = ("hard", r["i"])
        F, fb = full[(x, 0.001)], full.get((x, r["mip_rel_gap"]), full[(x, 0.001)])
        t = r["time"] + r.get("pre_s", 0.0) + r.get("relax_s", 0.0) + r.get("guard_s", 0.0)
        mt = U.meta(src, "solver" if r["rule"] == "full MILP" else "no learning", "complete", "S2-hard")
        nm = f"S2·hard | {r['rule']} @{100 * r['mip_rel_gap']:g} %"
        if r["feasible"]:
            REG.put(nm, x, r["obj"], U.slack(r) < TOL, t, F["time"], mt)
        else:
            REG.put(nm, x, fb["obj"], U.slack(fb) < TOL, t + fb["time"], F["time"], mt, fellback=True)


def load_longmilp(res):
    for r in recs(os.path.join(res, "longmilp_*.jsonl"), lambda r: (r["bench"], r["set"], r["k"])):
        x = inst_key(r["bench"], r["set"], r["k"])
        REG.bound(x, r["bound"], "unified/retime/longmilp")
        if U.finite(r.get("obj")):
            REG.put("S2 | long MILP (bound solve)", x, r["obj"], U.slack(r) < TOL, None, None,
                    U.meta("unified/retime/longmilp", "solver", "none", "S2-long"))


# ============================================================================ tables
ARM_A = [("Arm A: TRM + safety net", "w1_trm30_fb"), ("Arm A: TRM, mixed + repair", "w1_trm30_mix5r"),
         ("Arm A: TRM, 1 LP + repair", "w1_trm30_1r"), ("Arm A control: deep network + safety net", "w1_deep_fb"),
         ("Arm A control: deep network, mixed + repair", "w1_deep_mix5r"),
         ("Arm A control: one-shot + safety net", "w1_oneshot_fb"), ("Arm A control: one-shot, mixed + repair", "w1_oneshot_mix5r"),
         ("Temporal-GNN ensemble, mixed + repair", "m1x_mix5r")]
T12 = ([("Full MILP, 0.1 % gap (reference)", "S2·fix | full@0.1%"), ("Full MILP, 0.25 % gap", "S2·fix | full@0.25%"),
        ("Full MILP, 0.5 % gap", "S2·fix | full@0.5%"), ("Full MILP, 1 % gap", "S2·fix | full@1%"),
        ("Recommended pipeline (0.5 %)", "S2·fix | pipe@0.5%"), ("Pipeline + bound check", "S2·fix | pipe+bc"),
        ("Ensemble fixings + guards, 0.1 %", "S2·fix | pipe@0.1%"),
        ("Learning to Fix, kNN, eps 1 %", "S2·fix | ltf_knn_1"), ("Learning to Fix, kNN, eps 5 %", "S2·fix | ltf_knn_5"),
        ("Learning to Fix, kNN, eps 10 %", "S2·fix | ltf_knn_10"), ("Learning to Fix, kNN + our LP guard", "S2·fix | ltf_knn_1+lpg"),
        ("Learning to Fix on the BCE GNN, eps 1 %", "S2·fix | ltf_bce_1"),
        ("Learning to Fix on the LP relaxation (no learning)", "S2·fix | ltf_lp"),
        ("LP-integral fixing (no learning)", "S2·fix | lpint"), ("LP-integral fixing + guards (no learning)", "S2·fix | lpint+g"),
        ("LP rounding + repair (no learning)", "S2·fix | lpround"),
        ("Arm B, 24 steps + safety net", "S2·B | @24 + safety net"), ("Arm B, 24 steps (no MILP)", "S2·B | @24 steps"),
        ("Arm B, halting head", "S2·B | halting head"), ("Arm B without feedback + safety net", "S2·B0 | @24 + safety net"),
        ("Arm B without feedback, 24 steps", "S2·B0 | @24 steps"), ("LP-checked local search (no learning)", "S2·B | LP-checked local search")]
       + [(lab, f"S2·A | {k}") for lab, k in ARM_A])
EARLY = ([(f"{m} · {d}", f"S2·fix | e2e:{m}:{d}") for m in ("mlp", "gnn", "gnn_sym", "gnn_lp", "gnn_lp_rl")
          for d in ("top1", "adeq", "block", "screen")] + [("kNN-5 schedules", "S2·fix | knn5"), ("kNN-20 schedules", "S2·fix | knn20")])
SHIFT = [("Full MILP, 0.1 %", "S2·fix | full@0.1%"), ("Full MILP, 0.5 %", "S2·fix | full@0.5%"),
         ("Recommended pipeline", "S2·fix | pipe@0.5%"), ("Pipeline + bound check", "S2·fix | pipe+bc"),
         ("Learning to Fix, kNN, eps 1 %", "S2·fix | ltf_knn_1"), ("Learning to Fix, kNN + our LP guard", "S2·fix | ltf_knn_1+lpg"),
         ("Learning to Fix, BCE GNN, eps 1 %", "S2·fix | ltf_bce_1"),
         ("Learning to Fix on the LP relaxation (no learning)", "S2·fix | ltf_lp"),
         ("LP-integral fixing + guards (no learning)", "S2·fix | lpint+g"),
         ("Arm B (line outages only), 24 steps + safety net", "S2·B | @24 + safety net")]
U24 = [("Full MILP, 0.1 % gap (reference)", "S2·fix | full@0.1%"), ("Full MILP, 0.5 % gap", "S2·fix | full@0.5%"),
       ("Full MILP, 1 % gap", "S2·fix | full@1%"), ("Learning to Fix on the LP relaxation, 0.1 %", "S2·fix | ltf_lp"),
       ("Learning to Fix on the LP relaxation, 0.5 %", "S2·fix | ltf_lp@0.5%"), ("LP-integral fixing", "S2·fix | lpint"),
       ("LP-integral fixing + guards, 0.1 %", "S2·fix | lpint+g"), ("LP-integral fixing + guards, 0.5 %", "S2·fix | lpint+g@0.5%"),
       ("LP rounding + repair", "S2·fix | lpround"),
       ("Arm B zero-shot, 24 steps + safety net", "S2·B | @24 + safety net"), ("Arm B zero-shot, 24 steps", "S2·B | @24 steps"),
       ("LP-checked local search (no learning)", "S2·B | LP-checked local search")]
HARD = ([(f"Full MILP, {g} % gap" + (" (reference)" if g == "0.1" else ""), f"S2·hard | full MILP @{g} %") for g in ("0.1", "0.5", "1")]
        + [(f"{lab}, {g} %", f"S2·hard | {r} @{g} %") for r, lab in (("LtF-LP", "Learning to Fix on the LP relaxation"),
                                                                      ("LP-int+g", "LP-integral fixing + guards"),
                                                                      ("LP pipeline", "LP pipeline: LP values + guards"))
           for g in ("0.1", "0.5")]
        + [("Arm B zero-shot, 24 steps + safety net", "S2·B | @24 + safety net"), ("Arm B zero-shot, 24 steps", "S2·B | @24 steps"),
           ("LP rounding + repair (no learning)", "S2·B | LP rounding + repair, 5 LPs")])
TABLES = {"12 h, test_fresh": (("tf", 120), T12), "12 h, test_fresh: early one-shot models (no MILP)": (("tf", 120), EARLY),
          "24 hours (uc24)": (("u24", 40), U24), "Congested lines (ratings × 0.6)": (("hard", 60), HARD)}
# step-1 counterpart of a step-2 method (same method, saved records), for the side-by-side comparison
STEP1 = {"S2·fix | pipe@0.5%": "Y6 | pipeline: B-hg + LP guard, gap 0.5%", "S2·fix | pipe+bc": "Z1 | pipeline + bound check",
         "S2·fix | full@0.5%": "A·gpu | milp05", "S2·fix | ltf_knn_1": "X6 | faithful LtF kNN eps=1%",
         "S2·fix | ltf_lp": "Y1 | LtF on LP relaxation eps=1%", "S2·fix | lpround": "A·gpu | lpround5r",
         "S2·B | @24 + safety net": "B·gpu fb | @24 + safety net", "S2·B | @24 steps": "B·gpu fb | @24 steps",
         "S2·B0 | @24 + safety net": "B·gpu nofb | @24 + safety net", "S2·A | w1_trm30_fb": "A·gpu | w1_trm30_fb",
         "S2·A | w1_deep_fb": "A·gpu | w1_deep_fb", "S2·A | m1x_mix5r": "A·gpu | m1x_mix5r"}


def ref_times(insts):
    """per instance, every process's full-MILP (0.1 %) time"""
    names = [n for n in ("S2·fix | full@0.1%", "S2·A | milp01", "S2·B | full MILP 0.1 %", "S2·B0 | full MILP 0.1 %",
                         "S2·hard | full MILP @0.1 %") if n in REG.methods]
    out = {}
    for x in insts:
        ts = [REG.methods[n]["data"][x][2] for n in names if x in REG.methods[n]["data"] and REG.methods[n]["data"][x][2]]
        if ts:
            out[x] = ts
    return out


def geo(v):
    return math.exp(sum(math.log(z) for z in v) / len(v)) if v else None


def table(title, insts, rows, R, L):
    have = [(lab, n) for lab, n in rows if n in REG.methods and U.complete_on(n, insts)]
    miss = [n for lab, n in rows if not (n in REG.methods and U.complete_on(n, insts))]
    if not have:
        L += [f"## {title}", "", "No complete method yet.", ""]
        return
    both, cs = U.common_served_gap([n for _, n in have], insts)
    rt = ref_times(insts)
    T = []
    for lab, n in have:
        s = U.stats(n, insts)
        d = REG.methods[n]["data"]
        cr = [sorted(rt[x])[len(rt[x]) // 2] / d[x][2] for x in insts if x in rt and d[x][2]]
        s.update(label=lab, method=n, gap_common_served=cs[n], sp_common=geo(cr) if len(cr) == len(insts) else None)
        T.append(s)
    T.sort(key=lambda s: s["gap_mean"])
    R["tables"][title] = dict(n=len(insts), n_common_served=len(both), rows=T, missing=miss)
    L += [f"## {title}", "", f"{len(insts)} instances; 'common': gap on the {len(both)} every method here serves.", "",
          "| method | gap mean [95 % CI] | median | p95 | max | > 1 % | common | served | fallback | time s | speed-up geo / median / mean | common ref |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in T:
        ci = s["gap_ci"] or [None, None]
        sp = f"{s['sp_geo']:.1f}× / {s['sp_median']:.1f}× / {s['sp_mean']:.1f}×" if "sp_geo" in s else "–"
        L.append(f"| {s['label']} | {s['gap_mean']:.3f} [{U.fmt(ci[0], 3)}, {U.fmt(ci[1], 3)}] | {s['gap_median']:.3f} | "
                 f"{s['gap_p95']:.2f} | {s['gap_max']:.2f} | {s['share_gt1']:.0f} % | {U.fmt(s['gap_common_served'], 3)} | "
                 f"{s['served']:.1f} % | {s['fallback']:.0f} % | {U.fmt(s.get('time_mean'))} | {sp} | "
                 f"{U.fmt(s['sp_common'], 1) + '×' if s['sp_common'] else '–'} |")
    if miss:
        L += ["", "Not complete: " + ", ".join(miss)]
    L.append("")


# ============================================================================ checks
def checks(insts_all):
    out = {}
    # identity: step-2 LP relaxation costs against step 1's
    dis, n = [], 0
    for x, vals in REG.crel.items():
        s2 = [v for v, s in vals if s.startswith("unified/retime")]
        s1 = [v for v, s in vals if not s.startswith("unified/retime")]
        if s2 and s1:
            n += 1
            ref = s1[0]
            if any(abs(v - ref) > 1e-6 * abs(ref) for v in s2):
                dis.append((x, ref, s2))
    out["identity"] = dict(instances=n, disagreements=len(dis), examples=[str(d) for d in dis[:5]])
    # hardware and solver of the step-2 records
    hw = {}
    for p in glob.glob(os.path.join(ARGS.res, "uc12_fix_[ts]*.jsonl")) + glob.glob(os.path.join(ARGS.res, "uc24_fix_[ts]*.jsonl")):
        for r in U.jl(p):                                   # (the congested script's records carry no CPU field;
                                                            #  the manifest covers every process)
            hw[(r.get("cpu_model"), r.get("highs"))] = hw.get((r.get("cpu_model"), r.get("highs")), 0) + 1
    man = {}
    for p in glob.glob(os.path.join(ARGS.res, "manifest_[ts]*.jsonl")):
        for r in U.jl(p):
            man[r.get("cpu_model")] = man.get(r.get("cpu_model"), 0) + 1
            if r["rc"] != 0:
                out.setdefault("failed_processes", []).append(f"{r['bench']}:{r['set']}:{r['k']} {r['kind']} rc {r['rc']}")
    out["hardware"] = dict(fix_records={f"{a} / HiGHS {b}": c for (a, b), c in hw.items()}, manifest_cpu=man)
    # same models, same schedules
    same = {}
    for s2, s1 in STEP1.items():
        if s2 in REG.methods and s1 in REG.methods:
            a, b = REG.methods[s2]["data"], REG.methods[s1]["data"]
            xs = [x for x in a if x in b]
            if xs:
                same[s2] = dict(vs=s1, n=len(xs), identical=sum(abs(a[x][0] - b[x][0]) <= 1e-6 * abs(b[x][0]) for x in xs),
                                within_0_1pct=sum(abs(a[x][0] - b[x][0]) <= 1e-3 * abs(b[x][0]) for x in xs))
    out["same_models"] = same
    # timing noise of the repeated full-MILP reference
    rt = ref_times(insts_all)
    ratios = sorted(max(v) / min(v) for v in rt.values() if len(v) >= 2)
    if ratios:
        out["reference_spread"] = dict(instances=len(ratios), median=U.pct(ratios, 0.5), p90=U.pct(ratios, 0.9),
                                       max=ratios[-1])
    # bounds
    below = []
    for name, M in REG.methods.items():
        if not name.startswith("S2"):
            continue
        for x, (c, *_r) in M["data"].items():
            db = U.db_of(x)
            if db is not None and c < db * (1 - 1e-7):
                below.append((name, x, c, db))
    out["cost_below_bound"] = dict(n=len(below), examples=[str(b) for b in below[:5]])
    unres = {}
    for x in insts_all:
        best = min((M["data"][x][0] for M in REG.methods.values() if x in M["data"]), default=None)
        db = U.db_of(x)
        if best is not None and db is not None:
            unres[x] = (best - db) / best * 100
    out["unresolved_gt_0_1pct"] = sorted(([str(x), round(g, 3)] for x, g in unres.items() if g > 0.1), key=lambda z: -z[1])
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--partial", action="store_true", help="tables over the instances present (smoke runs)")
    ap.add_argument("--out", default="results/unified/step2")
    ARGS = a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    U.load_all()
    load_fix(a.res)
    load_fix_addon(a.res, "fixknn10")
    load_arm_a(a.res)
    load_arm_b(a.res, "uc12", "armB", "S2·B")
    load_arm_b(a.res, "uc12", "armB0", "S2·B0")
    for s in (1, 2, 3):
        load_arm_b(a.res, "uc12", f"armBs{s}", f"S2·Bs{s}")
    for b in ("uc24", "hard", "line_out"):
        load_arm_b(a.res, b, "armB", "S2·B")
    load_hard(a.res)
    load_longmilp(a.res)
    R = dict(tables={}, notes=[n for n in REG.notes if n.startswith("S2")])
    L = ["# Step 2: every method re-timed on one machine", "",
         "Generated by `scripts/unified_step2.py` from `results/unified/retime/`; protocol as in `docs/unified_metrics.md`. "
         "Speed-ups are paired within each process; 'common ref' uses the median full-MILP time of all processes on the "
         "instance.", ""]
    present = {x for n, M in REG.methods.items() if n.startswith("S2") for x in M["data"]}
    all_insts = []
    for title, ((key, n), rows) in TABLES.items():
        insts = [(key, k) for k in range(n)]
        if a.partial:
            insts = [x for x in insts if x in present and all(x in REG.methods[m]["data"] for _, m in rows if m in REG.methods)] \
                or [x for x in insts if x in present]
        all_insts += insts
        if insts:
            table(title, insts, rows, R, L)
    L += ["## Shifted 12-hour sets: mean gap · served · speed-up (geometric mean)", "",
          "| method | " + " | ".join(U.SHIFTS) + " |", "|---|" + "---|" * len(U.SHIFTS)]
    for lab, n in SHIFT:
        cells = []
        for sft in U.SHIFTS:
            insts = [(sft, k) for k in range(30)]
            if a.partial:
                insts = [x for x in insts if n in REG.methods and x in REG.methods[n]["data"]]
            if insts and n in REG.methods and U.complete_on(n, insts):
                s = U.stats(n, insts)
                R["tables"].setdefault("shifted", {}).setdefault(lab, {})[sft] = s
                cells.append(f"{s['gap_mean']:.2f} % · {s['served']:.0f} % · " + (f"{s['sp_geo']:.1f}×" if "sp_geo" in s else "–"))
            else:
                cells.append("–")
        all_insts += [(sft, k) for sft in U.SHIFTS for k in range(30)]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    # training seeds: the pipeline's three ensembles (5 networks each, own thresholds) and arm B's four training runs
    tf = [("tf", k) for k in range(120)]
    if a.partial:
        tf = [x for x in tf if x in present]
    SEEDS = [("Recommended pipeline (0.5 %)", ["S2·fix | pipe@0.5%", "S2·fix | pipe@0.5%:b", "S2·fix | pipe@0.5%:c"]),
             ("Pipeline + bound check", ["S2·fix | pipe+bc", "S2·fix | pipe+bc:b", "S2·fix | pipe+bc:c"]),
             ("Arm B, 24 steps + safety net", ["S2·B | @24 + safety net"] + [f"S2·Bs{s} | @24 + safety net" for s in (1, 2, 3)]),
             ("Arm B, 24 steps (no MILP)", ["S2·B | @24 steps"] + [f"S2·Bs{s} | @24 steps" for s in (1, 2, 3)])]
    L += ["## Training seeds (12 h, test_fresh): mean gap · served · speed-up (geometric mean) per training run", "",
          "| method | run 1 | run 2 | run 3 | run 4 | mean gap | spread (max − min) |", "|---|---|---|---|---|---|---|"]
    R["seeds"] = {}
    for lab, names in SEEDS:
        st = [U.stats(n, tf) if n in REG.methods and tf and U.complete_on(n, tf) else None for n in names]
        cells = [f"{s['gap_mean']:.3f} % · {s['served']:.0f} % · " + (f"{s['sp_geo']:.1f}×" if "sp_geo" in s else "–")
                 if s else "–" for s in st] + ["–"] * (4 - len(st))
        g = [s["gap_mean"] for s in st if s]
        R["seeds"][lab] = dict(runs=names, stats=st)
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {U.fmt(sum(g) / len(g) if g else None, 3)} | "
                 f"{U.fmt(max(g) - min(g) if len(g) > 1 else None, 3)} pp |")
    L += ["", "Pipeline runs: ensembles of the networks with seeds 0-4 (the one the original study tuned on), 5-9 and "
          "10-14, each with Learning to Fix's thresholds tuned on the same validation set. Arm B runs: w2_rloop_rl (seed 0, "
          "the GPU round's tested model) and g3_plain_s1-3_rl (seeds 1-3, the same configuration, trained for the "
          "attention-residual comparison); settings of the GPU round's test plan for all four.", ""]
    C = checks(sorted(set(x for x in all_insts if x in present)))
    R["checks"] = C
    L += ["## Checks", ""]
    L.append(f"* Identity: LP relaxation cost of {C['identity']['instances']} instances against step 1's records: "
             f"{C['identity']['disagreements']} disagreements.")
    L.append(f"* Hardware: {C['hardware']}")
    if C.get("failed_processes"):
        L.append(f"* Failed processes: {len(C['failed_processes'])}: {C['failed_processes'][:10]}")
    for s2, v in C["same_models"].items():
        L.append(f"* Same method as step 1, {s2} vs {v['vs']}: {v['identical']}/{v['n']} identical costs, "
                 f"{v['within_0_1pct']}/{v['n']} within 0.1 %.")
    if "reference_spread" in C:
        r = C["reference_spread"]
        L.append(f"* Full-MILP reference re-solved by several processes on the same core: time max / min per instance "
                 f"median {r['median']:.2f}, p90 {r['p90']:.2f}, max {r['max']:.2f} ({r['instances']} instances).")
    L.append(f"* Delivered costs below the tightest bound: {C['cost_below_bound']['n']}.")
    L.append(f"* Instances still more than 0.1 % unresolved (best known vs tightest bound): {len(C['unresolved_gt_0_1pct'])}.")
    L.append("")
    if R["notes"]:
        L += ["## Notes", ""] + [f"* {n}" for n in R["notes"][:50]] + [""]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    json.dump(R, open(a.out + ".json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
