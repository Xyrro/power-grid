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

Also: paired comparisons on 12 h (mean gap difference and ratio of paired speed-ups, 95 % bootstrap intervals over
instances), arm A's every pipeline, and the step-1 methods that step 2 could not re-run, re-scored against the bounds
known now (their speed-ups stay their own study's, measured on another machine).
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
# methods whose thresholds were tuned on this cluster: the first tuning pass oversubscribed its cores (19 processes on 16)
# and Learning to Fix's tuning uses time-limited MILPs, so every tuning was repeated with one core each (tune2) and these
# methods re-run in two add-on passes with those thresholds, "fixtune2" (kNN at eps 5 / 10 %, the pipelines) and
# "fixtune3" (kNN at eps 1 % with and without the LP guard, BCE at eps 1 %: their tunings took longest, so they were
# re-run last); where these passes exist, their records are the methods' rows and the first pass is kept as
# "S2·fix(contended)" for the sensitivity table
TUNED = ("ltf_knn_1", "ltf_knn_5", "ltf_knn_10", "ltf_knn_1+lpg", "ltf_bce_1", "pipe@0.1%", "pipe@0.5%", "pipe+bc",
         "pipe@0.5%:b", "pipe+bc:b", "pipe@0.5%:c", "pipe+bc:c")


def load_fix(res):
    retuned = bool(glob.glob(os.path.join(res, "uc12_fixtune[23]_*.jsonl")))
    for bench in ("uc12", "uc24"):
        for r in recs(os.path.join(res, f"{bench}_fix_*.jsonl"), lambda r: (r["bench"], r["set"], r["k"])):
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
                lab = "S2·fix(contended)" if retuned and name in TUNED else "S2·fix"
                REG.put(f"{lab} | {name}", x, v["cost"], v["shed"] < TOL and v["short"] < TOL, v["time"], tref,
                        U.meta(src, family(name), "complete", "S2-fix", cpu=r.get("cpu_model"), highs=r.get("highs")),
                        fellback=bool(v.get("fallback")))


def load_fix_addon(res, name):
    """an add-on pass of unified_fix_eval.py (a few methods re-run later, e.g. after a re-tuning), records
    {bench}_{name}_*.jsonl: its methods paired with its own full MILP at 0.1 %; methods the main pass already has for
    the instance are not loaded again"""
    for bench in ("uc12", "uc24"):
        for r in recs(os.path.join(res, f"{bench}_{name}_*.jsonl"), lambda r: (r["bench"], r["set"], r["k"])):
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
    for r in recs(os.path.join(res, "uc12_armA_*.jsonl"), lambda r: r["i"]):
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
    for r in recs(os.path.join(res, f"{bench}_{kind}_*.jsonl"), lambda r: r["k"]):
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
    R = recs(os.path.join(res, "hard_fix_*.jsonl"), lambda r: (r["i"], r["rule"], r["mip_rel_gap"]))
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
    for p in sorted(glob.glob(os.path.join(res, "longmilp_*.jsonl"))):          # every solve's bound counts (all passes)
        for r in U.jl(p):
            REG.bound(inst_key(r["bench"], r["set"], r["k"]), r["bound"], "unified/retime/longmilp")
    for p in sorted(glob.glob(os.path.join(res, "bounds_progress_*.jsonl"))):   # bounds proven part-way through a solve
        for line in open(p):                                                    # (kept if the solve was killed; a line
            try:                                                                #  cut off by the kill is skipped)
                r = json.loads(line)
            except ValueError:
                continue
            REG.bound(inst_key(r["bench"], r["set"], r["k"]), r["bound"], "unified/retime/longmilp (progress)")
    best = {}
    for p in sorted(glob.glob(os.path.join(res, "longmilp_*.jsonl"))):
        for r in U.jl(p):
            key = (r["bench"], r["set"], r["k"])
            if U.finite(r.get("obj")) and (key not in best or r["obj"] < best[key]["obj"]):
                best[key] = r
    for r in best.values():
        REG.put("S2 | long MILP (bound solve)", inst_key(r["bench"], r["set"], r["k"]), r["obj"], U.slack(r) < TOL, None,
                None, U.meta("unified/retime/longmilp", "solver", "none", "S2-long"))


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


# paired comparisons on 12 h test_fresh: (label A, method A, label B, method B, only where both serve)
PAIRED = [("Recommended pipeline", "S2·fix | pipe@0.5%", "full MILP, 0.5 %", "S2·fix | full@0.5%", False),
          ("Pipeline + bound check", "S2·fix | pipe+bc", "full MILP, 0.5 %", "S2·fix | full@0.5%", False),
          ("Recommended pipeline", "S2·fix | pipe@0.5%", "full MILP, 1 %", "S2·fix | full@1%", False),
          ("Recommended pipeline", "S2·fix | pipe@0.5%", "full MILP, 0.1 %", "S2·fix | full@0.1%", False),
          ("Recommended pipeline", "S2·fix | pipe@0.5%", "Learning to Fix, kNN, eps 1 %", "S2·fix | ltf_knn_1", False),
          ("Recommended pipeline", "S2·fix | pipe@0.5%", "Learning to Fix on the BCE GNN", "S2·fix | ltf_bce_1", False),
          ("Recommended pipeline", "S2·fix | pipe@0.5%", "Learning to Fix on the LP relaxation", "S2·fix | ltf_lp", False),
          ("Recommended pipeline", "S2·fix | pipe@0.5%", "arm B + safety net", "S2·B | @24 + safety net", False),
          ("Arm B + safety net", "S2·B | @24 + safety net", "full MILP, 0.5 %", "S2·fix | full@0.5%", False),
          ("Arm B + safety net", "S2·B | @24 + safety net", "arm B without feedback + safety net", "S2·B0 | @24 + safety net", False),
          ("Arm B, 24 steps", "S2·B | @24 steps", "arm B without feedback, 24 steps", "S2·B0 | @24 steps", False),
          ("Arm B, 24 steps", "S2·B | @24 steps", "arm B without feedback, 24 steps", "S2·B0 | @24 steps", True),
          ("Arm B, 24 steps", "S2·B | @24 steps", "LP rounding + repair", "S2·fix | lpround", False),
          ("Arm A: TRM + safety net", "S2·A | w1_trm30_fb", "one-shot control + safety net", "S2·A | w1_oneshot_fb", False),
          ("Arm A: TRM + safety net", "S2·A | w1_trm30_fb", "deep control + safety net", "S2·A | w1_deep_fb", False),
          ("Arm A: TRM, 1 LP + repair", "S2·A | w1_trm30_1r", "one-shot control, 1 LP + repair", "S2·A | w1_oneshot_1r", False),
          ("Arm A: TRM, 5 thresholds + repair", "S2·A | w1_trm30_t5r", "one-shot control, 5 thresholds + repair",
           "S2·A | w1_oneshot_t5r", False),
          ("Arm A: TRM, mixed + repair", "S2·A | w1_trm30_mix5r", "one-shot control, mixed + repair", "S2·A | w1_oneshot_mix5r", False)]
# arm A: every model under every decoding of the GPU round's test plan
ARM_A_GRID = [("Recursive TRM, 16 steps", "w1_trm30", ("mix5r", "1r", "t5r", "fb")),
              ("One-shot control, same network", "w1_oneshot", ("mix5r", "1r", "t5r", "fb")),
              ("Deep unshared control, 13.1 M", "w1_deep", ("mix5r", "1r", "t5r", "fb")),
              ("Temporal-GNN ensemble (m1x)", "m1x", ("mix5r", "1r", "5r", None))]
# step-1 methods step 2 could not re-run (models or training data not on the cluster), against today's bounds
NOT_RERUN = [("12 h, test_fresh", ("tf", 120), [("Hybrid, 0.1 % gap", "Y6 | hybrid"),
                                                 ("Recommended pipeline, original models (same study, for comparison)",
                                                  "Y6 | pipeline: B-hg + LP guard, gap 0.5%")]),
             ("12 h, first 60 fresh instances", ("tf", 60),
              [("Hybrid", "X6 | hybrid he_bce_s0_e1_n360"),
               ("Ours: error-cost + adequacy guard, 90 %", "X6 | ours: error-cost + adequacy guard 90% (BCE)"),
               ("Ours: combined pipeline, 98 %", "X6 | ours: combined st+error-cost+adequacy+LP guard 98%")]),
             ("24 hours", ("u24", 40),
              [("Learned recipe with a cross-validated margin", "Z2fu | m24cal"),
               ("Learned recipe: ensemble fixings + LP guard, 0.5 % (tuned thresholds)", "Z2fu | m24"),
               ("Guarded 95 % rule (imitation GNN)", "Z2 | guarded 95%"),
               ("Learning to Fix, kNN, eps 1 %", "X8 | LtF knn eps=1%")]),
             ("Congested lines", ("hard", 60),
              [("Learned pipeline, 0.5 %", "Z3 | pipeline @0.5 %"), ("Learned pipeline, 0.1 %", "Z3 | pipeline @0.1 %"),
               ("Learning to Fix, kNN, eps 1 %", "Z3 | LtF-kNN @0.1 %")])]


def paired(na, nb, insts, served_only):
    """mean of the per-instance gap difference A − B, and the geometric mean of A's paired speed-up over B's"""
    da, dd = REG.methods[na]["data"], REG.methods[nb]["data"]
    xs = [x for x in insts if not served_only or (da[x][1] and dd[x][1])]
    g = lambda d, x: (d[x][0] - U.db_of(x)) / d[x][0] * 100
    dg = [g(da, x) - g(dd, x) for x in xs]
    out = dict(a=na, b=nb, served_only=served_only, n=len(xs), gap_a=sum(g(da, x) for x in xs) / len(xs),
               gap_b=sum(g(dd, x) for x in xs) / len(xs), diff=sum(dg) / len(dg), diff_ci=U.boot_ci(dg, na + nb + str(len(xs))),
               a_better=sum(v < -1e-9 for v in dg), b_better=sum(v > 1e-9 for v in dg))
    if all(da[x][2] and da[x][3] and dd[x][2] and dd[x][3] for x in xs):
        lr = [math.log((da[x][3] / da[x][2]) / (dd[x][3] / dd[x][2])) for x in xs]
        ci = U.boot_ci(lr, "sp" + na + nb + str(len(xs)))
        out.update(speed=math.exp(sum(lr) / len(lr)), speed_ci=[math.exp(ci[0]), math.exp(ci[1])])
    return out


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
    for p in glob.glob(os.path.join(ARGS.res, "uc12_fix_*.jsonl")) + glob.glob(os.path.join(ARGS.res, "uc24_fix_*.jsonl")):
        for r in U.jl(p):                                   # (the congested script's records carry no CPU field;
                                                            #  the manifest covers every process)
            hw[(r.get("cpu_model"), r.get("highs"))] = hw.get((r.get("cpu_model"), r.get("highs")), 0) + 1
    man = {}
    for p in glob.glob(os.path.join(ARGS.res, "manifest_*.jsonl")):
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
    load_fix_addon(a.res, "fixtune2")
    load_fix_addon(a.res, "fixtune3")
    load_fix_addon(a.res, "fixorig")
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
    # thresholds: tuned with one core each (used above), tuned under CPU contention (first pass), the original study's
    L += ["## Learning to Fix's thresholds: three tuning runs of the same procedure (12 h, test_fresh)", "",
          "Test: mean gap · served · speed-up (geometric mean); in brackets the share of decisions the thresholds fix on "
          "the validation set they were tuned on.", "",
          "| method | one core per tuning (used) | first pass, oversubscribed cores | original study's run |",
          "|---|---|---|---|"]
    R["tuning_sensitivity"] = {}
    tfile = {"ltf_knn_1": "ltfx_tune_knn_1.json", "ltf_knn_5": "ltfx_tune_knn_5.json", "ltf_knn_10": "ltfx_tune_knn_10.json",
             "ltf_bce_1": "ltfx_tune_bce_1.json", "pipe@0.5%": ("m1x_tune_hgP.json", "m1x_tune_hg_pol_n2000_gnnt_ens5.json")}

    def vfix(folder, m):
        f = tfile[m] if isinstance(tfile[m], str) else tfile[m][folder == "results/uc12"]
        f = os.path.join(folder, f)
        nc = f.replace(".json", ".nonconverged.json")       # the first pass's own eps-10 % run (set aside, not converged)
        if os.path.exists(nc) and os.path.basename(folder) == "tune":
            f = nc
        if not os.path.exists(f):
            return None
        j = json.load(open(f))
        return j["val_fixed_share"] * 100, j["converged"]
    for m in ("ltf_knn_1", "ltf_knn_5", "ltf_knn_10", "ltf_bce_1", "pipe@0.5%"):
        cells = []
        for n, folder in ((f"S2·fix | {m}", os.path.join(a.res, "tune2")), (f"S2·fix(contended) | {m}", os.path.join(a.res, "tune")),
                          (f"S2·fix | {m}:orig", "results/uc12")):
            ok = n in REG.methods and tf and U.complete_on(n, tf)
            s = U.stats(n, tf) if ok else None
            v = vfix(folder, m)
            R["tuning_sensitivity"].setdefault(m, []).append(dict(test=s, val_fixed=v))
            vv = f" [{v[0]:.1f} %{'' if v[1] else ', not converged'}]" if v else ""
            cells.append((f"{s['gap_mean']:.3f} % · {s['served']:.0f} % · " + (f"{s['sp_geo']:.1f}×" if "sp_geo" in s else "–")
                          if s else "–") + vv)
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "Same models in all three columns; only the thresholds differ. Learning to Fix's tuning solves time-limited "
          "MILPs (relaxations 6 s, checks 60 s, master 120 s), so repeated runs on the same validation set reach different "
          "thresholds; the pipeline's guard-aware tuning is stable.", ""]
    # paired comparisons (12 h)
    L += ["## Paired comparisons (12 h, test_fresh)", "",
          "Per instance: A's gap minus B's (negative: A is closer to the bound), and A's paired speed-up over B's "
          "(geometric mean; above 1: A is faster). 95 % bootstrap intervals over instances.", "",
          "| A | B | instances | gap A | gap B | A − B, pp [95 % CI] | A better / B better | speed-up A over B [95 % CI] |",
          "|---|---|---|---|---|---|---|---|"]
    R["paired"] = []
    for la, na, lb, nb, so in PAIRED:
        if not (na in REG.methods and nb in REG.methods and tf and U.complete_on(na, tf) and U.complete_on(nb, tf)):
            continue
        p = paired(na, nb, tf, so)
        p.update(label_a=la, label_b=lb)
        R["paired"].append(p)
        sp = f"{p['speed']:.2f}× [{p['speed_ci'][0]:.2f}, {p['speed_ci'][1]:.2f}]" if "speed" in p else "–"
        L.append(f"| {la} | {lb}{' (where both serve)' if so else ''} | {p['n']} | {p['gap_a']:.3f} | {p['gap_b']:.3f} | "
                 f"{p['diff']:+.3f} [{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}] | {p['a_better']} / {p['b_better']} | {sp} |")
    L.append("")
    # arm A: every decoding
    L += ["## Arm A, every decoding (12 h, test_fresh): mean gap · speed-up (geometric mean)", "",
          "| model | mixed + repair | 1 LP + repair | 5 thresholds + repair | + safety net |", "|---|---|---|---|---|"]
    R["arm_a_grid"] = {}
    for lab, pre, decs in ARM_A_GRID:
        cells = []
        for dcd in decs:
            n = f"S2·A | {pre}_{dcd}" if dcd else None
            if n and n in REG.methods and tf and U.complete_on(n, tf):
                st = U.stats(n, tf)
                R["arm_a_grid"].setdefault(lab, {})[dcd] = st
                cells.append(f"{st['gap_mean']:.3f} % · " + (f"{st['sp_geo']:.1f}×" if "sp_geo" in st else "–"))
            else:
                cells.append("–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    # step-1 methods not re-run, against the bounds known now
    L += ["## Not re-run in step 2: step 1's records against today's bounds", "",
          "Gaps use the tightest bound known now (including step 2's solves); speed-ups are each study's own paired "
          "measurement on its own machine, so they compare only loosely with the tables above.", "",
          "| benchmark | method | run | gap mean [95 % CI] | served | speed-up (geometric mean, own study) |", "|---|---|---|---|---|---|"]
    R["not_rerun"] = []
    for bench, (key, n_i), rows in NOT_RERUN:
        insts = [(key, k) for k in range(n_i)]
        for lab, n in rows:
            if n not in REG.methods or not U.complete_on(n, insts):
                continue
            st = U.stats(n, insts)
            st.update(benchmark=bench, label=lab, method=n)
            R["not_rerun"].append(st)
            ci = st["gap_ci"] or [None, None]
            L.append(f"| {bench} | {lab} | {st['run']} | {st['gap_mean']:.3f} [{U.fmt(ci[0], 3)}, {U.fmt(ci[1], 3)}] | "
                     f"{st['served']:.1f} % | " + (f"{st['sp_geo']:.1f}×" if "sp_geo" in st else "–") + " |")
    L += ["", "PGLib-UC California was not re-run either; no step-2 solve touched its bounds, so its numbers in "
          "`results/unified/rescored.md` stand.", ""]
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
