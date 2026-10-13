"""Cost-aware check study, steps 2-4: validation pass, timing check, test pass (one core, paired timing, resumable).

    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_run.py --stage val      # val + val_extra + val_extra2 (360)
    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_run.py --stage select   # thresholds from validation only
    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_run.py --stage retime   # >= 10 pipeline instances re-solved
    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_run.py --stage test     # test_fresh 0-119 + 6 shifted sets

Pipeline (scripts/uc_m1x_pipeline.py): B-hg fixings -> adequacy + rows + LP-relaxation guard -> reduced MILP, 0.5 %.
Candidate checks (fixed before any run; thresholds from the validation pass only):
  C1 pre-solve LP-cost check   Delta = (z_RR - z_LP) / z_LP of the relaxed reduced LP vs the full LP relaxation;
                               if Delta > d_pre: release fixings by their first-order saving in the relaxed reduced LP
                               (otsl.cguard.release_by_saving) until Delta <= d_pre; then the reduced MILP at 0.5 %
  C2 post-solve bound check    g = (C_red - z_LP) / C_red of the reduced MILP's solution against the LP relaxation; if
     + gradient polish         g > d_post: dispatch-LP gradient at the solution, release the 60 fixings with the largest
                               predicted saving, re-solve from the incumbent at 0.5 % (otsl.cguard.gradient_polish);
                               keep the better solution
  C3 post-solve bound check    if g > d_post: the full MILP at 0.5 % (cold), keep the better solution
     + full-MILP fallback
z_LP is the LP relaxation already solved as model input; the bound check itself costs nothing.

val:    per instance the pipeline's reduced MILP (0.5 %), Delta, g; if g > 1 % (lowest grid value) also C2's polish and
        C3's full MILP at 0.5 %; gaps against the dataset MILP's dual bound. results/uc12/cguard_val.jsonl
select: d_pre, d_post = the smallest grid value that fires on at most 5 % of the 360 validation instances.
        results/uc12/cguard_select.json
retime: the reference full MILP (0.1 %) and the pipeline's reduced MILP of 14 test instances (2 per set) re-solved here;
        the pipeline records (results/uc12/m1x_pipeline_eval_s*.jsonl) are reused only if the times agree within 10 %
        (median and ratio of means). results/uc12/cguard_retime.json
test:   per instance: LP relaxation and fixings rebuilt (fixed share checked against the record), Delta, g from the
        record's cost; C1 solved where it fires; C2 solved where g > d_post (the pipeline's reduced MILP is re-solved
        once to obtain its solution vector, its time is not billed twice: the record's is used); C3 from the records
        (pipeline + full MILP at 0.5 %). results/uc12/cguard_test.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uc_cguard_common import (BHG, R_FULL, R_FULL5, R_PIPE, RES, ROOT, data_instance, fixed_share, load,  # noqa: E402
                              pipeline_records, shifted_instance, slack, test_jobs, timed)
from otsl.cguard import (bound_gap, gradient_polish, lp_increase, relaxed_reduced_lp, release_by_saving,  # noqa: E402
                         solve_with_start)
from otsl.ood import OODModel, solve_milp  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402

GAP5, TL, N_POLISH = 5e-3, 60.0, 60
GRID_PRE = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
GRID_POST = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)
MAX_RATE = 0.05


def rec_of(r):
    return {k: v for k, v in r.items() if k not in ("x", "u")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "select", "retime", "test", "testB", "testfull"], required=True)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.stage == "select":
        V = [json.loads(l) for l in open(os.path.join(RES, "cguard_val.jsonl"))]
        n = len(V)
        rate = lambda key, d: float(np.mean([v[key] > d for v in V]))
        d_pre = next((d for d in GRID_PRE if rate("delta", d) <= MAX_RATE), GRID_PRE[-1])
        d_post = next((d for d in GRID_POST if rate("g", d) <= MAX_RATE), GRID_POST[-1])
        out = dict(n=n, d_pre=d_pre, d_post=d_post, max_rate=MAX_RATE,
                   rate_pre={str(d): rate("delta", d) for d in GRID_PRE}, rate_post={str(d): rate("g", d) for d in GRID_POST},
                   rule="smallest grid value that fires on at most 5 % of the validation instances (val + val_extra + val_extra2)")
        json.dump(out, open(os.path.join(RES, "cguard_select.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    G = sysm.G
    bhg = BHG(sysm)
    if a.stage == "val":
        out = os.path.join(RES, "cguard_val.jsonl")
        done = {(r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
        V = {p: load(os.path.join(ROOT, f"{p}.npz")) for p in ("val", "val_extra", "val_extra2")}
        jobs = [(p, k) for p in V for k in range(len(V[p]["load"])) if (p, k) not in done]
        if jobs:
            bhg.warm(m, data_instance(V[jobs[0][0]], jobs[0][1]))
        print(f"{len(jobs)} validation instances", flush=True)
        for n, (p, k) in enumerate(jobs):
            d = V[p]
            sc = data_instance(d, k)
            db = float(d["obj"][k] * (1 - d["gap"][k]))
            rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
            _, fix, info = bhg.run(m, sc, rel)
            rr = relaxed_reduced_lp(m, sc, fix)
            red = solve_with_start(m, sc, fix, TL, GAP5)
            g = bound_gap(red["obj"], rel.obj) if red["feasible"] else np.inf
            rec = dict(set=p, k=k, db=db, obj_ref=float(d["obj"][k]), z_lp=float(rel.obj), relax_s=relax_s,
                       pre_s=info["pre_s"], guard_s=info["guard_s"], rr_s=rr["seconds"], fixed=fixed_share(fix, (), 12, G),
                       delta=float(lp_increase(rr["cost"], rel.obj)), g=float(g), pipe=rec_of(red))
            if g > GRID_POST[0] and red["feasible"]:
                pol, pinfo = gradient_polish(m, sc, fix, red["x"], red["u"], N_POLISH, TL, GAP5)
                full5 = solve_milp(m, sc, {}, TL, GAP5)
                full5.pop("u", None)
                rec.update(polish=rec_of(pol), polish_info=pinfo, full5=full5)
            with open(out, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            gp = lambda c: (c - db) / c * 100
            print(f"[{n + 1}/{len(jobs)}] {p} {k} pipe gap {gp(red['obj']):.2f}% g {g * 100:.2f}% delta "
                  f"{rec['delta'] * 100:.3f}% t {red['time']:.1f}s"
                  + (f" | polish {gp(rec['polish']['obj']):.2f}% full5 {gp(rec['full5']['obj']):.2f}%" if "polish" in rec else ""),
                  flush=True)
        return
    tf = load(os.path.join(ROOT, "test_fresh.npz"))
    R = pipeline_records()
    get = lambda s, k: data_instance(tf, k) if s == "tf" else shifted_instance(s, k)
    if a.stage == "retime":
        picks = [("tf", k) for k in (0, 1)] + [(s, k) for s in ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
                                               for k in (0, 1)]
        out = []
        bhg.warm(m, get(*picks[0]))
        for s, k in picks:
            sc = get(s, k)
            rr = R[(s, k)]
            full = solve_milp(m, sc, {}, TL, 1e-3)
            rel = m.solve_dispatch(sc, None, relax=True)
            _, fix, _ = bhg.run(m, sc, rel)
            red = solve_milp(m, sc, fix, TL, GAP5)
            o_f, o_p = rr[R_FULL], rr[R_PIPE]
            out.append(dict(set=s, k=k, cpu_old=o_f["cpu"], full_old=o_f["time"], full_new=full["time"], pipe_old=o_p["time"],
                            pipe_new=red["time"], same_full=abs(full["obj"] - o_f["obj"]) <= 1e-6 * abs(o_f["obj"]),
                            same_pipe=abs(red["obj"] - o_p["obj"]) <= 1e-6 * abs(o_p["obj"]),
                            fixed_match=abs(fixed_share(fix, sc.g_out, 12, G) - o_p["fixed_share"]) < 1e-9))
            print(out[-1], flush=True)
        rf = np.array([o["full_new"] / o["full_old"] for o in out])
        rp = np.array([o["pipe_new"] / o["pipe_old"] for o in out])
        summ = dict(n=len(out), full_ratio_median=float(np.median(rf)),
                    full_ratio_of_means=float(sum(o["full_new"] for o in out) / sum(o["full_old"] for o in out)),
                    full_within_10pct=float(np.mean(np.abs(rf - 1) <= 0.1) * 100), pipe_ratio_median=float(np.median(rp)),
                    pipe_ratio_of_means=float(sum(o["pipe_new"] for o in out) / sum(o["pipe_old"] for o in out)),
                    pipe_within_10pct=float(np.mean(np.abs(rp - 1) <= 0.1) * 100),
                    same_objectives=int(sum(o["same_full"] and o["same_pipe"] for o in out)),
                    fixings_match=int(sum(o["fixed_match"] for o in out)))
        summ["agree_within_10pct"] = bool(all(abs(summ[k] - 1) <= 0.1 for k in ("full_ratio_median", "full_ratio_of_means",
                                                                                "pipe_ratio_median", "pipe_ratio_of_means")))
        json.dump(dict(summary=summ, instances=out), open(os.path.join(RES, "cguard_retime.json"), "w"), indent=1, default=float)
        print(json.dumps(summ, indent=1))
        return
    if a.stage == "testfull":
        # the timing check failed (results/uc12/cguard_retime.json: this core now runs the same solves ~30 % faster than the
        # m1x records), so nothing of the pipeline records is reused: per instance, back to back on this core, the full
        # MILP at 0.1 % (dual bound, speed-up denominator) and at 0.5 %, the pipeline, and the checks. C2's polish is run
        # wherever g > 1 % (the lowest grid value), so any d_post >= 1 % (main rule or arm B) is read from the records.
        sel = json.load(open(os.path.join(RES, "cguard_select.json")))
        d_pre = sel["d_pre"]
        out = os.path.join(RES, "cguard_test_full.jsonl")
        done = {(r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
        jobs = [j for j in test_jobs() if j not in done]
        if jobs:
            bhg.warm(m, get(*jobs[0]))
        print(f"{len(jobs)} test instances (full re-run); d_pre {d_pre}", flush=True)
        for n, (s, k) in enumerate(jobs):
            sc = get(s, k)
            full = solve_milp(m, sc, {}, TL, 1e-3)
            full.pop("u", None)
            full5 = solve_milp(m, sc, {}, TL, GAP5)
            full5.pop("u", None)
            rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
            _, fix, info = bhg.run(m, sc, rel)
            red = solve_with_start(m, sc, fix, TL, GAP5)
            g = float(bound_gap(red["obj"], rel.obj)) if red["feasible"] else np.inf
            rrl = relaxed_reduced_lp(m, sc, fix)
            delta = float(lp_increase(rrl["cost"], rel.obj))
            old_rec = R.get((s, k), {})
            rec = dict(set=s, k=k, z_lp=float(rel.obj), relax_s=relax_s, pre_s=info["pre_s"], guard_s=info["guard_s"],
                       released_lp=info.get("released_lp"), fixed=fixed_share(fix, sc.g_out, 12, G), delta=delta,
                       rr_s=rrl["seconds"], g=g, d_pre=d_pre, full=full, full5=full5, pipe=rec_of(red),
                       same_as_m1x=dict(full=abs(full["obj"] - old_rec[R_FULL]["obj"]) <= 1e-6 * abs(full["obj"]) if old_rec else None,
                                        pipe=abs(red["obj"] - old_rec[R_PIPE]["obj"]) <= 1e-6 * abs(red["obj"]) if old_rec else None))
            if delta > d_pre:
                fix1, i1 = release_by_saving(m, sc, fix, rel.obj, d_pre)
                r1 = solve_milp(m, sc, fix1, TL, GAP5)
                r1.pop("u", None)
                rec["c1"] = dict(r1, fixed_share=fixed_share(fix1, sc.g_out, 12, G), extra_s=i1["seconds"], info=i1)
            else:
                rec["c1"] = None
            if g > GRID_POST[0] and red["feasible"]:
                pol, pinfo = gradient_polish(m, sc, fix, red["x"], red["u"], N_POLISH, TL, GAP5)
                rec["polish"] = dict(rec_of(pol), **pinfo)
            else:
                rec["polish"] = None
            with open(out, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(f"[{n + 1}/{len(jobs)}] {s} {k} full {full['time']:.1f}s full5 {full5['time']:.1f}s pipe {red['time']:.1f}s "
                  f"g {g * 100:.2f}% delta {delta * 100:.3f}% c1 {rec['c1'] is not None} polish {rec['polish'] is not None}",
                  flush=True)
        return
    if a.stage == "testB":
        # arm B (recalibrated on in-distribution + shifted validation, scripts/uc_cguard_shiftval.py): C2's polish on the
        # test instances where g > d_post_B and the main pass did not already polish (g <= d_post)
        d_b = json.load(open(os.path.join(RES, "cguard_select_B.json")))["d_post_B"]
        T = {(r["set"], r["k"]): r for r in map(json.loads, open(os.path.join(RES, "cguard_test.jsonl")))}
        out = os.path.join(RES, "cguard_test_B.jsonl")
        done = {(r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
        jobs = [sk for sk, r in T.items() if r["g"] > d_b and r["c2"] is None and sk not in done]
        print(f"arm B: d_post_B {d_b}; {len(jobs)} instances need a polish", flush=True)
        if jobs:
            bhg.warm(m, get(*jobs[0]))
        for s, k in jobs:
            sc = get(s, k)
            P = R[(s, k)][R_PIPE]
            rel = m.solve_dispatch(sc, None, relax=True)
            _, fix, _ = bhg.run(m, sc, rel)
            red = solve_with_start(m, sc, fix, TL, GAP5)
            pol, pinfo = gradient_polish(m, sc, fix, red["x"], red["u"], N_POLISH, TL, GAP5)
            rec = dict(set=s, k=k, d_post_B=d_b, c2=dict(rec_of(pol), same_as_record=abs(red["obj"] - P["obj"]) <= 1e-6 * abs(P["obj"]),
                                                       **pinfo))
            with open(out, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(s, k, "polished", flush=True)
        return
    # ---------------------------------------------------------------- test
    sel = json.load(open(os.path.join(RES, "cguard_select.json")))
    rt = json.load(open(os.path.join(RES, "cguard_retime.json")))["summary"]
    assert rt["agree_within_10pct"], "timing check failed: the pipeline records must be re-run, not reused"
    d_pre, d_post = sel["d_pre"], sel["d_post"]
    out = os.path.join(RES, "cguard_test.jsonl")
    done = {(r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    jobs = [j for j in test_jobs() if j not in done]
    if jobs:
        bhg.warm(m, get(*jobs[0]))
    print(f"{len(jobs)} test instances; d_pre {d_pre}, d_post {d_post}", flush=True)
    for n, (s, k) in enumerate(jobs):
        sc = get(s, k)
        rr = R[(s, k)]
        P = rr[R_PIPE]
        rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
        _, fix, info = bhg.run(m, sc, rel)
        assert abs(fixed_share(fix, sc.g_out, 12, G) - P["fixed_share"]) < 1e-9, (s, k)
        rrl = relaxed_reduced_lp(m, sc, fix)
        delta = float(lp_increase(rrl["cost"], rel.obj))
        g = float(bound_gap(P["obj"], rel.obj))
        rec = dict(set=s, k=k, z_lp=float(rel.obj), delta=delta, g=g, rr_s=rrl["seconds"], d_pre=d_pre, d_post=d_post,
                   fired_pre=delta > d_pre, fired_post=g > d_post)
        # C1: pre-solve LP-cost check (its LP is an added cost on every instance)
        if delta > d_pre:
            fix1, i1 = release_by_saving(m, sc, fix, rel.obj, d_pre)
            r1 = solve_milp(m, sc, fix1, TL, GAP5)
            r1.pop("u", None)
            rec["c1"] = dict(r1, fixed_share=fixed_share(fix1, sc.g_out, 12, G), extra_s=i1["seconds"], info=i1)
        else:
            rec["c1"] = None                      # identical to the pipeline; extra time = one relaxed reduced LP
        # C2: post-solve bound check + gradient polish
        if g > d_post:
            red = solve_with_start(m, sc, fix, TL, GAP5)        # the pipeline's solution vector (time from the record)
            same = abs(red["obj"] - P["obj"]) <= 1e-6 * abs(P["obj"])
            pol, pinfo = gradient_polish(m, sc, fix, red["x"], red["u"], N_POLISH, TL, GAP5)
            rec["c2"] = dict(rec_of(pol), same_as_record=same, **pinfo)
        else:
            rec["c2"] = None
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[{n + 1}/{len(jobs)}] {s} {k} delta {delta * 100:.3f}% g {g * 100:.2f}% fired pre {rec['fired_pre']} "
              f"post {rec['fired_post']}", flush=True)


if __name__ == "__main__":
    main()
