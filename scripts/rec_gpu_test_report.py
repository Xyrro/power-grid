"""GPU round: the single test pass of both arms on the 120 test_fresh instances, in ONE metric.

Reads results/recursive/gpu_round/rloop_eval_uc12_<model>_s*.jsonl (arm B, scripts/pace/pgu_test_armb.sbatch) and
trm_eval_test_fresh_s*.jsonl (arm A, scripts/pace/pgu_test_arma.sbatch), written after every choice was fixed on
validation. Pure standard library, so it runs anywhere.

Metric (the paper's, used throughout docs/RESEARCH.md): gap = (C - DB) / C, where DB is the tightest proven lower
bound of the instance over every full-MILP solve in these records (each one is a valid bound). Served = no load
shedding and no reserve shortfall. Speed-up = T(full MILP at 0.1 %) / T(method), both measured in the same process
on the same core (paired timing); mean and geometric mean. Paired bootstrap CIs (10,000 resamples) of per-instance
differences. Arm A and arm B solve their own MILPs, so cross-arm speed-ups compare ratios, not seconds.

    python3 scripts/rec_gpu_test_report.py
"""
import glob
import json
import math
import os
import random

RES = os.path.join("results", "recursive", "gpu_round")
MODELS_B = ("w2_rloop_rl", "w1_rloop_nofb_rl")
BOOT = 10000


def read(pattern, key):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[r[key]] = r
    return out


def served(shed, short):
    return shed < 1e-6 and short < 1e-6


def halt_step(rl, thr=0.5):
    for i, r in enumerate(rl):
        if r.get("halt", 0.0) >= thr:
            return i
    return len(rl) - 1


def arm_b_methods(r, tag):
    """per-instance (cost, served, time) of every arm B pipeline, and the reference MILP time"""
    m = {}
    f1, f5 = r["full"]["0.001"], r["full"]["0.005"]
    m[f"{tag}|milp01"] = (f1["obj"], served(f1["shed"], f1["short"]), f1["time"])
    m[f"{tag}|milp05"] = (f5["obj"], served(f5["shed"], f5["short"]), f5["time"])
    c = r["lpround"]["c"]
    j = min(range(len(c)), key=lambda i: c[i][0])
    m["lpround (no learning, 5 LPs)"] = (c[j][0], served(c[j][1], c[j][2]), r["lpround"]["t_pre"] + c[-1][3])
    ls = r["lpls"]["recs"][-1]
    m["LP-checked local search (no learning)"] = (ls["best"], served(ls["best_shed"], ls["best_short"]),
                                                  r["lpls"]["t_pre"] + ls["time"])
    rl, ks = r["rloop"], r["k_sel"]
    x = rl[ks - 1]
    m[f"{tag} @{ks} steps"] = (x["best"], served(x["best_shed"], x["best_short"]), x["time"])
    h = rl[halt_step(rl)]
    m[f"{tag} @halting head"] = (h["best"], served(h["best_shed"], h["best_short"]), h["time"])
    cost, srv, t = x["best"], served(x["best_shed"], x["best_short"]), x["time"]
    s = r.get("safety")
    if s:
        t += s["time"] + s.get("t_fix", 0.0)
        if s["feasible"] and served(s.get("shed") or 0.0, s.get("short") or 0.0) and (not srv or s["obj"] < cost):
            cost, srv = s["obj"], True
    m[f"{tag} @{ks} + safety net"] = (cost, srv, t)
    return m, f1["time"], f1["bound"]


def arm_a_methods(r):
    m = {}
    for k, v in r.items():
        if isinstance(v, dict) and "cost" in v and "time" in v and v.get("cost") is not None:
            m[k] = (v["cost"], served(v.get("shed", 0.0) or 0.0, v.get("short", 0.0) or 0.0), v["time"])
    return m, r["milp01"]["time"], r["milp01"]["bound"]


def ci(xs, rng):
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return ms[int(0.025 * BOOT)], ms[int(0.975 * BOOT) - 1]


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    B = {t: read(f"rloop_eval_uc12_{t}_s*.jsonl", "k") for t in MODELS_B}
    A = read("trm_eval_test_fresh_s*.jsonl", "i")
    ids = sorted(set.intersection(*[set(d) for d in list(B.values()) + ([A] if A else []) if d]))
    if not ids:
        raise SystemExit("no complete instances yet")
    rows, tref, db = {}, {}, {}
    for i in ids:
        bounds = []
        for t, d in B.items():
            if i in d:
                m, tm, b = arm_b_methods(d[i], t)
                bounds.append(b)
                for k, v in m.items():
                    if (k.startswith(("lpround", "LP-checked")) or "|milp" in k) and t != MODELS_B[0]:
                        continue                  # baselines and full MILPs once, from the first model's run
                    rows.setdefault(k, {})[i] = (v, tm)
        if i in A:
            m, tm, b = arm_a_methods(A[i])
            bounds.append(b)
            for k, v in m.items():
                rows.setdefault("A|" + k, {})[i] = (v, tm)
        db[i] = max(bounds)
    rng = random.Random(0)

    def per(name):
        d = rows[name]
        g = [(d[i][0][0] - db[i]) / d[i][0][0] * 100 for i in ids if i in d]
        s = [d[i][0][1] for i in ids if i in d]
        sp = [d[i][1] / max(d[i][0][2], 1e-9) for i in ids if i in d]
        return g, s, sp

    out = dict(n=len(ids), metric="(C - DB) / C, DB = tightest proven bound over this instance's full-MILP solves",
               methods={}, paired={})
    lines = [f"# GPU round: test results ({len(ids)} test_fresh instances, every choice fixed on validation)", "",
             "Gap = (C − DB)/C to the tightest proven bound; served = no shedding or reserve shortfall; speed-up = "
             "full MILP at 0.1 % / method, timed in the same process (mean / geometric mean).", "",
             "| method | served | gap mean | median | max | speed-up mean / geo |", "|---|---|---|---|---|---|"]
    order = sorted(rows, key=lambda k: sum(per(k)[0]) / len(per(k)[0]))
    for k in order:
        g, s, sp = per(k)
        n = len(g)
        st = dict(n=n, served=100 * sum(s) / n, gap_mean=sum(g) / n, gap_median=sorted(g)[n // 2], gap_max=max(g),
                  speedup_mean=sum(sp) / n, speedup_geo=math.exp(sum(math.log(x) for x in sp) / n))
        out["methods"][k] = st
        lines.append(f"| {k} | {st['served']:.1f} % | {st['gap_mean']:.3f} % | {st['gap_median']:.3f} % | "
                     f"{st['gap_max']:.2f} % | {st['speedup_mean']:.1f}× / {st['speedup_geo']:.1f}× |")

    def pair(a, b):
        da, dbb = rows.get(a), rows.get(b)
        if not da or not dbb:
            return None
        common = [i for i in ids if i in da and i in dbb]
        ga = [(da[i][0][0] - db[i]) / da[i][0][0] * 100 for i in common]
        gb = [(dbb[i][0][0] - db[i]) / dbb[i][0][0] * 100 for i in common]
        d = [y - x for x, y in zip(ga, gb)]                       # > 0: A better
        lt = [math.log(dbb[i][0][2] / da[i][0][2]) for i in common]   # > 0: A faster
        both = [k for k, i in enumerate(common) if da[i][0][1] and dbb[i][0][1]]
        db_ = [d[k] for k in both]
        r = dict(n=len(common), diff=sum(d) / len(d), diff_ci=ci(d, rng), log_time=sum(lt) / len(lt),
                 log_time_ci=ci(lt, rng), n_both_served=len(both),
                 diff_both=sum(db_) / len(db_) if db_ else float("nan"), diff_both_ci=ci(db_, rng) if len(db_) > 1 else None)
        out["paired"][f"{a} vs {b}"] = r
        return r

    pairs = [(f"{MODELS_B[0]} @24 steps", f"{MODELS_B[1]} @24 steps"),
             (f"{MODELS_B[0]} @halting head", f"{MODELS_B[1]} @halting head"),
             (f"{MODELS_B[0]} @24 + safety net", f"{MODELS_B[1]} @24 + safety net"),
             (f"{MODELS_B[0]} @24 steps", "lpround (no learning, 5 LPs)"),
             (f"{MODELS_B[0]} @24 steps", "LP-checked local search (no learning)"),
             (f"{MODELS_B[0]} @24 + safety net", f"{MODELS_B[0]}|milp05")]
    plan_p = os.path.join(RES, "trm_select_gpu.json")
    if A and os.path.exists(plan_p):
        plan = json.load(open(plan_p))
        head = plan.get("headline", "")
        fin = plan.get("final", "")
        for c in plan.get("trm", {}):
            if c != fin and head:
                pairs.append(("A|" + head, "A|" + c + head[len(fin):]))
        if head:
            pairs += [("A|" + head, "A|milp05"), ("A|" + head, "A|lpround5r"), ("A|" + head, "A|m1x_mix5r"),
                      ("A|" + head, f"{MODELS_B[0]} @24 steps")]
    lines += ["", "Paired differences (positive = the first method is better / faster; 95 % bootstrap CI):", "",
              "| first | second | n | Δ gap (pp) | Δ log time | n both served | Δ gap, both served |",
              "|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        r = pair(a, b)
        if r:
            dbs = (f"{r['diff_both']:+.3f} [{r['diff_both_ci'][0]:+.3f}, {r['diff_both_ci'][1]:+.3f}]"
                   if r["diff_both_ci"] else "—")
            lines.append(f"| {a} | {b} | {r['n']} | {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] | "
                         f"{r['log_time']:+.2f} [{r['log_time_ci'][0]:+.2f}, {r['log_time_ci'][1]:+.2f}] | "
                         f"{r['n_both_served']} | {dbs} |")
    with open(os.path.join(RES, "test_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RES, "test_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
