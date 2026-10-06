"""PGLib-UC California: score the evaluation records with the paper's metrics and write the result tables.

    python3 scripts/uc_pglib_report.py [--split test]

Reads results/pglib/pglib_eval_<split>.jsonl, the probe / validation / training / tuning records, and writes
results/pglib/pglib_results.json and results/pglib/pglib_results.md.

Paper metrics (Fritz et al., Sec. IV-C), per rule over feasible instances: gap (C - DB) / C with DB = dual bound of the
full MILP (generation run, 0.1 % target); runtime incl. inference, guards, reduced MILP; speed-up T_MILP / T_m per
instance (T_MILP = generation-run full MILP; the --b2b subset is also scored against the back-to-back re-solve);
fixed share of the free commitment decisions. Added: time-to-quality speed-up t_q / T_m with t_q the first time the
full MILP's incumbent log reaches the rule's cost (T_MILP if it never does), served share, ratio of mean times.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from uc_papereval_score import PAPER, stats  # noqa: E402

OUT = "results/pglib"
ROOT = "data/generated/pglib_ca"

NAMES = {
    "lprelax_integral": "no learning: fix the integral values of the LP relaxation",
    "knn_const_0.01": "kNN (k = 50), constant [0.01, 0.99]",
    "knn_const_0.05": "kNN (k = 50), constant [0.05, 0.95]",
    "ltf_knn_1": "**Learning to Fix, kNN, eps = 1 %** (paper's setting)",
    "ltf_knn_5": "Learning to Fix, kNN, eps = 5 %",
    "ltf_st_1": "Learning to Fix on our self-trained model, eps = 1 %",
    "ltf_lf_1": "Learning to Fix on our label-free model, eps = 1 %",
    "ltf_rl_1": "Learning to Fix on our REINFORCE model, eps = 1 %",
    "st_const_0.01": "RACLearn-style: self-trained p, constant [0.01, 0.99]",
    "lf_const_0.01": "RACLearn-style: label-free p, constant [0.01, 0.99]",
    "rl_const_0.01": "RACLearn-style: REINFORCE p, constant [0.01, 0.99]",
    "st_rac_95": "RACLearn: self-trained p, 95 % most confident",
    "lf_rac_95": "RACLearn: label-free p, 95 % most confident",
    "rl_rac_95": "RACLearn: REINFORCE p, 95 % most confident",
    "harm_90": "ours: error-cost ranking + adequacy guard, 90 %",
    "harm_95_lpg": "ours: error-cost + adequacy + LP guard, 95 % (combined)",
    "harm_98_lpg": "ours: error-cost + adequacy + LP guard, 98 % (combined)",
}


def tq(inc_t, inc_obj, target, t_full):
    for t, o in zip(inc_t, inc_obj):
        if o <= target * (1 + 1e-6) + 1e-9:
            return t
    return t_full


def fmt_row(r, cols):
    out = []
    for c in cols:
        v = r.get(c)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append("–")
        elif isinstance(v, str):
            out.append(v)
        elif c.startswith("speedup") or c.startswith("ttq"):
            out.append(f"{v:.1f}")
        elif c in ("n", "n_feas"):
            out.append(f"{int(v)}")
        else:
            out.append(f"{v:.2f}")
    return "| " + " | ".join(out) + " |"


def table(rows, cols, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "\n".join(fmt_row(r, cols) for r in rows) + "\n"


def paired(recs, a, b, key="gap"):
    """per-instance gap difference a - b over instances feasible for both; bootstrap 95 % CI"""
    da = []
    for r in recs:
        ra, rb = r["rules"].get(a), r["rules"].get(b)
        if ra and rb and ra["feasible"] and rb["feasible"]:
            ga = (ra["obj"] - r["bound"]) / ra["obj"] * 100
            gb = (rb["obj"] - r["bound"]) / rb["obj"] * 100
            da.append(ga - gb)
    da = np.array(da)
    if len(da) == 0:
        return None
    bi = np.random.default_rng(0).integers(0, len(da), (2000, len(da)))
    return dict(n=len(da), mean=float(da.mean()), ci=[float(x) for x in np.percentile(da[bi].mean(1), [2.5, 97.5])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    by_i = {}
    for f in sorted(os.listdir(OUT)):                 # merge evaluation passes (pglib_eval_<split><tag>.jsonl)
        if f.startswith(f"pglib_eval_{a.split}") and f.endswith(".jsonl"):
            for x in open(os.path.join(OUT, f)):
                r = json.loads(x)
                if r["i"] not in by_i:
                    by_i[r["i"]] = r
                else:
                    q = by_i[r["i"]]
                    q["rules"].update(r["rules"])
                    if r["e2e"] and not q["e2e"]:
                        q["e2e"] = r["e2e"]
                    if "full" in r and "full" not in q:
                        q["full"] = r["full"]
    recs = sorted(by_i.values(), key=lambda r: r["i"])
    common = set.intersection(*[set(r["rules"]) for r in recs])
    for r in recs:
        r["rules"] = {k: v for k, v in r["rules"].items() if k in common}
    n = len(recs)
    T, G = 24, None
    db = np.array([r["bound"] for r in recs])
    t_full = np.array([r["t_full"] for r in recs])
    ref = np.array([r["obj"] for r in recs])
    order = list(NAMES)
    names = sorted(recs[0]["rules"].keys(), key=lambda k: order.index(k) if k in order else 99)
    G = 410
    out = dict(n=n, rows=[], e2e=[], b2b={}, paired={})
    # full MILP row
    full = [dict(obj=r["obj"], shed=0.0, short=0.0, time=r["t_full"], feasible=True, fixed=0.0) for r in recs]
    row = stats(full, db, t_full, np.zeros(n), ref, "full MILP (HiGHS, 1 thread, 0.1 % target)")
    row["ttq_mean"] = 1.0
    out["rows"].append(row)
    for nm in names:
        rs, ov, ttq = [], [], []
        for r in recs:
            v = r["rules"][nm]
            tm = v["time"] + v["extra"] + v.get("lpg_s", 0.0)
            rs.append(dict(obj=v["obj"], shed=v["shed"] if np.isfinite(v["shed"]) else 0, short=v["short"] if np.isfinite(v["short"]) else 0,
                           time=tm, feasible=bool(v["feasible"]), fixed=v["n_fixed"] / (T * G)))
            ov.append(v["extra"] + v.get("lpg_s", 0.0))
        row = stats(rs, db, t_full, np.zeros(n), ref, NAMES.get(nm, nm))
        f = [k for k in range(n) if rs[k]["feasible"]]
        if f:
            tt = np.array([tq(recs[k]["inc_t"], recs[k]["inc_obj"], rs[k]["obj"], t_full[k]) / rs[k]["time"] for k in f])
            row.update(ttq_mean=float(tt.mean()), ttq_median=float(np.median(tt)),
                       overhead_mean=float(np.mean([ov[k] for k in f])), key=nm,
                       n_limit=int(sum(recs[k]["rules"][nm]["status"] != "Optimal" for k in f)))
        out["rows"].append(row)
    # end-to-end rows (no MILP)
    srcs = sorted({e["src"] for e in recs[0]["e2e"]}) if all(r["e2e"] for r in recs) else []
    for s in srcs:
        for mode in ("th0.5", "screen"):
            rs, ttq = [], []
            for r in recs:
                es = [e for e in r["e2e"] if e["src"] == s]
                cold = r["e2e"][0]["lp_s"]        # the instance's first dispatch LP (model loaded, no warm start)
                if mode == "th0.5":
                    e = [x for x in es if abs(x["th"] - 0.5) < 1e-9][0]
                    tm = e["t_in"] + e["rep_s"] + cold
                else:                              # inference + all repairs + one cold and k - 1 warm-started LPs
                    e = min(es, key=lambda x: x["cost"])
                    tm = es[0]["t_in"] + sum(x["rep_s"] + x["lp_s"] for x in es) - es[0]["lp_s"] + cold
                rs.append(dict(obj=e["cost"], shed=e["shed"], short=e["short"], time=tm, feasible=True, fixed=1.0))
            row = stats(rs, db, t_full, np.zeros(n), ref, f"end-to-end, {s} p, " + ("threshold 0.5, 1 LP" if mode == "th0.5" else f"screening {len(es)} thresholds"))
            tt = np.array([tq(recs[k]["inc_t"], recs[k]["inc_obj"], rs[k]["obj"], t_full[k]) / rs[k]["time"] for k in range(n)])
            row.update(ttq_mean=float(tt.mean()), ttq_median=float(np.median(tt)), key=f"e2e_{s}_{mode}")
            out["e2e"].append(row)
    # back-to-back subset
    b2 = [r for r in recs if "full" in r]
    if b2:
        tb = np.array([r["full"]["time"] for r in b2])
        tg = np.array([r["t_full"] for r in b2])
        out["b2b"] = dict(n=len(b2), t_gen_mean=float(tg.mean()), t_b2b_mean=float(tb.mean()),
                          ratio_mean=float(np.mean(tb / tg)), obj_rel_diff_max=float(max(abs(r["full"]["obj"] - r["obj"]) / r["obj"] for r in b2)))
        sub = []
        for nm in names:
            sps = []
            for r in b2:
                v = r["rules"][nm]
                if v["feasible"]:
                    sps.append((r["full"]["time"] / (v["time"] + v["extra"] + v.get("lpg_s", 0.0)),
                                r["t_full"] / (v["time"] + v["extra"] + v.get("lpg_s", 0.0))))
            if sps:
                sps = np.array(sps)
                sub.append(dict(key=nm, speedup_b2b=float(sps[:, 0].mean()), speedup_gen=float(sps[:, 1].mean())))
        out["b2b"]["rules"] = sub
    # paired comparisons against faithful LtF-kNN at eps = 1 %
    for nm in names:
        if nm != "ltf_knn_1" and "ltf_knn_1" in names:
            out["paired"][nm] = paired(recs, nm, "ltf_knn_1")
    # pareto: best mean gap at mean speed-up >= x (feasible >= 95 %)
    allrows = [r for r in out["rows"][1:] + out["e2e"] if r.get("feasible", 0) >= 95.0 and "gap_mean" in r]
    out["pareto"] = {}
    for x in (2, 5, 10, 20, 50, 100):
        c = [r for r in allrows if r["speedup_mean"] >= x]
        if c:
            b = min(c, key=lambda r: r["gap_mean"])
            out["pareto"][x] = dict(method=b["method"], gap_mean=b["gap_mean"], speedup_mean=b["speedup_mean"],
                                    ttq_mean=b.get("ttq_mean"))
    # side records
    for k, f in (("probe", "pglib_probe.jsonl"),):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            out[k] = [{kk: v for kk, v in json.loads(x).items() if kk != "inc"} for x in open(p)]
    for k, f in (("validate", "pglib_validate.json"), ("train_stats", "pglib_train_stats.json")):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            out[k] = json.load(open(p))
    out["ltf_tuning"] = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("pglib_ltf_") and f.endswith(".json"):
            J = json.load(open(os.path.join(OUT, f)))
            out["ltf_tuning"][f[10:-5]] = {k: J[k] for k in ("val_fixed_share", "val_off_share", "val_on_share", "verify_max",
                                                              "wall_s", "cpu_s", "budget_stop", "n_val") if k in J}
            out["ltf_tuning"][f[10:-5]].update(iterations=len(J["history"]), n_check_solves=J["stats"]["n_check_solves"],
                                                n_relax=J["stats"]["n_relax"], t_check=J["stats"]["t_check"], t_relax=J["stats"]["t_relax"],
                                                t_master=J["stats"]["t_master"])
    data = {}
    for sp in ("train", "val", "test"):
        p = os.path.join(ROOT, f"{sp}.npz")
        if os.path.exists(p):
            d = np.load(p)
            row = dict(n=int(len(d["load"])), t_label_mean=float(d["t_label"].mean()), t_rel_mean=float(d["t_rel"].mean()),
                       c_lf_over_rel_mean=float(np.mean(d["c_lf"] / d["c_rel"] - 1) * 100),
                       base_counts=np.bincount(d["base"], minlength=5).tolist(), n_reject=int(d["n_reject"].sum()) if "n_reject" in d else None)
            if np.isfinite(d["time"]).any():
                row.update(milp_time_mean=float(d["time"].mean()), milp_time_median=float(np.median(d["time"])),
                           milp_time_max=float(d["time"].max()), milp_time_min=float(d["time"].min()),
                           milp_opt_share=float(np.mean(d["opt"])), milp_gap_mean=float(d["gap"].mean() * 100),
                           milp_gap_max=float(d["gap"].max() * 100), c_lf_over_milp_mean=float(np.mean(d["c_lf"] / d["obj"] - 1) * 100),
                           c_rel_gap_to_milp_mean=float(np.mean(1 - d["c_rel"] / d["obj"]) * 100),
                           t_first_within_05=float(np.nanmean([tq(d["inc_t"][k][np.isfinite(d["inc_t"][k])], d["inc_obj"][k][np.isfinite(d["inc_t"][k])],
                                                                  d["obj"][k] * 1.005, d["time"][k]) for k in range(len(d["load"]))])),
                           core_hours=float(d["time"].sum() / 3600))
            data[sp] = row
    out["data"] = data
    json.dump(out, open(os.path.join(OUT, "pglib_results.json"), "w"), indent=1, default=float)

    cols = ["method", "feasible", "gap_mean", "gap_max", "time_mean", "time_max", "speedup_mean", "speedup_max",
            "speedup_ratio_of_means", "ttq_mean", "fixed_mean", "served"]
    head = ["method", "feasible %", "gap mean %", "gap max %", "runtime mean s", "runtime max s", "speed-up mean",
            "speed-up max", "ratio of mean times", "time-to-quality speed-up mean", "fixed %", "served %"]
    md = [f"# PGLib-UC California (24 h, 410 free units): results on {n} {a.split} instances\n"]
    md.append("Paper metrics (gap to the full MILP's dual bound; feasible instances; runtimes include inference and guards).\n")
    md.append(table(out["rows"] + out["e2e"], cols, head))
    md.append("\nPaper, Table I (Irish system, 72 h, Gurobi), for reference:\n")
    md.append("| method | feasible % | gap mean % | gap max % | runtime mean s | speed-up mean | speed-up max | fixed % |\n|---|---|---|---|---|---|---|---|\n")
    for p in PAPER:
        md.append(f"| {p[0]} | {p[1]:.2f} | {p[2]:.2f} | {p[3]:.2f} | {p[4]:.2f} | {p[6]:.2f} | {p[7]:.2f} | {p[8]:.2f} |\n")
    md.append("\nPareto (best mean gap at mean speed-up >= x, feasible >= 95 %):\n\n| x | method | gap mean % | speed-up mean | time-to-quality |\n|---|---|---|---|---|\n")
    for x, b in out["pareto"].items():
        md.append(f"| {x}x | {b['method']} | {b['gap_mean']:.3f} | {b['speedup_mean']:.1f} | {b['ttq_mean']:.1f} |\n")
    md.append("\nPaired gap differences vs Learning to Fix (kNN, eps = 1 %), rule minus LtF, pp [bootstrap 95 % CI]:\n\n| rule | n | mean | CI |\n|---|---|---|---|\n")
    for k, v in out["paired"].items():
        if v:
            md.append(f"| {NAMES.get(k, k)} | {v['n']} | {v['mean']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |\n")
    if out["b2b"]:
        b = out["b2b"]
        md.append(f"\nBack-to-back check ({b['n']} instances): full MILP {b['t_gen_mean']:.1f} s in the generation run vs "
                  f"{b['t_b2b_mean']:.1f} s back to back (mean ratio {b['ratio_mean']:.2f}); objective max rel. diff {b['obj_rel_diff_max']:.2e}.\n\n")
        md.append("| rule | speed-up vs back-to-back MILP | vs generation-run MILP |\n|---|---|---|\n")
        for r in b["rules"]:
            md.append(f"| {NAMES.get(r['key'], r['key'])} | {r['speedup_b2b']:.1f} | {r['speedup_gen']:.1f} |\n")
    md.append("\nData:\n\n```\n" + json.dumps(data, indent=1) + "\n```\n")
    md.append("\nLearning to Fix tuning:\n\n```\n" + json.dumps(out["ltf_tuning"], indent=1) + "\n```\n")
    open(os.path.join(OUT, "pglib_results.md"), "w").write("".join(md))
    print("".join(md))
