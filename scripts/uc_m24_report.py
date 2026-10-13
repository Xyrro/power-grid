"""m24 study, report: results/uc24/m24_results.md / .json.

    python3 scripts/uc_m24_report.py

Test metrics (Learning to Fix, Sec. IV-C): gap (C - DB) / C to the dual bound of the full MILP solved in the same
process (0.1 %, 300 s), feasible instances; per-instance speed-up T_full / T_method (mean, median, geometric mean) with
every overhead; served = no shedding, over-generation or reserve shortfall; fixed share after guards. Time to the same
quality (TTQ): the first time the full MILP's incumbent log reaches the method's cost (censored at the full MILP's
termination time when it never does), divided by the method's time. Paired differences: instance bootstrap 95 % CIs
(2,000 resamples) of Δ gap (pp), Δ log speed-up and Δ log TTQ speed-up on instances both rules solve; LtF-kNN from the
stored uc24ltf records (cost scored against this run's DB; speed-up against the full MILP of its own run).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import load_b3, time_to_reach  # noqa: E402

U24, DATA, RES = "data/generated/uc24", "data/generated/uc24_m24", "results/uc24"
NB = 2000
TG = 24 * 73
PIPE = "m24 pipeline (LP guard, gap=0.5%)"
ORDER = ["full", "full gap=0.5%", "LtF-LP gap=0.5%", "guarded 95%", "guarded 95% gap=0.5%", "m24 + LP guard", PIPE]
NAMES = {"full": "full MILP, 0.1 % (reference)", "full gap=0.5%": "full MILP, 0.5 % gap",
         "LtF-LP gap=0.5%": "learning-free LtF on the LP relaxation, 0.5 %", "guarded 95%": "guarded 95 % rule (0.1 %)",
         "guarded 95% gap=0.5%": "guarded 95 % rule, 0.5 %", "m24 + LP guard": "m24 fixings + LP guard (0.1 %)",
         PIPE: "**m24 pipeline (LP guard, 0.5 %)**", "LtF kNN eps=1% (stored)": "LtF-kNN ε = 1 % (stored run)"}


def boot(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    b = x[np.random.default_rng(seed).integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def f(x, p=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def arrays(R, F):
    """per-instance metrics of a rule; R, F: lists of records (rule, this run's full MILP)"""
    feas = np.array([bool(r["feasible"]) for r in R])
    cost = np.array([r["obj"] if r["feasible"] else np.nan for r in R], float)
    DB = np.array([x["bound"] for x in F], float)
    t_full = np.array([x["time"] for x in F], float)
    t_m = np.array([r.get("t_method", r.get("time")) for r in R], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(feas, (cost - DB) / cost * 100, np.nan)
        sp = np.where(feas, t_full / t_m, np.nan)
    ttq = []
    for r, x in zip(R, F):
        if not r["feasible"]:
            ttq.append(np.nan)
            continue
        t = time_to_reach(x["inc"], r["obj"])
        ttq.append(x["time"] if not np.isfinite(t) else t)
    ttq = np.array(ttq)
    served = np.array([r["feasible"] and (r.get("shed") or 0) < 1e-6 and (r.get("short") or 0) < 1e-6 for r in R])
    fixed = np.array([r.get("n_fixed", 0) / TG * 100 for r in R], float)
    return dict(feas=feas, gap=gap, sp=sp, ttq=np.where(feas, ttq / t_m, np.nan), t_m=t_m, served=served, fixed=fixed,
                reached=np.array([np.isfinite(time_to_reach(x["inc"], r["obj"])) if r["feasible"] else False
                                  for r, x in zip(R, F)]))


def stats(x):
    f_ = x["feas"]
    g, sp, tq = x["gap"][f_], x["sp"][f_], x["ttq"][f_]
    return dict(n=int(len(f_)), feasible=float(f_.mean() * 100), gap_mean=float(g.mean()), gap_ci=boot(g),
                gap_median=float(np.median(g)), gap_max=float(g.max()), sp_mean=float(sp.mean()), sp_ci=boot(sp),
                sp_median=float(np.median(sp)), sp_geo=float(np.exp(np.log(sp).mean())),
                ttq_median=float(np.median(tq)), ttq_geo=float(np.exp(np.log(tq).mean())),
                served=float(x["served"].mean() * 100), fixed=float(x["fixed"][f_].mean()),
                time_mean=float(x["t_m"][f_].mean()))


def paired(a, b):
    both = a["feas"] & b["feas"]
    dg = (a["gap"] - b["gap"])[both]
    dl = (np.log(a["sp"]) - np.log(b["sp"]))[both]
    dq = (np.log(a["ttq"]) - np.log(b["ttq"]))[both]
    return dict(n=int(both.sum()), d_gap=float(dg.mean()), d_gap_ci=boot(dg), d_logsp=float(dl.mean()), d_logsp_ci=boot(dl),
                time_ratio=float(np.exp(dl.mean())), d_logttq=float(dq.mean()), d_logttq_ci=boot(dq))


def prob_section(L, res):
    tr = json.load(open(os.path.join(RES, "m24_train.json"))) if os.path.exists(os.path.join(RES, "m24_train.json")) else {}
    if not tr:
        return
    from otsl.m1x import aligned_labels, prob_metrics
    from otsl.uc import load_rts_gmlc
    from otsl.ucml import canonical_labels
    sysm = load_rts_gmlc()
    v = load_b3(os.path.join(U24, "uc24ltf_val.npz"))
    P = np.load(os.path.join(RES, "uc24ltf_probs.npz"))
    refs = {"LP relaxation values (learning-free LtF's score)": v["u_rel"].astype(np.float64),
            "kNN, k = 50, 120 labelled schedules (LtF-kNN)": P["knn_va"].astype(np.float64),
            "label-free imitation GNN (guarded 95 % rule)": P["bce_va"].astype(np.float64)}
    rows, out = [], {}
    yc = canonical_labels(sysm, v["u"], v["u0"])
    for nm, p in refs.items():
        mt = prob_metrics(p, aligned_labels(sysm, v["u"], v["u0"], p), yc)
        out[nm] = mt
        rows.append([nm, "–", "–"] + [f(mt[k], q) for k, q in (("logloss", 4), ("brier", 4), ("auc", 4), ("wrong_per_inst", 1),
                                                               ("fix99", 3), ("fix999", 3))])
    for tag, r in tr.items():
        mt = r["val"]
        out[tag] = mt
        rows.append([tag, r.get("n_train", "–"), f"{r['train_s'] / 60:.1f}" if r.get("train_s") else "–"] +
                    [f(mt[k], q) for k, q in (("logloss", 4), ("brier", 4), ("auc", 4), ("wrong_per_inst", 1),
                                              ("fix99", 3), ("fix999", 3))])
    res["validation_probabilities"] = out
    L += ["\n## Validation: probability quality (40 instances with full MILPs, labels aligned inside identical-unit groups)\n",
          table(["source", "n train", "train min", "log-loss", "Brier", "AUC", "wrong / inst. (of 1,752)", "fixable @99 %",
                 "fixable @99.9 %"], rows)]


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    res = {}
    L = ["# m24: the 12-hour recipe on the 24-hour benchmark — results\n",
         "Generated by `scripts/uc_m24_report.py`; method and discussion: [`docs/methods/m24.md`](../../docs/methods/m24.md).\n"]
    pj = os.path.join(DATA, "pilot.json")
    if os.path.exists(pj):
        p = json.load(open(pj))
        res["pilot"] = p
        L.append("## Labels\n\nPolishing pilot (teacher: 1 temporal GNN on 100 MILP-labelled instances; training instances 100–119; "
                 f"gap to their 0.5 %-gap MILP; label-free block-repaired label: median {p['label_free_blk']['gap_median_%']:.2f} %, "
                 f"mean {p['label_free_blk']['gap_mean_%']:.1f} %):\n")
        L.append(table(["time limit", "s / label", "gap median %", "gap mean %", "# > 0.5 %", "# > 1 %", "hit limit", "fixed %",
                        "unit-hours = MILP %"],
                       [[k[2:] + " s", f(v["time_mean_s"]), f(v["gap_median_%"], 3), f(v["gap_mean_%"], 3), v["n_gt_0.5%"],
                         v["n_gt_1%"], v["hit_limit"], f(v["fixed_%"], 1), f(v["agree_milp_%"], 2)]
                        for k, v in p.items() if k.startswith("tl")]))
    pol = sorted(glob.glob(os.path.join(DATA, "polish_*.npz")))
    if pol:
        Z = [np.load(x) for x in pol]
        cat = lambda k: np.concatenate([z[k] for z in Z])
        st = dict(n=int(len(cat("total_s"))), s_per_label=float(cat("total_s").mean()), core_h=float(cat("total_s").sum() / 3600),
                  from_milp=float(cat("used_milp").mean() * 100), hit_limit=float(cat("hit_limit").mean() * 100),
                  served=float(cat("served").mean() * 100), fixed=float(cat("n_fixed").mean() / TG * 100),
                  released_rows=float(cat("released_rows").mean()), released_lp=float(cat("released_lp").mean()))
        ex = sorted(glob.glob(os.path.join(DATA, "extra_*.npz")))
        st["n_new_scenarios"] = int(sum(len(np.load(x)["seed"]) for x in ex))
        st["gen_core_h"] = float(sum(np.load(x)["t_label"].sum() for x in ex) / 3600)
        res["polish"] = st
        L.append(f"\nPolished labels: {st['n']} instances (180 original training instances without MILP + {st['n_new_scenarios']} new "
                 f"training-day scenarios, generated in {st['gen_core_h']:.2f} core-h), {st['s_per_label']:.1f} s per label "
                 f"({st['core_h']:.2f} core-h), {st['fixed']:.1f} % fixed after guards (rows released {st['released_rows']:.1f}, LP "
                 f"guard {st['released_lp']:.1f} per instance), reduced MILP stopped at 10 s on {st['hit_limit']:.0f} %, label from the "
                 f"reduced MILP on {st['from_milp']:.0f} % (else the block-repaired label-free label), served {st['served']:.0f} %.")
    prob_section(L, res)
    tunes = sorted(glob.glob(os.path.join(RES, "m24_tune_*.json")))
    if tunes:
        rows = []
        b = json.load(open(os.path.join(RES, "base_tune_lp_1.json")))
        k1 = json.load(open(os.path.join(RES, "uc24ltf_tune_knn_1.json")))
        rows.append(["learning-free LtF on the LP relaxation (base)", "u_rel", "none", b["converged"], f(b["val_fixed_share"] * 100, 1),
                     "–", b["n_iter"], f(b["wall_s"] / 3600, 2)])
        rows.append(["LtF-kNN (uc24ltf)", "kNN", "none", k1["converged"], f(k1["val_fixed_share"] * 100, 1), "–",
                     k1.get("n_iter", "–"), f(k1.get("wall_s", np.nan) / 3600, 2)])
        res["tune"] = {}
        for t in tunes:
            r = json.load(open(t))
            res["tune"][r["run"]] = {k: r[k] for k in ("src", "eps", "guards", "converged", "val_fixed_share", "val_fixed_share_pre_guard",
                                                       "n_iter", "n_cut_sets", "verify_max", "n_verify_fail", "wall_s")}
            rows.append([f"m24 {r['run']}", r["src"], "+".join(r["guards"]), r["converged"], f(r["val_fixed_share"] * 100, 1),
                         f(r["val_fixed_share_pre_guard"] * 100, 1), r["n_iter"], f(r["wall_s"] / 3600, 2)])
        L += ["\n## Validation: Learning to Fix tuning (ε = 1 %, 40 instances)\n",
              table(["run", "score", "guards in the check", "converged", "val fixed % (after guards)", "before guards", "iterations",
                     "wall h"], rows)]
    cvp = os.path.join(RES, "m24_cv.jsonl")
    if os.path.exists(cvp):
        C = [json.loads(x) for x in open(cvp)]
        res["cv"] = {}
        rows = []
        lab = {"m24": "m24 (ensemble, guards)", "lp": "learning-free LtF (LP relaxation)"}
        for score in ("m24", "lp"):
            tun = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, f"m24_cv_{score}_f*.json")))]
            for kind in ("insample", "heldout"):
                rr = [r for r in C if r["score"] == score and r["kind"] == kind]
                if not rr:
                    continue
                inc = np.array([r["rel_inc"] * 100 if r["feasible"] else np.inf for r in rr])
                fe = np.isfinite(inc)
                st = dict(n=len(rr), infeasible=int((~fe).sum()), inc_mean=float(inc[fe].mean()),
                          inc_median=float(np.median(inc[fe])), inc_max=float(inc[fe].max()),
                          n_gt_1=int((inc > 1).sum()), n_gt_05=int((inc > 0.5).sum()),
                          fixed=float(np.mean([r["n_fixed"] for r in rr]) / TG * 100))
                if kind == "heldout":
                    st["tuned_fixed_share"] = [t["fixed_share"] * 100 for t in tun]
                    st["tuned_converged"] = [t["converged"] for t in tun]
                res["cv"][f"{score} {kind}"] = st
                desc = ("40-instance thresholds on the same 40 instances" if kind == "insample" else
                        f"2-fold: tuned on 20 ({', '.join(f(x, 1) for x in st['tuned_fixed_share'])} % fixed there), "
                        f"applied to the other 20")
                rows.append([lab[score], desc, st["n"], st["infeasible"], f(st["fixed"], 1), f(st["inc_mean"], 3),
                             f(st["inc_median"], 3), f(st["inc_max"], 2), f"{st['n_gt_05']} / {st['n']}",
                             f"**{st['n_gt_1']} / {st['n']}**"])
        L += ["\n## Validation-only diagnostic: does the validated ε hold out of sample? (2-fold cross-validation)\n",
              "Cost increase of the reduced MILP (0.1 %, 120 s; m24 with its test-time guards incl. the LP guard) over "
              "the validation instance's full-MILP objective C*. ε = 1 %.\n",
              table(["score", "thresholds", "n", "infeasible", "fixed %", "increase mean %", "median %", "max %",
                     "> 0.5 %", "> 1 % (= ε violated)"], rows)]
    path = os.path.join(RES, "m24_eval_test.jsonl")
    if os.path.exists(path):
        recs = [json.loads(x) for x in open(path)]
        recs.sort(key=lambda r: r["i"])
        F = [r["runs"]["full"] for r in recs]
        X = {nm: arrays([r["runs"][nm] for r in recs], F) for nm in ORDER if all(nm in r["runs"] for r in recs)}
        st_k = {}
        knn = {json.loads(x)["i"]: json.loads(x) for x in open(os.path.join(RES, "uc24ltf_eval_test.jsonl"))}
        KR = []
        tdat = load_b3(os.path.join(U24, "test.npz"))["time"]          # the uc24ltf protocol's full-MILP time
        for r in recs:
            s0 = knn[r["i"]]["LtF knn eps=1%"]
            KR.append(dict(s0, t_method=s0["t_milp"] + s0.get("pre_s", 0.0) + s0.get("t_guard", 0.0),
                           n_fixed=s0.get("n_fixed", 0), feasible=bool(s0.get("feasible", True))))
        X["LtF kNN eps=1% (stored)"] = arrays(KR, [dict(F[k], time=float(tdat[r["i"]])) for k, r in enumerate(recs)])
        S = {nm: stats(x) for nm, x in X.items()}
        res["test"] = dict(n=len(recs), stats=S, full_time_mean=float(np.mean([x["time"] for x in F])))
        L.append(f"\n## Test: {len(recs)} uc24 test instances; full MILP (0.1 %, 300 s) back to back {np.mean([x['time'] for x in F]):.0f} s "
                 f"mean (median {np.median([x['time'] for x in F]):.0f} s), core 1\n")
        rows = []
        for nm in ORDER + ["LtF kNN eps=1% (stored)"]:
            if nm not in S:
                continue
            s = S[nm]
            rows.append([NAMES.get(nm, nm), f(s["feasible"], 0), f"{f(s['gap_mean'])} [{f(s['gap_ci'][0])}, {f(s['gap_ci'][1])}]",
                         f(s["gap_median"]), f(s["gap_max"]), f(s["served"], 0),
                         f"{f(s['sp_mean'], 1)} [{f(s['sp_ci'][0], 1)}, {f(s['sp_ci'][1], 1)}]", f(s["sp_median"], 1), f(s["sp_geo"], 1),
                         f(s["ttq_median"], 2), f(s["ttq_geo"], 2), f(s["fixed"], 1)])
        L.append(table(["rule", "feasible %", "gap mean % [95 % CI]", "median", "max", "served %", "speed-up mean [CI]", "median",
                        "geo. mean", "TTQ speed-up median", "TTQ geo. mean", "fixed %"], rows))
        L.append("\nLtF-kNN: stored records of the uc24ltf test run (speed-up against the dataset run's full-MILP time, as published "
                 "there, i.e. not paired with this run's timing; cost scored against this run's dual bound; TTQ against this run's "
                 "incumbent log).\n")
        P = ["\nPaired, rule − reference (instances both solve; Δ log speed-up > 0: the rule is faster; time ratio = geometric mean "
             "of reference time / rule time):\n",
             "| rule | reference | n | Δ gap pp [CI] | Δ log speed-up [CI] | time ratio | Δ log TTQ speed-up [CI] |",
             "|---|---|---|---|---|---|---|"]
        res["test"]["paired"] = {}
        for a_ in (PIPE, "m24 + LP guard"):
            if a_ not in X:
                continue
            for b_ in ("LtF-LP gap=0.5%", "full gap=0.5%", "guarded 95% gap=0.5%", "guarded 95%", "LtF kNN eps=1% (stored)",
                       "m24 + LP guard"):
                if b_ not in X or b_ == a_:
                    continue
                p = paired(X[a_], X[b_])
                res["test"]["paired"][f"{a_} vs {b_}"] = p
                c = lambda v, ci, q=2: f"{v:+.{q}f} [{ci[0]:+.{q}f}, {ci[1]:+.{q}f}]"
                P.append(f"| {NAMES.get(a_, a_)} | {NAMES.get(b_, b_)} | {p['n']} | {c(p['d_gap'], p['d_gap_ci'])} | "
                         f"{c(p['d_logsp'], p['d_logsp_ci'])} | {p['time_ratio']:.2f}× | {c(p['d_logttq'], p['d_logttq_ci'])} |")
        p = paired(X["LtF-LP gap=0.5%"], X["full gap=0.5%"])
        res["test"]["paired"]["LtF-LP gap=0.5% vs full gap=0.5%"] = p
        P.append(f"| {NAMES['LtF-LP gap=0.5%']} | {NAMES['full gap=0.5%']} | {p['n']} | {p['d_gap']:+.2f} [{p['d_gap_ci'][0]:+.2f}, "
                 f"{p['d_gap_ci'][1]:+.2f}] | {p['d_logsp']:+.2f} [{p['d_logsp_ci'][0]:+.2f}, {p['d_logsp_ci'][1]:+.2f}] | "
                 f"{p['time_ratio']:.2f}× | {p['d_logttq']:+.2f} [{p['d_logttq_ci'][0]:+.2f}, {p['d_logttq_ci'][1]:+.2f}] |")
        L += P
        same = [abs(r["runs"]["guarded 95%"]["obj"] - (r["runs"]["guarded 95%"].get("stored_obj") or np.nan)) <= 1e-6 * r["runs"]["guarded 95%"]["obj"]
                for r in recs if r["runs"]["guarded 95%"]["feasible"]]
        res["test"]["guarded_reproduces_stored_obj_pct"] = float(np.mean(same) * 100) if same else None
    open(os.path.join(RES, "m24_results.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(RES, "m24_results.json"), "w"), indent=1, default=float)
    print("\n".join(L))
