"""m1x study, report: results/uc12/m1x_results.md / .json from the data, training, validation-fixing, tuning and test
records of the uc_m1x_* scripts.

    python3 scripts/uc_m1x_report.py

Test metrics as in scripts/uc_hybrid_report.py (imported): gap (C - DB) / C to the back-to-back full MILP's dual
bound, mean per-instance speed-up (method time includes the LP relaxation, inference of every ensemble member,
error-cost features and guards), feasible instances only; paired differences against the reference hybrid
(he_bce_s0_e1_n360, re-run in the same worker) with instance-bootstrap 95 % CIs (2,000 resamples).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.ucdata import load  # noqa: E402
from uc_hybrid_report import boot, fmt, paired, per_instance, stats  # noqa: E402

ROOT, DATA, RES = "data/generated/uc12", "data/generated/uc12_m1x", "results/uc12"
REF = "hybrid he_bce_s0_e1_n360"
MET = [("logloss", "log-loss", 4), ("brier", "Brier", 4), ("auc", "AUC", 4), ("ece", "calib. err.", 4),
       ("wrong_per_inst", "wrong / inst.", 1), ("fix99", "fixable @99 %", 3), ("fix999", "fixable @99.9 %", 3)]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def valfix_summary():
    path = os.path.join(RES, "m1x_valfix.jsonl")
    if not os.path.exists(path):
        return {}
    by = {}
    for r in map(json.loads, open(path)):
        by.setdefault((r["tag"], r["ratio"]), []).append(r)
    out = {}
    for (tag, ratio), rs in by.items():
        g = np.array([r["gap"] if r["feasible"] else np.nan for r in rs])
        out[f"{tag}@{ratio}"] = dict(n=len(rs), gap_mean=float(np.nanmean(g)), gap_mean_ci=boot(g[np.isfinite(g)]),
                                     gap_median=float(np.nanmedian(g)), n_gt1=int(np.sum(g > 1)),
                                     fixed=float(np.mean([r["fixed"] for r in rs]) * 100),
                                     time=float(np.mean([r["milp_s"] + r["guard_s"] for r in rs])),
                                     served=float(np.mean([r["shed"] < 1e-6 and r["short"] < 1e-6 for r in rs]) * 100),
                                     gaps=g.tolist())
    return out


def valfix_raw(tag, ratio=0.95):
    path = os.path.join(RES, "m1x_valfix.jsonl")
    if not os.path.exists(path):
        return {}
    return {r["i"]: r for r in map(json.loads, open(path)) if r["tag"] == tag and r["ratio"] == ratio}


def label_stats():
    out = {}
    pj = os.path.join(DATA, "pilot.json")
    if os.path.exists(pj):
        out["pilot"] = json.load(open(pj))
    ex = sorted(glob.glob(os.path.join(DATA, "extra_*.npz")), key=lambda p: int(p.split("_")[-1][:-4]))
    if ex:
        t_rel = np.concatenate([np.load(p)["t_rel"] for p in ex])
        t_lf = np.concatenate([np.load(p)["t_lf"] for p in ex])
        out["extra"] = dict(n=int(len(t_rel)), t_relax_mean=float(t_rel.mean()), t_label_free_mean=float(t_lf.mean()),
                            core_s=float(t_lf.sum()))
    po = sorted(glob.glob(os.path.join(DATA, "polish_*.npz")), key=lambda p: int(p.split("_")[-1][:-4]))
    if po:
        Z = [np.load(p) for p in po]
        E = [np.load(p.replace("polish_", "extra_")) for p in po]
        cost = np.concatenate([z["cost"] for z in Z])
        clf = np.concatenate([e["c_lf"] for e in E])
        ylf = np.concatenate([e["y_lf"] for e in E])
        y = np.concatenate([z["y"] for z in Z])
        out["polish"] = dict(n=int(len(cost)), cfg=str(Z[0]["cfg"]), from_milp=float(np.concatenate([z["used_milp"] for z in Z]).mean() * 100),
                             time_mean=float(np.concatenate([z["total_s"] for z in Z]).mean()),
                             hit_limit=float(np.concatenate([z["hit_limit"] for z in Z]).mean() * 100),
                             served=float(np.concatenate([z["served"] for z in Z]).mean() * 100),
                             cost_vs_lf_median=float(np.median(cost / clf - 1) * 100), cost_vs_lf_mean=float(np.mean(cost / clf - 1) * 100),
                             unit_hours_changed=float((y != ylf).mean() * 100),
                             unit_hours_vs_teacher=float((y != (np.concatenate([z["p_teacher"] for z in Z]) > 0.5)).mean() * 100),
                             core_s=float(np.concatenate([z["total_s"] for z in Z]).sum()))
    return out


def per_instance_recs(recs, d):
    """uc_hybrid_report.per_instance on a list of records (rules evaluated on the same instances)"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    try:
        return per_instance(f.name, d)
    finally:
        os.remove(f.name)


def test_block(lines, res, key, title, recs, d):
    X, common, t_full = per_instance_recs(recs, d)
    S = {r: stats(x, t_full) for r, x in X.items()}
    res[key] = dict(n=len(common), instances=common, full_milp_time_mean=float(t_full.mean()), stats=S)
    lines += [f"\n### {title}: {len(common)} instances; full MILP back to back {t_full.mean():.1f} s mean (core 3)\n"]
    rows = []
    order = ["full MILP", "faithful LtF BCE eps=1%", REF] + sorted(r for r in S if r.startswith("m1x "))
    for r in order:
        if r not in S:
            continue
        s_ = S[r]
        rows.append([r, fmt(s_["feasible"], 1), f"{fmt(s_['gap_mean'])} [{fmt(s_['gap_mean_ci'][0])}, {fmt(s_['gap_mean_ci'][1])}]",
                     fmt(s_["gap_median"], 3), fmt(s_["gap_max"]), fmt(s_["runtime_mean"], 1),
                     f"{fmt(s_['speedup_mean'], 1)} [{fmt(s_['speedup_mean_ci'][0], 1)}, {fmt(s_['speedup_mean_ci'][1], 1)}]",
                     fmt(s_["speedup_median"], 1), fmt(s_["speedup_ratio_of_means"]),
                     f"{fmt(s_['fixed_mean'], 1)} ({fmt(s_['fixed_pre_guard_mean'], 1)})", fmt(s_["served"], 1), s_["n_gt1"]])
    lines.append(table(["rule", "feasible %", "gap mean % [95 % CI]", "gap median %", "gap max %", "runtime s",
                        "speed-up mean [95 % CI]", "speed-up median", "speed-up (ratio of means)", "fixed % (pre-guard)",
                        "served %", "# > 1 % (dataset ref.)"], rows))
    if REF not in X:
        return
    lines.append(f"\nPaired against the reference hybrid ({REF}, re-run in the same worker); instance bootstrap 95 % CI. "
                 "Gap and speed-up on instances feasible for both rules.\n")
    rows, pj = [], {}
    for r in order[3:] + ["faithful LtF BCE eps=1%"]:
        if r not in X:
            continue
        a, b = X[r], X[REF]
        both = a["feas"] & b["feas"]
        m = lambda x: np.where(both, x, np.nan)
        dg = paired(m(a["gap_db"]), m(b["gap_db"]), "gap")
        ds = paired(m(a["sp"]), m(b["sp"]), "sp")
        dl = paired(m(np.log(a["sp"])), m(np.log(b["sp"])), "lsp")
        df = paired(m(a["fixed"]), m(b["fixed"]), "fixed")
        dt = paired(m(np.log(a["t_m"])), m(np.log(b["t_m"])), "lt")
        dr = paired(a["g_ref"], b["g_ref"], "gref")
        pj[r] = dict(gap=dg, speedup=ds, log_speedup=dl, fixed=df, log_time=dt, gap_dataset_ref=dr)
        c = lambda x, p=3: f"{x['diff']:+.{p}f} [{x['ci'][0]:+.{p}f}, {x['ci'][1]:+.{p}f}]"
        rows.append([r, c(dg), c(ds, 2), c(dl), f"{np.exp(-dt['diff']):.2f}× [{np.exp(-dt['ci'][1]):.2f}, {np.exp(-dt['ci'][0]):.2f}]",
                     c(df, 2), c(dr), dg["n"]])
    res[key]["paired_vs_ref"] = pj
    lines.append(table(["rule", "Δ gap to DB, pp", "Δ mean speed-up", "Δ log speed-up", "time ratio ref / rule (geo. mean)",
                        "Δ fixed share, pp", "Δ gap to dataset ref. (all inst., fallback), pp", "n"], rows))


def test_section(lines, res):
    path = os.path.join(RES, "m1x_eval_test_fresh.jsonl")
    if not os.path.exists(path):
        return
    d = load(os.path.join(ROOT, "test_fresh.npz"))
    recs = [json.loads(line) for line in open(path)]
    lines.append("\n## Test: test_fresh (read once, by scripts/uc_m1x_eval.py)\n\nPaper metrics (gap to the back-to-back "
                 "full MILP's dual bound, feasible instances; mean per-instance speed-up with every overhead included: LP "
                 "relaxation, forward passes of all ensemble members, error-cost features, guards).")
    # pairing check against the hybrid study's own test run (same instances, other core and load)
    hp = os.path.join(RES, "hybrid_eval_test_fresh.jsonl")
    if os.path.exists(hp):
        old = {(r["rule"], r["i"]): r for r in map(json.loads, open(hp))}
        new = {(r["rule"], r["i"]): r for r in recs}
        ii = sorted({r["i"] for r in recs})
        tf = [(new[("full MILP", i)]["time"], old[("full MILP", i)]["time"]) for i in ii if ("full MILP", i) in old and ("full MILP", i) in new]
        ro = [(new[(REF, i)]["obj"], old[(REF, i)]["obj"]) for i in ii if (REF, i) in old and (REF, i) in new]
        if tf:
            r_t = np.array([a / b for a, b in tf])
            same = np.mean([abs(a - b) <= 1e-6 * abs(b) for a, b in ro]) * 100 if ro else float("nan")
            res["timing_check"] = dict(n=len(tf), full_time_here=float(np.mean([a for a, _ in tf])),
                                       full_time_hybrid_study=float(np.mean([b for _, b in tf])),
                                       ratio_median=float(np.median(r_t)), ratio_mean=float(r_t.mean()),
                                       within_10pct=float(np.mean(np.abs(r_t - 1) <= 0.1) * 100),
                                       ref_rule_same_objective_pct=float(same))
            lines.append(f"\nTiming check against the hybrid study's test run on the same {len(tf)} instances: full MILP "
                         f"{np.mean([a for a, _ in tf]):.1f} s here vs {np.mean([b for _, b in tf]):.1f} s there (per-instance "
                         f"ratio median {np.median(r_t):.2f}, {np.mean(np.abs(r_t - 1) <= 0.1) * 100:.0f} % within 10 %), so the "
                         f"stored times are not reused: every speed-up is paired against the full MILP solved here, on the same "
                         f"core, back to back. The re-run reference rule reaches the same objective as in the hybrid study on "
                         f"{same:.0f} % of the instances.")
    first = [r for r in recs if r["i"] < 60]
    test_block(lines, res, "test_first60", "First 60 instances, every rule", first, d)
    rules_all = {r["rule"] for r in recs if r["i"] >= 60}
    if rules_all:
        test_block(lines, res, "test_all", "All instances evaluated with the reference and the selected rule",
                   [r for r in recs if r["rule"] in rules_all], d)


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    res = {"labels": label_stats()}
    tr = json.load(open(os.path.join(RES, "m1x_train.json"))) if os.path.exists(os.path.join(RES, "m1x_train.json")) else {}
    vf = valfix_summary()
    res["train"], res["valfix"] = tr, {k: {kk: vv for kk, vv in v.items() if kk != "gaps"} for k, v in vf.items()}
    lines = ["# m1x: better Model 1 probabilities (data scaling, deep ensembles, temporal mixing) — results\n",
             "Generated by `scripts/uc_m1x_report.py`; method and discussion: [`docs/methods/m1x.md`](../../docs/methods/m1x.md).\n"]
    L = res["labels"]
    if "pilot" in L:
        lines.append("## Labels\n\nPolishing pilot (24 training instances, out-of-fold BCE probabilities as the teacher; gap of the "
                     "label to the training MILP objective, used only to choose the setting):\n")
        rows = [[k, fmt(v.get("time_mean_s"), 2), fmt(v["gap_median_%"], 3), fmt(v["gap_mean_%"], 2), v.get("n_gt_1%", "–"),
                 fmt(v.get("served_%"), 1), v.get("hit_limit", "–")] for k, v in L["pilot"].items()]
        lines.append(table(["setting", "s / label", "gap median %", "gap mean %", "# > 1 %", "served %", "hit 5 s limit"], rows))
    if "extra" in L:
        e = L["extra"]
        lines.append(f"\nExtra scenarios: {e['n']} (training days, sampler of `otsl.ucdata.generate`, seed 111); LP relaxation "
                     f"{e['t_relax_mean']:.2f} s, relaxation + label-free label + its dispatch LP {e['t_label_free_mean']:.2f} s per "
                     f"instance ({e['core_s'] / 3600:.2f} core-h).")
    if "polish" in L:
        p = L["polish"]
        lines.append(f"\nTeacher-polished labels ({p['n']} instances, {p['cfg']}): {p['time_mean']:.2f} s per label "
                     f"({p['core_s'] / 3600:.2f} core-h), reduced MILP hit the limit on {p['hit_limit']:.0f} %, label from the "
                     f"reduced MILP on {p['from_milp']:.0f} % (else the label-free label), served (no shedding / shortfall) "
                     f"{p['served']:.0f} %, cost vs the label-free label median {p['cost_vs_lf_median']:+.2f} % (mean "
                     f"{p['cost_vs_lf_mean']:+.2f} %), {p['unit_hours_changed']:.2f} % of unit-hours changed; the label differs from the "
                     f"teacher's rounded prediction on {p['unit_hours_vs_teacher']:.2f} % of unit-hours.")
    if tr:
        lines.append("\n## Validation: probability quality (360 instances: val + val_extra + val_extra2)\n\n"
                     "Against MILP labels aligned to the prediction inside identical-unit groups; *fixable @P*: largest share of "
                     "unit-hours that can be fixed, most confident first, with precision ≥ P (pooled). Guarded rule: error-cost "
                     "ranking (harm ensemble s0) at 95 % + adequacy + min up/down rows + LP-relaxation guard, reduced MILPs on "
                     "the first 60 instances of val_extra2, gap to the stored validation MILP.\n")
        rows = []
        lab = lambda r: {"lf": "label-free", "milp": "MILP"}.get(r.get("label"), r.get("label")) if r.get("label") != "pol" else \
            ("MILP" if r.get("n") == 500 else "500 MILP + polished")
        for tag, r in tr.items():
            v = r["val"]["all360"]
            f = vf.get(f"{tag}@0.95")
            gr = (f"{fmt(f['gap_mean'], 3)} / {fmt(f['gap_median'], 3)} / {f['n_gt1']} / {fmt(f['fixed'], 1)} / {fmt(f['time'], 2)}"
                  if f else "–")
            ep = r.get("epochs", "–")
            kind = r.get("kind") if not str(r.get("kind")).startswith("ens:") else f"ensemble of {len(r['kind'].split(','))}"
            rows.append([tag, lab(r), r.get("n"), kind, ep,
                         fmt(r["train_s"] / 60, 1) if r.get("train_s") else "–"] + [fmt(v[k], p) for k, _, p in MET] + [gr])
        lines.append(table(["source", "labels", "n train", "model", "epochs", "train min"] + [h for _, h, _ in MET] +
                           ["guarded 95 %: gap mean / median / # > 1 % / fixed % / s"], rows))
        vref = valfix_raw("ref_bce_s0")
        if vref:
            lines.append("\nGuarded validation rule, paired against the reference model (ref_bce_s0 = uc_model1_4.pt) on the same 60 "
                         "instances (instance bootstrap 95 % CI):\n")
            rows, res["valfix_paired"] = [], {}
            for tag in tr:
                v = valfix_raw(tag)
                if not v or tag == "ref_bce_s0" or len(v) != len(vref):
                    continue
                dg = np.array([v[i]["gap"] - vref[i]["gap"] for i in sorted(vref)])
                dfx = np.array([v[i]["fixed"] - vref[i]["fixed"] for i in sorted(vref)]) * 100
                res["valfix_paired"][tag] = dict(gap=float(dg.mean()), gap_ci=boot(dg), fixed=float(dfx.mean()), fixed_ci=boot(dfx))
                rows.append([tag, f"{dg.mean():+.3f} [{boot(dg)[0]:+.3f}, {boot(dg)[1]:+.3f}]",
                             f"{dfx.mean():+.2f} [{boot(dfx)[0]:+.2f}, {boot(dfx)[1]:+.2f}]",
                             f"{np.mean([x['released_lp'] for x in v.values()]):.1f}"])
            lines.append(table(["source", "Δ gap, pp", "Δ fixed share, pp", "fixings released by the LP guard (mean)"], rows))
        dis = [(t, r["val"]["all360"]) for t, r in tr.items() if "fix999_dis1" in r["val"]["all360"]]
        if dis:
            lines.append("\nEnsemble disagreement as an extra filter (rank by |p̄ − 0.5| − k·std over members):\n")
            lines.append(table(["ensemble", "fixable @99 % (k = 0 / 1 / 2)", "fixable @99.9 % (k = 0 / 1 / 2)"],
                               [[t, f"{fmt(v['fix99'], 3)} / {fmt(v['fix99_dis1'], 3)} / {fmt(v['fix99_dis2'], 3)}",
                                 f"{fmt(v['fix999'], 3)} / {fmt(v['fix999_dis1'], 3)} / {fmt(v['fix999_dis2'], 3)}"] for t, v in dis]))
    tunes = sorted(glob.glob(os.path.join(RES, "m1x_tune_*.json")))
    if tunes:
        lines.append("\n## Validation: Learning to Fix tuning (guard-aware, eps = 1 %, 360 instances)\n")
        ref = json.load(open(os.path.join(RES, "hybrid_tune_he_bce_s0_e1_n360.json")))
        rows = [["reference: he_bce_s0_e1_n360", ref["score"], "–", fmt(ref["val_fixed_share"] * 100, 2),
                 fmt(ref["val_fixed_share_pre_guard"] * 100, 2), ref["n_iter"], fmt(ref["wall_s"] / 3600, 2) + " (warm)"]]
        res["tune"] = {}
        for p in tunes:
            r = json.load(open(p))
            res["tune"][r["run"]] = {k: r[k] for k in ("score", "eps", "guards", "converged", "val_fixed_share",
                                                        "val_fixed_share_pre_guard", "n_iter", "wall_s", "verify_max")}
            rows.append([r["run"], r["score"], r["converged"], fmt(r["val_fixed_share"] * 100, 2),
                         fmt(r["val_fixed_share_pre_guard"] * 100, 2), r["n_iter"], fmt(r["wall_s"] / 3600, 2)])
        lines.append(table(["run", "score", "converged", "val fixed % after guards", "before guards", "iterations", "wall h"], rows))
    test_section(lines, res)
    open(os.path.join(RES, "m1x_results.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(RES, "m1x_results.json"), "w"), indent=1, default=float)
    print("\n".join(lines))
