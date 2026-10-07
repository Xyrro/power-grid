"""Robustness study, follow-up: LP-relaxation-based rules and fixes on the existing shifted and in-distribution sets.

    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_followup.py --stage retime     # timing check (>= 10 instances)
    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_followup.py --stage val        # veto variant on val (60)
    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_followup.py --stage test       # all 7 sets, resumable

New rules (nothing tuned on shifted data):
  LtF LP-relaxation eps=1%   Learning to Fix's joint tuning with pi = the instance's LP-relaxation values (thresholds of
                             the baselines study, results/uc12/base_tune_lp_1.json, tuned on 180 validation instances);
                             eq. (5) on the instance's own relaxation, no guards (as tuned)
  hybrid + LP guard          the hybrid's fixings and tuned guards (adequacy, rows), then the LP-relaxation guard
  LtF BCE + LP guard         faithful LtF on the BCE GNN, then min up/down row release and the LP-relaxation guard
  LtF kNN + LP guard         faithful LtF kNN, then row release and the LP-relaxation guard
  <rule> + LP veto           the rule's fixings, then the OFF-fixing veto chosen on the original validation set (stage
                             val): release OFF fixings the instance's LP relaxation contradicts (otsl.ood.lp_off_veto);
                             applied to the hybrid, LtF kNN and LtF on the LP relaxation
Outages: the same overrides as the main run (probabilities of unavailable units already 0 in the stored instance
files; fixings of unavailable units forced OFF last; guards on the available fleet).

Timing. The full MILP and the base rules are not re-solved: their times are the main run's (results/uc12/
ood_eval.jsonl), after `--stage retime` re-solved >= 10 of them on the same core and confirmed agreement within 10 %.
A new rule whose final fixings equal its base rule's (guard / veto released nothing) has the same reduced MILP; its
record reuses the base rule's solve (objective, time) and adds the newly measured guard / veto time ("reused": true).
The base fixings are rebuilt from the stored probabilities and checked against the stored fixed shares.
Every other reduced MILP is solved here (60 s, 0.1 %, highspy, one thread). Method time = reduced MILP + inference +
LP relaxation (stored relax time of the instance, for every rule that reads the relaxation) + guards / veto.
Output: results/uc12/ood_fu_retime.json, ood_fu_val.jsonl, ood_fu_select.json, ood_fu_eval.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import harm_from_file, harm_scores  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import apply_guards, harm_to_score  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.ood import (OODModel, OODScenario, avail_view, fixed_share, force_out, lp_off_veto,  # noqa: E402
                      rarely_on_units, solve_milp)
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

ROOT, OUTD, OUT = "data/generated/uc12", "data/generated/uc12_ood", "results/uc12"
TL, GAP = 60.0, 1e-3
ORDER = ("id", "load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
VETO_TOL, RARE_FREQ = 1e-3, 0.01          # fixed a priori: LP numerical tolerance; "rarely on" = < 1 % of training unit-hours


def thr(name):
    r = json.load(open(os.path.join(OUT, name)))
    return np.array(r["lo"]), np.array(r["hi"])


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


def variants(rare):
    """the two veto variants tried (fixed before any run): name -> function(fix, u_rel)"""
    return {"unit-hour veto": lambda fix, u: lp_off_veto(fix, u, VETO_TOL, None, False),
            "rarely-on unit veto": lambda fix, u: lp_off_veto(fix, u, VETO_TOL, rare, True)}


def main_records():
    by = {}
    for line in open(os.path.join(OUT, "ood_eval.jsonl")):
        r = json.loads(line)
        by[(r["shift"], r["k"], r["rule"])] = r
    return by


def load_inst(sh, k):
    x = dict(np.load(os.path.join(OUTD, sh, f"{k:03d}.npz")))
    g_out = tuple(int(g) for g in x["g_out"])
    sc = OODScenario(load=x["load"], avail=x["avail"], u0=x["u0"], sr=x["sr"], g_out=g_out,
                     l_out=tuple(int(l) for l in x["l_out"]), start=int(x["start"]))
    di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], u_rel=x["u_rel"][None],
              lmp_rel=x["lmp_rel"][None], flow_rel=x["flow_rel"][None])
    return x, sc, di


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["retime", "val", "test"], required=True)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--n_retime", type=int, default=2, help="instances per set for the timing check")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    S = load_rts_gmlc()
    M = OODModel(S, T=12)
    T, G = 12, S.G
    HARM = harm_from_file(os.path.join(OUT, "combo_harm_s0.pt"))
    FF = FixFeaturizer(S, 12)
    HNORM = json.load(open(os.path.join(OUT, "hybrid_probs.json")))["harm_norm_logh_mean_sd"]["bce_s0"]
    TH = {"knn": thr("ltfx_tune_knn_1.json"), "bce": thr("ltfx_tune_bce_1.json"),
          "hyb": thr("hybrid_tune_he_bce_s0_e1_n360.json"), "lp": thr("base_tune_lp_1.json")}
    RARE = rarely_on_units(load(os.path.join(ROOT, "train.npz"))["u"], RARE_FREQ)
    VAR = variants(RARE)

    def base_fix(name, P, H, sc, sv):
        """pre-guard fixings and the tuned guards of a base rule"""
        if name == "hyb":
            return fix_dict(harm_to_score(H, P["bce"], *HNORM), *TH["hyb"]), ("adeq", "rows")
        if name == "knn":
            return fix_dict(P["knn"], *TH["knn"]), ()
        if name == "bce":
            return fix_dict(P["bce"], *TH["bce"]), ()
        if name == "lp":
            return fix_dict(P["lp"], *TH["lp"]), ()
        raise ValueError(name)

    if a.stage == "retime":
        by = main_records()
        picks = []
        for sh in ORDER:
            ks = sorted(k for (s_, k, ru) in by if s_ == sh and ru == "full MILP" and 5.0 <= by[(s_, k, ru)]["time"] <= 55.0)
            picks += [(sh, k) for k in ks[:a.n_retime]]
        out = []
        for sh, k in picks:
            x, sc, di = load_inst(sh, k)
            sv = avail_view(S, sc.g_out)
            full = solve_milp(M, sc, {}, TL, GAP)
            P = {"bce": x["p_bce"]}
            H = harm_scores(HARM, FF, P["bce"][None], di, [0])[0]
            f0, gu = base_fix("hyb", P, H, sc, sv)
            f1, _ = apply_guards(M, sv, sc, f0, gu)
            f1 = force_out(f1, sc.g_out, T)
            hyb = solve_milp(M, sc, f1, TL, GAP)
            o_full, o_hyb = by[(sh, k, "full MILP")], by[(sh, k, "hybrid eps=1%")]
            rec = dict(shift=sh, k=k, full_old=o_full["time"], full_new=full["time"], full_obj_old=o_full["obj"],
                       full_obj_new=full["obj"], hyb_old=o_hyb["time"], hyb_new=hyb["time"], hyb_obj_old=o_hyb["obj"],
                       hyb_obj_new=hyb["obj"], hyb_fixed_match=abs(fixed_share(f1, sc.g_out, T, G) - o_hyb["fixed_share"]) < 1e-9)
            out.append(rec)
            print(f"{sh} k={k} full {o_full['time']:.1f} -> {full['time']:.1f}s, hybrid {o_hyb['time']:.1f} -> "
                  f"{hyb['time']:.1f}s, fixings match {rec['hyb_fixed_match']}", flush=True)
        rf = np.array([r["full_new"] / r["full_old"] for r in out])
        rh = np.array([r["hyb_new"] / r["hyb_old"] for r in out])
        summ = dict(n=len(out), full_ratio_median=float(np.median(rf)), full_ratio_mean=float(rf.mean()),
                    full_within_10pct=float((np.abs(rf - 1) <= 0.1).mean() * 100),
                    full_ratio_of_means=float(sum(r["full_new"] for r in out) / sum(r["full_old"] for r in out)),
                    hyb_ratio_median=float(np.median(rh)), hyb_ratio_mean=float(rh.mean()),
                    hyb_within_10pct=float((np.abs(rh - 1) <= 0.1).mean() * 100),
                    objectives_identical_full=int(sum(abs(r["full_obj_new"] - r["full_obj_old"]) <= 1e-6 * abs(r["full_obj_old"]) for r in out)),
                    objectives_identical_hyb=int(sum(abs(r["hyb_obj_new"] - r["hyb_obj_old"]) <= 1e-6 * abs(r["hyb_obj_old"]) for r in out)),
                    fixings_match=int(sum(r["hyb_fixed_match"] for r in out)))
        summ["agree_within_10pct"] = bool(abs(summ["full_ratio_median"] - 1) <= 0.1 and abs(summ["full_ratio_of_means"] - 1) <= 0.1)
        json.dump({"summary": summ, "instances": out}, open(os.path.join(OUT, "ood_fu_retime.json"), "w"), indent=1)
        print(json.dumps(summ, indent=1))

    elif a.stage == "val":
        # veto variant chosen on the original validation set (val.npz, 60 instances); rule fixed before the run:
        # lowest mean (over the hybrid and LtF-kNN) validation gap to the dataset MILP's dual bound; within 0.02 pp the
        # faster (higher mean log speed-up of the reduced MILP + veto time against the unvetoed rule)
        d = load(os.path.join(ROOT, "val.npz"))
        Pz = np.load(os.path.join(OUT, "hybrid_probs.npz"))
        path = os.path.join(OUT, "ood_fu_val.jsonl")
        done = {json.loads(l)["i"] for l in open(path)} if os.path.exists(path) else set()
        for i in range(min(a.n_val, len(d["load"]))):
            if i in done:
                continue
            sc = scenario_from(d, i)
            P = {"knn": Pz["knn_va"][i], "bce": Pz["bce_s0_va"][i].astype(np.float64)}
            H = Pz["harm_bce_s0_va"][i]
            DB = float(d["obj"][i] * (1 - d["gap"][i]))
            recs = []
            for name in ("hyb", "knn"):
                f0, gu = base_fix(name, P, H, sc, S)
                if gu:
                    f0, _ = apply_guards(M, S, sc, f0, gu)
                base = solve_milp(M, sc, f0, TL, GAP)
                base.pop("u", None)
                recs.append(dict(i=i, rule=name, variant="none", n_fixed=len(f0), released=0, veto_s=0.0, DB=DB, **base))
                for vn, vf in VAR.items():
                    (f1, rel), dt = timed(vf, f0, d["u_rel"][i])
                    if rel == 0:
                        r = {kk: v for kk, v in base.items()}
                        r["reused"] = True
                    else:
                        r = solve_milp(M, sc, f1, TL, GAP)
                        r.pop("u", None)
                        r["reused"] = False
                    recs.append(dict(i=i, rule=name, variant=vn, n_fixed=len(f1), released=rel, veto_s=dt, DB=DB, **r))
            with open(path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, default=float) + "\n")
            print(f"val {i}: " + ", ".join(f"{r['rule']}/{r['variant']} rel {r['released']} t {r['time']:.1f}" for r in recs), flush=True)
        R = [json.loads(l) for l in open(path)]
        sel = {}
        for vn in VAR:
            g, ls = [], []
            for name in ("hyb", "knn"):
                base = {r["i"]: r for r in R if r["rule"] == name and r["variant"] == "none"}
                for r in R:
                    if r["rule"] == name and r["variant"] == vn and r["feasible"] and base[r["i"]]["feasible"]:
                        g.append((r["obj"] - r["DB"]) / r["obj"] * 100)
                        ls.append(np.log(base[r["i"]]["time"] / (r["time"] + r["veto_s"])))
            sel[vn] = dict(gap_mean=float(np.mean(g)), logsp_vs_base=float(np.mean(ls)),
                           released_mean=float(np.mean([r["released"] for r in R if r["variant"] == vn])))
        for name in ("hyb", "knn"):
            g = [(r["obj"] - r["DB"]) / r["obj"] * 100 for r in R if r["rule"] == name and r["variant"] == "none" and r["feasible"]]
            sel[f"base {name}"] = dict(gap_mean=float(np.mean(g)))
        best = min(VAR, key=lambda v: sel[v]["gap_mean"])
        close = [v for v in VAR if sel[v]["gap_mean"] <= sel[best]["gap_mean"] + 0.02]
        chosen = max(close, key=lambda v: sel[v]["logsp_vs_base"])
        out = dict(chosen=chosen, stats=sel, n=len({r["i"] for r in R}), rarely_on_units=[int(g) for g in RARE],
                   rule="lowest mean validation gap over hybrid and LtF-kNN; within 0.02 pp the faster", tol=VETO_TOL,
                   rare_freq=RARE_FREQ)
        json.dump(out, open(os.path.join(OUT, "ood_fu_select.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))

    else:
        by = main_records()
        chosen = json.load(open(os.path.join(OUT, "ood_fu_select.json")))["chosen"]
        veto = VAR[chosen]
        path = os.path.join(OUT, "ood_fu_eval.jsonl")
        done = set()
        if os.path.exists(path):
            done = {(json.loads(l)["shift"], json.loads(l)["k"]) for l in open(path)}
        n_of = {sh: len([1 for (s_, k, ru) in by if s_ == sh and ru == "full MILP"]) for sh in ORDER}
        jobs = [(sh, k) for k in range(max(n_of.values())) for sh in ORDER if k < n_of[sh] and (sh, k) not in done]
        print(f"{len(jobs)} instances to run", flush=True)
        t_start = time.time()
        for j, (sh, k) in enumerate(jobs):
            x, sc, di = load_inst(sh, k)
            g_out = sc.g_out
            sv = avail_view(S, g_out)
            P = {"knn": x["p_knn"], "bce": x["p_bce"], "lp": x["u_rel"]}
            H = harm_scores(HARM, FF, P["bce"][None], di, [0])[0]
            stored = {nm: by[(sh, k, ru)] for nm, ru in (("hyb", "hybrid eps=1%"), ("knn", "LtF kNN eps=1%"),
                                                          ("bce", "LtF BCE eps=1%"))}
            relax_s = stored["hyb"]["relax_s"]
            base = dict(shift=sh, k=k, g_out=list(g_out), l_out=list(sc.l_out))
            recs = []
            final = {}
            for nm in ("hyb", "knn", "bce", "lp"):
                (f0, gu), dt = timed(base_fix, nm, P, H, sc, sv)
                if gu:
                    (f1, _), dg = timed(apply_guards, M, sv, sc, f0, gu)
                else:
                    f1, dg = f0, 0.0
                final[nm] = (f0, force_out(f1, g_out, T), dt, dg)
                if nm in stored:
                    fs = fixed_share(final[nm][1], g_out, T, G)
                    assert abs(fs - stored[nm]["fixed_share"]) < 1e-9, (sh, k, nm, fs, stored[nm]["fixed_share"])

            def emit(rule, fix, base_nm, pre_s, uses_rel, guard_s, info):
                same = base_nm is not None and fix == final[base_nm][1]
                if same:
                    s0 = stored[base_nm]
                    r = {kk: s0[kk] for kk in ("feasible", "obj", "time", "status", "mip_gap", "shed", "short", "spill", "bound")
                         if kk in s0}
                    r["reused"] = True
                else:
                    r = solve_milp(M, sc, fix, TL, GAP)
                    r.pop("u", None)
                    r["reused"] = False
                recs.append(dict(base, rule=rule, base_rule=base_nm, fixed_share=fixed_share(fix, g_out, T, G),
                                 pre_s=pre_s, relax_s=relax_s if uses_rel else 0.0, guard_s=guard_s, **info, **r))

            # 1. LtF on the LP relaxation (faithful: no guards)
            f0, f1, dt, _ = final["lp"]
            emit("LtF LP-relaxation eps=1%", f1, None, dt, True, 0.0, {})
            # 2. LP-relaxation guard after the tuned fixings
            for nm, rule, gu in (("hyb", "hybrid + LP guard", ("adeq", "rows", "lp")),
                                 ("bce", "LtF BCE + LP guard", ("rows", "lp")), ("knn", "LtF kNN + LP guard", ("rows", "lp"))):
                f0 = final[nm][0]
                (f2, info), dg = timed(apply_guards, M, sv, sc, f0, gu)
                info = {kk: v for kk, v in info.items() if kk != "guard_s"}
                emit(rule, force_out(f2, g_out, T), nm, stored[nm]["pre_s"], nm != "knn", dg, info)
            # 3. LP veto on OFF fixings (variant chosen on validation)
            for nm, rule in (("hyb", "hybrid + LP veto"), ("knn", "LtF kNN + LP veto"), ("lp", "LtF LP-relaxation + LP veto")):
                f_in = final[nm][1]
                (f2, rel), dv = timed(veto, f_in, x["u_rel"])
                f2 = force_out(f2, g_out, T)
                if nm == "lp":
                    pre, gs, bn = final["lp"][2], dv, None
                    if f2 == f_in:                       # nothing released: same reduced MILP as the plain LP rule
                        lp_rec = [r for r in recs if r["rule"] == "LtF LP-relaxation eps=1%"][0]
                        recs.append(dict(lp_rec, rule=rule, guard_s=dv, released_veto=0, reused=True))
                        continue
                else:
                    pre, gs, bn = stored[nm]["pre_s"], stored[nm]["guard_s"] + dv, nm
                emit(rule, f2, bn, pre, True, gs, {"released_veto": rel})
            with open(path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, default=float) + "\n")
            print(f"[{j + 1}/{len(jobs)} {time.time() - t_start:.0f}s] {sh} k={k} solved "
                  f"{sum(not r['reused'] for r in recs)}/{len(recs)}, infeasible {[r['rule'] for r in recs if not r['feasible']]}",
                  flush=True)
