"""Robustness study, step 2: every shifted instance's full MILP and every method back to back, on one core.

    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_eval.py --n 40

Instances: the in-distribution reference ("id": the first n instances of test_fresh, stored scenarios) and n
instances of each shift of scripts/uc_ood_gen.py (data/generated/uc12_ood/specs.json). Jobs run round-robin over the
shifts (instance k of every shift before instance k + 1), so a partial run is balanced; finished instances are
skipped on restart (resume).

Per instance, in one process: full MILP (60 s, 0.1 %, HiGHS one thread) -> LP relaxation (timed; input of the GNNs,
the error-cost model and the no-learning baseline) -> probabilities (each model timed) -> every rule's fixings,
guards, reduced MILP (60 s, 0.1 %) -> end-to-end decoders and their dispatch LPs. Nothing was tuned on shifted data:
thresholds, targets and models are the ones fixed on the original validation set.

Rules (names as in the records):
  LtF kNN eps=1%            faithful Learning to Fix, kNN (k = 50, Table II features), results/uc12/ltfx_tune_knn_1.json
  LtF BCE eps=1%            faithful Learning to Fix on the MILP-label BCE GNN (uc_model1_4.pt), ltfx_tune_bce_1.json
  hybrid eps=1%             guard-aware LtF on error-cost scores (BCE GNN, harm seed 0), hybrid_tune_he_bce_s0_e1_n360.json,
                            adequacy guard + min up/down row release
  guarded error-cost 90%    error-cost ranking + adequacy guard, 90 % (BCE GNN, harm seed 0)
  guarded error-cost 95%    error-cost ranking + adequacy guard + row release + LP-relaxation guard, 95 % target
  combined 98%              self-trained GNN (combo_st_s0.pt) + error cost + adequacy + rows + LP guard, 98 % target
  no learning               fix every unit-hour that is integral (0 or 1) in the LP relaxation; adequacy guard + rows
  e2e RL 1 LP / screen      label-free + REINFORCE GNN (constrained_rl_lf.pt): threshold 0.5 (one LP) or the 7
                            thresholds 0.3-0.9 (best by the exact LP); block adequacy + min up/down repair; no MILP
  e2e lag 1 LP / screen     combined end-to-end pipeline (constrained_lag_D.pt, val threshold 0.6; screening as above)
  line_out only: "... (GNN sees topology)" = hybrid and guarded 90 % with the BCE GNN's messages over the outaged
                 lines switched off (edge gate 0)
Outages: unavailable units are forced off in every LP / MILP (otsl.ood.OODModel); the learned predictors do not see
availability, so their on-probabilities of unavailable units are overridden to 0 and their fixings forced OFF after
the guards; the solver-free guards / repairs use the available fleet (otsl.ood.avail_view).
Output: results/uc12/ood_eval.jsonl (one record per instance and rule), data/generated/uc12_ood/<shift>/<k>.npz
(the instance, its full-MILP solution, LP relaxation and every model's probabilities).
Recovery test (scripts/uc_ood_finetune.py): --bce replaces the BCE GNN checkpoint, --knn_extra adds labelled instances
to the kNN pool, --rules restricts the fixing rules (no end-to-end rows), --out / --inst_dir redirect the outputs.
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
from otsl.combo import THRESHOLDS, adequacy_guard, harm_from_file, harm_scores, rule_fixings  # noqa: E402
from otsl.constrained import adequacy_repair_blocks  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import apply_guards, harm_to_score  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.ood import (OODModel, OODScenario, avail_view, build_shift_scenario, fixed_share, force_out,  # noqa: E402
                      override_probs, predict_gated, scenario_to_dict, solve_milp)
from otsl.uc import load_rts_gmlc, repair_min_updown  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402

ROOT, OUTD, OUT = "data/generated/uc12", "data/generated/uc12_ood", "results/uc12"
TL, GAP = 60.0, 1e-3
ORDER = ("id", "load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
GNN_FILES = {"bce": "uc_model1_4.pt", "st": "combo_st_s0.pt", "rl": "constrained_rl_lf.pt", "lag": "constrained_lag_D.pt"}
E2E = {"RL": ("rl", 0.5), "lag": ("lag", 0.6)}      # model, single-LP threshold (lag: val-calibrated, combo study)


def scenario(shift, k, specs, tf):
    if shift == "id":
        sc = OODScenario(load=tf["load"][k], avail=tf["avail"][k], u0=tf["u0"][k], sr=tf["sr"][k],
                         day=int(tf["day"][k]), start=int(tf["start"][k]))
        return sc, {"day": int(tf["day"][k]), "start": int(tf["start"][k]), "seed": -1}
    sp = specs["shifts"][shift]
    job = sp["jobs"][k]
    return build_shift_scenario(SYS, M, sp["kw"], job, specs.get("line_pool")), job


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


def run_instance(shift, k, sc, job):
    m, s = M, SYS
    T, G = 12, s.G
    g_out = tuple(sc.g_out)
    sv = avail_view(s, g_out)
    recs = []
    base = dict(shift=shift, k=k, day=job["day"], start=job["start"], g_out=list(g_out), l_out=list(sc.l_out))
    # ---------------- full MILP (reference: objective, dual bound, time)
    full = solve_milp(m, sc, {}, TL, GAP)
    u_full = full.pop("u", None)
    recs.append(dict(base, rule="full MILP", fixed_share=0.0, pre_s=0.0, relax_s=0.0, guard_s=0.0, **full))
    # ---------------- LP relaxation (input of the GNNs / error-cost features / no-learning baseline)
    rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
    extra = {}
    if g_out or sc.l_out:                                        # how much the outage itself costs (not timed)
        intact = OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr)
        extra["c_rel_intact"] = float(m.solve_dispatch(intact, None, relax=True).obj)
    di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], start=np.array([sc.start]),
              u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
    # ---------------- probabilities (each timed), overrides for unavailable units
    P, Ti = {}, {}
    P["knn"], Ti["knn"] = timed(lambda: KNN.predict(di)[0])
    for name, m1 in GNN.items():
        P[name], Ti[name] = timed(lambda: m1.predict(di)[0].astype(np.float64))
    if sc.l_out:
        gate = np.ones((1, s.L), np.float32)
        gate[0, list(sc.l_out)] = 0.0
        P["bce_topo"], Ti["bce_topo"] = timed(lambda: predict_gated(GNN["bce"], di, gate)[0].astype(np.float64))
    P = {key: override_probs(v, g_out) for key, v in P.items()}
    H, Th = {}, {}
    for src in [x for x in ("bce", "st", "bce_topo") if x in P]:
        H[src], Th[src] = timed(lambda: harm_scores(HARM, FF, P[src][None], di, [0])[0])
    # ---------------- fixing rules: (name, fixings, pre-solve seconds, uses LP relaxation, guards)
    specs = []
    for name, (th, src) in LTF.items():
        fix, dt = timed(fix_dict, P[src], th[0], th[1])
        specs.append((name, fix, Ti[src] + dt, src != "knn", ()))
    for tag, src in (("", "bce"), (" (GNN sees topology)", "bce_topo")):
        if src not in P:
            continue
        def hyb():
            return fix_dict(harm_to_score(H[src], P[src], *HNORM), HYB[0], HYB[1])
        fix, dt = timed(hyb)
        specs.append(("hybrid eps=1%" + tag, fix, Ti[src] + Th[src] + dt, True, ("adeq", "rows")))
        fix, dt = timed(rule_fixings, "harm", 0.90, P[src], sc, sv, H[src])
        specs.append(("guarded error-cost 90%" + tag, fix, Ti[src] + Th[src] + dt, True, ()))
    fix, dt = timed(rule_fixings, "harm", 0.95, P["bce"], sc, sv, H["bce"])
    specs.append(("guarded error-cost 95%", fix, Ti["bce"] + Th["bce"] + dt, True, ("rows", "lp")))
    fix, dt = timed(rule_fixings, "harm", 0.98, P["st"], sc, sv, H["st"])
    specs.append(("combined 98%", fix, Ti["st"] + Th["st"] + dt, True, ("rows", "lp")))
    ur = rel.u
    integ = (np.abs(ur) < 1e-6) | (np.abs(ur - 1) < 1e-6)

    def nolearn():
        f = {(int(t), int(g)): int(round(ur[t, g])) for t, g in zip(*np.where(integ))}
        return adequacy_guard(f, sc.load, sc.avail, sc.sr, sv)[0]
    fix, dt = timed(nolearn)
    specs.append(("no learning: LP-integral fixings + guards", fix, dt, True, ("rows",)))
    if RULES_SEL is not None:
        specs = [x for x in specs if x[0] in RULES_SEL]
    for name, fix, pre_s, uses_rel, guards in specs:
        n_pre = fixed_share(fix, g_out, T, G)
        info, guard_s = {}, 0.0
        if guards:            # in order: adequacy, min up/down row release, LP-relaxation guard (otsl.hybrid.apply_guards)
            (fix, info), guard_s = timed(apply_guards, m, sv, sc, fix, guards)
            info = {kk: v for kk, v in info.items() if kk != "guard_s"}
        fix = force_out(fix, g_out, T)
        r = solve_milp(m, sc, fix, TL, GAP)
        u = r.pop("u", None)
        recs.append(dict(base, rule=name, fixed_share=fixed_share(fix, g_out, T, G), fixed_pre_guard=n_pre, pre_s=pre_s,
                         relax_s=relax_s if uses_rel else 0.0, guard_s=guard_s, **info, **r,
                         wrong_vs_full=int(((u != u_full).sum()) if (u is not None and u_full is not None) else -1)))
    # ---------------- end-to-end (no MILP): decode -> block adequacy repair -> min up/down repair -> dispatch LP
    for tag, (src, th1) in (E2E.items() if RULES_SEL is None else ()):
        cache, rows = {}, []
        for th in THRESHOLDS:
            t0 = time.time()
            u = (P[src] > th).astype(np.int8)
            u = adequacy_repair_blocks(u, sc.u0, sc.load, sc.avail, sc.sr, sv)
            u = repair_min_updown(u, sc.u0, s.min_up, s.min_dn)
            u[:, list(g_out)] = 0
            dec_s = time.time() - t0
            key = u.tobytes()
            if key not in cache:
                sol, lp_s = timed(m.solve_dispatch, sc, u)
                cache[key] = (sol, lp_s)
                new = True
            else:
                new = False
            sol, lp_s = cache[key]
            rows.append(dict(th=th, obj=float(sol.obj), shed=float(sol.shed), short=float(sol.short), dec_s=dec_s,
                             lp_s=lp_s if new else 0.0, lp_s_any=lp_s, units=float(u.sum() / T),
                             wrong=int((u != u_full).sum()) if u_full is not None else -1))
        one = [r for r in rows if abs(r["th"] - th1) < 1e-9][0]
        recs.append(dict(base, rule=f"e2e {tag} 1 LP", feasible=True, obj=one["obj"], shed=one["shed"], short=one["short"],
                         time=one["lp_s_any"], pre_s=Ti[src] + one["dec_s"], relax_s=relax_s, guard_s=0.0, fixed_share=1.0,
                         n_lp=1, threshold=th1, units=one["units"], wrong_vs_full=one["wrong"]))
        best = min(rows, key=lambda r: r["obj"])
        recs.append(dict(base, rule=f"e2e {tag} screen", feasible=True, obj=best["obj"], shed=best["shed"],
                         short=best["short"], time=float(sum(r["lp_s"] for r in rows)),
                         pre_s=Ti[src] + float(sum(r["dec_s"] for r in rows)), relax_s=relax_s, guard_s=0.0, fixed_share=1.0,
                         n_lp=len(cache), threshold=best["th"], units=best["units"], wrong_vs_full=best["wrong"]))
    for r in recs:
        r.update(extra)
    inst = scenario_to_dict(sc, {"u_full": u_full if u_full is not None else np.zeros((T, G), np.int8),
                                 "obj_full": full["obj"], "bound_full": full["bound"], "time_full": full["time"],
                                 "u_rel": rel.u, "lmp_rel": rel.lmp, "flow_rel": rel.flow, "c_rel": rel.obj,
                                 **{f"p_{key}": v for key, v in P.items()}})
    return recs, inst


def load_thr(name):
    r = json.load(open(os.path.join(OUT, name)))
    return np.array(r["lo"]), np.array(r["hi"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--shifts", default=",".join(ORDER))
    ap.add_argument("--out", default=os.path.join(OUT, "ood_eval.jsonl"))
    ap.add_argument("--rules", default="", help="comma-separated subset of fixing rules (default: all rules + end-to-end)")
    ap.add_argument("--bce", default="", help="replacement checkpoint for the BCE GNN (recovery test)")
    ap.add_argument("--knn_extra", default="", help="labelled instances (.npz) added to the kNN pool (recovery test)")
    ap.add_argument("--inst_dir", default=OUTD, help="where the per-instance .npz files go")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    SYS = load_rts_gmlc()
    M = OODModel(SYS, T=12)
    tr = load(os.path.join(ROOT, "train.npz"))
    tf = load(os.path.join(ROOT, "test_fresh.npz"))
    specs = json.load(open(os.path.join(OUTD, "specs.json")))
    trs = strip(tr)
    RULES_SEL = set(x for x in a.rules.split(",") if x) or None
    files = dict(GNN_FILES, **({"bce": os.path.relpath(a.bce, OUT)} if a.bce else {}))
    GNN = {k: load_model1(os.path.join(OUT, f), SYS, trs, 12) for k, f in files.items()}
    pool = tr
    if a.knn_extra:
        ex = load(a.knn_extra)
        pool = {key: np.concatenate([tr[key], ex[key]]) for key in ("load", "avail", "u0", "sr", "start", "u")}
    KNN = KNNProb(SYS, pool, k=50)
    HARM = harm_from_file(os.path.join(OUT, "combo_harm_s0.pt"))
    FF = FixFeaturizer(SYS, 12)
    HNORM = json.load(open(os.path.join(OUT, "hybrid_probs.json")))["harm_norm_logh_mean_sd"]["bce_s0"]
    LTF = {"LtF kNN eps=1%": (load_thr("ltfx_tune_knn_1.json"), "knn"),
           "LtF BCE eps=1%": (load_thr("ltfx_tune_bce_1.json"), "bce")}
    HYB = load_thr("hybrid_tune_he_bce_s0_e1_n360.json")
    # warm-up (first-call overheads are not billed to the first instance)
    d0 = {k: v[:1] for k, v in tf.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(tf["load"])}
    for m1 in GNN.values():
        m1.predict(d0)
    KNN.predict(d0)
    harm_scores(HARM, FF, GNN["bce"].predict(d0).astype(np.float64), d0, [0])
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["shift"], r["k"]))
    shifts = [x for x in a.shifts.split(",") if x]
    jobs = [(sh, k) for k in range(a.n) for sh in shifts if (sh, k) not in done]
    print(f"{len(jobs)} instances to run ({len(done)} done)", flush=True)
    t_start = time.time()
    for j, (sh, k) in enumerate(jobs):
        sc, job = scenario(sh, k, specs, tf)
        t0 = time.time()
        recs, inst = run_instance(sh, k, sc, job)
        os.makedirs(os.path.join(a.inst_dir, sh), exist_ok=True)
        np.savez_compressed(os.path.join(a.inst_dir, sh, f"{k:03d}.npz"), **inst)
        with open(a.out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        full = recs[0]
        bad = [r["rule"] for r in recs if not r["feasible"]]
        print(f"[{j + 1}/{len(jobs)} {time.time() - t_start:.0f}s] {sh} k={k} full {full['time']:.1f}s "
              f"({full['status']}), instance {time.time() - t0:.0f}s, infeasible {bad}", flush=True)
