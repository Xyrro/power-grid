"""No-learning baselines and fair solver budgets, test evaluation: every run of an instance back to back in one process
on one core (paired timing).

    python3 scripts/uc_base_eval.py --bench uc12 --idx 0-59 --highs_path <dir with highspy 1.12 = scipy's HiGHS>
    python3 scripts/uc_base_eval.py --bench uc24 --idx 0-19

Per test instance:
  1. LP relaxation (timed; input of every LP-based baseline and of the GNN-based reference rules);
  2. full MILP at the reference settings (uc12: 60 s, 0.1 %; uc24: 300 s, 0.1 %) with incumbent log and a sampled
     (primal, dual) trace: re-timed reference T_MILP and the anytime curve of the same-budget comparison;
  3. full MILP with mip_rel_gap = 0.25 / 0.5 / 1 % (same time limit);
  4. reference rules, re-run from their saved thresholds and probabilities (fixings checked against the stored
     records): uc12 faithful LtF kNN / BCE GNN eps = 1 %, hybrid (he_bce_s0_e1_n360), error-cost + adequacy 90 %;
     uc24 faithful LtF kNN eps = 1 %, guarded 95 % rule;
  5. no-learning baselines: LP-integral fixing (val-selected tolerance) without and with our guards, Learning to Fix
     on the LP-relaxation values (results/<bench>/base_tune_lp_1.json), LP rounding + block repair + dispatch LPs.
Reduced MILPs use the full MILP's settings and solver (uc12: highspy 1.12 = the HiGHS of scipy 1.17 used by the
reference runs, verified identical; uc24: highspy 1.15 as in uc24ltf). No fallback: a reduced problem without a
solution is infeasible. Method time = inference (kNN timed here; GNN forward pass and error-cost scoring: the stored
per-instance time of the reference run, milliseconds) + LP relaxation (rules that use it) + fixing rule + guards +
reduced MILP (or repairs + dispatch LPs). Output: results/<bench>/base_eval_test.jsonl (one line per instance;
resumable: instances already in the file are skipped).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="uc12", choices=["uc12", "uc24"])
ap.add_argument("--idx", default="0-59", help="instance range a-b (inclusive) or comma list")
ap.add_argument("--highs_path", default="")
ap.add_argument("--gaps", default="0.0025,0.005,0.01")
ap.add_argument("--skip", default="", help="comma-separated run names to leave out")
ap.add_argument("--out", default="", help="output path (default results/<bench>/base_eval_test.jsonl)")
ap.add_argument("--select", default="", help="default results/<bench>/base_val_select.json")
ap.add_argument("--tune", default="", help="default results/<bench>/base_tune_lp_1.json")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
if a.highs_path:
    sys.path.insert(0, a.highs_path)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from otsl.b3 import load_b3  # noqa: E402
from otsl.base import lp_integral_fixings, lp_round_screen, solve_reduced  # noqa: E402
from otsl.combo import rule_fixings  # noqa: E402
from otsl.fixpolicy import lp_guard, release_conflicting_rows  # noqa: E402
from otsl.hybrid import ALL_GUARDS, apply_guards, harm_to_score  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import scenario_from  # noqa: E402

CFG = {"uc12": dict(T=12, test="data/generated/uc12/test_fresh.npz", tl=60.0, gap=1e-3, trace=0.5),
       "uc24": dict(T=24, test="data/generated/uc24/test.npz", tl=300.0, gap=1e-3, trace=2.0)}


def parse_idx(s):
    if "-" in s:
        x, y = s.split("-")
        return list(range(int(x), int(y) + 1))
    return [int(x) for x in s.split(",") if x]


def jl(path):
    return [json.loads(x) for x in open(path)]


def timed(fn, *args, **kw):
    t0 = time.time()
    r = fn(*args, **kw)
    return r, time.time() - t0


def refs_uc12(sysm, d, idx):
    """{instance: [(name, fixings, pre_s, uses_lp_relaxation, guards, stored record)]}"""
    R = "results/uc12"
    stored = {}
    for r in jl(f"{R}/hybrid_eval_test_fresh.jsonl"):
        stored[(r["i"], r["rule"])] = r
    P = np.load(f"{R}/hybrid_probs.npz")
    norms = json.load(open(f"{R}/hybrid_probs.json"))["harm_norm_logh_mean_sd"]
    th = {k: json.load(open(f"{R}/{f}.json")) for k, f in
          (("knn", "ltfx_tune_knn_1"), ("bce", "ltfx_tune_bce_1"), ("hyb", "hybrid_tune_he_bce_s0_e1_n360"))}
    assert tuple(th["hyb"]["guards"]) == ("adeq", "rows"), th["hyb"]["guards"]
    knn = KNNProb(sysm, load_b3("data/generated/uc12/train.npz"), k=50)
    one = lambda i: {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}
    out = {}
    for i in idx:
        sc = scenario_from(d, i)
        pk, t_knn = timed(lambda: knn.predict(one(i))[0])
        assert np.abs(pk - P["knn_tf"][i]).max() < 1e-9
        pb, hb = P["bce_s0_tf"][i].astype(np.float64), P["harm_bce_s0_tf"][i]
        L = []
        fx, dt = timed(fix_dict, pk, np.array(th["knn"]["lo"]), np.array(th["knn"]["hi"]))
        L.append(("LtF kNN eps=1%", fx, t_knn + dt, False, (), stored[(i, "faithful LtF kNN eps=1%")]))
        s0 = stored[(i, "faithful LtF BCE eps=1%")]
        fx = fix_dict(pb, np.array(th["bce"]["lo"]), np.array(th["bce"]["hi"]))
        L.append(("LtF BCE GNN eps=1%", fx, s0["pre_s"], True, (), s0))
        s0 = stored[(i, "hybrid he_bce_s0_e1_n360")]
        fx = fix_dict(harm_to_score(hb, pb, *norms["bce_s0"]), np.array(th["hyb"]["lo"]), np.array(th["hyb"]["hi"]))
        L.append(("hybrid (he_bce_s0_e1_n360)", fx, s0["pre_s"], True, ("adeq", "rows"), s0))
        s0 = stored[(i, "ours: error-cost + adequacy guard 90% (BCE)")]
        fx = rule_fixings("harm", 0.90, pb, sc, sysm, hb)
        L.append(("ours: error-cost + adequacy 90%", fx, s0["pre_s"], True, (), s0))
        out[i] = L
    return out


def refs_uc24(sysm, d, idx):
    R = "results/uc24"
    stored = {r["i"]: r for r in jl(f"{R}/uc24ltf_eval_test.jsonl")}
    P = np.load(f"{R}/uc24ltf_probs.npz")
    th = json.load(open(f"{R}/uc24ltf_tune_knn_1.json"))
    from uc_uc24ltf_prep import load_train_lab
    from uc_b3_fix import make_specs
    knn = KNNProb(sysm, load_train_lab("data/generated/uc24/uc24ltf_train_lab.npz"), k=50)
    one = lambda i: {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}
    out = {}
    for i in idx:
        pk, t_knn = timed(lambda: knn.predict(one(i))[0])
        assert np.abs(pk - P["knn_te"][i]).max() < 1e-9
        L = []
        fx, dt = timed(fix_dict, pk, np.array(th["lo"]), np.array(th["hi"]))
        L.append(("LtF kNN eps=1%", fx, t_knn + dt, False, (), stored[i]["LtF knn eps=1%"]))
        s0 = stored[i]["old: 95%|best=bce_g"]
        (_, arr, g), = make_specs(i, d, {"bce": P["bce_te"].astype(np.float64), "rl": P["rl_te"].astype(np.float64)},
                                  sysm, ["bce_g"], [0.95])
        assert g
        fx = {(int(t), int(gg)): int(v) for t, gg, v in arr}
        L.append(("ours: guarded 95% (imitation GNN)", fx, s0["pre_s"], True, ("b3_lp", "rows"), s0))
        out[i] = L
    return out


def guard(m, sysm, sc, fix, guards):
    """the guard chain of a rule; returns (fix, info with guard_s)"""
    if not guards:
        return fix, {"guard_s": 0.0}
    if guards == ("b3_lp", "rows"):               # scripts/uc_b3_fix.py / uc_uc24ltf_eval.py kind "b3_guard"
        t0 = time.time()
        fix, r_lp, _ = lp_guard(m, sysm, sc, fix) if fix else (fix, 0, 0.0)
        fix, r_c = release_conflicting_rows(fix, sysm, sc.u0)
        return fix, {"guard_s": time.time() - t0, "released_lp": r_lp, "released_rows": r_c}
    return apply_guards(m, sysm, sc, fix, guards)


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    import highspy
    cfg = CFG[a.bench]
    print("highspy", highspy.Highs().version(), "bench", a.bench, flush=True)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=cfg["T"], network=True)
    d = load_b3(cfg["test"])
    idx = parse_idx(a.idx)
    R = os.path.join("results", a.bench)
    sel = json.load(open(a.select or os.path.join(R, "base_val_select.json")))
    tol_none, tol_g = sel["none"]["selected_tol"], sel["guards"]["selected_tol"]
    ltf_lp = json.load(open(a.tune or os.path.join(R, "base_tune_lp_1.json")))
    lo_lp, hi_lp = np.array(ltf_lp["lo"]), np.array(ltf_lp["hi"])
    gaps = [float(x) for x in a.gaps.split(",") if x]
    skip = set(x for x in a.skip.split(",") if x)
    path = a.out or os.path.join(R, "base_eval_test.jsonl")
    done = {json.loads(x)["i"] for x in open(path)} if os.path.exists(path) else set()
    idx = [i for i in idx if i not in done]
    refs = (refs_uc12 if a.bench == "uc12" else refs_uc24)(sysm, d, idx)
    TG = cfg["T"] * sysm.G
    print(f"tol (no guards) {tol_none:g}, tol (guards) {tol_g:g}; LtF-LP converged {ltf_lp['converged']}, val fixed "
          f"{ltf_lp['val_fixed_share'] * 100:.1f}%; {len(idx)} instances to run", flush=True)
    if a.dry:
        i = idx[0]
        for nm, fx, pre, rel, gd, s0 in refs[i]:
            n_st = s0.get("n_fixed_pre", round(s0.get("fixed_pre_guard", 0) * TG))
            print(f"  {nm:40s} fixed (pre-guard) {len(fx)} stored {n_st} pre {pre * 1000:.1f} ms", flush=True)
        sys.exit(0)
    t_start = time.time()
    for k, i in enumerate(idx):
        sc = scenario_from(d, i)
        rec = dict(i=i, loadavg0=os.getloadavg()[0])
        rel, t_rel = timed(m.solve_dispatch, sc, None, relax=True)
        u_rel = rel.u
        rec.update(t_rel=t_rel, c_rel=float(rel.obj), u_rel_maxdiff=float(np.abs(u_rel - d["u_rel"][i]).max()))
        runs = {}
        runs["full"] = solve_reduced(m, sc, None, cfg["tl"], cfg["gap"], trace_every=cfg["trace"])
        for g in gaps:
            nm = f"full gap={g * 100:g}%"
            if nm not in skip:
                r = solve_reduced(m, sc, None, cfg["tl"], g)
                r.pop("trace")
                runs[nm] = r
        specs = []                              # (name, fixings, pre_s, t_rel added, guards, kind, stored record)
        for nm, fx, pre, use_rel, gd, s0 in refs[i]:
            specs.append((nm, fx, pre, use_rel, gd, "ref", s0))
        fx, dt = timed(lp_integral_fixings, u_rel, tol_none)
        specs.append((f"LP-integral (tol {tol_none:g})", fx, dt, True, (), "base", None))
        fx, dt = timed(lp_integral_fixings, u_rel, tol_g)
        specs.append((f"LP-integral (tol {tol_g:g}) + guards", fx, dt, True, ALL_GUARDS, "base", None))
        fx, dt = timed(fix_dict, u_rel, lo_lp, hi_lp)
        specs.append(("LtF on LP relaxation eps=1%", fx, dt, True, (), "base", None))
        for nm, fx, pre, use_rel, gd, kind, s0 in specs:
            if nm in skip:
                continue
            n_pre = len(fx)
            fx2, ginfo = guard(m, sysm, sc, fx, gd)
            r = solve_reduced(m, sc, fx2, cfg["tl"], cfg["gap"])
            r.pop("trace")
            r.update(kind=kind, pre_s=float(pre), t_rel=float(t_rel if use_rel else 0.0), t_guard=float(ginfo["guard_s"]),
                     n_fixed_pre=n_pre, **{kk: v for kk, v in ginfo.items() if kk.startswith("released")})
            r["t_method"] = r["time"] + r["pre_s"] + r["t_rel"] + r["t_guard"]
            if s0 is not None:                  # reproducibility of the stored reference run
                r["stored_obj"] = s0["obj"]
                r["stored_time"] = s0.get("time", s0.get("t_milp"))
                r["stored_n_fixed"] = s0.get("n_fixed", round(s0.get("fixed_share", np.nan) * TG))
            runs[nm] = r
        if "LP rounding + repair + LPs" not in skip:
            e = lp_round_screen(m, sysm, sc, u_rel)
            runs["LP rounding + repair + LPs"] = dict(
                kind="e2e", feasible=True, obj=e["cost"], shed=e["shed"], short=e["short"], th=e["th"], n_lp=e["n_lp"],
                t_repair=e["t_repair"], t_lp=e["t_lp"], costs=e["costs"], t_rel=t_rel,
                t_method=t_rel + e["t_repair"] + e["t_lp"], n_fixed=TG, inc=[])
        rec["runs"] = runs
        rec["loadavg1"] = os.getloadavg()[0]
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        full = runs["full"]
        msg = " | ".join(f"{nm[:24]}: " + (f"{(r['obj'] - full['obj']) / full['obj'] * 100:+.2f}% {r['t_method'] if 't_method' in r else r['time']:.1f}s"
                                           if r["feasible"] else "INF") for nm, r in runs.items() if nm != "full")
        print(f"[{k + 1}/{len(idx)} {time.time() - t_start:.0f}s] i={i} full {full['time']:.1f}s | {msg}", flush=True)
    print("done", flush=True)
