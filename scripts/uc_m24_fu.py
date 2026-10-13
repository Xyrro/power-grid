"""m24 follow-up: (1) calibrate the m24 rule so its held-out eps-violation rate matches learning-free LtF's, (2) the
cost-aware bound check with a full-MILP fallback of the cguard study (otsl.cguard, C3) on 24 hours.

    taskset -c 0 python3 scripts/uc_m24_fu.py --stage cv_margin            # logit margins on the 2-fold CV thresholds
    taskset -c 1 python3 scripts/uc_m24_fu.py --stage cv_eps --eps 0.005   # tighter eps, 2-fold CV (tuning per fold)
    python3 scripts/uc_m24_fu.py --stage select                            # calibration choice -> m24_fu_select.json
    taskset -c 0 python3 scripts/uc_m24_fu.py --stage val --rules m24,lp,m24cal   # rules on the 40 validation instances
    python3 scripts/uc_m24_fu.py --stage cguard                            # bound-check thresholds -> m24_fu_cguard.json
    taskset -c 1 python3 scripts/uc_m24_fu.py --stage test                 # 40 test instances, timing check vs stored records
    taskset -c 0 python3 scripts/uc_m24_fu.py --stage paired --part 0/2    # fully paired re-solve (even / odd instances)
    taskset -c 1 python3 scripts/uc_m24_fu.py --stage paired --part 1/2

Calibration (validation only; the 2-fold split, measurement and violation criterion of scripts/uc_m24_cv.py: reduced
MILP at 0.1 %, 120 s, test-time guards, violation = cost increase over C* above 1 %). Two families, each ordered from the
smallest change:
  * margin delta (nats): the thresholds of eq. (5) moved outwards in logit space, lo' = sigmoid(logit(lo) - delta),
    hi' = sigmoid(logit(hi) + delta) (lo = 0 and hi = 1 unchanged), applied to each fold's tuned thresholds
    (results/uc24/m24_cv_m24_f<k>.json); grid 0.5, 1, 1.5, 2, 3, 4, 6, evaluated upwards until the target is met;
  * tighter eps: m24's tuning (otsl.m24.M24Tuner, same settings) at eps = 0.5 %, then 0.25 %, on each fold's 20
    training instances;
target: at most 3 of the 40 held-out cases above 1 % (learning-free LtF's rate in the same CV). Within each family the
smallest change that meets it; across families the one that keeps the larger held-out fixed share. The chosen change is
then applied to thresholds tuned on all 40 instances (margin: results/uc24/m24_tune_hg_fin.json; eps: a new run of
scripts/uc_m24_tune.py).

Bound check (C3 of docs/methods/cguard.md): g = (C_red - z_LP) / C_red with z_LP the LP relaxation solved as model
input; if g > d (or the reduced problem is infeasible), the full MILP at 0.5 % is solved and the better solution kept.
d per rule: the smallest value of the grid 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10 % (cguard's grid 1-10 %, extended
downwards because 24-hour relaxation gaps are small: median 0.26 % on validation) that fires on at most 5 % of the 40
validation instances, from the rule's reduced MILP at 0.5 % (300 s) on them.

Test, stage "test": per instance in one process on core 1 (the core of the stored records
results/uc24/m24_eval_test.jsonl): LP relaxation, ensemble inference, the original m24 fixings and the calibrated ones,
guards, reduced MILP at 0.5 % (300 s). The original pipeline reproduces its stored objectives, but in 1.35-1.65x the
time on instances 0-25 (stored before a container restart, on a faster machine) and 0.9-1.0x on 26-39, so the stored
full-MILP times cannot serve as denominators. Stage "paired" therefore re-solves everything per instance back to back
in one process (full MILP at 0.1 % with its incumbent log and at 0.5 %, LtF-LP at 0.5 %, original and calibrated m24 at
0.5 %), even instances on core 0 and odd ones on core 1. Output: results/uc24/m24_fu_*.json[l].
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import load_b3  # noqa: E402
from otsl.base import solve_reduced  # noqa: E402
from otsl.ltfx import fix_dict, fix_masks  # noqa: E402
from otsl.m24 import M24Tuner, guard_chain  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

U24, DATA, RES = "data/generated/uc24", "data/generated/uc24_m24", "results/uc24"
DELTAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EPSS = (0.005, 0.0025)
TARGET = 3
GRID = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)
MAX_RATE = 0.05
TG = 24 * 73


def folds(n=40, seed=0):
    p = np.random.default_rng(seed).permutation(n)
    return [np.sort(p[:n // 2]), np.sort(p[n // 2:])]


def margin(lo, hi, delta):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if delta == 0:
        return lo, hi
    lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    sg = lambda z: 1 / (1 + np.exp(-z))
    lo2 = np.where(lo <= 0, lo, sg(lg(lo) - delta))
    hi2 = np.where(hi >= 1, hi, sg(lg(hi) + delta))
    return lo2, hi2


def sc_of(d, i):
    return UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])


def done_keys(path, keys):
    out = set()
    if os.path.exists(path):
        for x in open(path):
            r = json.loads(x)
            out.add(tuple(r[k] for k in keys))
    return out


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def measure_cv(path, tag, k, idx, lo, hi, m, sysm, d, pi, done, tl=120.0):
    """the CV measurement of scripts/uc_m24_cv.py (m24 guards incl. the LP guard, reduced MILP 0.1 %)"""
    for i in idx:
        if (tag, k, int(i)) in done:
            continue
        sc = sc_of(d, i)
        fx = fix_dict(pi[i], lo, hi)
        n_pre = len(fx)
        fx, info = guard_chain(m, sysm, sc, fx, lp=True)
        r = solve_reduced(m, sc, fx, tl, 1e-3)
        c_star = float(d["obj"][i])
        rec = dict(tag=tag, fold=k, i=int(i), feasible=r["feasible"], obj=r["obj"], c_star=c_star,
                   rel_inc=(r["obj"] - c_star) / c_star if r["feasible"] else None, n_fixed_pre=n_pre, n_fixed=len(fx),
                   time=r["time"], status=r["status"], **{kk: v for kk, v in info.items() if kk.startswith("released")})
        append(path, rec)
        done.add((tag, k, int(i)))
        ri = rec["rel_inc"]
        print(f"{tag} f{k} i={i}: " + (f"{ri * 100:+.3f}%" if ri is not None else "INFEASIBLE") +
              f" fixed {len(fx) / TG * 100:.1f}% {r['time']:.1f}s", flush=True)


def n_viol(path, tag):
    rr = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    rr = [r for r in rr if r["tag"] == tag]
    v = sum(1 for r in rr if not r["feasible"] or r["rel_inc"] > 0.01)
    return v, len(rr), (float(np.mean([r["n_fixed"] for r in rr]) / TG) if rr else None)


def load_rule(name, sel=None):
    """(score name, lo, hi) of a rule's 40-instance thresholds"""
    if name == "lp":
        b = json.load(open(os.path.join(RES, "base_tune_lp_1.json")))
        return "lp", np.array(b["lo"]), np.array(b["hi"])
    t = json.load(open(os.path.join(RES, "m24_tune_hg_fin.json")))
    lo, hi = np.array(t["lo"]), np.array(t["hi"])
    if name == "m24":
        return "m24", lo, hi
    sel = sel or json.load(open(os.path.join(RES, "m24_fu_select.json")))
    if sel["family"] == "margin":
        lo, hi = margin(lo, hi, sel["param"])
    else:
        t = json.load(open(os.path.join(RES, f"m24_tune_{sel['run']}.json")))
        assert t["converged"], sel["run"]
        lo, hi = np.array(t["lo"]), np.array(t["hi"])
    return "m24", lo, hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cv_margin", "cv_eps", "select", "val", "cguard", "test", "paired"])
    ap.add_argument("--part", default="0/1", help="paired stage: k/n = instances with i % n == k")
    ap.add_argument("--reverse", action="store_true", help="paired stage: take the part's instances from the end, "
                    "write to m24_fu_paired_<k>r.jsonl and skip instances any paired file already has (load balancing)")
    ap.add_argument("--eps", default="0.005,0.0025")
    ap.add_argument("--rules", default="m24,lp,m24cal")
    ap.add_argument("--budget_min", type=float, default=60.0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=24, network=True)
    dv = load_b3(os.path.join(U24, "uc24ltf_val.npz"))
    PI = {"m24": np.load(os.path.join(DATA, "probs", "fin_ens_probs.npz"))["val"].astype(np.float64),
          "lp": dv["u_rel"].astype(np.float64)}
    F = folds()
    P_MAR, P_EPS = os.path.join(RES, "m24_fu_cv_margin.jsonl"), os.path.join(RES, "m24_fu_cv_eps.jsonl")

    if a.stage == "cv_margin":
        done = done_keys(P_MAR, ("tag", "fold", "i"))
        th = [json.load(open(os.path.join(RES, f"m24_cv_m24_f{k}.json"))) for k in (0, 1)]
        for delta in DELTAS:
            tag = f"margin{delta:g}"
            for k in (0, 1):
                lo, hi = margin(th[k]["lo"], th[k]["hi"], delta)
                measure_cv(P_MAR, tag, k, F[k], lo, hi, m, sysm, dv, PI["m24"], done)
            v, n, fx = n_viol(P_MAR, tag)
            print(f"== {tag}: {v} / {n} held-out violations, fixed {fx * 100:.1f}%", flush=True)
            if v <= TARGET:
                break

    if a.stage == "cv_eps":
        done = done_keys(P_EPS, ("tag", "fold", "i"))
        for eps in [float(x) for x in a.eps.split(",")]:
            tag = f"eps{eps * 100:g}"
            for k in (0, 1):
                out = os.path.join(RES, f"m24_fu_cv_{tag}_f{k}.json")
                tr = F[1 - k]
                if not os.path.exists(out):
                    sub = {kk: v[tr] for kk, v in dv.items()}
                    logf = open(out.replace(".json", ".log"), "w")
                    log = lambda s: (logf.write(s + "\n"), logf.flush())
                    log(f"m24 fu cv: eps {eps} fold {k} (tuned on {tr.tolist()})")
                    t0 = time.time()
                    tu = M24Tuner(m, sysm, sub, PI["m24"][tr], eps, K_max=10, Q=20, check_tl=60.0, relax_tl=8.0,
                                  master_tl=120.0, log=log, time_budget=a.budget_min * 60, start_tl=15.0)
                    lo, hi = tu.run()
                    conv = tu.history[-1].get("result") == "all instances feasible"
                    po, pn = tu.guarded_masks(lo, hi)
                    json.dump(dict(eps=eps, fold=k, train_idx=tr.tolist(), converged=conv, lo=lo.tolist(), hi=hi.tolist(),
                                   fixed_share=float((po | pn).mean()), n_iter=len(tu.history), wall_s=time.time() - t0),
                              open(out, "w"), indent=1)
                    log(f"done: converged={conv} fixed {float((po | pn).mean()) * 100:.2f}% wall {time.time() - t0:.0f}s")
                    print(f"tuned {tag} fold {k}: converged={conv} fixed {float((po | pn).mean()) * 100:.2f}% "
                          f"wall {time.time() - t0:.0f}s", flush=True)
                t = json.load(open(out))
                measure_cv(P_EPS, tag, k, F[k], np.array(t["lo"]), np.array(t["hi"]), m, sysm, dv, PI["m24"], done)
            v, n, fx = n_viol(P_EPS, tag)
            print(f"== {tag}: {v} / {n} held-out violations, fixed {fx * 100:.1f}%", flush=True)
            if v <= TARGET:
                break

    if a.stage == "select":
        cv = [json.loads(x) for x in open(os.path.join(RES, "m24_cv.jsonl"))]
        ref = sum(1 for r in cv if r["kind"] == "heldout" and r["score"] == "lp" and (not r["feasible"] or r["rel_inc"] > 0.01))
        assert ref == TARGET, ref
        fam = {}
        for name, path, params in (("margin", P_MAR, DELTAS), ("eps", P_EPS, EPSS)):
            rows = []
            for p in params:
                tag = f"margin{p:g}" if name == "margin" else f"eps{p * 100:g}"
                v, n, fx = n_viol(path, tag)
                if n == 0:
                    continue
                rows.append(dict(param=p, tag=tag, violations=v, n=n, heldout_fixed=fx))
            ok = [r for r in rows if r["n"] == 40 and r["violations"] <= TARGET]
            fam[name] = dict(grid=rows, chosen=ok[0] if ok else None)
        cands = [(f, v["chosen"]) for f, v in fam.items() if v["chosen"]]
        assert cands, "no candidate meets the target"
        f_, c = max(cands, key=lambda x: x[1]["heldout_fixed"])
        sel = dict(family=f_, param=c["param"], tag=c["tag"], heldout_violations=c["violations"],
                   heldout_fixed=c["heldout_fixed"], target=TARGET, families=fam,
                   rule="per family the smallest change with <= 3 / 40 held-out violations; across families the larger "
                        "held-out fixed share")
        if f_ == "eps":
            sel["run"] = f"hg_fin_e{c['param'] * 1000:g}"
        json.dump(sel, open(os.path.join(RES, "m24_fu_select.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in sel.items() if k != "families"}), flush=True)
        for f_, v in fam.items():
            print(f_, [(r["tag"], r["violations"], round(r["heldout_fixed"] * 100, 1)) for r in v["grid"]], flush=True)

    if a.stage == "val":
        path = os.path.join(RES, "m24_fu_val.jsonl")
        done = done_keys(path, ("rule", "i"))
        for rule in [x for x in a.rules.split(",") if x]:
            score, lo, hi = load_rule(rule)
            for i in range(40):
                if (rule, i) in done:
                    continue
                sc = sc_of(dv, i)
                fx = fix_dict(PI[score][i], lo, hi)
                info = {}
                if score == "m24":
                    fx, info = guard_chain(m, sysm, sc, fx, lp=True)
                r = solve_reduced(m, sc, fx, 300.0, 5e-3)
                z = float(dv["c_rel"][i])
                g = (r["obj"] - z) / r["obj"] if r["feasible"] else np.inf
                rec = dict(rule=rule, i=i, feasible=r["feasible"], obj=r["obj"], z_lp=z, g=g, c_star=float(dv["obj"][i]),
                           n_fixed=len(fx), time=r["time"], status=r["status"])
                append(path, rec)
                print(f"val {rule} i={i}: g {g * 100:.3f}% vs C* {((r['obj'] or np.nan) / rec['c_star'] - 1) * 100:+.3f}% "
                      f"fixed {len(fx) / TG * 100:.1f}% {r['time']:.1f}s", flush=True)

    if a.stage == "cguard":
        V = [json.loads(x) for x in open(os.path.join(RES, "m24_fu_val.jsonl"))]
        out = {}
        for rule in sorted({r["rule"] for r in V}):
            g = np.array([r["g"] if r["feasible"] else np.inf for r in V if r["rule"] == rule])
            assert len(g) == 40, (rule, len(g))
            rate = {str(d): float((g > d).mean()) for d in GRID}
            d = next((d for d in GRID if (g > d).mean() <= MAX_RATE), GRID[-1])
            out[rule] = dict(d_post=d, rate=rate, n=len(g), g_median=float(np.median(g)), g_p90=float(np.percentile(g, 90)))
            print(rule, "d_post", d, "fires on", f"{(g > d).mean() * 100:.1f}%", flush=True)
        json.dump(dict(rules=out, grid=GRID, max_rate=MAX_RATE,
                       rule="smallest grid value firing on at most 5 % of the 40 validation instances (cguard C3)"),
                  open(os.path.join(RES, "m24_fu_cguard.json"), "w"), indent=1)

    if a.stage == "paired":
        # fully paired pass: the stored records of results/uc24/m24_eval_test.jsonl came from a container that ran
        # ~1.5x faster (instances 0-25; the "test" stage re-solves the original m24 pipeline with identical objectives
        # in 1.35-1.65x the time), so every comparator is re-solved here, per instance back to back in one process
        from uc_m24_train import featurizer, load_member
        k_, n_ = (int(x) for x in a.part.split("/"))
        path = os.path.join(RES, f"m24_fu_paired_{k_}{'r' if a.reverse else ''}.jsonl")

        def done_any():
            out = set()
            for q in glob.glob(os.path.join(RES, "m24_fu_paired_*.jsonl")):
                out |= {json.loads(x)["i"] for x in open(q)}
            return out
        d = load_b3(os.path.join(U24, "test.npz"))
        feat = featurizer(sysm)
        members = [load_member(p, sysm, feat) for p in sorted(glob.glob(os.path.join(DATA, "models", "fin_s*.pt")))]
        assert len(members) == 5
        rules = {"m24": load_rule("m24")[1:], "m24cal": load_rule("m24cal")[1:]}
        _, lo_lp, hi_lp = load_rule("lp")
        done = {json.loads(x)["i"] for x in open(path)} if os.path.exists(path) else set()
        one = {k: d[k][:1] for k in ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel")}
        for m1 in members:
            m1.predict(one)
        todo = [i for i in range(40) if i % n_ == k_ and i not in done]
        if a.reverse:
            todo = [i for i in todo[::-1] if i not in done_any()]
        print(f"paired part {k_}/{n_}: {len(todo)} instances on cpu {sorted(os.sched_getaffinity(0))}", flush=True)
        t_start = time.time()
        for n, i in enumerate(todo):
            if a.reverse and i in done_any():
                print(f"i={i} done by the other process; stopping", flush=True)
                break
            sc = sc_of(d, i)
            rec = dict(i=i, loadavg0=os.getloadavg()[0], cpu=sorted(os.sched_getaffinity(0)))
            t0 = time.time()
            rel = m.solve_dispatch(sc, None, relax=True)
            t_rel = time.time() - t0
            rec.update(t_rel=t_rel, c_rel=float(rel.obj))
            runs = {}
            for nm, gap in (("full", 1e-3), ("full gap=0.5%", 5e-3)):
                r = {k: v for k, v in solve_reduced(m, sc, None, 300.0, gap).items() if k != "trace"}
                r.update(pre_s=0.0, t_rel=0.0, t_guard=0.0, t_method=r["time"], mip_rel_gap=gap)
                runs[nm] = r
            t0 = time.time()
            fx = fix_dict(rel.u, lo_lp, hi_lp)
            dt = time.time() - t0
            r = {k: v for k, v in solve_reduced(m, sc, fx, 300.0, 5e-3).items() if k != "trace"}
            r.update(pre_s=dt, t_rel=t_rel, t_guard=0.0, n_fixed_pre=len(fx), mip_rel_gap=5e-3)
            r["t_method"] = r["time"] + r["pre_s"] + r["t_rel"]
            runs["LtF-LP gap=0.5%"] = r
            di = {kk: d[kk][i:i + 1] for kk in ("load", "avail", "u0", "sr")}
            di.update(u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
            t0 = time.time()
            p = np.mean([m1.predict(di)[0].astype(np.float64) for m1 in members], 0)
            t_inf = time.time() - t0
            for nm, (lo, hi) in rules.items():
                t0 = time.time()
                fx = fix_dict(p, lo, hi)
                dt = time.time() - t0
                fx2, info = guard_chain(m, sysm, sc, fx, lp=True)
                r = {k: v for k, v in solve_reduced(m, sc, fx2, 300.0, 5e-3).items() if k != "trace"}
                r.update(pre_s=t_inf + dt, t_rel=t_rel, t_guard=info["guard_s"], n_fixed_pre=len(fx), mip_rel_gap=5e-3,
                         **{kk: v for kk, v in info.items() if kk.startswith("released")})
                r["t_method"] = r["time"] + r["pre_s"] + r["t_rel"] + r["t_guard"]
                runs[nm] = r
            rec["runs"] = runs
            rec["loadavg1"] = os.getloadavg()[0]
            append(path, rec)
            full = runs["full"]
            print(f"[{n + 1}/{len(todo)} {time.time() - t_start:.0f}s] i={i} full {full['time']:.1f}s | " + " | ".join(
                f"{nm}: " + (f"{(r['obj'] - full['obj']) / full['obj'] * 100:+.2f}% {r['t_method']:.1f}s" if r["feasible"] else "INF")
                for nm, r in runs.items() if nm != "full"), flush=True)
        print("done", flush=True)

    if a.stage == "test":
        from uc_m24_train import featurizer, load_member
        path = os.path.join(RES, "m24_fu_test.jsonl")
        d = load_b3(os.path.join(U24, "test.npz"))
        feat = featurizer(sysm)
        members = [load_member(p, sysm, feat) for p in sorted(glob.glob(os.path.join(DATA, "models", "fin_s*.pt")))]
        assert len(members) == 5
        rules = {"m24": load_rule("m24")[1:], "m24cal": load_rule("m24cal")[1:]}
        done = {json.loads(x)["i"] for x in open(path)} if os.path.exists(path) else set()
        one = {k: d[k][:1] for k in ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel")}
        for m1 in members:
            m1.predict(one)
        t_start = time.time()
        todo = [i for i in range(40) if i not in done]
        print(f"test: {len(todo)} instances; calibration {json.load(open(os.path.join(RES, 'm24_fu_select.json')))['tag']}",
              flush=True)
        for n, i in enumerate(todo):
            sc = sc_of(d, i)
            rec = dict(i=i, loadavg0=os.getloadavg()[0], cpu=sorted(os.sched_getaffinity(0)))
            t0 = time.time()
            rel = m.solve_dispatch(sc, None, relax=True)
            t_rel = time.time() - t0
            rec.update(t_rel=t_rel, c_rel=float(rel.obj))
            di = {kk: d[kk][i:i + 1] for kk in ("load", "avail", "u0", "sr")}
            di.update(u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
            t0 = time.time()
            p = np.mean([m1.predict(di)[0].astype(np.float64) for m1 in members], 0)
            t_inf = time.time() - t0
            runs = {}
            for nm, (lo, hi) in rules.items():
                t0 = time.time()
                fx = fix_dict(p, lo, hi)
                dt = time.time() - t0
                fx2, info = guard_chain(m, sysm, sc, fx, lp=True)
                r = solve_reduced(m, sc, fx2, 300.0, 5e-3)
                r = {k: v for k, v in r.items() if k != "trace"}
                r.update(pre_s=t_inf + dt, t_rel=t_rel, t_guard=info["guard_s"], n_fixed_pre=len(fx), mip_rel_gap=5e-3,
                         **{kk: v for kk, v in info.items() if kk.startswith("released")})
                r["t_method"] = r["time"] + r["pre_s"] + r["t_rel"] + r["t_guard"]
                runs[nm] = r
            rec["runs"] = runs
            rec["loadavg1"] = os.getloadavg()[0]
            append(path, rec)
            print(f"[{n + 1}/{len(todo)} {time.time() - t_start:.0f}s] i={i} " + " | ".join(
                f"{nm}: {r['obj'] if r['feasible'] else 'INF'} {r['t_method']:.1f}s fixed {r['n_fixed'] / TG * 100:.1f}%"
                for nm, r in runs.items()), flush=True)
        print("done", flush=True)
