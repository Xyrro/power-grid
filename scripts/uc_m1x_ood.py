"""m1x study, robustness: the selected rule B-hg (Learning to Fix's tuning on the probabilities of the 5-member temporal
GNN ensemble, results/uc12/m1x_tune_hg_pol_n2000_gnnt_ens5.json) on the shifted 12-hour test sets of the robustness
study (data/generated/uc12_ood/<set>/<k>.npz, built by scripts/uc_ood_gen.py / uc_ood_eval.py), with and without the
post-hoc LP-relaxation guard. Nothing is tuned here: thresholds, models and guards are the ones fixed on validation.

    OTSL_THREADS=1 taskset -c 3 python3 scripts/uc_m1x_ood.py --retime --ks 0,1     # timing check (14 instances)
    OTSL_THREADS=1 taskset -c 3 python3 scripts/uc_m1x_ood.py [--retime]             # every instance (resume)
    python3 scripts/uc_m1x_ood.py --report                                           # results/uc12/m1x_ood.{md,json}

Outage handling exactly as in scripts/uc_ood_eval.py (otsl.ood, imported read-only): the UC model forces unavailable
units off and removes outaged lines (OODModel), the GNN's on-probabilities of unavailable units are overridden to 0,
the adequacy guard / row release / LP guard get the available fleet (avail_view), and fixings of unavailable units are
forced OFF after the guards; the fixed share counts available units only. MILPs through otsl.ood.solve_milp (highspy,
one thread, 60 s, 0.1 %), as for the robustness study's records.
Per instance: [full MILP if --retime] -> LP relaxation (timed; GNN input) -> 5 forward passes (timed) -> eq. (5) on the
mean probability -> guards (timed) -> reduced MILP. Rules: "m1x B-hg" (adequacy + rows, as tuned) and
"m1x B-hg + LP guard" (adequacy + rows + LP-relaxation guard).
Output: results/uc12/m1x_ood_eval.jsonl (one record per instance and rule; resume skips finished instances).
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
from otsl.hybrid import apply_guards  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.ood import OODModel, OODScenario, avail_view, fixed_share, force_out, override_probs, solve_milp  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from uc_m1x_train import featurizer, load_net  # noqa: E402

OODD, RES = "data/generated/uc12_ood", "results/uc12"
SETS = ("id", "load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
N = {"id": 40}
NAMES = {"id": "in-distribution (test_fresh 0-39)", "load_hi": "load +15 %", "load_lo": "load −15 %",
         "ren_hi": "wind + solar ×1.5", "gen_out": "2-3 thermal units out", "line_out": "1-2 lines out",
         "night": "windows across midnight"}
SRC, RUN = "pol_n2000_gnnt_ens5", "hg_pol_n2000_gnnt_ens5"
RULES = {"m1x B-hg": ("adeq", "rows"), "m1x B-hg + LP guard": ("adeq", "rows", "lp")}
OUT = os.path.join(RES, "m1x_ood_eval.jsonl")


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


def instance(shift, k):
    z = np.load(os.path.join(OODD, shift, f"{k:03d}.npz"))
    sc = OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"], day=int(z["day"]), start=int(z["start"]),
                     g_out=tuple(int(g) for g in z["g_out"]), l_out=tuple(int(x) for x in z["l_out"]))
    return sc, z


def run(a):
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    feat = featurizer(sysm)
    reg = json.load(open(os.path.join(RES, "m1x_sources.json")))
    find = lambda pth: pth if os.path.exists(pth) else os.path.join(RES, "m1x_" + os.path.basename(pth))
    members = [load_net(find(p), kind, sysm, feat) for p, kind in reg[SRC]]
    tune = json.load(open(os.path.join(RES, f"m1x_tune_{RUN}.json")))
    lo, hi = np.array(tune["lo"]), np.array(tune["hi"])
    T, G = 12, sysm.G
    done = set()
    if os.path.exists(a.out):
        done = {(r["shift"], r["k"]) for r in map(json.loads, open(a.out))}
    ks = [int(x) for x in a.ks.split(",") if x]
    sets = [s for s in a.sets.split(",") if s]
    jobs = [(s, k) for k in range(max(N.values()) if not ks else max(ks) + 1) for s in sets
            if k < N.get(s, 30) and (not ks or k in ks) and (s, k) not in done]
    sc0, _ = instance(sets[0], 0)                                          # warm-up (first-call overheads)
    r0 = m.solve_dispatch(sc0, None, relax=True)
    d0 = dict(load=sc0.load[None], avail=sc0.avail[None], u0=sc0.u0[None], sr=sc0.sr[None], u_rel=r0.u[None],
              lmp_rel=r0.lmp[None], flow_rel=r0.flow[None])
    for m1 in members:
        m1.predict(d0)
    print(f"{len(jobs)} instances to run ({len(done)} done)", flush=True)
    t_start = time.time()
    for j, (shift, k) in enumerate(jobs):
        sc, z = instance(shift, k)
        g_out = tuple(sc.g_out)
        sv = avail_view(sysm, g_out)
        base = dict(shift=shift, k=k, g_out=list(g_out), l_out=list(sc.l_out))
        recs = []
        if a.retime:
            full = solve_milp(m, sc, {}, 60.0, 1e-3)
            full.pop("u", None)
            recs.append(dict(base, rule="full MILP (core 3)", fixed_share=0.0, pre_s=0.0, relax_s=0.0, guard_s=0.0, **full))
        rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
        di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], u_rel=rel.u[None],
                  lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
        p, t_inf = timed(lambda: np.mean([m1.predict(di)[0].astype(np.float64) for m1 in members], 0))
        p = override_probs(p, g_out)
        fix0, t_fix = timed(fix_dict, p, lo, hi)
        for name, guards in RULES.items():
            (fix, info), guard_s = timed(apply_guards, m, sv, sc, dict(fix0), guards)
            info = {kk: v for kk, v in info.items() if kk != "guard_s"}
            fix = force_out(fix, g_out, T)
            r = solve_milp(m, sc, fix, 60.0, 1e-3)
            r.pop("u", None)
            recs.append(dict(base, rule=name, fixed_share=fixed_share(fix, g_out, T, G),
                             fixed_pre_guard=fixed_share(fix0, g_out, T, G), pre_s=t_inf + t_fix, relax_s=relax_s,
                             guard_s=guard_s, **info, **r))
        with open(a.out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        print(f"[{j + 1}/{len(jobs)} {time.time() - t_start:.0f}s] {shift} k={k} " +
              ", ".join(f"{r['rule'].replace('m1x ', '')} {r['time']:.1f}s{'' if r['feasible'] else ' INFEASIBLE'}"
                        for r in recs), flush=True)


# ============================================================================ report
NB = 2000


def boot(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    b = x[np.random.default_rng(seed).integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def arrays(R, F, Ft, ks):
    """per-instance metrics of one rule (the robustness study's definitions, scripts/uc_ood_report.per_rule): gap to
    the dual bound DB of the robustness study's full MILP (the same DB for every rule), speed-up against the full-MILP
    time Ft (the re-timed one for the m1x rules when the timing check failed, else the stored one)"""
    feas = np.array([bool(R[k]["feasible"]) for k in ks])
    cost = np.array([R[k]["obj"] for k in ks], float)
    DB = np.array([F[k]["bound"] if np.isfinite(F[k]["bound"]) else F[k]["obj"] * (1 - (F[k]["mip_gap"] or 0)) for k in ks])
    C_full = np.array([F[k]["obj"] for k in ks])
    t_full = np.array([Ft[k]["time"] for k in ks])
    t_m = np.array([R[k]["time"] + R[k]["pre_s"] + R[k]["relax_s"] + R[k]["guard_s"] for k in ks])
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(feas, (cost - DB) / cost * 100, np.nan)
        sp = np.where(feas, t_full / t_m, np.nan)
    slack = np.array([(R[k].get("shed", 0) or 0) + (R[k].get("spill", 0) or 0) + (R[k].get("short", 0) or 0) for k in ks])
    slack_f = np.array([(F[k].get("shed", 0) or 0) + (F[k].get("spill", 0) or 0) + (F[k].get("short", 0) or 0) for k in ks])
    c_fb = np.where(feas, cost, C_full)
    return dict(feas=feas, gap=gap, sp=sp, fixed=np.array([R[k].get("fixed_share", 0) for k in ks]) * 100,
                served=np.where(feas, slack < 1e-6, slack_f < 1e-6), g_ref=(c_fb - C_full) / C_full * 100)


def stats(x):
    f = x["feas"]
    g, sp = x["gap"][f], x["sp"][f]
    return dict(n=int(len(f)), feasible=float(f.mean() * 100), gap_mean=float(g.mean()), gap_ci=boot(g),
                gap_median=float(np.median(g)), gap_max=float(g.max()), sp_mean=float(sp.mean()), sp_ci=boot(sp),
                sp_median=float(np.median(sp)), fixed=float(x["fixed"][f].mean()), served=float(x["served"].mean() * 100),
                n_gt10=int((x["g_ref"] > 10).sum()))


def paired(a, b):
    both = a["feas"] & b["feas"]
    dg = (a["gap"] - b["gap"])[both]
    dl = (np.log(a["sp"]) - np.log(b["sp"]))[both]
    dsp = (a["sp"] - b["sp"])[both]
    df = (a["fixed"] - b["fixed"])[both]
    return dict(n=int(both.sum()), d_gap=float(dg.mean()), d_gap_ci=boot(dg), d_logsp=float(dl.mean()), d_logsp_ci=boot(dl),
                d_sp=float(dsp.mean()), d_sp_ci=boot(dsp), d_fixed=float(df.mean()), d_fixed_ci=boot(df))


def report():
    ood = {}
    for r in map(json.loads, open(os.path.join(RES, "ood_eval.jsonl"))):
        ood.setdefault((r["shift"], r["rule"]), {})[r["k"]] = r
    mine = {}
    for r in map(json.loads, open(OUT)):
        mine.setdefault((r["shift"], r["rule"]), {})[r["k"]] = r
    # ---------------- timing check: full MILP re-timed on core 3 vs the robustness study's stored time
    pairs = [(sh, k, r["time"], ood[(sh, "full MILP")][k]["time"], r["obj"], ood[(sh, "full MILP")][k]["obj"])
             for (sh, rule), rr in mine.items() if rule == "full MILP (core 3)" for k, r in rr.items()]
    rt = np.array([p[2] / p[3] for p in pairs]) if pairs else np.array([])
    tc = dict(n=len(pairs), ratio_median=float(np.median(rt)) if len(rt) else None,
              ratio_mean=float(rt.mean()) if len(rt) else None,
              ratio_of_means=float(sum(p[2] for p in pairs) / sum(p[3] for p in pairs)) if pairs else None,
              within_10pct=float(np.mean(np.abs(rt - 1) <= 0.1) * 100) if len(rt) else None,
              same_objective=int(sum(abs(p[4] - p[5]) <= 1e-6 * abs(p[5]) for p in pairs)))
    stored_ok = bool(len(rt) >= 10 and abs(tc["ratio_median"] - 1) <= 0.1 and abs(tc["ratio_of_means"] - 1) <= 0.1)
    tc["use_stored_times"] = stored_ok
    res = dict(timing_check=tc, sets={})
    L = ["# m1x selected rule (B-hg) under distribution shift: results", "",
         "Generated by `scripts/uc_m1x_ood.py --report`; method: [`docs/methods/m1x.md`](../../docs/methods/m1x.md) (section "
         "*Robustness under distribution shift*); shifted sets and the other rules: [`docs/methods/ood.md`](../../docs/methods/ood.md), "
         "`results/uc12/ood_eval.jsonl` (read only).", "",
         "B-hg = Learning to Fix's joint ε = 1 % tuning on the mean probability of 5 temporal GNNs (500 MILP + 1,500 polished "
         "labels), adequacy guard + min up/down row release (as tuned); + LP guard = the LP-relaxation guard added at test time. "
         "Outage handling as in the robustness study. Gap = (C − DB) / C to the dual bound of the robustness study's full MILP "
         "(the same DB for every rule), feasible instances; speed-up = per-instance T_full / T_method with every overhead "
         "(LP relaxation, 5 forward passes, guards).", ""]
    L.append(f"**Timing check** ({tc['n']} full MILPs re-solved on core 3 vs the stored times of the robustness study): "
             f"per-instance ratio median {tc['ratio_median']:.2f}, ratio of means {tc['ratio_of_means']:.2f}, "
             f"{tc['within_10pct']:.0f} % within 10 %, identical objective on {tc['same_objective']} / {tc['n']}. "
             + ("Agreement within 10 %: the stored full-MILP times are the reference for the m1x rules."
                if stored_ok else "No agreement within 10 %: every full MILP was re-timed on core 3 and the m1x speed-ups use "
                                  "those times (the other rules keep their own, paired times)."))
    L += ["", "| set | rule | n | feasible % | gap mean % [95 % CI] | gap max % | speed-up mean [95 % CI] | median | "
          "fixed % | served % | # > 10 % |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    prow = ["", "Paired against the robustness study's records on the same instances (rule − reference, instances both "
            "solve; bootstrap 95 % CI):", "",
            "| set | rule | reference | n | Δ gap pp | Δ log speed-up | Δ mean speed-up | Δ fixed pp |",
            "|---|---|---|---|---|---|---|---|"]
    for sh in SETS:
        F = ood[(sh, "full MILP")]
        Ft_m = mine.get((sh, "full MILP (core 3)"), {})
        ks = sorted(k for k in F if all(k in mine.get((sh, r), {}) for r in RULES))
        if not stored_ok:
            ks = [k for k in ks if k in Ft_m]
        if not ks:
            continue
        X = {}
        for rule in ("LtF kNN eps=1%", "hybrid eps=1%"):
            X[rule] = arrays(ood[(sh, rule)], F, F, ks)
        for rule in RULES:
            X[rule] = arrays(mine[(sh, rule)], F, F if stored_ok else Ft_m, ks)
        res["sets"][sh] = dict(n=len(ks), ks=ks, stats={r: stats(x) for r, x in X.items()}, paired={})
        for r in ("LtF kNN eps=1%", "hybrid eps=1%") + tuple(RULES):
            s = res["sets"][sh]["stats"][r]
            L.append(f"| {NAMES[sh]} | {r} | {s['n']} | {s['feasible']:.0f} | {s['gap_mean']:.2f} [{s['gap_ci'][0]:.2f}, "
                     f"{s['gap_ci'][1]:.2f}] | {s['gap_max']:.2f} | {s['sp_mean']:.1f} [{s['sp_ci'][0]:.1f}, {s['sp_ci'][1]:.1f}] | "
                     f"{s['sp_median']:.1f} | {s['fixed']:.1f} | {s['served']:.0f} | {s['n_gt10']} |")
        for r in RULES:
            for ref in ("hybrid eps=1%", "LtF kNN eps=1%"):
                p = paired(X[r], X[ref])
                res["sets"][sh]["paired"][f"{r} vs {ref}"] = p
                c = lambda v, ci, q=2: f"{v:+.{q}f} [{ci[0]:+.{q}f}, {ci[1]:+.{q}f}]"
                prow.append(f"| {NAMES[sh]} | {r} | {ref} | {p['n']} | {c(p['d_gap'], p['d_gap_ci'])} | "
                            f"{c(p['d_logsp'], p['d_logsp_ci'])} | {c(p['d_sp'], p['d_sp_ci'], 1)} | {c(p['d_fixed'], p['d_fixed_ci'], 1)} |")
    L += prow
    open(os.path.join(RES, "m1x_ood.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(RES, "m1x_ood.json"), "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retime", action="store_true", help="also re-solve (and time) the full MILP on this core")
    ap.add_argument("--ks", default="", help="only these instance indices of every set (timing check)")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.report:
        report()
    else:
        run(a)
