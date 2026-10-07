"""m1x study, end-to-end test of the recommended pipeline:
    B-hg fixings (Learning to Fix's validation-tuned thresholds on the 5-member temporal-GNN ensemble,
    results/uc12/m1x_tune_hg_pol_n2000_gnnt_ens5.json) -> adequacy guard + min up/down rows (as tuned) ->
    LP-relaxation guard -> reduced MILP at mip_rel_gap 0.5 %
against, in the same process and per instance: the full MILP at the reference gap (0.1 %; speed-up denominator and dual
bound), the full MILP at 0.5 %, the hybrid at the reference gap (results/uc12/hybrid_tune_he_bce_s0_e1_n360.json), and
B-hg + LP guard at the reference gap. Nothing is tuned here.

    OTSL_THREADS=1 taskset -c N python3 scripts/uc_m1x_pipeline.py --shard i --nshards 3     # one process per core
    python3 scripts/uc_m1x_pipeline.py --report                                             # results/uc12/m1x_pipeline.{md,json}

Instances: test_fresh 0-119 (stored scenarios) and the six shifted sets of the robustness study (30 each,
data/generated/uc12_ood/<shift>/<k>.npz, read only); its in-distribution set is test_fresh 0-39 (verified identical), so
those 40 are reported from the test_fresh runs. Outages as in scripts/uc_ood_eval.py (otsl.ood): the model forces
unavailable units off and removes outaged lines, on-probabilities of unavailable units are set to 0, guards use the
available fleet, fixings of unavailable units are forced OFF after the guards, fixed share over available units.
Every MILP through otsl.ood.solve_milp (highspy, one thread, 60 s). Jobs are interleaved over the sets and split
round-robin into shards; each shard appends to results/uc12/m1x_pipeline_eval_s<i>.jsonl (resume skips finished ones).
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
from otsl.combo import harm_from_file, harm_scores  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import apply_guards, harm_to_score  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.ood import OODModel, OODScenario, avail_view, fixed_share, force_out, override_probs, solve_milp  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402
from uc_m1x_train import featurizer, load_net  # noqa: E402

ROOT, OODD, RES = "data/generated/uc12", "data/generated/uc12_ood", "results/uc12"
SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
SETS = ("tf",) + SHIFTS
NSET = {"tf": 120}
REF_GAP, LOOSE = 1e-3, 5e-3
NAMES = {"tf": "test_fresh (120)", "id": "in-distribution (test_fresh 0-39)", "load_hi": "load +15 %",
         "load_lo": "load −15 %", "ren_hi": "wind + solar ×1.5", "gen_out": "2-3 thermal units out",
         "line_out": "1-2 lines out", "night": "windows across midnight"}
R_FULL, R_FULL5, R_HYB, R_BLP, R_PIPE = ("full MILP", "full MILP gap 0.5%", "hybrid", "B-hg + LP guard",
                                         "pipeline: B-hg + LP guard, gap 0.5%")


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


def jobs_all():
    return [(s, k) for k in range(max(NSET.values())) for s in SETS if k < NSET.get(s, 30)]


def instance(s, k, tf):
    if s == "tf":
        return OODScenario(load=tf["load"][k], avail=tf["avail"][k], u0=tf["u0"][k], sr=tf["sr"][k], day=int(tf["day"][k]),
                           start=int(tf["start"][k]))
    z = np.load(os.path.join(OODD, s, f"{k:03d}.npz"))
    return OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"], day=int(z["day"]), start=int(z["start"]),
                       g_out=tuple(int(g) for g in z["g_out"]), l_out=tuple(int(x) for x in z["l_out"]))


def run(a):
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    T, G = 12, sysm.G
    tr = load(os.path.join(ROOT, "train.npz"))
    tf = load(os.path.join(ROOT, "test_fresh.npz"))
    feat = featurizer(sysm)
    reg = json.load(open(os.path.join(RES, "m1x_sources.json")))
    find = lambda p: p if os.path.exists(p) else os.path.join(RES, "m1x_" + os.path.basename(p))
    gnnt = [load_net(find(p), kind, sysm, feat) for p, kind in reg["pol_n2000_gnnt_ens5"]]
    bce = load_model1(os.path.join(RES, "uc_model1_4.pt"), sysm, strip(tr), 12)
    harm = harm_from_file(os.path.join(RES, "combo_harm_s0.pt"))
    ff = FixFeaturizer(sysm, 12)
    hnorm = json.load(open(os.path.join(RES, "hybrid_probs.json")))["harm_norm_logh_mean_sd"]["bce_s0"]
    th = lambda f: (np.array(json.load(open(os.path.join(RES, f)))["lo"]), np.array(json.load(open(os.path.join(RES, f)))["hi"]))
    hyb_lo, hyb_hi = th("hybrid_tune_he_bce_s0_e1_n360.json")
    b_lo, b_hi = th("m1x_tune_hg_pol_n2000_gnnt_ens5.json")
    out = os.path.join(RES, f"m1x_pipeline_eval_s{a.shard}.jsonl")
    done = {(r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    jobs = [j for j in jobs_all()[a.shard::a.nshards] if j not in done]
    sc0 = instance("tf", 0, tf)                                        # warm-up (first-call overheads)
    r0 = m.solve_dispatch(sc0, None, relax=True)
    d0 = dict(load=sc0.load[None], avail=sc0.avail[None], u0=sc0.u0[None], sr=sc0.sr[None], start=np.array([sc0.start]),
              u_rel=r0.u[None], lmp_rel=r0.lmp[None], flow_rel=r0.flow[None])
    for m1 in gnnt + [bce]:
        m1.predict(d0)
    harm_scores(harm, ff, bce.predict(d0).astype(np.float64), d0, [0])
    print(f"shard {a.shard}/{a.nshards} on CPU {os.sched_getaffinity(0)}: {len(jobs)} instances to run ({len(done)} done)",
          flush=True)
    t_start = time.time()
    for j, (s, k) in enumerate(jobs):
        sc = instance(s, k, tf)
        g_out = tuple(sc.g_out)
        sv = avail_view(sysm, g_out)
        base = dict(set=s, k=k, shard=a.shard, cpu=sorted(os.sched_getaffinity(0)), g_out=list(g_out), l_out=list(sc.l_out))
        recs = []
        for name, gap in ((R_FULL, REF_GAP), (R_FULL5, LOOSE)):
            r = solve_milp(m, sc, {}, 60.0, gap)
            r.pop("u", None)
            recs.append(dict(base, rule=name, mip_rel_gap=gap, fixed_share=0.0, fixed_pre_guard=0.0, pre_s=0.0, relax_s=0.0,
                             guard_s=0.0, **r))
        rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
        di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], start=np.array([sc.start]),
                  u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
        # hybrid (reference gap): BCE GNN -> error-cost scores -> thresholds -> adequacy + rows
        p_b, t_b = timed(lambda: override_probs(bce.predict(di)[0].astype(np.float64), g_out))
        h_b, t_h = timed(lambda: harm_scores(harm, ff, p_b[None], di, [0])[0])
        fix0, t_f = timed(lambda: fix_dict(harm_to_score(h_b, p_b, *hnorm), hyb_lo, hyb_hi))
        specs = [(R_HYB, fix0, t_b + t_h + t_f, ("adeq", "rows"), (REF_GAP,))]
        # B-hg: mean of 5 temporal GNNs -> thresholds -> adequacy + rows + LP-relaxation guard (computed once)
        p_g, t_g = timed(lambda: override_probs(np.mean([m1.predict(di)[0].astype(np.float64) for m1 in gnnt], 0), g_out))
        fixg, t_fg = timed(fix_dict, p_g, b_lo, b_hi)
        specs.append(("B-hg", fixg, t_g + t_fg, ("adeq", "rows", "lp"), (REF_GAP, LOOSE)))
        for name, fix_pre, pre_s, guards, gaps in specs:
            (fix, info), guard_s = timed(apply_guards, m, sv, sc, dict(fix_pre), guards)
            info = {kk: v for kk, v in info.items() if kk != "guard_s"}
            fix = force_out(fix, g_out, T)
            for gap in gaps:
                r = solve_milp(m, sc, fix, 60.0, gap)
                r.pop("u", None)
                rule = name if name == R_HYB else (R_BLP if gap == REF_GAP else R_PIPE)
                recs.append(dict(base, rule=rule, mip_rel_gap=gap, fixed_share=fixed_share(fix, g_out, T, G),
                                 fixed_pre_guard=fixed_share(fix_pre, g_out, T, G), pre_s=pre_s, relax_s=relax_s,
                                 guard_s=guard_s, **info, **r))
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        print(f"[{j + 1}/{len(jobs)} {time.time() - t_start:.0f}s] {s} k={k} " +
              ", ".join(f"{r['rule'].split(':')[0]} {r['time']:.1f}s{'' if r['feasible'] else ' INF'}" for r in recs), flush=True)


# ============================================================================ report
NB = 2000


def boot(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    b = x[np.random.default_rng(seed).integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def slack(r):
    return (r.get("shed", 0) or 0) + (r.get("spill", 0) or 0) + (r.get("short", 0) or 0)


def arrays(R, F, ks, Ft=None):
    """R: rule records by k; F: this run's reference-gap full MILP by k (dual bound for every rule, incl. stored
    LtF-kNN costs); Ft: full-MILP records whose time is the speed-up denominator (default F; a stored rule uses the
    full MILP of its own run)"""
    Ft = Ft or F
    feas = np.array([bool(R[k]["feasible"]) for k in ks])
    cost = np.array([R[k]["obj"] for k in ks], float)
    DB = np.array([F[k]["bound"] if np.isfinite(F[k]["bound"]) else F[k]["obj"] * (1 - (F[k]["mip_gap"] or 0)) for k in ks])
    t_full = np.array([Ft[k]["time"] for k in ks])
    t_m = np.array([R[k]["time"] + R[k].get("pre_s", 0) + R[k].get("relax_s", 0) + R[k].get("guard_s", 0) for k in ks])
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(feas, (cost - DB) / cost * 100, np.nan)
        sp = np.where(feas, t_full / t_m, np.nan)
    served = np.where(feas, [slack(R[k]) < 1e-6 for k in ks], [slack(F[k]) < 1e-6 for k in ks])
    return dict(feas=feas, gap=gap, sp=sp, served=served, fixed=np.array([R[k].get("fixed_share", 0) for k in ks]) * 100)


def stats(x):
    f = x["feas"]
    g, sp = x["gap"][f], x["sp"][f]
    return dict(n=int(len(f)), feasible=float(f.mean() * 100), gap_mean=float(g.mean()), gap_ci=boot(g),
                gap_max=float(g.max()), served=float(x["served"].mean() * 100), sp_mean=float(sp.mean()), sp_ci=boot(sp),
                sp_median=float(np.median(sp)), fixed=float(x["fixed"][f].mean()))


def paired(a, b):
    both = a["feas"] & b["feas"]
    dg = (a["gap"] - b["gap"])[both]
    dl = (np.log(a["sp"]) - np.log(b["sp"]))[both]
    ds = (a["served"].astype(float) - b["served"].astype(float)) * 100
    df = (a["fixed"] - b["fixed"])[both]
    return dict(n=int(both.sum()), d_gap=float(dg.mean()), d_gap_ci=boot(dg), d_logsp=float(dl.mean()), d_logsp_ci=boot(dl),
                time_ratio=float(np.exp(dl.mean())), d_served=float(ds.mean()), d_served_ci=boot(ds), d_fixed=float(df.mean()),
                d_fixed_ci=boot(df))


def report():
    mine = {}
    for path in sorted(glob.glob(os.path.join(RES, "m1x_pipeline_eval_s*.jsonl"))):
        for r in map(json.loads, open(path)):
            mine.setdefault((r["set"], r["rule"]), {})[r["k"]] = r
    ood = {}
    for r in map(json.loads, open(os.path.join(RES, "ood_eval.jsonl"))):
        ood.setdefault((r["shift"], r["rule"]), {})[r["k"]] = r
    hyb = {}
    for r in map(json.loads, open(os.path.join(RES, "hybrid_eval_test_fresh.jsonl"))):
        hyb.setdefault(r["rule"], {})[r["i"]] = r
    # stored LtF-kNN records (cost scored against this run's dual bound; speed-up against the full MILP of its own run)
    knn = {s: (ood.get((s, "LtF kNN eps=1%"), {}), ood.get((s, "full MILP"), {})) for s in SHIFTS + ("id",)}
    knn["tf"] = (hyb.get("faithful LtF kNN eps=1%", {}), hyb.get("full MILP", {}))
    # timing agreement of this run's reference full MILP with the stored references (information; nothing reused)
    agree = {}
    for s in SHIFTS + ("tf",):
        st = ood.get((s if s != "tf" else "id", "full MILP"), {})
        for k, r in mine.get((s, R_FULL), {}).items():
            if k in st:
                c = r["cpu"][0]
                agree.setdefault(c, []).append((r["time"] / st[k]["time"], abs(r["obj"] - st[k]["obj"]) <= 1e-6 * abs(st[k]["obj"])))
    tc = {str(c): dict(n=len(v), ratio_median=float(np.median([x[0] for x in v])),
                       within_10pct=float(np.mean([abs(x[0] - 1) <= 0.1 for x in v]) * 100),
                       same_objective=int(sum(x[1] for x in v))) for c, v in sorted(agree.items())}
    res = dict(timing_vs_stored=tc, sets={})
    rules = (R_FULL, R_FULL5, R_HYB, R_BLP, R_PIPE)
    L = ["# m1x recommended pipeline, end to end: results", "",
         "Generated by `scripts/uc_m1x_pipeline.py --report`; method: [`docs/methods/m1x.md`](../../docs/methods/m1x.md), section "
         "*Recommended pipeline, end to end*.", "",
         "Pipeline = B-hg fixings (validation-tuned) → adequacy guard + min up/down rows → LP-relaxation guard → reduced MILP at "
         "mip_rel_gap 0.5 %. Per instance, in one single-threaded process pinned to one core (cores 0, 1, 3): full MILP at the "
         "reference gap 0.1 % (dual bound DB and speed-up denominator), full MILP at 0.5 %, hybrid (reference gap), B-hg + LP guard "
         "(reference gap), pipeline. Gap = (C − DB) / C, feasible instances; speed-up = T_full(0.1 %) / T_method with every "
         "overhead; served = no shedding, over-generation or reserve shortfall. LtF-kNN: stored records (test_fresh 0–59 from "
         "`hybrid_eval_test_fresh.jsonl`, shifted sets from `ood_eval.jsonl`), cost scored against this run's DB, speed-up "
         "against the full MILP of its own run.", "",
         "Reference full MILP of this run vs the stored references (information only, nothing reused): " +
         "; ".join(f"core {c}: n {v['n']}, time ratio median {v['ratio_median']:.2f}, {v['within_10pct']:.0f} % within 10 %, "
                   f"identical objective {v['same_objective']} / {v['n']}" for c, v in tc.items()) + ".", "",
         "| set | rule | n | feasible % | gap mean % [95 % CI] | gap max % | served % | speed-up mean [95 % CI] | median | fixed % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    P = ["", "Paired differences (rule − reference, instances both solve; served over all instances; bootstrap 95 % CI). "
         "Time ratio = reference time / rule time, geometric mean (> 1: the rule is faster).", "",
         "| set | rule | reference | n | Δ gap pp | Δ log speed-up | time ratio | Δ served pp | Δ fixed pp |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in ("tf", "id") + SHIFTS:
        src = "tf" if s == "id" else s
        F = mine.get((src, R_FULL), {})
        ks = sorted(k for k in F if all(k in mine.get((src, r), {}) for r in rules) and (s != "id" or k < 40))
        if not ks:
            continue
        X = {r: arrays(mine[(src, r)], F, ks) for r in rules}
        kk, kf = knn[s]
        ks_knn = [k for k in ks if k in kk and k in kf]
        S = {r: stats(x) for r, x in X.items()}
        if ks_knn:
            S["LtF kNN eps=1% (stored)"] = stats(arrays(kk, F, ks_knn, kf))
        res["sets"][s] = dict(n=len(ks), stats=S, paired={})
        for r, st in S.items():
            L.append(f"| {NAMES[s]} | {r} | {st['n']} | {st['feasible']:.0f} | {st['gap_mean']:.2f} [{st['gap_ci'][0]:.2f}, "
                     f"{st['gap_ci'][1]:.2f}] | {st['gap_max']:.2f} | {st['served']:.0f} | {st['sp_mean']:.1f} "
                     f"[{st['sp_ci'][0]:.1f}, {st['sp_ci'][1]:.1f}] | {st['sp_median']:.1f} | {st['fixed']:.1f} |")
        refs = [(R_FULL5, X[R_FULL5], ks), (R_HYB, X[R_HYB], ks)]
        for r in (R_PIPE, R_BLP):
            for refname, xref, kref in refs:
                p = paired(X[r], xref)
                res["sets"][s]["paired"][f"{r} vs {refname}"] = p
                P.append(row(s, r, refname, p))
            if ks_knn:
                idx = [ks.index(k) for k in ks_knn]
                sub = {key: v[idx] for key, v in X[r].items()}
                p = paired(sub, arrays(kk, F, ks_knn, kf))
                res["sets"][s]["paired"][f"{r} vs LtF kNN eps=1% (stored)"] = p
                P.append(row(s, r, "LtF kNN eps=1% (stored)", p))
    L += P
    open(os.path.join(RES, "m1x_pipeline.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(RES, "m1x_pipeline.json"), "w"), indent=1, default=float)
    print("\n".join(L))


def row(s, r, ref, p):
    c = lambda v, ci, q=2: f"{v:+.{q}f} [{ci[0]:+.{q}f}, {ci[1]:+.{q}f}]"
    return (f"| {NAMES[s]} | {r} | {ref} | {p['n']} | {c(p['d_gap'], p['d_gap_ci'])} | {c(p['d_logsp'], p['d_logsp_ci'])} | "
            f"{p['time_ratio']:.2f}× | {c(p['d_served'], p['d_served_ci'], 1)} | {c(p['d_fixed'], p['d_fixed_ci'], 1)} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.report:
        report()
    else:
        run(a)
