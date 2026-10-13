"""Cost-aware check study: tables (results/uc12/cguard_results.{md,json}).

    python3 scripts/uc_cguard_report.py

Rows per set (test_fresh 0-119 and the six shifted sets): full MILP at 0.5 %, the pipeline, and the pipeline with each
check (C1 pre-solve LP-cost check, C2 post-solve bound check + gradient polish, C3 post-solve bound check + full-MILP
fallback). Gap = (C - DB) / C against the reference full MILP's dual bound (0.1 %, from the pipeline records); speed-up =
T_full(0.1 %) / T_method with every overhead: the pipeline's inference, LP relaxation, guards and reduced MILP (records);
for C1 also its relaxed reduced LP (every instance) and, where it fires, the release LPs and the new reduced MILP; for
C2 / C3, where they fire, the gradient LP and polish MILP / the full MILP at 0.5 %. Paired bootstrap CIs (2,000
resamples) against the pipeline and against the full MILP at 0.5 %.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uc_cguard_common import R_FULL, R_FULL5, R_PIPE, RES, SHIFTS, pipeline_records, slack  # noqa: E402

NAMES = {"tf": "test_fresh (120)", "load_hi": "load +15 %", "load_lo": "load −15 %", "ren_hi": "wind + solar ×1.5",
         "gen_out": "2–3 units out", "line_out": "1–2 lines out", "night": "across midnight"}
RULES = ("full MILP 0.5 %", "pipeline", "pipeline + C1 (pre-solve LP-cost check)", "pipeline + C2 (bound check + gradient polish)",
         "pipeline + C3 (bound check + full-MILP fallback)")
NB = 2000


def boot(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    b = x[np.random.default_rng(seed).integers(0, len(x), (NB, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def method_time(r):
    return r["time"] + r.get("pre_s", 0) + r.get("relax_s", 0) + r.get("guard_s", 0)


def build(set_, ks, R, T):
    """per-instance arrays for every rule"""
    out = {ru: dict(cost=[], t=[], served=[], fixed=[], fired=[]) for ru in RULES}
    db, tfull = [], []
    for k in ks:
        rr, c = R[(set_, k)], T[(set_, k)]
        F, F5, P = rr[R_FULL], rr[R_FULL5], rr[R_PIPE]
        db.append(F["bound"] if np.isfinite(F["bound"]) else F["obj"] * (1 - (F["mip_gap"] or 0)))
        tfull.append(F["time"])
        tp = method_time(P)

        def add(ru, cost, t, served, fixed, fired):
            for key, v in zip(("cost", "t", "served", "fixed", "fired"), (cost, t, served, fixed, fired)):
                out[ru][key].append(v)
        add(RULES[0], F5["obj"], F5["time"], slack(F5) < 1e-6, 0.0, False)
        add(RULES[1], P["obj"], tp, slack(P) < 1e-6, P["fixed_share"], False)
        if c["c1"] is None:
            add(RULES[2], P["obj"], tp + c["rr_s"], slack(P) < 1e-6, P["fixed_share"], False)
        else:
            r1 = c["c1"]
            t1 = P.get("pre_s", 0) + P.get("relax_s", 0) + P.get("guard_s", 0) + c["rr_s"] + r1["extra_s"] + r1["time"]
            add(RULES[2], r1["obj"], t1, slack(r1) < 1e-6, r1["fixed_share"], True)
        if c["c2"] is None:
            add(RULES[3], P["obj"], tp, slack(P) < 1e-6, P["fixed_share"], False)
        else:
            r2 = c["c2"]
            better = r2["feasible"] and r2["obj"] < P["obj"]
            src = r2 if better else P
            add(RULES[3], src["obj"], tp + r2["grad_s"] + r2["time"], slack(src) < 1e-6, P["fixed_share"], True)
        if c["fired_post"]:
            better = F5["obj"] < P["obj"]
            src = F5 if better else P
            add(RULES[4], src["obj"], tp + F5["time"], slack(src) < 1e-6, P["fixed_share"], True)
        else:
            add(RULES[4], P["obj"], tp, slack(P) < 1e-6, P["fixed_share"], False)
    db, tfull = np.array(db), np.array(tfull)
    X = {}
    for ru, v in out.items():
        cost = np.array(v["cost"], float)
        X[ru] = dict(gap=(cost - db) / cost * 100, sp=tfull / np.array(v["t"]), served=np.array(v["served"], bool),
                     fixed=np.array(v["fixed"]) * 100, fired=np.array(v["fired"], bool))
    return X


def stats(x):
    g, sp = x["gap"], x["sp"]
    return dict(n=int(len(g)), gap_mean=float(g.mean()), gap_ci=boot(g), gap_max=float(g.max()),
                served=float(x["served"].mean() * 100), sp_mean=float(sp.mean()), sp_ci=boot(sp),
                sp_median=float(np.median(sp)), fixed=float(x["fixed"].mean()), fired=float(x["fired"].mean() * 100))


def paired(a, b):
    dg = a["gap"] - b["gap"]
    dl = np.log(a["sp"]) - np.log(b["sp"])
    ds = (a["served"].astype(float) - b["served"].astype(float)) * 100
    return dict(d_gap=float(dg.mean()), d_gap_ci=boot(dg), d_logsp=float(dl.mean()), d_logsp_ci=boot(dl),
                time_ratio=float(np.exp(dl.mean())), d_served=float(ds.mean()), d_served_ci=boot(ds))


def c(v, ci, q=2):
    return f"{v:+.{q}f} [{ci[0]:+.{q}f}, {ci[1]:+.{q}f}]"


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    R = pipeline_records()
    T = {}
    for line in open(os.path.join(RES, "cguard_test.jsonl")):
        r = json.loads(line)
        T[(r["set"], r["k"])] = r
    sel = json.load(open(os.path.join(RES, "cguard_select.json")))
    rt = json.load(open(os.path.join(RES, "cguard_retime.json")))["summary"]
    res = dict(select=sel, retime=rt, sets={})
    L = ["# Cost-aware checks for the recommended pipeline: results", "",
         "Generated by `scripts/uc_cguard_report.py`; method: [`docs/methods/cguard.md`](../../docs/methods/cguard.md).", "",
         f"Thresholds from the original validation set (360 instances; {sel['rule']}): d_pre = {sel['d_pre'] * 100:g} %, "
         f"d_post = {sel['d_post'] * 100:g} % (validation firing rates {sel['rate_pre'][str(sel['d_pre'])] * 100:.1f} % / "
         f"{sel['rate_post'][str(sel['d_post'])] * 100:.1f} %).",
         f"Timing check (14 pipeline instances re-solved on core 0): full MILP time ratio median {rt['full_ratio_median']:.3f}, "
         f"ratio of means {rt['full_ratio_of_means']:.3f}; pipeline reduced MILP median {rt['pipe_ratio_median']:.3f}, ratio of "
         f"means {rt['pipe_ratio_of_means']:.3f}; identical objectives {rt['same_objectives']} / {rt['n']}; agreement within "
         f"10 %: {rt['agree_within_10pct']} → the pipeline records are reused.", "",
         "| set | rule | n | fired % | gap mean % [95 % CI] | max % | served % | speed-up mean [95 % CI] | median | fixed % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    PL = ["", "Paired (rule − reference; Δ gap pp, Δ log speed-up, time ratio = geometric mean of reference / rule time, "
          "Δ served pp; bootstrap 95 % CI):", "",
          "| set | rule | vs pipeline: Δ gap | Δ log speed-up | vs full MILP 0.5 %: Δ gap | Δ log speed-up | time ratio | Δ served |",
          "|---|---|---|---|---|---|---|---|"]
    for s in ("tf",) + SHIFTS:
        ks = sorted(k for (s_, k) in T if s_ == s)
        if not ks:
            continue
        X = build(s, ks, R, T)
        S = {ru: stats(x) for ru, x in X.items()}
        Pp = {ru: paired(X[ru], X[RULES[1]]) for ru in RULES[2:]}
        Pf = {ru: paired(X[ru], X[RULES[0]]) for ru in RULES[1:]}
        res["sets"][s] = dict(n=len(ks), stats=S, vs_pipeline=Pp, vs_full5=Pf)
        for ru in RULES:
            st = S[ru]
            fx = "–" if ru == RULES[0] else f"{st['fixed']:.1f}"
            L.append(f"| {NAMES[s]} | {ru} | {st['n']} | {st['fired']:.0f} | {st['gap_mean']:.2f} [{st['gap_ci'][0]:.2f}, "
                     f"{st['gap_ci'][1]:.2f}] | {st['gap_max']:.2f} | {st['served']:.0f} | {st['sp_mean']:.1f} "
                     f"[{st['sp_ci'][0]:.1f}, {st['sp_ci'][1]:.1f}] | {st['sp_median']:.1f} | {fx} |")
        for ru in RULES[1:]:
            pp = Pp.get(ru)
            pf = Pf[ru]
            PL.append(f"| {NAMES[s]} | {ru} | " + (f"{c(pp['d_gap'], pp['d_gap_ci'])} | {c(pp['d_logsp'], pp['d_logsp_ci'])}"
                                                   if pp else "– | –")
                      + f" | {c(pf['d_gap'], pf['d_gap_ci'])} | {c(pf['d_logsp'], pf['d_logsp_ci'])} | {pf['time_ratio']:.2f}× | "
                        f"{c(pf['d_served'], pf['d_served_ci'], 1)} |")
    # validation pass summary
    V = [json.loads(l) for l in open(os.path.join(RES, "cguard_val.jsonl"))]
    gp = lambda c_, db: (c_ - db) / c_ * 100
    fired = [v for v in V if v["g"] > sel["d_post"]]
    vs = dict(n=len(V), pipe_gap_mean=float(np.mean([gp(v["pipe"]["obj"], v["db"]) for v in V])),
              n_fired=len(fired),
              fired_pipe=float(np.mean([gp(v["pipe"]["obj"], v["db"]) for v in fired])) if fired else None,
              fired_polish=float(np.mean([gp(min(v["polish"]["obj"], v["pipe"]["obj"]), v["db"]) for v in fired])) if fired else None,
              fired_full5=float(np.mean([gp(min(v["full5"]["obj"], v["pipe"]["obj"]), v["db"]) for v in fired])) if fired else None,
              g_quantiles={str(q): float(np.quantile([v["g"] for v in V], q)) for q in (0.5, 0.9, 0.95, 0.99)},
              delta_quantiles={str(q): float(np.quantile([v["delta"] for v in V], q)) for q in (0.5, 0.9, 0.95, 0.99)})
    res["validation"] = vs
    L += PL
    L += ["", f"Validation (360, gap to the dataset MILP's dual bound): pipeline mean {vs['pipe_gap_mean']:.3f} %; the bound check "
              f"at d_post fires on {vs['n_fired']} instances, where the pipeline has {vs['fired_pipe']:.2f} %, + gradient polish "
              f"{vs['fired_polish']:.2f} %, + full MILP 0.5 % {vs['fired_full5']:.2f} %. Quantiles of g (50 / 90 / 95 / 99 %): "
              + " / ".join(f"{vs['g_quantiles'][q] * 100:.2f}" for q in ("0.5", "0.9", "0.95", "0.99")) + " %; of Delta: "
              + " / ".join(f"{vs['delta_quantiles'][q] * 100:.3f}" for q in ("0.5", "0.9", "0.95", "0.99")) + " %."]
    open(os.path.join(RES, "cguard_results.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(RES, "cguard_results.json"), "w"), indent=1, default=float)
    print("\n".join(L))
