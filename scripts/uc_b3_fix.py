"""B3 (24-hour UC): solver acceleration by partial fixing + reduced MILP, against the full MILP.

    python scripts/uc_b3_fix.py --split val --n 12 --ratios 0.9,0.95 --rankings bce_asym_g,rl_g,bce_g   # selection
    python scripts/uc_b3_fix.py --split test --ratios 0.8,0.9,0.95 --rankings raclearn,best --resolve_full 8

Rankings (probabilities from the label-free models of scripts/uc_b3_train.py; no MILP labels anywhere):
  raclearn    RACLearn-style confidence rule: fix the decisions with the largest |p - 0.5| of the imitation
              (LF-BCE) model; no guard (baseline).
  bce_g       same ranking + adequacy guard (5 % margin) + LP-relaxation guard.
  bce_asym_g  LF-BCE, OFF error odds x 10 (asymmetric) + adequacy guard + LP-relaxation guard.
  rl_g        REINFORCE probabilities, symmetric + adequacy guard + LP-relaxation guard.
  best        per ratio, the guarded ranking selected on val (results/uc24/b3_fix_select.json).
Every fixing set is cleared of min up/down conflicts (release_conflicting_rows) before solving.

Each worker processes one instance: the LP guard, every reduced MILP and (for the first --resolve_full
instances) a back-to-back re-solve of the full MILP, in one process (same CPU conditions). Time of a pipeline =
LP relaxation (features) + guard LPs + reduced MILP (+ fallback). Speed-up = full-MILP time / pipeline time,
with the full-MILP time from data generation (and, for the re-solved subset, from the same run). Time-to-quality:
first time the full MILP's incumbent log reaches the reduced solution's cost.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.b3 import inc_from_arrays, load_b3, solve_milp_hs, subset, time_to_reach  # noqa: E402
from otsl.fixpolicy import lp_guard, release_conflicting_rows  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_fixing import adequacy_guard, choose_fixings  # noqa: E402

OUT = os.path.join("results", "uc24")
_W = {}


def _init(cfg):
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=cfg["T"], network=True), cfg=cfg)


def _job(job):
    i, load, avail, u0, sr, specs, resolve_full = job
    s, m, cfg = _W["s"], _W["m"], _W["cfg"]
    sc = UCScenario(load=load, avail=avail, u0=u0, sr=sr)
    out = {"i": int(i)}
    if resolve_full:
        sol = solve_milp_hs(m, sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"])
        out["full_b2b"] = dict(obj=sol["obj"], time=sol["time"], cpu=sol["cpu"], gap=sol["gap"], status=sol["status"],
                               inc=sol["inc"], loadavg=os.getloadavg()[0])
    for name, fix_arr, use_lp_guard in specs:
        fix = {(int(t), int(g)): int(v) for t, g, v in fix_arr}
        t0, c0 = time.time(), time.process_time()
        rel_lp = 0
        if use_lp_guard and fix:
            fix, rel_lp, _ = lp_guard(m, s, sc, fix)
        fix, rel_c = release_conflicting_rows(fix, s, u0)
        t_guard, c_guard = time.time() - t0, time.process_time() - c0
        sol = solve_milp_hs(m, sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], z_fix=fix or None)
        t, c, fb = sol["time"], sol["cpu"], 0
        if sol["u"] is None:                       # infeasible fixings: solve without them
            fb = 1
            sol = solve_milp_hs(m, sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"])
            t, c = t + sol["time"], c + sol["cpu"]
        out[name] = dict(obj=sol["obj"], shed=sol.get("shed", np.nan), short=sol.get("short", np.nan), t_milp=t,
                         c_milp=c, t_guard=t_guard, c_guard=c_guard, fallback=fb, status=sol["status"],
                         gap=sol["gap"], n_fixed=len(fix), released_lp=rel_lp, released_conflict=rel_c,
                         units_on=float(sol["u"].sum() / sol["u"].shape[0]) if sol["u"] is not None else np.nan,
                         loadavg=os.getloadavg()[0], inc=sol["inc"], bound=sol["bound"])
    return out


def make_specs(i, d, probs, sysm, rankings, ratios, kappa=10.0):
    """[(name, fix array [k, 3], lp_guard flag)] for instance i"""
    specs = []
    for ratio in ratios:
        for rk in rankings:
            src = "rl" if rk.startswith("rl") else "bce"
            p = probs[src][i]
            kap = kappa if "asym" in rk else 1.0
            tt, gg = choose_fixings(p, ratio, kap)
            fix = {(int(t), int(g)): int(p[t, g] > 0.5) for t, g in zip(tt, gg)}
            guard = rk.endswith("_g")
            if guard:
                fix, _ = adequacy_guard(fix, d["load"][i], d["avail"][i], d["sr"][i], sysm)
            arr = np.array([(t, g, v) for (t, g), v in fix.items()], np.int64).reshape(-1, 3)
            specs.append((f"{int(round(ratio * 100))}%|{rk}", arr, guard))
    return specs


def summarize(res, d, idx, names, t_rel, incs=None):
    """per method: gaps vs the generation-run full MILP, served share, speed-ups, time-to-quality"""
    ref = d["obj"][idx]
    lf = not np.isfinite(ref).all()
    if lf:                                  # split without full MILPs: gaps to the LP-relaxation bound
        ref = d["c_rel"][idx]
    t_full = d["time"][idx]
    c_full = d["cpu"][idx] if "cpu" in d else np.full(len(idx), np.nan)
    incs = incs or [inc_from_arrays(d["inc_t"][i], d["inc_obj"][i]) for i in idx]
    bound = d["bound"][idx]
    rows = [{"method": "LP relaxation (reference; no full MILP)" if lf else "full MILP (generation run)", "n": len(idx), "gap_mean_%": 0.0, "gap_median_%": 0.0,
             "served_%": float(((d["shed"][idx] < 1e-6) & (d["short"][idx] < 1e-6)).mean() * 100),
             "time_mean_s": float(t_full.mean()), "time_median_s": float(np.median(t_full)),
             "cpu_mean_s": float(np.nanmean(c_full)), "milp_gap_%": float(d["gap"][idx].mean() * 100)}]
    by_i = {r["i"]: r for r in res}
    for name in names:
        R = [by_i[i][name] for i in idx]
        obj = np.array([r["obj"] for r in R])
        gap = (obj - ref) / ref * 100
        served = np.array([(r["shed"] < 1e-6) and (r["short"] < 1e-6) for r in R])
        t_pipe = np.array([r["t_milp"] + r["t_guard"] for r in R]) + t_rel[idx]
        c_pipe = np.array([r["c_milp"] + r["c_guard"] for r in R]) + t_rel[idx]
        ttq = np.array([time_to_reach(inc, o) for inc, o in zip(incs, obj)])
        reached = np.isfinite(ttq)
        ttq_c = np.where(reached, ttq, t_full)                      # censored at the full MILP's time
        rows.append({"method": name, "n": len(idx), "fixed_share_%": float(np.mean([r["n_fixed"] for r in R]) / d["u"][0].size * 100),
                     "gap_mean_%": float(gap.mean()), "gap_median_%": float(np.median(gap)),
                     "gap_to_full_bound_mean_%": float(np.mean((obj - bound) / obj * 100)),
                     "gap_p90_%": float(np.percentile(gap, 90)), "gap_max_%": float(gap.max()),
                     "served_%": float(served.mean() * 100),
                     "beats_or_matches_full_%": float((gap <= 1e-3).mean() * 100),
                     "time_mean_s": float(t_pipe.mean()), "time_median_s": float(np.median(t_pipe)),
                     "speedup_mean_x": float(t_full.mean() / t_pipe.mean()),
                     "speedup_median_x": float(np.median(t_full / t_pipe)),
                     "cpu_speedup_mean_x": float(np.nanmean(c_full) / c_pipe.mean()) if np.isfinite(c_full).any() else np.nan,
                     "ttq_reached_%": float(reached.mean() * 100),
                     "ttq_mean_s_censored": float(ttq_c.mean()),
                     "ttq_speedup_median_x": float(np.median(ttq_c / t_pipe)),
                     "ttq_speedup_mean_x": float(ttq_c.mean() / t_pipe.mean()),
                     "fallback_%": float(np.mean([r["fallback"] for r in R]) * 100),
                     "released_lp": float(np.mean([r["released_lp"] for r in R])),
                     "limit_hit_%": float(np.mean([r["status"] != "optimal" for r in R]) * 100),
                     "units_on": float(np.mean([r["units_on"] for r in R]))})
    return rows


def fmt(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    f = lambda v: f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:.1f}" if isinstance(v, float) else str(v))
    return head + "".join("| " + " | ".join(f(r.get(c, "")) for c in cols) + " |\n" for r in rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ratios", default="0.8,0.9,0.95")
    ap.add_argument("--rankings", default="raclearn,best")
    ap.add_argument("--resolve_full", type=int, default=0, help="re-solve the full MILP back to back on the first k")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--select", action="store_true",
                    help="val: per ratio, pick the guarded ranking with the lowest mean gap -> b3_fix_select.json")
    ap.add_argument("--cfg", default="uc24")
    ap.add_argument("--summarize_only", action="store_true", help="summarise the instances already in the jsonl")
    a = ap.parse_args()
    root = os.path.join("data", "generated", a.cfg)
    if a.cfg != "uc24":
        OUT = os.path.join("results", "uc24", "b3_debug")
    d = load_b3(os.path.join(root, f"{a.split}.npz"))
    gen_cfg = json.load(open(os.path.join(root, f"{a.split}.json")))["cfg"]
    cfg = {"T": int(d["load"].shape[1]), "time_limit": a.limit or gen_cfg["time_limit"], "mip_gap": a.gap or gen_cfg["mip_gap"]}
    n = a.n or len(d["load"])
    idx = np.arange(n)
    sysm = load_rts_gmlc()
    import torch
    from otsl.ucml import UCFeaturizer, build_uc_model1
    from uc_b3_train import strip
    tr = strip(load_b3(os.path.join(root, "train.npz")))
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    probs = {}
    for key, path in (("bce", "b3_lf_bce.pt"), ("rl", "b3_rl_selected.pt")):
        m1 = build_uc_model1(sysm, feat, cfg["T"], "gnn", seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(OUT, path)))
        probs[key] = m1.predict(d)
    ratios = [float(x) for x in a.ratios.split(",")]
    rankings = a.rankings.split(",")
    if "best" in rankings:
        sel = json.load(open(os.path.join(OUT, "b3_fix_select.json")))["best"]
        rankings = [r for r in rankings if r != "best"]
    specs_all = []
    for i in idx:
        sp_ = make_specs(i, d, probs, sysm, rankings, ratios)
        if "best" in a.rankings.split(","):
            for ratio in ratios:
                rk = sel[str(int(round(ratio * 100)))]
                (nm, arr, g), = make_specs(i, d, probs, sysm, [rk], [ratio])
                sp_.append((f"{int(round(ratio * 100))}%|best={rk}", arr, g))
        specs_all.append(sp_)
    names = [s[0] for s in specs_all[0]]
    print("methods:", names, "cfg", cfg, flush=True)
    jobs = [(int(i), d["load"][i], d["avail"][i], d["u0"][i], d["sr"][i], specs_all[k], k < a.resolve_full)
            for k, i in enumerate(idx)]
    # re-solved instances first so the back-to-back subset finishes early
    res, t0 = [], time.time()
    jl = os.path.join(OUT, f"b3_fix_{a.split}{a.tag}.jsonl")
    if a.summarize_only:
        res = [json.loads(x) for x in open(jl)]
        idx = np.array(sorted(r["i"] for r in res))
        n, jobs = len(idx), []
    with open(jl, "a" if a.summarize_only else "w") as fjl, \
            mp.get_context("spawn").Pool(a.workers if jobs else 1, initializer=_init, initargs=(cfg,)) as pool:
        for r in (pool.imap_unordered(_job, jobs, chunksize=1) if jobs else []):
            res.append(r)
            fjl.write(json.dumps(r, default=float) + "\n")
            fjl.flush()
            i = r["i"]
            msg = " ".join(f"{nm.split('|')[0]}:{nm.split('|')[1][:10]} {(r[nm]['obj'] - d['obj'][i]) / d['obj'][i] * 100:+.3f}% "
                           f"{r[nm]['t_milp']:.0f}s" for nm in names)
            print(f"[{len(res)}/{n}] inst {i} full {d['time'][i]:.0f}s" +
                  (f" (b2b {r['full_b2b']['time']:.0f}s)" if "full_b2b" in r else "") + " | " + msg +
                  f" ({time.time() - t0:.0f}s)", flush=True)
    rows = summarize(res, d, idx, names, d["t_rel"])
    b2b = [r for r in res if "full_b2b" in r]
    extra = {"cfg": cfg, "n": n, "wall_s": time.time() - t0}
    if b2b:
        bi = np.array(sorted(r["i"] for r in b2b))
        tb = np.array([r["full_b2b"]["time"] for r in b2b])
        extra["b2b"] = {"n": len(b2b), "full_time_gen_mean": float(d["time"][bi].mean()), "full_time_b2b_mean": float(tb.mean()),
                        "full_cpu_b2b_mean": float(np.mean([r["full_b2b"]["cpu"] for r in b2b])),
                        "full_obj_b2b_vs_gen_%": float(np.mean([(r["full_b2b"]["obj"] - d["obj"][r["i"]]) / d["obj"][r["i"]] * 100 for r in b2b]))}
        d_b = dict(d)
        d_b["time"] = d["time"].copy()
        d_b["cpu"] = d["cpu"].copy() if "cpu" in d else np.full(len(d["time"]), np.nan)
        for r in b2b:
            d_b["time"][r["i"]] = r["full_b2b"]["time"]
            d_b["cpu"][r["i"]] = r["full_b2b"]["cpu"]
        extra["rows_b2b"] = summarize(res, d_b, bi, names, d["t_rel"],
                                      incs=[[tuple(x) for x in r["full_b2b"]["inc"]] for r in sorted(b2b, key=lambda r: r["i"])])
        extra["b2b"]["full_inc_b2b"] = {int(r["i"]): r["full_b2b"]["inc"] for r in b2b}
    cols = ["method", "n", "fixed_share_%", "gap_mean_%", "gap_median_%", "gap_max_%", "served_%", "time_mean_s",
            "speedup_mean_x", "speedup_median_x", "cpu_speedup_mean_x", "ttq_reached_%", "ttq_speedup_median_x",
            "limit_hit_%", "fallback_%"]
    md = fmt(rows, cols)
    if b2b:
        md += f"\n\nBack-to-back subset ({len(b2b)} instances, full MILP re-solved in the same worker):\n\n" + fmt(extra["rows_b2b"], cols)
    print(md, flush=True)
    with open(os.path.join(OUT, f"b3_fix_{a.split}{a.tag}.md"), "w") as f:
        f.write(f"uc24 {a.split}, {n} instances, cfg {cfg}\n\n" + md)
    json.dump({"rows": rows, **extra}, open(os.path.join(OUT, f"b3_fix_{a.split}{a.tag}.json"), "w"), indent=1, default=float)
    if a.select:
        # pre-declared rule: per ratio, lowest mean gap among guarded rankings; ties (within 0.01 pp) -> faster
        best = {}
        for ratio in ratios:
            key = str(int(round(ratio * 100)))
            cand = [r for r in rows[1:] if r["method"].startswith(key + "%|") and r["method"].endswith("_g")]
            lo = min(r["gap_mean_%"] for r in cand)
            near = [r for r in cand if r["gap_mean_%"] <= lo + 0.01]
            best[key] = min(near, key=lambda r: r["time_mean_s"])["method"].split("|")[1]
        json.dump({"best": best, "rule": "per ratio: lowest mean val cost (gap to the val LP-relaxation bound) among "
                   "guarded rankings, ties within 0.01 pp -> lowest pipeline time", "val_rows": rows}, open(os.path.join(OUT, "b3_fix_select.json"), "w"),
                  indent=1, default=float)
        print("selected:", best, flush=True)
