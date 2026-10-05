"""Evaluation of the combined pipelines and the multi-seed confirmations (B2, uc12).

    python scripts/uc_combo_eval.py --stage valfix --n 15          # val: probability source for the fixing pipeline
    python scripts/uc_combo_eval.py --stage calib                  # val: thresholds + end-to-end predictor
    python scripts/uc_combo_eval.py --stage e2e --split test_fresh # end-to-end, all 120 instances, one LP per decoder
    python scripts/uc_combo_eval.py --stage fix --split test_fresh --n 60   # fixing, full MILP back to back

Decision rules are fixed before any test set is read (see docs/methods/combo.md):
  * threshold of every end-to-end model: argmin over THRESHOLDS of the mean val cost of block-repaired schedules
    divided by the LP-relaxation cost (label-free expected cost incl. shedding / reserve penalties, the rule of
    scripts/uc_constrained.py --stage calib, extended to thresholds 0.3 / 0.4);
  * end-to-end predictor family: lowest seed-averaged val criterion at the calibrated thresholds;
  * screening: all THRESHOLDS of the chosen predictor, block-repaired, best by the exact LP (k <= 7 LPs);
  * probability source of the fixing pipeline: lowest mean val gap (first n val instances, back-to-back reduced
    MILPs, harm seed 0, 95 % target, adequacy + LP-relaxation guards); within 0.1 pp the faster one.
Outputs: results/uc12/combo_{valfix.jsonl,valfix.json,calib.json,e2e_<split>.json/.npz,fix_<split>.jsonl}.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import (THRESHOLDS, block_decode, e2e_metrics, harm_from_file, harm_scores, mud_decode,  # noqa: E402
                        rule_fixings, solver_guards)
from otsl.constrained import HourlyOracle  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_constrained import load_model1, setup, strip  # noqa: E402

OUT = os.path.join("results", "uc12")
ROOT = os.path.join("data", "generated", "uc12")
KEYS = {"train": 0, "val": 10 ** 6, "test": 2 * 10 ** 6, "test_fresh": 3 * 10 ** 6}
TL, GAP = 60.0, 1e-3
SEEDS = (0, 1, 2)


def model_path(name):
    """registry of Model 1 checkpoints (seed 0 of rl / lag = the original runs)"""
    fixed = {"rl_s0": "constrained_rl_lf.pt", "lag_s0": "constrained_lag_D.pt", "lf_bce": "constrained_lf_bce.pt",
             "bce": "uc_model1_4.pt", "milp_rl": "uc_model1_rl.pt", "st_orig": "selftrain_m3.pt",
             "strl": "selftrain_strl.pt"}
    return os.path.join(OUT, fixed.get(name, f"combo_{name}.pt"))


FAMILIES = {"rl": [f"rl_s{s}" for s in SEEDS], "lag": [f"lag_s{s}" for s in SEEDS],
            "st": [f"st_s{s}" for s in SEEDS], "lf_bce": ["lf_bce"]}


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


_PROBS = {}


def probs(name, d, split, tr, sysm):
    key = (name, split)
    if key not in _PROBS:
        _PROBS[key] = load_model1(model_path(name), sysm, tr, 12).predict(d).astype(np.float64)
    return _PROBS[key]


# ----------------------------------------------------------------------------------- MILP workers
_W = {}


def _init():
    torch.set_num_threads(1)
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=12, network=True))


def _solve(sc, fix):
    m = _W["m"]
    sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP, z_fix=fix or None)
    t_, fb = sol.time, 0
    if sol.u is None:                      # fixings conflict with min up/down: solve without them
        fb = 1
        sol = m.solve_uc(sc, time_limit=TL, mip_gap=GAP)
        t_ += sol.time
    return dict(obj=float(sol.obj), shed=float(sol.shed), short=float(sol.short), time=float(t_), status=sol.status,
                fallback=fb, units_on=float(sol.u.sum() / sol.u.shape[0]) if sol.u is not None else float("nan"))


def _job(args):
    """full MILP, then every reduced MILP of the same instance, back to back in one process"""
    i, sc, specs, full = args
    out = []
    if full:
        out.append(dict(i=i, config="full", seed=-1, ratio=0.0, fixed_share=0.0, **_solve(sc, {})))
    for cfg_name, seed, ratio, fix, lpg in specs:
        extra, lp_s = {}, 0.0
        if lpg:                                   # solver-aware guards, timed as part of the rule
            fix, extra, lp_s = solver_guards(_W["m"], _W["s"], sc, fix)
        r = _solve(sc, fix)
        r["time"] += lp_s
        out.append(dict(i=i, config=cfg_name, seed=seed, ratio=ratio,
                        fixed_share=len(fix) / (sc.load.shape[0] * len(sc.u0)), **extra, **r))
    return out


def run_fixing(jobs, path, workers):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            done.add(json.loads(line)["i"])
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} instances to solve ({len(done)} done), {len(jobs[0][2]) if jobs else 0} reduced MILPs each",
          flush=True)
    t0 = time.time()
    if not jobs:
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(path, "a") as f:
        for k, recs in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            full = [r for r in recs if r["config"] == "full"]
            ft = full[0]["time"] if full else float("nan")
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} full {ft:.1f}s, reduced total "
                  f"{sum(r['time'] for r in recs if r['config'] != 'full'):.1f}s", flush=True)


def summarize_fix(path, ref_obj, n_max=None):
    """per (config, seed, ratio): gap / served / speed-up vs the back-to-back full MILP, over instances that have
    every configuration"""
    recs = [json.loads(line) for line in open(path)]
    by = {}
    for r in recs:
        if n_max is not None and r["i"] >= n_max:
            continue
        by.setdefault((r["config"], r["seed"], r["ratio"]), {})[r["i"]] = r
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    full = by[("full", -1, 0.0)]
    t_full = np.array([full[i]["time"] for i in common])
    rows = []
    for (c, s, q), d in sorted(by.items()):
        rr = [d[i] for i in common]
        cost = np.array([r["obj"] for r in rr])
        ref = ref_obj[common]
        gap = (cost - ref) / ref * 100
        served = np.array([(r["shed"] < 1e-6) and (r["short"] < 1e-6) for r in rr])
        tm = np.array([r["time"] for r in rr])
        rows.append(dict(config=c, seed=s, target=q, fixed_share=float(np.mean([r["fixed_share"] for r in rr]) * 100),
                         gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_max=float(gap.max()),
                         n_gt1=int((gap > 1).sum()), served=float(served.mean() * 100),
                         match=float((gap <= 1e-3).mean() * 100), time_s=float(tm.mean()),
                         speedup=float(t_full.mean() / tm.mean()), fallback=float(np.mean([r["fallback"] for r in rr]) * 100),
                         n=len(rr)))
    return rows, common


# ----------------------------------------------------------------------------------- end-to-end
def e2e_eval(models, d, split, oracle, sysm, tr, ths_of, arrays, rows, extra_mud=True):
    """models: names; ths_of(name) -> thresholds to evaluate with the block decoder"""
    n = len(d["load"])
    keys = np.arange(n) + KEYS[split]
    for name in models:
        p = probs(name, d, split, tr, sysm)
        decs = [("mud", 0.5)] if extra_mud else []
        decs += [("block", th) for th in ths_of(name)]
        for dec, th in decs:
            tag = f"{name}|{dec}|{th}"
            if tag in arrays:
                continue
            U = (mud_decode if dec == "mud" else block_decode)(p, d, sysm, th)
            c, sh, so = oracle.scores(d, np.arange(n), U, keys)
            units = U.sum((1, 2)) / U.shape[1]
            arrays[tag] = np.stack([c, sh, so, units])
            r = e2e_metrics(c, sh, so, d["obj"], units.mean()) if "obj" in d else {}
            g_rel = (c - d["c_rel"]) / d["c_rel"]
            r.update(model=name, decoder=dec, threshold=th, crit=float(np.mean(c / d["c_rel"])),
                     served_lf=float(((sh < 1e-6) & (so < 1e-6)).mean() * 100),
                     gap_rel_served=float(g_rel[(sh < 1e-6) & (so < 1e-6)].mean() * 100) if ((sh < 1e-6) & (so < 1e-6)).any() else float("nan"))
            rows.append(r)
            print(f"  [{split}] {tag:28s} served {r['served_lf']:5.1f}%  crit {r['crit']:.4f}  "
                  + (f"median {r['gap_median']:7.3f}%  served-mean {r['gap_mean_served']:6.3f}%  mean {r['gap_mean']:8.2f}%  "
                     f"units {r['units_on']:.2f}" if "gap_median" in r else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["valfix", "calib", "e2e", "fix"])
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    cfg, tr_full, va_full, _te_unused, sysm, rep = setup()
    T = cfg["T"]
    tr = strip(tr_full)
    ff = FixFeaturizer(sysm, T)

    if a.stage == "valfix":
        d = va_full
        n = min(a.n, len(d["load"]))
        idx = list(range(n))
        ens = harm_from_file(os.path.join(OUT, "combo_harm_s0.pt"))
        P = np.load(os.path.join(OUT, "fixpolicy_probs.npz"))
        pb = probs("bce", d, "val", tr, sysm)
        print("BCE probabilities match fixpolicy_probs (val):", float(np.abs(pb - P["bce_va"]).max()), flush=True)
        sources = {"bce": pb, "st": probs("st_s0", d, "val", tr, sysm), "rl": probs("rl_s0", d, "val", tr, sysm)}
        jobs = []
        for i in idx:
            sc = scenario_from(d, i)
            specs = []
            for src, p in sources.items():
                h = harm_scores(ens, ff, p[i:i + 1], d, [i])[0]
                specs.append((f"{src}+harm+lp", 0, 0.95, rule_fixings("harm", 0.95, p[i], sc, sysm, h), True))
            jobs.append((i, sc, specs, True))
        path = os.path.join(OUT, f"combo_valfix{a.tag}.jsonl")
        run_fixing(jobs, path, a.workers)
        rows, common = summarize_fix(path, d["obj"], n)
        for r in rows:
            print(f"  {r['config']:16s} gap mean {r['gap_mean']:7.3f}%  median {r['gap_median']:6.3f}%  served "
                  f"{r['served']:5.1f}%  fixed {r['fixed_share']:5.1f}%  speed-up {r['speedup']:5.2f}x", flush=True)
        cand = [r for r in rows if r["config"] != "full"]
        best = min(r["gap_mean"] for r in cand)
        close = [r for r in cand if r["gap_mean"] <= best + 0.1]
        pick = max(close, key=lambda r: r["speedup"])["config"].split("+")[0]
        dump(os.path.join(OUT, f"combo_valfix{a.tag}.json"), {"rows": rows, "n": len(common), "chosen_source": pick,
                                                              "rule": "lowest mean gap; within 0.1 pp the faster"})
        print("chosen probability source for the fixing pipeline:", pick, flush=True)

    elif a.stage == "calib":
        d = va_full
        oracle = HourlyOracle(cfg, a.workers)
        rows, arrays = [], {}
        models = [m for fam in FAMILIES.values() for m in fam]
        e2e_eval(models, d, "val", oracle, sysm, tr, lambda name: THRESHOLDS, arrays, rows, extra_mud=False)
        oracle.close()
        cal = {}
        for name in models:
            rr = [r for r in rows if r["model"] == name and r["decoder"] == "block"]
            b = min(rr, key=lambda r: r["crit"])
            cal[name] = {"threshold": b["threshold"], "crit": b["crit"], "served_lf": b["served_lf"],
                         "gap_rel_served": b["gap_rel_served"]}
        fam_crit = {f: float(np.mean([cal[m]["crit"] for m in ms])) for f, ms in FAMILIES.items()}
        chosen = min(fam_crit, key=fam_crit.get)
        out = {"thresholds": cal, "family_crit": fam_crit, "chosen_family": chosen, "rows": rows,
               "rule": "threshold: argmin mean(LP cost / LP relaxation cost) on val (block decoder); family: lowest "
                       "seed-averaged criterion at the calibrated thresholds"}
        dump(os.path.join(OUT, f"combo_calib{a.tag}.json"), out)
        np.savez_compressed(os.path.join(OUT, f"combo_calib{a.tag}_arrays.npz"), **{k.replace("|", "__"): v for k, v in arrays.items()})
        print("calibrated thresholds:", {k: v["threshold"] for k, v in cal.items()}, flush=True)
        print("family criterion:", fam_crit, "-> chosen", chosen, flush=True)

    elif a.stage == "e2e":
        d = load(os.path.join(ROOT, f"{a.split}.npz"))
        cal = json.load(open(os.path.join(OUT, "combo_calib.json")))
        chosen = cal["chosen_family"]
        oracle = HourlyOracle(cfg, a.workers)
        rows, arrays = [], {}

        def ths_of(name):
            th = {0.5, cal["thresholds"][name]["threshold"]}
            fam = [f for f, ms in FAMILIES.items() if name in ms][0]
            return sorted(set(THRESHOLDS) if fam == chosen else th)
        models = FAMILIES[chosen] + [m for f, ms in FAMILIES.items() if f != chosen for m in ms]
        e2e_eval(models, d, a.split, oracle, sysm, tr, ths_of, arrays, rows)
        # reference: the MILP's own commitment through the same LP
        n = len(d["load"])
        c, sh, so = oracle.scores(d, np.arange(n), d["u"], np.arange(n) + KEYS[a.split])
        arrays["milp|ref|0"] = np.stack([c, sh, so, d["u"].sum((1, 2)) / T])
        r = e2e_metrics(c, sh, so, d["obj"], d["u"].sum((1, 2)).mean() / T)
        r.update(model="milp", decoder="ref", threshold=0)
        rows.append(r)
        print("  MILP commitment:", r, flush=True)
        oracle.close()
        dump(os.path.join(OUT, f"combo_e2e_{a.split}{a.tag}.json"), {"rows": rows, "chosen_family": chosen,
                                                                     "thresholds": {k: v["threshold"] for k, v in cal["thresholds"].items()}})
        np.savez_compressed(os.path.join(OUT, f"combo_e2e_{a.split}{a.tag}_arrays.npz"),
                            **{k.replace("|", "__"): v for k, v in arrays.items()})

    elif a.stage == "fix":
        d = load(os.path.join(ROOT, f"{a.split}.npz"))
        n = min(a.n, len(d["load"]))
        sub = {k: v[:n] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}
        src = json.load(open(os.path.join(OUT, "combo_valfix.json")))["chosen_source"]
        src_names = {"bce": ["bce"] * 3, "st": [f"st_s{s}" for s in SEEDS], "rl": [f"rl_s{s}" for s in SEEDS]}[src]
        harms = [harm_from_file(os.path.join(OUT, f"combo_harm_s{s}.pt")) for s in SEEDS]
        pb = probs("bce", sub, a.split, tr, sysm)
        p_st = [probs(f"st_s{s}", sub, a.split, tr, sysm) for s in SEEDS]
        p_rl = [probs(f"rl_s{s}", sub, a.split, tr, sysm) for s in SEEDS]
        p_mrl = probs("milp_rl", sub, a.split, tr, sysm)
        p_src = [probs(nm, sub, a.split, tr, sysm) for nm in src_names]
        jobs = []
        for i in range(n):
            sc = scenario_from(d, i)
            specs = []
            hb = [harm_scores(harms[s], ff, pb[i:i + 1], d, [i])[0] for s in SEEDS]
            hsrc = hb if src == "bce" else [harm_scores(harms[s], ff, p_src[s][i:i + 1], d, [i])[0] for s in SEEDS]
            for s in SEEDS:
                # confirmations (components as published, one seed of each trainable part)
                specs.append(("harm+guard", s, 0.90, rule_fixings("harm", 0.90, pb[i], sc, sysm, hb[s]), False))
                specs.append(("harm+lp", s, 0.95, rule_fixings("harm", 0.95, pb[i], sc, sysm, hb[s]), True))
                specs.append(("st asym+guard", s, 0.95, rule_fixings("asym", 0.95, p_st[s][i], sc, sysm), False))
                specs.append(("rl rac", s, 0.95, rule_fixings("rac", 0.95, p_rl[s][i], sc, sysm), False))
                # combined pipeline: chosen probabilities + harm ranking + adequacy guard + LP-relaxation guard
                for q in (0.95, 0.98):
                    if src == "bce" and q == 0.95:
                        continue                       # identical to harm+lp @ 95 %
                    specs.append((f"combo {src}+harm+lp", s, q, rule_fixings("harm", q, p_src[s][i], sc, sysm, hsrc[s]), True))
                for q in (0.95, 0.98):                 # the same without the LP-relaxation guard (faster point)
                    if src == "bce" and q == 0.95:
                        continue
                    specs.append((f"combo {src}+harm+guard", s, q, rule_fixings("harm", q, p_src[s][i], sc, sysm, hsrc[s]), False))
            # RACLearn-style baselines (MILP-label BCE, deterministic) and the original REINFORCE ranking model
            specs.append(("rac", 0, 0.90, rule_fixings("rac", 0.90, pb[i], sc, sysm), False))
            specs.append(("rac", 0, 0.95, rule_fixings("rac", 0.95, pb[i], sc, sysm), False))
            specs.append(("rac+lp", 0, 0.95, rule_fixings("rac", 0.95, pb[i], sc, sysm), True))
            specs.append(("rac+lp", 0, 0.98, rule_fixings("rac", 0.98, pb[i], sc, sysm), True))
            specs.append(("milp_rl rac", 0, 0.95, rule_fixings("rac", 0.95, p_mrl[i], sc, sysm), False))
            jobs.append((i, sc, specs, True))
        path = os.path.join(OUT, f"combo_fix_{a.split}{a.tag}.jsonl")
        run_fixing(jobs, path, a.workers)
        rows, common = summarize_fix(path, d["obj"], n)
        for r in rows:
            print(f"  {r['config']:22s} s{r['seed']:2d} {r['target']:.2f}  fixed {r['fixed_share']:5.1f}%  gap mean "
                  f"{r['gap_mean']:7.3f}%  median {r['gap_median']:6.3f}%  served {r['served']:5.1f}%  "
                  f"speed-up {r['speedup']:5.2f}x", flush=True)
        dump(os.path.join(OUT, f"combo_fix_{a.split}{a.tag}.json"), {"rows": rows, "n": len(common), "source": src})
