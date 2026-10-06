"""Untimed overheads of our pipelines, measured per instance, for the Learning-to-Fix-style runtime
(T_m = inference + every guard / repair + downstream optimisation).

    python scripts/uc_papereval_overhead.py            # uc12 test_fresh + test (120 each) and uc24 test (40)

What the earlier evaluations timed (read from the code): reduced-MILP wall time (+ the fallback re-solve), the
LP-relaxation guard's LPs (lp_guard_s in uc12; t_guard in uc24, which also includes the min up/down row release);
uc24 also adds the LP relaxation used as GNN input. Not timed (measured here, single process, one instance at a
time, as in deployment):
  t_rel    LP relaxation of the instance (GNN / error-cost features; uc12 only, uc24 stores t_rel)
  t_gnn    GNN forward pass for one instance (featurisation included)
  t_harm   error-cost features (FixFeaturizer) + harm-ensemble scoring for one instance
  t_rank   ranking + building the fixings (fix_from_ranking)
  t_adeq   adequacy guard (OFF fixes released in merit order)
  t_rows   min up/down row release (release_conflicting_rows; untimed in uc12, timed in uc24)
  t_block  block adequacy repair + min up/down repair of one schedule (end-to-end decoding)
  LP times of the end-to-end pipelines (dispatch LP per distinct schedule), with their costs re-computed and checked
  against the stored per-instance arrays.
Load calibration: the earlier runs were timed on a busier machine. For the first n_cal instances the LP-relaxation
guard of RACLearn + LP guard at a 95 % target is recomputed (same fixings, same LPs) and its seconds compared with
the recorded lp_guard_s; for uc24 the LP relaxation is re-timed and compared with the stored t_rel. The median ratio
(recorded / now) scales the overheads measured here to the conditions of the earlier runs.
Output: results/papereval_overhead.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from otsl.b3 import load_b3, rep_block  # noqa: E402
from otsl.combo import (THRESHOLDS, adequacy_guard, block_decode, harm_from_file, mud_decode,  # noqa: E402
                        rule_fixings, solver_guards)
from otsl.fixpolicy import FixFeaturizer, fix_from_ranking, release_conflicting_rows  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1  # noqa: E402

OUT = os.path.join(ROOT, "results", "papereval_overhead.json")


def one(d, i):
    return {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}


def tic():
    return time.perf_counter()


def uc12(split, n, n_cal, sysm, m, tr_feat, log):
    from uc_combo_eval import model_path
    from uc_constrained import load_model1
    d = load(os.path.join(ROOT, "data/generated/uc12", f"{split}.npz"))
    n = min(n, len(d["load"]))
    T = 12
    arr = np.load(os.path.join(ROOT, "results/uc12", f"combo_e2e_{split}_arrays.npz"))
    gnn = {nm: load_model1(model_path(nm), sysm, tr_feat, T) for nm in ("bce", "rl_s0", "rl_s1", "rl_s2", "lag_s0",
                                                                         "lag_s1", "lag_s2")}
    ens = harm_from_file(os.path.join(ROOT, "results/uc12/combo_harm_s0.pt"))
    ff = FixFeaturizer(sysm, T)
    # batched probabilities (identical to the evaluation runs) for decoding
    P = {nm: g.predict(d).astype(np.float64) for nm, g in gnn.items()}
    rec = {k: [] for k in ("t_rel", "rel_check", "t_gnn", "t_harm", "t_rank", "t_adeq", "t_rows", "t_block",
                           "t_lp", "cost_check")}
    e2e = {}
    for i in range(n):
        sc = scenario_from(d, i)
        t0 = tic()
        rel = m.solve_dispatch(sc, None, relax=True)
        rec["t_rel"].append(tic() - t0)
        rec["rel_check"].append(abs(rel.obj - d["c_rel"][i]) / d["c_rel"][i])
        oi = one(d, i)
        t0 = tic()
        with torch.no_grad():
            pb = gnn["bce"].predict(oi)[0].astype(np.float64)
        rec["t_gnn"].append(tic() - t0)
        t0 = tic()
        h = ens.score(ff(pb, d, i)[None])[0]
        rec["t_harm"].append(tic() - t0)
        t0 = tic()
        fix = fix_from_ranking(h, (pb > 0.5).astype(int), 0.95)
        rec["t_rank"].append(tic() - t0)
        t0 = tic()
        fix, _ = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
        rec["t_adeq"].append(tic() - t0)
        t0 = tic()
        release_conflicting_rows(fix, sysm, sc.u0)
        rec["t_rows"].append(tic() - t0)
        t0 = tic()
        block_decode(P["lag_s0"][i:i + 1], oi, sysm, 0.6)
        rec["t_block"].append(tic() - t0)
        # end-to-end, seed 0: every distinct schedule of the reported decoders through the exact LP
        scheds = {"rl|mud|0.5": mud_decode(P["rl_s0"][i:i + 1], oi, sysm, 0.5)[0],
                  "rl|block|0.5": block_decode(P["rl_s0"][i:i + 1], oi, sysm, 0.5)[0]}
        for th in THRESHOLDS:
            scheds[f"lag|block|{th}"] = block_decode(P["lag_s0"][i:i + 1], oi, sysm, th)[0]
        lp_t, cc = [], []
        cache = {}
        for key, u in scheds.items():
            b = u.tobytes()
            if b not in cache:
                t0 = tic()
                sol = m.solve_dispatch(sc, u)
                lp_t.append(tic() - t0)
                cache[b] = sol.obj
            mdl, dec, th = key.split("|")
            stored = arr[f"{mdl}_s0__{dec}__{th}"][0, i]
            cc.append(abs(cache[b] - stored) / stored)
        rec["t_lp"].append(float(np.mean(lp_t)))
        rec["cost_check"].append(float(max(cc)))
        # distinct LPs of the screening pipeline (lag, 7 thresholds), per seed
        for s in (0, 1, 2):
            us = {block_decode(P[f"lag_s{s}"][i:i + 1], oi, sysm, th)[0].tobytes() for th in THRESHOLDS}
            e2e.setdefault(f"screen_nlp_s{s}", []).append(len(us))
        if (i + 1) % 20 == 0:
            log(f"  {split} {i + 1}/{n}: t_rel {np.mean(rec['t_rel']):.3f}s t_gnn {np.mean(rec['t_gnn']):.4f}s "
                f"t_harm {np.mean(rec['t_harm']):.4f}s t_adeq {np.mean(rec['t_adeq']):.4f}s t_rows "
                f"{np.mean(rec['t_rows']):.4f}s t_block {np.mean(rec['t_block']):.4f}s t_lp {np.mean(rec['t_lp']):.3f}s "
                f"cost check max {max(rec['cost_check']):.2e} rel check max {max(rec['rel_check']):.2e}")
    # load calibration: RACLearn + LP guard at 95 %, same fixings and LPs as the recorded run
    recs = {}
    for line in open(os.path.join(ROOT, "results/uc12", f"combo_fix_{split}.jsonl")):
        r = json.loads(line)
        if r["config"] == "rac+lp" and r["ratio"] == 0.95:
            recs[r["i"]] = r
    pb_all = P["bce"]
    cal = []
    for i in range(min(n_cal, n)):
        sc = scenario_from(d, i)
        fix = rule_fixings("rac", 0.95, pb_all[i], sc, sysm)
        _, info, secs = solver_guards(m, sysm, sc, fix)
        r = recs[i]
        cal.append(dict(i=i, now_s=secs, then_s=r["lp_guard_s"], same=(info["released_rows"] == r["released_rows"]
                                                                      and info["released_lp"] == r["released_lp"])))
    ratio = [c["then_s"] / c["now_s"] for c in cal if c["same"] and c["now_s"] > 0]
    log(f"  {split} calibration: {sum(c['same'] for c in cal)}/{len(cal)} identical guard outcomes, "
        f"recorded / now = {np.median(ratio):.3f} (median), {np.mean(ratio):.3f} (mean)")
    return dict(n=n, **{k: [float(x) for x in v] for k, v in rec.items()}, **e2e, calibration=cal,
                load_factor=float(np.median(ratio)))


def uc24(sysm, log, n_cal=10):
    from uc_b3_train import strip
    from uc_model1 import candidates_from_probs
    d = load_b3(os.path.join(ROOT, "data/generated/uc24/test.npz"))
    tr = strip(load_b3(os.path.join(ROOT, "data/generated/uc24/train.npz")))
    T, n = 24, len(d["load"])
    m = UCModel(sysm, T=T, network=True)
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    g = build_uc_model1(sysm, feat, T, "gnn", seed=0)
    g.net.load_state_dict(torch.load(os.path.join(ROOT, "results/uc24/b3_rl_selected.pt")))
    p_rl = np.load(os.path.join(ROOT, "results/uc24/b3_probs_test_lf_rl.npy")).astype(np.float64)
    p_bce = np.load(os.path.join(ROOT, "results/uc24/b3_probs_test_lf_bce.npy")).astype(np.float64)
    rng = np.random.default_rng(0)                       # the order of scripts/uc_b3_train.py --stage test
    cl_bce = candidates_from_probs(p_bce, 8, rng)        # (consumes the generator as in the original run)
    cl_rl = candidates_from_probs(p_rl, 8, rng)
    rec = {k: [] for k in ("t_gnn", "prob_check", "t_adeq", "t_block", "t_lp", "cost_block", "shed_block",
                           "short_block", "cost_screen", "shed_screen", "short_screen", "n_screen", "t_screen_lp",
                           "t_screen_block")}
    cal = []
    for i in range(n):
        sc = scenario_from(d, i)
        if i < n_cal:
            rel = m.solve_dispatch(sc, None, relax=True)
            cal.append(dict(i=i, now_s=rel.time, then_s=float(d["t_rel"][i])))
        oi = one(d, i)
        t0 = tic()
        with torch.no_grad():
            p = g.predict(oi)[0].astype(np.float64)
        rec["t_gnn"].append(tic() - t0)
        rec["prob_check"].append(float(np.abs(p - p_rl[i]).max()))
        # adequacy guard on a 95 % confidence fixing of the imitation model (uc_b3_fix's guarded rankings)
        pb = p_bce[i]
        err = np.minimum(pb, 1 - pb)
        fix = fix_from_ranking(err, (pb > 0.5).astype(int), 0.95)
        t0 = tic()
        adequacy_guard(fix, d["load"][i], d["avail"][i], d["sr"][i], sysm)
        rec["t_adeq"].append(tic() - t0)
        t0 = tic()
        u = rep_block((p_rl[i] > 0.5).astype(np.int8), d, i, sysm)
        rec["t_block"].append(tic() - t0)
        t0 = tic()
        sol = m.solve_dispatch(sc, u)
        rec["t_lp"].append(tic() - t0)
        rec["cost_block"].append(sol.obj)
        rec["shed_block"].append(sol.shed)
        rec["short_block"].append(sol.short)
        best, tl, tb = None, 0.0, 0.0
        for c in cl_rl[i]:
            t0 = tic()
            uc = rep_block(c, d, i, sysm)
            tb += tic() - t0
            t0 = tic()
            s_ = m.solve_dispatch(sc, uc)
            tl += tic() - t0
            if best is None or s_.obj < best.obj:
                best = s_
        rec["cost_screen"].append(best.obj)
        rec["shed_screen"].append(best.shed)
        rec["short_screen"].append(best.short)
        rec["n_screen"].append(len(cl_rl[i]))
        rec["t_screen_lp"].append(tl)
        rec["t_screen_block"].append(tb)
        if (i + 1) % 10 == 0:
            log(f"  uc24 {i + 1}/{n}: t_gnn {np.mean(rec['t_gnn']):.4f}s t_adeq {np.mean(rec['t_adeq']):.4f}s "
                f"t_block {np.mean(rec['t_block']):.4f}s t_lp {np.mean(rec['t_lp']):.3f}s screen LPs "
                f"{np.mean(rec['n_screen']):.1f} in {np.mean(rec['t_screen_lp']):.2f}s; prob check {max(rec['prob_check']):.1e}")
    ratio = [c["then_s"] / c["now_s"] for c in cal]
    log(f"  uc24 calibration (LP relaxation, stored t_rel / now): median {np.median(ratio):.3f}")
    return dict(n=n, **{k: [float(x) for x in v] for k, v in rec.items()}, calibration=cal,
                load_factor=float(np.median(ratio)), n_bce_candidates=float(np.mean([len(c) for c in cl_bce])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n_cal", type=int, default=20)
    ap.add_argument("--parts", default="test_fresh,test,uc24")
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(1)
    t00 = time.time()

    def log(msg):
        print(f"[{time.time() - t00:.0f}s] {msg}", flush=True)

    sysm = load_rts_gmlc()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    parts = a.parts.split(",")
    if any(p in ("test_fresh", "test") for p in parts):
        from uc_constrained import setup, strip
        _, tr_full, *_ = setup()
        tr_feat = strip(tr_full)
        m = UCModel(sysm, T=12, network=True)
        for split in ("test_fresh", "test"):
            if split in parts:
                out[split] = uc12(split, a.n, a.n_cal, sysm, m, tr_feat, log)
                json.dump(out, open(OUT, "w"), indent=0)
    if "uc24" in parts:
        out["uc24"] = uc24(sysm, log)
        json.dump(out, open(OUT, "w"), indent=0)
    log("done")
