"""Validation pass of the TRM study (arm A): decode, price with the dispatch LP, select (validation only).

    python3 scripts/rec_trm_val.py --tag main          # a trained model (data/generated/recursive/models/<tag>.pt)
    python3 scripts/rec_trm_val.py --tag m1x           # baseline: the m1x temporal-GNN ensemble (stored val probs)
    python3 scripts/rec_trm_val.py --tag lpround       # baseline: LP rounding + repair + 5 LPs (no learning)

360 instances (val + val_extra + val_extra2), gap to the stored validation MILP objective C* (60 s, 0.1 %).
For a model, with the probabilities after supervision steps 1..K_MAX (non-recursive models: one pass):
  phase 1  decoder (viterbi | threshold) x threshold grid at the training step count k_def = n_sup -> 1-LP selection:
           lowest mean gap (ties within 0.01 pp: viterbi at 0.5);
  phase 2  test-time compute: the selected decoder and threshold after k = 1, 2, 4, 8, 16, 32 steps;
  phase 3  screening with <= 5 LPs: (a) the schedules of the last 5 supervision steps up to k_def, (b) 5 thresholds
           of the selected decoder at k_def, chosen by greedy forward selection on validation;
  halting  the learned halting head: stop at the first step with q > 0 (cap K_MAX).
Probability metrics against the targets of the model's symmetry handling (BCE, wrong unit-hours) at every step.
Outputs: data/generated/recursive/val/<tag>.npz (per-candidate cost / shed / short, probabilities summary),
results/recursive/trm_val_<tag>.json (metrics and the selection).
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
from otsl import trm  # noqa: E402
from otsl.base import lp_round_screen  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

OUT = os.path.join("data", "generated", "recursive")
RES = os.path.join("results", "recursive")
K_MAX = 32
K_CURVE = (1, 2, 4, 8, 16, 32)
TH = {"viterbi": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), "threshold": (0.3, 0.4, 0.5, 0.6, 0.7)}


class Pricer:
    """dispatch LPs of (instance, schedule), memoised"""

    def __init__(self, sysm, d):
        self.m = UCModel(sysm, T=d["load"].shape[1])
        self.d, self.cache, self.n, self.t = d, {}, 0, 0.0

    def __call__(self, i, u):
        key = (int(i), np.asarray(u, np.int8).tobytes())
        if key not in self.cache:
            d = self.d
            t0 = time.time()
            sol = self.m.solve_dispatch(UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i]), u)
            self.t += time.time() - t0
            self.n += 1
            self.cache[key] = (float(sol.obj), float(sol.shed), float(sol.short))
        return self.cache[key]


def metrics(c, sh, so, ref):
    gap = (np.asarray(c) - ref) / ref * 100
    served = (np.asarray(sh) < 1e-6) & (np.asarray(so) < 1e-6)
    return dict(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_max=float(gap.max()),
                gap_p90=float(np.percentile(gap, 90)), served=float(served.mean() * 100),
                gap_mean_served=float(gap[served].mean()) if served.any() else float("nan"),
                n_gt1=int((gap > 1).sum()))


def run_candidates(name, sched_fn, idx, price, store, rows, ref):
    """price schedule sched_fn(i) for every instance; store arrays and metrics under name"""
    if name in store:
        return store[name]
    out = np.array([price(i, sched_fn(i)) for i in idx])
    store[name] = out
    rows[name] = metrics(out[:, 0], out[:, 1], out[:, 2], ref[idx])
    return out


def greedy_screen(costs, k=5):
    """greedy forward selection of k candidates (columns of costs [N, C]) minimising the mean of the row-wise min"""
    chosen = []
    best = np.full(costs.shape[0], np.inf)
    for _ in range(k):
        j = int(np.argmin([np.mean(np.minimum(best, costs[:, c])) if c not in chosen else np.inf
                           for c in range(costs.shape[1])]))
        chosen.append(j)
        best = np.minimum(best, costs[:, j])
    return chosen


def prob_metrics(P, tgt, S):
    out = {}
    for k in range(P.shape[1]):
        p = np.clip(P[:, k], 1e-6, 1 - 1e-6)
        bce = float(-(tgt * np.log(p) + (1 - tgt) * np.log(1 - p)).mean())
        wr = trm.wrong_count(torch.logit(torch.as_tensor(p)), torch.as_tensor(tgt), None if S is None else torch.as_tensor(S))
        out[k + 1] = dict(bce=bce, wrong=float(wr.float().mean()))
    return out


def evaluate_model(tag, sysm, va, price, threads):
    path = os.path.join(OUT, "models", f"{tag}.pt")
    model, feats, cfg, ck = trm.load_model(path, sysm, "cpu", which="ema")
    xt, xb = feats(va)
    t0 = time.time()
    K = K_MAX if cfg.mode == "trm" else 1
    P, Q = trm.run_steps(model, xt, xb, K, bs=40)
    t_inf = time.time() - t0
    if K == 1:
        P = np.repeat(P, K_MAX, 1)
    sgs = trm.subgroups(sysm, va["u0"]) if cfg.sym == "count" else [None] * len(va["load"])
    S = trm.subgroup_mean_matrix(sysm, va["u0"]) if cfg.sym == "count" else None
    tgt = trm.targets(sysm, va, va["u"], cfg.sym)
    info = dict(tag=tag, cfg=cfg.to_json(), params=trm.n_params(model), infer_s_batched=t_inf,
                prob=prob_metrics(P, tgt, S), halt_q=Q.tolist() if cfg.mode == "trm" else None)
    k_def = cfg.n_sup if cfg.mode == "trm" else 1
    return P, Q, sgs, info, k_def


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(OUT, "val"), exist_ok=True)
    sysm = load_rts_gmlc()
    va = trm.subset(trm.val_data(), np.arange(a.n))
    ref = va["obj"]
    idx = np.arange(a.n)
    price = Pricer(sysm, va)
    store, rows, t0 = {}, {}, time.time()
    sel = {}
    if a.tag == "lpround":
        m = price.m
        out, out1 = [], []
        for i in idx:
            sc = UCScenario(load=va["load"][i], avail=va["avail"][i], u0=va["u0"][i], sr=va["sr"][i])
            b = lp_round_screen(m, sysm, sc, va["u_rel"][i])
            out.append((b["cost"], b["shed"], b["short"], b["n_lp"]))
            u = trm.decode(va["u_rel"][i], va, i, sysm, 0.5, "threshold")
            out1.append(price(i, u))
        out, out1 = np.array(out), np.array(out1)
        store["screen5"], store["th0.5"] = out[:, :3], out1
        rows["screen5"] = dict(metrics(out[:, 0], out[:, 1], out[:, 2], ref), lps=float(out[:, 3].mean()))
        rows["th0.5"] = metrics(out1[:, 0], out1[:, 1], out1[:, 2], ref)
        info = dict(tag="lpround")
    else:
        if a.tag == "m1x":
            P1 = np.load(os.path.join("data", "generated", "uc12_m1x", "probs", "pol_n2000_gnnt_ens5_probs.npz"))["val"]
            P = np.repeat(P1[:a.n, None].astype(np.float64), K_MAX, 1)
            sgs = [None] * a.n
            info, k_def = dict(tag="m1x", source="pol_n2000_gnnt_ens5 (5 x CommitGNNT, 500 MILP + 1,500 polished)"), 1
        else:
            P, Q, sgs, info, k_def = evaluate_model(a.tag, sysm, va, price, a.threads)
        recursive = a.tag not in ("m1x",) and info.get("cfg", {}).get("mode") == "trm"
        # phase 1: decoder x threshold at k_def
        for how, ths in TH.items():
            for th in ths:
                run_candidates(f"{how}|{th}|k{k_def}", lambda i: trm.decode(P[i, k_def - 1], va, i, sysm, th, how, sgs[i]),
                               idx, price, store, rows, ref)
        cands = [(how, th) for how, ths in TH.items() for th in ths]
        gaps = {c: rows[f"{c[0]}|{c[1]}|k{k_def}"]["gap_mean"] for c in cands}
        g0 = gaps[("viterbi", 0.5)]
        best = min(cands, key=lambda c: gaps[c])
        if gaps[best] > g0 - 0.01:
            best = ("viterbi", 0.5)
        how, th = best
        sel.update(decoder=how, th=th, k_def=k_def, one_lp=rows[f"{how}|{th}|k{k_def}"])
        # phase 2: test-time compute curve
        if recursive:
            for k in K_CURVE:
                run_candidates(f"{how}|{th}|k{k}", lambda i: trm.decode(P[i, k - 1], va, i, sysm, th, how, sgs[i]),
                               idx, price, store, rows, ref)
            sel["curve"] = {k: rows[f"{how}|{th}|k{k}"] for k in K_CURVE}
            # phase 3a: last 5 steps up to k_def (and up to 2 k_def)
            for kk in sorted({k_def, min(2 * k_def, K_MAX)}):
                ks = list(range(max(1, kk - 4), kk + 1))
                pick, nlp = [], []
                for i in idx:
                    us = {trm.decode(P[i, k - 1], va, i, sysm, th, how, sgs[i]).tobytes(): k for k in ks}
                    res = [price(i, np.frombuffer(b, np.int8).reshape(P.shape[2:])) for b in us]
                    pick.append(min(res, key=lambda r: r[0]))
                    nlp.append(len(us))
                pick = np.array(pick)
                rows[f"steps5|k{kk}"] = dict(metrics(pick[:, 0], pick[:, 1], pick[:, 2], ref), lps=float(np.mean(nlp)))
                store[f"steps5|k{kk}"] = pick
            sel["screen_steps"] = rows[f"steps5|k{k_def}"]
            # halting head: first step with q > 0
            kh = np.array([int(np.argmax(Q[i] > 0)) + 1 if (Q[i] > 0).any() else K_MAX for i in idx])
            hh = np.array([price(i, trm.decode(P[i, kh[i] - 1], va, i, sysm, th, how, sgs[i])) for i in idx])
            rows["halt_q"] = dict(metrics(hh[:, 0], hh[:, 1], hh[:, 2], ref), steps_mean=float(kh.mean()),
                                  steps_median=float(np.median(kh)))
            store["halt_q"] = hh
        # phase 3b: 5 thresholds of the selected decoder, greedy on validation
        names = [f"{how}|{t}|k{k_def}" for t in TH[how]]
        C = np.stack([store[nm][:, 0] for nm in names], 1)
        ch = greedy_screen(C, 5)
        pick_names = [names[c] for c in ch]
        allc = np.stack([store[nm] for nm in pick_names], 1)
        j = allc[:, :, 0].argmin(1)
        pick = allc[np.arange(len(allc)), j]
        rows["th5"] = metrics(pick[:, 0], pick[:, 1], pick[:, 2], ref)
        store["th5"] = pick
        sel.update(screen_th=[TH[how][c] for c in ch], screen_th_metrics=rows["th5"])
        # the alternative decoder at its own best threshold (for the decoder comparison)
        for hw in TH:
            bt = min(TH[hw], key=lambda t: rows[f"{hw}|{t}|k{k_def}"]["gap_mean"])
            sel[f"best_{hw}"] = dict(th=bt, **rows[f"{hw}|{bt}|k{k_def}"])
    info.update(selection=sel, rows=rows, lps=price.n, lp_s=price.t, wall_s=time.time() - t0)
    np.savez_compressed(os.path.join(OUT, "val", f"{a.tag}.npz"), **{k.replace("|", "__"): v for k, v in store.items()})
    with open(os.path.join(RES, f"trm_val_{a.tag}.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    print(json.dumps(dict(tag=a.tag, selection={k: v for k, v in sel.items() if k != "curve"}), default=float), flush=True)
    if "curve" in sel:
        print("curve", json.dumps({k: (round(v["gap_mean"], 3), round(v["served"], 1)) for k, v in sel["curve"].items()}))


if __name__ == "__main__":
    main()
