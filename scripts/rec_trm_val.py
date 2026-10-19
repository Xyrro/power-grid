"""Validation pass of the TRM study (arm A): decode, price with the dispatch LP, select (validation only).

    python3 scripts/rec_trm_val.py --tag main          # a trained model (data/generated/recursive/models/<tag>.pt)
    python3 scripts/rec_trm_val.py --tag m1x           # baseline: the m1x temporal-GNN ensemble (stored val probs)
    python3 scripts/rec_trm_val.py --tag lpround       # baseline: LP rounding + repair + 5 LPs (no learning)

360 instances (val + val_extra + val_extra2), gap to the stored validation MILP objective C* (60 s, 0.1 %).
Every candidate schedule is priced plainly (1 LP) and with the LP-guided adequacy repair (otsl.trm.lp_repair: only
when the LP sheds or misses reserve, <= 2 extra LPs): pipelines "plain" and "rep".
For a model, with the probabilities after supervision steps 1..K_MAX (non-recursive models: one pass):
  phase 1  decoder (viterbi | threshold) x threshold grid at the training step count k_def = n_sup; for each pipeline
           (plain, rep) the lowest mean gap (ties within 0.01 pp: viterbi at 0.5);
  phase 2  test-time compute: the selected decoder and threshold (rep) after k = 1, 2, 4, 8, 16, 32 steps;
  phase 3  screening with <= 5 LPs (+ repair of the cheapest if it fails): (a) the schedules of the last 5 supervision
           steps up to k_def, (b) 5 thresholds of the selected decoder at k_def by greedy forward selection;
  halting  the learned halting head: stop at the first step with q > 0 (cap K_MAX).
Probability metrics against the model's targets (BCE, wrong unit-hours) at every step.
Outputs: data/generated/recursive/val/<tag>.npz (per-candidate cost / shed / short / extra LPs),
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
from otsl.base import lp_round_candidates  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

OUT = os.path.join("data", "generated", "recursive")
RES = os.path.join("results", "recursive")
K_MAX = 32
K_CURVE = (1, 2, 4, 8, 16, 32)
TH = {"viterbi": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), "threshold": (0.3, 0.4, 0.5, 0.6, 0.7),
      "blend0.3": (0.3, 0.5), "blend0.5": (0.3, 0.5), "blend0.7": (0.3, 0.5)}
LPR = (0.001, 0.05, 0.2, 0.5, 0.8)        # LP-rounding thresholds (otsl.base.E2E_TH)


class Pricer:
    """dispatch LPs of (instance, schedule), memoised; LP-guided repair of failing schedules, memoised"""

    def __init__(self, sysm, d):
        self.s = sysm
        self.m = UCModel(sysm, T=d["load"].shape[1])
        self.d, self.cache, self.rcache, self.n, self.t = d, {}, {}, 0, 0.0

    def sc(self, i):
        d = self.d
        return UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])

    def __call__(self, i, u):
        key = (int(i), np.asarray(u, np.int8).tobytes())
        if key not in self.cache:
            t0 = time.time()
            sol = self.m.solve_dispatch(self.sc(i), u)
            self.t += time.time() - t0
            self.n += 1
            self.cache[key] = (float(sol.obj), float(sol.shed), float(sol.short), sol)
        return self.cache[key][:3]

    def repaired(self, i, u):
        """(cost, shed, short, extra LPs) after the LP-guided repair (no-op when the schedule is served)"""
        key = (int(i), np.asarray(u, np.int8).tobytes())
        c, sh, so = self(i, u)
        if sh < 1e-6 and so < 1e-6:
            return c, sh, so, 0
        if key not in self.rcache:
            _, s2, n, t = trm.lp_repair(self.m, self.sc(i), np.asarray(u, np.int8), self.s, self.cache[key][3])
            self.n += n
            self.t += t
            self.rcache[key] = (float(s2.obj), float(s2.shed), float(s2.short), n)
        return self.rcache[key]


def metrics(arr, ref):
    c, sh, so = arr[:, 0], arr[:, 1], arr[:, 2]
    gap = (c - ref) / ref * 100
    served = (sh < 1e-6) & (so < 1e-6)
    out = dict(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_max=float(gap.max()),
               gap_p90=float(np.percentile(gap, 90)), served=float(served.mean() * 100),
               gap_mean_served=float(gap[served].mean()) if served.any() else float("nan"), n_gt1=int((gap > 1).sum()))
    if arr.shape[1] > 3:
        out["extra_lps"] = float(arr[:, 3].mean())
    return out


def greedy_screen(costs, k=5):
    """greedy forward selection of k candidates (columns of costs [N, C]) minimising the mean of the row-wise min"""
    chosen, best = [], np.full(costs.shape[0], np.inf)
    for _ in range(min(k, costs.shape[1])):
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


class Evaluator:
    def __init__(self, va, sysm, price, idx):
        self.va, self.s, self.price, self.idx = va, sysm, price, idx
        self.store, self.rows, self.sched = {}, {}, {}

    def candidate(self, name, fn):
        """schedule fn(i) for every instance -> plain and repaired arrays under name|plain, name|rep"""
        if name + "|plain" in self.store:
            return
        us = [fn(i) for i in self.idx]
        self.sched[name] = us
        plain = np.array([self.price(i, u) for i, u in zip(self.idx, us)])
        rep = np.array([self.price.repaired(i, u) for i, u in zip(self.idx, us)])
        ref = self.va["obj"][self.idx]
        for kind, arr in (("plain", plain), ("rep", rep)):
            self.store[f"{name}|{kind}"] = arr
            self.rows[f"{name}|{kind}"] = metrics(arr, ref)

    def screen(self, name, names, kind):
        """cheapest of the candidates `names` (plain prices), then the repair of that schedule if it fails"""
        ref = self.va["obj"][self.idx]
        C = np.stack([self.store[f"{n}|plain"][:, 0] for n in names], 1)
        j = C.argmin(1)
        out, nlp = [], []
        for r, i in enumerate(self.idx):
            u = self.sched[names[j[r]]][r]
            distinct = len({self.sched[n][r].tobytes() for n in names})
            if kind == "rep":
                c, sh, so, n = self.price.repaired(i, u)
                out.append((c, sh, so, distinct + n))
            else:
                c, sh, so = self.price(i, u)
                out.append((c, sh, so, distinct))
        arr = np.array(out)
        self.store[f"{name}|{kind}"] = arr
        self.rows[f"{name}|{kind}"] = dict(metrics(arr[:, :3], ref), lps=float(arr[:, 3].mean()))
        return self.rows[f"{name}|{kind}"]


def select(rows, k_def, kind):
    cands = [(how, th) for how, ths in TH.items() for th in ths]
    gaps = {c: rows[f"{c[0]}|{c[1]}|k{k_def}|{kind}"]["gap_mean"] for c in cands}
    best = min(cands, key=lambda c: gaps[c])
    if gaps[best] > gaps[("viterbi", 0.5)] - 0.01:
        best = ("viterbi", 0.5)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--res", default=RES, help="folder for trm_val_<tag>.json (e.g. a round's own results folder)")
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="ablation pass: viterbi 0.2 / 0.3 / 0.5 and blend0.5 only, no "
                    "5-threshold screening (same metrics otherwise)")
    a = ap.parse_args()
    global TH
    if a.quick:
        TH = {"viterbi": (0.2, 0.3, 0.5), "blend0.5": (0.3, 0.5)}
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(OUT, "val"), exist_ok=True)
    sysm = load_rts_gmlc()
    va = trm.subset(trm.val_data(), np.arange(a.n))
    idx = np.arange(a.n)
    price = Pricer(sysm, va)
    ev = Evaluator(va, sysm, price, idx)
    t0 = time.time()
    sel, info = {}, dict(tag=a.tag)
    if a.tag == "lpround":
        def sc(i):
            return UCScenario(load=va["load"][i], avail=va["avail"][i], u0=va["u0"][i], sr=va["sr"][i])
        cands = {i: lp_round_candidates(va["u_rel"][i], sc(i), sysm) for i in idx}
        names = []
        for j, th in enumerate((0.001, 0.05, 0.2, 0.5, 0.8)):
            ev.candidate(f"round|{th}", lambda i, j=j: cands[i][j][1])
            names.append(f"round|{th}")
        for kind in ("plain", "rep"):
            sel[f"screen5_{kind}"] = ev.screen("screen5", names, kind)
    else:
        if a.tag == "m1x":
            P1 = np.load(os.path.join("data", "generated", "uc12_m1x", "probs", "pol_n2000_gnnt_ens5_probs.npz"))["val"]
            P = np.repeat(P1[:a.n, None].astype(np.float64), K_MAX, 1)
            sgs, k_def, recursive = [None] * a.n, 1, False
            info.update(source="pol_n2000_gnnt_ens5 (5 x CommitGNNT, 500 MILP + 1,500 polished)")
        else:
            ck_path = os.path.join(OUT, "models", f"{a.tag}.pt")
            model, feats, cfg, ck = trm.load_model(ck_path, sysm, "cpu", which="ema")
            # record exactly which weights were evaluated: the checkpoint's training step, its run's best step and the
            # configured length, so a swapped or overwritten checkpoint shows up in the result file
            ck_log = ck.get("log") or {}
            info.update(checkpoint=dict(path=ck_path, step=int(ck.get("step", -1)), steps_cfg=int(cfg.steps),
                                        best_step=ck_log.get("best_step"), best_val=ck_log.get("best_val"),
                                        mtime=os.path.getmtime(ck_path)))
            xt, xb = feats(va)
            recursive = cfg.mode == "trm"
            ti = time.time()
            P, Q = trm.run_steps(model, xt, xb, K_MAX if recursive else 1, bs=40)
            info["infer_s_batched"] = time.time() - ti
            if not recursive:
                P = np.repeat(P, K_MAX, 1)
            sgs = trm.subgroups(sysm, va["u0"]) if cfg.sym == "count" else [None] * a.n
            S = trm.subgroup_mean_matrix(sysm, va["u0"]) if cfg.sym == "count" else None
            info.update(cfg=cfg.to_json(), params=trm.n_params(model),
                        prob=prob_metrics(P, trm.targets(sysm, va, va["u"], cfg.sym), S))
            k_def = cfg.n_sup if recursive else 1
            if recursive:
                info["halt_q"] = Q.tolist()

        def dec(how, th, k):
            return lambda i: trm.decode(P[i, k - 1], va, i, sysm, th, how, sgs[i])

        for how, ths in TH.items():
            for th in ths:
                ev.candidate(f"{how}|{th}|k{k_def}", dec(how, th, k_def))
        for kind in ("plain", "rep"):
            how, th = select(ev.rows, k_def, kind)
            sel[kind] = dict(decoder=how, th=th, k_def=k_def, one_lp=ev.rows[f"{how}|{th}|k{k_def}|{kind}"])
            for hw in TH:
                bt = min(TH[hw], key=lambda t: ev.rows[f"{hw}|{t}|k{k_def}|{kind}"]["gap_mean"])
                sel[kind][f"best_{hw}"] = dict(th=bt, **ev.rows[f"{hw}|{bt}|k{k_def}|{kind}"])
            # 5 thresholds of the selected decoder (greedy on validation, plain prices)
            hs = how if len(TH[how]) >= 5 else "viterbi"
            if len(TH.get(hs, ())) >= 5:
                names = [f"{hs}|{t}|k{k_def}" for t in TH[hs]]
                C = np.stack([ev.store[f"{n}|plain"][:, 0] for n in names], 1)
                ch = greedy_screen(C, 5)
                sel[kind]["screen_dec"] = hs
                sel[kind]["screen_th"] = [TH[hs][c] for c in ch]
                sel[kind]["screen_th_metrics"] = ev.screen(f"th5_{kind}", [names[c] for c in ch], kind)
        # mixed screening: the model's selected 1-LP schedule + 4 LP roundings (greedy on validation), <= 5 LPs
        lpr = os.path.join(OUT, "val", "lpround.npz")
        if os.path.exists(lpr):
            def sc(i):
                return UCScenario(load=va["load"][i], avail=va["avail"][i], u0=va["u0"][i], sr=va["sr"][i])
            rc = {i: lp_round_candidates(va["u_rel"][i], sc(i), sysm) for i in idx}
            for j, t in enumerate(LPR):
                ev.candidate(f"round|{t}", lambda i, j=j: rc[i][j][1])
            for kind in ("plain", "rep"):
                h1, t1 = sel[kind]["decoder"], sel[kind]["th"]
                own = f"{h1}|{t1}|k{k_def}"
                C = np.stack([np.minimum(ev.store[f"{own}|plain"][:, 0], ev.store[f"round|{t}|plain"][:, 0]) for t in LPR], 1)
                chosen = []
                best = ev.store[f"{own}|plain"][:, 0].copy()
                for _ in range(4):
                    j = int(np.argmin([np.mean(np.minimum(best, ev.store[f"round|{t}|plain"][:, 0])) if t not in chosen
                                       else np.inf for t in LPR]))
                    chosen.append(LPR[j])
                    best = np.minimum(best, ev.store[f"round|{LPR[j]}|plain"][:, 0])
                sel[kind]["mix_th"] = chosen
                sel[kind]["mix_metrics"] = ev.screen(f"mix5_{kind}", [own] + [f"round|{t}" for t in chosen], kind)
        how, th = sel["rep"]["decoder"], sel["rep"]["th"]
        if recursive:
            for k in K_CURVE:
                ev.candidate(f"{how}|{th}|k{k}", dec(how, th, k))
            sel["curve"] = {k: dict(plain=ev.rows[f"{how}|{th}|k{k}|plain"], rep=ev.rows[f"{how}|{th}|k{k}|rep"])
                            for k in K_CURVE}
            for kk in ([k_def] if a.quick else sorted({k_def, min(2 * k_def, K_MAX)})):
                ks = list(range(max(1, kk - 4), kk + 1))
                for k in ks:
                    ev.candidate(f"{how}|{th}|k{k}", dec(how, th, k))
                for kind in ("plain", "rep"):
                    sel[f"screen_steps_k{kk}_{kind}"] = ev.screen(f"steps5_k{kk}_{kind}", [f"{how}|{th}|k{k}" for k in ks], kind)
            kh = np.array([int(np.argmax(np.array(Q[i]) > 0)) + 1 if (np.array(Q[i]) > 0).any() else K_MAX for i in idx])
            ev.candidate("halt_q", lambda i: trm.decode(P[i, kh[i] - 1], va, i, sysm, th, how, sgs[i]))
            sel["halt_q"] = dict(steps_mean=float(kh.mean()), steps_median=float(np.median(kh)),
                                 plain=ev.rows["halt_q|plain"], rep=ev.rows["halt_q|rep"])
    info.update(selection=sel, rows=ev.rows, lps=price.n, lp_s=price.t, wall_s=time.time() - t0, quick=a.quick)
    np.savez_compressed(os.path.join(OUT, "val", f"{a.tag}.npz"), **{k.replace("|", "__"): v for k, v in ev.store.items()})
    os.makedirs(a.res, exist_ok=True)
    with open(os.path.join(a.res, f"trm_val_{a.tag}.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    short = {k: (v.get("decoder"), v.get("th"), round(v.get("one_lp", v).get("gap_mean", np.nan), 3),
                 round(v.get("one_lp", v).get("served", np.nan), 1)) if isinstance(v, dict) and ("one_lp" in v or "gap_mean" in v)
             else None for k, v in sel.items() if k != "curve"}
    print(json.dumps(dict(tag=a.tag, selection=short), default=float), flush=True)
    if "curve" in sel:
        print("curve (rep)", json.dumps({k: (round(v["rep"]["gap_mean"], 3), round(v["rep"]["served"], 1))
                                          for k, v in sel["curve"].items()}))
    for k, v in sel.items():
        if isinstance(v, dict) and "screen_th_metrics" in v:
            print(k, "5 thresholds", v["screen_th"], round(v["screen_th_metrics"]["gap_mean"], 3), round(v["screen_th_metrics"]["served"], 1))
        if isinstance(v, dict) and "mix_metrics" in v:
            print(k, "mix5", v["mix_th"], round(v["mix_metrics"]["gap_mean"], 3), round(v["mix_metrics"]["gap_median"], 3),
                  round(v["mix_metrics"]["served"], 1))


if __name__ == "__main__":
    main()
