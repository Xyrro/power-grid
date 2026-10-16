"""rloop (arm B) validation (val + val_extra + val_extra2, 360 instances; gap to the stored MILP dual bound).

    python3 scripts/rec_rloop_val.py --model main [--decode viterbi|round] [--K 16]   # per-step records of a model
    python3 scripts/rec_rloop_val.py --baselines                                       # LP rounding, m1x end to end
    python3 scripts/rec_rloop_val.py --report                                          # selection table

Model records: results/recursive/rloop_val_<tag>[_<decode>].npz with cost / shed / short / halting probability per
(step, instance). Baselines (same decoding and LP code): LP relaxation rounded at the PGLib thresholds + block repair
+ min up/down repair, best of 5 LPs; the m1x 5-member temporal-GNN ensemble (stored validation probabilities) rounded
at each threshold of a grid + the same repairs, one LP per distinct schedule; the threshold sets for 1, 3, 5 LPs are
chosen here on validation (greedy forward selection on the mean gap); also the ensemble decoded with Viterbi.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rec_rloop_common as C  # noqa: E402
from otsl import rloop  # noqa: E402
from otsl.base import E2E_TH  # noqa: E402

M1X_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def gap(c, db):
    return (np.asarray(c) - db) / np.asarray(c) * 100


def run_model(a):
    sysm = C.system("uc12")
    path = os.path.join(C.OUT, f"{a.model}.pt")
    net, ck, feats = C.load_model(path, sysm)
    va = C.val_data()
    idx = list(range(len(va["load"]))) if a.n == 0 else list(range(a.n))
    insts = C.insts_of(va, idx)
    pool = rloop.LPPool(workers=0)
    mode = a.decode or ck["cfg"].get("decode", "viterbi")
    starts = None
    if a.start == "lpround":        # the LP-rounding baseline's best schedule (5 LPs) as the starting point
        B = dict(np.load(os.path.join(C.RES, "rloop_val_baselines.npz")))
        jbest = B["lpround_c"].argmin(0)
        starts = []
        for i, inst in zip(idx, insts):
            th = float(B["lpround_th"][jbest[i]])
            starts.append(rloop.decode((inst["u_rel"] > th).astype(np.int8), inst, sysm, "round")[1])
        starts = np.stack(starts)
    t0 = time.time()
    out = {k: [] for k in ("cost", "shed", "short", "halt")}
    for s in range(0, len(idx), 60):
        st = rloop.rollout_batch(net, feats, sysm, insts[s:s + 60], a.K, pool, "cpu",
                                 [("va", i) for i in idx[s:s + 60]], mode=mode, bias=a.bias,
                                 starts=None if starts is None else starts[s:s + 60])
        for j, k in enumerate(("cost", "shed", "short", "halt")):
            out[k].append(np.array([[x[j] for x in step] for step in st]))
    res = {k: np.concatenate(v, 1) for k, v in out.items()}
    tag = a.model + (f"_{a.decode}" if a.decode else "") + (f"_b{a.bias:g}" if a.bias else "") + \
        (f"_start-{a.start}" if a.start else "")
    np.savez(os.path.join(C.RES, f"rloop_val_{tag}.npz"), db=va["db"][idx], c_rel=va["c_rel"][idx], **res)
    best = np.minimum.accumulate(res["cost"], 0)
    print(f"{tag}: {time.time() - t0:.0f}s, mean gap best-of-k:",
          [round(float(gap(best[k], va["db"][idx]).mean()), 3) for k in range(a.K)], flush=True)


def lpround_starts(va, insts, sysm, idx):
    """the LP-rounding baseline's best schedule per validation instance (best of its 5 thresholds)"""
    B = dict(np.load(os.path.join(C.RES, "rloop_val_baselines.npz")))
    jb = B["lpround_c"].argmin(0)
    r = range(len(jb))
    return ([rloop.decode((insts[j]["u_rel"] > float(B["lpround_th"][jb[i]])).astype(np.int8), insts[j], sysm, "round")[1]
             for j, i in enumerate(idx)], B["lpround_c"].min(0)[idx], B["lpround_shed"][jb, r][idx],
            B["lpround_short"][jb, r][idx])


def union_starts(va, insts, sysm, idx):
    """the best of the LP-rounding (5 LPs) and m1x (5 LPs, validation-selected thresholds) candidates"""
    B = dict(np.load(os.path.join(C.RES, "rloop_val_baselines.npz")))
    P = np.load(os.path.join(C.DM1X, "probs", "pol_n2000_gnnt_ens5_probs.npz"))["val"].astype(float)
    sel = json.load(open(os.path.join(C.RES, "rloop_val_summary.json")))["m1x_thresholds"]["5"]["th"]
    th = list(B["m1x_th"])
    J = [th.index(t) for t in sel]
    s0, c0, sh0, so0 = lpround_starts(va, insts, sysm, idx)
    for j, i in enumerate(idx):
        jm = J[int(np.argmin(B["m1x_c"][J, i]))]
        if B["m1x_c"][jm, i] < c0[j]:
            s0[j] = rloop.decode((P[i] > th[jm]).astype(np.int8), insts[j], sysm, "round")[1]
            c0[j], sh0[j], so0[j] = B["m1x_c"][jm, i], B["m1x_shed"][jm, i], B["m1x_short"][jm, i]
    return s0, c0, sh0, so0


def run_guided(a):
    """LP-verified local search from the LP-rounding schedule (K LPs), moves ranked by the LP's first-order prediction
    minus lam x the log-odds of a prior: none (learning-free), the m1x ensemble (static), or a recursive model (re-run
    on the LP feedback of the current best)"""
    sysm = C.system("uc12")
    va = C.val_data()
    n = len(va["load"])
    idx = list(range(a.n0, min(a.n1 or n, n)))
    insts = C.insts_of(va, idx)
    starts, c0, sh0, so0 = (union_starts if a.seed == "union" else lpround_starts)(va, insts, sysm, idx)
    P = np.load(os.path.join(C.DM1X, "probs", "pol_n2000_gnnt_ens5_probs.npz"))["val"].astype(float) \
        if a.guide == "m1x" else None
    net = feats = None
    if a.guide in ("net", "net_once"):
        net, ck, feats = C.load_model(os.path.join(C.OUT, f"{a.model}.pt"), sysm)
    lp = rloop.FastLP(sysm, 12)
    K = a.K
    out = {k: np.zeros((K + 1, len(idx))) for k in ("cost", "shed", "short", "time")}
    t0 = time.time()
    for j, i in enumerate(idx):
        recs, _, _ = rloop.guided_search(insts[j], starts[j], K, lp, sysm, a.lam, prior=P[i] if P is not None else None,
                                      net=net, feats=feats, bias=a.bias, net_once=a.guide == "net_once")
        out["cost"][0, j], out["shed"][0, j], out["short"][0, j] = c0[j], sh0[j], so0[j]
        for k in range(K):
            x = recs[min(k, len(recs) - 1)] if recs else None
            if x is None or x["best"] >= c0[j]:
                out["cost"][k + 1, j], out["shed"][k + 1, j], out["short"][k + 1, j] = c0[j], sh0[j], so0[j]
            else:
                out["cost"][k + 1, j], out["shed"][k + 1, j], out["short"][k + 1, j] = x["best"], x["best_shed"], x["best_short"]
            out["time"][k + 1, j] = x["time"] if x else 0.0
    tag = f"guided_{a.guide}" + (f"-{a.model}" if a.guide in ("net", "net_once") else "") + f"_lam{a.lam:g}" + \
        (f"_b{a.bias:g}" if a.bias else "") + ("_seed-union" if a.seed == "union" else "") + \
        (f"_{a.n0}-{idx[-1] + 1}" if (a.n0 or a.n1) else "")
    np.savez(os.path.join(C.RES, f"rloop_val_{tag}.npz"), db=va["db"][idx], idx=np.array(idx), **out)
    db = va["db"][idx]
    print(f"{tag}: {time.time() - t0:.0f}s, mean gap by LPs:",
          [round(float(gap(out["cost"][k], db).mean()), 3) for k in range(K + 1)], flush=True)


def candidates(p, inst, sysm, ths, mode="round"):
    out = []
    for th in ths:
        r = (p > th).astype(np.int8) if mode == "round" else p
        out.append(rloop.decode(r if mode == "round" else p, inst, sysm, mode)[1])
    return out


def run_baselines(a):
    sysm = C.system("uc12")
    va = C.val_data()
    n = len(va["load"])
    insts = C.insts_of(va)
    P = np.load(os.path.join(C.DM1X, "probs", "pol_n2000_gnnt_ens5_probs.npz"))["val"].astype(np.float64)
    lp = rloop.FastLP(sysm, 12)
    rec = dict(db=va["db"], c_rel=va["c_rel"])
    memo = {}

    def price(i, s):
        key = (i, s.tobytes())
        if key not in memo:
            sol = lp.solve(rloop.scenario_of(insts[i]), s)
            memo[key] = (sol["obj"], float(sol["shed"].sum() + sol["spill"].sum()), float(sol["short"].sum()))
        return memo[key]

    t0 = time.time()
    for name, src, ths in (("lpround", "u_rel", E2E_TH), ("m1x", "m1x", M1X_GRID)):
        C3 = np.zeros((len(ths), n, 3))
        for i in range(n):
            p = P[i] if src == "m1x" else va["u_rel"][i]
            for j, s in enumerate(candidates(p, insts[i], sysm, ths)):
                C3[j, i] = price(i, s)
        rec[f"{name}_th"] = np.array(ths)
        rec[f"{name}_c"], rec[f"{name}_shed"], rec[f"{name}_short"] = C3[..., 0], C3[..., 1], C3[..., 2]
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
    V = np.zeros((n, 3))
    for i in range(n):
        V[i] = price(i, rloop.decode(P[i], insts[i], sysm, "viterbi")[1])
    rec["m1x_vit_c"], rec["m1x_vit_shed"], rec["m1x_vit_short"] = V[:, 0], V[:, 1], V[:, 2]
    np.savez(os.path.join(C.RES, "rloop_val_baselines.npz"), **rec)
    print(f"baselines done ({time.time() - t0:.0f}s)", flush=True)


def select_thresholds(c, ths, db, kmax=5):
    """greedy forward selection of threshold subsets (1..kmax) minimising the mean gap of the best candidate"""
    chosen, out = [], {}
    for k in range(1, kmax + 1):
        best = None
        for j in range(len(ths)):
            if j in chosen:
                continue
            g = gap(c[chosen + [j]].min(0), db).mean()
            if best is None or g < best[0]:
                best = (g, j)
        chosen.append(best[1])
        out[k] = dict(th=[float(ths[j]) for j in chosen], gap=float(best[0]))
    return out


def summary(cost, shed, short, db):
    g = gap(cost, db)
    served = (shed < 1e-6) & (short < 1e-6)
    return dict(served=float(served.mean() * 100), gap_mean=float(g.mean()), gap_median=float(np.median(g)),
                gap_max=float(g.max()), n=int(len(g)))


def report(a):
    rows = {}
    bp = os.path.join(C.RES, "rloop_val_baselines.npz")
    sel = {}
    if os.path.exists(bp):
        B = dict(np.load(bp))
        db = B["db"]
        for name in ("lpround", "m1x"):
            c, sh, so = B[f"{name}_c"], B[f"{name}_shed"], B[f"{name}_short"]
            ths = B[f"{name}_th"]
            if name == "lpround":
                j = c.argmin(0)
                rows["LP rounding + repair, best of 5 LPs"] = summary(c.min(0), sh[j, range(len(db))], so[j, range(len(db))], db)
            else:
                sel = select_thresholds(c, ths, db)
                for k, v in sel.items():
                    J = [list(ths).index(t) for t in v["th"]]
                    jj = np.array(J)[c[J].argmin(0)]
                    rows[f"m1x ensemble + repair, {k} LP(s) (th {v['th']})"] = summary(
                        c[jj, range(len(db))], sh[jj, range(len(db))], so[jj, range(len(db))], db)
        rows["m1x ensemble, Viterbi decoding + repair, 1 LP"] = summary(B["m1x_vit_c"], B["m1x_vit_shed"], B["m1x_vit_short"], db)
    for f in sorted(os.listdir(C.RES)):
        if f.startswith("rloop_val_") and f.endswith(".npz") and f != "rloop_val_baselines.npz":
            R = dict(np.load(os.path.join(C.RES, f)))
            if f.startswith("rloop_val_guided_") and "-" in f.split("_lam")[-1]:
                continue                                 # shards: merged by --merge
            off = 1 if ("start-" in f or f.startswith("rloop_val_guided_")) else 0   # row 0: the starting schedule
            K = R["cost"].shape[0] - off
            for k in sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, K} & set(range(1 - off, K + 1))):
                c = R["cost"][:k + off]
                j = c.argmin(0)
                n = c.shape[1]
                rows[f"{f[10:-4]} K={k}"] = summary(c.min(0), R["shed"][:k + off][j, range(n)],
                                                    R["short"][:k + off][j, range(n)], R["db"])
    json.dump(dict(rows=rows, m1x_thresholds=sel), open(os.path.join(C.RES, "rloop_val_summary.json"), "w"), indent=1)
    w = max(len(k) for k in rows) if rows else 10
    print(f"{'method':{w}s}  served  mean   median  max    n")
    for k, v in rows.items():
        print(f"{k:{w}s}  {v['served']:5.1f}  {v['gap_mean']:.3f}  {v['gap_median']:.3f}  {v['gap_max']:6.2f}  {v['n']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--decode", default=None, choices=[None, "viterbi", "round"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--bias", type=float, default=0.0, help="logit shift before decoding")
    ap.add_argument("--start", default=None, choices=[None, "lpround"], help="seed the recursion with a schedule")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--guide", default=None, choices=[None, "none", "m1x", "net", "net_once"],
                    help="LP-verified local search from the LP-rounding schedule, guided by this prior")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", default="lpround", choices=["lpround", "union"], help="starting schedule of the search")
    ap.add_argument("--n0", type=int, default=0)
    ap.add_argument("--n1", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.chdir(C.ROOT)
    os.makedirs(C.RES, exist_ok=True)
    if a.guide:
        run_guided(a)
    elif a.model:
        run_model(a)
    if a.baselines:
        run_baselines(a)
    if a.report:
        report(a)
