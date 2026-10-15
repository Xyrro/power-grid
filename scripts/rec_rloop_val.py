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
    t0 = time.time()
    out = {k: [] for k in ("cost", "shed", "short", "halt")}
    for s in range(0, len(idx), 60):
        st = rloop.rollout_batch(net, feats, sysm, insts[s:s + 60], a.K, pool, "cpu",
                                 [("va", i) for i in idx[s:s + 60]], mode=mode)
        for j, k in enumerate(("cost", "shed", "short", "halt")):
            out[k].append(np.array([[x[j] for x in step] for step in st]))
    res = {k: np.concatenate(v, 1) for k, v in out.items()}
    tag = a.model + (f"_{a.decode}" if a.decode else "")
    np.savez(os.path.join(C.RES, f"rloop_val_{tag}.npz"), db=va["db"][idx], c_rel=va["c_rel"][idx], **res)
    best = np.minimum.accumulate(res["cost"], 0)
    print(f"{tag}: {time.time() - t0:.0f}s, mean gap best-of-k:",
          [round(float(gap(best[k], va["db"][idx]).mean()), 3) for k in range(a.K)], flush=True)


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
            K = R["cost"].shape[0]
            for k in sorted({1, 2, 3, 4, 6, 8, 12, 16, K} & set(range(1, K + 1))):
                c = R["cost"][:k]
                j = c.argmin(0)
                n = c.shape[1]
                rows[f"{f[10:-4]} K={k}"] = summary(c.min(0), R["shed"][:k][j, range(n)], R["short"][:k][j, range(n)], R["db"])
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
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.chdir(C.ROOT)
    os.makedirs(C.RES, exist_ok=True)
    if a.model:
        run_model(a)
    if a.baselines:
        run_baselines(a)
    if a.report:
        report(a)
