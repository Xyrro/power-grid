"""RACLearn-style confidence-aware fixing for DC-OTS (Park et al., IEEE TPS 2024, arXiv 2211.15755).

    python scripts/run_confidence.py --cfg case118 --n_test 100

RACLearn: MC-dropout on the commitment decoder (p = 0.5, penultimate layer), Conf = 1/sigma over T
samples, fix the X % most confident binaries to their predictions, repair feasibility, solve the
reduced MILP. Here the binaries are line statuses. Compared selections:
  * MC-dropout confidence 1/sigma            (RACLearn)
  * probability margin |p - 0.5|              (the entropy-style baseline RACLearn compares against)
  * "keep the 10 most likely lines free"      (our partial fixing, F9)
Feasibility repair for OTS: if fixing makes the MILP infeasible, drop the fixed-OPEN decisions and keep
the fixed-CLOSED ones (always feasible, because all-closed is). We report feasibility before repair.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import _W, _init, load, make_model  # noqa: E402
from otsl.features import Featurizer  # noqa: E402
from otsl.pipeline import metrics  # noqa: E402
from otsl.train import build_model1, train_bce  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402
from run_model1 import fmt_table  # noqa: E402


def add_head_dropout(m1, p=0.5):
    """Insert dropout before the last linear layer of the switching head (as RACLearn does)."""
    head = m1.net.head
    layers = list(head.children())
    m1.net.head = nn.Sequential(*layers[:-1], nn.Dropout(p), layers[-1])
    return m1


@torch.no_grad()
def mc_predict(m1, d, T=30, bs=128):
    m1.net.eval()
    for mod in m1.net.head.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
    n = len(d["pd"])
    mus, sds = [], []
    for i in range(0, n, bs):
        idx = np.arange(i, min(i + bs, n))
        s = torch.stack([torch.sigmoid(m1.logits(d, idx)) for _ in range(T)])  # [T, B, L]
        mus.append(s.mean(0).numpy()); sds.append(s.std(0).numpy())
    m1.net.eval()
    return np.concatenate(mus), np.concatenate(sds)


def _solve(args):
    pd, fix, K = args
    m = _W["m"]
    cfg = _W["cfg"]
    t0 = time.time()
    s = m.solve_ots(pd, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], z_fix=fix)
    feas_before = s.z is not None
    if not feas_before:   # repair: keep only the closed fixes
        s = m.solve_ots(pd, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"],
                        z_fix={l: v for l, v in fix.items() if v == 1})
    return (s.obj if s.z is not None else np.inf, s.z, feas_before, time.time() - t0)


def fixes_by_confidence(mu, conf, sw, ratio, K):
    """Fix the `ratio` most confident switchable decisions to round(mu); at most K fixed open."""
    out = []
    swi = np.where(sw)[0]
    for i in range(len(mu)):
        order = swi[np.argsort(-conf[i, swi])]
        chosen = order[:int(round(ratio * len(swi)))]
        fix = {int(l): (0 if mu[i, l] > 0.5 else 1) for l in chosen}
        opens = [l for l, v in fix.items() if v == 0]
        if len(opens) > K:
            for l in sorted(opens, key=lambda l: -mu[i, l])[K:]:
                del fix[l]
        out.append(fix)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = CONFIGS[a.cfg]
    root = os.path.join("data", "generated", a.cfg)
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    te = {k: (v[:a.n_test] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in te.items()}
    m = make_model(cfg)
    case, K = m.case, cfg["budget"]
    sw = ~m.fixed_closed
    feat = Featurizer(case, tr, duals=True, fixed_closed=m.fixed_closed)
    m1 = add_head_dropout(build_model1(case, feat, sw, "gnn", seed=a.seed), 0.5)
    t = time.time()
    train_bce(m1, tr, va, epochs=a.epochs, seed=a.seed, log_every=25)
    print(f"trained in {time.time() - t:.0f}s", flush=True)
    mu, sd = mc_predict(m1, te, T=a.T)
    y = 1 - te["z"]                                     # 1 = MILP opened the line
    pred = (mu > 0.5).astype(int)
    conf_mc = 1.0 / np.maximum(sd, 1e-6)
    conf_margin = np.abs(mu - 0.5)

    # accuracy of the most confident subset (RACLearn Fig. 6), over switchable lines
    acc_curve = {}
    for name, conf in [("mc_dropout", conf_mc), ("margin", conf_margin)]:
        c, ok = conf[:, sw].ravel(), (pred[:, sw] == y[:, sw]).ravel()
        isopen = y[:, sw].ravel() == 1
        order = np.argsort(-c)
        acc_curve[name] = {}
        for r in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]:
            k = max(1, int(r * len(order)))
            sel = order[:k]
            acc_curve[name][r] = {"accuracy": float(ok[sel].mean()),
                                  "share_of_opened_lines_in_subset": float(isopen[sel].sum() / max(isopen.sum(), 1))}
        print(name, {r: round(v["accuracy"], 5) for r, v in acc_curve[name].items()}, flush=True)

    jobs, labels = [], []
    for name, conf in [("MC-dropout (RACLearn)", conf_mc), ("probability margin", conf_margin)]:
        for r in [0.5, 0.9, 0.97, 1.0]:
            fx = fixes_by_confidence(mu, conf, sw, r, K)
            labels.append((name, r)); jobs.append(fx)
    # our partial fixing: 10 most likely lines free, the rest fixed closed
    fx = []
    for i in range(len(mu)):
        order = np.argsort(-np.where(sw, mu[i], -1))
        free = set(order[:10].tolist())
        fx.append({int(l): 1 for l in np.where(sw)[0] if l not in free})
    labels.append(("keep 10 most likely lines free (ours)", None)); jobs.append(fx)

    flat = [(te["pd"][i], f[i], K) for f in jobs for i in range(len(te["pd"]))]
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_solve, flat, chunksize=1)
    n = len(te["pd"])
    rows = []
    for j, (name, r) in enumerate(labels):
        rr = res[j * n:(j + 1) * n]
        cost = np.array([x[0] for x in rr])
        z = np.array([x[1] if x[1] is not None else np.ones(case.n_line, np.int8) for x in rr])
        fixed_n = np.mean([len(f) for f in jobs[j]]) / sw.sum() * 100
        row = metrics(cost, te, z, f"{name}" + ("" if r is None else f", fix {int(r * 100)} %"),
                      fixed_pct=float(fixed_n),
                      feasible_before_repair_pct=float(np.mean([x[2] for x in rr]) * 100),
                      time_s=float(np.mean([x[3] for x in rr])),
                      milp_time_s_reference=float(te["ots_time"].mean()))
        rows.append(row)
        print(f"  {row['method']:45s} fixed {fixed_n:5.1f}%  feas-before-repair {row['feasible_before_repair_pct']:5.1f}%"
              f"  gap {row['gap_mean_%']:.4f}%  closed {row['gap_closed_%']:.1f}%  time {row['time_s']:.2f}s", flush=True)
    cols = ["method", "fixed_pct", "feasible_before_repair_pct", "gap_mean_%", "gap_closed_%", "beats_or_ties_milp_%",
            "time_s", "milp_time_s_reference"]
    md = fmt_table(rows, cols)
    print(md)
    out = os.path.join("results", a.cfg)
    with open(os.path.join(out, "confidence_results.md"), "w") as f:
        f.write(md + "\n\nAccuracy of the most-confident subset (switchable lines):\n\n" + json.dumps(acc_curve, indent=1))
    with open(os.path.join(out, "confidence_results.json"), "w") as f:
        json.dump({"rows": rows, "acc_curve": acc_curve}, f, indent=1, default=float)
