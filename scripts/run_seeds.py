"""Seed robustness of the key Model 1 claims (top-1 decoding, one LP per scenario).

    python scripts/run_seeds.py --cfg case118 --seeds 1 2

For each seed: GNN + duals trained by BCE on (a) MILP labels and (b) equivalence-aware soft labels,
then (c) REINFORCE fine-tuning of (a) with the exact LP as critic.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import load, make_model  # noqa: E402
from otsl.features import Featurizer  # noqa: E402
from otsl.pipeline import LPOracle, decode_threshold, equivalent_label_targets, line_neighbourhood, metrics  # noqa: E402
from otsl.train import build_model1, train_bce, train_reinforce  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = CONFIGS[a.cfg]
    root = os.path.join("data", "generated", a.cfg)
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    m = make_model(cfg)
    case, K, sc = m.case, cfg["budget"], float(tr["switch_cost"])
    sw = ~m.fixed_closed
    oracle = LPOracle(cfg, a.workers)
    keys = np.arange(len(te["pd"])) + 4 * 10 ** 7
    feat = Featurizer(case, tr, duals=True, fixed_closed=m.fixed_closed)
    nb = line_neighbourhood(case, 2) if case.n_line > 100 else None
    soft, _ = equivalent_label_targets(oracle, tr, sw, sc, neighbourhood=nb)
    c0 = metrics(oracle.costs(te["pd"], np.ones_like(te["z"]), keys), te, np.ones_like(te["z"]), "all-closed")
    rows = []
    for seed in a.seeds:
        out = {"seed": seed}
        m_bce = build_model1(case, feat, sw, "gnn", seed=seed)
        train_bce(m_bce, tr, va, epochs=a.epochs, seed=seed, log_every=0)
        m_eq = build_model1(case, feat, sw, "gnn", seed=seed)
        train_bce(m_eq, tr, va, epochs=a.epochs, seed=seed, log_every=0, soft_targets=soft)
        m_rl = copy.deepcopy(m_bce)
        train_reinforce(m_rl, tr, oracle, K, sc, steps=a.rl_steps, seed=seed, log_every=0)
        for name, mm in [("bce", m_bce), ("equiv", m_eq), ("bce+reinforce", m_rl)]:
            z = decode_threshold(mm.predict(te), K, sw)
            r = metrics(oracle.costs(te["pd"], z, keys), te, z, name)
            out[name] = {"gap_mean_%": r["gap_mean_%"], "gap_closed_%": r["gap_closed_%"], "n_open": r["n_open"],
                         "feasible_%": r["feasible_%"]}
            print(f"seed {seed} {name:14s} gap closed {r['gap_closed_%']:.1f}%  gap {r['gap_mean_%']:.4f}%  "
                  f"opened {r['n_open']:.2f}", flush=True)
        rows.append(out)
    summary = {k: {"mean": float(np.mean([r[k]["gap_closed_%"] for r in rows])),
                   "min": float(np.min([r[k]["gap_closed_%"] for r in rows])),
                   "max": float(np.max([r[k]["gap_closed_%"] for r in rows]))} for k in ["bce", "equiv", "bce+reinforce"]}
    print(json.dumps(summary, indent=1))
    with open(os.path.join("results", a.cfg, "seeds_results.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary, "allclosed_gap_%": c0["gap_mean_%"]}, f, indent=1, default=float)
    oracle.close()
