"""Does Model 1 need MILP labels at all?

    python scripts/run_label_free.py --cfg case118 --steps 1500

Trains the dual-feature GNN with REINFORCE + exact LP critic from a random initialisation (no
MILP solved), and compares with the BCE-imitation model saved by run_model1.py. Also reports the
compute spent on supervision: MILP seconds for the imitation labels vs LP solves for RL.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import load, make_model  # noqa: E402
from otsl.features import Featurizer  # noqa: E402
from otsl.pipeline import LPOracle, candidates_from_probs, decode_threshold, metrics, pick_best  # noqa: E402
from otsl.train import build_model1, train_reinforce  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402
from run_model1 import fmt_table  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = CONFIGS[a.cfg]
    root = os.path.join("data", "generated", a.cfg)
    tr, te = load(os.path.join(root, "train.npz")), load(os.path.join(root, "test.npz"))
    m = make_model(cfg)
    case, K, sc = m.case, cfg["budget"], float(tr["switch_cost"])
    sw = ~m.fixed_closed
    out_dir = os.path.join("results", a.cfg)
    oracle = LPOracle(cfg, a.workers)
    keys = np.arange(len(te["pd"])) + 3 * 10 ** 7
    feat = Featurizer(case, tr, duals=True, fixed_closed=m.fixed_closed)
    rows, extra = [], {"milp_label_cpu_s_total": float(tr["ots_time"].sum()), "n_train": int(len(tr["pd"]))}

    def evaluate(m1, label):
        p = m1.predict(te)
        z1 = decode_threshold(p, K, sw)
        rows.append(metrics(oracle.costs(te["pd"], z1, keys), te, z1, f"{label}: top-1 decode -> LP",
                            LPs_per_scenario=1.0))
        cl = candidates_from_probs(p, K, sw, n_samples=16)
        zb, cb, nl = pick_best(oracle, te["pd"], cl, keys, sc)
        rows.append(metrics(cb, te, zb, f"{label}: candidate screening -> LP", LPs_per_scenario=float(nl.mean())))
        print(rows[-2]["method"], round(rows[-2]["benefit_captured_%"], 2), "|", rows[-1]["method"],
              round(rows[-1]["benefit_captured_%"], 2), flush=True)

    sd = os.path.join(out_dir, "model1_GNN-BCE_+duals.pt")
    if os.path.exists(sd):
        m_bce = build_model1(case, feat, sw, "gnn", seed=a.seed)
        m_bce.net.load_state_dict(torch.load(sd))
        evaluate(m_bce, "BCE imitation of MILP labels (reference)")

    m_rl = build_model1(case, feat, sw, "gnn", seed=a.seed)
    # start "closed": every line has a small opening probability, so early samples are mostly feasible
    p0 = K / (2.0 * sw.sum())
    with torch.no_grad():
        m_rl.net.head[-1].bias.fill_(float(np.log(p0 / (1 - p0))))
    t = time.time()
    n0 = oracle.n_solves
    _, hist = train_reinforce(m_rl, tr, oracle, K, sc, steps=a.steps, lr=1e-3, seed=a.seed, log_every=100)
    extra["rl_scratch_time_s"] = time.time() - t
    extra["rl_scratch_lp_solves"] = int(oracle.n_solves - n0)
    extra["rl_scratch_reward_curve_%"] = [float(np.mean(hist[i:i + 50]) * 100) for i in range(0, len(hist), 50)]
    evaluate(m_rl, f"REINFORCE from scratch, no MILP labels ({a.steps} steps)")
    torch.save(m_rl.net.state_dict(), os.path.join(out_dir, "model1_GNN_duals_RLscratch.pt"))

    cols = ["method", "gap_mean_%", "gap_p95_%", "gap_closed_%", "benefit_captured_%", "beats_or_ties_milp_%", "feasible_%", "n_open",
            "LPs_per_scenario"]
    md = fmt_table(rows, cols)
    print(md)
    print(extra)
    with open(os.path.join(out_dir, "label_free_results.md"), "w") as f:
        f.write(md + "\n" + "\n".join(f"- {k}: {v}" for k, v in extra.items() if not isinstance(v, list)) + "\n")
    with open(os.path.join(out_dir, "label_free_results.json"), "w") as f:
        json.dump({"rows": rows, "extra": extra}, f, indent=1, default=float)
    oracle.close()
