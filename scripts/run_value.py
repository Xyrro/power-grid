"""Learned switching values (no MILP labels needed).

    python scripts/run_value.py --cfg case118 --n_train 600

1. exhaustive greedy (exact LP for every line, every step) on train scenarios -> states + exact values
2. train ValueGNN on those states
3. test: exhaustive greedy (upper bound of greedy) vs learned greedy (R LP verifications per step)
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
from otsl.pipeline import metrics  # noqa: E402
from otsl.value import (GreedyOracle, ValueGNN, exhaustive_greedy, learned_beam, learned_greedy,  # noqa: E402
                        stack_states, train_value)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402
from run_model1 import fmt_table  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = CONFIGS[a.cfg]
    root = os.path.join("data", "generated", a.cfg)
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    m = make_model(cfg)
    case, K, sc = m.case, cfg["budget"], float(tr["switch_cost"])
    sw = ~m.fixed_closed
    oracle = GreedyOracle(cfg, a.workers)
    out_dir = os.path.join("results", a.cfg)
    os.makedirs(out_dir, exist_ok=True)
    rows, extra = [], {}

    # ---------------------------------------------------------------- labels by exhaustive greedy
    t = time.time()
    _, _, nlp_tr, st_tr = exhaustive_greedy(oracle, tr["pd"][:a.n_train], sw, K, sc, record=True)
    _, _, _, st_va = exhaustive_greedy(oracle, va["pd"][:150], sw, K, sc, record=True)
    extra["label_time_s_per_scenario"] = (time.time() - t) / (a.n_train + min(150, len(va["pd"])))
    extra["label_LPs_per_scenario"] = float(nlp_tr.mean())
    extra["n_states_train"] = len(st_tr)
    print(f"value labels: {len(st_tr)} states, {nlp_tr.mean():.0f} LPs/scenario, "
          f"{extra['label_time_s_per_scenario']:.2f}s/scenario wall ({a.workers} workers)")

    feat = Featurizer(case, stack_states(st_tr), duals=True, fixed_closed=m.fixed_closed)
    torch.manual_seed(a.seed)
    model = ValueGNN(case, feat.node_dim, feat.edge_dim)
    t = time.time()
    train_value(model, feat, st_tr, st_va, sw, epochs=a.epochs, seed=a.seed)
    extra["train_time_s"] = time.time() - t
    torch.save(model.state_dict(), os.path.join(out_dir, "value_gnn.pt"))

    # value-prediction quality on validation states
    with torch.no_grad():
        dv = stack_states(st_va)
        x, e = feat(dv)
        lg, val = model(x, e, torch.as_tensor(dv["z"]))
        v = np.array([s["v"] for s in st_va])
        c = np.array([s["c"] for s in st_va])
        mask = sw[None] & (dv["z"] > 0)
        true = np.where(np.isfinite(v), v / c[:, None] * 100, np.inf)
        pred = val.numpy() * model.scale + 1e3 * (torch.sigmoid(lg).numpy() > 0.5)
        hit1, hit3, rho = [], [], []
        for i in range(len(v)):
            mk = np.where(mask[i])[0]
            bt = mk[np.argmin(true[i, mk])]
            order = mk[np.argsort(pred[i, mk])]
            hit1.append(order[0] == bt); hit3.append(bt in order[:3])
            fin = np.isfinite(true[i, mk])
            rho.append(np.corrcoef(np.argsort(np.argsort(true[i, mk][fin])),
                                   np.argsort(np.argsort(pred[i, mk][fin])))[0, 1])
        extra["val_best_line_top1_%"] = float(np.mean(hit1) * 100)
        extra["val_best_line_in_top3_%"] = float(np.mean(hit3) * 100)
        extra["val_spearman"] = float(np.nanmean(rho))
        # same statistics for the analytic first-order estimate -gamma*f
        est = np.where(mask, -dv["gamma0"] * dv["flow0"], np.inf)
        h1 = [np.argmin(est[i]) == np.argmin(np.where(mask[i], true[i], np.inf)) for i in range(len(v))]
        h3 = [np.argmin(np.where(mask[i], true[i], np.inf)) in np.argsort(est[i])[:3] for i in range(len(v))]
        extra["dual_estimate_best_line_top1_%"] = float(np.mean(h1) * 100)
        extra["dual_estimate_best_line_in_top3_%"] = float(np.mean(h3) * 100)
    print({k: round(v, 3) for k, v in extra.items() if isinstance(v, float)})

    # ---------------------------------------------------------------- test
    t = time.time()
    zg, cg, nl, _ = exhaustive_greedy(oracle, te["pd"], sw, K, sc)
    rows.append(metrics(cg, te, zg, "exhaustive greedy (exact LP for all lines)", LPs_per_scenario=float(nl.mean()),
                        time_ms=1000 * (time.time() - t) / len(cg)))
    for R in [1, 2, 4]:
        t = time.time()
        zl, cl, nl = learned_greedy(model, feat, oracle, te["pd"], sw, K, sc, R=R)
        rows.append(metrics(cl, te, zl, f"learned-value greedy (R={R} LP checks/step)",
                            LPs_per_scenario=float(nl.mean()), time_ms=1000 * (time.time() - t) / len(cl)))
        print(rows[-1])
    for Bw, R in [(3, 3), (5, 5)]:
        t = time.time()
        zl, cl, nl = learned_beam(model, feat, oracle, te["pd"], sw, K, sc, B=Bw, R=R)
        rows.append(metrics(cl, te, zl, f"learned-value beam search (B={Bw}, R={R})",
                            LPs_per_scenario=float(nl.mean()), time_ms=1000 * (time.time() - t) / len(cl)))
        print(rows[-1])
    cols = ["method", "gap_mean_%", "gap_p95_%", "benefit_captured_%", "beats_or_ties_milp_%", "feasible_%",
            "n_open", "LPs_per_scenario", "time_ms"]
    md = fmt_table(rows, cols)
    print(md)
    with open(os.path.join(out_dir, "value_results.md"), "w") as f:
        f.write(md + "\n" + "\n".join(f"- {k}: {v:.3f}" for k, v in extra.items() if isinstance(v, float)) + "\n")
    with open(os.path.join(out_dir, "value_results.json"), "w") as f:
        json.dump({"rows": rows, "extra": extra}, f, indent=1, default=float)
    oracle.close()
