"""Generalisation to unseen base-case topologies (one random line outage per test scenario).

    python scripts/run_topology.py

Train on the intact grid (case30) or on a mix with random outages (case30_topo); test on the intact
test set and on a test set where every scenario has a random line out of service (case30_topo_test).
kNN reuses stored topologies (which may not fit the outaged grid); the GNN sees the outage through
its in-service edge feature and edge gating.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import load, make_model  # noqa: E402
from otsl.features import Featurizer  # noqa: E402
from otsl.pipeline import (LPOracle, candidates_from_probs, decode_threshold, knn_candidates, metrics,  # noqa: E402
                           pick_best, run_dual_greedy)
from otsl.train import build_model1, train_bce  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402
from run_model1 import fmt_table  # noqa: E402

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    G = lambda c, s: load(os.path.join("data", "generated", c, f"{s}.npz"))
    trains = {"intact": (G("case30", "train"), G("case30", "val")),
              "with outages": (G("case30_topo", "train"), G("case30_topo", "val"))}
    tests = {"intact grid": G("case30", "test"), "unseen outage": G("case30_topo_test", "test")}
    cfg = CONFIGS["case30_topo"]
    m = make_model(cfg)
    case, K = m.case, cfg["budget"]
    sw = ~m.fixed_closed
    oracle = LPOracle(cfg, 2)
    rows = []

    def base_of(d):
        return d["base"] if "base" in d else np.ones_like(d["z"])

    def add(label, test_name, te, z, cost, nl):
        r = metrics(cost, te, z, f"{label} [{test_name}]", LPs_per_scenario=float(np.mean(nl)))
        rows.append(r)
        print(f"  {r['method']:70s} captured {r['benefit_captured_%']:.1f}%  feas {r['feasible_%']:.1f}%", flush=True)

    for tn, te in tests.items():
        keys = np.arange(len(te["pd"])) + (7 if tn == "intact grid" else 8) * 10 ** 7
        bt = base_of(te)
        zg, cg, nl = run_dual_greedy(cfg, te["pd"], K, 5, 2, bases=bt)
        add("dual-greedy (no learning)", tn, te, zg, cg, nl)
        for trn, (tr, _) in trains.items():
            zk, ck, nl = pick_best(oracle, te["pd"], knn_candidates(tr["pd"], tr["z"], te["pd"], 20), keys,
                                   float(te["switch_cost"]), base=bt)
            add(f"kNN-LP k=20, trained {trn}", tn, te, zk, ck, nl)

    for trn, (tr, va) in trains.items():
        for kind in ["mlp", "gnn"]:
            feat = Featurizer(case, tr, duals=True, fixed_closed=m.fixed_closed, topo=True)
            m1 = build_model1(case, feat, sw, kind)
            print(f"training {kind} on {trn}")
            train_bce(m1, tr, va, epochs=epochs, log_every=50)
            for tn, te in tests.items():
                keys = np.arange(len(te["pd"])) + (7 if tn == "intact grid" else 8) * 10 ** 7
                bt = base_of(te)
                p = m1.predict(te)
                z1 = decode_threshold(p, K, sw) * bt
                add(f"{kind.upper()}+duals, trained {trn}: top-1", tn, te, z1, oracle.costs(te["pd"], z1, keys),
                    np.ones(len(z1)))
                cl = candidates_from_probs(p, K, sw, n_samples=16)
                zb, cb, nl = pick_best(oracle, te["pd"], cl, keys, float(te["switch_cost"]), base=bt)
                add(f"{kind.upper()}+duals, trained {trn}: screening", tn, te, zb, cb, nl)

    cols = ["method", "gap_mean_%", "benefit_captured_%", "beats_or_ties_milp_%", "feasible_%", "LPs_per_scenario"]
    md = fmt_table(rows, cols)
    print(md)
    os.makedirs("results/case30", exist_ok=True)
    with open("results/case30/topology_results.md", "w") as f:
        f.write(md)
    with open("results/case30/topology_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    oracle.close()
