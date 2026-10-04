"""Model 1 study: how should the switching decision be learned and decoded?

    python scripts/run_model1.py --cfg case118 [--raw case118_raw]

Compares (all decoded topologies are scored by the exact fixed-topology LP):
  baselines   all-closed DC-OPF, dual-sensitivity greedy (no learning), kNN-LP (Johnson et al. 2020)
  framework   MLP / GNN trained by BCE on MILP labels, top-1 decode
  new         + dual features from one all-closed DC-OPF
              + candidate screening (several decodes scored by LP, all-closed fallback)
              + REINFORCE fine-tuning with the LP as the critic (cost-aware Model-1 loss)
              + confidence-based partial fixing + MILP on the uncertain lines
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
from otsl.pipeline import (LPOracle, candidates_from_probs, decode_threshold, equivalent_label_targets,  # noqa: E402
                           knn_candidates, line_neighbourhood, metrics, pick_best, run_dual_greedy)
from otsl.train import build_model1, train_bce, train_reinforce  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402


def fmt_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(f"{r.get(c):.3f}" if isinstance(r.get(c), (float, np.floating)) else str(r.get(c, ""))
                                  for c in cols) + " |\n"
    return head + body


def mse_vs_milp(oracle_model, test, z, pd):
    """Testing-box metric of the framework: MSE of (pg, va) against the MILP solution."""
    pg_err, va_err = [], []
    for i in range(len(z)):
        s = oracle_model.solve_lp(pd[i], z[i])
        if not s.ok:
            s = oracle_model.solve_lp(pd[i])
        pg_err.append(((s.pg - test["pg"][i]) ** 2).mean())
        va_err.append(((s.va - test["va"][i]) ** 2).mean())
    return np.array(pg_err), np.array(va_err)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--raw", default=None, help="config with raw (no switching cost) training labels")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = CONFIGS[a.cfg]
    root = os.path.join("data", "generated", a.cfg)
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    m = make_model(cfg)
    case, K, sc = m.case, cfg["budget"], float(tr["switch_cost"])
    sw = ~m.fixed_closed
    print(f"{a.cfg}: train {len(tr['pd'])} val {len(va['pd'])} test {len(te['pd'])}  switch_cost {sc:.2f}")
    oracle = LPOracle(cfg, a.workers)
    rows, extra = [], {}
    keys = np.arange(len(te["pd"])) + 10 ** 7
    out_dir = os.path.join("results", a.cfg)
    os.makedirs(out_dir, exist_ok=True)

    def add(label, z, cost, n_lp, t, **kw):
        r = metrics(cost, te, z, label, LPs_per_scenario=float(np.mean(n_lp)), time_ms=1000 * t / len(z), **kw)
        rows.append(r)
        print(f"  {label:45s} gap {r['gap_mean_%']:.4f}%  captured {r['benefit_captured_%']:.1f}%  "
              f"feas {r['feasible_%']:.1f}%  LPs {r['LPs_per_scenario']:.1f}", flush=True)
        return r

    # ---------------------------------------------------------------- reference numbers
    b = (te["c0"] - te["c_ots"] - sc * (1 - te["z"]).sum(1)) / te["c0"] * 100
    extra["test_benefit_mean_%"] = float(b.mean())
    extra["test_frac_benefit_gt_0.01%"] = float((b > 0.01).mean())
    extra["milp_time_mean_s"] = float(te["ots_time"].mean())
    extra["milp_time_median_s"] = float(np.median(te["ots_time"]))
    print(f"mean benefit {b.mean():.3f}%  MILP time mean {te['ots_time'].mean():.2f}s")
    if "alt_c" in te:
        cs = te["c_ots"] + sc * (1 - te["z"]).sum(1)
        rel = (te["alt_c"] - cs[:, None]) / cs[:, None]
        for tol in [1e-5, 1e-4, 1e-3]:
            extra[f"frac_with_alt_optimum_within_{tol:g}"] = float((rel <= tol).any(1).mean())
        extra["mean_n_alt_within_1e-4"] = float((rel <= 1e-4).sum(1).mean())
        print("label ambiguity:", {k: round(v, 3) for k, v in extra.items() if "alt" in k})

    # ---------------------------------------------------------------- baselines
    t = time.time()
    c = oracle.costs(te["pd"], np.ones_like(te["z"]), keys)
    add("all-closed DC-OPF", np.ones_like(te["z"]), c, np.ones(len(c)), time.time() - t)

    t = time.time()
    zg, cg, nl = run_dual_greedy(cfg, te["pd"], K, R=5, workers=a.workers)
    add("dual-greedy heuristic (R=5, no learning)", zg, cg, nl, time.time() - t)

    for k in [5, 20]:
        t = time.time()
        cl = knn_candidates(tr["pd"], tr["z"], te["pd"], k)
        zk, ck, nl = pick_best(oracle, te["pd"], cl, keys, sc)
        add(f"kNN-LP k={k} (Johnson et al.)", zk, ck, nl, time.time() - t)

    # ---------------------------------------------------------------- learned Model 1
    feats = {"base": Featurizer(case, tr, duals=False, fixed_closed=m.fixed_closed),
             "duals": Featurizer(case, tr, duals=True, fixed_closed=m.fixed_closed)}
    variants = [("MLP-BCE", "mlp", "base", None), ("GNN-BCE", "gnn", "base", None),
                ("GNN-BCE +duals", "gnn", "duals", None), ("GNN-BCE +duals +equiv-labels", "gnn", "duals", "equiv")]
    t = time.time()
    nb = line_neighbourhood(case, 2) if case.n_line > 100 else None   # all swaps on small grids
    soft, n_eq = equivalent_label_targets(oracle, tr, sw, sc, neighbourhood=nb)
    extra["equiv_label_time_s"] = time.time() - t
    extra["train_frac_with_equivalent_topology"] = float((n_eq > 1).mean())
    extra["train_mean_equivalent_topologies"] = float(n_eq.mean())
    print(f"equivalence-aware labels: {(n_eq > 1).mean() * 100:.1f}% of training scenarios have an equally "
          f"optimal single-swap topology (mean {n_eq.mean():.2f} topologies) [{time.time() - t:.0f}s]")
    if a.raw:
        trr = load(os.path.join("data", "generated", a.raw, "train.npz"))
        variants.append(("GNN-BCE +duals [raw MILP labels]", "gnn", "duals", trr))
    models = {}
    for name, kind, fk, alt_train in variants:
        print(f"training {name}")
        m1 = build_model1(case, feats[fk], sw, kind, seed=a.seed)
        t = time.time()
        # raw-label set: same load scenarios (same seed), labels from DC-OTS without switching cost
        soft_t = soft if isinstance(alt_train, str) and alt_train == "equiv" else None
        train_d = alt_train if isinstance(alt_train, dict) else tr
        train_bce(m1, train_d, va, epochs=a.epochs // (4 if a.quick else 1), seed=a.seed, soft_targets=soft_t)
        extra[f"train_time_s[{name}]"] = time.time() - t
        models[name] = m1
        t = time.time()
        p = m1.predict(te)
        t_pred = time.time() - t
        z1 = decode_threshold(p, K, sw)
        c1 = oracle.costs(te["pd"], z1, keys)
        add(f"{name} | top-1 decode -> LP", z1, c1, np.ones(len(c1)), t_pred + 0)
        t = time.time()
        cl = candidates_from_probs(p, K, sw, n_samples=16)
        zb, cb, nl = pick_best(oracle, te["pd"], cl, keys, sc)
        add(f"{name} | candidate screening -> LP", zb, cb, nl, t_pred + time.time() - t)
        torch.save(m1.net.state_dict(), os.path.join(out_dir, f"model1_{name.replace(' ', '_')}.pt"))

    # ---------------------------------------------------------------- cost-aware fine-tuning
    print("REINFORCE fine-tuning of GNN-BCE +duals with the LP as critic")
    import copy
    m1 = models["GNN-BCE +duals"]
    m1rl = copy.deepcopy(m1)
    t = time.time()
    _, hist = train_reinforce(m1rl, tr, oracle, K, sc, steps=a.rl_steps // (4 if a.quick else 1), seed=a.seed)
    extra["rl_time_s"] = time.time() - t
    extra["rl_reward_curve"] = [float(np.mean(hist[i:i + 10])) for i in range(0, len(hist), 10)]
    p = m1rl.predict(te)
    z1 = decode_threshold(p, K, sw)
    add("GNN +duals +REINFORCE | top-1 decode -> LP", z1, oracle.costs(te["pd"], z1, keys), np.ones(len(z1)), 0)
    cl = candidates_from_probs(p, K, sw, n_samples=16)
    zb, cb, nl = pick_best(oracle, te["pd"], cl, keys, sc)
    add("GNN +duals +REINFORCE | candidate screening -> LP", zb, cb, nl, 0)
    models["GNN +duals +REINFORCE"] = m1rl
    torch.save(m1rl.net.state_dict(), os.path.join(out_dir, "model1_GNN_duals_REINFORCE.pt"))

    # ---------------------------------------------------------------- partial fixing + MILP
    p = models["GNN-BCE +duals"].predict(te)
    n_pf = min(len(te["pd"]), 100 if not a.quick else 20)
    for keep in [10, 25]:
        gaps, times, costs, zs = [], [], [], []
        for i in range(n_pf):
            order = np.argsort(-np.where(sw, p[i], -1))
            free = set(order[:keep].tolist())
            fix = {l: 1 for l in range(case.n_line) if sw[l] and l not in free}
            s = m.solve_ots(te["pd"][i], time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], z_fix=fix)
            costs.append(s.obj if s.z is not None else np.inf)
            zs.append(s.z if s.z is not None else np.ones(case.n_line, np.int8))
            times.append(s.time)
        sub = {k: v[:n_pf] for k, v in te.items() if isinstance(v, np.ndarray) and v.ndim >= 1}
        sub["switch_cost"] = te["switch_cost"]
        r = metrics(np.array(costs), sub, np.array(zs), f"GNN +duals | partial-fix MILP (free={keep})",
                    LPs_per_scenario=float("nan"), time_ms=1000 * float(np.mean(times)),
                    milp_time_ms_same_subset=1000 * float(te["ots_time"][:n_pf].mean()))
        rows.append(r)
        print(f"  partial-fix free={keep}: gap {r['gap_mean_%']:.4f}% time {np.mean(times):.2f}s "
              f"vs full MILP {te['ots_time'][:n_pf].mean():.2f}s")

    # ---------------------------------------------------------------- testing-box metric
    pg_mse, va_mse = mse_vs_milp(m, te, zb, te["pd"])
    tot_b = cb + sc * (1 - zb).sum(1)
    cs = te["c_ots"] + sc * (1 - te["z"]).sum(1)
    gap_b = (tot_b - cs) / cs * 100
    near_opt = gap_b <= 1e-3
    extra["mse_analysis"] = {
        "share_near_optimal(gap<=0.001%)": float(near_opt.mean()),
        "share_near_optimal_with_different_topology": float((near_opt & (zb != te["z"]).any(1)).mean()),
        "va_mse_near_optimal_mean": float(va_mse[near_opt].mean()) if near_opt.any() else None,
        "va_mse_near_optimal_max": float(va_mse[near_opt].max()) if near_opt.any() else None,
        "pg_mse_near_optimal_max": float(pg_mse[near_opt].max()) if near_opt.any() else None,
        "spearman_mse_vs_gap": float(np.corrcoef(np.argsort(np.argsort(va_mse + pg_mse)),
                                                 np.argsort(np.argsort(gap_b)))[0, 1]),
    }
    print("MSE analysis:", extra["mse_analysis"])

    cols = ["method", "gap_mean_%", "gap_p95_%", "benefit_captured_%", "beats_or_ties_milp_%", "feasible_%",
            "z_exact_match_%", "n_open", "LPs_per_scenario", "time_ms"]
    md = fmt_table(rows, cols)
    print(md)
    with open(os.path.join(out_dir, "model1_results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "model1_results.json"), "w") as f:
        json.dump({"rows": rows, "extra": extra}, f, indent=1, default=float)
    oracle.close()
