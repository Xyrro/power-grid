"""Unit commitment, Model 1 without MILP labels.

    python scripts/uc_label_free.py --cfg uc1

The framework trains Model 1 on MILP solutions; on the 12-hour benchmark each label costs ~17 s.
Here no MILP is solved for training:

* LF-BCE        imitate the rounded LP relaxation after adequacy (+ min up/down) repair - one LP each;
* + REINFORCE   then fine-tune with the dispatch LP as critic (RLOO: the per-instance reward offset
                cancels, so the critic needs no reference cost; the LP relaxation cost is used for logs);
* scratch       REINFORCE from a random initialisation.

Compared with the MILP-label rows of results/<cfg>/uc_model1_results.json on the same test set.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import UCModel, adequacy_repair, load_rts_gmlc, repair_min_updown  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from otsl.ucml import DispatchOracle, UCFeaturizer, build_uc_model1, train_uc_bce, train_uc_reinforce, uc_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model1 import fmt_table  # noqa: E402
from uc_gen import UC_CONFIGS  # noqa: E402
from uc_model1 import candidates_from_probs  # noqa: E402

MILP_KEYS = ("u", "p", "va", "r", "obj", "time", "gap", "opt", "shed", "alt_u", "alt_c")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="uc1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--scratch_steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=0, help="evaluate on the first n test instances (0 = all)")
    a = ap.parse_args()
    cfg = UC_CONFIGS[a.cfg]
    T = cfg["T"]
    root = os.path.join("data", "generated", a.cfg)
    tr_full, va_full, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    if a.n_test:
        te = {k: v[:a.n_test] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(te["load"]) else v
              for k, v in te.items()}
    sysm = load_rts_gmlc()
    out_dir = os.path.join("results", a.cfg)
    rep = (lambda u, u0: repair_min_updown(u, u0, sysm.min_up, sysm.min_dn)) if T > 1 else None
    fixrep = (lambda U, U0: np.array([rep(U[i], U0[i]) for i in range(len(U))])) if rep else (lambda U, U0: U)

    def adequate(U, d):
        V = np.array([adequacy_repair(U[i], d["load"][i], d["avail"][i], d["sr"][i], sysm) for i in range(len(U))])
        return fixrep(V, d["u0"])

    # training data with every MILP-derived field removed
    strip = lambda d: dict({k: v for k, v in d.items() if k not in MILP_KEYS}, obj=d["c_rel"])
    tr, va = strip(tr_full), strip(va_full)
    y_tr = adequate((tr["u_rel"] > 0.5).astype(np.int8), tr)
    va["u_target"] = adequate((va["u_rel"] > 0.5).astype(np.int8), va)
    # label cost: one LP relaxation per instance vs one MILP
    m = UCModel(sysm, T=T, network=True)
    t0 = time.time()
    for i in range(20):
        m.solve_dispatch(scenario_from(tr_full, i), None, relax=True)
    extra = {"lp_relaxation_s_per_label": (time.time() - t0) / 20, "milp_s_per_label": float(tr_full["time"].mean()),
             "labels_agree_with_milp_unit_hour_%": float((y_tr == tr_full["u"]).mean() * 100)}
    print(extra, flush=True)

    oracle = DispatchOracle(cfg, a.workers)
    n_te = len(te["load"])
    keys_te = np.arange(n_te) + 5 * 10 ** 7
    rows, rng = [], np.random.default_rng(a.seed)

    def evaluate(name, m1):
        p = m1.predict(te)
        for label, U in [("top-1 -> LP", fixrep((p > 0.5).astype(np.int8), te["u0"])),
                         ("top-1 + adequacy repair -> LP", adequate((p > 0.5).astype(np.int8), te))]:
            c, sh, so = oracle.evaluate(te, np.arange(n_te), U, keys_te)
            rows.append(uc_metrics(c, sh, so, te, U, f"{name}: {label}", LPs_per_instance=1.0))
            print("  ", rows[-1]["method"], round(rows[-1]["no_shed_no_shortfall_%"], 1), round(rows[-1]["gap_median_%"], 4), flush=True)
        cl = candidates_from_probs(p, 8, rng)
        flat_i = np.concatenate([np.full(len(x), i) for i, x in enumerate(cl)])
        flat_u = fixrep(np.concatenate(cl), te["u0"][flat_i])
        c, sh, so = oracle.evaluate(te, flat_i, flat_u, keys_te[flat_i])
        best = np.array([np.where(flat_i == i)[0][np.argmin(c[flat_i == i])] for i in range(n_te)])
        rows.append(uc_metrics(c[best], sh[best], so[best], te, flat_u[best], f"{name}: candidate screening -> LP",
                               LPs_per_instance=float(np.mean([len(x) for x in cl]))))
        print("  ", rows[-1]["method"], round(rows[-1]["no_shed_no_shortfall_%"], 1), round(rows[-1]["gap_median_%"], 4), flush=True)

    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    m_lf = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
    t0 = time.time()
    train_uc_bce(m_lf, tr, va, y_tr, epochs=a.epochs, seed=a.seed)
    extra["lf_bce_train_s"] = time.time() - t0
    evaluate("LF: imitate repaired LP relaxation (no MILP)", m_lf)
    m_rl = copy.deepcopy(m_lf)
    t0 = time.time()
    _, hist = train_uc_reinforce(m_rl, tr, oracle, steps=a.rl_steps, seed=a.seed, repair=rep)
    extra["lf_rl_s"] = time.time() - t0
    evaluate("LF + REINFORCE (LP critic, no MILP)", m_rl)
    m_sc = build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)
    t0 = time.time()
    _, hist_sc = train_uc_reinforce(m_sc, tr, oracle, steps=a.scratch_steps, seed=a.seed, repair=rep)
    extra["scratch_rl_s"] = time.time() - t0
    evaluate(f"REINFORCE from scratch ({a.scratch_steps} steps, no MILP)", m_sc)

    ref_path = os.path.join(out_dir, "uc_model1_results.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["rows"]
        rows += [dict(r, method="[MILP labels] " + r["method"]) for r in ref
                 if r["method"].startswith("GNN + symmetry + LP-relaxation features") and "confidence" not in r["method"]]
    cols = ["method", "no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%",
            "matches_or_beats_milp_%", "unit_hour_accuracy_%", "LPs_per_instance"]
    md = fmt_table(rows, cols)
    print(md)
    with open(os.path.join(out_dir, "uc_label_free_results.md"), "w") as f:
        f.write(md + "\n" + "\n".join(f"- {k}: {v}" for k, v in extra.items()) + "\n")
    with open(os.path.join(out_dir, "uc_label_free_results.json"), "w") as f:
        json.dump({"rows": rows, "extra": extra}, f, indent=1, default=float)
    oracle.close()
