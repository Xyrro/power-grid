"""Unit commitment, Model 1 study: how should the on/off decisions be learned and used?

    python scripts/uc_model1.py --cfg uc1        (single period)
    python scripts/uc_model1.py --cfg uc12       (12-hour look-ahead, min up/down repair)

Every predicted commitment is scored by the exact fixed-commitment dispatch LP (the framework's
"LP solver"); costs include VOLL-priced load shedding / over-generation and reserve shortfall.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import UCModel, adequacy_repair, load_rts_gmlc, repair_min_updown  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from otsl.ucml import (DispatchOracle, UCFeaturizer, build_uc_model1, canonical_labels, train_uc_bce,  # noqa: E402
                       train_uc_reinforce, uc_metrics)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model1 import fmt_table  # noqa: E402
from uc_gen import UC_CONFIGS  # noqa: E402


def knn_candidates(tr, te, k):
    f = lambda d: np.c_[d["load"].reshape(len(d["load"]), -1), d["avail"].reshape(len(d["avail"]), -1), d["u0"]]
    a, b = f(tr), f(te)
    mu, sd = a.mean(0), a.std(0) + 1e-9
    a, b = (a - mu) / sd, (b - mu) / sd
    dist = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None] - 2 * b @ a.T
    nn_idx = np.argsort(dist, 1)[:, :k]
    return [tr["u"][nn_idx[i]] for i in range(len(te["load"]))]


def candidates_from_probs(p, n_samples, rng):
    """Thresholds 0.2..0.8, plus Bernoulli samples."""
    out = []
    for i in range(len(p)):
        c = [(p[i] > th).astype(np.int8) for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
        c += [(rng.random(p[i].shape) < p[i]).astype(np.int8) for _ in range(n_samples)]
        out.append(np.unique(np.array(c), axis=0))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="uc1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n_fix", type=int, default=200, help="test instances for the partial-fixing MILPs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = UC_CONFIGS[a.cfg]
    T = cfg["T"]
    root = os.path.join("data", "generated", a.cfg)
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    sysm = load_rts_gmlc()
    out_dir = os.path.join("results", a.cfg)
    os.makedirs(out_dir, exist_ok=True)
    rep = (lambda u, u0: repair_min_updown(u, u0, sysm.min_up, sysm.min_dn)) if T > 1 else None
    fixrep = (lambda U, U0: np.array([rep(U[i], U0[i]) for i in range(len(U))])) if rep else (lambda U, U0: U)

    def adequate(U, d):
        """adequacy repair (capacity / minimum-output check per period), then min up/down repair"""
        V = np.array([adequacy_repair(U[i], d["load"][i], d["avail"][i], d["sr"][i], sysm) for i in range(len(U))])
        return fixrep(V, d["u0"])
    oracle = DispatchOracle(cfg, a.workers)
    n_te = len(te["load"])
    keys_te = np.arange(n_te) + 10 ** 7
    rows, extra = [], {"n_train": int(len(tr["load"])), "n_test": n_te, "milp_time_mean_s": float(te["time"].mean()),
                       "milp_optimal_%": float(te["opt"].mean() * 100)}
    print(f"{a.cfg}: T={T} train {len(tr['load'])} test {n_te}  MILP {te['time'].mean():.2f}s")

    def score(label, U, t_extra=0.0, **kw):
        c, sh, so = oracle.evaluate(te, np.arange(n_te), U, keys_te)
        r = uc_metrics(c, sh, so, te, U, label, **kw)
        rows.append(r)
        print(f"  {label:58s} gap mean {r['gap_mean_%']:9.3f}%  median {r['gap_median_%']:.3f}%  "
              f"no-shed {r['no_shed_no_shortfall_%']:5.1f}%  match {r['matches_or_beats_milp_%']:5.1f}%", flush=True)
        return c

    def screen(label, cand_lists, **kw):
        flat_i = np.concatenate([np.full(len(cl), i) for i, cl in enumerate(cand_lists)])
        flat_u = np.concatenate(cand_lists)
        if rep:
            flat_u = fixrep(flat_u, te["u0"][flat_i])
        c, sh, so = oracle.evaluate(te, flat_i, flat_u, keys_te[flat_i])
        best = np.array([np.where(flat_i == i)[0][np.argmin(c[flat_i == i])] for i in range(n_te)])
        r = uc_metrics(c[best], sh[best], so[best], te, flat_u[best], label,
                       LPs_per_instance=float(np.mean([len(x) for x in cand_lists])), **kw)
        rows.append(r)
        print(f"  {label:58s} gap mean {r['gap_mean_%']:9.3f}%  median {r['gap_median_%']:.3f}%  "
              f"no-shed {r['no_shed_no_shortfall_%']:5.1f}%  LPs {r['LPs_per_instance']:.1f}", flush=True)
        return flat_u[best]

    # ------------------------------------------------------------- label ambiguity
    if "alt_c" in te:
        rel = (te["alt_c"] - te["obj"][:, None]) / te["obj"][:, None]
        extra["frac_alt_commitment_within_1e-6"] = float((rel <= 1e-6).any(1).mean())
        extra["frac_alt_commitment_within_1e-4"] = float((rel <= 1e-4).any(1).mean())
    canon_te = canonical_labels(sysm, te["u"], te["u0"])
    extra["frac_labels_changed_by_canonicalisation"] = float((canon_te != te["u"]).any((1, 2)).mean())
    print("ambiguity:", {k: round(v, 3) for k, v in extra.items() if "frac" in k})

    # ------------------------------------------------------------- baselines
    score("persistence (keep units that were running)", np.repeat(te["u0"][:, None], T, 1))
    score("rounded LP relaxation", fixrep((te["u_rel"] > 0.5).astype(np.int8), te["u0"]))
    score("rounded LP relaxation + adequacy repair", adequate((te["u_rel"] > 0.5).astype(np.int8), te))
    score("persistence + adequacy repair", adequate(np.repeat(te["u0"][:, None], T, 1), te))
    score("merit-order priority list (no learning)", adequate(np.zeros_like(te["u"]), te))
    rows.append(uc_metrics(te["c_rel"], np.zeros(n_te), np.zeros(n_te), te, None, "LP relaxation (lower bound)"))
    for k in (5, 20):
        screen(f"kNN-LP k={k} (Xavier et al. 2021 style)", knn_candidates(tr, te, k))

    # ------------------------------------------------------------- learned Model 1
    canon_tr, canon_va = canonical_labels(sysm, tr["u"], tr["u0"]), canonical_labels(sysm, va["u"], va["u0"])
    variants = [
        ("MLP, BCE on MILP labels (framework)", "mlp", dict(relax=False, sym=False), False),
        ("GNN, BCE on MILP labels (framework)", "gnn", dict(relax=False, sym=False), False),
        ("GNN + symmetry rank + canonical labels", "gnn", dict(relax=False, sym=True), True),
        ("GNN + symmetry + LP-relaxation features", "gnn", dict(relax=True, sym=True), True),
    ]
    models, rng = {}, np.random.default_rng(a.seed)
    for name, kind, fo, canon in variants:
        feat = UCFeaturizer(sysm, tr, **fo)
        m1 = build_uc_model1(sysm, feat, T, kind, seed=a.seed)
        va_t = dict(va, u_target=canon_va if canon else va["u"])
        t0 = time.time()
        train_uc_bce(m1, tr, va_t, canon_tr if canon else tr["u"], epochs=a.epochs, seed=a.seed)
        extra[f"train_s[{name}]"] = time.time() - t0
        p = m1.predict(te)
        U = fixrep((p > 0.5).astype(np.int8), te["u0"])
        score(f"{name}: top-1 -> LP", U)
        score(f"{name}: top-1 + adequacy repair -> LP", adequate((p > 0.5).astype(np.int8), te))
        screen(f"{name}: candidate screening -> LP", candidates_from_probs(p, 8, rng))
        models[name] = m1
        torch.save(m1.net.state_dict(), os.path.join(out_dir, f"uc_model1_{len(models)}.pt"))

    # ------------------------------------------------------------- cost-aware fine-tuning (LP critic)
    import copy
    best_name = "GNN + symmetry + LP-relaxation features"
    m_rl = copy.deepcopy(models[best_name])
    t0 = time.time()
    _, hist = train_uc_reinforce(m_rl, tr, oracle, steps=a.rl_steps, seed=a.seed, repair=rep)
    extra["rl_time_s"] = time.time() - t0
    extra["rl_curve_gap_%"] = [float(np.mean(hist[i:i + 10]) * 100) for i in range(0, len(hist), 10)]
    p_rl = m_rl.predict(te)
    score(f"{best_name} + REINFORCE (LP critic): top-1 -> LP", fixrep((p_rl > 0.5).astype(np.int8), te["u0"]))
    score(f"{best_name} + REINFORCE: top-1 + adequacy repair -> LP", adequate((p_rl > 0.5).astype(np.int8), te))
    screen(f"{best_name} + REINFORCE: candidate screening -> LP", candidates_from_probs(p_rl, 8, rng))
    torch.save(m_rl.net.state_dict(), os.path.join(out_dir, "uc_model1_rl.pt"))

    # ------------------------------------------------------------- RACLearn-style confidence fixing + MILP
    m = UCModel(sysm, T=T, network=cfg.get("network", True))
    p = models[best_name].predict(te)
    conf = np.abs(p - 0.5)
    nf = min(a.n_fix, n_te)
    for ratio in (0.5, 0.8, 0.9, 0.95):
        costs, times, sheds, shorts, us = [], [], [], [], []
        for i in range(nf):
            flat = np.argsort(-conf[i].reshape(-1))[:int(ratio * conf[i].size)]
            tt, gg = np.unravel_index(flat, conf[i].shape)
            fix = {(int(t_), int(g_)): int(p[i, t_, g_] > 0.5) for t_, g_ in zip(tt, gg)}
            sol = m.solve_uc(scenario_from(te, i), time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], z_fix=fix)
            if sol.u is None:     # repair: drop the fixings
                sol = m.solve_uc(scenario_from(te, i), time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"])
            costs.append(sol.obj); times.append(sol.time); sheds.append(sol.shed); shorts.append(sol.short); us.append(sol.u)
        sub = {k: v[:nf] for k, v in te.items() if isinstance(v, np.ndarray) and v.ndim >= 1}
        r = uc_metrics(np.array(costs), np.array(sheds), np.array(shorts), sub, np.array(us),
                       f"confidence fixing {int(ratio * 100)} % + MILP (RACLearn-style)",
                       time_s=float(np.mean(times)), milp_time_s=float(te["time"][:nf].mean()))
        rows.append(r)
        print(f"  {r['method']:58s} gap {r['gap_mean_%']:.4f}%  time {r['time_s']:.2f}s vs {r['milp_time_s']:.2f}s", flush=True)

    # ------------------------------------------------------------- the framework's test metric
    c_b, _, _ = oracle.evaluate(te, np.arange(n_te), fixrep((p > 0.5).astype(np.int8), te["u0"]), keys_te)
    extra["note_mse"] = "computed in uc_model2.py from dispatch solutions"

    cols = ["method", "no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%", "matches_or_beats_milp_%",
            "unit_hour_accuracy_%", "exact_match_%", "LPs_per_instance", "time_s"]
    md = fmt_table(rows, cols)
    print(md)
    with open(os.path.join(out_dir, "uc_model1_results.md"), "w") as f:
        f.write(md + "\n" + "\n".join(f"- {k}: {v}" for k, v in extra.items() if not isinstance(v, list)) + "\n")
    with open(os.path.join(out_dir, "uc_model1_results.json"), "w") as f:
        json.dump({"rows": rows, "extra": extra}, f, indent=1, default=float)
    oracle.close()
