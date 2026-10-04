"""Model 2 study: predicting the DC solution for a given topology, and using Model 2 as a critic.

    python scripts/run_model2.py --cfg case118

Part A  direct (PG, VA) regression (original framework) vs physics-consistent decoder
        (PG -> power-balance repair -> VA from B(z) theta = P)  -> violations, cost error.
Part B  Model 2 as a fast screener: rank candidate topologies by predicted cost, LP-verify top-m.
Part C  the dashed arrow: train Model 1 through a frozen Model 2 (surrogate critic) and score the
        resulting topologies with the exact LP (tests for surrogate exploitation).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.data import load, make_model, solve_lps  # noqa: E402
from otsl.features import Featurizer  # noqa: E402
from otsl.pipeline import LPOracle, candidates_from_probs, decode_threshold, metrics, pick_best  # noqa: E402
from otsl.train import build_model1, build_model2, predict_model2, train_bce, train_model2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import CONFIGS  # noqa: E402
from run_model1 import fmt_table  # noqa: E402

SCEN_KEYS = ["pd", "c0", "lmp0", "mu0", "gamma0", "flow0", "va0", "pg0"]


def perturbed_topologies(d, sw, K, n_rand, rng):
    """For every scenario: MILP topology, all-closed, and n_rand random topologies that open 1..K
    switchable lines (Model 1 will make mistakes, so Model 2 must see non-optimal topologies)."""
    L = d["z"].shape[1]
    swi = np.where(sw)[0]
    rows, zs = [], []
    for i in range(len(d["pd"])):
        cand = [d["z"][i], np.ones(L, np.int8)]
        for _ in range(n_rand):
            z = np.ones(L, np.int8)
            z[rng.choice(swi, rng.integers(1, K + 1), replace=False)] = 0
            cand.append(z)
        for z in cand:
            rows.append(i); zs.append(z)
    return np.array(rows), np.array(zs)


def build_m2_set(cfg, d, sw, K, n_rand, rng, workers):
    rows, zs = perturbed_topologies(d, sw, K, n_rand, rng)
    c, pg, va, fl = solve_lps(cfg, d["pd"][rows], zs, workers)
    ok = np.isfinite(c)
    out = {k: d[k][rows][ok] for k in SCEN_KEYS}
    out.update(zz=zs[ok], lp_c=c[ok], lp_pg=pg[ok], lp_va=va[ok], lp_f=fl[ok], scen=rows[ok],
               is_milp=np.tile(np.r_[1, np.zeros(1 + n_rand)], len(d["pd"]))[ok].astype(bool))
    return out, (~ok).mean()


def evaluate_m2(m2, feat, d):
    t = time.time()
    pg, va = predict_model2(m2, feat, d)
    dt = (time.time() - t) / len(pg)
    P = m2.phys
    with torch.no_grad():
        v = P.violations(torch.as_tensor(pg), torch.as_tensor(va), torch.as_tensor(d["pd"]),
                         torch.as_tensor(d["zz"], dtype=torch.float64))
        cost = P.gen_cost(torch.as_tensor(pg)).numpy()
    v = {k: x.numpy() for k, x in v.items()}
    base = d["pd"].sum(1).mean()
    tol = 1e-3  # 0.1 MW on a 100 MVA base
    feas = (v["kcl_max"] < tol) & (v["gen_max"] < tol) & (v["line_max"] < tol)
    return {
        "pg_mse": float(((pg - d["lp_pg"]) ** 2).mean()), "va_mse": float(((va - d["lp_va"]) ** 2).mean()),
        "kcl_max_pu": float(v["kcl_max"].mean()), "kcl_total_rel_%": float((v["kcl_sum"] / base).mean() * 100),
        "gen_viol_max_pu": float(v["gen_max"].mean()), "line_viol_max_pu": float(v["line_max"].mean()),
        "lines_overloaded": float(v["line_n"].mean()), "feasible_%": float(feas.mean() * 100),
        "cost_abs_err_%": float((np.abs(cost - d["lp_c"]) / d["lp_c"]).mean() * 100),
        "cost_signed_err_%": float(((cost - d["lp_c"]) / d["lp_c"]).mean() * 100),
        "time_ms_per_sample": dt * 1000,
    }, pg, va, cost, v


def m2_cost_scores(m2, feat, pds_rows, scen_d, zs, w_over=1e4):
    """Predicted cost of candidate topologies (+ penalty for predicted overloads)."""
    d = {k: scen_d[k][pds_rows] for k in SCEN_KEYS}
    d["zz"] = zs
    pg, va = predict_model2(m2, feat, d)
    P = m2.phys
    with torch.no_grad():
        f = P.flows(torch.as_tensor(va), torch.as_tensor(zs, dtype=torch.float64))
        over = F.relu(f.abs() - P.fmax * torch.as_tensor(zs, dtype=torch.float64)).sum(-1).numpy()
        cost = P.gen_cost(torch.as_tensor(pg)).numpy()
    return cost + w_over * over


def train_model1_through_critic(m1, m2, feat2, tr, K, sc, epochs=40, bs=64, lr=3e-4, w_budget=1.0,
                                w_over=1e3, straight_through=False, seed=0):
    """The dashed arrow: minimise Model-2-predicted cost of Model 1's (relaxed) topology."""
    rng = np.random.default_rng(seed)
    for p_ in m2.parameters():
        p_.requires_grad_(False)
    m2.eval()
    opt = torch.optim.Adam(m1.net.parameters(), lr=lr)
    P = m2.phys
    n = len(tr["pd"])
    for ep in range(epochs):
        m1.net.train()
        perm = rng.permutation(n)
        tot = []
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg = m1.logits(tr, idx)
            p_open = torch.sigmoid(lg)
            if straight_through:
                hard = torch.zeros_like(p_open)
                top = lg.topk(K, -1).indices
                hard.scatter_(1, top, 1.0)
                hard = hard * (p_open > 0.5)
                p_open = hard + p_open - p_open.detach()
            z = 1.0 - p_open
            x, e = feat2(tr, idx)
            pd = torch.as_tensor(tr["pd"][idx])
            pg, va = m2(x, e, z.double(), pd)
            f = P.flows(va, z.double())
            c0 = torch.as_tensor(tr["c0"][idx])
            loss = ((P.gen_cost(pg) + sc * p_open.double().sum(-1)) / c0).mean() \
                + w_over * (F.relu(f.abs() - P.fmax * z.double()) ** 2).sum(-1).mean() \
                + w_budget * F.relu(p_open.sum(-1) - K).double().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append(loss.item())
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [critic-trained M1 st={straight_through}] ep {ep} surrogate loss {np.mean(tot):.6f}", flush=True)
    return m1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="case118")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_rand", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = CONFIGS[a.cfg]
    root = os.path.join("data", "generated", a.cfg)
    tr, va, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    m = make_model(cfg)
    case, K, sc = m.case, cfg["budget"], float(tr["switch_cost"])
    sw = ~m.fixed_closed
    rng = np.random.default_rng(a.seed)
    out_dir = os.path.join("results", a.cfg)
    os.makedirs(out_dir, exist_ok=True)
    res = {}

    # ------------------------------------------------------------------ Part A
    sub = {k: v[:a.n_train] for k, v in tr.items() if isinstance(v, np.ndarray) and v.ndim >= 1}
    d_tr, inf_tr = build_m2_set(cfg, sub, sw, K, a.n_rand, rng, a.workers)
    d_va, _ = build_m2_set(cfg, va, sw, K, a.n_rand, rng, a.workers)
    d_te, inf_te = build_m2_set(cfg, te, sw, K, a.n_rand, rng, a.workers)
    res["m2_data"] = {"train": len(d_tr["pd"]), "val": len(d_va["pd"]), "test": len(d_te["pd"]),
                      "random_topology_infeasible_frac": float(inf_tr)}
    print("Model-2 data:", res["m2_data"])
    feat = Featurizer(case, d_tr, duals=False, fixed_closed=m.fixed_closed)
    variants = [("direct (PG,VA) regression [framework]", "direct", 0.0),
                ("physics decoder", "physics", 0.0),
                ("physics decoder + overload penalty", "physics", 100.0)]
    rows, m2s = [], {}
    for name, mode, wf in variants:
        print("training Model 2:", name)
        m2 = build_model2(case, feat, mode, seed=a.seed)
        t = time.time()
        train_model2(m2, feat, d_tr, d_va, epochs=a.epochs, w_flow=wf, seed=a.seed)
        r, *_ = evaluate_m2(m2, feat, d_te)
        r["method"], r["train_s"] = name, time.time() - t
        rows.append(r)
        m2s[name] = m2
        torch.save(m2.state_dict(), os.path.join(out_dir, f"model2_{mode}_{int(wf)}.pt"))
        print("  ", {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
    # LP timing for reference
    t = time.time()
    _ = solve_lps(cfg, d_te["pd"][:400], d_te["zz"][:400], 1)
    res["lp_time_ms_per_sample_1core"] = (time.time() - t) / 400 * 1000
    cols = ["method", "pg_mse", "va_mse", "kcl_max_pu", "kcl_total_rel_%", "gen_viol_max_pu", "line_viol_max_pu",
            "lines_overloaded", "feasible_%", "cost_abs_err_%", "time_ms_per_sample"]
    md_a = fmt_table(rows, cols)
    print(md_a)

    # ------------------------------------------------------------------ Part B: screening
    oracle = LPOracle(cfg, a.workers)
    keys = np.arange(len(te["pd"])) + 2 * 10 ** 7
    featd = Featurizer(case, tr, duals=True, fixed_closed=m.fixed_closed)
    m1 = build_model1(case, featd, sw, "gnn", seed=a.seed)
    sd_path = os.path.join(out_dir, "model1_GNN-BCE_+duals.pt")
    if os.path.exists(sd_path):
        m1.net.load_state_dict(torch.load(sd_path))
    else:
        train_bce(m1, tr, va, epochs=100)
    p = m1.predict(te)
    cl = candidates_from_probs(p, K, sw, n_samples=32, top_m=K + 5)
    rows_idx = np.concatenate([np.full(len(c), i) for i, c in enumerate(cl)])
    zs = np.concatenate(cl)
    true_c = oracle.costs(te["pd"][rows_idx], zs, keys[rows_idx]) + sc * (1 - zs).sum(1)
    brows = []
    zb_all, cb_all, _ = pick_best(oracle, te["pd"], cl, keys, sc)
    brows.append(metrics(cb_all, te, zb_all, "LP-verify ALL candidates",
                         LPs_per_scenario=float(np.mean([len(c) for c in cl]))))
    spear = {}
    for name in [v[0] for v in variants]:
        score = m2_cost_scores(m2s[name], feat, rows_idx, te, zs) + sc * (1 - zs).sum(1)
        rho = []
        for i in range(len(cl)):
            mk = rows_idx == i
            tc, sc_ = true_c[mk], score[mk]
            fin = np.isfinite(tc)
            if fin.sum() > 2:
                rho.append(np.corrcoef(np.argsort(np.argsort(tc[fin])), np.argsort(np.argsort(sc_[fin])))[0, 1])
        spear[name] = float(np.nanmean(rho))
        for mtop in [1, 3]:
            zsel, csel = [], []
            for i in range(len(cl)):
                mk = np.where(rows_idx == i)[0]
                top = mk[np.argsort(score[mk])[:mtop]]
                # always also verify all-closed (fallback) -> never worse than DC-OPF
                cand = np.concatenate([top, mk[(zs[mk] == 1).all(1)]])
                j = cand[np.argmin(true_c[cand])]
                zsel.append(zs[j]); csel.append(true_c[j] - sc * (1 - zs[j]).sum())
            brows.append(metrics(np.array(csel), te, np.array(zsel), f"Model 2 [{name}] picks top-{mtop} -> LP",
                                 LPs_per_scenario=float(mtop + 1), spearman=spear[name]))
    # random screening baseline
    for mtop in [3]:
        zsel, csel = [], []
        for i in range(len(cl)):
            mk = np.where(rows_idx == i)[0]
            top = rng.choice(mk, min(mtop, len(mk)), replace=False)
            cand = np.concatenate([top, mk[(zs[mk] == 1).all(1)]])
            j = cand[np.argmin(true_c[cand])]
            zsel.append(zs[j]); csel.append(true_c[j] - sc * (1 - zs[j]).sum())
        brows.append(metrics(np.array(csel), te, np.array(zsel), f"random top-{mtop} -> LP",
                             LPs_per_scenario=float(mtop + 1)))
    res["spearman_per_scenario"] = spear
    cols_b = ["method", "gap_mean_%", "benefit_captured_%", "beats_or_ties_milp_%", "LPs_per_scenario", "spearman"]
    md_b = fmt_table(brows, cols_b)
    print(md_b)

    # ------------------------------------------------------------------ Part C: dashed arrow
    crows = []
    base = metrics(oracle.costs(te["pd"], decode_threshold(p, K, sw), keys), te, decode_threshold(p, K, sw),
                   "BCE-trained Model 1 (reference)")
    crows.append(base)
    for st in [False, True]:
        for crit_name in ["physics decoder + overload penalty", "direct (PG,VA) regression [framework]"]:
            m1c = build_model1(case, featd, sw, "gnn", seed=a.seed)
            m1c.net.load_state_dict(m1.net.state_dict())          # start from the imitation model
            m2c = m2s[crit_name]
            # the critic takes base features (demand + z); build them for the training scenarios
            train_model1_through_critic(m1c, m2c, feat, tr, K, sc, epochs=30, straight_through=st, seed=a.seed)
            pc = m1c.predict(te)
            zc = decode_threshold(pc, K, sw)
            cc = oracle.costs(te["pd"], zc, keys)
            # what the critic believed vs truth
            sc_pred = m2_cost_scores(m2c, feat, np.arange(len(zc)), te, zc, w_over=0.0)
            fin = np.isfinite(cc)
            r = metrics(cc, te, zc, f"M1 trained via frozen Model 2 [{crit_name}] st={st}",
                        critic_cost_err_on_own_choice_pct=float(np.mean(np.abs(sc_pred[fin] - cc[fin]) / cc[fin]) * 100))
            crows.append(r)
            print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    cols_c = ["method", "gap_mean_%", "benefit_captured_%", "feasible_%", "n_open", "critic_cost_err_on_own_choice_pct"]
    md_c = fmt_table(crows, cols_c)
    print(md_c)
    with open(os.path.join(out_dir, "model2_results.md"), "w") as f:
        f.write("## A. Model 2 accuracy and constraint satisfaction (test topologies)\n\n" + md_a +
                f"\nLP solve time (HiGHS, 1 core): {res['lp_time_ms_per_sample_1core']:.1f} ms/sample\n\n"
                "## B. Model 2 as a candidate screener\n\n" + md_b +
                "\n## C. Training Model 1 through Model 2 (dashed arrow)\n\n" + md_c)
    with open(os.path.join(out_dir, "model2_results.json"), "w") as f:
        json.dump({"A": rows, "B": brows, "C": crows, "extra": res}, f, indent=1, default=float)
    oracle.close()
