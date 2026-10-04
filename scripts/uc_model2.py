"""Unit commitment, Model 2 study: predicting (PG, VA) for a given commitment, and the dashed arrow.

    python scripts/uc_model2.py --cfg uc1

A  direct (PG, VA) regression (framework) vs physics decoder (unit position in [pmin, pmax] -> exact
   balance repair -> VA from DC power flow): violations and cost error on test (demand, commitment).
B  Model 2 as a screener of candidate commitments (kNN candidates) vs LP-checking them all.
C  the dashed arrow: Model 1 fine-tuned through a frozen Model 2 (relaxed / straight-through u),
   scored with the exact dispatch LP.
D  the framework's test metric: MSE of (PG, VA) for alternative optimal commitments.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import RES_SHORT, UCModel, UCScenario, load_rts_gmlc, repair_min_updown  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import (DispatchOracle, UCDispatchNet, UCFeaturizer, build_uc_model1, canonical_labels,  # noqa: E402
                       train_uc_bce, train_uc_dualgrad, uc_metrics)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model1 import fmt_table  # noqa: E402
from uc_gen import UC_CONFIGS  # noqa: E402
from uc_model1 import knn_candidates  # noqa: E402

_W = {}


def _init(cfg):
    s = load_rts_gmlc()
    _W.update(m=UCModel(s, T=cfg["T"], network=True))


def _full(args):
    load_, avail, u0, sr, u = args
    sol = _W["m"].solve_dispatch(UCScenario(load=load_, avail=avail, u0=u0, sr=sr), u)
    return sol.obj, sol.p, sol.va, sol.r, sol.shed, sol.short


def dispatch_all(cfg, d, idx, us, workers):
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_full, [(d["load"][i], d["avail"][i], d["u0"][i], d["sr"][i], u) for i, u in zip(idx, us)],
                       chunksize=8)
    return [np.array([r[k] for r in res]) for k in range(6)]


def perturb(sysm, d, n_rand, rng, T):
    idx, us = [], []
    for i in range(len(d["load"])):
        cands = [d["u"][i]]
        for _ in range(n_rand):
            u = d["u"][i].copy()
            for _ in range(rng.integers(1, 4)):        # flip 1-3 units over a random window
                t, g = rng.integers(0, T), rng.integers(0, sysm.G)
                u[t:t + rng.integers(1, T - t + 1), g] = 1 - u[t, g]
            if T > 1:
                u = repair_min_updown(u, d["u0"][i], sysm.min_up, sysm.min_dn)
            cands.append(u)
        for u in cands:
            idx.append(i); us.append(u)
    return np.array(idx), np.array(us)


def sub(d, idx):
    return {k: v[idx] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}


def m2_forward(m2, feat, dd, idx, u):
    xb, xg, xe = feat(dd, idx)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    return m2(xb, xg, xe, u, t(dd["load"][idx]), t(dd["avail"][idx]), t(dd["u0"][idx]))


def train_m2(m2, feat, d, us, lp, epochs, w_line, seed=0, bs=64):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    P = m2.phys
    pg_t, va_t = torch.as_tensor(lp[1]), torch.as_tensor(lp[2])
    scale = torch.as_tensor(np.maximum(lp[1].std(0), 1e-3)), torch.as_tensor(np.maximum(lp[2].std(0), 1e-3))
    n = len(us)
    t0 = time.time()
    for ep in range(epochs):
        m2.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            u = torch.as_tensor(us[j], dtype=torch.float64)
            pg, va, r = m2_forward(m2, feat, d, j, u)
            loss = (((pg - pg_t[j]) / scale[0]) ** 2).mean() + (((va - va_t[j]) / scale[1]) ** 2).mean()
            if w_line > 0:
                loss = loss + w_line * (F.relu(P.flows(va).abs() - P.fmax) ** 2).sum((-1, -2)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [uc-m2:{m2.mode}] ep {ep} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return m2


@torch.no_grad()
def eval_m2(m2, feat, d, us, lp, label):
    m2.eval()
    P = m2.phys
    res = {"kcl": [], "gen": [], "line": [], "ramp": [], "cost": [], "cost_full": [], "mis": []}
    for i in range(0, len(us), 256):
        j = np.arange(i, min(i + 256, len(us)))
        u = torch.as_tensor(us[j], dtype=torch.float64)
        pg, va, r = m2_forward(m2, feat, d, j, u)
        ld = torch.as_tensor(d["load"][j])
        v = P.violations(pg, va, u, ld, r)
        mis_t = (pg.sum(-1) + r.sum(-1) - ld.sum(-1)).abs()                     # [B, T]
        u0 = torch.as_tensor(d["u0"][j], dtype=torch.float64)
        res["kcl"].append(v["kcl_max"].numpy()); res["gen"].append(v["gen_max"].numpy())
        res["line"].append(v["line_max"].numpy()); res["ramp"].append(v["ramp_max"].numpy())
        # cost of the predicted PG alone (what the framework's Model 2 output prices) ...
        res["cost"].append(P.cost(pg, u, u0, torch.zeros_like(mis_t)).numpy())
        # ... and with the slacks the dispatch LP would pay: imbalance at VOLL, spinning-reserve
        # shortfall (headroom sum(pmax u - p) below the requirement) at the reserve penalty
        short = F.relu(torch.as_tensor(d["sr"][j]) - (u * P.pmax - pg).clamp_min(0).sum(-1))
        res["cost_full"].append((P.cost(pg, u, u0, mis_t) + RES_SHORT * short.sum(-1)).numpy())
        res["mis"].append(mis_t.sum(-1).numpy())
    R = {k: np.concatenate(v) for k, v in res.items()}
    ok = np.isfinite(lp[0]) & (lp[4] < 1e-6)
    tol = 1e-3
    feas = (R["kcl"] < tol) & (R["gen"] < tol) & (R["line"] < tol) & (R["ramp"] < tol)
    return {"method": label, "fully_feasible_%": float(feas.mean() * 100),
            "worst_KCL_MW": float(R["kcl"].mean() * 100), "worst_overload_MW": float(R["line"].mean() * 100),
            "worst_gen_limit_MW": float(R["gen"].mean() * 100), "worst_ramp_MW": float(R["ramp"].mean() * 100),
            "cost_abs_err_%": float((np.abs(R["cost"][ok] - lp[0][ok]) / lp[0][ok]).mean() * 100)}, R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="uc1")
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dg_steps", type=int, default=300)
    ap.add_argument("--critic_epochs", type=int, default=15)
    ap.add_argument("--parts", default="ABCD", help="subset of the study to run (A is needed by B and C)")
    ap.add_argument("--reuse", action="store_true", help="load Model 2 weights saved by an earlier run")
    a = ap.parse_args()
    cfg = UC_CONFIGS[a.cfg]
    T = cfg["T"]
    root = os.path.join("data", "generated", a.cfg)
    tr, va_, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    sysm = load_rts_gmlc()
    rng = np.random.default_rng(0)
    out_dir = os.path.join("results", a.cfg)
    trs, tes = sub(tr, np.arange(min(a.n_train, len(tr["load"])))), sub(te, np.arange(min(a.n_test, len(te["load"]))))
    res = {}

    # ---------------------------------------------------------------- A
    i_tr, u_tr = perturb(sysm, trs, 2, rng, T)
    i_te, u_te = perturb(sysm, tes, 2, rng, T)
    lp_tr = dispatch_all(cfg, trs, i_tr, u_tr, a.workers)
    lp_te = dispatch_all(cfg, tes, i_te, u_te, a.workers)
    dtr, dte = sub(trs, i_tr), sub(tes, i_te)
    feat = UCFeaturizer(sysm, dtr, relax=False, sym=True)
    rows, m2s = [], {}
    for name, mode, wl in [("direct (PG, VA) regression [framework]", "direct", 0.0),
                           ("physics decoder", "physics", 0.0),
                           ("physics decoder + overload penalty", "physics", 10.0)]:
        torch.manual_seed(0)
        m2 = UCDispatchNet(sysm, feat, T, mode)
        t0 = time.time()
        wpath = os.path.join(out_dir, f"m2_{mode}_w{wl:g}.pt")
        if a.reuse and os.path.exists(wpath):
            m2.load_state_dict(torch.load(wpath))
        else:
            train_m2(m2, feat, dtr, u_tr, lp_tr, a.epochs, wl)
            torch.save(m2.state_dict(), wpath)
        r, _ = eval_m2(m2, feat, dte, u_te, lp_te, name)
        r["train_s"] = time.time() - t0
        rows.append(r); m2s[name] = m2
        print("  ", r, flush=True)
    t0 = time.time()
    _ = dispatch_all(cfg, tes, i_te[:200], u_te[:200], 1)
    res["lp_ms_per_dispatch_1core"] = (time.time() - t0) / 200 * 1000
    md_a = fmt_table(rows, ["method", "fully_feasible_%", "worst_KCL_MW", "worst_overload_MW", "worst_gen_limit_MW",
                            "worst_ramp_MW", "cost_abs_err_%"])
    print(md_a)

    # ---------------------------------------------------------------- B: screening kNN candidates
    oracle = DispatchOracle(cfg, a.workers)
    keys = np.arange(len(tes["load"])) + 3 * 10 ** 7
    cl = knn_candidates(tr, tes, 20)
    flat_i = np.concatenate([np.full(len(c), i) for i, c in enumerate(cl)])
    flat_u = np.concatenate(cl)
    if T > 1:
        flat_u = np.array([repair_min_updown(u, tes["u0"][i], sysm.min_up, sysm.min_dn) for i, u in zip(flat_i, flat_u)])
    c_true, sh, so = oracle.evaluate(tes, flat_i, flat_u, keys[flat_i])
    brows = []

    def pick(scores, m, label, lps=None, **kw):
        sel = []
        for i in range(len(cl)):
            mk = np.where(flat_i == i)[0]
            top = mk[np.argsort(scores[mk])[:m]]
            sel.append(top[np.argmin(c_true[top])])
        sel = np.array(sel)
        brows.append(uc_metrics(c_true[sel], sh[sel], so[sel], tes, flat_u[sel], label, LPs=float(lps or m), **kw))
    pick(c_true, 10 ** 6, "LP-check all 20 kNN candidates", lps=20)
    dsub = sub(tes, flat_i)
    for name, m2 in m2s.items():
        _, R = eval_m2(m2, feat, dsub, flat_u, (c_true, None, None, None, sh, so), name)
        for ck, what in [("cost", "PG cost"), ("cost_full", "PG cost + implied shed/reserve slack")]:
            rho = np.nanmean([np.corrcoef(np.argsort(np.argsort(c_true[flat_i == i])),
                                          np.argsort(np.argsort(R[ck][flat_i == i])))[0, 1] for i in range(len(cl))])
            pick(R[ck], 3, f"Model 2 [{name}], {what}: top-3 -> LP", spearman=float(rho))
    pick(rng.random(len(flat_u)), 3, "random 3 -> LP")
    md_b = fmt_table(brows, ["method", "no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%",
                             "LPs", "spearman"])
    print(md_b)

    # ---------------------------------------------------------------- C: the dashed arrow
    md_c, crows = "", []
    if "C" in a.parts:
        canon_tr = canonical_labels(sysm, tr["u"], tr["u0"])
        feat1 = UCFeaturizer(sysm, tr, relax=True, sym=True)
        m1 = build_uc_model1(sysm, feat1, T, "gnn", seed=0)
        train_uc_bce(m1, tr, dict(va_, u_target=canonical_labels(sysm, va_["u"], va_["u0"])), canon_tr, epochs=a.epochs)
        keys_all = np.arange(len(tes["load"])) + 4 * 10 ** 7

        def score1(model, label):
            p = model.predict(tes)
            U = (p > 0.5).astype(np.int8)
            if T > 1:
                U = np.array([repair_min_updown(U[i], tes["u0"][i], sysm.min_up, sysm.min_dn) for i in range(len(U))])
            c, s_, o_ = oracle.evaluate(tes, np.arange(len(U)), U, keys_all)
            crows.append(uc_metrics(c, s_, o_, tes, U, label))
            print("  ", crows[-1]["method"], round(crows[-1]["gap_median_%"], 4), round(crows[-1]["no_shed_no_shortfall_%"], 1), flush=True)
        score1(m1, "imitation (BCE, canonical labels) reference")
        import copy
        for crit in ["physics decoder + overload penalty", "direct (PG, VA) regression [framework]"]:
            for st in (False, True):
                mc = copy.deepcopy(m1)
                m2 = m2s[crit]
                for p_ in m2.parameters():
                    p_.requires_grad_(False)
                m2.eval()
                opt = torch.optim.Adam(mc.net.parameters(), lr=3e-4)
                P = m2.phys
                for ep in range(a.critic_epochs):
                    perm = rng.permutation(len(tr["load"]))
                    for i in range(0, len(perm), 64):
                        j = perm[i:i + 64]
                        lg = mc.logits(tr, j)
                        pu = torch.sigmoid(lg)
                        if st:
                            pu = (pu > 0.5).float() + pu - pu.detach()
                        xb, xg, xe = feat(sub(tr, j))
                        tt = lambda a_: torch.as_tensor(a_, dtype=torch.float64)
                        pg, va, r = m2(xb, xg, xe, pu.double(), tt(tr["load"][j]), tt(tr["avail"][j]), tt(tr["u0"][j]))
                        ld = tt(tr["load"][j])
                        mis = (pg.sum(-1) + r.sum(-1) - ld.sum(-1)).abs()
                        cost = P.cost(pg, pu.double(), tt(tr["u0"][j]), mis)
                        loss = (cost / tt(tr["obj"][j])).mean() + 10 * (F.relu(P.flows(va).abs() - P.fmax) ** 2).sum((-1, -2)).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                score1(mc, f"M1 trained through frozen Model 2 [{crit}] st={st}")
        mdg = copy.deepcopy(m1)
        train_uc_dualgrad(mdg, tr, cfg, steps=a.dg_steps, bs=32, workers=a.workers)
        score1(mdg, "M1 fine-tuned with exact LP sensitivities (dual gradient)")
        md_c = fmt_table(crows, ["method", "no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%",
                                 "unit_hour_accuracy_%", "units_on"])
        print(md_c)

    # ---------------------------------------------------------------- D: MSE of alternative optima
    dd = {}
    if "D" in a.parts and "alt_u" in te:
        rel = (te["alt_c"][:, 0] - te["obj"]) / te["obj"]
        tie = np.where(rel <= 1e-6)[0][:300]
        if len(tie):
            alt = dispatch_all(cfg, te, tie, te["alt_u"][tie, 0], a.workers)
            dd = {"n_tied_instances": int(len(tie)),
                  "pg_mse_tied_mean": float(((alt[1] - te["p"][tie]) ** 2).mean()),
                  "pg_mse_tied_max": float(((alt[1] - te["p"][tie]) ** 2).mean((1, 2)).max()),
                  "va_mse_tied_mean": float(((alt[2] - te["va"][tie]) ** 2).mean()),
                  "commitment_hamming_tied_mean": float((te["alt_u"][tie, 0] != te["u"][tie]).sum((1, 2)).mean())}
            print("MSE of tied optima:", dd)
    res["mse_of_tied_optima"] = dd
    tag = "" if a.parts == "ABCD" else "_" + a.parts
    with open(os.path.join(out_dir, f"uc_model2{tag}_results.md"), "w") as f:
        f.write("## A. Model 2 accuracy (test (demand, commitment) pairs)\n\n" + md_a +
                f"\nDispatch LP: {res['lp_ms_per_dispatch_1core']:.0f} ms on one core\n\n## B. Model 2 as screener\n\n" + md_b +
                "\n## C. Dashed arrow\n\n" + md_c + "\n## D. MSE between tied optimal solutions\n\n" + json.dumps(dd, indent=1))
    with open(os.path.join(out_dir, f"uc_model2{tag}_results.json"), "w") as f:
        json.dump({"A": rows, "B": brows, "C": crows, "extra": res}, f, indent=1, default=float)
    oracle.close()
