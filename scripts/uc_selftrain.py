"""Self-training with the solver as teacher (expert iteration) for 12-hour UC (B2), without MILP labels.

    python scripts/uc_selftrain.py --stage all          (resumable: finished stages are skipped)

Stages
  r0      label-free Model 1 (BCE on the repaired LP relaxation, as uc_label_free.py stage 1), its label pool
          scored by the dispatch LP, and the label-free + REINFORCE baseline (same seed and steps)
  pilot   fixing share q for label generation: q = 0.8 vs 0.9 on 30 training instances (label quality / second)
  rounds  R rounds: predict -> fix the q most confident decisions that agree with the current label (+ adequacy
          guard) -> reduced MILP (short limit) -> dispatch LP -> keep if cheaper -> retrain from scratch.
          The training set is split into R chunks; round r relabels chunk r with the round r-1 model.
  rl      REINFORCE fine-tuning of the last self-trained model: plain (RLOO) and with the solver label added to
          each sample group (self-imitation, train_reinforce_sil)
  eval    test set, one LP per decoder: top-1 (+ min up/down repair), + adequacy repair, candidate screening
  fix     first n_fix test instances: full MILP and reduced MILPs (80/90/95 % fixed) back to back per instance
  (tables: scripts/uc_selftrain_report.py -> results/uc12/selftrain_results.md / .json)

Validation uses no MILP information: gaps there are to the LP-relaxation bound c_rel. MILP objectives of training
instances appear only as a diagnostic of label quality, never in training or selection.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.selftrain import (SolverPool, _fixeval, _reduced, fix_array, label_fixings, train_bce_fixed,  # noqa: E402
                            train_reinforce_sil)
from otsl.uc import UCModel, adequacy_repair, load_rts_gmlc, repair_min_updown  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from otsl.ucml import (DispatchOracle, UCFeaturizer, build_uc_model1, train_uc_bce, train_uc_reinforce,  # noqa: E402
                       uc_metrics)
from run_model1 import fmt_table  # noqa: E402
from uc_fixing import adequacy_guard, choose_fixings  # noqa: E402
from uc_gen import UC_CONFIGS  # noqa: E402
from uc_model1 import candidates_from_probs  # noqa: E402

MILP_KEYS = ("u", "p", "va", "r", "obj", "time", "gap", "opt", "shed", "alt_u", "alt_c")
STAGES = ["r0", "pilot", "rounds", "rl", "eval", "fix"]


def sub(d, idx):
    n = len(d["load"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in d.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--tag", default="", help="file prefix suffix (smoke tests)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.0, help="fixing share for label generation (0 = pilot decides)")
    ap.add_argument("--pilot_n", type=int, default=30)
    ap.add_argument("--label_limit", type=float, default=15.0, help="time limit of label-generating reduced MILPs")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--rl_steps", type=int, default=120)
    ap.add_argument("--n_fix", type=int, default=60)
    ap.add_argument("--n_train", type=int, default=0, help="smoke tests: use the first n training instances")
    ap.add_argument("--n_test", type=int, default=0, help="smoke tests: use the first n test / val instances")
    ap.add_argument("--fix_max_specs", type=int, default=0, help="smoke tests: reduced MILPs per instance (0 = all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    cfg = UC_CONFIGS["uc12"]
    T = cfg["T"]
    root = os.path.join("data", "generated", "uc12")
    tr_full, va_full, te = (load(os.path.join(root, f"{s}.npz")) for s in ["train", "val", "test"])
    if a.n_train:
        tr_full = sub(tr_full, np.arange(a.n_train))
    if a.n_test:
        va_full, te = sub(va_full, np.arange(a.n_test)), sub(te, np.arange(a.n_test))
    sysm = load_rts_gmlc()
    OUT = os.path.join("results", "uc12")
    P = lambda name: os.path.join(OUT, f"selftrain{a.tag}_{name}")
    state_path = P("state.json")
    S = json.load(open(state_path)) if os.path.exists(state_path) else {}

    def save_state():
        with open(state_path, "w") as f:
            json.dump(S, f, indent=1, default=float)

    rep = lambda u, u0: repair_min_updown(u, u0, sysm.min_up, sysm.min_dn)
    fixrep = lambda U, U0: np.array([rep(U[i], U0[i]) for i in range(len(U))])

    def adequate(U, d):
        V = np.array([adequacy_repair(U[i], d["load"][i], d["avail"][i], d["sr"][i], sysm) for i in range(len(U))])
        return fixrep(V, d["u0"])

    strip = lambda d: dict({k: v for k, v in d.items() if k not in MILP_KEYS}, obj=d["c_rel"])
    tr, va = strip(tr_full), strip(va_full)
    N = len(tr["load"])
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    keys_tr, keys_va, keys_te = np.arange(N), np.arange(len(va["load"])) + 10 ** 6, np.arange(len(te["load"])) + 2 * 10 ** 6

    def new_model():
        return build_uc_model1(sysm, feat, T, "gnn", seed=a.seed)

    def load_model(path):
        m1 = new_model()
        m1.net.load_state_dict(torch.load(path))
        m1.net.eval()
        return m1

    _oracle = {}

    def oracle():
        if "o" not in _oracle:
            _oracle["o"] = DispatchOracle(cfg, a.workers)
        return _oracle["o"]

    def pool_load():
        z = np.load(P("pool.npz"))
        return {k: z[k].copy() for k in z.files}

    def pool_save(pl):
        np.savez_compressed(P("pool.npz"), **pl)

    def label_stats(pl, idx=None):
        """label quality; gap to the training MILP objective is a diagnostic only (never used for training)"""
        idx = np.arange(N) if idx is None else idx
        c, ref, rel = pl["C"][idx], tr_full["obj"][idx], tr["c_rel"][idx]
        served = (pl["shed"][idx] < 1e-6) & (pl["short"][idx] < 1e-6)
        g = (c - ref) / ref * 100
        return {"n": int(len(idx)), "gap_to_LP_bound_mean_%": float(((c - rel) / rel * 100).mean()),
                "gap_to_LP_bound_median_%": float(np.median((c - rel) / rel * 100)),
                "served_%": float(served.mean() * 100),
                "diag_gap_to_MILP_mean_%": float(g.mean()), "diag_gap_to_MILP_median_%": float(np.median(g)),
                "diag_gap_to_MILP_mean_served_%": float(g[served].mean()) if served.any() else float("nan"),
                "diag_label_le_MILP_%": float((g <= 1e-3).mean() * 100),
                "units_on": float(pl["Y"][idx].sum((1, 2)).mean() / T),
                "diag_MILP_units_on": float(tr_full["u"][idx].sum((1, 2)).mean() / T)}

    def val_eval(m1, name):
        """validation without MILP information: one LP per decoder, gap to the LP-relaxation bound"""
        p = m1.predict(va)
        n = len(va["load"])
        out = {}
        for lab, U in [("top-1", fixrep((p > 0.5).astype(np.int8), va["u0"])),
                       ("adequacy", adequate((p > 0.5).astype(np.int8), va))]:
            c, sh, so = oracle().evaluate(va, np.arange(n), U, keys_va)
            r = uc_metrics(c, sh, so, va, None, lab)
            out[lab] = {k: r[k] for k in ("no_shed_no_shortfall_%", "gap_median_%", "gap_mean_served_%", "gap_mean_%")}
            out[lab]["units_on"] = float(U.sum((1, 2)).mean() / T)
        print(f"  [val] {name}: top-1 served {out['top-1']['no_shed_no_shortfall_%']:.1f}% median gap-to-bound "
              f"{out['top-1']['gap_median_%']:.2f}% | adequacy served {out['adequacy']['no_shed_no_shortfall_%']:.1f}% "
              f"median {out['adequacy']['gap_median_%']:.2f}% units {out['adequacy']['units_on']:.1f}", flush=True)
        S.setdefault("val", {})[name] = out
        save_state()
        return out

    def label_jobs(p, pl, idx, q, limit):
        jobs, released = [], 0
        for i in idx:
            fix = label_fixings(p[i], pl["Y"][i], q, agree=True)
            fix, rel = adequacy_guard(fix, tr["load"][i], tr["avail"][i], tr["sr"][i], sysm)
            released += rel
            jobs.append((int(i), tr["load"][i], tr["avail"][i], tr["u0"][i], tr["sr"][i], fix_array(fix), limit, cfg["mip_gap"]))
        return jobs, released

    def absorb(pl, res, rnd):
        """keep each reduced-MILP schedule whose dispatch cost beats the current label"""
        acc, milp_s, lp_s, to, gains = 0, 0.0, 0.0, 0, []
        for r in res:
            i = r["i"]
            milp_s += r["milp_s"]; lp_s += r["lp_s"]; to += r["status"] != "optimal"
            gains.append((pl["C"][i] - r["cost"]) / pl["C"][i] * 100 if np.isfinite(r["cost"]) else -np.inf)
            if r["u"] is not None and r["cost"] < pl["C"][i] * (1 - 1e-7):
                pl["Y"][i], pl["C"][i], pl["shed"][i], pl["short"][i] = r["u"], r["cost"], r["shed"], r["short"]
                pl["src"][i] = rnd
                acc += 1
            pl["visits"][i] += 1
        g = np.array(gains)
        return {"n_solved": len(res), "accepted_%": acc / max(len(res), 1) * 100, "milp_core_s": milp_s, "lp_core_s": lp_s,
                "milp_s_mean": milp_s / max(len(res), 1), "hit_time_limit_%": to / max(len(res), 1) * 100,
                "cost_reduction_mean_%": float(np.mean(np.maximum(g, 0))), "cost_reduction_median_%": float(np.median(np.maximum(g, 0))),
                "infeasible": int(np.sum(~np.isfinite(g)))}

    stages = STAGES if a.stage == "all" else a.stage.split(",")
    done = S.setdefault("done", [])

    # ======================================================================= r0
    if "r0" in stages and "r0" not in done:
        t_stage = time.time()
        y0 = adequate((tr["u_rel"] > 0.5).astype(np.int8), tr)
        va_t = dict(va, u_target=adequate((va["u_rel"] > 0.5).astype(np.int8), va))
        m_ = UCModel(sysm, T=T, network=True)
        t0 = time.time()
        for i in range(10):
            m_.solve_dispatch(scenario_from(tr_full, i), None, relax=True)
        lp_relax_s = (time.time() - t0) / 10
        t0 = time.time()
        c0, sh0, so0 = oracle().evaluate(tr, np.arange(N), y0, keys_tr)
        score_core_s = (time.time() - t0) * a.workers
        pl = dict(Y=y0.astype(np.int8), C=c0, shed=sh0, short=so0, src=np.zeros(N, np.int16), visits=np.zeros(N, np.int16),
                  Y0=y0.astype(np.int8), C0=c0.copy())
        pool_save(pl)
        S["label_cost"] = {"lp_relaxation_s_per_label": lp_relax_s, "lp_relaxation_core_s": lp_relax_s * N,
                           "score_lp_core_s_round0": score_core_s, "full_milp_label_core_s": float(tr_full["time"].sum()),
                           "full_milp_s_per_label": float(tr_full["time"].mean())}
        S["labels"] = {"round0": label_stats(pl)}
        print("label pool round 0:", S["labels"]["round0"], flush=True)
        m0 = new_model()
        t0 = time.time()
        train_uc_bce(m0, tr, va_t, y0, epochs=a.epochs, seed=a.seed)
        S["train_s"] = {"m0_bce": time.time() - t0}
        torch.save(m0.net.state_dict(), P("m0.pt"))
        val_eval(m0, "LF-BCE (round 0)")
        # control: the fixed-epoch trainer used in later rounds, on the round-0 labels
        m0f = new_model()
        train_bce_fixed(m0f, tr, y0, epochs=a.epochs, seed=a.seed)
        torch.save(m0f.net.state_dict(), P("m0_fixed.pt"))
        val_eval(m0f, "LF-BCE, fixed-epoch trainer (control)")
        m_lfrl = copy.deepcopy(m0)
        t0 = time.time()
        train_uc_reinforce(m_lfrl, tr, oracle(), steps=a.rl_steps, seed=a.seed, repair=rep)
        S["train_s"]["lf_rl"] = time.time() - t0
        torch.save(m_lfrl.net.state_dict(), P("lfrl.pt"))
        val_eval(m_lfrl, "LF + REINFORCE")
        S["stage_s_r0"] = time.time() - t_stage
        done.append("r0"); save_state()

    # ======================================================================= pilot
    perm = np.random.default_rng(a.seed + 1).permutation(N)
    chunks = np.array_split(perm, a.rounds)
    if "pilot" in stages and "pilot" not in done and not a.q:
        t_stage = time.time()
        pl = pool_load()
        m0 = load_model(P("m0.pt"))
        p = m0.predict(tr)
        pidx = chunks[0][:a.pilot_n]
        sp = SolverPool(cfg, a.workers)
        S["pilot"] = {}
        best = {}
        for q in (0.8, 0.9):
            jobs, released = label_jobs(p, pl, pidx, q, a.label_limit)
            res = sp.run(_reduced, jobs, log_every=10, tag=f"pilot q={q}")
            tmp = {k: v.copy() for k, v in pl.items()}
            st = absorb(tmp, res, 1)
            st.update(released_per_instance=released / len(pidx), labels_after=label_stats(tmp, pidx))
            S["pilot"][str(q)] = st
            best[q] = res
            print(f"  pilot q={q}: {st}", flush=True)
        sp.close()
        st0 = label_stats(pl, pidx)
        S["pilot"]["before"] = st0
        g8, g9 = (S["pilot"][k]["labels_after"]["gap_to_LP_bound_mean_%"] for k in ("0.8", "0.9"))
        # better label quality wins; within 0.1 pp the cheaper (faster) setting
        q_sel = 0.8 if g8 < g9 - 0.1 else 0.9
        S["q"] = q_sel
        S["pilot"]["chosen_q"] = q_sel
        # keep the best schedule found by either setting; both count as label cost
        for q in (0.8, 0.9):
            st = absorb(pl, best[q], 1)
        pl["visits"][pidx] = 1
        pool_save(pl)
        S["pilot_core_s"] = sum(S["pilot"][k]["milp_core_s"] + S["pilot"][k]["lp_core_s"] for k in ("0.8", "0.9"))
        S["pilot"]["labels_best_of_both"] = label_stats(pl, pidx)
        S["stage_s_pilot"] = time.time() - t_stage
        done.append("pilot"); save_state()
    if a.q:
        S["q"] = a.q

    # ======================================================================= rounds
    if "rounds" in stages and "rounds" not in done:
        q = S["q"]
        S.setdefault("rounds", [])
        model = load_model(P(f"m{len(S['rounds'])}.pt"))
        for r in range(len(S["rounds"]) + 1, a.rounds + 1):
            t_stage = time.time()
            pl = pool_load()
            p = model.predict(tr)
            idx = np.array([i for i in chunks[r - 1] if pl["visits"][i] == 0])
            sp = SolverPool(cfg, a.workers)
            jobs, released = label_jobs(p, pl, idx, q, a.label_limit)
            before = label_stats(pl, chunks[r - 1])
            res = sp.run(_reduced, jobs, log_every=25, tag=f"round {r} q={q}")
            sp.close()
            st = absorb(pl, res, r)
            pool_save(pl)
            st.update(round=r, q=q, released_per_instance=released / max(len(idx), 1), chunk_before=before,
                      chunk_after=label_stats(pl, chunks[r - 1]), pool_after=label_stats(pl))
            print(f"  round {r}: {json.dumps({k: v for k, v in st.items() if not isinstance(v, dict)}, default=float)}", flush=True)
            print(f"    chunk before {before}\n    chunk after  {st['chunk_after']}", flush=True)
            model = new_model()
            t0 = time.time()
            train_bce_fixed(model, tr, pl["Y"], epochs=a.epochs, seed=a.seed)
            st["train_s"] = time.time() - t0
            torch.save(model.net.state_dict(), P(f"m{r}.pt"))
            st["wall_s"] = time.time() - t_stage
            S["rounds"].append(st)
            save_state()
            val_eval(model, f"ST round {r}")
        done.append("rounds"); save_state()

    # ======================================================================= rl
    R = a.rounds
    if "rl" in stages and "rl" not in done:
        pl = pool_load()
        m_st = load_model(P(f"m{R}.pt"))
        S.setdefault("train_s", {})
        if not os.path.exists(P("strl.pt")):
            m_ = copy.deepcopy(m_st)
            t0 = time.time()
            train_uc_reinforce(m_, tr, oracle(), steps=a.rl_steps, seed=a.seed, repair=rep)
            S["train_s"]["st_rl"] = time.time() - t0
            torch.save(m_.net.state_dict(), P("strl.pt"))
            val_eval(m_, "ST + REINFORCE")
        if not os.path.exists(P("stsil.pt")):
            m_ = copy.deepcopy(m_st)
            t0 = time.time()
            train_reinforce_sil(m_, tr, oracle(), pl["Y"], pl["C"], steps=a.rl_steps, seed=a.seed, repair=rep)
            S["train_s"]["st_sil"] = time.time() - t0
            torch.save(m_.net.state_dict(), P("stsil.pt"))
            val_eval(m_, "ST + REINFORCE with solver labels (SIL)")
        S["rl_LPs_total"] = oracle().n
        done.append("rl"); save_state()

    # ======================================================================= models for test
    def models_for_test():
        ms = {"MILP-label BCE (uc_model1_4)": os.path.join(OUT, "uc_model1_4.pt"),
              "MILP-label + REINFORCE (uc_model1_rl)": os.path.join(OUT, "uc_model1_rl.pt"),
              "LF-BCE (round 0)": P("m0.pt"), "LF + REINFORCE": P("lfrl.pt")}
        for r in range(1, R + 1):
            ms[f"ST round {r}"] = P(f"m{r}.pt")
        ms["ST + REINFORCE"] = P("strl.pt")
        ms["ST + REINFORCE + SIL"] = P("stsil.pt")
        return {k: v for k, v in ms.items() if os.path.exists(v)}

    if "eval" in stages and "eval" not in done:
        n_te = len(te["load"])
        rows = S.setdefault("test_rows", [])
        have = {r["method"] for r in rows}
        milp_units = float(te["u"].sum((1, 2)).mean() / T)
        for name, path in models_for_test().items():
            m1 = load_model(path)
            p = m1.predict(te)
            decs = [("top-1", lambda: fixrep((p > 0.5).astype(np.int8), te["u0"])),
                    ("top-1 + adequacy repair", lambda: adequate((p > 0.5).astype(np.int8), te))]
            for lab, mk in decs:
                if f"{name}: {lab}" in have:
                    continue
                U = mk()
                c, sh, so = oracle().evaluate(te, np.arange(n_te), U, keys_te)
                rows.append(uc_metrics(c, sh, so, te, U, f"{name}: {lab}", LPs_per_instance=1.0, MILP_units_on=milp_units))
                print("  [test]", rows[-1]["method"], round(rows[-1]["no_shed_no_shortfall_%"], 1),
                      round(rows[-1]["gap_median_%"], 3), round(rows[-1]["gap_mean_served_%"], 3), flush=True)
                save_state()
            if f"{name}: screening" in have or name in ("ST round 1", "ST round 2"):
                continue
            cl = candidates_from_probs(p, 8, np.random.default_rng(a.seed))
            flat_i = np.concatenate([np.full(len(x), i) for i, x in enumerate(cl)])
            flat_u = fixrep(np.concatenate(cl), te["u0"][flat_i])
            c, sh, so = oracle().evaluate(te, flat_i, flat_u, keys_te[flat_i])
            best = np.array([np.where(flat_i == i)[0][np.argmin(c[flat_i == i])] for i in range(n_te)])
            rows.append(uc_metrics(c[best], sh[best], so[best], te, flat_u[best], f"{name}: screening",
                                   LPs_per_instance=float(np.mean([len(x) for x in cl])), MILP_units_on=milp_units))
            print("  [test]", rows[-1]["method"], round(rows[-1]["no_shed_no_shortfall_%"], 1),
                  round(rows[-1]["gap_median_%"], 3), round(rows[-1]["gap_mean_served_%"], 3), flush=True)
            save_state()
        done.append("eval"); save_state()

    # ======================================================================= fix
    if "fix" in stages and "fix" not in done:
        if "o" in _oracle:
            _oracle.pop("o").close()
        nf = min(a.n_fix, len(te["load"]))
        ms = models_for_test()
        probs = {k: load_model(v).predict(sub(te, np.arange(nf))) for k, v in ms.items()
                 if k in ("MILP-label BCE (uc_model1_4)", "MILP-label + REINFORCE (uc_model1_rl)", "LF + REINFORCE",
                          f"ST round {R}", "ST + REINFORCE", "ST + REINFORCE + SIL")}
        plan = [("MILP-label BCE (uc_model1_4)", "sym", (0.8, 0.9, 0.95)), ("MILP-label BCE (uc_model1_4)", "asym+guard", (0.9, 0.95)),
                ("MILP-label + REINFORCE (uc_model1_rl)", "sym", (0.9, 0.95)), ("LF + REINFORCE", "sym", (0.9, 0.95)),
                (f"ST round {R}", "sym", (0.8, 0.9, 0.95)), (f"ST round {R}", "asym+guard", (0.9, 0.95)),
                ("ST + REINFORCE", "sym", (0.9, 0.95)), ("ST + REINFORCE + SIL", "sym", (0.9, 0.95))]
        jobs = []
        for i in range(nf):
            specs = [("full MILP", np.zeros((0, 3), np.int64))]
            for mname, rule, ratios in plan:
                if mname not in probs:
                    continue
                pi = probs[mname][i]
                for ratio in ratios:
                    tt, gg = choose_fixings(pi, ratio, 10.0 if rule == "asym+guard" else 1.0)
                    fix = {(int(t_), int(g_)): int(pi[t_, g_] > 0.5) for t_, g_ in zip(tt, gg)}
                    if rule == "asym+guard":
                        fix, _ = adequacy_guard(fix, te["load"][i], te["avail"][i], te["sr"][i], sysm)
                    specs.append((f"{int(round(ratio * 100))} %: {mname}, {rule}", fix_array(fix)))
            if a.fix_max_specs:
                specs = specs[:a.fix_max_specs + 1]
            jobs.append((i, te["load"][i], te["avail"][i], te["u0"][i], te["sr"][i], specs, cfg["time_limit"], cfg["mip_gap"]))
        t0 = time.time()
        sp = SolverPool(cfg, a.workers)
        res = dict(sp.run(_fixeval, jobs, log_every=5, tag="fix"))
        sp.close()
        names = [s[0] for s in jobs[0][5]]
        subte = sub(te, np.arange(nf))
        frows, per = [], {}
        t_full = np.array([res[i]["full MILP"]["time"] for i in range(nf)])
        for nm in names:
            g = lambda k: np.array([res[i][nm][k] for i in range(nf)])
            U = np.array([res[i][nm]["u"] for i in range(nf)])
            t_ = g("time")
            r = uc_metrics(g("obj"), g("shed"), g("short"), subte, U, nm, fixed_share_=float(g("n_fixed").mean() / (T * sysm.G) * 100),
                           time_s=float(t_.mean()), speedup_x=float(t_full.mean() / t_.mean()),
                           speedup_median_x=float(np.median(t_full / t_)), fallback_=float(g("fallback").mean() * 100),
                           hit_time_limit_=float(np.mean([res[i][nm]["status"] != "optimal" for i in range(nf)]) * 100))
            frows.append(r)
            per[nm] = {"obj": g("obj").tolist(), "time": t_.tolist(), "shed": g("shed").tolist(), "short": g("short").tolist()}
        S["fix_rows"] = frows
        S["fix_wall_s"] = time.time() - t0
        with open(P("fix_per_instance.json"), "w") as f:
            json.dump(per, f, default=float)
        done.append("fix"); save_state()
        cols = ["method", "fixed_share_", "gap_mean_%", "gap_median_%", "no_shed_no_shortfall_%", "matches_or_beats_milp_%",
                "time_s", "speedup_x", "units_on", "fallback_", "hit_time_limit_"]
        print(fmt_table(frows, cols), flush=True)

    if "o" in _oracle:
        _oracle["o"].close()
    print("stages done:", S.get("done"), flush=True)
