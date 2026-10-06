"""Solver-side alternatives to hard fixing: one process, one core, every variant of an instance back to back.

    taskset -c 0 python3 scripts/uc_solver_eval.py --bench uc12 --split val --n 40 --variants val12 --out <jsonl>
    taskset -c 0 python3 scripts/uc_solver_eval.py --bench uc12 --split test_fresh --n 60 --variants test12 --out <jsonl>
    taskset -c 0 python3 scripts/uc_solver_eval.py --bench uc24 --split test --n 40 --variants test24 --out <jsonl>

Per instance, in order: inference (GNN forward pass, error-cost features + ensemble, kNN; timed), the LP relaxation
that feeds the GNN (timed), the full MILP (cold, incumbent trace; the reference for time, dual bound and time to
quality), then every variant. All MILPs go through otsl.solver.solve_uc_hs (highspy, 1 thread, 0.1 % gap, the
benchmark's time limit unless a variant sets its own). Records: one JSON line per instance with one entry per variant;
the script resumes after the last completed instance.

Variant names (see otsl/solver.py and docs/methods/solver.md):
  full                     full MILP, cold start
  ref:<rule>               hard-fixing reference rules re-run here (uc12: ltf_knn, ltf_bce, hybrid, ec90, comb98;
                           uc24: ltf_knn, g95)
  hard:<q0>:<q1>           error-cost (uc12) / confidence (uc24) ranking within each predicted class, q0 of the OFF and
                           q1 of the ON predictions, adequacy guard + min up/down row release, hard-fixed
  pas:<q0>:<q1>:<D>[:tl]   the same set as a Predict-and-Search trust region (at most D deviations), no hard fixing
  core:<base>:<q0>:<q1>:<D>  base rule's fixings hard, trust region over the extra decisions of hard:<q0>:<q1>
  warmfull:dec             full MILP warm-started from the decoded schedule (threshold, block adequacy repair, min
                           up/down repair, dispatch LP)
  warmred:<base>           base rule's reduced MILP warm-started from the decoded schedule made consistent with its fixings
  ftp:<base>[:tl]          full MILP warm-started from the base rule's reduced solution (fix, then prove / polish)
  lb:<base>:<r>[:tl]       local branching: full MILP + Hamming ball of radius r around the base incumbent, warm start
  grad:<base>:<m>[:tl]     gradient release: the m fixings with the largest first-order saving at the base incumbent
                           released, reduced MILP re-solved from the incumbent
  rins:<base>[:tl]         RINS neighbourhood: every decision where the base incumbent equals the full LP relaxation
                           (computed anyway as GNN input) fixed, the rest free, warm start from the incumbent
  lpfix                    no-learning reference: the decisions the LP relaxation sets integrally fixed (+ row release)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import adequacy_guard, block_decode, harm_from_file, harm_scores, rule_fixings  # noqa: E402
from otsl.fixpolicy import FixFeaturizer, lp_guard, release_conflicting_rows  # noqa: E402
from otsl.hybrid import apply_guards, harm_to_score  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.solver import (dispatch_x, flip_savings, gradient_release, local_branching_row, masks_from_fix,  # noqa: E402
                         pick_confident, schedule_for_fixings, solve_uc_hs, trust_region_row)
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

BENCH = {"uc12": dict(T=12, tl=60.0, gap=1e-3, root="data/generated/uc12", out="results/uc12", th_dec=0.5),
         "uc24": dict(T=24, tl=300.0, gap=1e-3, root="data/generated/uc24", out="results/uc24", th_dec=0.3)}
SPLIT_KEY12 = {"val": "va", "val_extra": "vx", "val_extra2": "vx2", "test_fresh": "tf"}

VARIANT_SETS = {
    # validation (uc12): only variants with parameters to tune; anytime runs are cut post hoc at tau <= their limit
    "val12": ["full", "ref:hybrid", "ref:ec90", "ref:comb98",
              "ftp:hybrid:10", "lb:hybrid:10:10", "lb:hybrid:30:10", "grad:hybrid:20:10", "grad:hybrid:60:10",
              "ftp:comb98:10", "lb:comb98:10:10", "lb:comb98:30:10", "grad:comb98:20:10", "grad:comb98:60:10",
              "hard:0.97:0.9", "core:hybrid:0.97:0.9:10", "core:hybrid:1:1:20",
              "pas:0.97:0.9:10:30", "pas:1:1:20:30"],
    # supplementary validation pass on the same instances (merged by instance index in the report)
    "val12b": ["ref:hybrid", "ref:comb98", "grad:hybrid:120:10", "grad:comb98:120:10", "rins:hybrid:10",
               "rins:comb98:10", "lpfix"],
    # third validation pass: warm starts (to choose time limits for warm-started anytime variants)
    "val12c": ["warmfull:dec", "warmred:hybrid", "warmred:ec90", "warmred:comb98"],
    # uc12 test (fixed on validation, docs/methods/solver.md "uc12 selection")
    "test12": ["full", "ref:ltf_knn", "ref:ltf_bce", "ref:hybrid", "ref:ec90", "ref:comb98", "lpfix",
               "warmfull:dec", "warmred:hybrid", "warmred:ec90", "warmred:comb98", "ftp:hybrid",
               "rins:hybrid:10", "lb:hybrid:10:10", "rins:comb98:10", "grad:comb98:60:10", "lb:comb98:10:10",
               "hard:0.97:0.9", "core:hybrid:0.97:0.9:10", "pas:0.97:0.9:10:30"],
    # uc24 validation (uc24ltf_val; the dataset's full MILP is the reference, not re-solved)
    "val24": ["fullstored", "ref:g95", "warmred:g95", "rins:g95:60", "grad:g95:100:60", "grad:g95:250:60",
              "hard:0.97:0.9", "pas:0.97:0.9:20:150"],
}


# ============================================================================ benchmark context
class Bench12:
    def __init__(self, split, need_knn):
        from otsl.ucdata import load, scenario_from
        from uc_constrained import load_model1, strip
        cfg = BENCH["uc12"]
        self.cfg, self.T = cfg, 12
        self.set_guards = ("adeq", "rows")              # guards of the hybrid rule
        self.s = load_rts_gmlc()
        self.m = UCModel(self.s, T=12, network=True)
        tr = load(os.path.join(cfg["root"], "train.npz"))
        self.d = load(os.path.join(cfg["root"], f"{split}.npz"))
        self.scen = lambda i: scenario_from(self.d, i)
        out = cfg["out"]
        trs = strip(tr)
        self.gnn = {"bce": load_model1(os.path.join(out, "uc_model1_4.pt"), self.s, trs, 12),
                    "st": load_model1(os.path.join(out, "combo_st_s0.pt"), self.s, trs, 12)}
        h0 = harm_from_file(os.path.join(out, "combo_harm_s0.pt"))   # harm ensemble seed 0 for both sources
        self.harm = {"bce": h0, "st": h0}                              # (as scripts/uc_hybrid_eval.py)
        self.ff = FixFeaturizer(self.s, 12)
        self.knn = KNNProb(self.s, tr, k=50) if need_knn else None
        self.norm = json.load(open(os.path.join(out, "hybrid_probs.json")))["harm_norm_logh_mean_sd"]["bce_s0"]
        th = lambda f: json.load(open(os.path.join(out, f)))
        self.th = {"ltf_knn": th("ltfx_tune_knn_1.json"), "ltf_bce": th("ltfx_tune_bce_1.json"),
                   "hybrid": th("hybrid_tune_he_bce_s0_e1_n360.json")}
        assert tuple(self.th["hybrid"]["guards"]) == ("adeq", "rows")
        self.chk = np.load(os.path.join(out, "hybrid_probs.npz"))
        self.key = SPLIT_KEY12.get(split)
        N = len(self.d["load"])
        self.one = lambda i: {k: v[i:i + 1] for k, v in self.d.items()
                              if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == N}
        i0 = self.one(0)                                   # warm-up (first call allocations)
        for k in self.gnn:
            self.gnn[k].predict(i0)

    def n(self):
        return len(self.d["load"])

    def infer(self, i):
        """probabilities, error costs and their wall times for instance i"""
        di = self.one(i)
        P, H, Tm = {}, {}, {}
        for k, g in self.gnn.items():
            t0 = time.time(); P[k] = g.predict(di)[0].astype(np.float64); Tm[k] = time.time() - t0
            t0 = time.time(); H[k] = harm_scores(self.harm[k], self.ff, P[k][None], self.d, [i])[0]
            Tm["harm_" + k] = time.time() - t0
        if self.knn is not None:
            t0 = time.time(); P["knn"] = self.knn.predict(di)[0]; Tm["knn"] = time.time() - t0
        if self.key is not None:                           # same model outputs as the stored ones
            s = f"_{self.key}"
            assert np.abs(P["bce"] - self.chk["bce_s0" + s][i]).max() < 1e-5
            assert np.abs(H["bce"] / self.chk["harm_bce_s0" + s][i] - 1).max() < 1e-3
            assert np.abs(P["st"] - self.chk["st_s0" + s][i]).max() < 1e-5
        return P, H, Tm

    def score(self, P, H):
        """ranking score for hard / pas variants: the learned expected error cost of the BCE GNN"""
        return H["bce"], (P["bce"] > 0.5).astype(int)

    def ref_fix(self, rule, sc, P, H, Tm):
        """(fixings before solver-free guards were applied?, pre-solve seconds, uses GNN) for a reference rule.
        Returns (fix, pre_s, guards_to_apply)."""
        t0 = time.time()
        if rule == "ltf_knn":
            r = self.th["ltf_knn"]
            fix = fix_dict(P["knn"], np.array(r["lo"]), np.array(r["hi"]))
            return fix, Tm["knn"] + time.time() - t0, (), False
        if rule == "ltf_bce":
            r = self.th["ltf_bce"]
            fix = fix_dict(P["bce"], np.array(r["lo"]), np.array(r["hi"]))
            return fix, Tm["bce"] + time.time() - t0, (), True
        if rule == "hybrid":
            r = self.th["hybrid"]
            sco = harm_to_score(H["bce"], P["bce"], *self.norm)
            fix = fix_dict(sco, np.array(r["lo"]), np.array(r["hi"]))
            return fix, Tm["bce"] + Tm["harm_bce"] + time.time() - t0, ("adeq", "rows"), True
        if rule == "ec90":
            fix = rule_fixings("harm", 0.90, P["bce"], sc, self.s, H["bce"])
            return fix, Tm["bce"] + Tm["harm_bce"] + time.time() - t0, (), True
        if rule == "comb98":
            fix = rule_fixings("harm", 0.98, P["st"], sc, self.s, H["st"])
            return fix, Tm["st"] + Tm["harm_st"] + time.time() - t0, ("rows", "lp"), True
        raise ValueError(rule)

    def dec_probs(self, P):
        return P["bce"]


class Bench24:
    def __init__(self, split, need_knn):
        from otsl.b3 import load_b3
        from otsl.ucml import UCFeaturizer, build_uc_model1
        from uc_b3_train import strip
        cfg = BENCH["uc24"]
        self.cfg, self.T = cfg, 24
        self.set_guards = ("adeq", "lp", "rows")        # guards of the uc24 guarded rule (scripts/uc_b3_fix.py)
        self.s = load_rts_gmlc()
        self.m = UCModel(self.s, T=24, network=True)
        f = {"test": "test.npz", "val": "uc24ltf_val.npz"}[split]
        self.d = load_b3(os.path.join(cfg["root"], f))
        self.scen = lambda i: UCScenario(load=self.d["load"][i], avail=self.d["avail"][i], u0=self.d["u0"][i],
                                         sr=self.d["sr"][i])
        out = cfg["out"]
        tr = strip(load_b3(os.path.join(cfg["root"], "train.npz")))
        feat = UCFeaturizer(self.s, tr, relax=True, sym=True)
        self.gnn = {}
        for k, fn in (("bce", "b3_lf_bce.pt"), ("rl", "b3_rl_selected.pt")):
            g = build_uc_model1(self.s, feat, 24, "gnn", seed=0)
            g.net.load_state_dict(torch.load(os.path.join(out, fn)))
            self.gnn[k] = g
        if need_knn:
            from uc_uc24ltf_prep import load_train_lab
            self.knn = KNNProb(self.s, load_train_lab(os.path.join(cfg["root"], "uc24ltf_train_lab.npz")), k=50)
        else:
            self.knn = None
        self.th = {"ltf_knn": json.load(open(os.path.join(out, "uc24ltf_tune_knn_1.json")))}
        self.chk = np.load(os.path.join(out, "uc24ltf_probs.npz"))
        self.key = {"test": "te", "val": "va"}[split]
        N = len(self.d["load"])
        self.one = lambda i: {k: v[i:i + 1] for k, v in self.d.items()
                              if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == N}
        for g in self.gnn.values():
            g.predict(self.one(0))

    def n(self):
        return len(self.d["load"])

    def infer(self, i):
        di = self.one(i)
        P, Tm = {}, {}
        for k, g in self.gnn.items():
            t0 = time.time(); P[k] = g.predict(di)[0].astype(np.float64); Tm[k] = time.time() - t0
        if self.knn is not None:
            t0 = time.time(); P["knn"] = self.knn.predict(di)[0]; Tm["knn"] = time.time() - t0
        for k in P:
            assert np.abs(P[k] - self.chk[f"{k}_{self.key}"][i]).max() < 1e-5, k
        return P, {}, Tm

    def score(self, P, H):
        """ranking score for hard / pas variants on uc24: error probability of the imitation GNN (no MILP labels,
        so no error-cost model)"""
        return np.minimum(P["bce"], 1 - P["bce"]), (P["bce"] > 0.5).astype(int)

    def ref_fix(self, rule, sc, P, H, Tm):
        t0 = time.time()
        if rule == "ltf_knn":
            r = self.th["ltf_knn"]
            fix = fix_dict(P["knn"], np.array(r["lo"]), np.array(r["hi"]))
            return fix, Tm["knn"] + time.time() - t0, (), False
        if rule == "g95":            # scripts/uc_b3_fix.py "95%|bce_g": confidence ranking, adequacy guard, then
            from uc_fixing import choose_fixings      # LP-relaxation guard and min up/down conflict release
            p = P["bce"]
            tt, gg = choose_fixings(p, 0.95, 1.0)
            fix = {(int(t), int(g)): int(p[t, g] > 0.5) for t, g in zip(tt, gg)}
            fix, _ = adequacy_guard(fix, sc.load, sc.avail, sc.sr, self.s)
            return fix, Tm["bce"] + time.time() - t0, ("lp", "rows"), True
        raise ValueError(rule)

    def dec_probs(self, P):
        return P["bce"]


# ============================================================================ helpers
def rec_of(res, **extra):
    ok = res["x"] is not None
    out = dict(feasible=ok, obj=float(res["obj"]) if ok else None, bound=float(res["bound"]), status=res["status"],
               gap=float(res["gap"]) if ok else None, t_build=float(res["t_build"]), t_run=float(res["time"]),
               cpu=float(res["cpu"]), nodes=int(res["nodes"]), inc=res["inc"], start_obj=res["start_obj"],
               start_accepted=res["start_accepted"], shed=res.get("shed"), short=res.get("short"))
    out.update(extra)
    return out


def guard(B, sc, fix, guards):
    """solver-free and LP guards in the given order (names: adeq, rows, lp); returns (fix, seconds, info)"""
    t0 = time.time()
    info = {}
    fix = dict(fix)
    for gname in guards:
        if gname == "adeq":
            fix, info["released_adeq"] = adequacy_guard(fix, sc.load, sc.avail, sc.sr, B.s)
        elif gname == "rows":
            fix, info["released_rows"] = release_conflicting_rows(fix, B.s, sc.u0)
        elif gname == "lp":
            fix, info["released_lp"], _ = lp_guard(B.m, B.s, sc, fix)
    return fix, time.time() - t0, info


def ranked_set(B, sc, P, H, q0, q1):
    """hard / pas set: q0 / q1 shares of the predicted OFF / ON decisions with the lowest score, + the benchmark's guard
    chain (uc12: adequacy guard, min up/down row release; uc24: adequacy, LP-relaxation guard, row release); returns
    (fix, seconds, guard info)"""
    t0 = time.time()
    sco, yhat = B.score(P, H)
    S0, S1 = pick_confident(sco, yhat, q0, q1)
    fix = {(int(t), int(g)): 0 for t, g in zip(*np.where(S0))}
    fix.update({(int(t), int(g)): 1 for t, g in zip(*np.where(S1))})
    fix, _, info = guard(B, sc, fix, B.set_guards)
    return fix, time.time() - t0, info


def run_instance(B, i, variants, log):
    cfg, T, G = B.cfg, B.T, B.s.G
    TL, GAP = cfg["tl"], cfg["gap"]
    sc = B.scen(i)
    out = {"i": int(i), "loadavg0": os.getloadavg()[0]}
    P, H, Tm = B.infer(i)
    out["t_inf"] = {k: float(v) for k, v in Tm.items()}
    t0 = time.time()
    rel = B.m.solve_dispatch(sc, None, relax=True)    # LP relaxation: input of the GNN features (timed)
    t_rel = time.time() - t0
    u_rel = np.asarray(rel.u, float)
    out["t_rel"] = t_rel
    base = {}                                          # rule -> (fix, pre_s incl. t_rel, result)
    dec = {}

    def decoded():
        if "u" not in dec:
            t0 = time.time()
            p = B.dec_probs(P)
            dd = {k: B.d[k][i:i + 1] for k in ("u0", "load", "avail", "sr")}
            dec["u"] = block_decode(p[None], dd, B.s, cfg["th_dec"])[0]
            dec["t"] = time.time() - t0
        return dec["u"], dec["t"]

    def get_base(rule):
        if rule not in base:
            fix, pre, guards, gnn = B.ref_fix(rule, sc, P, H, Tm)
            n_pre = len(fix)
            fix, t_g, ginfo = guard(B, sc, fix, guards)
            res = solve_uc_hs(B.m, sc, TL, GAP, z_fix=fix or None)
            pre_s = pre + t_g + (t_rel if gnn else 0.0)
            base[rule] = (fix, pre_s, res)
            out[f"ref:{rule}"] = rec_of(res, pre_s=pre_s, n_fixed=len(fix), n_fixed_pre=n_pre,
                                        n_off=sum(1 for v in fix.values() if v == 0), **ginfo)
        return base[rule]

    for v in variants:
        if v in out:
            continue
        p = v.split(":")
        kind = p[0]
        t_v = time.time()
        if kind == "full":
            res = solve_uc_hs(B.m, sc, TL, GAP)
            out["full"] = rec_of(res, pre_s=0.0, loadavg=os.getloadavg()[0])
        elif kind == "fullstored":       # validation only: the dataset's full MILP (time, bound, incumbent log) as the
            dd = B.d                     # reference, without re-solving it (uc24 validation, to save solver time)
            inc = [(float(t), float(o), float("nan")) for t, o in zip(dd["inc_t"][i], dd["inc_obj"][i]) if np.isfinite(t)]
            out["full"] = dict(feasible=True, obj=float(dd["obj"][i]), bound=float(dd["bound"][i]),
                               status="optimal" if dd["opt"][i] else "time_limit", gap=float(dd["gap"][i]), t_build=0.0,
                               t_run=float(dd["time"][i]), cpu=float(dd["cpu"][i]), nodes=int(dd["nodes"][i]), inc=inc,
                               start_obj=None, start_accepted=None, shed=float(dd["shed"][i]), short=float(dd["short"][i]),
                               pre_s=0.0, stored=True)
            out["fullstored"] = {"note": "full = dataset run"}
        elif kind == "ref":
            get_base(p[1])
        elif kind in ("hard", "pas"):
            q0, q1 = float(p[1]), float(p[2])
            fix, t_set, ginfo = ranked_set(B, sc, P, H, q0, q1)
            pre_s = t_rel + Tm["bce"] + Tm.get("harm_bce", 0.0) + t_set
            if kind == "hard":
                res = solve_uc_hs(B.m, sc, TL, GAP, z_fix=fix or None)
                out[v] = rec_of(res, pre_s=pre_s, n_fixed=len(fix), **ginfo)
            else:
                D = float(p[3])
                tl = float(p[4]) if len(p) > 4 else TL
                off, on = masks_from_fix(fix, T, G)
                res = solve_uc_hs(B.m, sc, tl, GAP, rows=[trust_region_row(B.m, off, on, D)])
                out[v] = rec_of(res, pre_s=pre_s, n_tr=len(fix), delta=D, tl=tl, **ginfo)
        elif kind == "core":
            fixb, preb, _ = get_base(p[1])
            q0, q1, D = float(p[2]), float(p[3]), float(p[4])
            fix, t_set, ginfo = ranked_set(B, sc, P, H, q0, q1)
            band = {k: val for k, val in fix.items() if k not in fixb}
            off, on = masks_from_fix(band, T, G)
            res = solve_uc_hs(B.m, sc, TL, GAP, z_fix=fixb or None, rows=[trust_region_row(B.m, off, on, D)])
            out[v] = rec_of(res, pre_s=preb + t_set, n_fixed=len(fixb), n_band=len(band), delta=D)
        elif kind == "warmfull":
            u, t_dec = decoded()
            x, c, t_lp = dispatch_x(B.m, sc, u)
            pre_s = t_rel + Tm["bce"] + t_dec + t_lp
            res = solve_uc_hs(B.m, sc, TL, GAP, start_x=x)
            out[v] = rec_of(res, pre_s=pre_s, dec_obj=c)
        elif kind == "warmred":
            fixb, preb, resb = get_base(p[1])
            u, t_dec = decoded()
            t0 = time.time()
            uf = schedule_for_fixings(u, fixb, B.s, sc.u0)
            t_cons = time.time() - t0
            if uf is None:
                out[v] = dict(feasible=False, note="no schedule consistent with the fixings")
                continue
            x, c, t_lp = dispatch_x(B.m, sc, uf)
            # the decode + LP are extra work on top of the base rule's pre-solve time
            res = solve_uc_hs(B.m, sc, TL, GAP, z_fix=fixb or None, start_x=x)
            out[v] = rec_of(res, pre_s=preb + t_dec + t_cons + t_lp, dec_obj=c, n_fixed=len(fixb))
        elif kind == "lpfix":            # no-learning reference: fix every decision the LP relaxation sets integrally
            t0 = time.time()
            integral = np.abs(u_rel - np.round(u_rel)) < 1e-6
            fix = {(int(t), int(g)): int(round(u_rel[t, g])) for t, g in zip(*np.where(integral))}
            n_pre = len(fix)
            fix, t_g, ginfo = guard(B, sc, fix, ("rows",))
            pre_s = t_rel + time.time() - t0
            res = solve_uc_hs(B.m, sc, TL, GAP, z_fix=fix or None)
            out[v] = rec_of(res, pre_s=pre_s, n_fixed=len(fix), n_fixed_pre=n_pre, **ginfo)
        elif kind in ("ftp", "lb", "grad", "rins"):
            fixb, preb, resb = get_base(p[1])
            if resb["x"] is None:
                out[v] = dict(feasible=False, note="base infeasible")
                continue
            base_time = preb + resb["t_build"] + resb["time"]
            if kind == "ftp":
                tl = float(p[2]) if len(p) > 2 else TL
                res = solve_uc_hs(B.m, sc, tl, GAP, start_x=resb["x"])
                out[v] = rec_of(res, pre_s=base_time, base_obj=resb["obj"], tl=tl)
            elif kind == "lb":
                r_, tl = float(p[2]), (float(p[3]) if len(p) > 3 else TL)
                res = solve_uc_hs(B.m, sc, tl, GAP, start_x=resb["x"], rows=[local_branching_row(B.m, resb["u"], r_)])
                out[v] = rec_of(res, pre_s=base_time, base_obj=resb["obj"], tl=tl, radius=r_)
            elif kind == "rins":         # RINS neighbourhood: keep the base incumbent where it equals the LP relaxation
                tl = float(p[2]) if len(p) > 2 else TL
                t0 = time.time()
                ub_ = resb["u"].astype(float)
                agree = np.abs(u_rel - ub_) < 1e-6
                fix2 = {(int(t), int(g)): int(ub_[t, g]) for t, g in zip(*np.where(agree))}
                t_r = time.time() - t0
                res = solve_uc_hs(B.m, sc, tl, GAP, z_fix=fix2 or None, start_x=resb["x"])
                out[v] = rec_of(res, pre_s=base_time + t_r, base_obj=resb["obj"], tl=tl, n_fixed=len(fix2),
                                n_released=int((~agree).sum()))
            else:
                mrel, tl = int(p[2]), (float(p[3]) if len(p) > 3 else TL)
                sav, t_g = flip_savings(B.m, sc, resb["u"])
                t0 = time.time()
                fix2, rel = gradient_release(fixb, sav, mrel)
                t_r = time.time() - t0
                res = solve_uc_hs(B.m, sc, tl, GAP, z_fix=fix2 or None, start_x=resb["x"])
                out[v] = rec_of(res, pre_s=base_time + t_g + t_r, base_obj=resb["obj"], tl=tl, n_released=len(rel),
                                n_fixed=len(fix2), t_grad=t_g)
        else:
            raise ValueError(v)
        out.setdefault(v, {})["wall_variant"] = time.time() - t_v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=list(BENCH))
    ap.add_argument("--split", default="val")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--idx", default="", help="explicit comma-separated instance indices (overrides start / n)")
    ap.add_argument("--variants", default="val12", help="a VARIANT_SETS key or a comma-separated list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    variants = VARIANT_SETS.get(a.variants) or [v for v in a.variants.split(",") if v]
    need_knn = any(v.startswith("ref:ltf_knn") for v in variants)
    B = (Bench12 if a.bench == "uc12" else Bench24)(a.split, need_knn)
    idx = [int(x) for x in a.idx.split(",")] if a.idx else list(range(a.start, min(a.start + a.n, B.n())))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["i"] for line in open(a.out) if line.strip()}
    todo = [i for i in idx if i not in done]
    print(f"[solver-eval] {a.bench} {a.split}: {len(todo)} of {len(idx)} instances to do; {len(variants)} variants: "
          f"{variants}", flush=True)
    t0 = time.time()
    for k, i in enumerate(todo):
        r = run_instance(B, i, variants, print)
        r["variants"] = variants
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        full = r.get("full", {})
        db = full.get("bound", np.nan)
        msg = " | ".join(f"{v}: " + (f"{(r[v]['obj'] - db) / r[v]['obj'] * 100:.2f}% "
                                     f"{r[v].get('pre_s', 0) + r[v]['t_build'] + r[v]['t_run']:.1f}s"
                                     if r[v].get("feasible") else "INF") for v in variants if v != "full")
        print(f"[{k + 1}/{len(todo)} {time.time() - t0:.0f}s] i={i} full {full.get('t_run', np.nan):.1f}s "
              f"load {r['loadavg0']:.1f} | {msg}", flush=True)
    print("done", flush=True)
