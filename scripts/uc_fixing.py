"""Unit commitment: which predicted decisions should be fixed before the MILP? (RACLearn-style partial fixing)

    python scripts/uc_fixing.py --cfg uc1 --n_fix 200

Uses the Model 1 weights saved by uc_model1.py (LP-feature GNN, BCE; and its REINFORCE fine-tune).

* symmetric   RACLearn: fix the X % (unit, hour) decisions with the largest |p - 0.5|.
* asymmetric  a wrong OFF fix can force load shedding (VOLL); a wrong ON fix costs at most the unit's
              no-load cost. Rank by error probability min(p, 1-p), multiplied by kappa for OFF decisions.
* guard       after choosing the fixings, per hour: if the units NOT fixed off cannot cover net load +
              reserve (+ margin), unfix OFF-fixed units in merit order until they can (no solver needed).
* REINFORCE   rank by the cost-aware model's probabilities instead.

The full MILP is re-solved in the same process so speed-ups are measured under the same CPU load.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, uc_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model1 import fmt_table  # noqa: E402
from uc_gen import UC_CONFIGS  # noqa: E402


def choose_fixings(p, ratio, kappa=1.0):
    """indices (t, g) of the ratio*T*G decisions with the smallest risk-weighted error probability"""
    err = np.minimum(p, 1 - p) * np.where(p < 0.5, kappa, 1.0)
    flat = np.argsort(err.reshape(-1), kind="stable")[:int(ratio * p.size)]
    return np.unravel_index(flat, p.shape)


def adequacy_guard(fix, load_, avail, sr, sysm, margin=0.05):
    """unfix OFF-fixed units (cheapest first) until, every hour, the units not fixed off cover
    net load + reserve with a margin. Returns the guarded fixings and the number of units released."""
    avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    order = np.argsort(avg)
    released = 0
    for t in range(load_.shape[0]):
        need = (1 + margin) * (load_[t].sum() - avail[t].sum() + sr[t])
        off = {g for (tt, g), v in fix.items() if tt == t and v == 0}
        cap = sum(sysm.pmax[g] for g in range(sysm.G) if g not in off)
        for g in order:
            if cap >= need:
                break
            if g in off:
                del fix[(t, int(g))]
                cap += sysm.pmax[g]
                released += 1
    return fix, released


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="uc1")
    ap.add_argument("--n_fix", type=int, default=200)
    ap.add_argument("--ratios", default="0.8,0.9,0.95,0.98")
    ap.add_argument("--kappa", type=float, default=10.0)
    a = ap.parse_args()
    cfg = UC_CONFIGS[a.cfg]
    T = cfg["T"]
    root = os.path.join("data", "generated", a.cfg)
    tr, te = load(os.path.join(root, "train.npz")), load(os.path.join(root, "test.npz"))
    sysm = load_rts_gmlc()
    out_dir = os.path.join("results", a.cfg)
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    probs = {}
    for tag, path in [("BCE", "uc_model1_4.pt"), ("REINFORCE", "uc_model1_rl.pt")]:
        m1 = build_uc_model1(sysm, feat, T, "gnn", seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(out_dir, path)))
        probs[tag] = m1.predict(te)
    nf = min(a.n_fix, len(te["load"]))
    sub = {k: v[:nf] for k, v in te.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(te["load"])}
    m = UCModel(sysm, T=T, network=cfg.get("network", True))

    def solve_all(label, make_fix):
        costs, times, sheds, shorts, us, nfixed, rel, fallback = [], [], [], [], [], [], [], 0
        for i in range(nf):
            sc = scenario_from(te, i)
            fix, r_ = make_fix(i, sc)
            sol = m.solve_uc(sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"], z_fix=fix or None)
            t_ = sol.time
            if sol.u is None:                       # fixings conflict with min up/down: solve without them
                fallback += 1
                sol = m.solve_uc(sc, time_limit=cfg["time_limit"], mip_gap=cfg["mip_gap"])
                t_ += sol.time
            costs.append(sol.obj); times.append(t_); sheds.append(sol.shed); shorts.append(sol.short); us.append(sol.u)
            nfixed.append(len(fix) / (T * sysm.G)); rel.append(r_)
        r = uc_metrics(np.array(costs), np.array(sheds), np.array(shorts), sub, np.array(us), label,
                       fixed_share_=float(np.mean(nfixed) * 100), released_per_instance=float(np.mean(rel)),
                       fallback_=float(fallback / nf * 100), time_s=float(np.mean(times)))
        print(f"  {label:52s} fixed {r['fixed_share_']:5.1f}%  gap mean {r['gap_mean_%']:8.4f}%  "
              f"no-shed {r['no_shed_no_shortfall_%']:5.1f}%  match {r['matches_or_beats_milp_%']:5.1f}%  "
              f"time {r['time_s']:.3f}s", flush=True)
        return r

    rows = [solve_all("full MILP (same run, for timing)", lambda i, sc: ({}, 0))]
    t_full = rows[0]["time_s"]
    for ratio in [float(x) for x in a.ratios.split(",")]:
        for tag, kappa, guard in [("BCE", 1.0, False), ("BCE", a.kappa, False), ("BCE", 1.0, True),
                                  ("BCE", a.kappa, True), ("REINFORCE", 1.0, False), ("REINFORCE", a.kappa, True)]:
            def make_fix(i, sc, tag=tag, kappa=kappa, guard=guard):
                p = probs[tag][i]
                tt, gg = choose_fixings(p, ratio, kappa)
                fix = {(int(t_), int(g_)): int(p[t_, g_] > 0.5) for t_, g_ in zip(tt, gg)}
                return adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm) if guard else (fix, 0)
            name = (f"{int(ratio * 100)} %: {tag}, " + ("symmetric" if kappa == 1 else f"asymmetric k={kappa:g}")
                    + (" + adequacy guard" if guard else ""))
            rows.append(solve_all(name, make_fix))
            rows[-1]["speedup_x"] = t_full / rows[-1]["time_s"]
    cols = ["method", "fixed_share_", "gap_mean_%", "gap_median_%", "no_shed_no_shortfall_%", "matches_or_beats_milp_%",
            "time_s", "speedup_x", "released_per_instance", "fallback_"]
    md = fmt_table(rows, cols)
    print(md)
    with open(os.path.join(out_dir, "uc_fixing_results.md"), "w") as f:
        f.write(f"{nf} test instances, {a.cfg}\n\n" + md)
    with open(os.path.join(out_dir, "uc_fixing_results.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=1, default=float)
