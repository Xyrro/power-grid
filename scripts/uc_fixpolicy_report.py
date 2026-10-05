"""Summarise the fixing-policy evaluation: table per (rule, share), Pareto frontier (mean gap vs speed-up),
best gap attainable at equal speed-up, plot.

    python scripts/uc_fixpolicy_report.py

Reads results/uc12/fixpolicy_eval_{test,test80,val}.jsonl (pass A: 90/95/97 %, pass B: 80 %, val selection); writes results/uc12/fixpolicy_results.{md,json} and
results/uc12/fixpolicy_pareto.png.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otsl.ucdata import load  # noqa: E402
from uc_fixpolicy_eval import summarize  # noqa: E402

NAMES = {"full": "full MILP (same run)", "rac": "RACLearn (confidence margin)", "ltf": "Learning to Fix (generator thresholds)",
         "asym": "BCE, OFF x10 + adequacy guard (ours, previous)", "rl": "REINFORCE probabilities (ours, previous)",
         "harm_c": "harm policy, compensated labels (proposed)", "harm_u": "harm policy, uncompensated labels",
         "harm_c_guard": "harm policy (compensated) + adequacy guard", "rac_lp": "RACLearn + LP-relaxation guard (post hoc)",
         "harm_c_lp": "harm policy + adequacy + LP-relaxation guard (post hoc)", "harm_u_guard": "harm policy (uncompensated) + adequacy guard"}


def best_gap_at(points, s, key="gap_mean"):
    """best (lowest) gap among a rule's shares that reach at least speed-up s (None if none does): the
    attainable trade-off of the rule when the share is chosen for the target speed"""
    ok = [r[key] for r in points if r["speedup"] >= s]
    return min(ok) if ok else None


def load_pass(tag, n, obj, only=None):
    path = os.path.join("results/uc12", f"fixpolicy_eval_{tag}.jsonl")
    if not os.path.exists(path):
        return [], []
    recs = [json.loads(line) for line in open(path)]
    recs = [r for r in recs if r["i"] < n and (only is None or r["i"] in only)]
    # keep only instances on which every (rule, share) of this pass has been solved
    groups = {}
    for r in recs:
        groups.setdefault((r["method"], r["ratio"]), set()).add(r["i"])
    done = set.intersection(*groups.values()) if groups else set()
    rows, common = summarize([r for r in recs if r["i"] in done], obj)
    for r in rows:
        r["name"] = NAMES.get(r["method"], r["method"])
        r["pass"] = tag
    return rows, common


def table(rows, with_speed=True):
    L = ["| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s |"
         + (" speed-up |" if with_speed else ""), "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if with_speed else "")]
    order = list(NAMES)
    for r in sorted(rows, key=lambda r: (r["method"] != "full", r["target"], order.index(r["method"]) if r["method"] in order else 99)):
        L.append(f"| {r['name']} | {'' if r['method'] == 'full' else int(round(r['target'] * 100))} | "
                 f"{r['fixed_share']:.1f} | {r['gap_mean']:.3f} | {r['gap_median']:.3f} | {r['gap_p90']:.3f} | "
                 f"{r['gap_max']:.2f} | {r['served']:.1f} | {r['match']:.1f} | {r['time_s']:.2f} |"
                 + (f" {r['speedup']:.1f}x |" if with_speed else ""))
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="fixpolicy_results")
    a = ap.parse_args()
    out_dir = "results/uc12"
    te, va = (load(os.path.join("data/generated/uc12", f"{s_}.npz")) for s_ in ("test", "val"))
    rowsA, comA = load_pass("test", a.n, te["obj"])
    rowsBC, comBC = load_pass("test_bc", a.n, te["obj"])
    rowsV, comV = load_pass("val", 20, va["obj"])
    rowsVL, comVL = load_pass("val_lp", 20, va["obj"])
    fullA = [r for r in rowsA if r["method"] == "full"][0]
    # frontier: pass A + the combined pass (80 % and LP-guard rows; speed-up vs each pass's own full MILP) on the
    # instances both passes cover (pass A recomputed on that subset if the combined pass is shorter)
    if rowsBC:
        sub = set(comBC)
        rowsA_sub, _ = load_pass("test", a.n, te["obj"], only=sub) if sorted(comBC) != sorted(comA) else (rowsA, comA)
        front = [r for r in rowsA_sub + rowsBC if r["method"] != "full"]
        n_front = len(comBC)
    else:
        front, n_front = [r for r in rowsA if r["method"] != "full"], len(comA)
    for r in front:
        r["pareto"] = not any((q["gap_mean"] <= r["gap_mean"] and q["speedup"] >= r["speedup"]) and
                              (q["gap_mean"] < r["gap_mean"] or q["speedup"] > r["speedup"]) for q in front)
    methods = [m for m in NAMES if m != "full" and any(r["method"] == m for r in front)]
    grid = [2, 3, 4, 5, 7, 10, 15, 20, 30]
    eq = {m: {s: best_gap_at([r for r in front if r["method"] == m], s) for s in grid} for m in methods}
    eq_med = {m: {s: best_gap_at([r for r in front if r["method"] == m], s, "gap_median") for s in grid} for m in methods}

    L = [f"# Fixing policy learned from solver outcomes — B2 (uc12), first {len(comA)} test instances\n",
         "Every reduced MILP is solved in the same worker process right after the full MILP of the same instance "
         "(60 s limit, 0.1 % MIP gap, the dataset settings), 2 workers, machine shared with other jobs. Gaps are to "
         "`test['obj']` (negative = cheaper than the dataset MILP); served = no load shedding and no reserve "
         "shortfall. Same probability model (saved BCE GNN `uc_model1_4.pt`) for RACLearn, Learning to Fix, the "
         "asymmetric rule and the proposed policy; the REINFORCE rule uses `uc_model1_rl.pt`.\n",
         f"## 90 / 95 / 97 % fixed (pass A, {len(comA)} instances)\n",
         f"Full MILP in this pass: mean {fullA['time_s']:.1f} s, serves {fullA['served']:.1f} %, mean gap "
         f"{fullA['gap_mean']:.3f} % to the dataset MILP.\n"] + table(rowsA)
    if rowsBC:
        full_ = [r for r in rowsBC if r["method"] == "full"][0]
        head = (f"first {len(comBC)} instances, one back-to-back full MILP per instance shared by both tables: mean "
                f"{full_['time_s']:.1f} s, serves {full_['served']:.1f} %, mean gap {full_['gap_mean']:.3f} %")
        L += ["", f"## 80 % fixed (pass B; {head})\n"] + table([r for r in rowsBC if r["method"] == "full" or
                                                                  (r["target"] == 0.8 and not r["method"].endswith("_lp"))])
        L += ["", f"## Post-hoc extension: row feasibility check + LP-relaxation guard (pass C; same run as pass B)\n",
              "Added after pass A showed catastrophic shedding / shortfall outliers that the capacity guard misses; "
              "no parameter, checked on 10 val instances before this run; the guard's LP time is included.\n"]
        L += table([r for r in rowsBC if r["method"].endswith("_lp")])
    L += ["", "## Pareto data and mean gap at equal speed-up\n",
          "Points marked on the frontier are not dominated (lower-or-equal mean gap and higher-or-equal speed-up) by any "
          f"other rule/share on the {n_front} instances covered by all passes" + ": " +
          ", ".join(f"{r['name']} @ {int(round(r['target'] * 100))} %" for r in front if r.get("pareto")) + ".\n",
          "Best mean gap (%) a rule attains with a share whose speed-up is at least the column value (blank = no share of that rule is that fast):\n",
          "| rule | " + " | ".join(f"{s}x" for s in grid) + " |", "|---|" + "---|" * len(grid)]
    for m in methods:
        L.append(f"| {NAMES[m]} | " + " | ".join("" if eq[m][s] is None else f"{eq[m][s]:.3f}" for s in grid) + " |")
    L += ["", "Best median gap (%) at a speed-up of at least the column value:\n", "| rule | " + " | ".join(f"{s}x" for s in grid) + " |",
          "|---|" + "---|" * len(grid)]
    for m in methods:
        L.append(f"| {NAMES[m]} | " + " | ".join("" if eq_med[m][s] is None else f"{eq_med[m][s]:.3f}" for s in grid) + " |")
    if rowsV:
        L += ["", f"## Model selection on validation (first {len(comV)} val instances, no full MILP in this run; "
              "selection between label variants and the guard)\n"] + table(rowsV, with_speed=False)
    if rowsVL:
        L += ["", f"## LP-relaxation guard on validation (first {len(comVL)} val instances, sanity check before pass C)\n"]
        L += table(rowsVL, with_speed=False)
    md = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, f"{a.out}.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, f"{a.out}.json"), "w") as f:
        json.dump({"pass_A_instances": comA, "pass_BC_instances": comBC, "val_instances": comV,
                   "rows_A": rowsA, "rows_BC": rowsBC, "rows_val": rowsV, "rows_val_lp": rowsVL, "frontier_rows": front,
                   "gap_mean_at_speedup": eq, "gap_median_at_speedup": eq_med},
                  f, indent=1, default=float)
    print(md)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for m in methods:
            rr = sorted([r for r in front if r["method"] == m], key=lambda r: r["speedup"])
            for k, key in enumerate(("gap_mean", "gap_median")):
                ax[k].plot([r["speedup"] for r in rr], [max(r[key], 1e-3) for r in rr], "o-", label=NAMES[m])
                for r in rr:
                    ax[k].annotate(f"{int(round(r['target'] * 100))}", (r["speedup"], max(r[key], 1e-3)), fontsize=7)
        for k, lab in enumerate(("mean gap % (log)", "median gap % (log, floored at 0.001)")):
            ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel("speed-up vs full MILP (log)")
            ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fixpolicy_pareto.png"), dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
