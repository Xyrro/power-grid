"""Consolidate result JSONs into the tables used in docs/RESEARCH.md.

    python scripts/tables.py case30 case118   ->  docs/tables_<case>.md

'gap closed' = 1 - mean gap(method) / mean gap(all-closed DC-OPF), recomputed from the logged gaps so
that runs made before the metric was added are comparable.
"""
import json
import os
import sys


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def row_md(r, cols):
    out = []
    for c in cols:
        v = r.get(c, "")
        out.append(f"{v:.3f}" if isinstance(v, float) else str(v))
    return "| " + " | ".join(out) + " |"


def table(rows, cols, names):
    s = "| " + " | ".join(names) + " |\n|" + "---|" * len(cols) + "\n"
    return s + "\n".join(row_md(r, cols) for r in rows) + "\n"


if __name__ == "__main__":
    for cfg in sys.argv[1:]:
        d = os.path.join("results", cfg)
        m1 = load(os.path.join(d, "model1_results.json"))
        if m1 is None:
            continue
        gap0 = next(r["gap_mean_%"] for r in m1["rows"] if r["method"].startswith("all-closed"))
        fix = lambda rows: [dict(r, **{"gap_closed_%": 100 * (1 - r["gap_mean_%"] / gap0),
                                       "method": r["method"].replace(" | ", ": ")}) for r in rows]
        md = [f"# Result tables: {cfg}\n", f"All-closed DC-OPF mean gap to the MILP: {gap0:.4f} %\n"]
        cols = ["method", "gap_mean_%", "gap_closed_%", "feasible_%", "beats_or_ties_milp_%", "n_open", "LPs_per_scenario"]
        names = ["method", "mean gap %", "gap closed %", "feasible %", "matches/beats MILP %", "lines opened", "LPs / scenario"]
        md += ["## Model 1 study\n", table(fix(m1["rows"]), cols, names)]
        for name, title in [("value_results.json", "Learned switching values"),
                            ("label_free_results.json", "Label-free REINFORCE")]:
            r = load(os.path.join(d, name))
            if r:
                md += [f"## {title}\n", table(fix(r["rows"]), cols, names)]
        m2 = load(os.path.join(d, "model2_results.json"))
        if m2:
            ca = ["method", "pg_mse", "kcl_max_pu", "kcl_total_rel_%", "line_viol_max_pu", "feasible_%", "cost_abs_err_%"]
            na = ["Model 2", "PG MSE", "worst KCL (p.u.)", "total KCL / demand %", "worst overload (p.u.)",
                  "feasible %", "cost error %"]
            md += ["## Model 2 accuracy\n", table(m2["A"], ca, na)]
            cb = ["method", "gap_mean_%", "gap_closed_%", "LPs_per_scenario", "spearman"]
            md += ["## Model 2 as screener\n", table(fix(m2["B"]), cb, ["method", "mean gap %", "gap closed %",
                                                                         "LPs / scenario", "Spearman"])]
            cc = ["method", "gap_mean_%", "gap_closed_%", "feasible_%", "n_open"]
            md += ["## Training Model 1 through Model 2\n", table(fix(m2["C"]), cc, ["method", "mean gap %", "gap closed %",
                                                                                    "feasible %", "lines opened"])]
        out = os.path.join("docs", "ots", f"tables_{cfg}.md")
        open(out, "w").write("\n".join(md))
        print("wrote", out)
