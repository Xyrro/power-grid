"""Reproducibility audit: every headline number of docs/RESEARCH.md's TL;DR and of the §6 X5-X8 tables, recomputed
from the saved results (summary JSON / MD files, and for X6-X8 also from the per-instance records with the documented
definitions). Prints a markdown table (claim, recomputed value, source, status) and writes it as JSON.

    python3 scripts/uc_base_audit.py [--out results/uc12/base_audit.json] [--lp 1]

A claim is "ok" when the recomputed value rounds to the stated one at the stated precision (ranges: both ends);
"MISMATCH" otherwise. --lp 1 also recomputes the MILP's served shares of B1 / B2 (needs the dispatch LP of every
stored MILP schedule: about a minute on one core).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ROWS = []


def J(path):
    return json.load(open(os.path.join(ROOT, path)))


def JL(path):
    return [json.loads(x) for x in open(os.path.join(ROOT, path))]


def md_rows(path):
    """{method: [cells as floats or str]} of every markdown table row, and the '- key: value' extras"""
    rows, extra = {}, {}
    for line in open(os.path.join(ROOT, path)):
        if line.startswith("| ") and not line.startswith("| method") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            vals = []
            for c in cells[1:]:
                try:
                    vals.append(float(c))
                except ValueError:
                    vals.append(c)
            rows.setdefault(cells[0], vals)
        m = re.match(r"- ([^:]+): (.+)$", line.strip())
        if m:
            try:
                extra[m.group(1)] = float(m.group(2))
            except ValueError:
                extra[m.group(1)] = m.group(2)
    return rows, extra


def rounds_to(value, claimed):
    """does value round to the decimal string claimed (e.g. '0.24', '5.2', '34')?"""
    s = claimed.strip().replace("−", "-").replace("+", "")
    dec = len(s.split(".")[1]) if "." in s else 0
    return abs(value - float(s)) <= 0.5 * 10 ** (-dec) + 1e-9


def chk(loc, what, claimed, value, src, note=""):
    """claimed: a number string, or 'a–b' for a range (value then a pair)"""
    if isinstance(value, (tuple, list)) and "–" in claimed:
        a, b = claimed.split("–")
        ok = rounds_to(value[0], a) and rounds_to(value[1], b)
        shown = f"{value[0]:.4g}–{value[1]:.4g}"
    else:
        ok = rounds_to(float(value), claimed)
        shown = f"{float(value):.4g}"
    ROWS.append(dict(loc=loc, what=what, claimed=claimed, recomputed=shown, src=src, status="ok" if ok else "MISMATCH",
                     note=note))


def note(loc, what, claimed, shown, src, text):
    ROWS.append(dict(loc=loc, what=what, claimed=claimed, recomputed=shown, src=src, status="note", note=text))


# ============================================================================ TL;DR, items 2-11 (B1 / B2 studies)
def tldr_early(lp):
    u1, x1 = md_rows("results/uc1/uc_model1_results.md")
    u2, x2 = md_rows("results/uc12/uc_model1_results.md")
    s = "results/uc1/uc_model1_results.md"
    chk("TL;DR 2", "B1 hours with an exactly tied optimum, %", "34", x1["frac_alt_commitment_within_1e-6"] * 100, s)
    chk("TL;DR 2", "B1 labels changed by canonical order, %", "21", x1["frac_labels_changed_by_canonicalisation"] * 100, s)
    chk("TL;DR 2", "B2 labels changed by canonical order, %", "62.5", x2["frac_labels_changed_by_canonicalisation"] * 100,
        "results/uc12/uc_model1_results.md")
    one = lambda R: [v[0] for k, v in R.items() if k.endswith(": top-1 -> LP") and "REINFORCE" not in k]
    v1, v2 = one(u1), one(u2)
    chk("TL;DR 3", "B1 one-shot served, range over the four imitation Model 1s, %", "59–74", (min(v1), max(v1)), s)
    chk("TL;DR 3", "B2 one-shot served, range over the four imitation Model 1s, %", "10–17", (min(v2), max(v2)),
        "results/uc12/uc_model1_results.md",
        "the GNN + symmetry + canonical labels variant serves 8.3 % (V2 table); the B1 range includes the same variant")
    if lp:
        from otsl.uc import UCModel, load_rts_gmlc
        from otsl.ucdata import load, scenario_from
        sysm = load_rts_gmlc()
        for tag, T, f, claimed in (("B1", 1, "data/generated/uc1/test.npz", "98.7"), ("B2", 12, "data/generated/uc12/test.npz", "92.5")):
            m = UCModel(sysm, T=T, network=True)
            d = load(os.path.join(ROOT, f))
            ok = []
            for i in range(len(d["obj"])):
                r = m.solve_dispatch(scenario_from(d, i), d["u"][i].reshape(T, -1))
                ok.append(r.shed < 1e-6 and r.short < 1e-6)
            chk("TL;DR 3", f"{tag} MILP served (dispatch LP of the stored MILP schedules), %", claimed, np.mean(ok) * 100, f)
    g = lambda R, k: R[k][0]
    chk("TL;DR 4", "B1 REINFORCE one-shot served, %", "93.9",
        g(u1, "GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP"), s)
    chk("TL;DR 4", "B2 REINFORCE one-shot served, %", "80.8",
        g(u2, "GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP"), "results/uc12/uc_model1_results.md")
    chk("TL;DR 4", "B1 REINFORCE + screening served, %", "97.4",
        g(u1, "GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP"), s)
    chk("TL;DR 4", "B2 REINFORCE + screening served, %", "87.5",
        g(u2, "GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP"), "results/uc12/uc_model1_results.md")
    l1, y1 = md_rows("results/uc1/uc_label_free_results.md")
    l2, y2 = md_rows("results/uc12/uc_label_free_results.md")
    chk("TL;DR 5", "B1 label cost ratio MILP / LP relaxation", "18", y1["milp_s_per_label"] / y1["lp_relaxation_s_per_label"],
        "results/uc1/uc_label_free_results.md")
    chk("TL;DR 5", "B2 label cost ratio MILP / LP relaxation", "87", y2["milp_s_per_label"] / y2["lp_relaxation_s_per_label"],
        "results/uc12/uc_label_free_results.md")
    chk("TL;DR 5", "B1 label-free + REINFORCE one-shot served, %", "94.1", g(l1, "LF + REINFORCE (LP critic, no MILP): top-1 -> LP"),
        "results/uc1/uc_label_free_results.md")
    chk("TL;DR 5", "B2 label-free + REINFORCE one-shot served, %", "84.2", g(l2, "LF + REINFORCE (LP critic, no MILP): top-1 -> LP"),
        "results/uc12/uc_label_free_results.md")
    m2, _ = md_rows("results/uc1/uc_model2_results.md")
    units = {k: v[-1] for k, v in m2.items() if k.startswith("M1 trained through") or k.startswith("imitation")}
    dpg = [v for k, v in units.items() if "direct (PG, VA)" in k]
    chk("TL;DR 6", "units committed through the direct (PG, VA) Model 2, range", "18–27", (min(dpg), max(dpg)),
        "results/uc1/uc_model2_results.md")
    chk("TL;DR 6", "units committed by plain imitation", "15", units["imitation (BCE, canonical labels) reference"],
        "results/uc1/uc_model2_results.md")
    ff = [m2[k][0] for k in ("physics decoder", "physics decoder + overload penalty")]
    chk("TL;DR 7", "physics decoder fully feasible, range %", "70–74", (min(ff), max(ff)), "results/uc1/uc_model2_results.md")
    chk("TL;DR 7", "direct (PG, VA) fully feasible, %", "0", m2["direct (PG, VA) regression [framework]"][0],
        "results/uc1/uc_model2_results.md")
    f1, _ = md_rows("results/uc1/uc_fixing_results.md")
    f2, _ = md_rows("results/uc12/uc_fixing_core_results.md")
    chk("TL;DR 8", "B1 RACLearn 90 % speed-up", "4", f1["90 %: BCE, symmetric"][6], "results/uc1/uc_fixing_results.md")
    chk("TL;DR 8", "B2 RACLearn 80 % speed-up (28.3 s / 14.8 s)", "1.9",
        f2["full MILP (time from data generation)"][5] / u2["confidence fixing 80 % + MILP (RACLearn-style)"][-1],
        "results/uc12/uc_fixing_core_results.md + uc_model1_results.md")
    chk("TL;DR 8", "B2 90 %, OFF x 10 + adequacy guard, mean gap %", "0.41", f2["90 %: BCE, asymmetric k=10 + adequacy guard"][1],
        "results/uc12/uc_fixing_core_results.md")
    chk("TL;DR 8", "B2 90 %, RACLearn, mean gap %", "1.38", f2["90 %: BCE, symmetric"][1], "results/uc12/uc_fixing_core_results.md")
    chk("TL;DR 8", "B2 90 %, OFF x 10 + adequacy guard, speed-up", "4.7", f2["90 %: BCE, asymmetric k=10 + adequacy guard"][6],
        "results/uc12/uc_fixing_core_results.md")
    chk("TL;DR 8", "B1 95 %, REINFORCE ranking, mean gap %", "1.3", f1["95 %: REINFORCE, symmetric"][1], "results/uc1/uc_fixing_results.md")
    chk("TL;DR 8", "B1 95 %, RACLearn, mean gap %", "18", f1["95 %: BCE, symmetric"][1], "results/uc1/uc_fixing_results.md")
    chk("TL;DR 8", "B2 95 %, REINFORCE ranking, mean gap %", "5.5", f2["95 %: REINFORCE, symmetric"][1], "results/uc12/uc_fixing_core_results.md")
    chk("TL;DR 8", "B2 95 %, RACLearn, mean gap %", "38.5", f2["95 %: BCE, symmetric"][1], "results/uc12/uc_fixing_core_results.md")
    chk("TL;DR 8", "B2 95 %, REINFORCE ranking, speed-up", "27", f2["95 %: REINFORCE, symmetric"][6], "results/uc12/uc_fixing_core_results.md")
    srv = [v[2] for k, v in u2.items() if "REINFORCE" in k]
    chk("TL;DR 9", "B2 REINFORCE extra cost on served instances, range %", "4.7–9.1", (min(srv), max(srv)),
        "results/uc12/uc_model1_results.md")
    combo = open(os.path.join(ROOT, "results/uc12/combo_results.md")).read()
    m = re.search(r"\| label-free \+ REINFORCE: threshold 0\.5 \+ block repair \| 1 \| ([\d.]+) %", combo)
    chk("TL;DR 10", "block repair, one LP, served % (fresh, 3 seeds)", "95", float(m.group(1)), "results/uc12/combo_results.md")
    st = open(os.path.join(ROOT, "results/uc12/selftrain_results.md")).read()
    m = re.search(r"Self-training labels cost \*\*([\d.]+) %\*\*", st)
    chk("TL;DR 10", "self-training label cost, % of MILP labels", "37", float(m.group(1)), "results/uc12/selftrain_results.md")
    m = re.search(r"\| error-cost ranking \+ adequacy guard \| 90 % \| [\d.]+ % \[[^]]+\] \| ([\d.]+) % .*\| ([\d.]+) \[[\d., ]+\]× \| 3 \|", combo)
    chk("TL;DR 10", "error-cost ranking 90 %: mean gap %", "0.28", float(m.group(1)), "results/uc12/combo_results.md")
    chk("TL;DR 10", "error-cost ranking 90 %: speed-up", "3.2", float(m.group(2)), "results/uc12/combo_results.md")
    m = re.search(r"\| RACLearn \(BCE confidence\) \| 90 % \| [\d.]+ % \| ([\d.]+) %", combo)
    chk("TL;DR 10", "RACLearn 90 %: mean gap %", "12.4", float(m.group(1)), "results/uc12/combo_results.md")
    m = re.search(r"screening over 7 thresholds \+ block repair \| ([\d.]+) distinct \(≤ 7\) \| ([\d.]+) % \[[^]]+\] \| ([\d.]+) %", combo)
    chk("TL;DR 11", "end-to-end + screening: served %", "93.3", float(m.group(2)), "results/uc12/combo_results.md")
    chk("TL;DR 11", "end-to-end + screening: median gap %", "1.04", float(m.group(3)), "results/uc12/combo_results.md")
    chk("TL;DR 11", "end-to-end + screening: LPs", "6", float(m.group(1)), "results/uc12/combo_results.md")


# ============================================================================ papereval (X5, TL;DR 11)
def pe_row(rows, method):
    return [r for r in rows if r["method"] == method][0]


def x5():
    P = J("results/papereval_results.json")
    tf, te, u24 = P["test_fresh"]["fixing_first60"], P["test"]["fixing_first60"], P["uc24"]
    src = "results/papereval_results.json"
    specs = [("12 h fresh: error-cost + both guards, 95 %", tf, "error-cost ranking + adequacy + LP guard, 95 % target", "0.59", "8.0", "93"),
             ("12 h fresh: combined pipeline, 98 %", tf, "combined: self-trained p + error cost + both guards, 98 % target", "0.84", "12.6", "92"),
             ("12 h fresh: REINFORCE ranking, 95 %", tf, "REINFORCE ranking (MILP-label REINFORCE p), 95 % target", "3.37", "32.5", "95"),
             ("12 h original: error-cost + both guards", te, "error-cost ranking + adequacy + LP guard, 95 % target", "0.63", "10.4", "92"),
             ("24 h: guarded rule, 95 % target", u24, "val-selected: imitation + adequacy + LP guards, 95% target", "0.67", "11.6", "85"),
             ("12 h fresh: paper's cost-ranked kNN", tf, "cost-ranked kNN (k = 50, best of 50 LPs)", "12.0", "5.4", "100")]
    for nm, rows, meth, g, s, fx in specs:
        r = pe_row(rows, meth)
        chk("X5", f"{nm}: gap mean %", g, r["gap_mean"], src)
        nt = ("11.6× is the load-corrected re-run of uc24ltf (11.61×); the papereval record cited by X5 gives 11.55× (→ 11.5×, "
              "as in methods/papereval.md)") if "24 h" in nm else ""
        chk("X5", f"{nm}: speed-up mean", s, r["speedup_mean"], src, nt)
        chk("X5", f"{nm}: fixed %", fx, r["fixed_mean"], src)
    for tag, claimed, g in (("TL;DR 11", "0.59", "error-cost ranking + adequacy + LP guard, 95 % target"),
                            ("TL;DR 11", "0.84", "combined: self-trained p + error cost + both guards, 98 % target")):
        r = pe_row(tf, g)
        chk(tag, f"{g}: gap mean % (paper metrics)", claimed, r["gap_mean"], src)
    chk("TL;DR 11", "error-cost + both guards 95 %: speed-up", "8.0", pe_row(tf, "error-cost ranking + adequacy + LP guard, 95 % target")["speedup_mean"], src)
    chk("TL;DR 11", "combined 98 %: speed-up", "12.6", pe_row(tf, "combined: self-trained p + error cost + both guards, 98 % target")["speedup_mean"], src)
    # "the paper's own baselines are 1.7-4.6x worse in gap and mostly 1.3-2.6x slower"
    paper = {r["method"]: r for r in P["paper_table_I"]}
    pairs = [("kNN, [0.1, 0.9]", "kNN (k = 50, IDW), [0.1, 0.90]"), ("kNN, [0.05, 0.95]", "kNN (k = 50, IDW), [0.05, 0.95]"),
             ("kNN, [0.01, 0.99]", "kNN (k = 50, IDW), [0.01, 0.99]"),
             ("cost-ranked kNN (k = 50, best of 50 LPs)", "cost-ranked kNN (k = 50, best of 50 LPs)"),
             ("kNN, tau = 0.5 (hard threshold)", "kNN (k = 50, IDW), hard threshold 0.5")]
    gr = {a: pe_row(tf, b)["gap_mean"] / paper[a]["gap_mean"] for a, b in pairs}
    sr = {a: paper[a]["speedup_mean"] / pe_row(tf, b)["speedup_mean"] for a, b in pairs}
    soft = [a for a, _ in pairs if "hard" not in a]
    chk("X5 text", "gap ratio ours / paper, constant thresholds + cost-ranked kNN", "1.7–4.6",
        (min(gr[a] for a in soft), max(gr[a] for a in soft)), src,
        f"hard threshold 0.5: ratio {gr['kNN, tau = 0.5 (hard threshold)']:.2f} (better here than in the paper), not covered by the sentence")
    chk("X5 text", "slow-down paper / ours, same baselines", "1.3–2.6", (min(sr[a] for a in soft), max(sr[a] for a in soft)), src,
        f"hard threshold: {sr['kNN, tau = 0.5 (hard threshold)']:.1f}× ('mostly')")


# ============================================================================ X6 hybrid
def x6():
    H = J("results/uc12/hybrid_results.json")
    src = "results/uc12/hybrid_results.json"
    R = H["rows"]
    for nm, row, g, s, fx in (("faithful LtF kNN eps = 1 %", "faithful LtF kNN eps=1%", "0.40", "4.6", "68"),
                              ("faithful LtF BCE eps = 1 %", "faithful LtF BCE eps=1%", "0.28", "5.2", "84"),
                              ("hybrid, 360 val", "hybrid he_bce_s0_e1_n360", "0.24", "5.2", "86"),
                              ("hybrid, other seed, 180 val", "hybrid he_bce_s1_e1_n180", "0.23", "7.6", "84")):
        chk("X6", f"{nm}: gap %", g, R[row]["gap_mean"], src)
        chk("X6", f"{nm}: speed-up", s, R[row]["speedup_mean"], src)
        chk("X6", f"{nm}: fixed %", fx, R[row]["fixed_mean"], src)
    chk("X6", "hybrid 360: max gap %", "1.78", R["hybrid he_bce_s0_e1_n360"]["gap_max"], src)
    p = H["paired"]["hybrid he_bce_s0_e1_n360 vs faithful LtF kNN eps=1%"]["gap_db"]
    chk("X6", "hybrid − LtF-kNN, gap pp", "-0.16", p["diff"], src)
    chk("X6", "hybrid − LtF-kNN, CI low", "-0.33", p["ci"][0], src)
    chk("X6", "hybrid − LtF-kNN, CI high", "-0.03", p["ci"][1], src, "also in methods/hybrid.md")
    p = H["paired"]["hybrid he_bce_s0_e1_n360 vs faithful LtF BCE eps=1%"]["gap_db"]
    chk("X6", "hybrid − LtF-BCE, gap pp", "-0.03", p["diff"], src)
    chk("X6", "hybrid − LtF-BCE, CI", "-0.15–0.06", tuple(p["ci"]), src)
    chk("X6", "worst test gap, 180 val (seed 0) %", "3.14", R["hybrid he_bce_s0_e1_n180"]["gap_max"], src)
    pil = J("results/uc12/hybrid_pilot_allguards_6inst.json")
    chk("X6", "LP guard in the check: fixed before the guards %", "99.8", pil["val_fixed_share_pre_guard"] * 100,
        "results/uc12/hybrid_pilot_allguards_6inst.json")
    chk("TL;DR 12", "12 h: hybrid gap vs LtF-kNN gap %", "0.24", R["hybrid he_bce_s0_e1_n360"]["gap_mean"], src)
    chk("TL;DR 12", "12 h: LtF-kNN gap %", "0.40", R["faithful LtF kNN eps=1%"]["gap_mean"], src)
    # independent recomputation from the per-instance records (definitions of methods/hybrid.md)
    recs = JL("results/uc12/hybrid_eval_test_fresh.jsonl")
    by = {}
    for r in recs:
        by.setdefault(r["rule"], {})[r["i"]] = r
    idx = sorted(by["full MILP"])
    full = by["full MILP"]
    db = np.array([full[i]["obj"] * (1 - full[i]["mip_gap"]) for i in idx])
    tf = np.array([full[i]["time"] for i in idx])
    for nm, rule, g, s in (("hybrid 360", "hybrid he_bce_s0_e1_n360", "0.24", "5.2"), ("LtF kNN", "faithful LtF kNN eps=1%", "0.40", "4.6"),
                           ("LtF BCE", "faithful LtF BCE eps=1%", "0.28", "5.2")):
        rr = [by[rule][i] for i in idx]
        f = np.array([r["feasible"] for r in rr])
        c = np.array([r["obj"] if r["feasible"] else np.nan for r in rr], float)
        tm = np.array([r["time"] + r["pre_s"] + r["relax_s"] + r["guard_s"] for r in rr])
        chk("X6 (per-instance)", f"{nm}: gap %", g, ((c - db) / c * 100)[f].mean(), "results/uc12/hybrid_eval_test_fresh.jsonl")
        chk("X6 (per-instance)", f"{nm}: speed-up", s, (tf / tm)[f].mean(), "results/uc12/hybrid_eval_test_fresh.jsonl")


# ============================================================================ X7 PGLib
def x7():
    P = J("results/pglib/pglib_results.json")
    src = "results/pglib/pglib_results.json"
    rows = {r["method"]: r for r in P["rows"] + P["e2e"]}
    for nm, key, fe, g, s, med in (("LtF kNN eps = 1 %", "**Learning to Fix, kNN, eps = 1 %** (paper's setting)", "96.7", "0.55", "25.5", "5.8"),
                                   ("ours 98 %", "ours: error-cost + adequacy + LP guard, 98 % (combined)", "100", "0.195", "12.4", "8.2"),
                                   ("ours 95 %", "ours: error-cost + adequacy + LP guard, 95 % (combined)", "100", "0.043", "8.5", None),
                                   ("learned end-to-end (5 LPs)", "end-to-end, lf p, screening 5 thresholds", "96.7", "0.65", "23.5", None),
                                   ("no learning: LP rounding + repair + 5 LPs", "end-to-end, lprelax p, screening 5 thresholds", "100", "0.33", "24.0", None)):
        r = rows[key]
        chk("X7", f"{nm}: feasible %", fe, r["feasible"], src)
        chk("X7", f"{nm}: gap %", g, r["gap_mean"], src)
        nt = ("the label-free source gives 23.59× (self-trained 23.43×, REINFORCE 23.68×); methods/pglib.md's table says 23.6×"
              if "learned" in nm else "")
        chk("X7", f"{nm}: speed-up mean", s, r["speedup_mean"], src, nt)
        if med:
            chk("X7", f"{nm}: speed-up median", med, r["speedup_median"], src)
    p = P["paired"]["harm_98_lpg"]
    chk("X7", "ours 98 % − LtF-kNN, gap pp", "-0.35", p["mean"], src)
    chk("X7", "ours 98 % − LtF-kNN, CI", "-0.70–-0.12", tuple(p["ci"]), src)
    chk("TL;DR 12", "PGLib LtF-kNN: gap %", "0.55", rows["**Learning to Fix, kNN, eps = 1 %** (paper's setting)"]["gap_mean"], src)
    chk("TL;DR 12", "PGLib ours: gap %", "0.195", rows["ours: error-cost + adequacy + LP guard, 98 % (combined)"]["gap_mean"], src)
    chk("TL;DR 12", "PGLib LP rounding: speed-up", "24", rows["end-to-end, lprelax p, screening 5 thresholds"]["speedup_mean"], src)
    # per-instance: LtF-kNN's four fastest instances, and the LP relaxation's distance to the optimum
    recs = {}
    for f in ("a", "b"):
        for r in JL(f"results/pglib/pglib_eval_test_{f}.jsonl"):
            recs[r["i"]] = r
    sps = []
    for i, r in sorted(recs.items()):
        v = r["rules"]["ltf_knn_1"]
        if v["feasible"] and v["shed"] + v["short"] <= 1e-4:
            sps.append(r["t_full"] / (v["time"] + v["extra"] + v.get("lpg_s", 0.0)))
    sps = np.sort(sps)[::-1]
    chk("X7", "LtF-kNN: the four largest per-instance speed-ups, range", "74–210", (sps[3], sps[0]),
        "results/pglib/pglib_eval_test_{a,b}.jsonl")
    chk("X7", "LtF-kNN: median speed-up (per-instance)", "5.8", float(np.median(sps)), "results/pglib/pglib_eval_test_{a,b}.jsonl")
    d = dict(np.load(os.path.join(ROOT, "data/generated/pglib_ca/test.npz")))
    gl = (d["obj"] - d["c_rel"]) / d["obj"] * 100
    dv = dict(np.load(os.path.join(ROOT, "data/generated/pglib_ca/val.npz")))
    gv = (dv["obj"] - dv["c_rel"]) / dv["obj"] * 100
    chk("X7", "LP relaxation below the MILP optimum, median over val %", "0.03", float(np.median(gv)),
        "data/generated/pglib_ca/val.npz",
        f"'within 0.03 %' is the median; val mean {gv.mean():.3f} %, max {gv.max():.2f} %; test median {np.median(gl):.3f} %, "
        f"mean {gl.mean():.3f} %, max {gl.max():.2f} %")


# ============================================================================ X8 uc24ltf
def x8():
    U = J("results/uc24/uc24ltf_results.json")
    src = "results/uc24/uc24ltf_results.json"
    rows = {r["method"]: r for r in U["rows"] + U["stored_rows"]}
    for nm, key, fe, g, s, lc, ttq in (("LtF kNN eps = 1 %", "LtF knn eps=1%", "100", "0.60", "7.1", "5.6", "1.3"),
                                       ("LtF kNN eps = 5 %", "LtF knn eps=5%", "92.5", "2.14", "25.5", "20", "0.9"),
                                       ("ours guarded 95 %", "old: 95%|best=bce_g", "100", "0.67", "14.8", "11.6", "2.6"),
                                       ("ours REINFORCE 90 %", "stored: 90%|rl", "100", "3.06", "36.6", None, "2.8")):
        r = rows[key]
        chk("X8", f"{nm}: feasible %", fe, r["feasible"], src)
        chk("X8", f"{nm}: gap %", g, r["gap_mean"], src)
        chk("X8", f"{nm}: speed-up", s, r["speedup_mean"], src)
        if lc:
            chk("X8", f"{nm}: load-corrected speed-up", lc, r["speedup_mean_loadcorr"], src)
        chk("X8", f"{nm}: time-to-quality speed-up", ttq, r["ttq_speedup_median"], src)
    p = [x for x in U["pairs"] if x["ltf"] == "LtF knn eps=1%" and x["other"] == "old: 95%|best=bce_g"][0]
    chk("X8", "guarded 95 % − LtF-kNN, gap pp", "0.06", p["diff_mean"], src)
    chk("X8", "guarded 95 % − LtF-kNN, CI", "-0.14–0.26", tuple(p["diff_ci"]), src)
    chk("TL;DR 12", "24 h: speed ratio guarded / LtF-kNN (load-corrected)", "2",
        rows["old: 95%|best=bce_g"]["speedup_mean_loadcorr"] / rows["LtF knn eps=1%"]["speedup_mean_loadcorr"], src)
    # per-instance recomputation (definitions of methods/uc24ltf.md)
    d = dict(np.load(os.path.join(ROOT, "data/generated/uc24/test.npz")))
    ev = JL("results/uc24/uc24ltf_eval_test.jsonl")
    idx = np.array([r["i"] for r in ev])
    for nm, key, g, s in (("LtF kNN eps = 1 %", "LtF knn eps=1%", "0.60", "7.1"), ("ours guarded 95 %", "old: 95%|best=bce_g", "0.67", "14.8")):
        f, gap, sp = [], [], []
        for r in ev:
            v = r[key]
            tm = v["pre_s"] + v["t_guard"] + v["t_milp"] + (r["t_rel"] if "knn" not in key else 0.0)
            if v["feasible"]:
                gap.append((v["obj"] - d["bound"][r["i"]]) / v["obj"] * 100)
                sp.append(d["time"][r["i"]] / tm)
        chk("X8 (per-instance)", f"{nm}: gap %", g, np.mean(gap), "results/uc24/uc24ltf_eval_test.jsonl")
        chk("X8 (per-instance)", f"{nm}: speed-up", s, np.mean(sp), "results/uc24/uc24ltf_eval_test.jsonl")
    b3 = open(os.path.join(ROOT, "results/uc24/b3_results.md")).read()
    m = re.search(r"t_within_0\.5%_of_final_median_s ([\d.]+)", b3)
    chk("TL;DR 11", "24 h: full MILP within 0.5 % of its final cost, median s", "28", float(m.group(1)), "results/uc24/b3_results.md")


# ============================================================================ LtF on 12 h (X3 numbers quoted in the TL;DR)
def ltfx():
    L = J("results/uc12/ltfx_results.json")["rows"]
    src = "results/uc12/ltfx_results.json"
    chk("TL;DR 11", "LtF kNN eps = 1 %: gap to DB %", "0.40", L["knn ltf eps=1%"]["gap_db_mean"], src)
    chk("TL;DR 11", "LtF kNN eps = 1 %: speed-up", "4.7", L["knn ltf eps=1%"]["speedup_mean"], src)
    chk("TL;DR 11", "LtF BCE eps = 1 %: gap %", "0.28", L["bce ltf eps=1%"]["gap_db_mean"], src)
    chk("TL;DR 11", "LtF BCE eps = 1 %: speed-up", "5.2", L["bce ltf eps=1%"]["speedup_mean"], src)
    chk("TL;DR 11", "LtF self-trained eps = 10 %: gap %", "0.85", L["st ltf eps=10%"]["gap_db_mean"], src)
    chk("TL;DR 11", "LtF self-trained eps = 10 %: speed-up", "13.5", L["st ltf eps=10%"]["speedup_mean"], src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--lp", type=int, default=0)
    a = ap.parse_args()
    tldr_early(a.lp)
    x5()
    x6()
    x7()
    x8()
    ltfx()
    n_bad = sum(r["status"] == "MISMATCH" for r in ROWS)
    print(f"{len(ROWS)} checks, {n_bad} mismatches\n")
    print("| where | quantity | stated | recomputed | status | source / note |\n|---|---|---|---|---|---|")
    for r in ROWS:
        if r["status"] != "ok" or r["note"]:
            print(f"| {r['loc']} | {r['what']} | {r['claimed']} | {r['recomputed']} | {r['status']} | {r['src']}"
                  + (f"; {r['note']}" if r["note"] else "") + " |")
    if a.out:
        json.dump(dict(n_checks=len(ROWS), n_mismatch=n_bad, rows=ROWS), open(a.out, "w"), indent=1)
