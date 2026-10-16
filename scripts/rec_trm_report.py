"""Tables of the TRM study (arm A): validation ablations, test results with paired bootstrap CIs.

    python3 scripts/rec_trm_report.py

Reads results/recursive/trm_val_*.json, trm_train_*.json, trm_select.json, trm_eval_<split>_s*.jsonl; writes
results/recursive/trm_results.{md,json}.
Test metrics (paper metrics as in docs/methods/base.md): gap (C - DB) / C to the dual bound DB of the full MILP at
0.1 % solved in the same process (every end-to-end schedule has a dispatch, the LP prices shedding and reserve
shortfall, so the mean includes those penalties); served = no shedding and no reserve shortfall; speed-up =
T_MILP(0.1 %) / T_method per instance (mean, median, geometric mean); paired instance bootstrap (2,000 resamples) of
the gap difference (pp) and of the log speed-up difference.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.base import boot_ci, paired  # noqa: E402

RES = os.path.join("results", "recursive")
K_CURVE = (1, 2, 4, 8, 16, 32)
BASE = [("milp01", "full MILP, 0.1 % (reference)"), ("milp05", "full MILP, 0.5 %"),
        ("lpround5", "LP rounding + repair, 5 thresholds (no learning)"),
        ("lpround5r", "LP rounding, 5 thresholds + LP repair"),
        ("m1x_1", "m1x temporal-GNN ensemble, 1 LP"), ("m1x_1r", "m1x ensemble, 1 LP + LP repair"),
        ("m1x_5", "m1x ensemble, 5 thresholds"), ("m1x_5r", "m1x ensemble, 5 thresholds + LP repair"),
        ("m1x_mix5", "m1x ensemble + 4 LP roundings"), ("m1x_mix5r", "m1x ensemble + 4 LP roundings + LP repair"),
        ("rl_rl1", "REINFORCE end-to-end (label-free + RL), 1 LP"), ("rl_rl1r", "REINFORCE (label-free + RL), 1 LP + LP repair"),
        ("rl_lag7", "REINFORCE end-to-end (Lagrangian + KL), 7 thresholds"),
        ("rl_lag7r", "REINFORCE (Lagrangian + KL), 7 thresholds + LP repair")]
SUFFIX = [("_1", "1 LP"), ("_1r", "1 LP + LP repair"), ("_t5", "5 thresholds"), ("_t5r", "5 thresholds + LP repair"),
          ("_mix5", "+ 4 LP roundings"), ("_mix5r", "+ 4 LP roundings + LP repair"),
          ("_s5", "last 5 steps screened"), ("_s5r", "last 5 steps + LP repair"), ("_halt", "halting head, 1 LP"),
          ("_haltr", "halting head + LP repair"), ("_halt2r", "cost-based halting head + LP repair"),
          ("_fb", "1 LP + LP repair + safety-net reduced MILP")]


def load_eval(split="test_fresh"):
    recs = {}
    for f in sorted(glob.glob(os.path.join(RES, f"trm_eval_{split}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs[r["i"]] = r
    return [recs[i] for i in sorted(recs)]


def stats(R, key):
    db = np.array([r["milp01"]["bound"] for r in R])
    tf = np.array([r["milp01"]["time"] for r in R])
    c = np.array([r[key]["cost"] for r in R], float)
    t = np.array([r[key]["time"] for r in R], float)
    sv = np.array([r[key]["shed"] < 1e-6 and r[key]["short"] < 1e-6 for r in R])
    nlp = np.array([r[key].get("n_lp", np.nan) for r in R], float)
    g = (c - db) / c * 100
    sp = tf / t
    out = dict(n=len(R), served=float(sv.mean() * 100), gap_mean=float(g.mean()), gap_ci=boot_ci(g),
               gap_median=float(np.median(g)), gap_max=float(g.max()),
               gap_mean_served=float(g[sv].mean()) if sv.any() else np.nan, n_gt1=int((g > 1).sum()),
               time_mean=float(t.mean()), time_median=float(np.median(t)), speedup_mean=float(sp.mean()),
               speedup_ci=boot_ci(sp), speedup_median=float(np.median(sp)), speedup_geo=float(np.exp(np.log(sp).mean())),
               lps=float(np.nanmean(nlp)) if np.isfinite(nlp).any() else np.nan)
    if key.endswith("_fb"):
        out["fired"] = float(np.mean([r[key].get("fired", False) for r in R]) * 100)
        out["from_milp"] = float(np.mean([r[key].get("from_milp", False) for r in R]) * 100)
    if "k" in R[0][key]:
        out["steps_mean"] = float(np.mean([r[key]["k"] for r in R]))
    return out, g, np.log(sp)


def fmt_row(name, s):
    lps = "–" if not np.isfinite(s["lps"]) else f"{s['lps']:.1f}"
    return (f"| {name} | {s['served']:.1f} % | {s['gap_mean']:.2f} % [{s['gap_ci'][0]:.2f}, {s['gap_ci'][1]:.2f}] | "
            f"{s['gap_median']:.2f} % | {s['gap_max']:.1f} % | {s['time_mean']:.2f} s | "
            f"{s['speedup_mean']:.1f}× [{s['speedup_ci'][0]:.1f}, {s['speedup_ci'][1]:.1f}] | {s['speedup_median']:.1f}× | "
            f"{s['speedup_geo']:.1f}× | {lps} |")


def test_tables(R, plan, lines, out):
    keys = [(k, n) for k, n in BASE if k in R[0]]
    for tag in plan["trm"]:
        keys += [(tag + s, f"TRM `{tag}`: {n}" if tag == plan["final"] else f"`{tag}`: {n}") for s, n in SUFFIX
                 if tag + s in R[0]]
    S, G, L = {}, {}, {}
    for k, _ in keys:
        S[k], G[k], L[k] = stats(R, k)
    out["test"] = dict(n=len(R), stats=S, plan=plan)
    lines += [f"## Test ({len(R)} instances of test_fresh; paired timing, one core per instance)", "",
              "| method | served | gap to DB, mean [95 % CI] | median | max | time | speed-up mean [CI] | median | geo. mean | LPs |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [fmt_row(n, S[k]) for k, n in keys]
    for k, _ in keys:
        if k.endswith("_fb"):
            lines.append(f"\n`{k}`: the safety net fired on {S[k]['fired']:.1f} % of the instances; the reduced MILP gave the "
                         f"cheaper schedule on {S[k]['from_milp']:.1f} %.")
    fin = plan["final"]
    ctrl = [t for t in plan["trm"] if t != fin]
    lines += ["", "### Paired differences (TRM − reference): Δ gap pp [95 % CI] / Δ log speed-up [CI] (> 0: TRM faster)", ""]
    head = plan.get("headline", fin + "_1r")
    hs = head[len(fin):]
    lines += [f"**Headline pipeline {head}** (chosen on validation)", "", "| reference | Δ gap, pp | Δ log speed-up |",
              "|---|---|---|"]
    for r in ["lpround5r", "lpround5", "m1x_mix5r", "m1x_5r", "rl_lag7r", "milp05", "milp01"] + \
            [c + sx for c in ctrl for sx in (hs, "_t5r")]:
        if r in S and r != head:
            dg, dl = paired(G[head], G[r]), paired(L[head], L[r])
            out["paired"] = out.get("paired", {})
            out["paired"][f"{head}|{r}"] = dict(gap=dg, logspeed=dl)
            lines.append(f"| {r} | {dg['diff']:+.2f} [{dg['ci'][0]:+.2f}, {dg['ci'][1]:+.2f}] | "
                         f"{dl['diff']:+.2f} [{dl['ci'][0]:+.2f}, {dl['ci'][1]:+.2f}] |")
    lines += ["", "**Control models' pipelines against the no-learning baseline and the loosened MILP**", "",
              "| pipeline | vs lpround5r: Δ gap / Δ log speed-up | vs milp05: Δ gap / Δ log speed-up |", "|---|---|---|"]
    for c in [fin] + ctrl + ["m1x"]:
        for sx in ("_1r", "_t5r", "_5r", "_mix5r", "_s5r", "_fb"):
            k = c + sx
            if k not in S:
                continue
            cells = []
            for r in ("lpround5r", "milp05"):
                dg, dl = paired(G[k], G[r]), paired(L[k], L[r])
                out.setdefault("paired", {})[f"{k}|{r}"] = dict(gap=dg, logspeed=dl)
                cells.append(f"{dg['diff']:+.2f} [{dg['ci'][0]:+.2f}, {dg['ci'][1]:+.2f}] / {dl['diff']:+.2f} "
                             f"[{dl['ci'][0]:+.2f}, {dl['ci'][1]:+.2f}]")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    comps = [("_1r", ["lpround5r", "m1x_1r", "m1x_5r", "rl_lag7r", "milp05"] + [c + "_1r" for c in ctrl]),
             ("_1", ["lpround5", "m1x_1", "m1x_5", "rl_lag7"] + [c + "_1" for c in ctrl]),
             ("_s5r", ["lpround5r", "m1x_5r", "rl_lag7r", "milp05", fin + "_1r"] + [c + "_t5r" for c in ctrl]),
             ("_t5r", ["lpround5r", "m1x_5r", "rl_lag7r", "milp05"] + [c + "_t5r" for c in ctrl]),
             ("_fb", ["lpround5r", "m1x_5r", "milp05", fin + "_1r"])]
    out.setdefault("paired", {})
    for sfx, refs in comps:
        for tag in [fin] + ctrl:
            k = tag + sfx
            if k not in S:
                continue
            refs_ = [r for r in refs if r in S and r != k]
            lines += [f"**{k}**", "", "| reference | Δ gap, pp | Δ log speed-up |", "|---|---|---|"]
            for r in refs_:
                dg, dl = paired(G[k], G[r]), paired(L[k], L[r])
                out["paired"][f"{k}|{r}"] = dict(gap=dg, logspeed=dl)
                lines.append(f"| {r} | {dg['diff']:+.2f} [{dg['ci'][0]:+.2f}, {dg['ci'][1]:+.2f}] | "
                             f"{dl['diff']:+.2f} [{dl['ci'][0]:+.2f}, {dl['ci'][1]:+.2f}] |")
            lines.append("")
            if tag != fin:
                break
    lines += ["### Test-time compute on test (1 LP after k supervision steps; plain / + LP repair)", "",
              "| model | " + " | ".join(f"k = {k}" for k in K_CURVE) + " |", "|---|" + "---|" * len(K_CURVE)]
    out["curve_test"] = {}
    for tag in plan["trm"]:
        if f"{tag}_k1" not in R[0]:
            continue
        cs = {k: (stats(R, f"{tag}_k{k}")[0], stats(R, f"{tag}_k{k}r")[0]) for k in K_CURVE}
        out["curve_test"][tag] = cs
        lines.append(f"| {tag} | " + " | ".join(f"{a['gap_mean']:.2f} % ({a['served']:.0f} %) / {b['gap_mean']:.2f} % "
                                                 f"({b['served']:.0f} %), {b['time_mean']:.2f} s" for a, b in cs.values()) + " |")
    rel = [r["relax"]["urel_diff_stored"] for r in R]
    tf = np.array([r["milp01"]["time"] for r in R])
    lines += ["", f"Re-solved LP relaxation vs stored: max |Δu_rel| = {max(rel):.2e}. Full MILP at 0.1 %: mean {tf.mean():.1f} s "
                  f"(median {np.median(tf):.1f} s), {np.mean([r['milp01']['status'] != 'optimal' for r in R]) * 100:.0f} % "
                  f"at the 60 s limit; LP relaxation {np.mean([r['relax']['time'] for r in R]):.2f} s.", ""]


def val_tables(lines, out):
    vals = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "trm_val_*.json")))}
    out["val"] = {}
    lines += ["## Validation (360 instances; gap to the stored validation MILP objective C*)", "",
              "1-LP pipelines with their selected decoder / threshold; 5 thresholds = greedy-selected screening; "
              "\"+ rep\" = with the LP-guided repair (selected separately).", "",
              "| model | mode | n | T | n_sup | LP feats | symmetry | params | BCE @k | wrong @k | 1 LP | 1 LP + rep | "
              "5 thr. + rep | last 5 steps + rep | + 4 LP roundings + rep |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(r):
        return f"{r['gap_mean']:.2f} / {r['gap_median']:.2f} % ({r['served']:.0f} %)" if r else "–"

    for tag, v in vals.items():
        s = v.get("selection", {})
        if tag == "lpround":
            lines.append(f"| LP rounding (no learning) | – | – | – | – | – | – | 0 | – | – | – | – | "
                         f"{cell(s.get('screen5_rep'))} | – | – |")
            out["val"][tag] = s
            continue
        c = v.get("cfg", {})
        kd = s.get("rep", {}).get("k_def", 1)
        pm = v.get("prob", {}).get(str(kd), {})
        steps = s.get(f"screen_steps_k{kd}_rep")
        lines.append(f"| {tag} | {c.get('mode', 'one-shot')} | {c.get('n', '–')} | {c.get('T', '–')} | {c.get('n_sup', '–')} | "
                     f"{c.get('lp_feats', True)} | {c.get('sym', 'rank')} | {v.get('params', '–')} | "
                     f"{pm.get('bce', float('nan')):.4f} | {pm.get('wrong', float('nan')):.1f} | "
                     f"{cell(s['plain']['one_lp'])} ({s['plain']['decoder'][:3]} {s['plain']['th']}) | "
                     f"{cell(s['rep']['one_lp'])} ({s['rep']['decoder'][:3]} {s['rep']['th']}) | "
                     f"{cell(s['rep'].get('screen_th_metrics'))} | {cell(steps)} | {cell(s['rep'].get('mix_metrics'))} |")
        out["val"][tag] = dict(selection=s, cfg=c, prob=v.get("prob"), params=v.get("params"))
    lines += ["", "### Test-time compute on validation (1 LP + repair after k steps, selected decoder; mean gap (served))", "",
              "| model | " + " | ".join(f"k = {k}" for k in K_CURVE) + " | halting head (mean steps) |",
              "|---|" + "---|" * (len(K_CURVE) + 1)]
    for tag, v in vals.items():
        s = v.get("selection", {})
        if "curve" not in s:
            continue
        cv, hq = s["curve"], s["halt_q"]
        lines.append(f"| {tag} | " + " | ".join(f"{cv[str(k)]['rep']['gap_mean']:.2f} % ({cv[str(k)]['rep']['served']:.0f} %)"
                                                 for k in K_CURVE)
                     + f" | {hq['rep']['gap_mean']:.2f} % ({hq['steps_mean']:.1f}) |")
    for f in sorted(glob.glob(os.path.join(RES, "trm_halt_*.json"))):
        h = json.load(open(f))
        out.setdefault("halt_posthoc", {})[h["tag"]] = {k: v for k, v in h.items() if k != "w"}
        fx = h["val_fixed_steps"]
        lines.append(f"\nPost-hoc halting head of `{h['tag']}` (fitted on {h['n_fit']} training instances, target: within "
                     f"{h['tol'] * 100:.1f} % of the best known cost; {h['good_rate_fit'] * 100:.0f} % of the fitted steps are "
                     f"good): validation {h['val']['gap_mean']:.2f} % (median {h['val']['gap_median']:.2f} %, served "
                     f"{h['val']['served']:.1f} %) at {h['val']['steps_mean']:.1f} steps on average, against "
                     f"{fx['1']['gap_mean']:.2f} % at 1 step and {fx[str(h['K'])]['gap_mean']:.2f} % at {h['K']}; the best step "
                     f"per instance in hindsight (oracle): {h['val_oracle_best_step_gap']:.2f} %.")
    lines += ["", "### Probability quality by supervision step (validation: BCE against the model's targets / wrong "
              "unit-hours per instance)", "", "| model | " + " | ".join(f"k = {k}" for k in K_CURVE) + " |",
              "|---|" + "---|" * len(K_CURVE)]
    for tag, v in vals.items():
        if v.get("cfg", {}).get("mode") != "trm":
            continue
        lines.append(f"| {tag} | " + " | ".join(f"{v['prob'][str(k)]['bce']:.4f} / {v['prob'][str(k)]['wrong']:.1f}"
                                                 for k in K_CURVE) + " |")
    lines.append("")


def train_table(lines, out):
    tr = {}
    for f in sorted(glob.glob(os.path.join(RES, "trm_train_*.json"))):
        d = json.load(open(f))
        last = d["train"][-1] if d.get("train") else (d["hist"][-1] if d.get("hist") else {})
        tr[d["tag"]] = dict(stage=d["stage"], params=d.get("params"), train_s=d.get("train_s"), last=last,
                            val=d["val"][-1] if d.get("val") else None)
    out["train"] = tr
    lines += ["## Training runs", "", "| tag | stage | params | wall time | last training log | last val (60, EMA) |",
              "|---|---|---|---|---|---|"]
    for t, d in tr.items():
        last = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (d["last"] or {}).items()
                if k in ("bce", "wrong", "steps_per_inst", "epoch", "gap_rel", "served")}
        val = {k: round(v, 4) for k, v in (d["val"] or {}).items() if k != "step"}
        lines.append(f"| {t} | {d['stage']} | {d['params']} | {d['train_s'] / 60:.0f} min | {json.dumps(last)} | {json.dumps(val)} |")
    lines.append("")


def main():
    os.chdir(os.path.dirname(HERE))
    out, lines = {}, ["# TRM study (arm A): results", "",
                      "Generated by `scripts/rec_trm_report.py`. Method and setup: `docs/recursive/trm.md`.", ""]
    R = load_eval()
    sp = os.path.join(RES, "trm_select.json")
    if R and os.path.exists(sp):
        test_tables(R, json.load(open(sp)), lines, out)
    val_tables(lines, out)
    train_table(lines, out)
    with open(os.path.join(RES, "trm_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "trm_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
