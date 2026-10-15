"""Tables of the TRM study (arm A): validation ablations, test results with paired bootstrap CIs.

    python3 scripts/rec_trm_report.py

Reads results/recursive/trm_val_*.json, trm_train_*.json, trm_select.json, trm_eval_<split>_s*.jsonl; writes
results/recursive/trm_results.{md,json}.
Test metrics (Learning to Fix's paper metrics, as in docs/methods/base.md): gap (C - DB) / C to the dual bound DB of the
full MILP at 0.1 % solved in the same process; every end-to-end schedule has an LP dispatch (feasible by construction
of the slack-priced LP), served = no shedding and no reserve shortfall; speed-up = T_MILP(0.1 %) / T_method per
instance (mean, median, geometric mean); paired instance bootstrap (2,000 resamples) of the gap difference (pp) and
of the log speed-up difference.
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
BASE_ORDER = ["milp01", "milp05", "lpround5", "m1x_1", "m1x_5", "rl_rl1", "rl_lag7"]
LABEL = {"milp01": "full MILP, 0.1 % (reference)", "milp05": "full MILP, 0.5 %",
         "lpround5": "LP rounding + repair + 5 LPs (no learning)", "m1x_1": "m1x temporal-GNN ensemble, 1 LP",
         "m1x_5": "m1x temporal-GNN ensemble, 5 thresholds (<= 5 LPs)",
         "rl_rl1": "REINFORCE end-to-end (label-free + RL), 1 LP", "rl_lag7": "REINFORCE end-to-end (Lagrangian + KL), 7 thresholds"}
SUFFIX = {"_1": "1 LP", "_s5": "screen last 5 steps (<= 5 LPs)", "_t5": "5 thresholds (<= 5 LPs)",
          "_halt": "halting head, 1 LP", "_fb": "1 LP + safety-net reduced MILP"}


def load_eval(split="test_fresh"):
    recs = {}
    for f in sorted(glob.glob(os.path.join(RES, f"trm_eval_{split}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs[r["i"]] = r
    return [recs[i] for i in sorted(recs)]


def arrays(R, key):
    c = np.array([r[key]["cost"] if key in r else np.nan for r in R], float)
    t = np.array([r[key]["time"] if key in r else np.nan for r in R], float)
    sv = np.array([(r[key]["shed"] < 1e-6 and r[key]["short"] < 1e-6) if key in r else False for r in R])
    nlp = np.array([r[key].get("n_lp", np.nan) if key in r else np.nan for r in R], float)
    return c, t, sv, nlp


def stats(R, key):
    db = np.array([r["milp01"]["bound"] for r in R])
    tf = np.array([r["milp01"]["time"] for r in R])
    c, t, sv, nlp = arrays(R, key)
    g = (c - db) / c * 100
    sp = tf / t
    out = dict(n=int(np.isfinite(c).sum()), served=float(sv.mean() * 100), gap_mean=float(np.nanmean(g)),
               gap_ci=boot_ci(g), gap_median=float(np.nanmedian(g)), gap_max=float(np.nanmax(g)),
               gap_mean_served=float(np.nanmean(g[sv])) if sv.any() else np.nan, n_gt1=int((g > 1).sum()),
               time_mean=float(np.nanmean(t)), time_median=float(np.nanmedian(t)), speedup_mean=float(np.nanmean(sp)),
               speedup_ci=boot_ci(sp), speedup_median=float(np.nanmedian(sp)),
               speedup_geo=float(np.exp(np.nanmean(np.log(sp)))), lps=float(np.nanmean(nlp)))
    if key.endswith("_fb"):
        out["fired"] = float(np.mean([r[key].get("fired", False) for r in R]) * 100)
        out["from_milp"] = float(np.mean([r[key].get("from_milp", False) for r in R]) * 100)
    return out, g, np.log(sp)


def fmt_row(name, s):
    return (f"| {name} | {s['served']:.1f} % | {s['gap_mean']:.2f} % [{s['gap_ci'][0]:.2f}, {s['gap_ci'][1]:.2f}] | "
            f"{s['gap_median']:.2f} % | {s['gap_max']:.2f} % | {s['time_mean']:.2f} s | "
            f"{s['speedup_mean']:.1f}× [{s['speedup_ci'][0]:.1f}, {s['speedup_ci'][1]:.1f}] | {s['speedup_median']:.1f}× | "
            f"{s['speedup_geo']:.1f}× | {s['lps']:.1f} |")


def val_tables(lines, out):
    vals = {}
    for f in sorted(glob.glob(os.path.join(RES, "trm_val_*.json"))):
        tag = os.path.basename(f)[8:-5]
        vals[tag] = json.load(open(f))
    out["val"] = {}
    lines += ["## Validation (360 instances; gap to the stored validation MILP objective C*)", "",
              "| model | mode | n | T | n_sup | LP feats | symmetry | params | BCE @k_def | wrong / inst. @k_def | "
              "1 LP: decoder, th | 1 LP: gap mean / median / max | served | last-5 steps | 5 thresholds |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, v in vals.items():
        s = v.get("selection", {})
        rows = v.get("rows", {})
        if tag == "lpround":
            r = rows["screen5"]
            lines.append(f"| LP rounding (no learning) | – | – | – | – | – | – | 0 | – | – | 5 thresholds | "
                         f"{r['gap_mean']:.2f} / {r['gap_median']:.2f} / {r['gap_max']:.1f} % | {r['served']:.1f} % | – | – |")
            out["val"][tag] = rows
            continue
        c = v.get("cfg", {})
        kd = s.get("k_def", 1)
        pm = v.get("prob", {}).get(str(kd), {}) if "prob" in v else {}
        one = s.get("one_lp", {})
        st = s.get("screen_steps")
        lines.append(f"| {tag} | {c.get('mode', 'm1x')} | {c.get('n', '–')} | {c.get('T', '–')} | {c.get('n_sup', '–')} | "
                     f"{c.get('lp_feats', True)} | {c.get('sym', 'rank')} | {v.get('params', '–')} | "
                     f"{pm.get('bce', float('nan')):.4f} | {pm.get('wrong', float('nan')):.1f} | {s.get('decoder')}, {s.get('th')} | "
                     f"{one.get('gap_mean', np.nan):.2f} / {one.get('gap_median', np.nan):.2f} / {one.get('gap_max', np.nan):.1f} % | "
                     f"{one.get('served', np.nan):.1f} % | "
                     + (f"{st['gap_mean']:.2f} % ({st.get('lps', np.nan):.1f} LPs)" if st else "–") + " | "
                     f"{s.get('screen_th_metrics', {}).get('gap_mean', np.nan):.2f} % |")
        out["val"][tag] = dict(selection=s, cfg=c, prob=v.get("prob"), params=v.get("params"))
    lines += ["", "### Test-time compute on validation (selected decoder and threshold; mean gap / served)", "",
              "| model | " + " | ".join(f"k = {k}" for k in K_CURVE) + " | halting head (mean steps) |",
              "|---|" + "---|" * (len(K_CURVE) + 1)]
    for tag, v in vals.items():
        s = v.get("selection", {})
        if "curve" not in s:
            continue
        cv = s["curve"]
        hq = v["rows"].get("halt_q", {})
        lines.append(f"| {tag} | " + " | ".join(f"{cv[str(k)]['gap_mean']:.2f} % / {cv[str(k)]['served']:.0f} %"
                                                 for k in K_CURVE)
                     + f" | {hq.get('gap_mean', np.nan):.2f} % ({hq.get('steps_mean', np.nan):.1f}) |")
    lines += ["", "### Probability quality by supervision step (validation, BCE against the model's targets / wrong "
              "unit-hours per instance)", "", "| model | " + " | ".join(f"k = {k}" for k in K_CURVE) + " |",
              "|---|" + "---|" * len(K_CURVE)]
    for tag, v in vals.items():
        if "prob" not in v or v.get("cfg", {}).get("mode") != "trm":
            continue
        lines.append(f"| {tag} | " + " | ".join(f"{v['prob'][str(k)]['bce']:.4f} / {v['prob'][str(k)]['wrong']:.1f}"
                                                 for k in K_CURVE) + " |")
    lines.append("")


def train_table(lines, out):
    tr = {}
    for f in sorted(glob.glob(os.path.join(RES, "trm_train_*.json"))):
        d = json.load(open(f))
        tr[d["tag"]] = dict(stage=d["stage"], params=d.get("params"), train_s=d.get("train_s"),
                            last=d["train"][-1] if d.get("train") else (d["hist"][-1] if d.get("hist") else None),
                            val=d["val"][-1] if d.get("val") else None)
    out["train"] = tr
    lines += ["## Training runs", "", "| tag | stage | params | wall time | last train log | last val (60, EMA) |",
              "|---|---|---|---|---|---|"]
    for t, d in tr.items():
        lines.append(f"| {t} | {d['stage']} | {d['params']} | {d['train_s'] / 60:.0f} min | "
                     f"{json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in (d['last'] or {}).items() if k in ('bce', 'wrong', 'steps_per_inst', 'epoch', 'gap_rel', 'served')})} | "
                     f"{json.dumps({k: round(v, 4) for k, v in (d['val'] or {}).items() if k != 'step'})} |")
    lines.append("")


def main():
    os.chdir(os.path.dirname(HERE))
    out, lines = {}, ["# TRM study (arm A): results", "",
                      "Generated by `scripts/rec_trm_report.py`. Method and setup: `docs/recursive/trm.md`.", ""]
    R = load_eval()
    plan = json.load(open(os.path.join(RES, "trm_select.json"))) if os.path.exists(os.path.join(RES, "trm_select.json")) else None
    if R and plan:
        keys = [k for k in BASE_ORDER if k in R[0]]
        trm_keys = []
        for tag in plan["trm"]:
            trm_keys += [tag + sfx for sfx in SUFFIX if tag + sfx in R[0]]
        keys += trm_keys
        S, G, L = {}, {}, {}
        for k in keys:
            S[k], G[k], L[k] = stats(R, k)
        out["test"] = dict(n=len(R), stats=S, plan=plan)
        lines += [f"## Test ({len(R)} instances of test_fresh, paired timing on one core per instance)", "",
                  "| method | served | gap to DB, mean [95 % CI] | median | max | time | speed-up mean [CI] | median | geo. mean | LPs |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            name = LABEL.get(k)
            if name is None:
                tag = next(t for t in plan["trm"] if k.startswith(t + "_"))
                name = f"TRM `{tag}`: {SUFFIX[k[len(tag):]]}"
            lines.append(fmt_row(name, S[k]))
        fb = [k for k in keys if k.endswith("_fb")]
        if fb:
            lines += [""] + [f"* `{k}`: safety net fired on {S[k]['fired']:.1f} % of the instances, the reduced MILP gave the "
                             f"better schedule on {S[k]['from_milp']:.1f} %." for k in fb]
        # paired comparisons
        refs = [r for r in ("lpround5", "m1x_1", "m1x_5", "rl_lag7", "milp05") if r in keys]
        ctrl = [t + "_1" for t in plan["trm"] if t != plan["final"] and t + "_1" in keys]
        lines += ["", "### Paired differences (method − reference): Δ gap pp [95 % CI] / Δ log speed-up [CI] "
                      "(> 0: method faster)", "",
                  "| method | " + " | ".join(f"vs {r}" for r in refs + ctrl) + " |", "|---|" + "---|" * len(refs + ctrl)]
        out["paired"] = {}
        for k in trm_keys:
            cells = []
            for r in refs + ctrl:
                if r == k:
                    cells.append("–")
                    continue
                dg, dl = paired(G[k], G[r]), paired(L[k], L[r])
                out["paired"][f"{k}|{r}"] = dict(gap=dg, logspeed=dl)
                cells.append(f"{dg['diff']:+.2f} [{dg['ci'][0]:+.2f}, {dg['ci'][1]:+.2f}] / "
                             f"{dl['diff']:+.2f} [{dl['ci'][0]:+.2f}, {dl['ci'][1]:+.2f}]")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        # test-time compute on test
        lines += ["", "### Test-time compute on test (1 LP after k supervision steps)", "",
                  "| model | " + " | ".join(f"k = {k}" for k in K_CURVE) + " |", "|---|" + "---|" * len(K_CURVE)]
        out["curve_test"] = {}
        for tag in plan["trm"]:
            if f"{tag}_k1" not in R[0]:
                continue
            cs = {k: stats(R, f"{tag}_k{k}")[0] for k in K_CURVE}
            out["curve_test"][tag] = cs
            lines.append(f"| {tag} | " + " | ".join(f"{cs[k]['gap_mean']:.2f} % / {cs[k]['served']:.0f} % / "
                                                     f"{cs[k]['time_mean']:.2f} s" for k in K_CURVE) + " |")
        lines.append("")
        rel = [r["relax"]["urel_diff_stored"] for r in R]
        lines.append(f"Re-solved LP relaxation vs stored: max |Δu_rel| = {max(rel):.2e}. Full MILP 0.1 %: mean "
                     f"{np.mean([r['milp01']['time'] for r in R]):.1f} s, at the 60 s limit on "
                     f"{np.mean([r['milp01']['status'] != 'optimal' for r in R]) * 100:.0f} %.")
        lines.append("")
    val_tables(lines, out)
    train_table(lines, out)
    with open(os.path.join(RES, "trm_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "trm_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
