"""Tables for the combined pipelines and the multi-seed confirmations -> results/uc12/combo_results.md / .json

    python scripts/uc_combo_report.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import THRESHOLDS, e2e_metrics  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from uc_combo_eval import summarize_fix  # noqa: E402

OUT = os.path.join("results", "uc12")
ROOT = os.path.join("data", "generated", "uc12")
SEEDS = (0, 1, 2)
FAM_LABEL = {"rl": "label-free + REINFORCE", "lag": "label-free + Lagrangian + KL (lag_D)", "st": "self-trained BCE",
             "lf_bce": "label-free BCE (no fine-tuning)"}
METRICS = ("served", "gap_median", "gap_mean_served", "gap_mean", "units_on")


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def fmt(a, dig=2, pct=True):
    if a["n"] == 0:
        return "–"
    u = " %" if pct else ""
    if a["n"] == 1:
        return f"{a['mean']:.{dig}f}{u}"
    return f"{a['mean']:.{dig}f}{u} [{a['min']:.{dig}f}, {a['max']:.{dig}f}]"


# ----------------------------------------------------------------------------------- end-to-end
def e2e_tables(split, res):
    js = jload(f"combo_e2e_{split}.json")
    if js is None:
        return []
    z = np.load(os.path.join(OUT, f"combo_e2e_{split}_arrays.npz"))
    arr = {k.replace("__", "|"): z[k] for k in z.files}
    d = load(os.path.join(ROOT, f"{split}.npz"))
    ref = d["obj"]
    th = js["thresholds"]
    chosen = js["chosen_family"]
    fams = {"rl": [f"rl_s{s}" for s in SEEDS], "lag": [f"lag_s{s}" for s in SEEDS], "st": [f"st_s{s}" for s in SEEDS],
            "lf_bce": ["lf_bce"]}

    def met(tag):
        a = arr[tag]
        r = e2e_metrics(a[0], a[1], a[2], ref, a[3].mean())
        r["_raw"] = a[:3]
        return r

    Bidx = np.random.default_rng(0).integers(0, len(ref), (2000, len(ref)))

    def boot(raws):
        """instance bootstrap (95 %) of the seed-averaged served share and served-instance mean gap"""
        sv, gm = [], []
        for c, sh, so in raws:
            gap = (c - ref) / ref * 100
            hard = ((sh < 1e-6) & (so < 1e-6)).astype(float)
            h = hard[Bidx]
            sv.append(h.mean(1) * 100)
            gm.append((gap[Bidx] * h).sum(1) / np.maximum(h.sum(1), 1))
        sv, gm = np.mean(sv, 0), np.mean(gm, 0)
        return {"served": [float(np.percentile(sv, 2.5)), float(np.percentile(sv, 97.5))],
                "gap_mean_served": [float(np.percentile(gm, 2.5)), float(np.percentile(gm, 97.5))]}

    def screen(name):
        tags = [f"{name}|block|{t}" for t in THRESHOLDS if f"{name}|block|{t}" in arr]
        A = np.stack([arr[t] for t in tags])                      # [k, 4, N]
        k = np.argmin(A[:, 0], 0)
        n = A.shape[2]
        b = A[k, :, np.arange(n)]                                 # [N, 4]
        # distinct schedules ~ distinct LP costs per instance (identical schedules are solved once)
        distinct = float(np.mean([len(np.unique(np.round(A[:, 0, i], 4))) for i in range(n)]))
        r = e2e_metrics(b[:, 0], b[:, 1], b[:, 2], ref, b[:, 3].mean())
        r["distinct_LPs"] = distinct
        r["_raw"] = b[:, :3].T
        return r, distinct

    rows = []

    def add(label, per_seed, lps="1", note=""):
        r = {"config": label, "LPs": lps, "note": note,
             "per_seed": [{k: v for k, v in p.items() if k != "_raw"} for p in per_seed]}
        for m in METRICS:
            r[m] = agg([p.get(m) for p in per_seed])
        r["boot95"] = boot([p["_raw"] for p in per_seed])
        rows.append(r)

    milp = met("milp|ref|0")
    add("MILP commitment (reference)", [milp], "MILP")
    for fam, ms in fams.items():
        lab = FAM_LABEL[fam]
        add(f"{lab}: top-1 + min up/down repair", [met(f"{m}|mud|0.5") for m in ms])
        add(f"{lab}: threshold 0.5 + block repair", [met(f"{m}|block|0.5") for m in ms])
        ths = [th[m] for m in ms]
        add(f"{lab}: val-calibrated threshold + block repair", [met(f"{m}|block|{th[m]}") for m in ms],
            note="thresholds " + "/".join(f"{t:g}" for t in ths))
        if fam == chosen:
            sc = [screen(m) for m in ms]
            add(f"{lab}: screening over {len(THRESHOLDS)} thresholds + block repair", [s[0] for s in sc],
                lps=f"{np.mean([s[1] for s in sc]):.1f} distinct (≤ {len(THRESHOLDS)})")
    res[f"e2e_{split}"] = {"rows": rows, "chosen_family": chosen, "thresholds": th, "n": int(len(ref))}
    lines = [f"| configuration ({split}, {len(ref)} instances) | LPs | served | median gap | mean gap, served | mean gap | units on |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']}{' (' + r['note'] + ')' if r['note'] else ''} | {r['LPs']} | {fmt(r['served'], 1)} | "
                     f"{fmt(r['gap_median'])} | {fmt(r['gap_mean_served'])} | {fmt(r['gap_mean'])} | "
                     f"{fmt(r['units_on'], 2, False)} |")
    lines += ["", "Instance bootstrap (95 %, 2,000 resamples of the instances, seed-averaged metric): ",
              "", "| configuration | served | mean gap, served |", "|---|---|---|"]
    for r in rows:
        b = r["boot95"]
        lines.append(f"| {r['config']} | {b['served'][0]:.1f}–{b['served'][1]:.1f} % | "
                     f"{b['gap_mean_served'][0]:.2f}–{b['gap_mean_served'][1]:.2f} % |")
    return lines


def e2e_paired(split, res):
    """seed-matched paired comparisons on the instances both configurations serve"""
    js = jload(f"combo_e2e_{split}.json")
    if js is None:
        return []
    z = np.load(os.path.join(OUT, f"combo_e2e_{split}_arrays.npz"))
    arr = {k.replace("__", "|"): z[k] for k in z.files}
    ref = load(os.path.join(ROOT, f"{split}.npz"))["obj"]
    th = js["thresholds"]

    def scr(name):
        tags = [f"{name}|block|{t}" for t in THRESHOLDS if f"{name}|block|{t}" in arr]
        if len(tags) < len(THRESHOLDS):
            return None
        A = np.stack([arr[t] for t in tags])
        k = np.argmin(A[:, 0], 0)
        return A[k, :, np.arange(A.shape[2])].T

    def get(fam, dec, s_):
        m = f"{fam}_s{s_}"
        if dec == "cal":
            return arr[f"{m}|block|{th[m]}"]
        if dec == "screen":
            return scr(m)
        return arr[f"{m}|block|0.5"]
    pairs = [(("lag", "0.5"), ("rl", "0.5")), (("lag", "cal"), ("rl", "cal")), (("lag", "screen"), ("rl", "cal")),
             (("lag", "cal"), ("st", "cal"))]
    out, lines = [], ["| A | B | served by both | mean gap A | mean gap B | A cheaper | A only / B only |",
                      "|---|---|---|---|---|---|---|"]
    for (fa, da), (fb, db) in pairs:
        per = []
        for s_ in SEEDS:
            a, b = get(fa, da, s_), get(fb, db, s_)
            if a is None or b is None:
                continue
            ha = (a[1] < 1e-6) & (a[2] < 1e-6)
            hb = (b[1] < 1e-6) & (b[2] < 1e-6)
            both = ha & hb
            ga, gb = (a[0] - ref) / ref * 100, (b[0] - ref) / ref * 100
            per.append(dict(both=int(both.sum()), ga=float(ga[both].mean()), gb=float(gb[both].mean()),
                            a_cheaper=int((a[0][both] < b[0][both] - 1e-6).sum()), a_only=int((ha & ~hb).sum()),
                            b_only=int((hb & ~ha).sum())))
        if not per:
            continue
        la, lb = f"{FAM_LABEL[fa]} ({da})", f"{FAM_LABEL[fb]} ({db})"
        out.append({"A": la, "B": lb, "per_seed": per})
        f_ = lambda k: "/".join(str(p[k]) if isinstance(p[k], int) else f"{p[k]:.2f}" for p in per)
        lines.append(f"| {la} | {lb} | {f_('both')} | {f_('ga')} % | {f_('gb')} % | {f_('a_cheaper')} | "
                     f"{f_('a_only')} / {f_('b_only')} |")
    res[f"e2e_paired_{split}"] = out
    return ["Seed-matched pairs (seed 0 / 1 / 2); gaps are means over the instances both serve.", ""] + lines


# ----------------------------------------------------------------------------------- fixing
FIX_LABEL = {"rac": "RACLearn (BCE confidence)", "rac+lp": "RACLearn + LP-relaxation guard",
             "harm+guard": "error-cost ranking + adequacy guard", "harm+lp": "error-cost ranking + adequacy + LP guard",
             "st asym+guard": "self-trained, asymmetric + adequacy guard", "rl rac": "LF + REINFORCE probabilities",
             "milp_rl rac": "MILP-label + REINFORCE probabilities (original model, 1 seed)"}


def fix_tables(split, res):
    path = os.path.join(OUT, f"combo_fix_{split}.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], []
    d = load(os.path.join(ROOT, f"{split}.npz"))
    rows, common = summarize_fix(path, d["obj"])
    js = {"n": len(common)}
    src = jload("combo_valfix.json")["chosen_source"]
    # instance bootstrap of the seed-averaged mean gap (95 % interval)
    recs = [json.loads(line) for line in open(path)]
    gaps = {}
    for r in recs:
        if r["i"] in set(common):
            gaps.setdefault((r["config"], r["seed"], r["ratio"]), {})[r["i"]] = (r["obj"] - d["obj"][r["i"]]) / d["obj"][r["i"]] * 100
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(common), (2000, len(common)))
    ci = {}
    for (c, s_, q), gi in gaps.items():
        ci.setdefault((c, q), []).append(np.array([gi[i] for i in common]))
    for k, gs in ci.items():
        m = np.mean([g[B].mean(1) for g in gs], 0)
        ci[k] = (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
    full = [r for r in rows if r["config"] == "full"][0]
    groups = {}
    for r in rows:
        if r["config"] == "full":
            continue
        groups.setdefault((r["config"], r["target"]), []).append(r)
    out = []
    for (c, q), rr in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if c.startswith("combo") and c.endswith("+lp"):
            lab = f"combined: {src} probabilities + error-cost + adequacy + LP guard"
        elif c.startswith("combo"):
            lab = f"combined without LP guard: {src} probabilities + error-cost + adequacy guard"
        else:
            lab = FIX_LABEL.get(c, c)
        g = {"config": c, "label": lab,
             "target": q, "seeds": [r["seed"] for r in rr]}
        for m in ("fixed_share", "gap_mean", "gap_median", "gap_max", "served", "speedup", "time_s", "n_gt1"):
            g[m] = agg([r[m] for r in rr])
        g["gap_mean_boot95"] = ci[(c, q)]
        out.append(g)
    # Pareto: best seed-averaged mean gap at a seed-averaged speed-up >= x, per pipeline
    fam = {}
    for g in out:
        key = g["label"]
        if src == "bce" and g["config"] == "harm+lp":
            fam.setdefault(f"combined: {src} probabilities + error-cost + adequacy + LP guard", []).append(g)
        fam.setdefault(key, []).append(g)
    xs = (2, 3, 5, 10, 20)
    par = {}
    for k, gs in fam.items():
        par[k] = {}
        for x in xs:
            ok = [g for g in gs if g["speedup"]["mean"] >= x]
            if ok:
                b = min(ok, key=lambda g: g["gap_mean"]["mean"])
                par[k][x] = {"gap_mean": b["gap_mean"], "target": b["target"], "speedup": b["speedup"]}
    # summary rows: best confidence-ranked (RACLearn-style) rule vs best new rule
    for lab, keys in (("best RACLearn-style (confidence ranking, ± LP guard)", [k for k in fam if k.startswith("RACLearn")]),
                      ("best new rule (error cost / self-trained / REINFORCE / combined)",
                       [k for k in fam if not k.startswith("RACLearn") and "original model" not in k])):
        par[lab] = {}
        for x in xs:
            cand = [(par[k][x], k) for k in keys if x in par[k]]
            if cand:
                b, k = min(cand, key=lambda c: c[0]["gap_mean"]["mean"])
                par[lab][x] = dict(b, rule=k)
    res[f"fix_{split}"] = {"rows": out, "full": full, "source": src, "pareto": par, "n": js["n"]}
    lines = [f"Full MILP (60 s, 0.1 %), back to back: {full['time_s']:.1f} s mean, serves {full['served']:.1f} %, "
             f"mean gap to the reference {full['gap_mean']:.3f} % ({js['n']} instances).", "",
             "| rule | target | fixed share | mean gap | (instance bootstrap 95 %) | median gap | instances > 1 % | served | speed-up | seeds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for g in out:
        lines.append(f"| {g['label']} | {g['target'] * 100:.0f} % | {fmt(g['fixed_share'], 1)} | {fmt(g['gap_mean'])} | "
                     f"{g['gap_mean_boot95'][0]:.2f}–{g['gap_mean_boot95'][1]:.2f} | {fmt(g['gap_median'], 3)} | {fmt(g['n_gt1'], 1, False)} | {fmt(g['served'], 1)} | "
                     f"{fmt(g['speedup'], 1, False)}× | {len(g['seeds'])} |")
    plines = ["| pipeline | " + " | ".join(f"≥ {x}×" for x in xs) + " |", "|---|" + "---|" * len(xs)]
    for k, v in par.items():
        cells = []
        for x in xs:
            if x in v:
                who = f"{v[x]['rule'].split(':')[0]}, " if "rule" in v[x] else ""
                cells.append(f"{v[x]['gap_mean']['mean']:.2f} % ({who}{v[x]['target'] * 100:.0f} %, {v[x]['speedup']['mean']:.1f}×)")
            else:
                cells.append("–")
        plines.append(f"| {k} | " + " | ".join(cells) + " |")
    return lines, plines



def orig_compare(res):
    """original test: the earlier single-seed per-instance records vs this study's seeds, on the instances done here"""
    path = os.path.join(OUT, "combo_fix_test.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    d = load(os.path.join(ROOT, "test.npz"))
    ref = d["obj"]
    rows, common = summarize_fix(path, ref)
    new = {}
    for line in open(path):
        r = json.loads(line)
        if r["i"] in set(common):
            new.setdefault((r["config"], r["ratio"]), {}).setdefault(r["seed"], {})[r["i"]] = r
    old = {}                                                    # (config, ratio) -> {i: (obj, time)}, plus full times
    for f, mp in (("fixpolicy_eval_test.jsonl", {("harm_c_guard", 0.9): ("harm+guard", 0.9), ("rac", 0.9): ("rac", 0.9),
                                                 ("rac", 0.95): ("rac", 0.95), ("rl", 0.95): ("milp_rl rac", 0.95)}),
                  ("fixpolicy_eval_test_bc.jsonl", {("harm_c_lp", 0.95): ("harm+lp", 0.95), ("rac_lp", 0.95): ("rac+lp", 0.95)})):
        full = {}
        recs = [json.loads(x) for x in open(os.path.join(OUT, f))]
        for r in recs:
            if r["method"] == "full":
                full[r["i"]] = r["time"]
        for r in recs:
            k = (r["method"], r["ratio"])
            if k in mp:
                old.setdefault(mp[k], {})[r["i"]] = (r["obj"], r["time"], full[r["i"]])
    st = json.load(open(os.path.join(OUT, "selftrain_fix_per_instance.json")))
    for key, cfg in (("95 %: ST round 3, asym+guard", ("st asym+guard", 0.95)),):
        old[cfg] = {i: (st[key]["obj"][i], st[key]["time"][i], st["full MILP"]["time"][i]) for i in range(len(st[key]["obj"]))}
    full_new = new[("full", 0.0)][-1]
    out, lines = [], [f"Original test, first {len(common)} instances done in this study (same instances for every column; "
                      "old = the earlier single-seed run, its own back-to-back full MILP).", "",
                      "| rule | target | old: mean gap / speed-up | this study, seed 0 | seeds 0–2: mean gap [min, max] / speed-up |",
                      "|---|---|---|---|---|"]
    for (c, q), o in sorted(old.items()):
        if (c, q) not in new:
            continue
        ids = [i for i in common if i in o]
        if not ids:
            continue
        g_old = np.mean([(o[i][0] - ref[i]) / ref[i] * 100 for i in ids])
        sp_old = np.mean([o[i][2] for i in ids]) / np.mean([o[i][1] for i in ids])
        per_seed = []
        for s_, recs in sorted(new[(c, q)].items()):
            g = np.mean([(recs[i]["obj"] - ref[i]) / ref[i] * 100 for i in ids])
            sp = np.mean([full_new[i]["time"] for i in ids]) / np.mean([recs[i]["time"] for i in ids])
            per_seed.append((s_, g, sp))
        g0 = [x for x in per_seed if x[0] == 0][0]
        gs = [x[1] for x in per_seed]
        sps = [x[2] for x in per_seed]
        out.append(dict(config=c, target=q, n=len(ids), old_gap=g_old, old_speedup=sp_old, seeds=per_seed))
        lines.append(f"| {FIX_LABEL.get(c, c)} | {q * 100:.0f} % | {g_old:.2f} % / {sp_old:.1f}× ({len(ids)} inst.) | "
                     f"{g0[1]:.2f} % / {g0[2]:.1f}× | {np.mean(gs):.2f} % [{min(gs):.2f}, {max(gs):.2f}] / {np.mean(sps):.1f}× |")
    res["fix_test_vs_old"] = out
    return lines


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    res = {}
    md = ["# Combined pipelines and 3-seed confirmation (B2, 12-hour UC)", "",
          "Generated by `scripts/uc_combo_report.py`. Seeds 0, 1, 2 for every trainable part; cells are "
          "mean [min, max] over seeds (single value: one model). Gaps are to the MILP reference objective of each "
          "instance; served = no shedding / spill and no reserve shortfall.", ""]
    cal = jload("combo_calib.json")
    if cal:
        res["calib"] = {k: v for k, v in cal.items() if k != "rows"}
        md += ["## Validation decisions (made before any test set was read)", "",
               f"* Decision thresholds (block decoder, argmin val LP cost / LP-relaxation cost): " +
               ", ".join(f"{k} {v['threshold']:g}" for k, v in cal["thresholds"].items()),
               f"* Family criterion (seed-averaged val cost / relaxation cost): " +
               ", ".join(f"{k} {v:.4f}" for k, v in cal["family_crit"].items()) + f" -> **{cal['chosen_family']}**"]
    vf = jload("combo_valfix.json")
    if vf:
        res["valfix"] = vf
        md.append(f"* Fixing probability source ({vf['n']} val instances, 95 % target, harm seed 0, both guards): " +
                  ", ".join(f"{r['config']} {r['gap_mean']:.3f} % at {r['speedup']:.1f}×" for r in vf["rows"] if r["config"] != "full")
                  + f" -> **{vf['chosen_source']}**")
    md.append("")
    for split in ("test_fresh", "test"):
        lines = e2e_tables(split, res)
        if lines:
            md += [f"## End-to-end, {split}", ""] + lines + [""] + e2e_paired(split, res) + [""]
    for split in ("test_fresh", "test"):
        lines, plines = fix_tables(split, res)
        if lines:
            md += [f"## Fixing + reduced MILP, {split} (first {res['fix_' + split]['n']} instances)", ""] + lines + [""]
            md += ["### Best seed-averaged mean gap at a seed-averaged speed-up of at least x (target, speed-up)", ""] + plines + [""]
    ol = orig_compare(res)
    if ol:
        md += ["### Original test: earlier single-seed runs vs this study on the same instances", ""] + ol + [""]
    tr = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("combo_train_") and f.endswith(".json"):
            j = json.load(open(os.path.join(OUT, f)))
            tr[f[len("combo_train_"):-5]] = {k: v for k, v in j.items() if k != "hist"}
    res["train"] = tr
    md += ["## Training runs", "", "| run | wall s | LPs | reproduction check |", "|---|---|---|---|"]
    for k, v in tr.items():
        rp = v.get("repro_max_abs_diff_val", v.get("repro_max_rel_diff_val", ""))
        md.append(f"| {k} | {v.get('train_s', 0):.0f} | {v.get('train_LPs', '')} | {rp} |")
    with open(os.path.join(OUT, "combo_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "combo_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\n".join(md))
