"""Learning to Fix (faithful) vs the earlier per-generator reconstruction: wrong fixes by direction.

    python3 scripts/uc_ltfx_diag.py [--joint_check_old 60]

For every rule: wrong OFF / ON fixes per instance against the dataset MILP solution aligned inside identical-unit
groups (as in scripts/uc_ltf_diag.py), on instances with a gap > 10 % and <= 1 % (test_fresh, first 60), and the
same counts on the validation instances (where the faithful thresholds are feasible by construction).
--joint_check_old N: apply the paper's joint check (8b)-(8g) with eps = 1 % to the earlier reconstruction's kNN
thresholds (budget, tau = 1 %) on the first N validation instances (the earlier calibration set).
Output: results/uc12/ltfx_diag.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.fixpolicy import align_to_prediction  # noqa: E402
from otsl.ltfx import InstanceProblems, fix_masks  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402

ROOT, OUT = "data/generated/uc12", "results/uc12"


def wrong_counts(sysm, d, i, off, on, groups):
    yh = np.where(on, 1, 0).astype(np.int8)
    ua = align_to_prediction(sysm, d["u"][i], yh, d["u0"][i], groups)
    return int((off & (ua == 1)).sum()), int((on & (ua == 0)).sum())


def old_masks(theta, p):
    """earlier reconstruction: fix round(p) where max(p, 1 - p) >= theta_g"""
    conf = np.maximum(p, 1 - p)
    fixed = conf >= theta[None, :]
    return fixed & (p <= 0.5), fixed & (p > 0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_check_old", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    sysm = load_rts_gmlc()
    groups = sysm.identical_groups()
    tf = load(os.path.join(ROOT, "test_fresh.npz"))
    va = load(os.path.join(ROOT, "val.npz"))
    vx = load(os.path.join(ROOT, "val_extra.npz"))
    vd = {k: np.concatenate([va[k], vx[k]]) for k in ("u", "u0", "load", "avail", "sr", "obj")}
    P = np.load(os.path.join(OUT, "ltfx_probs.npz"))
    recs = [json.loads(line) for line in open(os.path.join(OUT, "ltfx_eval_test_fresh.jsonl"))]
    gap = {(r["rule"], r["i"]): ((r["obj"] if r["feasible"] else r["fb_obj"]) - tf["obj"][r["i"]]) / tf["obj"][r["i"]] * 100
           for r in recs}
    out = {"test_fresh": {}, "val": {}}
    rules = sorted({r for r, _ in gap if " ltf " in r})
    tune = {f"{json.load(open(f))['model']} ltf eps={json.load(open(f))['eps'] * 100:g}%": json.load(open(f))
            for f in [os.path.join(OUT, x) for x in os.listdir(OUT) if x.startswith("ltfx_tune_") and x.endswith(".json")]}
    print("| rule | test: n gap > 10 % (wrong OFF / ON per inst.) | test: n gap <= 1 % (OFF / ON) | test all (OFF / ON) | val all (OFF / ON) |")
    print("|---|---|---|---|---|")
    for rule in rules:
        key = rule.replace(" (not converged)", "")
        if key not in tune:
            continue
        t = tune[key]
        lo, hi = np.array(t["lo"]), np.array(t["hi"])
        model = t["model"]
        rows = []
        for i in sorted({i for (r, i) in gap if r == rule}):
            off, on = fix_masks(P[f"{model}_tf"][i], lo, hi)
            wo, wn = wrong_counts(sysm, tf, i, off, on, groups)
            rows.append(dict(i=i, gap=gap[(rule, i)], wrong_off=wo, wrong_on=wn, n_fixed=int(off.sum() + on.sum())))
        pv = np.concatenate([P[f"{model}_va"], P[f"{model}_vx"]])[:t["n_val"]]
        vrows = []
        for i in range(len(pv)):
            off, on = fix_masks(pv[i], lo, hi)
            wo, wn = wrong_counts(sysm, vd, i, off, on, groups)
            vrows.append(dict(i=i, wrong_off=wo, wrong_on=wn, n_fixed=int(off.sum() + on.sum())))
        out["test_fresh"][rule] = rows
        out["val"][rule] = vrows
        bad = [r for r in rows if r["gap"] > 10]
        good = [r for r in rows if r["gap"] <= 1]
        f = lambda rs, k: np.mean([r[k] for r in rs]) if rs else float("nan")
        print(f"| {rule} | {len(bad)} ({f(bad, 'wrong_off'):.1f} / {f(bad, 'wrong_on'):.1f}) | {len(good)} ({f(good, 'wrong_off'):.1f} / "
              f"{f(good, 'wrong_on'):.1f}) | {f(rows, 'wrong_off'):.1f} / {f(rows, 'wrong_on'):.1f} | {f(vrows, 'wrong_off'):.1f} / "
              f"{f(vrows, 'wrong_on'):.1f} |")
    old = json.load(open(os.path.join(OUT, "ltf_diag_fresh.json")))
    for rule, rows in old.items():
        if not rule.startswith("ltf:"):
            continue
        bad = [r for r in rows if r["gap"] > 10]
        good = [r for r in rows if r["gap"] <= 1]
        f = lambda rs, k: np.mean([r[k] for r in rs]) if rs else float("nan")
        print(f"| earlier reconstruction {rule} | {len(bad)} ({f(bad, 'wrong_off'):.1f} / {f(bad, 'wrong_on'):.1f}) | {len(good)} "
              f"({f(good, 'wrong_off'):.1f} / {f(good, 'wrong_on'):.1f}) | {f(rows, 'wrong_off'):.1f} / {f(rows, 'wrong_on'):.1f} | – |")
        out["test_fresh"]["earlier " + rule] = rows
    if a.joint_check_old:
        th = np.array(json.load(open(os.path.join(OUT, "ltf_thresholds.json")))["knn"]["budget:comp:0.01"]["theta"])
        pold = np.load(os.path.join(OUT, "ltf_probs.npz"))["knn_va"]
        m = UCModel(sysm, T=12, network=True)
        res = []
        for i in range(min(a.joint_check_old, len(pold))):
            off, on = old_masks(th, pold[i])
            Pi = InstanceProblems(m, scenario_from(va, i), va["obj"][i], 0.01)
            ok, info = Pi.check(off, on)
            wo, wn = wrong_counts(sysm, va, i, off, on, groups)
            res.append(dict(i=i, ok=bool(ok), status=info["status"], wrong_off=wo, wrong_on=wn, fixed=float((off | on).mean())))
        n_bad = sum(not r["ok"] for r in res)
        out["joint_check_of_earlier_knn_tau1"] = dict(n=len(res), n_violating=n_bad, rows=res)
        print(f"earlier reconstruction (kNN, budget tau = 1 %) under the joint eps = 1 % check: {n_bad} of {len(res)} val "
              f"instances violate; wrong OFF per violating instance "
              f"{np.mean([r['wrong_off'] for r in res if not r['ok']]) if n_bad else float('nan'):.1f}, per passing "
              f"{np.mean([r['wrong_off'] for r in res if r['ok']]):.1f}")
    json.dump(out, open(os.path.join(OUT, "ltfx_diag.json"), "w"), indent=1, default=float)
