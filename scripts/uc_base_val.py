"""No-learning baselines, validation: tolerance of the LP-integral fixing rule and the LP-rounding baseline on the
validation instances (full MILPs known), for the 12-hour (uc12) or the 24-hour (uc24) benchmark.

    python3 scripts/uc_base_val.py --bench uc12 [--highs_path <dir with highspy 1.12>]
    python3 scripts/uc_base_val.py --bench uc24
    python3 scripts/uc_base_val.py --bench uc12 --select        # writes results/<bench>/base_val_select.json

Per validation instance (one process, one core): the reduced MILP of "fix every decision whose LP-relaxation value
is within tol of 0 / 1" for every tol of --tols, without and with our guards (otsl.hybrid.apply_guards: adequacy,
min up/down rows, LP-relaxation guard), with the test settings (uc12: 60 s, 0.1 %; uc24: 300 s, 0.1 %); and the
LP-rounding end-to-end baseline (5 thresholds, block repair, dispatch LPs). Records:
results/<bench>/base_val.jsonl (one line per instance; resumable).

Selection rule (fixed before any run), per guard setting: fewest infeasible reduced MILPs, then the lowest mean
gap to the validation MILP objective C* over feasible instances; ties within 0.02 pp go to the larger fixed share.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="uc12", choices=["uc12", "uc24"])
ap.add_argument("--tols", default="1e-6,0.05,0.2")
ap.add_argument("--n", type=int, default=10 ** 6)
ap.add_argument("--highs_path", default="", help="directory prepended to sys.path before highspy is imported")
ap.add_argument("--select", action="store_true")
a = ap.parse_args()
if a.highs_path:
    sys.path.insert(0, a.highs_path)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from otsl.b3 import load_b3  # noqa: E402
from otsl.base import lp_integral_fixings, lp_round_screen, solve_reduced  # noqa: E402
from otsl.hybrid import ALL_GUARDS, apply_guards  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from otsl.ucdata import scenario_from  # noqa: E402

CFG = {"uc12": dict(T=12, val="data/generated/uc12/val.npz", tl=60.0, gap=1e-3),
       "uc24": dict(T=24, val="data/generated/uc24/uc24ltf_val.npz", tl=300.0, gap=1e-3)}


def select(path, tols):
    recs = [json.loads(x) for x in open(path)]
    out = {}
    for gd in ("none", "guards"):
        rows = []
        for tol in tols:
            k = f"lpint_{tol:g}_{gd}"
            R = [r[k] for r in recs]
            feas = np.array([x["feasible"] for x in R])
            gap = np.array([(x["obj"] - r["c_star"]) / r["c_star"] * 100 for x, r in zip(R, recs) if x["feasible"]])
            rows.append(dict(tol=tol, rule=k, n=len(R), n_infeasible=int((~feas).sum()),
                             gap_mean=float(gap.mean()) if len(gap) else np.inf,
                             gap_max=float(gap.max()) if len(gap) else np.inf,
                             fixed=float(np.mean([x["n_fixed"] for x in R]) / recs[0]["TG"] * 100),
                             time_mean=float(np.mean([x["time"] + x.get("t_guard", 0.0) for x in R]))))
        best = sorted(rows, key=lambda r: (r["n_infeasible"], r["gap_mean"]))[0]
        tied = [r for r in rows if r["n_infeasible"] == best["n_infeasible"] and r["gap_mean"] <= best["gap_mean"] + 0.02]
        pick = max(tied, key=lambda r: r["fixed"])
        out[gd] = dict(selected_tol=pick["tol"], rows=rows)
    e2e = np.array([(r["lpround"]["cost"] - r["c_star"]) / r["c_star"] * 100 for r in recs])
    served = np.array([r["lpround"]["shed"] < 1e-6 and r["lpround"]["short"] < 1e-6 for r in recs])
    out["lpround"] = dict(gap_mean=float(e2e.mean()), gap_median=float(np.median(e2e)), served=float(served.mean() * 100))
    out["rule"] = ("per guard setting: fewest infeasible reduced MILPs, then the lowest mean gap to C* over feasible "
                   "instances; ties within 0.02 pp -> larger fixed share")
    out["n_val"] = len(recs)
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(HERE))
    cfg = CFG[a.bench]
    tols = [float(x) for x in a.tols.split(",")]
    path = os.path.join("results", a.bench, "base_val.jsonl")
    if a.select:
        res = select(path, tols)
        json.dump(res, open(os.path.join("results", a.bench, "base_val_select.json"), "w"), indent=1)
        print(json.dumps({k: (v["selected_tol"] if isinstance(v, dict) and "selected_tol" in v else v)
                          for k, v in res.items() if k != "rule"}, indent=1))
        sys.exit(0)
    import highspy
    print("highspy", highspy.Highs().version(), flush=True)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=cfg["T"], network=True)
    d = load_b3(cfg["val"])
    n = min(a.n, len(d["obj"]))
    TG = cfg["T"] * sysm.G
    done = {json.loads(x)["i"] for x in open(path)} if os.path.exists(path) else set()
    t_start = time.time()
    for i in range(n):
        if i in done:
            continue
        sc = scenario_from(d, i)
        u_rel = d["u_rel"][i]
        rec = dict(i=i, c_star=float(d["obj"][i]), c_rel=float(d["c_rel"][i]), TG=TG)
        for tol in tols:
            fix0 = lp_integral_fixings(u_rel, tol)
            for gd in ("none", "guards"):
                fix, info = (fix0, {"guard_s": 0.0}) if gd == "none" else apply_guards(m, sysm, sc, fix0, ALL_GUARDS)
                r = solve_reduced(m, sc, fix, cfg["tl"], cfg["gap"])
                r.pop("trace")
                r.update(t_guard=float(info["guard_s"]), n_fixed_pre=len(fix0),
                         **{k: v for k, v in info.items() if k.startswith("released")})
                rec[f"lpint_{tol:g}_{gd}"] = r
        e = lp_round_screen(m, sysm, sc, u_rel)
        rec["lpround"] = dict(cost=e["cost"], shed=e["shed"], short=e["short"], th=e["th"], n_lp=e["n_lp"],
                              t_repair=e["t_repair"], t_lp=e["t_lp"], costs=e["costs"])
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        def short(v):
            g = "INF" if not v["feasible"] else f"{(v['obj'] - rec['c_star']) / rec['c_star'] * 100:.2f}%"
            return f"{g}/{v['time']:.1f}s"
        msg = " ".join(f"{k[6:]}:{short(v)}" for k, v in rec.items() if k.startswith("lpint"))
        print(f"[{i + 1}/{n} {time.time() - t_start:.0f}s] {msg} | lpround {(e['cost'] - rec['c_star']) / rec['c_star'] * 100:.2f}%",
              flush=True)
    print("done", flush=True)
