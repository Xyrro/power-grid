"""Cost-aware check study, separate arm: shifted *validation* sets for recalibration.

    OTSL_THREADS=1 taskset -c 0 python3 scripts/uc_cguard_shiftval.py --n 20          # build + solve (resumable)
    python3 scripts/uc_cguard_shiftval.py --select                                    # threshold of arm B

The six shifts of the robustness study (otsl.ood, specifications of data/generated/uc12_ood/specs.json: load x 1.15 /
x 0.85, wind + solar x 1.5, 2-3 running units out, 1-2 N-1-screened lines out, windows across midnight), drawn on
*validation* calendar days (day % 5 == 2; never a test or training day) with new seeds (401-406). Per instance: LP
relaxation, the pipeline's fixings and reduced MILP at 0.5 %, the full MILP at 0.5 % (the comparison of the study and
the C3 fallback), the bound-check value g = (C_red - z_LP) / C_red. Output: data/generated/uc12_cguard/shiftval/<shift>/
<k>.npz (scenario) and results/uc12/cguard_shiftval.jsonl.

Arm B rule (fixed before this pass): d_post_B = the largest grid value that fires on every shifted-validation instance
where the pipeline is more than 1 pp above the full MILP at 0.5 %, subject to firing on at most 10 % of the 360 original
validation instances; if none qualifies, the smallest grid value with at most 10 % there. results/uc12/cguard_select_B.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uc_cguard_common import BHG, RES, SHIFTS, timed  # noqa: E402
from otsl.cguard import bound_gap, solve_with_start  # noqa: E402
from otsl.ood import OODModel, build_shift_scenario, fixed_share, scenario_to_dict  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402

OUTD = "data/generated/uc12_cguard/shiftval"
GRID_POST = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)
DAYS = np.arange(366)
VAL_DAYS = DAYS[DAYS % 5 == 2]
SEEDS = dict(zip(SHIFTS, range(401, 407)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    path = os.path.join(RES, "cguard_shiftval.jsonl")
    if a.select:
        S = [json.loads(l) for l in open(path)]
        V = [json.loads(l) for l in open(os.path.join(RES, "cguard_val.jsonl"))]
        bad = [s for s in S if s["excess_pp"] > 1.0]
        rate = lambda d: float(np.mean([v["g"] > d for v in V]))
        ok = [d for d in GRID_POST if rate(d) <= 0.10 and all(b["g"] > d for b in bad)]
        d_b = max(ok) if ok else min(d for d in GRID_POST if rate(d) <= 0.10)
        out = dict(d_post_B=d_b, n_shiftval=len(S), n_bad=len(bad), bad=[(b["set"], b["k"], b["excess_pp"], b["g"]) for b in bad],
                   rate_val={str(d): rate(d) for d in GRID_POST},
                   caught={str(d): int(sum(b["g"] > d for b in bad)) for d in GRID_POST},
                   fire_shiftval={str(d): float(np.mean([s["g"] > d for s in S])) for d in GRID_POST},
                   rule="largest grid value firing on every shifted-validation instance with pipeline - full MILP 0.5 % > 1 pp, "
                        "subject to <= 10 % firing on the original validation set; else the smallest value with <= 10 %")
        json.dump(out, open(os.path.join(RES, "cguard_select_B.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        sys.exit(0)
    specs = json.load(open("data/generated/uc12_ood/specs.json"))
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    bhg = BHG(sysm)
    done = {(r["set"], r["k"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    jobs = [(s, k) for k in range(a.n) for s in SHIFTS if (s, k) not in done]
    warm = False
    for n, (s, k) in enumerate(jobs):
        sp = specs["shifts"][s]
        s0, s1 = sp["starts"]
        rng = np.random.default_rng([SEEDS[s], k])
        days = VAL_DAYS[VAL_DAYS + 1 < 366] if s1 + 12 > 24 else VAL_DAYS
        job = dict(day=int(rng.choice(days)), start=int(rng.integers(s0, s1 + 1)), seed=int(rng.integers(1 << 31)))
        sc = build_shift_scenario(sysm, m, sp["kw"], job, specs.get("line_pool"))
        os.makedirs(os.path.join(OUTD, s), exist_ok=True)
        np.savez_compressed(os.path.join(OUTD, s, f"{k:03d}.npz"), **scenario_to_dict(sc))
        if not warm:
            bhg.warm(m, sc)
            warm = True
        rel, relax_s = timed(m.solve_dispatch, sc, None, relax=True)
        _, fix, info = bhg.run(m, sc, rel)
        red = solve_with_start(m, sc, fix, 60.0, 5e-3)
        full5 = solve_with_start(m, sc, {}, 60.0, 5e-3)
        g = float(bound_gap(red["obj"], rel.obj))
        rec = dict(set=s, k=k, **job, g_out=list(sc.g_out), l_out=list(sc.l_out), z_lp=float(rel.obj), g=g,
                   fixed=fixed_share(fix, sc.g_out, 12, sysm.G),
                   pipe={kk: v for kk, v in red.items() if kk not in ("x", "u")},
                   full5={kk: v for kk, v in full5.items() if kk not in ("x", "u")},
                   excess_pp=float((red["obj"] - full5["obj"]) / red["obj"] * 100))
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[{n + 1}/{len(jobs)}] {s} {k} g {g * 100:.2f}% pipeline - full5 {rec['excess_pp']:.2f} pp "
              f"t {red['time']:.1f} / {full5['time']:.1f}s", flush=True)
