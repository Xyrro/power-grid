"""Zero-shot extrapolation of the TRM study (arm A, stretch): models trained on 12-hour windows at the original line
ratings, run without retraining on 24-hour windows (uc24 test, 40 instances) and on the congested benchmark (RTS-GMLC
12 h, line ratings x 0.6, docs/methods/hard.md; test, 60 instances), with more supervision steps.

    python3 scripts/rec_trm_stretch.py --bench uc24 --tags main oneshot
    python3 scripts/rec_trm_stretch.py --bench hard --tags main oneshot

Every model keeps its 12-hour validation choices (decoder, threshold; trm_select.json or its trm_val_<tag>.json);
nothing is tuned on these sets. Pipelines: 1 LP + LP repair after k supervision steps (k = 1 .. K), and the model's
schedule + 4 LP roundings (+ repair); baseline LP rounding at 5 thresholds + repair. The reference is the stored full
MILP of the dataset run (uc24: 300 s limit; hard: 600 s), gap (C - DB) / C to its stored dual bound; the MILP was not
re-timed here, so its stored time is quoted for scale only (no paired speed-up).
Output: results/recursive/trm_stretch_<bench>.json (+ jsonl records).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl import trm  # noqa: E402
from otsl.base import lp_round_candidates  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

RES = os.path.join("results", "recursive")
OUT = os.path.join("data", "generated", "recursive")
LPR = (0.001, 0.05, 0.2, 0.5, 0.8)


def bench(name):
    if name == "uc24":
        d = trm.load_npz(os.path.join("data", "generated", "uc24", "test.npz"))
        return d, load_rts_gmlc(), 24, (1, 4, 8, 16, 32, 48)
    from otsl.hard import LINE_SCALE, load_split
    return load_split("test"), load_rts_gmlc(line_scale=LINE_SCALE), 12, (1, 4, 8, 16, 32)


def finish(m, sc, sysm, us, rep=True):
    best, n = None, 0
    seen = set()
    for u in us:
        if u.tobytes() in seen:
            continue
        seen.add(u.tobytes())
        sol = m.solve_dispatch(sc, u)
        n += 1
        if best is None or sol.obj < best[1].obj:
            best = (u, sol)
    u, sol = best
    if rep and (sol.shed > 1e-6 or sol.short > 1e-6):
        u, sol, k, _ = trm.lp_repair(m, sc, u, sysm, sol)
        n += k
    return dict(cost=float(sol.obj), shed=float(sol.shed), short=float(sol.short), n_lp=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["uc24", "hard"])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    d, sysm, T, ks = bench(a.bench)
    N = a.n or len(d["load"])
    m = UCModel(sysm, T=T)
    base_sys = load_rts_gmlc()
    plan = json.load(open(os.path.join(RES, "trm_select.json"))) if os.path.exists(os.path.join(RES, "trm_select.json")) else {}
    models = {}
    for tag in a.tags:
        model, feats, cfg, _ = trm.load_model(os.path.join(OUT, "models", f"{tag}.pt"), base_sys)
        fh = trm.TRMFeatures.from_state(sysm, feats.state())          # ratings of this benchmark in the inputs
        sel = plan.get("trm", {}).get(tag, {}).get("rep") or json.load(open(os.path.join(RES, f"trm_val_{tag}.json")))["selection"]["rep"]
        models[tag] = (model, fh, cfg, sel)
    path = os.path.join(RES, f"trm_stretch_{a.bench}.jsonl")
    done = {json.loads(line)["i"] for line in open(path)} if os.path.exists(path) else set()
    t0 = time.time()
    for i in range(N):
        if i in done:
            continue
        sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        d1 = {k: d[k][i:i + 1] for k in ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel")}
        rec = dict(i=i, bound=float(d["bound"][i]), milp_obj=float(d["obj"][i]), milp_time=float(d["time"][i]))
        tt = time.time()
        cands = lp_round_candidates(d["u_rel"][i], sc, sysm, LPR)
        rounds = {float(t): u for t, u in cands}
        rec["lpround5r"] = finish(m, sc, sysm, [u for _, u in cands])
        rec["lpround5r"]["time"] = time.time() - tt
        for tag, (model, fh, cfg, sel) in models.items():
            sg = trm.subgroups(sysm, d1["u0"])[0] if cfg.sym == "count" else None
            xt, xb = fh(d1)
            kmax = max(ks) if cfg.mode == "trm" else 1
            tt = time.time()
            P, _ = trm.run_steps(model, xt, xb, kmax, bs=1)
            t_inf = (time.time() - tt) / (kmax if cfg.mode == "trm" else 1)
            for k in (ks if cfg.mode == "trm" else (1,)):
                u = trm.decode(P[0, k - 1], d1, 0, sysm, sel["th"], sel["decoder"], sg)
                rec[f"{tag}_k{k}r"] = finish(m, sc, sysm, [u])
                if sel.get("mix_th"):
                    rec[f"{tag}_k{k}_mix5r"] = finish(m, sc, sysm, [u] + [rounds[t] for t in sel["mix_th"]])
            rec[f"{tag}_infer_s_per_step"] = t_inf
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{N} {time.time() - t0:.0f}s]", flush=True)
    R = [json.loads(line) for line in open(path)]
    out = dict(bench=a.bench, n=len(R), steps=ks, milp_time_mean=float(np.mean([r["milp_time"] for r in R])), methods={})
    keys = [k for k in R[0] if isinstance(R[0][k], dict)]
    db = np.array([r["bound"] for r in R])
    for k in keys:
        c = np.array([r[k]["cost"] for r in R])
        g = (c - db) / c * 100
        sv = np.array([r[k]["shed"] < 1e-6 and r[k]["short"] < 1e-6 for r in R])
        out["methods"][k] = dict(gap_mean=float(g.mean()), gap_median=float(np.median(g)), gap_max=float(g.max()),
                                 served=float(sv.mean() * 100), lps=float(np.mean([r[k]["n_lp"] for r in R])))
    gm = np.array([(r["milp_obj"] - r["bound"]) / r["milp_obj"] * 100 for r in R])
    out["methods"]["milp_stored"] = dict(gap_mean=float(gm.mean()), gap_median=float(np.median(gm)), gap_max=float(gm.max()))
    with open(os.path.join(RES, f"trm_stretch_{a.bench}.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["methods"].items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
