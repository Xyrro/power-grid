"""Exploration on validation instances for round 8 (docs/methods/agg.md): any mix of methods on 12-hour validation
instances, every method of an instance back to back in one single-threaded process (paired times). Validation only.

    OTSL_THREADS=1 python3 scripts/agg_probe.py --ks 0-60:3 --methods ref,afull:0.5,sand:s0:0.02 --out results/agg/explore/x.jsonl
    (on PACE: scripts/pace/pgu_agg_probe.sbatch runs one worker per allocated core with --claims)

Methods:
  ref                  the full MILP on the B-theta network at 0.1 % (the speed-up reference)
  full:<g>             the same at g %
  sfull:<g> / afull:<g>   the screened full MILP, per unit / with identical units aggregated, at g %
  cert:<pred>:<d>      round 7's certified pipeline (per unit), threshold d
  acert:<pred>:<d>[:<dc>]   the certified pipeline on the aggregated model (dc > 0: count fixings too)
  acertx:<pred>:<d>     acert with the exact fixings of Lagrangian probing in the proof step (otsl.probe)
  acerth:<pred>:<d>     acert with the solver's primal heuristics off in the proof step (the candidate is the incumbent)
  alad:<pred>:<d>:<d2>[:x]   the ladder: when the LP bound does not certify, a second reduced MILP with threshold d2
                        before the proof step (x: probing fixings in the proof)
  sand:<pred>:<d>[:lpf]     the sandwich certificate (lpf: keep integrality where the LP relaxation is fractional)
<pred>: s0 (round 7's network), ens5 (the five networks averaged), lp (the LP relaxation's values), knn (k = 50, as
Learning to Fix), lag<tau> (Lagrangian probing: fix where the margin is at least tau x z_LP; use d < 0.5), hyb<tau>
(s0, minus the fixings that the probing schedule contradicts with a margin of at least tau x z_LP), uni<tau> (s0, with
every decision whose margin is at least tau x z_LP set to the probing schedule's state).
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
sys.path.insert(0, HERE)
from screen_eval import instance  # noqa: E402
from otsl.agg import AggScreenedUC, agg_certified_pipeline, sandwich_pipeline  # noqa: E402
from otsl.ood import OODModel, override_probs, solve_milp  # noqa: E402
from otsl.probe import hourly_on_cost, unit_margins  # noqa: E402
from otsl.screen import ScreenedUC, certified_pipeline, monitored_from_lp  # noqa: E402


def parse_ks(s):
    if ":" in s or "-" in s:
        rng, _, step = s.partition(":")
        a, b = map(int, rng.split("-"))
        return list(range(a, b, int(step or 1)))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="val")
    ap.add_argument("--claims", default="", help="claim folder shared by workers (each instance runs once)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    methods = a.methods.split(",")
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["k"] for line in open(a.out)}
    nets, knn = None, None
    for k in parse_ks(a.ks):
        if k in done:
            continue
        if a.claims:
            os.makedirs(a.claims, exist_ok=True)
            try:
                os.close(os.open(os.path.join(a.claims, f"{a.set}_{k}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        sysm, T, tl, sc, _ = instance("uc12", a.set, k)
        if nets is None:
            from uc_m1x_train import featurizer, load_net
            feat = featurizer(sysm)
            need = {m.split(":")[1] for m in methods if m.split(":")[0] in ("cert", "acert", "acertx", "acerth", "alad", "sand")}
            need |= {"s0"} if any(x.startswith(("hyb", "uni")) for x in need) else set()
            nets = [load_net(f"results/uc12/m1x_pol_n2000_gnnt_s{s}.pt", "gnnt", sysm, feat)
                    for s in range(5 if "ens5" in need else 1)]
            if "knn" in need:
                from otsl.b3 import load_b3
                from otsl.ltfx import KNNProb
                knn = KNNProb(sysm, load_b3("data/generated/uc12/train.npz"), k=50)
        m = OODModel(sysm, T=T)
        scr, ag = ScreenedUC(sysm, T=T), AggScreenedUC(sysm, T=T)
        g_out = tuple(getattr(sc, "g_out", ()) or ())
        m.solve_dispatch(sc, None, relax=True)
        t0 = time.perf_counter()
        rel = m.solve_dispatch(sc, None, relax=True)
        t_rel = time.perf_counter() - t0
        di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None],
                  start=np.array([getattr(sc, "start", 0)]), u_rel=rel.u[None], lmp_rel=rel.lmp[None],
                  flow_rel=rel.flow[None])
        for n in nets:
            n.predict(di)
        if knn is not None:
            knn.predict(di)
        probs, t_inf = {}, {}
        for pred in ("s0", "ens5", "knn"):
            if pred == "ens5" and len(nets) < 5 or pred == "knn" and knn is None:
                continue
            t0 = time.perf_counter()
            if pred == "knn":
                q = knn.predict(di)[0]
            else:
                q = np.mean([n.predict(di)[0].astype(np.float64) for n in (nets if pred == "ens5" else nets[:1])], 0)
            probs[pred] = override_probs(q, g_out)
            t_inf[pred] = time.perf_counter() - t0
        probs["lp"], t_inf["lp"] = np.clip(rel.u, 0.0, 1.0), 0.0
        lag_preds = sorted({x.split(":")[1] for x in methods if x.split(":")[1:2] and
                            (x.split(":")[1].startswith(("lag", "hyb", "uni")))})
        if lag_preds:
            t0 = time.perf_counter()
            on = hourly_on_cost(sysm, rel.lmp, rel.mu)
            _, sched, margin = unit_margins(sysm, sc.u0, on, g_out)
            t_lag = time.perf_counter() - t0
            for pred in lag_preds:
                tau = float(pred[3:]) * rel.obj
                if pred.startswith("lag"):        # fix where the margin exceeds tau (p = 0.5 elsewhere: never fixed)
                    probs[pred] = np.where(margin >= tau, sched, 0.5).astype(float)
                    t_inf[pred] = t_lag
                elif pred.startswith("uni"):      # the network's p, overridden by the probing schedule where the margin
                    q = probs["s0"].copy()        # is at least tau (economically robust decisions fixed too)
                    big = margin >= tau
                    q[big] = sched[big]
                    probs[pred], t_inf[pred] = q, t_lag + t_inf["s0"]
                else:                             # the network's p, but 0.5 where the economics contradict it
                    q = probs["s0"].copy()
                    q[(np.round(q) != sched) & (margin >= tau)] = 0.5
                    probs[pred], t_inf[pred] = q, t_lag + t_inf["s0"]
        mon = monitored_from_lp(rel.flow, sysm.fmax)
        rec = dict(k=k, set=a.set, zlp=rel.obj, t_rel=t_rel, runs={})
        order = list(methods)
        np.random.default_rng(k).shuffle(order)
        for meth in order:
            parts = meth.split(":")
            t0 = time.perf_counter()
            info = {}
            if parts[0] in ("ref", "full"):
                g = 1e-3 if parts[0] == "ref" else float(parts[1]) / 100
                r = solve_milp(m, sc, {}, tl, g)
                u, info = r.get("u"), dict(bound=r["bound"], status=r["status"])
                dt = time.perf_counter() - t0
            elif parts[0] in ("sfull", "afull"):
                z = (scr if parts[0] == "sfull" else ag).solve(sc, mon, None, float(parts[1]) / 100, tl)
                u, info = z["u"], dict(bound=z["bound"], status=z["status"], rounds=z["rounds"])
                dt = time.perf_counter() - t0 + t_rel
            else:
                pred, dl = parts[1], float(parts[2])
                p = probs[pred]
                if parts[0] == "cert":
                    u, info = certified_pipeline(scr, sc, p, rel, dl, 5e-3, None, tl)
                elif parts[0] == "alad":            # alad:<pred>:<d>:<d2>[:x]  the ladder, probing in the proof with x
                    u, info = agg_certified_pipeline(ag, sc, p, rel, dl, 0.0, 5e-3, None, tl, counts=False,
                                                     probe=len(parts) > 4, ladder=float(parts[3]))
                elif parts[0] in ("acert", "acertx", "acerth"):
                    dc = float(parts[3]) if len(parts) > 3 else 0.0
                    po = dict(mip_heuristic_effort=0.0, mip_heuristic_run_rins=False, mip_heuristic_run_rens=False,
                              mip_heuristic_run_root_reduced_cost=False, mip_heuristic_run_feasibility_jump=False)
                    u, info = agg_certified_pipeline(ag, sc, p, rel, dl, dc, 5e-3, None, tl, counts=dc > 0,
                                                     probe=parts[0] == "acertx",
                                                     proof_opts=po if parts[0] == "acerth" else None)
                else:
                    u, info = sandwich_pipeline(ag, sc, p, rel, dl, 5e-3, tl, keep_lp_frac=len(parts) > 3)
                dt = time.perf_counter() - t0 + t_rel + t_inf[pred]
                ured = info.pop("u_red", None)
                if ured is not None:
                    pr = m.solve_dispatch(sc, ured)
                    info.update(red_cost=pr.obj, red_served=bool(pr.shed < 1e-6 and pr.short < 1e-6))
            pr = m.solve_dispatch(sc, u)
            rec["runs"][meth] = dict(time=dt, cost=pr.obj, served=bool(pr.shed < 1e-6 and pr.short < 1e-6),
                                     **{x: y for x, y in info.items() if not isinstance(y, (set, np.ndarray, dict))})
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(k, {x: round(y["time"], 1) for x, y in rec["runs"].items()}, flush=True)


if __name__ == "__main__":
    main()
