"""Round 8 (docs/methods/agg.md): identical units aggregated, Lagrangian probing and the certified pipelines built on them,
against the full MILP, round 7's pipelines and Learning to Fix, every method of an instance back to back in one
single-threaded process (paired speed-ups, as scripts/screen_eval.py).

    OTSL_THREADS=1 python3 scripts/agg_eval.py --bench uc12 --set val --k 7 --out results/agg/val/uc12_x.jsonl
    OTSL_THREADS=1 python3 scripts/agg_eval.py --bench uc12 --set tf --k 7 --acert s0:0.02 --out results/agg/test/x.jsonl

Instances as scripts/screen_eval.py: uc12 val (360 validation instances), tf (test_fresh) and the six shifted sets,
uc24 (40 test instances), hard (60 congested test instances). The LP relaxation of the full network is solved once
(timed) and charged to every method that reads it.

Methods (names as in the records):
  full@g            the full MILP on the B-theta network at gap g (full@0.1% is the speed-up reference)
  scr:full@g        round 7's screened network (copper plate + lazily added line limits), per unit
  agg:full@g        the screened network with identical units aggregated into integer counts (otsl.agg; exact)
  cert:s0:0.02      round 7's certified pipeline (per unit; its validation choice), uc12
  cert:lp:0.05      round 7's learning-free certified pipeline (LP values), every bench
  ltf_knn_1         Learning to Fix as published (Fritz et al. 2026): kNN (k = 50) probabilities, per-generator thresholds
                    tuned by its Benders procedure at eps = 1 % on the validation set (step 2's tuning), reduced MILP on
                    the B-theta network at 0.1 % (step 2's ltf_knn_1), uc12
  acert:<pred>:<d>  the certified pipeline on the aggregated model: fixings min(p, 1 - p) <= d, rows min up/down cannot
                    complete released -> reduced aggregated MILP on the screened network at 0.5 % -> certificate: the LP
                    bound within 0.5 %, else the aggregated screened full MILP at 0.5 % warm-started from the reduced
                    schedule; flags after <pred>: +x the exact fixings of Lagrangian probing (otsl.probe) in the proof
                    step, +r the reduced MILP at 0.1 % instead of 0.5 %, +f<e> cheapest candidates first: <pred>'s and
                    the LP values' roundings (0.5, 0.2; block adequacy and min up/down repaired, one dispatch LP each),
                    the cheapest delivered when the LP bound proves it within e % (learning-free preds: the LP's only)
  <pred>: s0 (round 7's network), ens5 (the five networks averaged), lp (LP values), knn (k = 50, as Learning to Fix),
          lag<tau> (Lagrangian probing: decisions whose margin at the LP prices is at least tau x z_LP, no learning),
          m0 / mens5 (round 9: the temporal GNN retrained with the probing margins as an input feature, one / five seeds)
The reduced schedule of every certified pipeline is priced too (not timed): the same run gives the pipeline without
its certificate (confidence fixing alone, RACLearn's rule) and with an LP certificate at a looser tolerance.
Every delivered schedule is priced by the exact dispatch LP on the full network (not timed).
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from screen_eval import Clock, host_info, instance, price  # noqa: E402
from otsl.agg import AggScreenedUC, agg_certified_pipeline  # noqa: E402
from otsl.constrained import adequacy_repair_blocks  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.ood import OODModel, avail_view, force_out, override_probs, solve_milp  # noqa: E402
from otsl.probe import hourly_on_cost, margin_feature, unit_margins  # noqa: E402
from otsl.screen import ScreenedUC, certified_pipeline, monitored_from_lp  # noqa: E402
from otsl.uc import repair_min_updown  # noqa: E402

EPS = 5e-3
G01 = 1e-3


def load_thr(path):
    j = json.load(open(path))
    return np.array(j["lo"]), np.array(j["hi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=("uc12", "uc24", "hard"))
    ap.add_argument("--set", default="tf")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--acert", default="s0:0.02,s0+x:0.02,s0+x+r:0.02,ens5:0.05,ens5+x:0.05,lag0.0003:0.4,"
                                          "lag0.0003+x:0.4,lag0.0003+x+r:0.4,lag0.0001+x:0.4,lag0.001+x:0.4,lp+x:0.05,knn+x:0.05",
                    help="certified pipelines <pred>:<delta> on the aggregated model")
    ap.add_argument("--methods", default="", help="comma list (default: every method of the bench and set)")
    ap.add_argument("--ltf", default="results/unified/retime/tune/ltfx_tune_knn_1.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    if a.bench != "uc12":
        a.set = "tf"
    key = dict(bench=a.bench, set=a.set, k=a.k)
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if (r["bench"], r["set"], r["k"]) == (a.bench, a.set, a.k):
                print(f"{key} already in {a.out}", flush=True)
                return
    sysm, T, tl, sc, stored_c_rel = instance(a.bench, a.set, a.k)
    m = OODModel(sysm, T=T)
    scr, ag = ScreenedUC(sysm, T=T), AggScreenedUC(sysm, T=T)
    g_out = tuple(getattr(sc, "g_out", ()) or ())
    sv = avail_view(sysm, g_out)                      # repairs count only the available fleet
    acerts = [c.split(":") for c in a.acert.split(",") if c]
    learned = a.bench == "uc12"
    if not learned:                                   # no learned model of uc24 / hard was kept
        acerts = [c for c in acerts if c[0].split("+")[0] in ("lp",) or c[0].startswith("lag")]
    nets, knn, thr_ltf, nets_mg = [], None, None, []
    if learned:
        from uc_m1x_train import featurizer, load_net
        feat = featurizer(sysm)
        nets = [load_net(f"results/uc12/m1x_pol_n2000_gnnt_s{s}.pt", "gnnt", sysm, feat) for s in range(5)]
        if any(c[0].split("+")[0].startswith("m") for c in acerts):     # round 9: the networks with the margin feature
            feat_mg = featurizer(sysm, margin=True)
            nets_mg = [load_net(f"data/generated/uc12_m1x/models/pol_n2000_gnnt_mg_s{s}.pt", "gnnt", sysm, feat_mg)
                       for s in range(5)]
        from otsl.b3 import load_b3
        knn = KNNProb(sysm, load_b3("data/generated/uc12/train.npz"), k=50)
        thr_ltf = load_thr(a.ltf)

    # LP relaxation (timed once, charged where read); warm-up first
    m.solve_dispatch(sc, None, relax=True)
    t0 = time.perf_counter()
    rel = m.solve_dispatch(sc, None, relax=True)
    t_rel = time.perf_counter() - t0
    di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None],
              start=np.array([getattr(sc, "start", 0)]), u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
    for net in nets:
        net.predict(di)
    if nets_mg:
        from otsl.probe import margin_feature
        di_mg = dict(di, margin=margin_feature(sysm, sc.u0, rel.lmp, rel.mu, rel.obj, g_out)[None])
        for net in nets_mg:
            net.predict(di_mg)
    if knn is not None:
        knn.predict(di)
    mon = monitored_from_lp(rel.flow, sysm.fmax)

    def with_rel(ck):
        ck.add("lp_relaxation", t_rel)
        return ck

    def probs(pred):
        if pred == "lp":
            return np.clip(rel.u, 0.0, 1.0)
        if pred == "knn":
            return override_probs(knn.predict(di)[0], g_out)
        if pred.startswith("lag"):
            on = hourly_on_cost(sysm, rel.lmp, rel.mu)
            _, sched, margin = unit_margins(sysm, sc.u0, on, g_out)
            return np.where(margin >= float(pred[3:]) * rel.obj, sched, 0.5).astype(float)
        if pred.startswith("m"):                     # round 9: margin feature computed here (timed), then the nets
            dm = dict(di, margin=margin_feature(sysm, sc.u0, rel.lmp, rel.mu, rel.obj, g_out)[None])
            use = nets_mg if pred == "mens5" else [nets_mg[int(pred[1:])]]
            return override_probs(np.mean([n.predict(dm)[0].astype(np.float64) for n in use], 0), g_out)
        use = nets if pred == "ens5" else [nets[int(pred[1:])]]
        return override_probs(np.mean([n.predict(di)[0].astype(np.float64) for n in use], 0), g_out)

    meth = {}
    for g in (G01, EPS):
        def full(ck, g=g):
            r = ck("milp", solve_milp, m, sc, {}, tl, g)
            return r.get("u"), dict(obj=r["obj"], bound=r["bound"], status=r["status"], mip_gap=r["mip_gap"])

        def afull(ck, g=g):
            with_rel(ck)
            z = ck("milp", ag.solve, sc, mon, None, g, tl)
            return z["u"], dict(obj=z["obj"], bound=z["bound"], status=z["status"], rounds=z["rounds"], n_mon=z["n_mon"],
                                lines_ok=z["ok"])
        meth[f"full@{g * 100:g}%"] = full
        meth[f"agg:full@{g * 100:g}%"] = afull

    def sfull(ck):
        with_rel(ck)
        z = ck("milp", scr.solve, sc, mon, None, EPS, tl)
        return z["u"], dict(obj=z["obj"], bound=z["bound"], status=z["status"], rounds=z["rounds"], lines_ok=z["ok"])
    meth["scr:full@0.5%"] = sfull

    def cert7(ck, pred, dl):
        with_rel(ck)
        p = ck("inference", probs, pred)
        return ck("certified", certified_pipeline, scr, sc, p, rel, dl, EPS, None, tl)
    if learned:
        meth["cert:s0:0.02"] = lambda ck: cert7(ck, "s0", 0.02)
    meth["cert:lp:0.05"] = lambda ck: cert7(ck, "lp", 0.05)

    if learned:
        def ltf(ck):
            fix = ck("rule", lambda: fix_dict(probs("knn"), *thr_ltf))
            fix = force_out(fix, g_out, T)
            r = ck("milp", solve_milp, m, sc, fix, tl, G01)
            u, info = r.get("u"), dict(obj=r["obj"], n_fixed=len(fix), status=r["status"])
            if not r["feasible"]:                       # min up/down-infeasible fixings: the full MILP (as step 2)
                r = ck("fallback", solve_milp, m, sc, {}, tl, G01)
                u, info["fallback"] = r.get("u"), True
            return u, info
        meth["ltf_knn_1"] = ltf

    def round_first(p, eps_r):
        """cheapest candidates first: p and the LP values rounded at 0.5 and 0.2, block adequacy and min up/down
        repaired, priced by the dispatch LP of the full network; the cheapest is returned when the LP bound proves it
        within eps_r"""
        best = None
        srcs = [p] if p is not None else []
        srcs.append(np.clip(rel.u, 0.0, 1.0))
        for src in srcs:
            for th in (0.5, 0.2):
                u = adequacy_repair_blocks((src > th).astype(np.int8), sc.u0, sc.load, sc.avail, sc.sr, sv)
                u = repair_min_updown(u, sc.u0, sysm.min_up, sysm.min_dn)
                if g_out:
                    u[:, list(g_out)] = 0
                c = m.solve_dispatch(sc, u).obj
                if best is None or c < best[1]:
                    best = (u, c)
        gap = (best[1] - rel.obj) / best[1]
        return (best[0] if gap <= eps_r else None), gap

    for spec, dl in acerts:
        flags = spec.split("+")
        pred, probe, red = flags[0], "x" in flags[1:], G01 if "r" in flags[1:] else None
        first = next((float(f[1:]) / 100 for f in flags[1:] if f.startswith("f")), None)

        def acert(ck, pred=pred, dl=float(dl), probe=probe, red=red, first=first):
            with_rel(ck)
            p = ck("inference", probs, pred)
            if first is not None:
                u0, g0 = ck("round_first", round_first, None if pred.startswith(("lag", "lp")) else p, first)
                if u0 is not None:
                    return u0, dict(cert="round", lp_gap=float(g0), n_fixed=0, red_s=0.0, cert_s=0.0)
            return ck("certified", agg_certified_pipeline, ag, sc, p, rel, dl, None, EPS, None, tl, None, False, None,
                      probe, None, red)
        meth[f"acert:{spec}:{dl}"] = acert

    if a.methods:
        keep = set(a.methods.split(","))
        meth = {k_: v for k_, v in meth.items() if k_ in keep}
    order = sorted(meth)
    np.random.default_rng([zlib.crc32(f"{a.bench}/{a.set}/{a.k}".encode()), a.seed]).shuffle(order)
    rec = dict(key, **host_info(), order=order, c_rel=float(rel.obj), stored_c_rel=stored_c_rel, t_rel=t_rel,
               g_out=list(g_out), l_out=list(getattr(sc, "l_out", ()) or ()), loadavg0=os.getloadavg()[0], runs={})
    t_start = time.time()
    for name in order:
        ck = Clock()
        u, info = meth[name](ck)
        pr = price(m, sc, u)
        if "u_red" in info:
            pr_red = price(m, sc, info.pop("u_red"))
            info.update(red_cost=pr_red["cost"], red_shed=pr_red["shed"], red_short=pr_red["short"])
        info = {x: y for x, y in info.items() if not isinstance(y, (set, np.ndarray, dict))}
        rec["runs"][name] = dict(pr, time=ck.wall, parts=ck.parts, **info)
        print(f"  {name:26s} {ck.wall:7.2f}s  cost {pr['cost']:.1f}  "
              f"{'served' if pr['shed'] is not None and pr['shed'] < 1e-6 and pr['short'] < 1e-6 else 'NOT served'}",
              flush=True)
    rec["loadavg1"] = os.getloadavg()[0]
    rec["wall"] = time.time() - t_start
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(f"{key}: {len(order)} methods in {rec['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
