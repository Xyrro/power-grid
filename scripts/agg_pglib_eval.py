"""Round 8 on the public PGLib-UC California system (24 h, 410 free units; docs/methods/agg.md): the certified pipeline
against the full MILP and Learning to Fix as published, every method of an instance back to back in one
single-threaded process (paired speed-ups).

    PGLIB=<pglib-uc> OTSL_THREADS=1 python3 scripts/agg_pglib_eval.py --k 0 --out results/agg/pglib/test_x.jsonl

Data, models and Learning to Fix's thresholds: scripts/pace/pgu_pglib.sbatch (docs/methods/pglib.md section 9). The LP
relaxation is solved once (timed) and charged to every method that reads it.

Methods:
  full@g          the full MILP at gap g (full@0.1% is the speed-up reference; 900 s limit)
  ltf_knn_1       Learning to Fix as published (Fritz et al. 2026): kNN (k = 50, Table II features, the self-training
                  labelled set) and per-generator thresholds tuned by its Benders procedure at eps = 1 % on the 30
                  validation instances; reduced MILP at 0.1 % (as the paper's regime in docs/methods/pglib.md); a reduced
                  MILP without a solution falls back to the full MILP (time added; no instance dropped)
  pcert:<pred>:<d>:<e>   the certified pipeline: every decision with min(p, 1 - p) <= d fixed, rows min up/down cannot
                  complete released -> reduced MILP at gap e -> certificate: the LP relaxation's bound within e of the
                  schedule's cost, else the full MILP at e warm-started from it (the better schedule kept). <pred>: lp
                  (the LP relaxation's values: no learning), knn (Learning to Fix's predictor), st (our self-trained
                  model, no MILP label). e = 0.5 % or 0.1 % (the reference's own gap: the same guarantee). A leading
                  'r' (rlp, rst) tries the cheapest candidates first: the probabilities rounded at 0.5 and 0.1, block
                  adequacy and min up/down repaired, priced by one dispatch LP each; the cheaper one is delivered when
                  the LP bound proves it within e, otherwise the pipeline continues as above.
Every delivered schedule is priced by the exact dispatch LP (not timed).
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.fixpolicy import release_conflicting_rows  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.pglib import PGModel, highs_run, load_bases, load_npz, subset  # noqa: E402

ROOT = "data/generated/pglib_ca"
ROUND_TH = (0.5, 0.1)


class Sc:
    def __init__(self, d, i):
        self.load, self.avail, self.sr, self.u0 = d["load"][i], d["avail"][i], d["sr"][i], d["u0"][i]
        self.avail_min = d["avail_min"][i] if "avail_min" in d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--pcert", default="lp:0.05:0.5,rlp:0.05:0.5,rlp:0.05:0.2,lp:0.05:0.1,rlp:0.05:0.1,knn:0.05:0.5,"
                                         "st:0.02:0.5,rst:0.02:0.5")
    ap.add_argument("--methods", default="")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    if os.path.exists(a.out):
        for line in open(a.out):
            if json.loads(line)["k"] == a.k:
                print(f"{a.k} already in {a.out}")
                return
    sysm, _ = load_bases(a.T)
    m = PGModel(sysm, T=a.T)
    d = load_npz(os.path.join(ROOT, f"{a.split}.npz"))
    sc = Sc(d, a.k)
    one = subset(d, np.array([a.k]))

    # models: kNN on the self-training labelled set, our self-trained model (no MILP label)
    import uc_pglib_train as PT
    from otsl.pglib_ml import KNNProb, PGFeaturizer
    full_tr = PT.strip(load_npz(os.path.join(ROOT, "train.npz")))
    n_all = len(full_tr["load"])
    tr = subset(full_tr, np.arange(n_all - 16))
    L = np.load(os.path.join(ROOT, "st_labels.npz"))
    knn = KNNProb(sysm, subset(tr, L["idx"]), L["u"], k=50)
    feat = PGFeaturizer(sysm, tr)
    st = PT.load_model(sysm, feat, "m1_st", 0)
    J = json.load(open("results/pglib/pglib_ltf_knn_1.json"))
    thr = (np.array(J["lo"]), np.array(J["hi"]))

    # LP relaxation (timed once; warm-up first)
    m.solve_dispatch(sc, None, relax=True)
    t0 = time.perf_counter()
    rel = m.solve_dispatch(sc, None, relax=True)
    t_rel = time.perf_counter() - t0
    one.update(u_rel=rel.u[None], lmp_rel=rel.lmp[None], res_rel=rel.res_price[None], c_rel=np.array([rel.obj]))
    knn.predict(one)
    st.predict(PT.strip(one))

    def probs(pred):
        if pred == "lp":
            return np.clip(rel.u, 0, 1), 0.0
        t0 = time.perf_counter()
        p = knn.predict(one)[0] if pred == "knn" else st.predict(PT.strip(one))[0]
        return p, time.perf_counter() - t0

    def milp(fix, gap, start=None, tl=a.tl):
        lo, hi, lb, ub = m.bounds_fixed(sc, fix)
        return highs_run(m.c, m.A, lo, hi, lb, ub, m.integ, tl, gap, 1, False, start=start)

    def u_of(x):
        return np.round(m._unpack(x)[0]).astype(np.int8)

    meth = {}
    for g in (1e-3, 5e-3):
        def full(g=g):
            t0 = time.perf_counter()
            r = milp(None, g)
            return (u_of(r["x"]) if r["x"] is not None else None), time.perf_counter() - t0, dict(
                obj=r["obj"], bound=r["bound"], status=r["status"])
        meth[f"full@{g * 100:g}%"] = full

    def ltf():
        p, t_p = probs("knn")
        t0 = time.perf_counter()
        fix = fix_dict(p, *thr)
        r = milp(fix, 1e-3)
        info = dict(n_fixed=len(fix), status=r["status"], obj=r["obj"], fallback=r["x"] is None)
        if r["x"] is None:
            r = milp(None, 1e-3)
        return (u_of(r["x"]) if r["x"] is not None else None), t_p + time.perf_counter() - t0, info
    meth["ltf_knn_1"] = ltf

    from otsl.pglib import rep_block
    inst = {x: d[x][a.k] for x in ("load", "avail", "sr", "u0")}

    for spec in a.pcert.split(","):
        rnd = spec.startswith("r")
        pred, dl, ep = spec.lstrip("r").split(":")

        def pcert(pred=pred, dl=float(dl), eps=float(ep) / 100, rnd=rnd):
            p, t_p = probs(pred)
            t0 = time.perf_counter()
            if rnd:                                     # cheapest candidates first: rounded and repaired schedules
                best = None
                for th in ROUND_TH:
                    u0 = rep_block((p > th).astype(np.int8), inst, sysm)
                    pr0 = m.solve_dispatch(sc, u0)
                    if best is None or pr0.obj < best[1]:
                        best = (u0, pr0.obj)
                lp0 = (best[1] - rel.obj) / best[1]
                if lp0 <= eps:
                    return best[0], t_rel + t_p + time.perf_counter() - t0, dict(cert="round", lp_gap=lp0, n_fixed=0,
                                                                                    round_s=time.perf_counter() - t0)
            fix = {(int(t), int(g)): int(p[t, g] > 0.5) for t, g in np.argwhere(np.minimum(p, 1 - p) <= dl)}
            fix, _ = release_conflicting_rows(fix, sysm, sc.u0)
            r = milp(fix, eps)
            red_s = time.perf_counter() - t0
            info = dict(n_fixed=len(fix), red_obj=r["obj"], red_s=red_s)
            if r["x"] is None:                          # cannot happen after the row release; kept for safety
                r = milp(None, eps)
                info["fallback"] = True
            C = r["obj"]
            lp_gap = (C - rel.obj) / C
            info.update(lp_gap=lp_gap, u_red=u_of(r["x"]))
            if lp_gap <= eps:
                info.update(cert="lp", cert_s=0.0)
                return u_of(r["x"]), t_rel + t_p + time.perf_counter() - t0, info
            t1 = time.perf_counter()
            z = milp(None, eps, start=r["x"], tl=max(a.tl - (time.perf_counter() - t0), 1.0))
            x = z["x"] if z["x"] is not None and z["obj"] <= C else r["x"]
            info.update(cert="milp", cert_s=time.perf_counter() - t1, cert_obj=z["obj"], cert_bound=z["bound"],
                        cert_status=z["status"])
            return u_of(x), t_rel + t_p + time.perf_counter() - t0, info
        meth[f"pcert:{spec}"] = pcert

    if a.methods:
        meth = {k: v for k, v in meth.items() if k in set(a.methods.split(","))}
    order = sorted(meth)
    np.random.default_rng(a.k).shuffle(order)
    import highspy
    h = highspy.Highs()
    rec = dict(k=a.k, split=a.split, base=int(d["base"][a.k]), c_rel=rel.obj, t_rel=t_rel, host=socket.gethostname(),
               highs=f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}", order=order, runs={})
    try:
        rec["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    for name in order:
        u, dt, info = meth[name]()
        pr = m.solve_dispatch(sc, u) if u is not None else None
        if "u_red" in info:
            prr = m.solve_dispatch(sc, info.pop("u_red"))
            info.update(red_cost=prr.obj, red_served=bool(prr.shed + prr.short < 1e-6))
        rec["runs"][name] = dict(time=dt, cost=pr.obj if pr else float("inf"),
                                 served=bool(pr is not None and pr.shed + pr.short < 1e-6), **info)
        print(f"  {name:22s} {dt:8.2f}s  cost {rec['runs'][name]['cost']:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    main()
