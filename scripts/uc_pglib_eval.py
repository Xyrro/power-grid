"""PGLib-UC California: test evaluation of every fixing rule (reduced MILPs) and the end-to-end mode.

    python3 scripts/uc_pglib_eval.py --split test --n 30 --b2b 6 --workers 2 [--src st]

Per instance, one worker solves (optionally) the full MILP and then every reduced MILP back to back. The full
MILP's time / dual bound / incumbent log come from the generation run (data/generated/pglib_ca/<split>.npz); the
first --b2b instances also re-solve the full MILP in the evaluation worker as a timing check. Method times include
inference (LP relaxation for our features: its generation-run time; model forward pass; kNN search; error-cost
features and scoring), every guard (adequacy, min up/down rows, LP-relaxation guard LPs) and the reduced MILP.
Records: results/pglib/pglib_eval_<split>.jsonl (one line per instance).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.combo import adequacy_guard, harm_from_file  # noqa: E402
from otsl.fixpolicy import FixFeaturizer, fix_from_ranking, release_conflicting_rows  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.pglib import load_bases, load_npz, rep_block, rep_minud, subset  # noqa: E402
from otsl.pglib_ml import SolverPool, dispatch_job, fix_array, fixeval_job, inst_arrays  # noqa: E402

ROOT, OUT = "data/generated/pglib_ca", "results/pglib"
E2E_TH = (0.001, 0.05, 0.2, 0.5, 0.8)


class Sc:
    def __init__(self, d, i):
        self.load, self.avail, self.sr, self.u0 = d["load"][i], d["avail"][i], d["sr"][i], d["u0"][i]


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--b2b", type=int, default=6)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--tl", type=float, default=900.0)
    ap.add_argument("--gap", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--src", default="st", help="probability source of Learning to Fix on our model (val-selected)")
    ap.add_argument("--ltf", default="knn_1,knn_5,{src}_1")
    ap.add_argument("--rules", default="all")
    ap.add_argument("--skip", default="", help="comma-separated rule names to leave out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_e2e", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    sysm, _ = load_bases(a.T)
    T, G = a.T, sysm.G
    d = load_npz(os.path.join(ROOT, f"{a.split}.npz"))
    idx = np.arange(a.start, min(a.start + a.n, len(d["load"])))
    d = subset(d, idx)
    n = len(idx)
    sp = "te" if a.split == "test" else "va"
    P = np.load(os.path.join(OUT, "pglib_probs.npz"))
    probs = {k: P[f"m1_{k}_{sp}"][idx] for k in ("lf", "rl", "st") if f"m1_{k}_{sp}" in P.files}
    t_fwd = {k: float(P[f"m1_{k}_{sp}_s"]) for k in probs}
    # kNN (paper's classifier, k = 50, Table II features) on the self-training labelled set: probabilities and
    # per-instance search time from the probs stage of uc_pglib_train.py
    p_knn, t_knn = P[f"knn_{sp}"][idx], float(P[f"knn_{sp}_s"])
    # Learning to Fix thresholds
    ltf = {}
    for tg in a.ltf.format(src=a.src).split(","):
        f = os.path.join(OUT, f"pglib_ltf_{tg}.json")
        if os.path.exists(f):
            J = json.load(open(f))
            ltf[tg] = (np.array(J["lo"]), np.array(J["hi"]))
    print("LtF thresholds:", list(ltf), flush=True)
    harm = harm_from_file(os.path.join(OUT, "pglib_harm.pt")) if os.path.exists(os.path.join(OUT, "pglib_harm.pt")) else None
    ff = FixFeaturizer(sysm, T)

    jobs, meta = [], {}
    for k in range(n):
        i = int(idx[k])
        sc = Sc(d, k)
        t_rel = float(d["t_rel"][k])
        specs = []          # (name, fix array, extra seconds, lp guard flag)

        def add(name, fix, extra, lpg=False, rows=True):
            t0 = time.time()
            if rows:
                fix, _ = release_conflicting_rows(fix, sysm, sc.u0)
            specs.append((name, fix_array(fix), extra + time.time() - t0, lpg))

        # no learning: fix every decision whose LP relaxation value is integral
        ur = d["u_rel"][k]
        integral = np.abs(ur - np.round(ur)) < 1e-6
        add("lprelax_integral", {(int(t), int(g)): int(round(ur[t, g])) for t, g in zip(*np.where(integral))}, t_rel)
        # paper's kNN: constant thresholds and Learning to Fix
        pk = p_knn[k]
        add("knn_const_0.01", fix_dict(pk, 0.01, 0.99), t_knn, rows=False)
        add("knn_const_0.05", fix_dict(pk, 0.05, 0.95), t_knn, rows=False)
        for tg, (lo, hi) in ltf.items():
            src = tg.split("_")[0]
            p = pk if src == "knn" else probs[src][k]
            ex = t_knn if src == "knn" else t_rel + t_fwd[src]
            add(f"ltf_{tg}", fix_dict(p, lo, hi), ex, rows=False)
        # RACLearn-style on our probabilities: constant thresholds and confidence ranking
        for src in probs:
            p = probs[src][k]
            ex = t_rel + t_fwd[src]
            add(f"{src}_const_0.01", fix_dict(p, 0.01, 0.99), ex, rows=False)
            yhat = (p > 0.5).astype(int)
            add(f"{src}_rac_95", fix_from_ranking(np.minimum(p, 1 - p), yhat, 0.95), ex, rows=False)
        # ours: error-cost ranking + guards (probabilities of --src)
        if harm is not None:                  # the harm model was trained on the self-trained model's probabilities
            p = probs["st"][k]
            yhat = (p > 0.5).astype(int)
            (X, t_f) = timed(ff, p, d, k)
            hs, t_s = timed(harm.score, X)
            ex = t_rel + t_fwd["st"] + t_f + t_s
            for ratio, lpg in ((0.90, False), (0.95, True), (0.98, True)):
                t0 = time.time()
                fix = fix_from_ranking(hs, yhat, ratio)
                fix, _ = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
                add(f"harm_{int(ratio * 100)}{'_lpg' if lpg else ''}", fix, ex + time.time() - t0, lpg=lpg)
        if a.rules != "all":
            keep = a.rules.split(",")
            specs = [s for s in specs if s[0] in keep]
        if a.skip:
            specs = [s for s in specs if s[0] not in a.skip.split(",")]
        meta[i] = {s[0]: dict(n_fixed=int(len(s[1])), extra=s[2]) for s in specs}
        jobs.append((i, inst_arrays(d, k), specs, a.tl, a.gap, k < a.b2b))

    # end-to-end (no MILP): threshold -> block adequacy repair -> min up/down repair -> dispatch LP
    e2e_jobs, e2e_meta = [], {}
    for k in range(n):
        i = int(idx[k])
        srcs = {"lprelax": (d["u_rel"][k], float(d["t_rel"][k]))}
        srcs.update({s: (probs[s][k], float(d["t_rel"][k]) + t_fwd[s]) for s in probs})
        us, names, ex = [], [], []
        for s, (p, t_in) in srcs.items():
            for th in E2E_TH:
                inst = {kk: d[kk][k] for kk in ("load", "avail", "sr", "u0")}
                u, t_r = timed(rep_block, (p > th).astype(np.int8), inst, sysm)
                us.append(u)
                names.append((s, th))
                ex.append((t_in, t_r))
        e2e_jobs.append((i, inst_arrays(d, k), us))
        e2e_meta[i] = (names, ex)

    out_path = os.path.join(OUT, f"pglib_eval_{a.split}{a.tag}.jsonl")
    pool = SolverPool(dict(T=T), a.workers)
    t0 = time.time()
    e2e = {}
    if not a.no_e2e:
        for i, r in pool.run(dispatch_job, e2e_jobs):
            e2e[i] = r
    print(f"end-to-end LPs done ({time.time() - t0:.0f}s)", flush=True)
    done = 0
    for i, res in pool.pool.imap_unordered(fixeval_job, jobs, chunksize=1):
        k = int(np.where(idx == i)[0][0])
        names, ex = e2e_meta[i]
        rec = dict(i=i, split=a.split, t_full=float(d["time"][k]), bound=float(d["bound"][k]), obj=float(d["obj"][k]),
                   gap_full=float(d["gap"][k]), c_rel=float(d["c_rel"][k]), t_rel=float(d["t_rel"][k]), base=int(d["base"][k]),
                   inc_t=[float(x) for x in d["inc_t"][k] if np.isfinite(x)],
                   inc_obj=[float(x) for x in d["inc_obj"][k][np.isfinite(d["inc_t"][k])]],
                   rules={}, e2e=[])
        for name, r in res.items():
            if name.startswith("__"):
                rec[name.strip("_")] = r
                continue
            r["extra"] = meta[i][name]["extra"]
            r["n_fixed"] = r["n_fixed_final"]
            rec["rules"][name] = r
        for (s, th), (c, sh, so, dt), (t_in, t_r) in zip(names, e2e.get(i, []), ex):
            rec["e2e"].append(dict(src=s, th=th, cost=c, shed=sh, short=so, lp_s=dt, t_in=t_in, rep_s=t_r, extra=t_in + t_r))
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        done += 1
        best = min(((v["obj"] - rec["bound"]) / v["obj"] * 100, nm) for nm, v in rec["rules"].items() if v["feasible"])
        print(f"[eval] {done}/{n} inst {i} full {rec['t_full']:.0f}s | " + " ".join(
            f"{nm}:{v['time'] + v['extra']:.0f}s/{(v['obj'] - rec['bound']) / v['obj'] * 100:.2f}%"
            for nm, v in rec["rules"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    pool.close()
