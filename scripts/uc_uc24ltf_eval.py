"""uc24ltf, step 4: test evaluation of Learning to Fix on uc24 (40 test instances), one worker, every rule of an
instance solved back to back in that worker.

    python3 scripts/uc_uc24ltf_eval.py --rules knn_1,knn_5,bce_1,bce_5,bce_g_1,bce_g_5 --old "95%|best=bce_g,95%|rl" --b2b 4

Rules (thresholds fixed on validation by scripts/uc_uc24ltf_tune.py; tag = <model>[_g]_<eps%>):
  LtF rules        fixings of eq. (5) with the tuned [lo_g, hi_g]; "_g" tags also run our guard chain (adequacy guard
                   -> LP-relaxation guard -> min up/down conflict release; otsl.uc24ltf.guard_fix), as in the tuning.
  cheap rules      --cheap "bce_const0.05,..." constant / worst-case thresholds (results/uc24/uc24ltf_cheap_thresholds.json)
  old rules        --old: the existing uc24 rules of scripts/uc_b3_fix.py re-run under the same conditions
                   (e.g. "95%|best=bce_g" = val-selected guarded rule, 95 % target; "95%|rl" = REINFORCE ranking).
Reduced MILPs: highspy, 0.1 % gap, 300 s limit, 1 thread, incumbent log (the full-MILP settings). No fallback
re-solve: a reduced problem without solution is infeasible for the method (the paper's convention); the report
charges such instances with the full MILP in our earlier convention.
Method time = inference (kNN: Table II features + neighbour search; GNN: forward pass + the LP relaxation that feeds
its features, timed in the worker) + fixing rule + guards (timed in the worker) + reduced MILP.
--b2b k: the full MILP is re-solved first on the first k instances, in the same worker (timing check).
Output: results/uc24/uc24ltf_eval_test.jsonl (one record per instance), log on stdout.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import load_b3, solve_milp_hs  # noqa: E402
from otsl.fixpolicy import lp_guard, release_conflicting_rows  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402
from otsl.uc24ltf import guard_fix  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1  # noqa: E402

ROOT, OUT = os.path.join("data", "generated", "uc24"), os.path.join("results", "uc24")
TL, GAP = 300.0, 1e-3
GNNS = {"bce": "b3_lf_bce.pt", "rl": "b3_rl_selected.pt"}
_W = {}


def _init():
    torch.set_num_threads(1)
    s = load_rts_gmlc()
    _W.update(s=s, m=UCModel(s, T=24, network=True))


def _rec(sol):
    ok = sol["u"] is not None
    return dict(feasible=ok, obj=float(sol["obj"]) if ok else None, status=sol["status"], t_milp=float(sol["time"]),
                c_milp=float(sol["cpu"]), mip_gap=float(sol["gap"]) if ok else None, bound=float(sol["bound"]),
                shed=float(sol.get("shed", np.nan)) if ok else None, short=float(sol.get("short", np.nan)) if ok else None,
                units_on=float(sol["u"].sum() / sol["u"].shape[0]) if ok else None, inc=sol["inc"],
                nodes=int(sol.get("nodes", -1)))


def _job(job):
    i, load, avail, u0, sr, specs, b2b = job
    s, m = _W["s"], _W["m"]
    sc = UCScenario(load=load, avail=avail, u0=u0, sr=sr)
    out = {"i": int(i), "loadavg0": os.getloadavg()[0]}
    if b2b:
        sol = solve_milp_hs(m, sc, time_limit=TL, mip_gap=GAP)
        out["full_b2b"] = dict(_rec(sol), loadavg=os.getloadavg()[0])
    t0 = time.time()
    m.solve_dispatch(sc, None, relax=True)          # LP relaxation: input of the GNN features (timed only)
    out["t_rel"] = time.time() - t0
    for name, fix_arr, kind, pre_s in specs:
        fix = {(int(t), int(g)): int(v) for t, g, v in fix_arr}
        n0 = len(fix)
        t0 = time.time()
        ginfo = {}
        if kind == "ltf_guard":
            fix, ginfo = guard_fix(m, s, sc, fix)
        elif kind == "ltf_guard_cr":                   # + min up/down conflict release (test-time safety net)
            fix, ginfo = guard_fix(m, s, sc, fix, use_conflict=True)
        elif kind == "ltf_guard_crf":                  # conflict release first, then the guards (test time)
            fix, ginfo = guard_fix(m, s, sc, fix, conflict_first=True)
        elif kind == "ltf_cr":
            fix, r_c = release_conflicting_rows(fix, s, u0)
            ginfo = dict(released_conflict=r_c)
        elif kind in ("b3", "b3_guard"):               # scripts/uc_b3_fix.py _job: LP guard (guarded rules), conflicts
            r_lp = 0
            if kind == "b3_guard" and fix:
                fix, r_lp, _ = lp_guard(m, s, sc, fix)
            fix, r_c = release_conflicting_rows(fix, s, u0)
            ginfo = dict(released_lp=r_lp, released_conflict=r_c)
        t_guard = time.time() - t0
        sol = solve_milp_hs(m, sc, time_limit=TL, mip_gap=GAP, z_fix=fix or None)
        out[name] = dict(_rec(sol), kind=kind, pre_s=float(pre_s), t_guard=t_guard, n_fixed_pre=n0, n_fixed=len(fix),
                         n_off=int(sum(1 for v in fix.values() if v == 0)), loadavg=os.getloadavg()[0],
                         **{k: v for k, v in ginfo.items()})
    return out


def thresholds(tags):
    """{tag: (rule name, model, lo, hi, kind)}; a tag ending in "+cr" adds the min up/down conflict release of our
    pipelines at test time (otsl.fixpolicy.release_conflicting_rows) to the rule"""
    th = {}
    cheap = json.load(open(os.path.join(OUT, "uc24ltf_cheap_thresholds.json"))) \
        if os.path.exists(os.path.join(OUT, "uc24ltf_cheap_thresholds.json")) else {}
    for tg0 in tags:
        crf = tg0.endswith("+crf")
        cr = tg0.endswith("+cr") or crf
        tg = tg0[:-4] if crf else (tg0[:-3] if cr else tg0)
        f = os.path.join(OUT, f"uc24ltf_tune_{tg}.json")
        if os.path.exists(f):
            r = json.load(open(f))
            nm = f"LtF{'+guards' if r['guarded'] else ''} {r['model']} eps={r['eps'] * 100:g}%" + \
                 (" + conflict release first" if crf else (" + conflict release" if cr else "")) + \
                 ("" if r["converged"] else " (not converged)")
            kind = ("ltf_guard" if r["guarded"] else "ltf") + ("_crf" if crf else ("_cr" if cr else ""))
            th[tg0] = (nm, r["model"], np.array(r["lo"]), np.array(r["hi"]), kind)
        elif tg in cheap:
            model, k2 = tg.split("_", 1)
            th[tg0] = (f"{model} {k2}" + (" + conflict release" if cr else ""), model, np.array(cheap[tg]["lo"]),
                       np.array(cheap[tg]["hi"]), "ltf_cr" if cr else "ltf")
        else:
            print(f"WARNING: no thresholds for {tg}", flush=True)
    return th


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="knn_1,knn_5,gnn_1,gnn_5,gnn_g_1,gnn_g_5", help="gnn = the GNN chosen in prep")
    ap.add_argument("--old", default="95%|best=bce_g,95%|rl")
    ap.add_argument("--b2b", type=int, default=4)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    d = load_b3(os.path.join(ROOT, "test.npz"))
    n = min(a.n, len(d["obj"]))
    N = len(d["obj"])
    one = lambda i: {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == N}
    choice = json.load(open(os.path.join(OUT, "uc24ltf_probs.json"))).get("gnn_choice", "bce")
    th = thresholds([t.replace("gnn", choice, 1) if t.startswith("gnn") else t for t in a.rules.split(",") if t])
    models = sorted({v[1] for v in th.values()} | ({"bce", "rl"} if a.old else set()))
    # ---------------- probabilities per instance, timed (inference cost of each model)
    from uc_uc24ltf_prep import load_train_lab
    from uc_b3_train import strip
    P, T_inf = {k: [] for k in models}, {k: [] for k in models}
    mods = {}
    if "knn" in models:
        mods["knn"] = KNNProb(sysm, load_train_lab(os.path.join(ROOT, "uc24ltf_train_lab.npz")), k=50)
    tr = strip(load_b3(os.path.join(ROOT, "train.npz")))
    feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
    for k in models:
        if k in GNNS:
            m1 = build_uc_model1(sysm, feat, 24, "gnn", seed=0)
            m1.net.load_state_dict(torch.load(os.path.join(OUT, GNNS[k])))
            mods[k] = m1
            m1.predict(one(0))                       # warm-up
    for i in range(n):
        di = one(i)
        for k in models:
            t0 = time.time()
            p = mods[k].predict(di)[0].astype(np.float64)
            T_inf[k].append(time.time() - t0)
            P[k].append(p)
    P = {k: np.array(v) for k, v in P.items()}
    ref = np.load(os.path.join(OUT, "uc24ltf_probs.npz"))
    for k in models:
        print(f"max |p - uc24ltf_probs| {k}: {np.abs(P[k] - ref[f'{k}_te'][:n]).max():.2e}", flush=True)
    old = [o for o in a.old.split(",") if o]
    if old:
        from uc_b3_fix import make_specs
    jobs = []
    for i in range(n):
        specs = []
        for tg, (nm, model, lo, hi, kind) in th.items():
            t0 = time.time()
            fix = fix_dict(P[model][i], lo, hi)
            pre = T_inf[model][i] + time.time() - t0
            arr = np.array([(t, g, v) for (t, g), v in fix.items()], np.int64).reshape(-1, 3)
            specs.append((nm, arr, kind, pre))
        for o in old:
            q, rk = o.split("|")
            rk = rk.replace("best=", "")
            t0 = time.time()
            (_, arr, g), = make_specs(i, d, {"bce": P["bce"], "rl": P["rl"]}, sysm, [rk], [int(q[:-1]) / 100])
            pre = T_inf["rl" if rk.startswith("rl") else "bce"][i] + time.time() - t0
            specs.append((f"old: {o}", arr, "b3_guard" if g else "b3", pre))
        jobs.append((i, d["load"][i], d["avail"][i], d["u0"][i], d["sr"][i], specs, i < a.b2b))
    names = [s_[0] for s_ in jobs[0][5]]
    print(f"{len(names)} rules per instance: {names}", flush=True)
    for nm, arr, kind, pre in jobs[0][5]:
        print(f"  {nm:45s} kind {kind:9s} fixed (pre-guard) {len(arr) / (24 * sysm.G) * 100:5.1f}%  pre {pre * 1000:.1f} ms",
              flush=True)
    sh = {nm: np.mean([len(j[5][k][1]) / (24 * sysm.G) for j in jobs]) for k, nm in enumerate(names)}
    print("mean pre-guard fixed share over the instances:", {k: round(v * 100, 1) for k, v in sh.items()}, flush=True)
    if a.dry:
        sys.exit(0)
    path = os.path.join(OUT, f"uc24ltf_eval_test{a.tag}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["i"] for line in open(path)}
    jobs = [j for j in jobs if j[0] not in done]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(path, "a") as f:
        for k, r in enumerate(pool.imap(_job, jobs, chunksize=1)):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            i = r["i"]
            msg = " | ".join(f"{nm.replace('LtF', '').strip()[:22]}: "
                             + (f"{(r[nm]['obj'] - d['obj'][i]) / d['obj'][i] * 100:+.2f}% {r[nm]['t_milp']:.0f}s"
                                if r[nm]["feasible"] else "INFEASIBLE") for nm in names)
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={i} full {d['time'][i]:.0f}s"
                  + (f" (b2b {r['full_b2b']['t_milp']:.0f}s)" if "full_b2b" in r else "") + " | " + msg, flush=True)
    print("done", flush=True)
