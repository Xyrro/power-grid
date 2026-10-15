"""rloop (arm B) test evaluation, paired: every method of an instance is run back to back in one process on one core.

    OTSL_THREADS=1 taskset -c 2 python3 scripts/rec_rloop_eval.py --bench uc12 --shard 0 --nshards 2
    OTSL_THREADS=1 taskset -c 3 python3 scripts/rec_rloop_eval.py --bench uc12 --shard 1 --nshards 2

Per instance (benchmarks: uc12 = test_fresh 120, uc24 = 40 test instances of 24 h, hard = 60 congested test
instances, line_out = 30 instances with 1-2 critical lines out):
  1. full MILP at 0.1 % (speed-up denominator and dual bound DB) [and at 0.5 %], highspy, one thread
  2. LP relaxation (UCModel path, the input of every method; its time is counted in every method)
  3. LP rounding + block repair + min up/down repair, 5 LPs (otsl.base.lp_round_candidates; no learning)
  4. m1x 5-member temporal-GNN ensemble (12 h only): rounding at the validation-selected thresholds + the same repairs,
     1 / 3 / 5 LPs; and Viterbi decoding + repair, 1 LP
  5. rloop: K outer steps (network pass + decoding + LP each), per-step cost and cumulative time
  6. safety net for rloop's best schedule after K_sel steps: if it sheds / misses reserve, or its certified gap to
     the LP relaxation exceeds theta_min, a reduced MILP (0.5 %) around it (fix the decisions where the last
     probabilities are within 0.05 of 0 / 1 and agree with the schedule; min up/down row release); larger thresholds
     are evaluated offline from the same records
All dispatch LPs through otsl.rloop.FastLP (one solver object per method and instance, warm-started across that
method's own LPs). Records: results/recursive/rloop_eval_<bench>_s<shard>.jsonl (resume skips finished instances).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rec_rloop_common as C  # noqa: E402
from otsl import rloop  # noqa: E402
from otsl.base import E2E_TH, lp_round_candidates, solve_reduced  # noqa: E402
from otsl.fixpolicy import release_conflicting_rows  # noqa: E402
from otsl.ood import OODModel  # noqa: E402

TL = {"uc12": 60.0, "line_out": 60.0, "hard": 600.0, "uc24": 300.0}


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


def price_seq(lp, sc, scheds, t0):
    """price schedules in order (distinct ones once); per candidate: cost, shed, short, cumulative time since t0"""
    lp.reset()
    seen, out = {}, []
    for s in scheds:
        k = s.tobytes()
        if k not in seen:
            sol = lp.solve(sc, s)
            seen[k] = (float(sol["obj"]), float(sol["shed"].sum() + sol["spill"].sum()), float(sol["short"].sum()))
        out.append(list(seen[k]) + [time.time() - t0])
    return out, len(seen)


class M1X:
    """the m1x ensemble (results/uc12/m1x_pol_n2000_gnnt_s0..4.pt) with its featurizer; fmax for derated ratings"""

    def __init__(self, sysm_bench):
        from uc_m1x_train import featurizer, load_net
        from otsl.uc import load_rts_gmlc
        s0 = load_rts_gmlc()
        self.feat = featurizer(s0)
        if not np.allclose(sysm_bench.fmax, s0.fmax):          # zero-shot on derated lines: same statistics
            self.feat.e_static = np.c_[np.log(s0.b), sysm_bench.fmax]
            self.feat.s = sysm_bench
        self.members = [load_net(os.path.join(C.ROOT, f"results/uc12/m1x_pol_n2000_gnnt_s{k}.pt"), "gnnt", s0, self.feat)
                        for k in range(5)]

    def predict(self, inst):
        d = {k: np.asarray(inst[k])[None] for k in ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel")}
        return np.mean([m.predict(d)[0].astype(np.float64) for m in self.members], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=sorted(TL))
    ap.add_argument("--model", default=None, help="checkpoint tag (default: results/recursive/rloop_select.json)")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--k_sel", type=int, default=None)
    ap.add_argument("--theta_min", type=float, default=0.5, help="% certified gap that triggers the safety net")
    ap.add_argument("--gaps", default=None, help="full-MILP gaps, comma separated (default 0.001,0.005)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.chdir(C.ROOT)
    sel = json.load(open(os.path.join(C.RES, "rloop_select.json"))) if os.path.exists(os.path.join(C.RES, "rloop_select.json")) else {}
    tag = a.model or sel["model"]
    k_sel = a.k_sel or sel.get("K", {}).get(a.bench) or 8
    K = a.K or max(16, 2 * k_sel)
    gaps = [float(g) for g in (a.gaps or "0.001,0.005").split(",")]
    m1x_ths = sel.get("m1x_thresholds", {"1": [0.5], "3": [0.3, 0.5, 0.7], "5": [0.2, 0.35, 0.5, 0.65, 0.8]})
    sysm = C.system(a.bench)
    d = C.bench_data(a.bench, "test")
    T = d["load"].shape[1]
    m = OODModel(sysm, T)
    lp = rloop.FastLP(sysm, T)
    net, ck, feats = C.load_model(os.path.join(C.OUT, f"{tag}.pt"), sysm)
    inf = rloop.Inference(net, feats, sysm, rloop.FastLP(sysm, T), mode=ck["cfg"].get("decode", "viterbi"))
    m1x = M1X(sysm) if T == 12 else None
    fmax = sysm.fmax
    n = len(d["load"]) if a.n == 0 else min(a.n, len(d["load"]))
    out = a.out or os.path.join(C.RES, f"rloop_eval_{a.bench}_s{a.shard}.jsonl")
    done = set()
    for f in os.listdir(C.RES):
        if f.startswith(f"rloop_eval_{a.bench}_") and f.endswith(".jsonl"):
            done |= {r["k"] for r in C.read_jsonl(os.path.join(C.RES, f))}
    todo = [k for k in range(a.start, n) if k % a.nshards == a.shard and k not in done]
    print(f"{a.bench} shard {a.shard}/{a.nshards} on CPU {sorted(os.sched_getaffinity(0))}: model {tag}, K {K}, "
          f"k_sel {k_sel}, {len(todo)} instances ({len(done)} done)", flush=True)
    # warm-up (first-call overheads)
    w = C.insts_of(d, [0])[0]
    inf.rollout(w, 2, fmax=fmax)
    if m1x:
        m1x.predict(w)
    for k in todo:
        t_start = time.time()
        inst0 = C.insts_of(d, [k])[0]
        sc = rloop.scenario_of(inst0)
        rec = dict(bench=a.bench, k=k, model=tag, cpu=sorted(os.sched_getaffinity(0)), T=T)
        full = {}
        for g in gaps:
            r = solve_reduced(m, sc, None, TL[a.bench], g)
            full[str(g)] = {kk: r[kk] for kk in ("feasible", "obj", "time", "bound", "status", "mip_gap", "shed", "short")}
        rec["full"] = full
        rel, t_rel = timed(m.solve_dispatch, sc, None, relax=True)
        inst = dict(inst0, u_rel=rel.u, lmp_rel=rel.lmp, flow_rel=rel.flow, c_rel=float(rel.obj))
        rec.update(t_rel=t_rel, c_rel=float(rel.obj), c_rel_stored=float(inst0["c_rel"]))
        # LP rounding + repair + 5 LPs
        t0 = time.time()
        cands = [u for _, u in lp_round_candidates(inst["u_rel"], sc, sysm, E2E_TH)]
        pr, nlp = price_seq(lp, sc, cands, t0)
        rec["lpround"] = dict(c=pr, n_lp=nlp, t_pre=t_rel)
        # m1x ensemble end to end
        if m1x is not None:
            t0 = time.time()
            p = m1x.predict(inst)
            t_inf = time.time() - t0
            for kk, ths in m1x_ths.items():
                t1 = time.time()
                cands = [rloop.decode((p > th).astype(np.int8), inst, sysm, "round")[1] for th in ths]
                pr, nlp = price_seq(lp, sc, cands, t1)
                rec[f"m1x_{kk}"] = dict(c=pr, n_lp=nlp, t_pre=t_rel + t_inf, th=ths)
            t1 = time.time()
            pr, nlp = price_seq(lp, sc, [rloop.decode(p, inst, sysm, "viterbi")[1]], t1)
            rec["m1x_vit"] = dict(c=pr, n_lp=nlp, t_pre=t_rel + t_inf)
        # rloop
        recs, best_u, last_p = inf.rollout(inst, K, t_pre=t_rel, fmax=fmax)
        rec["rloop"] = [{kk: v for kk, v in r.items()} for r in recs]
        # safety net at k_sel
        rs = recs[min(k_sel, K) - 1]
        served = rs["best_shed"] < 1e-6 and rs["best_short"] < 1e-6
        cert = (rs["best"] / inst["c_rel"] - 1) * 100
        rec["cert_gap"] = cert
        if (not served) or cert > a.theta_min:
            # the schedule and probabilities of the best step within the first k_sel steps
            r2, bu, lp_ = inf.rollout(inst, k_sel, t_pre=0.0, fmax=fmax)
            t0 = time.time()
            fix = {(int(t), int(g)): int(bu[t, g]) for t, g in zip(*np.where((np.abs(lp_ - 0.5) >= 0.45) &
                                                                          ((lp_ > 0.5) == (bu > 0))))}
            fix, nrel = release_conflicting_rows(fix, sysm, inst["u0"])
            t_fix = time.time() - t0
            r = solve_reduced(m, sc, fix, TL[a.bench], 0.005)
            if not r["feasible"]:
                r2_ = solve_reduced(m, sc, None, TL[a.bench], 0.005)
                r = dict(r2_, time=r["time"] + r2_["time"], fallback_full=True)
            rec["safety"] = dict(n_fixed=len(fix), released=nrel, t_fix=t_fix, time=r["time"], obj=r["obj"],
                                 feasible=r["feasible"], shed=r.get("shed"), short=r.get("short"), status=r["status"],
                                 full=bool(r.get("fallback_full", False)))
        rec["wall"] = time.time() - t_start
        C.append_jsonl(out, rec)
        f1 = full[str(gaps[0])]
        best = min(x["best"] for x in recs[:k_sel])
        print(f"[{a.bench} {k}] full {f1['time']:.1f}s obj {f1['obj']:.1f} | rloop@{k_sel} "
              f"{(best - f1['bound']) / best * 100:.3f}% in {recs[k_sel - 1]['time']:.2f}s | lpround "
              f"{(min(x[0] for x in rec['lpround']['c']) - f1['bound']) / min(x[0] for x in rec['lpround']['c']) * 100:.3f}%"
              f" | cert {cert:.2f}% {'safety ' + str(round(rec['safety']['time'], 1)) + 's' if 'safety' in rec else ''}"
              f" ({rec['wall']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
