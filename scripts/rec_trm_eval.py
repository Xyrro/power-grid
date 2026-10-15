"""Test evaluation of the TRM study (arm A) with paired timing: for each instance, in ONE process pinned to one core,
the full MILP at 0.1 % (dual bound, speed-up denominator) and at 0.5 %, the LP relaxation, the baselines and every TRM
pipeline, back to back. Read the test set once, after every choice was fixed on validation.

    taskset -c 0 python3 scripts/rec_trm_eval.py --split test_fresh --shard 0 --nshards 2
    taskset -c 1 python3 scripts/rec_trm_eval.py --split test_fresh --shard 1 --nshards 2

Methods (time = every overhead: LP relaxation where used, features, network passes, decoding / repairs, LPs):
  milp01, milp05                 full MILP (highspy, 1 thread, 60 s) at mip_rel_gap 0.1 % / 0.5 %
  lpround5                       LP rounding at 5 thresholds + block adequacy / min up/down repair + <= 5 LPs (no learning)
  m1x_1, m1x_5                   m1x temporal-GNN ensemble (5 members), validation-selected decoder / threshold, 1 LP;
                                 5 validation-selected thresholds, <= 5 LPs
  rl_lag7, rl_rl1                REINFORCE end-to-end pipelines of docs/methods/combo.md (seed 0): Lagrangian + KL model,
                                 screening over 7 thresholds; label-free + REINFORCE at its calibrated threshold, 1 LP
  <tag>_1, <tag>_s5, <tag>_t5    a TRM model: 1 LP at its training step count; screening the last 5 steps' schedules;
                                 screening 5 thresholds; <tag>_k<k>: 1 LP after k steps (test-time compute curve);
                                 <tag>_halt: stop at the first step whose halting logit is > 0
  <tag>_fb                       <tag>_1 + safety net: if the LP sheds / misses reserve or the bound gap
                                 (C - z_LP) / C >= g* (validation), reduced MILP (0.5 %) around the model's 90 % most
                                 confident decisions (+ adequacy guard, min up/down rows), keep the cheaper schedule
Plan of TRM tags, selections and g*: results/recursive/trm_select.json (written by scripts/rec_trm_select.py).
Records: results/recursive/trm_eval_<split>_s<shard>.jsonl (one line per instance; resumable).
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
from otsl import trm  # noqa: E402
from otsl.base import lp_round_screen, solve_reduced  # noqa: E402
from otsl.combo import THRESHOLDS, adequacy_guard  # noqa: E402
from otsl.fixpolicy import fix_from_ranking, release_conflicting_rows  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

RES = os.path.join("results", "recursive")
OUT = os.path.join("data", "generated", "recursive")
R12 = os.path.join("results", "uc12")
K_CURVE = (1, 2, 4, 8, 16, 32)


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


class Inst:
    def __init__(self, d, i, rel):
        self.sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        self.d1 = dict(load=d["load"][i:i + 1], avail=d["avail"][i:i + 1], u0=d["u0"][i:i + 1], sr=d["sr"][i:i + 1],
                       u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None], c_rel=np.array([rel.obj]))


def lp(m, inst, u):
    sol, dt = timed(m.solve_dispatch, inst.sc, u)
    return dict(cost=float(sol.obj), shed=float(sol.shed), short=float(sol.short)), dt


def screen(m, inst, us):
    """price distinct schedules, keep the cheapest; returns (record, LP seconds, n LPs)"""
    seen, t = {}, 0.0
    for u in us:
        k = u.tobytes()
        if k not in seen:
            seen[k], dt = lp(m, inst, u)
            t += dt
    best = min(seen.values(), key=lambda r: r["cost"])
    return dict(best), t, len(seen)


def fallback_fix(q, u, sc, sysm, ratio=0.9):
    """fixings of the safety-net reduced MILP: the model's decoded schedule on its `ratio` most confident decisions
    (|q - u| smallest, q = allocated probabilities), then the adequacy guard and the min up/down row release"""
    err = np.abs(q - u)
    fix = fix_from_ranking(err, u, ratio)
    fix, _ = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
    fix, _ = release_conflicting_rows(fix, sysm, sc.u0)
    return fix


class TRMRunner:
    def __init__(self, tag, sel, sysm):
        self.tag, self.sel, self.s = tag, sel, sysm
        self.model, self.feats, self.cfg, _ = trm.load_model(os.path.join(OUT, "models", f"{sel['ckpt']}.pt"), sysm)
        self.kmax = max(K_CURVE) if self.cfg.mode == "trm" else 1

    @torch.inference_mode()
    def infer(self, d1):
        """per-step probabilities and cumulative inference times (single instance, as deployed)"""
        t0 = time.time()
        xt, xb = self.feats(d1)
        t_feat = time.time() - t0
        m = self.model
        T, G = xt.shape[1], xt.shape[2]
        P, Q, tk = [], [], []
        t0 = time.time()
        with trm.autocast(self.cfg, "cpu"):
            x, bus = m.inputs(xt, xb)
            y, z = m.init_state(1, T, G, xt)
            for k in range(self.kmax):
                y, z, logit, q = m.step(x, bus, y, z, grad=False)
                P.append(torch.sigmoid(logit[0]).float().numpy().astype(np.float64))
                Q.append(float(q[0]))
                tk.append(time.time() - t0)
        return P, Q, tk, t_feat

    def run(self, m, inst, t_rel, rec, gstar):
        s, sel, tag = self.s, self.sel, self.tag
        P, Q, tk, t_feat = self.infer(inst.d1)
        t_base = (t_rel if self.cfg.lp_feats else 0.0) + t_feat
        sg = trm.subgroups(s, inst.d1["u0"])[0] if self.cfg.sym == "count" else None
        how, th, kd = sel["decoder"], sel["th"], sel["k_def"]
        dec = lambda k, t=th: timed(trm.decode, P[min(k, len(P)) - 1], inst.d1, 0, s, t, how, sg)
        # 1 LP at the training step count
        u1, t_d = dec(kd)
        r, t_lp = lp(m, inst, u1)
        rec[f"{tag}_1"] = dict(r, time=t_base + tk[min(kd, len(tk)) - 1] + t_d + t_lp, n_lp=1, k=kd)
        if self.cfg.mode == "trm":
            for k in K_CURVE:
                u, t_d2 = dec(k)
                r2, t_lp2 = lp(m, inst, u)
                rec[f"{tag}_k{k}"] = dict(r2, time=t_base + tk[k - 1] + t_d2 + t_lp2, n_lp=1, k=k)
            ks = list(range(max(1, kd - 4), kd + 1))
            us, t_ds = [], 0.0
            for k in ks:
                u, t_d2 = dec(k)
                us.append(u)
                t_ds += t_d2
            r2, t_lps, n = screen(m, inst, us)
            rec[f"{tag}_s5"] = dict(r2, time=t_base + tk[kd - 1] + t_ds + t_lps, n_lp=n, k=kd)
            kh = next((k + 1 for k, q in enumerate(Q) if q > 0), self.kmax)
            u, t_d2 = dec(kh)
            r2, t_lp2 = lp(m, inst, u)
            rec[f"{tag}_halt"] = dict(r2, time=t_base + tk[kh - 1] + t_d2 + t_lp2, n_lp=1, k=kh)
        if sel.get("screen_th"):
            us, t_ds = [], 0.0
            for t_ in sel["screen_th"]:
                u, t_d2 = dec(kd, t_)
                us.append(u)
                t_ds += t_d2
            r2, t_lps, n = screen(m, inst, us)
            rec[f"{tag}_t5"] = dict(r2, time=t_base + tk[min(kd, len(tk)) - 1] + t_ds + t_lps, n_lp=n, k=kd)
        if gstar is not None:          # safety net on the 1-LP pipeline
            base = rec[f"{tag}_1"]
            g = (base["cost"] - inst.d1["c_rel"][0]) / base["cost"]
            fire = (base["shed"] > 1e-6) or (base["short"] > 1e-6) or g >= gstar
            out = dict(base, fired=bool(fire), bound_gap=float(g))
            if fire:
                q = trm.allocate(P[kd - 1], sg) if sg else P[kd - 1]
                fix, t_fix = timed(fallback_fix, q, u1, inst.sc, s)
                red = solve_reduced(m, inst.sc, fix, 60.0, 5e-3)
                out["time"] = base["time"] + t_fix + red["time"]
                out["fixed"] = len(fix) / u1.size
                if red["feasible"] and red["obj"] < base["cost"]:
                    out.update(cost=red["obj"], shed=red["shed"], short=red["short"], from_milp=True)
            rec[f"{tag}_fb"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--plan", default=os.path.join(RES, "trm_select.json"))
    ap.add_argument("--no_milp", action="store_true", help="skip the full MILPs (dry runs only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12)
    d = trm.load_npz(os.path.join(trm.ROOT, f"{a.split}.npz"))
    plan = json.load(open(a.plan))
    out = a.out or os.path.join(RES, f"trm_eval_{a.split}_s{a.shard}.jsonl")
    done = {json.loads(line)["i"] for line in open(out)} if os.path.exists(out) else set()
    jobs = [i for i in range(a.n)[a.shard::a.nshards] if i not in done]
    # models
    from otsl.m1x import build_model
    from otsl.ucml import UCFeaturizer
    from uc_constrained import load_model1, strip
    from otsl.combo import block_decode
    tr = trm.load_npz(os.path.join(trm.ROOT, "train.npz"))
    feat = UCFeaturizer(sysm, strip(tr), relax=True, sym=True)
    gnnt = []
    for s_ in range(5):
        m1 = build_model("gnnt", sysm, feat, 12, seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(R12, f"m1x_pol_n2000_gnnt_s{s_}.pt")))
        m1.net.eval()
        gnnt.append(m1)
    lag = load_model1(os.path.join(R12, "constrained_lag_D.pt"), sysm, strip(tr), 12)
    rl = load_model1(os.path.join(R12, "constrained_rl_lf.pt"), sysm, strip(tr), 12)
    runners = [TRMRunner(tag, sel, sysm) for tag, sel in plan["trm"].items()]
    m1sel = plan["m1x"]
    # warm-up (first-call overheads), not recorded
    rel0 = m.solve_dispatch(Inst(d, 0, m.solve_dispatch(UCScenario(load=d["load"][0], avail=d["avail"][0], u0=d["u0"][0],
                                                                  sr=d["sr"][0]), None, relax=True)).sc, None, relax=True)
    w = Inst(d, 0, rel0)
    for m1 in gnnt + [lag, rl]:
        m1.predict(w.d1)
    for rn in runners:
        rn.infer(w.d1)
    print(f"shard {a.shard}/{a.nshards} on CPU {sorted(os.sched_getaffinity(0))}: {len(jobs)} instances ({len(done)} done); "
          f"TRM pipelines {[r.tag for r in runners]}", flush=True)
    t_start = time.time()
    for j, i in enumerate(jobs):
        sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        rec = dict(i=int(i), split=a.split, shard=a.shard, cpu=sorted(os.sched_getaffinity(0)))
        if not a.no_milp:
            for name, gap in (("milp01", 1e-3), ("milp05", 5e-3)):
                r = solve_reduced(m, sc, None, 60.0, gap)
                rec[name] = dict(cost=r["obj"], shed=r["shed"], short=r["short"], time=r["time"], bound=r["bound"],
                                 status=r["status"], mip_gap=r["mip_gap"])
        rel, t_rel = timed(m.solve_dispatch, sc, None, relax=True)
        inst = Inst(d, i, rel)
        rec["relax"] = dict(cost=float(rel.obj), time=t_rel, urel_diff_stored=float(np.abs(rel.u - d["u_rel"][i]).max()))
        # LP rounding + repair + 5 LPs (no learning)
        b, t_b = timed(lp_round_screen, m, sysm, sc, rel.u)
        rec["lpround5"] = dict(cost=b["cost"], shed=b["shed"], short=b["short"], time=t_rel + t_b, n_lp=b["n_lp"])
        # m1x ensemble
        p, t_p = timed(lambda: np.mean([m1.predict(inst.d1)[0].astype(np.float64) for m1 in gnnt], 0))
        u, t_d = timed(trm.decode, p, inst.d1, 0, sysm, m1sel["th"], m1sel["decoder"], None)
        r, t_lp = lp(m, inst, u)
        rec["m1x_1"] = dict(r, time=t_rel + t_p + t_d + t_lp, n_lp=1)
        us, t_ds = [], 0.0
        for th in m1sel["screen_th"]:
            u, t_d = timed(trm.decode, p, inst.d1, 0, sysm, th, m1sel["decoder"], None)
            us.append(u)
            t_ds += t_d
        r, t_lps, n = screen(m, inst, us)
        rec["m1x_5"] = dict(r, time=t_rel + t_p + t_ds + t_lps, n_lp=n)
        # REINFORCE end-to-end pipelines (combo study, seed 0, its validation-calibrated choices)
        p, t_p = timed(lambda: lag.predict(inst.d1).astype(np.float64))
        us, t_ds = [], 0.0
        for th in THRESHOLDS:
            u, t_d = timed(block_decode, p, inst.d1, sysm, th)
            us.append(u[0])
            t_ds += t_d
        r, t_lps, n = screen(m, inst, us)
        rec["rl_lag7"] = dict(r, time=t_rel + t_p + t_ds + t_lps, n_lp=n)
        p, t_p = timed(lambda: rl.predict(inst.d1).astype(np.float64))
        u, t_d = timed(block_decode, p, inst.d1, sysm, 0.9)
        r, t_lp = lp(m, inst, u[0])
        rec["rl_rl1"] = dict(r, time=t_rel + t_p + t_d + t_lp, n_lp=1)
        for rn in runners:
            rn.run(m, inst, t_rel, rec, plan.get("gstar", {}).get(rn.tag))
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        ms = rec.get("milp01", {})
        print(f"[{j + 1}/{len(jobs)} {time.time() - t_start:.0f}s] i={i} milp01 {ms.get('time', float('nan')):.1f}s "
              + " ".join(f"{k} {v['time']:.2f}s" for k, v in rec.items() if isinstance(v, dict) and "time" in v
                         and k.endswith(("_1", "lpround5", "_s5"))), flush=True)


if __name__ == "__main__":
    main()
