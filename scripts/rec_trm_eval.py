"""Test evaluation of the TRM study (arm A) with paired timing: for each instance, in ONE process pinned to one core,
the full MILP at 0.1 % (dual bound, speed-up denominator) and at 0.5 %, the LP relaxation, the baselines and every TRM
pipeline, back to back. Read the test set once, after every choice was fixed on validation.

    taskset -c 0 python3 scripts/rec_trm_eval.py --split test_fresh --shard 0 --nshards 2
    taskset -c 1 python3 scripts/rec_trm_eval.py --split test_fresh --shard 1 --nshards 2

Every end-to-end pipeline comes in two variants: plain (as specified: the decoded schedule's dispatch LP, or the
cheapest of <= 5 screened schedules) and "r" (+ the LP-guided adequacy repair of otsl.trm.lp_repair when that LP sheds
load or misses reserve: <= 2 extra LPs), each with its own validation-selected decoder / threshold.
Time = every overhead: LP relaxation where used, features, network passes, decoding / repairs, LPs.
  milp01, milp05               full MILP (highspy, 1 thread, 60 s) at mip_rel_gap 0.1 % / 0.5 %
  lpround5[r]                  LP rounding at 5 thresholds + block adequacy / min up/down repair, <= 5 LPs (no learning)
  m1x_1[r], m1x_5[r]           m1x temporal-GNN ensemble (5 members): 1 LP; 5 validation-selected thresholds
  rl_lag7[r], rl_rl1[r]        REINFORCE end-to-end pipelines of docs/methods/combo.md (seed 0, their own validation
                               choices): Lagrangian + KL model screened over 7 thresholds; label-free + REINFORCE, 1 LP
  <tag>_1[r]                   a TRM model, 1 LP at its training step count
  <tag>_s5[r], <tag>_t5[r]     screening the last 5 steps' schedules / 5 thresholds
  <tag>_mix5[r], m1x_mix5[r]   mixed screening: the model's 1-LP schedule + LP roundings at 4 validation-chosen
                               thresholds (<= 5 LPs)
  <tag>_k<k>[r]                1 LP after k supervision steps (test-time compute curve)
  <tag>_halt[r]                stop at the first step whose halting logit is > 0
  <tag>_fb                     <tag>_1r + safety net: if it still sheds / misses reserve or its bound gap
                               (C - z_LP) / C >= g* (validation), reduced MILP (0.5 %, 60 s) around the model's 90 % most
                               confident decisions (+ adequacy guard, min up/down rows); keep the cheaper schedule
Plan (tags, selections, g*): results/recursive/trm_select.json (scripts/rec_trm_select.py, validation only).
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
from otsl.base import lp_round_candidates, solve_reduced  # noqa: E402
from otsl.combo import THRESHOLDS, adequacy_guard, block_decode  # noqa: E402
from otsl.fixpolicy import fix_from_ranking, release_conflicting_rows  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

RES = os.path.join("results", "recursive")
OUT = os.path.join("data", "generated", "recursive")
R12 = os.path.join("results", "uc12")
K_CURVE = (1, 2, 4, 8, 16, 32)
LPROUND_TH = (0.001, 0.05, 0.2, 0.5, 0.8)


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


class Inst:
    def __init__(self, d, i, rel):
        self.sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        self.d1 = dict(load=d["load"][i:i + 1], avail=d["avail"][i:i + 1], u0=d["u0"][i:i + 1], sr=d["sr"][i:i + 1],
                       u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None], c_rel=np.array([rel.obj]))
        self.cache = {}


class Ctx:
    def __init__(self, m, sysm):
        self.m, self.s = m, sysm

    def lp(self, inst, u):
        """dispatch LP of u (memoised per instance: the same schedule priced by two pipelines is timed for each, the
        solve itself is reused only for the record, its time is charged every time)"""
        key = u.tobytes()
        if key not in inst.cache:
            sol, dt = timed(self.m.solve_dispatch, inst.sc, u)
            inst.cache[key] = (sol, dt)
        sol, dt = inst.cache[key]
        return sol, dt

    def finish(self, inst, us, t_before, rep):
        """price the distinct schedules us, keep the cheapest; with rep, LP-guided repair if it fails.
        Returns a record (cost, shed, short, time, n_lp)."""
        best, t_lp, seen = None, 0.0, set()
        for u in us:
            if u.tobytes() in seen:
                continue
            seen.add(u.tobytes())
            sol, dt = self.lp(inst, u)
            t_lp += dt
            if best is None or sol.obj < best[1].obj - 1e-9:
                best = (u, sol)
        u, sol = best
        n, t_rep = len(seen), 0.0
        if rep and (sol.shed > 1e-6 or sol.short > 1e-6):
            (u, sol, k, _), t_rep = timed(trm.lp_repair, self.m, inst.sc, u, self.s, sol)
            n += k
        return dict(cost=float(sol.obj), shed=float(sol.shed), short=float(sol.short), time=t_before + t_lp + t_rep,
                    n_lp=n), u


def fallback_fix(q, u, sc, sysm, ratio=0.9):
    """fixings of the safety-net reduced MILP: the model's schedule on its `ratio` most confident decisions
    (|q - u| smallest, q = allocated probabilities), then the adequacy guard and the min up/down row release"""
    fix = fix_from_ranking(np.abs(q - u), u, ratio)
    fix, _ = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
    fix, _ = release_conflicting_rows(fix, sysm, sc.u0)
    return fix


def prob_pipelines(ctx, inst, name, p_fn, t_base, sel, rounds, t_rounds, sg=None):
    """1-LP, 5-threshold and mixed (model + 4 LP roundings) pipelines of a one-pass probability model (plain and r)"""
    rec = {}
    p, t_p = timed(p_fn)
    for kind, sfx in (("plain", ""), ("rep", "r")):
        s = sel[kind]
        u1, t_d1 = timed(trm.decode, p, inst.d1, 0, ctx.s, s["th"], s["decoder"], sg)
        rec[f"{name}_1{sfx}"], _ = ctx.finish(inst, [u1], t_base + t_p + t_d1, kind == "rep")
        us, t_ds = [], 0.0
        for th in s["screen_th"]:
            u, t_d = timed(trm.decode, p, inst.d1, 0, ctx.s, th, s.get("screen_dec", s["decoder"]), sg)
            us.append(u)
            t_ds += t_d
        rec[f"{name}_5{sfx}"], _ = ctx.finish(inst, us, t_base + t_p + t_ds, kind == "rep")
        if s.get("mix_th"):
            rec[f"{name}_mix5{sfx}"], _ = ctx.finish(inst, [u1] + [rounds[t] for t in s["mix_th"]],
                                                     t_base + t_p + t_d1 + t_rounds, kind == "rep")
    return rec


class TRMRunner:
    def __init__(self, tag, sel, sysm, device="cpu"):
        self.tag, self.sel, self.s, self.dev = tag, sel, sysm, device
        self.model, self.feats, self.cfg, _ = trm.load_model(os.path.join(OUT, "models", f"{sel['ckpt']}.pt"), sysm,
                                                             device)
        self.kmax = max(K_CURVE) if self.cfg.mode == "trm" else 1

    def _sync(self):
        if self.dev != "cpu":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def infer(self, d1):
        """per-step probabilities and cumulative inference times (single instance, as deployed; on a GPU the times
        include the transfers and are synchronised)"""
        t0 = time.time()
        xt, xb = self.feats(d1, self.dev)
        self._sync()
        t_feat = time.time() - t0
        m = self.model
        T, G = xt.shape[1], xt.shape[2]
        P, Q, tk, H = [], [], [], []
        t0 = time.time()
        with trm.autocast(self.cfg, self.dev):
            x, bus = m.inputs(xt, xb)
            y, z = m.init_state(1, T, G, xt)
            for k in range(self.kmax):
                y, z, logit, q = m.step(x, bus, y, z, grad=False)
                P.append(torch.sigmoid(logit[0]).float().cpu().numpy().astype(np.float64))
                Q.append(float(q[0]))
                H.append(y[0].float().mean((0, 1)).cpu().numpy())
                self._sync()
                tk.append(time.time() - t0)
        self.H = H
        return P, Q, tk, t_feat

    def run(self, ctx, inst, t_rel, rec, gstar, rounds=None, t_rounds=0.0):
        s, tag, kd = self.s, self.tag, self.sel["k_def"]
        P, Q, tk, t_feat = self.infer(inst.d1)
        t_base = (t_rel if self.cfg.lp_feats else 0.0) + t_feat
        sg = trm.subgroups(s, inst.d1["u0"])[0] if self.cfg.sym == "count" else None
        K = len(P)

        def dec(k, kind, th=None):
            sl = self.sel[kind]
            how = sl["decoder"] if th is None else sl.get("screen_dec", sl["decoder"])
            return timed(trm.decode, P[min(k, K) - 1], inst.d1, 0, s, sl["th"] if th is None else th, how, sg)

        def ti(k):
            return tk[min(k, K) - 1]

        for kind, sfx in (("plain", ""), ("rep", "r")):
            rep = kind == "rep"
            u, t_d = dec(kd, kind)
            rec[f"{tag}_1{sfx}"], u_fin = ctx.finish(inst, [u], t_base + ti(kd) + t_d, rep)
            rec[f"{tag}_1{sfx}"]["k"] = kd
            if self.sel[kind].get("mix_th") and rounds is not None:
                # the LP roundings need the LP relaxation even when the model's features do not
                t_mix = t_base + ti(kd) + t_d + t_rounds + (0.0 if self.cfg.lp_feats else t_rel)
                rec[f"{tag}_mix5{sfx}"], _ = ctx.finish(inst, [u] + [rounds[t] for t in self.sel[kind]["mix_th"]],
                                                        t_mix, rep)
            if rep:
                u1r = u_fin
            us, t_ds = [], 0.0
            for th in self.sel[kind].get("screen_th") or []:
                u, t_d = dec(kd, kind, th)
                us.append(u)
                t_ds += t_d
            if us:
                rec[f"{tag}_t5{sfx}"], _ = ctx.finish(inst, us, t_base + ti(kd) + t_ds, rep)
            if self.cfg.mode != "trm":
                continue
            for k in K_CURVE:
                u, t_d = dec(k, "rep")          # the curve uses the repaired pipeline's decoder / threshold
                rec[f"{tag}_k{k}{sfx}"], _ = ctx.finish(inst, [u], t_base + ti(k) + t_d, rep)
            us, t_ds = [], 0.0
            for k in range(max(1, kd - 4), kd + 1):
                u, t_d = dec(k, kind)
                us.append(u)
                t_ds += t_d
            rec[f"{tag}_s5{sfx}"], _ = ctx.finish(inst, us, t_base + ti(kd) + t_ds, rep)
            kh = next((k + 1 for k, q in enumerate(Q) if q > 0), K)
            u, t_d = dec(kh, kind)
            rec[f"{tag}_halt{sfx}"], _ = ctx.finish(inst, [u], t_base + ti(kh) + t_d, rep)
            rec[f"{tag}_halt{sfx}"]["k"] = kh
        if self.cfg.mode == "trm" and self.sel.get("halt_w") is not None:   # post-hoc halting head (LP-priced targets)
            w, b, kcap = np.array(self.sel["halt_w"]), self.sel["halt_b"], self.sel.get("halt_K", 16)
            kh = next((k + 1 for k in range(min(kcap, K)) if float(self.H[k] @ w + b) > 0), min(kcap, K))
            u, t_d = dec(kh, "rep")
            rec[f"{tag}_halt2r"], _ = ctx.finish(inst, [u], t_base + ti(kh) + t_d, True)
            rec[f"{tag}_halt2r"]["k"] = kh
        if gstar is not None:          # safety net on top of the repaired 1-LP pipeline
            base = rec[f"{tag}_1r"]
            g = (base["cost"] - inst.d1["c_rel"][0]) / base["cost"]
            fire = base["shed"] > 1e-6 or base["short"] > 1e-6 or g >= gstar
            out = dict(base, fired=bool(fire), bound_gap=float(g), from_milp=False)
            if fire:
                q = trm.allocate(P[kd - 1], sg) if sg else P[kd - 1]
                fix, t_fix = timed(fallback_fix, q, u1r, inst.sc, s)
                red = solve_reduced(ctx.m, inst.sc, fix, 60.0, 5e-3)
                out["time"] = base["time"] + t_fix + red["time"]
                out["fixed"] = len(fix) / u1r.size
                if red["feasible"] and red["obj"] < base["cost"]:
                    out.update(cost=red["obj"], shed=red["shed"], short=red["short"], from_milp=True)
            rec[f"{tag}_fb"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--plan", required=True, help="test plan from scripts/rec_trm_select.py (no default: the "
                    "CPU-round plan results/recursive/trm_select.json must not be picked up by accident)")
    ap.add_argument("--no_milp", action="store_true", help="skip the full MILPs (dry runs only)")
    ap.add_argument("--out", required=True, help="this run's own records (jsonl); resuming reads only this file")
    ap.add_argument("--round", default="", help="label stored in every record (e.g. gpu)")
    ap.add_argument("--trm_device", default="cpu", help="device of the TRM forward passes (LPs / MILPs stay on the CPU)")
    ap.add_argument("--only", default="", help="comma list of instance indices (overrides --n / --shard / --nshards)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12)
    ctx = Ctx(m, sysm)
    d = trm.load_npz(os.path.join(trm.ROOT, f"{a.split}.npz"))
    plan = json.load(open(a.plan))
    out = a.out
    done = {json.loads(line)["i"] for line in open(out)} if os.path.exists(out) else set()
    jobs = [i for i in range(a.n)[a.shard::a.nshards] if i not in done]
    if a.only:
        jobs = [int(x) for x in a.only.split(",") if x and int(x) not in done]
    from otsl.m1x import build_model
    from otsl.ucml import UCFeaturizer
    from uc_constrained import load_model1, strip
    tr = trm.load_npz(os.path.join(trm.ROOT, "train.npz"))
    feat = UCFeaturizer(sysm, strip(tr), relax=True, sym=True)
    gnnt = []
    for s_ in range(5):
        m1 = build_model("gnnt", sysm, feat, 12, seed=0)
        m1.net.load_state_dict(torch.load(os.path.join(R12, f"m1x_pol_n2000_gnnt_s{s_}.pt")))
        m1.net.eval()
        gnnt.append(m1)
    # the REINFORCE end-to-end baselines of the combo study; their weights exist only where that study ran
    lag_p, rl_p = os.path.join(R12, "constrained_lag_D.pt"), os.path.join(R12, "constrained_rl_lf.pt")
    have_rl = os.path.exists(lag_p) and os.path.exists(rl_p)
    lag = load_model1(lag_p, sysm, strip(tr), 12) if have_rl else None
    rl = load_model1(rl_p, sysm, strip(tr), 12) if have_rl else None
    if not have_rl:
        print("REINFORCE end-to-end baselines (rl_lag7, rl_rl1) skipped: their weights are not here", flush=True)
    runners = [TRMRunner(tag, sel, sysm, a.trm_device) for tag, sel in plan["trm"].items()]
    # warm-up (first-call overheads), not recorded
    sc0 = UCScenario(load=d["load"][0], avail=d["avail"][0], u0=d["u0"][0], sr=d["sr"][0])
    w = Inst(d, 0, m.solve_dispatch(sc0, None, relax=True))
    for m1 in gnnt + ([lag, rl] if have_rl else []):
        m1.predict(w.d1)
    for rn in runners:
        rn.infer(w.d1)
    m.solve_dispatch(sc0, d["u"][0])
    print(f"shard {a.shard}/{a.nshards} on CPU {sorted(os.sched_getaffinity(0))}: {len(jobs)} instances ({len(done)} done); "
          f"TRM pipelines {[r.tag for r in runners]}", flush=True)
    t_start = time.time()
    for j, i in enumerate(jobs):
        sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
        rec = dict(i=int(i), split=a.split, shard=a.shard, cpu=sorted(os.sched_getaffinity(0)), round=a.round,
                   plan=os.path.basename(a.plan))
        if not a.no_milp:
            for name, gap in (("milp01", 1e-3), ("milp05", 5e-3)):
                r = solve_reduced(m, sc, None, 60.0, gap)
                rec[name] = dict(cost=r["obj"], shed=r["shed"], short=r["short"], time=r["time"], bound=r["bound"],
                                 status=r["status"], mip_gap=r["mip_gap"])
        rel, t_rel = timed(m.solve_dispatch, sc, None, relax=True)
        inst = Inst(d, i, rel)
        rec["relax"] = dict(cost=float(rel.obj), time=t_rel, urel_diff_stored=float(np.abs(rel.u - d["u_rel"][i]).max()))
        # LP rounding + repair + 5 LPs (no learning)
        cands, t_c = timed(lp_round_candidates, rel.u, sc, sysm, LPROUND_TH)
        rounds = {float(t): u for t, u in cands}
        for sfx, rep in (("", False), ("r", True)):
            rec[f"lpround5{sfx}"], _ = ctx.finish(inst, [u for _, u in cands], t_rel + t_c, rep)
        # m1x ensemble
        rec.update(prob_pipelines(ctx, inst, "m1x", lambda: np.mean([m1.predict(inst.d1)[0].astype(np.float64)
                                                                     for m1 in gnnt], 0), t_rel, plan["m1x"], rounds, t_c))
        # REINFORCE end-to-end pipelines (combo study, seed 0, its own validation-calibrated choices)
        if have_rl:
            p, t_p = timed(lambda: lag.predict(inst.d1).astype(np.float64))
            us, t_ds = [], 0.0
            for th in THRESHOLDS:
                u, t_d = timed(block_decode, p, inst.d1, sysm, th)
                us.append(u[0])
                t_ds += t_d
            p2, t_p2 = timed(lambda: rl.predict(inst.d1).astype(np.float64))
            u2, t_d2 = timed(block_decode, p2, inst.d1, sysm, 0.9)
            for sfx, rep in (("", False), ("r", True)):
                rec[f"rl_lag7{sfx}"], _ = ctx.finish(inst, us, t_rel + t_p + t_ds, rep)
                rec[f"rl_rl1{sfx}"], _ = ctx.finish(inst, [u2[0]], t_rel + t_p2 + t_d2, rep)
        for rn in runners:
            rn.run(ctx, inst, t_rel, rec, plan.get("gstar", {}).get(rn.tag), rounds, t_c)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        ms = rec.get("milp01", {})
        print(f"[{j + 1}/{len(jobs)} {time.time() - t_start:.0f}s] i={i} milp01 {ms.get('time', float('nan')):.1f}s "
              + " ".join(f"{k} {v['time']:.2f}s" for k, v in rec.items() if isinstance(v, dict) and "time" in v
                         and k.endswith(("_1r", "lpround5r", "_s5r", "_fb"))), flush=True)


if __name__ == "__main__":
    main()
