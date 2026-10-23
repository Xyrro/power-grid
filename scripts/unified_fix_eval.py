"""Step 2 of docs/unified_metrics.md: the fixing rules, the no-learning baselines and the early one-shot models of one
test instance, back to back in one process on one core, in a random order (seeded by the instance).

    OTSL_THREADS=1 python3 scripts/unified_fix_eval.py --bench uc12 --set tf --k 7 --out results/unified/retime/uc12_fix.jsonl
    OTSL_THREADS=1 python3 scripts/unified_fix_eval.py --bench uc12 --set gen_out --k 3 --out ...
    OTSL_THREADS=1 python3 scripts/unified_fix_eval.py --bench uc24 --k 5 --out results/unified/retime/uc24_fix.jsonl

Usually started by scripts/unified_retime.py, which runs this and the recursive arms' own evaluation scripts for the
same instance back to back on the same core.

Instance. uc12: test_fresh (--set tf, data/generated/uc12/test_fresh.npz) or a shifted set (data/generated/uc12_ood/
<set>/<k>.npz, outages through otsl.ood exactly as in scripts/uc_m1x_pipeline.py); uc24: data/generated/uc24/test.npz.
The LP relaxation is solved once (timed) and its time is charged to every method that reads it.

Methods (names as in the records; nothing is tuned here):
  full@g            full MILP at mip_rel_gap g (uc12: 60 s limit, uc24: 300 s), highspy, one thread; full@0.1% is the
                    speed-up reference and gives a dual bound
  lpround           LP relaxation rounded at 5 thresholds + block adequacy and min up/down repair, dispatch LP of each,
                    cheapest kept (otsl.base.lp_round_screen; no learning, no MILP)
  lpint             fix every unit-hour whose LP value is integral (val-selected tolerance), no guard (Y1)
  lpint+g           the same at the guard tolerance + adequacy guard, min up/down row release, LP-relaxation guard (Y1)
  ltf_lp            Learning to Fix's thresholds on the LP-relaxation values (results/<bench>/base_tune_lp_1.json)
  ltf_knn_e         Learning to Fix, kNN (k = 50), thresholds tuned at eps = e % on this cluster's validation set
  ltf_knn_1+lpg     the eps = 1 % fixings, then row release and the LP-relaxation guard (the robustness study's fix)
  ltf_bce_1         Learning to Fix's thresholds (eps = 1 %, tuned here) on the MILP-label BCE GNN (uc_model1_4.pt)
  pipe@g            recommended pipeline: 5-member temporal-GNN ensemble -> guard-aware thresholds (tuned here) ->
                    adequacy, rows, LP-relaxation guard -> reduced MILP at g
  pipe+bc           pipe@0.5% plus the post-solve bound check: if (C - z_LP) / C > 4 % (results/uc12/cguard_select.json),
                    also the full MILP at 0.5 %, the better schedule kept, both times charged
  pipe@0.5%:<e>, pipe+bc:<e>   the same with another training run of the pipeline (--ens_extra: an ensemble of five other
                    seeds with its own thresholds tuned on validation; training-seed spread, test_fresh only)
  <rule>:orig       diagnostic (--orig): ltf_knn_1 / ltf_knn_5 / ltf_bce_1 / pipe@0.5% with this cluster's models but the
                    original study's committed thresholds (results/uc12/), to separate the thresholds' share of a
                    difference from the models' (test_fresh only)
  e2e:<model>:<dec> the early one-shot models (scripts/uc_model1.py, retrained here; test_fresh only): MLP and GNN on MILP
                    labels (the framework), GNN + symmetry, GNN + symmetry + LP-relaxation features, and the latter
                    fine-tuned by REINFORCE; decoders top1 (round at 0.5, min up/down repair, 1 LP), adeq (+ per-hour
                    adequacy repair), block (+ block adequacy repair), screen (7 thresholds + 8 samples, cheapest by LP)
  knn<k>            kNN over training schedules (k = 5, 20): each neighbour's schedule, min up/down-repaired, priced by
                    the dispatch LP, cheapest kept (Xavier et al. 2021 style; test_fresh only)
A reduced MILP without a solution falls back to the full MILP at the same gap; that solve's time is added ("fallback").
Every delivered schedule is priced afterwards by the exact dispatch LP (not timed): cost with penalties, shed + spill,
reserve shortfall. Times are wall seconds around each method's own calls (model build, inference, guards, every LP
and MILP), plus the LP relaxation where used; process CPU seconds are recorded alongside.
Output: one JSON line per instance (resume skips instances already in --out).
"""
import argparse
import json
import os
import platform
import socket
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.b3 import load_b3  # noqa: E402
from otsl.base import lp_integral_fixings, lp_round_screen  # noqa: E402
from otsl.cguard import bound_gap  # noqa: E402
from otsl.constrained import adequacy_repair_blocks  # noqa: E402
from otsl.hybrid import ALL_GUARDS, apply_guards  # noqa: E402
from otsl.ltfx import KNNProb, fix_dict  # noqa: E402
from otsl.ood import OODModel, OODScenario, avail_view, fixed_share, force_out, override_probs, solve_milp  # noqa: E402
from otsl.uc import adequacy_repair, load_rts_gmlc, repair_min_updown  # noqa: E402

SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
CFG = {"uc12": dict(T=12, tl=60.0, data="data/generated/uc12/test_fresh.npz", res="results/uc12"),
       "uc24": dict(T=24, tl=300.0, data="data/generated/uc24/test.npz", res="results/uc24")}
G01, G05 = 1e-3, 5e-3
D_POST = json.load(open(os.path.join(os.path.dirname(HERE), "results/uc12/cguard_select.json")))["d_post"]
EARLY = (("mlp", "uc_model1_1.pt", "mlp", dict(relax=False, sym=False)),
         ("gnn", "uc_model1_2.pt", "gnn", dict(relax=False, sym=False)),
         ("gnn_sym", "uc_model1_3.pt", "gnn", dict(relax=False, sym=True)),
         ("gnn_lp", "uc_model1_4.pt", "gnn", dict(relax=True, sym=True)),
         ("gnn_lp_rl", "uc_model1_rl.pt", "gnn", dict(relax=True, sym=True)))
KEYS = ("load", "avail", "u0", "sr", "start", "u_rel", "lmp_rel", "flow_rel")


def host_info():
    model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        model = platform.processor()
    import highspy
    h = highspy.Highs()
    return dict(host=socket.gethostname(), cpu_model=model,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                highs=f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}", torch=torch.__version__,
                torch_threads=torch.get_num_threads(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"))


def load_thr(path):
    j = json.load(open(path))
    return np.array(j["lo"]), np.array(j["hi"])


class Clock:
    """wall and process-CPU seconds of a method, summed over the pieces it is timed in"""

    def __init__(self):
        self.wall, self.cpu, self.parts = 0.0, 0.0, {}

    def __call__(self, part, f, *a, **k):
        t0, c0 = time.perf_counter(), time.process_time()
        r = f(*a, **k)
        dw, dc = time.perf_counter() - t0, time.process_time() - c0
        self.wall += dw
        self.cpu += dc
        self.parts[part] = self.parts.get(part, 0.0) + dw
        return r

    def add(self, part, wall, cpu=None):
        self.wall += wall
        self.cpu += wall if cpu is None else cpu
        self.parts[part] = self.parts.get(part, 0.0) + wall


class Ctx:
    """everything one instance's methods read: system, model, scenario, LP relaxation, model inputs"""

    def __init__(self, a, cfg):
        self.a, self.cfg, self.T = a, cfg, cfg["T"]
        self.sysm = load_rts_gmlc()
        self.m = OODModel(self.sysm, T=self.T)
        self.G = self.sysm.G
        self.tl = cfg["tl"]

    def instance(self, k):
        a = self.a
        if a.bench == "uc12" and a.set != "tf":
            z = np.load(os.path.join("data/generated/uc12_ood", a.set, f"{k:03d}.npz"))
            sc = OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"], day=int(z["day"]),
                             start=int(z["start"]), g_out=tuple(int(g) for g in z["g_out"]),
                             l_out=tuple(int(x) for x in z["l_out"]))
            stored_c_rel = float(z["c_rel"]) if "c_rel" in z.files else None
        else:
            d = self.data
            sc = OODScenario(load=d["load"][k], avail=d["avail"][k], u0=d["u0"][k], sr=d["sr"][k],
                             day=int(d["day"][k]), start=int(d["start"][k]))
            stored_c_rel = float(d["c_rel"][k])
        return sc, stored_c_rel


def milp(ck, m, sc, fix, tl, gap, part="milp"):
    """(reduced) MILP; without a solution, the full MILP at the same gap (fallback, time added)"""
    r = ck(part, solve_milp, m, sc, fix, tl, gap)
    fb = None
    if fix and not r["feasible"]:
        fb = ck("fallback", solve_milp, m, sc, {}, tl, gap)
    out = r if fb is None else fb
    info = dict(status=r["status"], obj=r["obj"], bound=r["bound"], mip_gap=r["mip_gap"], milp_s=r["time"],
                n_fixed=len(fix or {}), fallback=fb is not None)
    if fb is not None:
        info.update(fb_status=fb["status"], fb_obj=fb["obj"], fb_s=fb["time"])
    return out.get("u"), info


def price(m, sc, u):
    """the delivered schedule's exact cost (dispatch LP, penalties included); not timed"""
    if u is None:
        return dict(cost=float("inf"), shed=None, short=None)
    lp = m.solve_dispatch(sc, u)
    return dict(cost=float(lp.obj), shed=float(lp.shed), short=float(lp.short))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=sorted(CFG))
    ap.add_argument("--set", default="tf", choices=("tf",) + SHIFTS)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tune_dir", default="results/unified/retime/tune",
                    help="thresholds tuned on this cluster: ltfx_tune_{knn_1,knn_5,knn_10,bce_1}.json, m1x_tune_hgP.json")
    ap.add_argument("--hg", default="m1x_tune_hgP.json", help="pipeline thresholds file in --tune_dir")
    ap.add_argument("--early_dir", default="results/unified/retime/models", help="uc_model1_*.pt retrained here")
    ap.add_argument("--ens_extra", default="b=5-9:m1x_tune_hgPb.json,c=10-14:m1x_tune_hgPc.json",
                    help="other pipeline training runs: name=first-last member seed:thresholds file in --tune_dir")
    ap.add_argument("--orig", action="store_true", help="also the :orig diagnostic rules (original thresholds)")
    ap.add_argument("--only", default="", help="comma list of method names (smoke tests)")
    ap.add_argument("--seed", type=int, default=0, help="order seed (combined with the instance)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    if a.bench == "uc24":
        a.set = "tf"
    key = dict(bench=a.bench, set=a.set, k=a.k)
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if (r["bench"], r["set"], r["k"]) == (a.bench, a.set, a.k):
                print(f"{key} already in {a.out}", flush=True)
                return
    cfg = CFG[a.bench]
    ctx = Ctx(a, cfg)
    sysm, m, T, G, tl = ctx.sysm, ctx.m, ctx.T, ctx.G, ctx.tl
    tf_like = a.set == "tf"
    ctx.data = load_b3(cfg["data"]) if tf_like else None
    sc, stored_c_rel = ctx.instance(a.k)
    g_out = tuple(sc.g_out)
    sv = avail_view(sysm, g_out)
    tr = load_b3("data/generated/uc12/train.npz") if a.bench == "uc12" else None
    notes = []

    # ---------------------------------------------------------------- what this instance runs, and its artifacts
    sel = json.load(open(os.path.join(cfg["res"], "base_val_select.json")))
    tol_none, tol_g = sel["none"]["selected_tol"], sel["guards"]["selected_tol"]
    th_lp = load_thr(os.path.join(cfg["res"], "base_tune_lp_1.json"))
    tuned = {}
    for name, f in (("knn_1", "ltfx_tune_knn_1.json"), ("knn_5", "ltfx_tune_knn_5.json"),
                    ("knn_10", "ltfx_tune_knn_10.json"), ("bce_1", "ltfx_tune_bce_1.json"), ("hg", a.hg)):
        p = os.path.join(a.tune_dir, f)
        if a.bench == "uc12" and os.path.exists(p):
            tuned[name] = load_thr(p)
        elif a.bench == "uc12":
            notes.append(f"missing {p}: its methods skipped")
    knn = gnnt = bce = None
    early, extra_ens = {}, {}
    if a.bench == "uc12":
        if any(k.startswith("knn") for k in tuned) or tf_like:
            knn = KNNProb(sysm, tr, k=50)
        from uc_constrained import load_model1, strip
        if "bce_1" in tuned:
            bce = load_model1("results/uc12/uc_model1_4.pt", sysm, strip(tr), 12)
        if "hg" in tuned:
            from uc_m1x_train import featurizer, load_net
            feat = featurizer(sysm)
            gnnt = [load_net(f"results/uc12/m1x_pol_n2000_gnnt_s{s}.pt", "gnnt", sysm, feat) for s in range(5)]
        if tf_like and a.ens_extra:
            from uc_m1x_train import featurizer, load_net
            feat = featurizer(sysm)
            for spec in a.ens_extra.split(","):
                name, rest = spec.split("=")
                seeds, f = rest.split(":")
                s0, s1 = map(int, seeds.split("-"))
                paths = [f"data/generated/uc12_m1x/models/pol_n2000_gnnt_s{s}.pt" for s in range(s0, s1 + 1)]
                pt = os.path.join(a.tune_dir, f)
                if all(os.path.exists(q) for q in paths) and os.path.exists(pt):
                    extra_ens[name] = ([load_net(q, "gnnt", sysm, feat) for q in paths], load_thr(pt))
                else:
                    notes.append(f"ensemble {name} (seeds {seeds}) or {pt} missing: skipped")
        if tf_like:
            from otsl.ucml import UCFeaturizer, build_uc_model1
            for name, f, kind, fo in EARLY:
                p = os.path.join(a.early_dir, f)
                if not os.path.exists(p):
                    notes.append(f"missing {p}: e2e:{name} skipped")
                    continue
                m1 = build_uc_model1(sysm, UCFeaturizer(sysm, tr, **fo), T, kind, seed=0)
                m1.net.load_state_dict(torch.load(p))
                m1.net.eval()
                early[name] = (m1, fo["relax"])

    # ---------------------------------------------------------------- LP relaxation (timed once, charged where read)
    sc0, _ = ctx.instance(0) if tf_like else (sc, None)
    m.solve_dispatch(sc0, None, relax=True)                                  # warm-up (first-call overheads)
    t0, c0 = time.perf_counter(), time.process_time()
    rel = m.solve_dispatch(sc, None, relax=True)
    t_rel, c_rel_cpu = time.perf_counter() - t0, time.process_time() - c0
    di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], start=np.array([sc.start]),
              u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
    for net in (gnnt or []) + ([bce] if bce else []) + [x[0] for x in early.values()] + \
            [n for nets, _ in extra_ens.values() for n in nets]:
        net.predict(di)                                                      # warm-up of every network
    if knn is not None:
        knn.predict(di)

    def with_rel(ck):
        ck.add("lp_relaxation", t_rel, c_rel_cpu)
        return ck

    # ---------------------------------------------------------------- methods: name -> callable(ck) -> (u, info)
    meth = {}
    full_gaps = (G01, 0.0025, G05, 1e-2) if a.bench == "uc12" and tf_like else (G01, G05, 1e-2) if tf_like else (G01, G05)
    for g in full_gaps:
        meth[f"full@{g * 100:g}%"] = (lambda g: lambda ck: milp(ck, m, sc, {}, tl, g))(g)

    def rule(fix_fn, guards, gap, uses_rel=True):
        def run(ck):
            if uses_rel:
                with_rel(ck)
            fix = ck("rule", fix_fn)
            n_pre = len(fix)
            info = {}
            if guards:
                fix, info = ck("guards", apply_guards, m, sv, sc, dict(fix), guards)
                info = {k: v for k, v in info.items() if k != "guard_s"}
            fix = force_out(fix, g_out, T)
            u, mi = milp(ck, m, sc, fix, tl, gap)
            return u, dict(mi, n_fixed_pre=n_pre, fixed_share=fixed_share(fix, g_out, T, G), **info)
        return run

    meth["lpint"] = rule(lambda: lp_integral_fixings(rel.u, tol_none), (), G01)
    meth["lpint+g"] = rule(lambda: lp_integral_fixings(rel.u, tol_g), ALL_GUARDS, G01)
    meth["ltf_lp"] = rule(lambda: fix_dict(rel.u, *th_lp), (), G01)
    if a.bench == "uc24":
        meth["lpint+g@0.5%"] = rule(lambda: lp_integral_fixings(rel.u, tol_g), ALL_GUARDS, G05)
        meth["ltf_lp@0.5%"] = rule(lambda: fix_dict(rel.u, *th_lp), (), G05)
    if tf_like:
        def lpround(ck):
            with_rel(ck)
            e = ck("repair+lps", lp_round_screen, m, sysm, sc, rel.u)
            return e["u"], dict(n_lp=e["n_lp"], th=e["th"])
        meth["lpround"] = lpround

    def p_knn():
        return override_probs(knn.predict(di)[0], g_out)
    for e in (1, 5, 10):
        if f"knn_{e}" in tuned and (tf_like or e == 1):
            meth[f"ltf_knn_{e}"] = rule((lambda e: lambda: fix_dict(p_knn(), *tuned[f"knn_{e}"]))(e), (), G01, uses_rel=False)
    if "knn_1" in tuned:
        meth["ltf_knn_1+lpg"] = rule(lambda: fix_dict(p_knn(), *tuned["knn_1"]), ("rows", "lp"), G01, uses_rel=False)
    if bce is not None:
        meth["ltf_bce_1"] = rule(lambda: fix_dict(override_probs(bce.predict(di)[0].astype(np.float64), g_out),
                                                  *tuned["bce_1"]), (), G01)

    def ens_fix(nets, th):
        return lambda: fix_dict(override_probs(np.mean([n.predict(di)[0].astype(np.float64) for n in nets], 0), g_out), *th)

    def pipe_bc_of(hg_fix):
        def pipe_bc(ck):
            u, info = rule(hg_fix, ALL_GUARDS, G05)(ck)
            obj = info["fb_obj"] if info["fallback"] else info["obj"]
            g = float(bound_gap(obj, rel.obj)) if np.isfinite(obj) else float("inf")
            info.update(bound_gap=g, d_post=D_POST, fired=g > D_POST)
            if g > D_POST:
                r = ck("check_full", solve_milp, m, sc, {}, tl, G05)
                info.update(check_obj=r["obj"], check_s=r["time"], check_status=r["status"])
                if r["feasible"] and (u is None or r["obj"] < obj):
                    u = r["u"]
                    info["kept"] = "full"
            return u, info
        return pipe_bc
    if gnnt is not None:
        hg_fix = ens_fix(gnnt, tuned["hg"])
        meth["pipe@0.1%"] = rule(hg_fix, ALL_GUARDS, G01)
        meth["pipe@0.5%"] = rule(hg_fix, ALL_GUARDS, G05)
        meth["pipe+bc"] = pipe_bc_of(hg_fix)
    for name, (nets, th) in extra_ens.items():
        meth[f"pipe@0.5%:{name}"] = rule(ens_fix(nets, th), ALL_GUARDS, G05)
        meth[f"pipe+bc:{name}"] = pipe_bc_of(ens_fix(nets, th))
    if a.orig and tf_like and a.bench == "uc12":
        orig = lambda f: load_thr(os.path.join("results/uc12", f))
        if knn is not None:
            for e in (1, 5):
                meth[f"ltf_knn_{e}:orig"] = rule((lambda th: lambda: fix_dict(p_knn(), *th))(orig(f"ltfx_tune_knn_{e}.json")),
                                                 (), G01, uses_rel=False)
        if bce is not None:
            th_b = orig("ltfx_tune_bce_1.json")
            meth["ltf_bce_1:orig"] = rule(lambda: fix_dict(bce.predict(di)[0].astype(np.float64), *th_b), (), G01)
        if gnnt is not None:
            meth["pipe@0.5%:orig"] = rule(ens_fix(gnnt, orig("m1x_tune_hg_pol_n2000_gnnt_ens5.json")), ALL_GUARDS, G05)

    # early one-shot models and kNN schedules (end to end: no MILP)
    rep = lambda u: repair_min_updown(u, sc.u0, sysm.min_up, sysm.min_dn)

    def screen(ck, cands):
        best, n_lp = None, 0
        seen = {}
        for u in cands:
            kb = u.tobytes()
            if kb not in seen:
                seen[kb] = ck("lps", m.solve_dispatch, sc, u).obj
                n_lp += 1
            if best is None or seen[kb] < best[0]:
                best = (seen[kb], u)
        return best[1], dict(n_lp=n_lp)

    def early_method(name, dec):
        m1, uses_rel = early[name]

        def run(ck):
            if uses_rel:
                with_rel(ck)
            p = ck("inference", lambda: m1.predict(di)[0])
            if dec == "screen":
                from uc_model1 import candidates_from_probs
                rng = np.random.default_rng(zlib.crc32(f"{a.set}/{a.k}".encode()))
                cands = ck("repair", lambda: [rep(u) for u in candidates_from_probs(p[None], 8, rng)[0]])
                return screen(ck, cands)
            u = (p > 0.5).astype(np.int8)
            if dec == "adeq":
                u = ck("repair", adequacy_repair, u, sc.load, sc.avail, sc.sr, sysm)
            elif dec == "block":
                u = ck("repair", adequacy_repair_blocks, u, sc.u0, sc.load, sc.avail, sc.sr, sysm)
            u = ck("repair", rep, u)
            return screen(ck, [u])
        return run
    for name in early:
        for dec in ("top1", "adeq", "block", "screen"):
            meth[f"e2e:{name}:{dec}"] = early_method(name, dec)
    if tf_like and a.bench == "uc12":
        from uc_model1 import knn_candidates
        one = {k_: v[None] for k_, v in dict(load=sc.load, avail=sc.avail, u0=sc.u0).items()}
        for kk in (5, 20):
            meth[f"knn{kk}"] = (lambda kk: lambda ck: screen(ck, ck("knn", lambda: [rep(u) for u in
                                                                                   knn_candidates(tr, one, kk)[0]])))(kk)

    if a.only:
        keep = set(a.only.split(","))
        meth = {k_: v for k_, v in meth.items() if k_ in keep}
    order = sorted(meth)
    np.random.default_rng([zlib.crc32(f"{a.bench}/{a.set}/{a.k}".encode()), a.seed]).shuffle(order)

    # ---------------------------------------------------------------- run, price, record
    rec = dict(key, **host_info(), order=order, notes=notes, c_rel=float(rel.obj), stored_c_rel=stored_c_rel,
               t_rel=t_rel, g_out=list(g_out), l_out=list(getattr(sc, "l_out", ()) or ()), d_post=D_POST,
               loadavg0=os.getloadavg()[0], runs={})
    t_start = time.time()
    for name in order:
        ck = Clock()
        u, info = meth[name](ck)
        pr = price(m, sc, u)
        rec["runs"][name] = dict(pr, time=ck.wall, cpu=ck.cpu, parts=ck.parts, **info)
        print(f"  {name:24s} {ck.wall:7.2f}s  cost {pr['cost']:.1f}  "
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
