"""Screened network and the certified pipeline (docs/methods/screen.md): every method of one instance back to back in
one single-threaded process, with the step-2 references re-solved in the same process (paired speed-ups, as
scripts/unified_fix_eval.py).

    OTSL_THREADS=1 python3 scripts/screen_eval.py --bench uc12 --set tf --k 7 --out results/screen/test/uc12_x.jsonl
    OTSL_THREADS=1 python3 scripts/screen_eval.py --bench uc12 --set val --k 7 --out results/screen/val/val_x.jsonl

Instances: uc12 val (the 360 validation instances: val + val_extra + val_extra2, for the choices), uc12 tf (test_fresh)
and the six shifted 12-hour sets (outages applied as in step 2), uc24 (40 test instances), hard (the congested
benchmark's 60 test instances: line ratings x 0.6). The LP relaxation of the full network is solved once (timed) and
charged to every method that reads it.

Methods (names as in the records):
  full@g          the full MILP on the B-theta network at mip_rel_gap g (step 2's; full@0.1% is the speed-up reference
                  and gives a dual bound)
  scr:full@g      the same MILP on the screened network (otsl.screen.ScreenedUC: copper plate + the line limits the LP
                  relaxation loads to >= 99 %, violated ones added until none is): exact, no learning
  pipe@0.5%, pipe+bc   the recommended pipeline of step 2 and its bound check (uc12; ensemble, thresholds and guards
                  exactly as scripts/unified_fix_eval.py)
  scr:pipe        the recommended pipeline's fixings, reduced MILP on the screened network at 0.5 % (uc12)
  cert:<model>:<delta>   the certified pipeline (otsl.screen.certified_pipeline): <model>'s on-probabilities -> every
                  decision with min(p, 1 - p) <= delta fixed to round(p), rows min up/down cannot complete released ->
                  reduced MILP on the screened network at 0.5 % -> certificate: the LP relaxation's bound within 0.5 %,
                  else the screened full MILP at 0.5 % warm-started from the reduced schedule. <model>: ens5 (the
                  pipeline's five temporal GNNs, averaged), s0 (its first network alone), lp (the LP relaxation's values:
                  no learning). The reduced schedule is also priced (not timed): the same run gives the pipeline without
                  the certificate and with an LP certificate at a looser tolerance (scripts/screen_report.py).
Every delivered schedule is priced afterwards by the exact dispatch LP on the full network (not timed).
Output: one JSON line per instance (resume skips instances already in --out).
"""
import argparse
import json
import os
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
from otsl.cguard import bound_gap  # noqa: E402
from otsl.hybrid import ALL_GUARDS, apply_guards  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.ood import OODModel, OODScenario, avail_view, force_out, override_probs, solve_milp  # noqa: E402
from otsl.screen import ScreenedUC, certified_pipeline, monitored_from_lp  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402

SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
VAL = ("val", "val_extra", "val_extra2")
EPS = 5e-3
D_POST = json.load(open(os.path.join(os.path.dirname(HERE), "results/uc12/cguard_select.json")))["d_post"]
KEYS = ("load", "avail", "u0", "sr", "start", "u_rel", "lmp_rel", "flow_rel")


def host_info():
    model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import highspy
    h = highspy.Highs()
    return dict(host=socket.gethostname(), cpu_model=model,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                highs=f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}", torch=torch.__version__,
                slurm_job=os.environ.get("SLURM_JOB_ID"))


def instance(bench, s, k):
    """(system, T, time limit, scenario, stored LP relaxation cost)"""
    if bench == "hard":
        from otsl.hard import load_split, sc_of
        sysm = load_rts_gmlc(line_scale=0.6)
        d = load_split("test")
        sc = sc_of(d, k)
        return sysm, 12, 600.0, OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr), float(d["c_rel"][k])
    sysm = load_rts_gmlc()
    if bench == "uc24":
        d = load_b3("data/generated/uc24/test.npz")
        T, tl = 24, 300.0
    elif s == "val":
        parts = [load_b3(f"data/generated/uc12/{n}.npz") for n in VAL]
        d = {key: np.concatenate([p[key] for p in parts]) for key in KEYS + ("day", "c_rel")}
        T, tl = 12, 60.0
    elif s == "tf":
        d = load_b3("data/generated/uc12/test_fresh.npz")
        T, tl = 12, 60.0
    else:
        z = np.load(os.path.join("data/generated/uc12_ood", s, f"{k:03d}.npz"))
        sc = OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"], day=int(z["day"]), start=int(z["start"]),
                         g_out=tuple(int(g) for g in z["g_out"]), l_out=tuple(int(x) for x in z["l_out"]))
        return sysm, 12, 60.0, sc, float(z["c_rel"]) if "c_rel" in z.files else None
    sc = OODScenario(load=d["load"][k], avail=d["avail"][k], u0=d["u0"][k], sr=d["sr"][k], day=int(d["day"][k]),
                     start=int(d["start"][k]) if "start" in d else 0)
    return sysm, T, tl, sc, float(d["c_rel"][k])


class Clock:
    def __init__(self):
        self.wall, self.parts = 0.0, {}

    def __call__(self, part, f, *a, **k):
        t0 = time.perf_counter()
        r = f(*a, **k)
        dt = time.perf_counter() - t0
        self.wall += dt
        self.parts[part] = self.parts.get(part, 0.0) + dt
        return r

    def add(self, part, dt):
        self.wall += dt
        self.parts[part] = self.parts.get(part, 0.0) + dt


def price(m, sc, u):
    if u is None:
        return dict(cost=float("inf"), shed=None, short=None)
    lp = m.solve_dispatch(sc, u)
    return dict(cost=float(lp.obj), shed=float(lp.shed), short=float(lp.short))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="uc12", choices=("uc12", "uc24", "hard"))
    ap.add_argument("--set", default="tf", choices=("tf", "val") + SHIFTS)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="", help="comma list (default: every method of the bench and set)")
    ap.add_argument("--cert", default="ens5:0.02,ens5:0.05,ens5:0.1,s0:0.02,s0:0.05,s0:0.1,lp:0.05",
                    help="certified pipelines <model>:<delta> (validation: the grid; test: the validation choice)")
    ap.add_argument("--hg", default="results/unified/retime/tune/m1x_tune_hgP.json", help="step 2's pipeline thresholds")
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
    scr = ScreenedUC(sysm, T=T)
    G = sysm.G
    g_out = tuple(getattr(sc, "g_out", ()) or ())
    sv = avail_view(sysm, g_out)

    # models (uc12 only): the pipeline's five temporal GNNs
    nets = []
    certs = [c.split(":") for c in a.cert.split(",") if c]
    if a.bench == "uc12":
        from uc_m1x_train import featurizer, load_net
        feat = featurizer(sysm)
        nets = [load_net(f"results/uc12/m1x_pol_n2000_gnnt_s{s}.pt", "gnnt", sysm, feat) for s in range(5)]
        j = json.load(open(a.hg))
        hg = (np.array(j["lo"]), np.array(j["hi"]))
    else:
        certs = [c for c in certs if c[0] == "lp"]

    # LP relaxation (timed once, charged where read); warm-up first
    m.solve_dispatch(sc, None, relax=True)
    t0 = time.perf_counter()
    rel = m.solve_dispatch(sc, None, relax=True)
    t_rel = time.perf_counter() - t0
    di = dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], start=np.array([getattr(sc, "start", 0)]),
              u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])
    for net in nets:
        net.predict(di)

    def with_rel(ck):
        ck.add("lp_relaxation", t_rel)
        return ck

    def probs(model):
        if model == "lp":
            return np.clip(rel.u, 0.0, 1.0)
        use = nets if model == "ens5" else [nets[int(model[1:])]]
        return override_probs(np.mean([n.predict(di)[0].astype(np.float64) for n in use], 0), g_out)

    meth = {}
    for g in (1e-3, EPS):
        def full(ck, g=g):
            r = ck("milp", solve_milp, m, sc, {}, tl, g)
            return r.get("u"), dict(obj=r["obj"], bound=r["bound"], status=r["status"], mip_gap=r["mip_gap"])

        def sfull(ck, g=g):
            with_rel(ck)
            z = ck("milp", scr.solve, sc, monitored_from_lp(rel.flow, sysm.fmax), None, g, tl)
            return z["u"], dict(obj=z["obj"], bound=z["bound"], status=z["status"], rounds=z["rounds"], n_mon=z["n_mon"],
                                lines_ok=z["ok"])
        meth[f"full@{g * 100:g}%"] = full
        meth[f"scr:full@{g * 100:g}%"] = sfull

    if a.bench == "uc12":
        def pipe_fix(ck):
            fix = ck("rule", lambda: fix_dict(probs("ens5"), *hg))
            fix, info = ck("guards", apply_guards, m, sv, sc, dict(fix), ALL_GUARDS)
            return force_out(fix, g_out, T), {k: v for k, v in info.items() if k != "guard_s"}

        def pipe(ck, bc=False):
            with_rel(ck)
            fix, info = pipe_fix(ck)
            r = ck("milp", solve_milp, m, sc, fix, tl, EPS)
            u, obj = r.get("u"), r["obj"]
            if not r["feasible"]:
                r = ck("fallback", solve_milp, m, sc, {}, tl, EPS)
                u, obj = r.get("u"), r["obj"]
                info["fallback"] = True
            info.update(obj=obj, n_fixed=len(fix))
            if bc:
                gb = float(bound_gap(obj, rel.obj)) if np.isfinite(obj) else float("inf")
                info.update(bound_gap=gb, fired=gb > D_POST)
                if gb > D_POST:
                    rf = ck("check_full", solve_milp, m, sc, {}, tl, EPS)
                    info.update(check_obj=rf["obj"])
                    if rf["feasible"] and (u is None or rf["obj"] < obj):
                        u = rf["u"]
            return u, info
        meth["pipe@0.5%"] = pipe
        meth["pipe+bc"] = lambda ck: pipe(ck, bc=True)

        def spipe(ck):
            with_rel(ck)
            fix, info = pipe_fix(ck)
            z = ck("milp", scr.solve, sc, monitored_from_lp(rel.flow, sysm.fmax), fix, EPS, tl)
            info.update(obj=z["obj"], n_fixed=len(fix), rounds=z["rounds"], lines_ok=z["ok"])
            return z["u"], info
        meth["scr:pipe"] = spipe

    for model, dl in certs:
        def cert(ck, model=model, dl=float(dl)):
            with_rel(ck)
            p = ck("inference", probs, model)
            u, info = ck("certified", certified_pipeline, scr, sc, p, rel, dl, EPS, None, tl)
            return u, info
        meth[f"cert:{model}:{dl}"] = cert

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
        if "u_red" in info:                          # the reduced schedule, priced (not timed): the uncertified pipeline
            pr_red = price(m, sc, info.pop("u_red"))
            info.update(red_cost=pr_red["cost"], red_shed=pr_red["shed"], red_short=pr_red["short"])
        rec["runs"][name] = dict(pr, time=ck.wall, parts=ck.parts, **info)
        print(f"  {name:22s} {ck.wall:7.2f}s  cost {pr['cost']:.1f}  "
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
