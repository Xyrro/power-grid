"""Unified re-scoring of every saved test result: 12 h (test_fresh and the six shifted sets), 24 h, congested lines
and PGLib-UC California (step 1 of docs/unified_metrics.md). Standard library only.

Nothing is re-solved: every number comes from the per-instance records the studies saved. The protocol:

  instances  test_fresh 0-119 ("tf", i) and the six shifted sets of the robustness study ((shift, k), 30 each); the
             robustness study's in-distribution set ("id", k) is test_fresh 0-39 (checked here)
  bound      DB(x) = the largest proven lower bound recorded for instance x by any full-MILP solve in any file
             (explicit 'bound', or obj * (1 - mip_gap) where only the relative gap is stored) or the LP relaxation
  cost       C = the penalty-inclusive objective of the schedule the method delivers. A reduced MILP without a solution
             falls back to the full MILP: its recorded fallback solve, or else the same run's full MILP at the method's
             MILP gap, with that solve's time added
  gap        (C - DB) / C in %, every instance counted (none dropped)
  served     no shed load, over-generation or reserve shortfall in any hour
  speed-up   T(full MILP at 0.1 %, same run, same core) / T(method) per instance: geometric mean (headline), median,
             arithmetic mean and the ratio of total times. 'timing' says what a record's time includes; only
             'complete' records (LP relaxation, inference, guards, every LP / MILP, fallbacks) are comparable

    python3 scripts/unified_rescore.py            # results/unified/rescored.{json,md}
"""
import glob
import json
import math
import os
import random
import struct
import zipfile
import zlib

R12, RREC = os.path.join("results", "uc12"), os.path.join("results", "recursive")
OUT = os.path.join("results", "unified")
TOL = 1e-6           # served: slack in p.u.
BOOT = 2000
SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")


# ============================================================================ helpers
def jl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def jls(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        out += jl(p)
    return out


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def npz_arrays(path):
    """{name: (shape, flat list of float64)} from an .npz of little-endian float64 arrays (no numpy needed)"""
    out = {}
    z = zipfile.ZipFile(path)
    for n in z.namelist():
        b = z.read(n)
        assert b[:6] == b"\x93NUMPY", n
        major = b[6]
        hl = struct.unpack("<H", b[8:10])[0] if major == 1 else struct.unpack("<I", b[8:12])[0]
        off = 10 if major == 1 else 12
        hdr = eval(b[off:off + hl].decode("latin1"))          # numpy's own header: a dict literal
        assert hdr["descr"] == "<f8" and not hdr["fortran_order"], (n, hdr)
        data = b[off + hl:]
        cnt = len(data) // 8
        out[n[:-4]] = (hdr["shape"], list(struct.unpack(f"<{cnt}d", data)))
    return out


def slack(r, prefix=""):
    return sum((r.get(prefix + k) or 0.0) for k in ("shed", "spill", "short"))


# ============================================================================ the registry
class Registry:
    """methods: name -> dict(meta, data: inst -> (cost, served, t or None, tref or None, fellback))"""

    def __init__(self):
        self.methods, self.bounds, self.crel, self.notes = {}, {}, {}, []

    def bound(self, inst, val, src):
        if finite(val):
            self.bounds.setdefault(inst, []).append((float(val), src))

    def lp(self, inst, val, src):
        if finite(val):
            self.crel.setdefault(inst, []).append((float(val), src))

    def put(self, name, inst, cost, served, t, tref, meta, fellback=False):
        m = self.methods.setdefault(name, dict(meta=meta, data={}))
        if inst in m["data"]:
            raise ValueError(f"duplicate record for {name} at {inst}")
        m["data"][inst] = (float(cost), bool(served), t if finite(t) else None, tref if finite(tref) else None, fellback)


REG = Registry()


def meta(src, family, timing, run, **kw):
    return dict(source=src, family=family, timing=timing, run=run, **kw)


# ---------------------------------------------------------------------------- recursive arms (records per instance)
def load_arm_a(pattern, prefix, run, src):
    for r in jls(pattern):
        x = ("tf", r["i"])
        REG.bound(x, r["milp01"]["bound"], src)
        REG.bound(x, r["milp05"]["bound"], src)
        REG.lp(x, r["relax"]["cost"], src)
        tref = r["milp01"]["time"]
        for k, v in r.items():
            if k == "relax" or not (isinstance(v, dict) and finite(v.get("cost")) and "time" in v):
                continue
            REG.put(f"{prefix}{k}", x, v["cost"], (v.get("shed") or 0) + (v.get("short") or 0) < TOL, v["time"], tref,
                    meta(src, "arm A" if not k.startswith(("milp", "lpround", "m1x", "rl_")) else "other", "complete", run))


def halt_step(rl, thr=0.5):
    for i, r in enumerate(rl):
        if r.get("halt", 0.0) >= thr:
            return i
    return len(rl) - 1


def best_of(c):
    j = min(range(len(c)), key=lambda i: c[i][0])
    return c[j][0], c[j][1] < TOL and c[j][2] < TOL, c[-1][3]


def load_arm_b_gpu(tag, label):
    src = f"recursive/gpu_round/rloop_eval_uc12_{tag}"
    run = f"gpu-B-{tag}"
    for r in jls(os.path.join(RREC, "gpu_round", f"rloop_eval_uc12_{tag}_s*.jsonl")):
        x = ("tf", r["k"])
        f1, f5 = r["full"]["0.001"], r["full"]["0.005"]
        REG.bound(x, f1["bound"], src)
        REG.bound(x, f5["bound"], src)
        REG.lp(x, r["c_rel"], src)
        tref = f1["time"]
        mt = lambda fam: meta(src, fam, "complete", run)
        REG.put(f"{label} | full MILP 0.1 %", x, f1["obj"], slack(f1) < TOL, f1["time"], tref, mt("solver"))
        REG.put(f"{label} | full MILP 0.5 %", x, f5["obj"], slack(f5) < TOL, f5["time"], tref, mt("solver"))
        c, s, t = best_of(r["lpround"]["c"])
        REG.put(f"{label} | LP rounding + repair, 5 LPs", x, c, s, r["lpround"]["t_pre"] + t, tref, mt("no learning"))
        ls = r["lpls"]["recs"][-1]
        REG.put(f"{label} | LP-checked local search", x, ls["best"], ls["best_shed"] < TOL and ls["best_short"] < TOL,
                r["lpls"]["t_pre"] + ls["time"], tref, mt("no learning"))
        rl, ks = r["rloop"], r["k_sel"]
        y = rl[ks - 1]
        srv = y["best_shed"] < TOL and y["best_short"] < TOL
        REG.put(f"{label} | @{ks} steps", x, y["best"], srv, y["time"], tref, mt("arm B"))
        h = rl[halt_step(rl)]
        REG.put(f"{label} | halting head", x, h["best"], h["best_shed"] < TOL and h["best_short"] < TOL, h["time"], tref,
                mt("arm B"))
        cost, t = y["best"], y["time"]
        s = r.get("safety")
        if s:
            t += s["time"] + s.get("t_fix", 0.0)
            if s["feasible"] and slack(s) < TOL and (not srv or s["obj"] < cost):
                cost, srv = s["obj"], True
        REG.put(f"{label} | @{ks} + safety net", x, cost, srv, t, tref, mt("arm B"))


def load_arm_b_cpu(bench="uc12", key="tf", label="B·cpu"):
    src, run = f"recursive/rloop_eval_{bench}", f"cpu-B-{bench}"
    for r in jls(os.path.join(RREC, f"rloop_eval_{bench}_s*.jsonl")):
        x = (key, r["k"])
        f = r["full"]
        for g in f.values():
            REG.bound(x, g.get("bound"), src)
        REG.lp(x, r["c_rel"], src)
        tref = f["0.001"]["time"]
        mt = lambda fam: meta(src, fam, "complete", run)
        c1, s1, t1 = best_of(r["lpround"]["c"])
        REG.put(f"{label} | LP rounding + repair, 5 LPs", x, c1, s1, r["lpround"]["t_pre"] + t1, tref, mt("no learning"))
        for rkey, name, fam in (("lpls", "LP-checked local search, +16 LPs", "no learning"),
                                ("rloop_guided", "local search guided by rloop, +16 LPs", "arm B")):
            L = r[rkey]
            y = L["recs"][min(16, len(L["recs"])) - 1] if L["recs"] else None
            if y is None or y["best"] >= c1:
                base = (c1, s1, L["t_pre"] + (y["time"] if y else 0.0))
            else:
                base = (y["best"], y["best_shed"] < TOL and y["best_short"] < TOL, L["t_pre"] + y["time"])
            REG.put(f"{label} | {name}", x, *base, tref, mt(fam))
            if rkey == "rloop_guided":
                guided = base
        y = r["rloop"][r["k_sel"] - 1]
        REG.put(f"{label} | whole-schedule recursion, {r['k_sel']} steps", x, y["best"],
                y["best_shed"] < TOL and y["best_short"] < TOL, y["time"], tref, mt("arm B"))
        assert r.get("main_pipe") == "rloop_guided"
        flag = (not guided[1]) or r["cert_gap"] > 1.0
        c, s, t = guided
        if flag and r.get("safety") and r["safety"]["feasible"]:
            sf = r["safety"]
            s = (slack(sf) < TOL) if sf["obj"] <= c else s
            c = min(c, sf["obj"])
            t = t + sf["t_fix"] + sf["time"]
        REG.put(f"{label} | guided search + safety net (theta 1 %)", x, c, s, t, tref, mt("arm B"))


# ---------------------------------------------------------------------------- fixing studies (one record per rule)
def load_rule_file(path, prefix, run, idx="i", fullname="full MILP", timing="complete", inst_of=None, family="fixing",
                   tref_from=None):
    """hybrid / ltfx / m1x_eval / ood style records: obj, feasible, time + pre_s + relax_s + guard_s, fb_* fallback"""
    recs = jl(path)
    src = os.path.relpath(path, "results")
    inst_of = inst_of or (lambda r: ("tf", r[idx]))
    full = {inst_of(r): r for r in recs if r["rule"] == fullname}
    for x, F in full.items():
        b = F.get("bound")
        REG.bound(x, b if finite(b) else F["obj"] * (1 - (F.get("mip_gap") or 0.0)), src)
    for r in recs:
        x = inst_of(r)
        F = full.get(x) or (tref_from or {}).get(x)
        tref = F["time"] if F else None
        t = r["time"] + r.get("pre_s", 0.0) + r.get("relax_s", 0.0) + r.get("guard_s", 0.0)
        fam = "solver" if r["rule"].startswith("full") else family
        if r.get("feasible", True):
            REG.put(f"{prefix}{r['rule']}", x, r["obj"], slack(r) < TOL, t, tref, meta(src, fam, timing, run))
        elif "fb_obj" in r:
            REG.put(f"{prefix}{r['rule']}", x, r["fb_obj"], slack(r, "fb_") < TOL, t + r["fb_time"], tref,
                    meta(src, fam, timing, run), fellback=True)
        elif F:
            REG.put(f"{prefix}{r['rule']}", x, F["obj"], slack(F) < TOL, t + F["time"], tref, meta(src, fam, timing, run),
                    fellback=True)
        else:
            REG.notes.append(f"{src}: {r['rule']} at {x} infeasible with no fallback record: dropped")


def load_m1x_pipeline():
    """Y6: test_fresh + shifted sets; fallback = the same run's full MILP at the rule's gap"""
    recs = jls(os.path.join(R12, "m1x_pipeline_eval_s*.jsonl"))
    veto = jls(os.path.join(R12, "m1x_pipeline_veto_s*.jsonl"))
    inst = lambda r: ("tf", r["k"]) if r["set"] == "tf" else (r["set"], r["k"])
    full = {}
    for r in recs + veto:
        if r["rule"].startswith("full MILP"):
            REG.bound(inst(r), r["bound"] if finite(r["bound"]) else r["obj"] * (1 - (r["mip_gap"] or 0)), "uc12/m1x_pipeline")
            full[(inst(r), r["rule"])] = r
    for r in recs + veto:
        x = inst(r)
        if r["rule"].startswith("full MILP (re-time"):
            continue
        F = full[(x, "full MILP")]
        t = r["time"] + r.get("pre_s", 0.0) + r.get("relax_s", 0.0) + r.get("guard_s", 0.0)
        fam = "solver" if r["rule"].startswith("full") else "fixing"
        mt = meta("uc12/m1x_pipeline_eval" if r in recs else "uc12/m1x_pipeline_veto", fam, "complete", "Y6")
        if r["feasible"]:
            REG.put(f"Y6 | {r['rule']}", x, r["obj"], slack(r) < TOL, t, F["time"], mt)
        else:
            fb = full[(x, "full MILP gap 0.5%")] if abs(r["mip_rel_gap"] - 0.005) < 1e-9 else F
            REG.put(f"Y6 | {r['rule']}", x, fb["obj"], slack(fb) < TOL, t + fb["time"], F["time"], mt, fellback=True)


def load_cguard():
    """Z1: the pipeline re-run with the post-solve bound check (C3: g > d_post -> also full MILP 0.5 %, keep better)"""
    d_post = json.load(open(os.path.join(R12, "cguard_select.json")))["d_post"]
    for r in jl(os.path.join(R12, "cguard_test_full.jsonl")):
        x = ("tf", r["k"]) if r["set"] == "tf" else (r["set"], r["k"])
        F, F5, P = r["full"], r["full5"], r["pipe"]
        for G in (F, F5):
            REG.bound(x, G["bound"] if finite(G["bound"]) else G["obj"] * (1 - (G["mip_gap"] or 0)), "uc12/cguard_test_full")
        tref = F["time"]
        mt = lambda fam: meta("uc12/cguard_test_full", fam, "complete", "Z1")
        REG.put("Z1 | full MILP 0.1 %", x, F["obj"], slack(F) < TOL, F["time"], tref, mt("solver"))
        REG.put("Z1 | full MILP 0.5 %", x, F5["obj"], slack(F5) < TOL, F5["time"], tref, mt("solver"))
        tp = r["relax_s"] + r["pre_s"] + r["guard_s"] + P["time"]
        if not P["feasible"]:
            REG.put("Z1 | pipeline", x, F5["obj"], slack(F5) < TOL, tp + F5["time"], tref, mt("fixing"), fellback=True)
            REG.put("Z1 | pipeline + bound check", x, F5["obj"], slack(F5) < TOL, tp + F5["time"], tref, mt("fixing"), True)
            continue
        REG.put("Z1 | pipeline", x, P["obj"], slack(P) < TOL, tp, tref, mt("fixing"))
        if r["g"] > d_post:
            src = F5 if F5["obj"] < P["obj"] else P
            REG.put("Z1 | pipeline + bound check", x, src["obj"], slack(src) < TOL, tp + F5["time"], tref, mt("fixing"))
        else:
            REG.put("Z1 | pipeline + bound check", x, P["obj"], slack(P) < TOL, tp, tref, mt("fixing"))


def load_base(bench="uc12", key="tf", prefix="Y1", run="Y1"):
    """Y1: no-learning baselines and looser full MILPs (no fallback recorded: same run's full MILP)"""
    src = f"{bench}/base_eval_test"
    recs = jl(os.path.join("results", bench, "base_eval_test.jsonl"))
    for r in recs:
        x = (key, r["i"])
        REG.lp(x, r["c_rel"], src)
        F = r["runs"]["full"]
        for nm, v in r["runs"].items():
            if nm.startswith("full"):
                REG.bound(x, v.get("bound"), src)
        for nm, v in r["runs"].items():
            t = v.get("t_method", v.get("time"))
            fam = "solver" if nm.startswith("full") else ("no learning" if nm.startswith(("LP", "LtF on LP")) else "fixing")
            mt = meta(src, fam, "complete", run)
            if v["feasible"]:
                REG.put(f"{prefix} | {nm}", x, v["obj"], slack(v) < TOL, t, F["time"], mt)
            else:
                REG.put(f"{prefix} | {nm}", x, F["obj"], slack(F) < TOL, t + F["time"], F["time"], mt, fellback=True)


def load_ltf_eval():
    """X3/X6: ltf_eval_fresh(_b): obj already includes the fallback; time = MILP solves only (no inference / LP
    relaxation): timing 'MILP only'"""
    for fn in ("ltf_eval_fresh.jsonl", "ltf_eval_fresh_b.jsonl"):
        for r in jl(os.path.join(R12, fn)):
            x = ("tf", r["i"])
            if r["rule"] == "full" and fn.endswith("_b.jsonl"):
                continue
            name = f"X3 | {r['rule']}"
            fam = "solver" if r["rule"] == "full" else "fixing"
            REG.put(name, x, r["obj"], slack(r) < TOL, r["time"] + r.get("lp_guard_s", 0.0), None,
                    meta(f"uc12/{fn}", fam, "MILP only", "X3"), fellback=bool(r.get("fallback")))
    # speed-up denominators: the full MILP of the same file
    m = REG.methods
    for name, M in m.items():
        if name.startswith("X3 | ") and name != "X3 | full":
            for x, v in list(M["data"].items()):
                M["data"][x] = (v[0], v[1], v[2], m["X3 | full"]["data"][x][2], v[4])


def load_combo_fix():
    """X2: combined fixing pipelines, 3 seeds; time = reduced MILP + LP guard (no inference): 'MILP + guards'"""
    full = {}
    recs = jl(os.path.join(R12, "combo_fix_test_fresh.jsonl"))
    for r in recs:
        if r["config"] == "full":
            full[r["i"]] = r
    for r in recs:
        x = ("tf", r["i"])
        name = "X2 | full" if r["config"] == "full" else f"X2 | {r['config']} @{r['ratio']:g} s{r['seed']}"
        REG.put(name, x, r["obj"], slack(r) < TOL, r["time"], full[r["i"]]["time"],
                meta("uc12/combo_fix_test_fresh", "solver" if r["config"] == "full" else "fixing", "MILP + guards", "X2",
                     seed=r["seed"]), fellback=bool(r.get("fallback")))


def load_combo_e2e():
    """X2 end-to-end (no MILP): stored arrays [cost, shed, short, *] per instance; no timing"""
    arr = npz_arrays(os.path.join(R12, "combo_e2e_test_fresh_arrays.npz"))
    n = arr[next(iter(arr))][0][1]
    for sd in (0, 1, 2):
        ths = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        for i in range(n):
            cands = []
            for th in ths:
                shape, flat = arr[f"lag_s{sd}__block__{th}"]
                cands.append((flat[i], flat[n + i], flat[2 * n + i]))
            c, sh, so = min(cands, key=lambda z: z[0])
            REG.put(f"X2 | e2e combined + screening (7 thresholds) s{sd}", ("tf", i), c, sh + so < TOL, None, None,
                    meta("uc12/combo_e2e_test_fresh_arrays", "end to end", "none", "X2", seed=sd))


def load_ood():
    """Y4 robustness study: 'id' = test_fresh 0-39; shifted sets as in Y6"""
    inst = lambda r: ("tf", r["k"]) if r["shift"] == "id" else (r["shift"], r["k"])
    path = os.path.join(R12, "ood_eval.jsonl")
    load_rule_file(path, "Y4 | ", "Y4", inst_of=inst)
    full = {inst(r): r for r in jl(path) if r["rule"] == "full MILP"}
    load_rule_file(os.path.join(R12, "ood_fu_eval.jsonl"), "Y4fu | ", "Y4fu", inst_of=inst, fullname="__none__",
                   tref_from=full)


def load_runs_file(pattern, key, prefix, run, src):
    """m24-style records: one line per instance, runs = {name: record with t_method, feasible, obj, bound}; the
    reference is runs['full'] (0.1 %); a run without a solution falls back to the full MILP at its own gap"""
    for r in jls(pattern):
        x = (key, r["i"])
        REG.lp(x, r.get("c_rel"), src)
        runs = r["runs"]
        full = {v.get("mip_rel_gap", 0.001): v for nm, v in runs.items() if nm.startswith("full")}
        for nm, v in runs.items():
            if nm.startswith("full"):
                REG.bound(x, v.get("bound"), src)
        F = runs["full"]
        tref = F.get("t_method", F["time"])
        for nm, v in runs.items():
            t = v.get("t_method", v["time"])
            fam = "solver" if nm.startswith("full") else "fixing"
            mt = meta(src, fam, "complete", run)
            if v["feasible"]:
                REG.put(f"{prefix} | {nm}", x, v["obj"], slack(v) < TOL, t, tref, mt)
            else:
                fb = full.get(v.get("mip_rel_gap", 0.001), F)
                REG.put(f"{prefix} | {nm}", x, fb["obj"], slack(fb) < TOL, t + fb.get("t_method", fb["time"]), tref, mt,
                        fellback=True)


def load_uc24ltf():
    """X8: Learning to Fix on 24 h; time = pre_s + guards + reduced MILP + LP relaxation. Only a few instances re-ran
    the full MILP in the same process (full_b2b), so there is no paired reference: quality only. A fixing without a
    solution falls back to the full MILP of the Z2 run (same settings)"""
    src = "uc24/uc24ltf_eval_test"
    zf = REG.methods["Z2 | full"]["data"]
    for r in jl(os.path.join("results", "uc24", "uc24ltf_eval_test.jsonl")):
        x = ("u24", r["i"])
        F = r.get("full_b2b")
        if F:
            REG.bound(x, F["bound"], src)
        for nm, v in r.items():
            if not isinstance(v, dict) or nm == "full_b2b" or "t_milp" not in v:
                continue
            t = v.get("pre_s", 0.0) + v.get("t_guard", 0.0) + v["t_milp"] + r["t_rel"]
            mt = meta(src, "fixing", "no paired reference", "X8")
            if v["feasible"]:
                REG.put(f"X8 | {nm}", x, v["obj"], slack(v) < TOL, t, None, mt)
            else:
                c, srv = zf[x][0], zf[x][1]
                REG.put(f"X8 | {nm}", x, c, srv, None, None, mt, fellback=True)


def load_stretch(bench, key):
    """arm A (CPU round) zero-shot on other benchmarks: costs per step count; only some records carry a time"""
    src = f"recursive/trm_stretch_{bench}"
    for r in jl(os.path.join(RREC, f"trm_stretch_{bench}.jsonl")):
        x = (key, r["i"])
        REG.bound(x, r["bound"], src)
        for nm, v in r.items():
            if isinstance(v, dict) and finite(v.get("cost")):
                tm = v.get("time")
                REG.put(f"A·cpu stretch {bench} | {nm}", x, v["cost"], (v.get("shed") or 0) + (v.get("short") or 0) < TOL, tm,
                        r["milp_time"], meta(src, "arm A", "complete" if tm is not None else "none", f"stretch-{bench}"))


def load_hard():
    """Z3: congested lines (60 test); every rule at 0.1 % and 0.5 %, full MILP at 0.1 / 0.5 / 1 %"""
    src = "hard/hard_eval"
    recs = jls(os.path.join("results", "hard", "hard_eval_s*.jsonl"))
    full = {}
    for r in recs:
        if r["rule"] == "full MILP":
            x = ("hard", r["i"])
            REG.bound(x, r["bound"] if finite(r["bound"]) else r["obj"] * (1 - (r["mip_gap"] or 0)), src)
            full[(x, r["mip_rel_gap"])] = r
    for r in recs:
        x = ("hard", r["i"])
        F, fb = full[(x, 0.001)], full.get((x, r["mip_rel_gap"]), full[(x, 0.001)])
        t = r["time"] + r.get("pre_s", 0.0) + r.get("relax_s", 0.0) + r.get("guard_s", 0.0)
        mt = meta(src, "solver" if r["rule"] == "full MILP" else "fixing", "complete", "Z3")
        nm = f"Z3 | {r['rule']} @{100 * r['mip_rel_gap']:g} %"
        if r["feasible"]:
            REG.put(nm, x, r["obj"], slack(r) < TOL, t, F["time"], mt)
        else:
            REG.put(nm, x, fb["obj"], slack(fb) < TOL, t + fb["time"], F["time"], mt, fellback=True)


def load_pglib():
    """X7: PGLib-UC California (610 units, 30 test); reference = the full MILP re-run back to back in the same process
    ('full'), where recorded; rule time = reduced MILP + extra (LP relaxation, inference, rule) + LP guard"""
    src = "pglib/pglib_eval_test"
    full_t = {}
    recs = jl(os.path.join("results", "pglib", "pglib_eval_test_a.jsonl")) + jl(os.path.join("results", "pglib", "pglib_eval_test_b.jsonl"))
    extra = jl(os.path.join("results", "pglib", "pglib_eval_test_c.jsonl"))
    for r in recs:
        x = ("ca", r["i"])
        REG.bound(x, r["bound"], src)
        REG.lp(x, r["c_rel"], src)
        F = r.get("full")
        if F:
            REG.bound(x, F["bound"], src)
        full_t[x] = (F or {}).get("time", r["t_full"])          # paired re-run where recorded (6 of 30), else stored
        REG.put("X7 | full MILP (stored label run)", x, r["obj"], True, r["t_full"], r["t_full"],
                meta(src, "solver", "unpaired reference", "X7"))
        if F:
            REG.put("X7 | full MILP (re-run)", x, F["obj"], slack(F) < TOL, F["time"], full_t[x], meta(src, "solver", "complete", "X7"))
    for r in recs + extra:
        x = ("ca", r["i"])
        for nm, v in r["rules"].items():
            t = v["time"] + v.get("extra", 0.0) + v.get("lpg_s", 0.0)
            mt = meta(src, "no learning" if nm.startswith("lprelax") else "fixing", "unpaired reference", "X7")
            name = f"X7 | {nm}"
            if name in REG.methods and x in REG.methods[name]["data"]:
                continue
            if v["feasible"]:
                REG.put(name, x, v["obj"], slack(v) < TOL, t, full_t[x], mt)
            else:
                REG.put(name, x, r["obj"], True, t + r["t_full"], full_t[x], mt, fellback=True)
        if r["e2e"]:
            cold = r["e2e"][0]["lp_s"]
            for srcn in sorted({e["src"] for e in r["e2e"]}):
                es = [e for e in r["e2e"] if e["src"] == srcn]
                b = min(es, key=lambda e: e["cost"])
                tm = es[0]["t_in"] + sum(e["rep_s"] + e["lp_s"] for e in es) - es[0]["lp_s"] + cold
                REG.put(f"X7 | end to end, {srcn} p, screening {len(es)} thresholds", x, b["cost"],
                        b["shed"] + b["short"] < TOL, tm, full_t[x],
                        meta(src, "no learning" if srcn == "lprelax" else "end to end", "unpaired reference", "X7"))


def load_others():
    load_runs_file(os.path.join("results", "uc24", "m24_eval_test.jsonl"), "u24", "Z2", "Z2", "uc24/m24_eval_test")
    load_runs_file(os.path.join("results", "uc24", "m24_fu_paired_*.jsonl"), "u24", "Z2fu", "Z2fu", "uc24/m24_fu_paired")
    load_base("uc24", "u24", "Y1·24", "Y1·24")
    load_uc24ltf()
    load_arm_b_cpu("uc24", "u24", "B·cpu 24 h")
    load_stretch("uc24", "u24")
    load_hard()
    load_arm_b_cpu("hard", "hard", "B·cpu congested")
    load_stretch("hard", "hard")
    load_pglib()


def load_all():
    load_arm_a(os.path.join(RREC, "gpu_round", "trm_eval_test_fresh_s*.jsonl"), "A·gpu | ", "gpu-A",
               "recursive/gpu_round/trm_eval_test_fresh")
    load_arm_a(os.path.join(RREC, "trm_eval_test_fresh_s*.jsonl"), "A·cpu | ", "cpu-A", "recursive/trm_eval_test_fresh")
    load_arm_b_gpu("w2_rloop_rl", "B·gpu fb")
    load_arm_b_gpu("w1_rloop_nofb_rl", "B·gpu nofb")
    load_arm_b_cpu()
    load_m1x_pipeline()
    load_cguard()
    for fn, pre, run in (("hybrid_eval_test_fresh.jsonl", "X6 | ", "X6"), ("ltfx_eval_test_fresh.jsonl", "X3x | ", "X3x"),
                         ("m1x_eval_test_fresh.jsonl", "Y3 | ", "Y3")):
        load_rule_file(os.path.join(R12, fn), pre, run)
    load_base()
    load_ltf_eval()
    load_combo_fix()
    load_combo_e2e()
    load_ood()
    load_others()


# ============================================================================ statistics
def db_of(x):
    vals = [v for v, _ in REG.bounds.get(x, [])] + [v for v, _ in REG.crel.get(x, [])]
    return max(vals) if vals else None


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def boot_ci(xs, key):
    if len(xs) < 2:
        return None
    rng = random.Random(zlib.crc32(key.encode()))
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return [ms[int(0.025 * BOOT)], ms[int(0.975 * BOOT) - 1]]


def stats(name, insts):
    M = REG.methods[name]
    d = M["data"]
    rows = [(x, d[x]) for x in insts]
    gaps = [(c - db_of(x)) / c * 100 for x, (c, s, t, tr, fb) in rows]
    srv = [s for x, (c, s, t, tr, fb) in rows]
    out = dict(n=len(rows), gap_mean=sum(gaps) / len(gaps), gap_ci=boot_ci(gaps, name + str(len(rows))),
               gap_median=pct(gaps, 0.5), gap_p95=pct(gaps, 0.95), gap_max=max(gaps),
               share_gt1=100 * sum(g > 1 for g in gaps) / len(gaps), served=100 * sum(srv) / len(srv),
               fallback=100 * sum(v[4] for _, v in rows) / len(rows), timing=M["meta"]["timing"], run=M["meta"]["run"],
               source=M["meta"]["source"], family=M["meta"]["family"])
    ts = [(t, tr) for x, (c, s, t, tr, fb) in rows if t is not None and tr is not None and t > 0]
    if len(ts) == len(rows):
        sp = [tr / t for t, tr in ts]
        out.update(time_mean=sum(t for t, _ in ts) / len(ts), sp_geo=math.exp(sum(math.log(v) for v in sp) / len(sp)),
                   sp_median=pct(sp, 0.5), sp_mean=sum(sp) / len(sp),
                   sp_total=sum(tr for _, tr in ts) / sum(t for t, _ in ts))
    return out


def gaps_of(name, insts):
    d = REG.methods[name]["data"]
    return [(d[x][0] - db_of(x)) / d[x][0] * 100 for x in insts]


def common_served_gap(names, insts):
    both = [x for x in insts if all(REG.methods[n]["data"][x][1] for n in names)]
    return both, {n: (sum(gaps_of(n, both)) / len(both) if both else None) for n in names}


# ============================================================================ checks
def checks():
    out = {}
    # 1. the LP relaxation is deterministic: every file's value for an instance must agree (a shifted index would not)
    bad = []
    for x, vals in REG.crel.items():
        v0 = vals[0][0]
        bad += [(x, s, v) for v, s in vals if abs(v - v0) > 1e-7 * abs(v0)]
    out["lp_relaxation_disagreements"] = len(bad)
    out["lp_relaxation_instances"] = len(REG.crel)
    # 2. every delivered cost must be >= the tightest bound (and every bound >= the LP relaxation)
    viol = []
    for name, M in REG.methods.items():
        for x, (c, *_r) in M["data"].items():
            db = db_of(x)
            if db is not None and c < db * (1 - 1e-6):
                viol.append((name, x, c, db))
    out["cost_below_bound"] = len(viol)
    out["cost_below_bound_examples"] = [dict(method=a, inst=list(b), cost=c, db=d) for a, b, c, d in viol[:10]]
    lpv = []
    for x, vals in REG.bounds.items():
        if x in REG.crel:
            lpv += [(x, v, s) for v, s in vals if v < REG.crel[x][0][0] * (1 - 1e-6)]
    out["bounds_below_lp_relaxation"] = len(lpv)
    # 3. how much the tightest bound tightens over each source's own bound (mean, in % of DB)
    tight = {}
    for x, vals in REG.bounds.items():
        best = db_of(x)
        for v, s in vals:
            tight.setdefault(s, []).append((best - v) / best * 100)
    out["own_bound_shortfall_pct"] = {s: dict(mean=sum(v) / len(v), max=max(v), n=len(v)) for s, v in sorted(tight.items())}
    # 3b. what is still unknown: best known cost of the instance vs its tightest bound
    best = {}
    for name, M in REG.methods.items():
        for x, v in M["data"].items():
            best[x] = min(best.get(x, float("inf")), v[0])
    unres = {x: (best[x] - db_of(x)) / best[x] * 100 for x in best if db_of(x) is not None}
    tf = [v for x, v in unres.items() if x[0] == "tf"]
    out["unresolved_gap_tf"] = dict(n=len(tf), mean=sum(tf) / len(tf), max=max(tf), above_0p1=sum(v > 0.1 for v in tf),
                                    above_0p5=sum(v > 0.5 for v in tf),
                                    worst=sorted(([x[1], round(v, 3)] for x, v in unres.items() if x[0] == "tf"),
                                                 key=lambda z: -z[1])[:8])
    sh = [v for x, v in unres.items() if x[0] != "tf"]
    out["unresolved_gap_shifted"] = dict(n=len(sh), mean=sum(sh) / len(sh), max=max(sh), above_0p1=sum(v > 0.1 for v in sh))
    st = [r["milp01"]["status"] for r in jls(os.path.join(RREC, "gpu_round", "trm_eval_test_fresh_s*.jsonl"))]
    out["reference_milp_time_limited"] = dict(n=len(st), time_limited=sum(x != "optimal" for x in st))
    # 4. robustness study's in-distribution set = test_fresh 0-39; shifted sets of Y4 = those of Y6 (full MILP costs)
    pairs = []
    y4, y6 = REG.methods.get("Y4 | full MILP"), REG.methods.get("Y6 | full MILP")
    ref = REG.methods.get("A·gpu | milp01")
    if y4 and ref:
        for x, v in y4["data"].items():
            if x[0] == "tf":
                pairs.append(("Y4 id vs gpu-A", x, v[0], ref["data"][x][0]))
    if y4 and y6:
        for x, v in y4["data"].items():
            if x[0] != "tf" and x in y6["data"]:
                pairs.append(("Y4 vs Y6 shifted", x, v[0], y6["data"][x][0]))
    agree = {}
    for tag, x, a, b in pairs:
        agree.setdefault(tag, []).append(abs(a - b) / b)
    out["full_milp_agreement"] = {t: dict(n=len(v), within_0p2pct=sum(e <= 2e-3 for e in v), max_rel=max(v))
                                  for t, v in agree.items()}
    # 5. the same rule measured in several files: share of instances with the same cost (deterministic up to time limits)
    dup = [("hybrid", ["X6 | hybrid he_bce_s0_e1_n360", "Y3 | hybrid he_bce_s0_e1_n360", "Y6 | hybrid",
                       "Y1 | hybrid (he_bce_s0_e1_n360)"]),
           ("LtF kNN eps=1%", ["X6 | faithful LtF kNN eps=1%", "X3x | knn ltf eps=1%", "Y1 | LtF kNN eps=1%"]),
           ("full MILP 0.1 %", ["A·gpu | milp01", "B·gpu fb | full MILP 0.1 %", "Y6 | full MILP", "Z1 | full MILP 0.1 %",
                                "X6 | full MILP", "Y1 | full"]),
           ("LP rounding", ["A·gpu | lpround5", "B·gpu fb | LP rounding + repair, 5 LPs"])]
    rep = {}
    for tag, names in dup:
        names = [n for n in names if n in REG.methods]
        base = REG.methods[names[0]]["data"]
        rep[tag] = {}
        for n in names[1:]:
            d = REG.methods[n]["data"]
            xs = [x for x in d if x in base]
            same = sum(abs(d[x][0] - base[x][0]) <= 1e-6 * base[x][0] for x in xs)
            close = sum(abs(d[x][0] - base[x][0]) <= 2e-3 * base[x][0] for x in xs)
            rep[tag][f"{n} vs {names[0]}"] = dict(n=len(xs), identical=same, within_0p2pct=close)
    out["same_rule_in_several_files"] = rep
    return out


# ============================================================================ reproduction of published numbers
def reproduce():
    """each study's own convention on its own records, against the numbers it published"""
    out = {}
    # GPU round (results/recursive/gpu_round/test_results.json): DB = max over the 0.1 % full-MILP bounds of the
    # GPU-round files only (arm A milp01, arm B full 0.1 % of both models), as scripts/rec_gpu_test_report.py
    pub = json.load(open(os.path.join(RREC, "gpu_round", "test_results.json")))["methods"]
    b01 = {}
    for r in jls(os.path.join(RREC, "gpu_round", "trm_eval_test_fresh_s*.jsonl")):
        b01.setdefault(r["i"], []).append(r["milp01"]["bound"])
    for tag in ("w2_rloop_rl", "w1_rloop_nofb_rl"):
        for r in jls(os.path.join(RREC, "gpu_round", f"rloop_eval_uc12_{tag}_s*.jsonl")):
            b01.setdefault(r["k"], []).append(r["full"]["0.001"]["bound"])
    mapping = {"w2_rloop_rl @24 steps": "B·gpu fb | @24 steps", "w2_rloop_rl @24 + safety net": "B·gpu fb | @24 + safety net",
               "w1_rloop_nofb_rl @24 steps": "B·gpu nofb | @24 steps", "A|w1_trm30_mix5r": "A·gpu | w1_trm30_mix5r",
               "A|w1_oneshot_mix5r": "A·gpu | w1_oneshot_mix5r", "A|w1_deep_fb": "A·gpu | w1_deep_fb",
               "A|lpround5r": "A·gpu | lpround5r", "A|milp05": "A·gpu | milp05"}
    rows = {}
    for p, mine in mapping.items():
        d = REG.methods[mine]["data"]
        xs = sorted(d)
        g = [(d[x][0] - max(b01[x[1]])) / d[x][0] * 100 for x in xs]
        rows[p] = dict(published=pub[p]["gap_mean"], recomputed=sum(g) / len(g))
    out["gpu_round"] = rows
    # Y6 (results/uc12/m1x_pipeline.json): own run's bound, feasible instances only, mean of per-instance speed-ups
    pubY = json.load(open(os.path.join(R12, "m1x_pipeline.json")))["sets"]["tf"]["stats"]
    rows = {}
    raw = jls(os.path.join(R12, "m1x_pipeline_eval_s*.jsonl"))
    F = {r["k"]: r for r in raw if r["set"] == "tf" and r["rule"] == "full MILP"}
    for rule in pubY:
        R = {r["k"]: r for r in raw if r["set"] == "tf" and r["rule"] == rule}
        if len(R) != 120:
            continue
        g, sp = [], []
        for k, r in R.items():
            if not r["feasible"]:
                continue
            db = F[k]["bound"] if finite(F[k]["bound"]) else F[k]["obj"] * (1 - F[k]["mip_gap"])
            g.append((r["obj"] - db) / r["obj"] * 100)
            sp.append(F[k]["time"] / (r["time"] + r["pre_s"] + r["relax_s"] + r["guard_s"]))
        rows[rule] = dict(published_gap=pubY[rule]["gap_mean"], recomputed_gap=sum(g) / len(g),
                          published_sp=pubY[rule]["sp_mean"], recomputed_sp=sum(sp) / len(sp))
    out["Y6"] = rows
    # California (results/pglib/pglib_results.json): DB = the stored label run's bound, feasible and shed-free only,
    # speed-up = stored MILP time / (reduced MILP + extra + LP guard), arithmetic mean
    pubP = {r["method"]: r for r in json.load(open(os.path.join("results", "pglib", "pglib_results.json")))["rows"]}
    recs = jl(os.path.join("results", "pglib", "pglib_eval_test_a.jsonl")) + jl(os.path.join("results", "pglib", "pglib_eval_test_b.jsonl"))
    pm = {"harm_90": "ours: error-cost ranking + adequacy guard, 90 %", "harm_98_lpg": "ours: error-cost + adequacy + LP guard, 98 % (combined)",
          "ltf_knn_1": "**Learning to Fix, kNN, eps = 1 %** (paper's setting)", "lprelax_integral": "no learning: fix the integral values of the LP relaxation"}
    rows = {}
    for k, pubname in pm.items():
        g, sp = [], []
        for r in recs:
            v = r["rules"][k]
            # that report counts a reduced problem that needs shedding / reserve shortfall as infeasible (PGLib's own
            # model has hard balance and reserve rows) and leaves it out of the mean
            if v["feasible"] and (v["shed"] or 0) + (v["short"] or 0) <= 1e-6:
                g.append((v["obj"] - r["bound"]) / v["obj"] * 100)
                sp.append(r["t_full"] / (v["time"] + v["extra"] + v.get("lpg_s", 0.0)))
        rows[k] = dict(published_gap=pubP[pubname]["gap_mean"], recomputed_gap=sum(g) / len(g),
                       published_sp=pubP[pubname]["speedup_mean"], recomputed_sp=sum(sp) / len(sp))
    out["California"] = rows
    # congested (results/hard/hard_results.json): DB = the instance's full MILP at 0.1 %, feasible instances only,
    # speed-up geometric mean against that MILP
    pubH = json.load(open(os.path.join("results", "hard", "hard_results.json")))["test"]["rows"]
    recs = jls(os.path.join("results", "hard", "hard_eval_s*.jsonl"))
    F = {r["i"]: r for r in recs if r["rule"] == "full MILP" and r["mip_rel_gap"] == 0.001}
    rows = {}
    for rule, gap, pubname in (("pipeline", 0.005, "pipeline @ 0.5%"), ("LtF-kNN", 0.001, "LtF-kNN @ 0.1%"),
                               ("full MILP", 0.005, "full MILP @ 0.5%"), ("LtF-LP", 0.005, "LtF-LP @ 0.5%")):
        R = [r for r in recs if r["rule"] == rule and r["mip_rel_gap"] == gap]
        g, lsp = [], []
        for r in R:
            if not r["feasible"]:
                continue
            f = F[r["i"]]
            db = f["bound"] if finite(f["bound"]) else f["obj"] * (1 - f["mip_gap"])
            g.append((r["obj"] - db) / r["obj"] * 100)
            lsp.append(math.log(f["time"] / (r["time"] + r["pre_s"] + r["relax_s"] + r["guard_s"])))
        rows[pubname] = dict(published_gap=pubH[pubname]["gap_mean"], recomputed_gap=sum(g) / len(g),
                             published_sp=pubH[pubname]["sp_geo"], recomputed_sp=math.exp(sum(lsp) / len(lsp)))
    out["congested"] = rows
    # 24 h follow-up (results/uc24/m24_results.json followup.test): DB = the run's full MILP bound, feasible only,
    # speed-up geometric mean against that run's full MILP
    pubU = json.load(open(os.path.join("results", "uc24", "m24_results.json")))["followup"]["test"]
    recs = jls(os.path.join("results", "uc24", "m24_fu_paired_*.jsonl"))
    rows = {}
    for nm in ("full gap=0.5%", "LtF-LP gap=0.5%", "m24", "m24cal"):
        if nm not in pubU:
            continue
        g, lsp = [], []
        for r in recs:
            v, f = r["runs"][nm], r["runs"]["full"]
            if not v["feasible"]:
                continue
            g.append((v["obj"] - f["bound"]) / v["obj"] * 100)
            lsp.append(math.log(f.get("t_method", f["time"]) / v.get("t_method", v["time"])))
        rows[nm] = dict(published_gap=pubU[nm]["gap_mean"], recomputed_gap=sum(g) / len(g),
                        published_sp=pubU[nm]["sp_geo"], recomputed_sp=math.exp(sum(lsp) / len(lsp)))
    out["uc24 follow-up"] = rows
    return out


# ============================================================================ tables
def ids(kind):
    if isinstance(kind, tuple):
        return [(kind[0], k) for k in range(kind[1])]
    if kind == "tf120":
        return [("tf", i) for i in range(120)]
    if kind == "tf60":
        return [("tf", i) for i in range(60)]
    return [(kind, k) for k in range(30)]


def complete_on(name, insts):
    d = REG.methods[name]["data"]
    return all(x in d for x in insts)


TABLES = {
    # name: (instance set, [(label, method)])
    "12 h, all 120 fresh instances (complete timing)": ("tf120", [
        ("Full MILP, 0.1 % gap (reference)", "A·gpu | milp01"),
        ("Full MILP, 0.5 % gap", "A·gpu | milp05"),
        ("Recommended pipeline (Y6)", "Y6 | pipeline: B-hg + LP guard, gap 0.5%"),
        ("Recommended pipeline + bound check (Z1)", "Z1 | pipeline + bound check"),
        ("Hybrid, 0.1 % gap (Y6)", "Y6 | hybrid"),
        ("Ensemble fixings + LP guard, 0.1 % gap (Y6)", "Y6 | B-hg + LP guard"),
        ("Arm B, 24 steps + safety net", "B·gpu fb | @24 + safety net"),
        ("Arm B without feedback + safety net", "B·gpu nofb | @24 + safety net"),
        ("Arm B, 24 steps (no MILP)", "B·gpu fb | @24 steps"),
        ("Arm B without feedback, 24 steps", "B·gpu nofb | @24 steps"),
        ("Arm B, halting head", "B·gpu fb | halting head"),
        ("Arm A control: deep network + safety net", "A·gpu | w1_deep_fb"),
        ("Arm A control: one-shot + safety net", "A·gpu | w1_oneshot_fb"),
        ("Arm A: TRM + safety net", "A·gpu | w1_trm30_fb"),
        ("Arm A control: deep network, mixed + repair", "A·gpu | w1_deep_mix5r"),
        ("Arm A control: one-shot, mixed + repair", "A·gpu | w1_oneshot_mix5r"),
        ("Arm A: TRM, mixed + repair", "A·gpu | w1_trm30_mix5r"),
        ("Arm A: TRM, 1 LP + repair", "A·gpu | w1_trm30_1r"),
        ("Temporal-GNN ensemble, mixed + repair", "A·gpu | m1x_mix5r"),
        ("Label-free + REINFORCE, 1 LP + repair (CPU round)", "A·cpu | rl_rl1r"),
        ("Lagrangian + KL, 7 thresholds + repair (CPU round)", "A·cpu | rl_lag7r"),
        ("CPU round arm B: guided search + safety net", "B·cpu | guided search + safety net (theta 1 %)"),
        ("CPU round arm B: guided local search, 16 LPs", "B·cpu | local search guided by rloop, +16 LPs"),
        ("LP-checked local search (no learning)", "B·gpu fb | LP-checked local search"),
        ("LP rounding + repair (no learning)", "A·gpu | lpround5r"),
    ]),
    "12 h, first 60 fresh instances: fixing rules and baselines (complete timing)": ("tf60", [
        ("Full MILP, 0.1 % gap (reference, Y1 run)", "Y1 | full"),
        ("Full MILP, 0.25 % gap", "Y1 | full gap=0.25%"),
        ("Full MILP, 0.5 % gap", "Y1 | full gap=0.5%"),
        ("Full MILP, 1 % gap", "Y1 | full gap=1%"),
        ("Recommended pipeline (Y6)", "Y6 | pipeline: B-hg + LP guard, gap 0.5%"),
        ("Recommended pipeline + bound check (Z1)", "Z1 | pipeline + bound check"),
        ("Hybrid (X6)", "X6 | hybrid he_bce_s0_e1_n360"),
        ("Learning to Fix, kNN, eps 1 % (paper's setting)", "X6 | faithful LtF kNN eps=1%"),
        ("Learning to Fix, kNN, eps 5 %", "X3x | knn ltf eps=5%"),
        ("Learning to Fix, kNN, eps 10 %", "X3x | knn ltf eps=10%"),
        ("Learning to Fix tuning on our BCE GNN, eps 1 %", "X6 | faithful LtF BCE eps=1%"),
        ("Ours: error-cost + adequacy guard, 90 %", "X6 | ours: error-cost + adequacy guard 90% (BCE)"),
        ("Ours: combined pipeline, 98 %", "X6 | ours: combined st+error-cost+adequacy+LP guard 98%"),
        ("Learning to Fix tuning on the LP relaxation (no learning)", "Y1 | LtF on LP relaxation eps=1%"),
        ("LP-integral fixing + guards (no learning)", "Y1 | LP-integral (tol 0.05) + guards"),
        ("LP rounding + repair (no learning, Y1 run)", "Y1 | LP rounding + repair + LPs"),
        ("Arm B, 24 steps + safety net", "B·gpu fb | @24 + safety net"),
        ("Arm B, 24 steps (no MILP)", "B·gpu fb | @24 steps"),
    ]),
    "12 h, first 60: rules whose saved time covers the MILP only (gap and served comparable; speed-ups not)": ("tf60", [
        ("Full MILP, 0.1 % gap (X3 run)", "X3 | full"),
        ("RACLearn confidence, 90 % fixed", "X3 | rac@0.9"),
        ("RACLearn confidence, 95 % fixed", "X3 | rac@0.95"),
        ("Asymmetric + adequacy guard, 90 %", "X3 | asym@0.9"),
        ("Asymmetric + adequacy guard, 95 %", "X3 | asym@0.95"),
        ("Error-cost ranking + guard, 90 %", "X3 | harm@0.9"),
        ("Error-cost ranking + guard + LP guard, 95 %", "X3 | harm@0.95+lp"),
        ("REINFORCE probabilities, 95 %", "X3 | rl@0.95"),
    ]),
}
OTHER_TABLES = {
    "24 hours (uc24), 40 test instances": (("u24", 40), [
        ("Full MILP, 0.1 % gap (reference)", "Z2fu | full"),
        ("Full MILP, 0.5 % gap", "Z2fu | full gap=0.5%"),
        ("Learned recipe: ensemble fixings + LP guard, 0.5 % (tuned thresholds)", "Z2fu | m24"),
        ("Learned recipe with a cross-validated margin", "Z2fu | m24cal"),
        ("Learning to Fix on the LP relaxation, 0.5 % (no learning)", "Z2fu | LtF-LP gap=0.5%"),
        ("Guarded 95 % rule (imitation GNN)", "Z2 | guarded 95%"),
        ("Learning to Fix, kNN, eps 1 %", "X8 | LtF knn eps=1%"),
        ("Arm B (CPU round): guided search + safety net", "B·cpu 24 h | guided search + safety net (theta 1 %)"),
        ("Arm B (CPU round): guided search, 16 LPs", "B·cpu 24 h | local search guided by rloop, +16 LPs"),
        ("LP-checked local search (no learning)", "B·cpu 24 h | LP-checked local search, +16 LPs"),
        ("LP rounding + repair (no learning)", "B·cpu 24 h | LP rounding + repair, 5 LPs"),
        ("Arm A (CPU round), zero-shot, 16 steps, mixed + repair", "A·cpu stretch uc24 | main_k16_mix5r"),
    ]),
    "24 hours, first 20 instances: baselines re-run back to back (Y1)": (("u24", 20), [
        ("Full MILP, 0.1 % gap (reference)", "Y1·24 | full"),
        ("Full MILP, 0.5 % gap", "Y1·24 | full gap=0.5%"),
        ("Full MILP, 1 % gap", "Y1·24 | full gap=1%"),
        ("Learning to Fix, kNN, eps 1 %", "Y1·24 | LtF kNN eps=1%"),
        ("Guarded 95 % rule (imitation GNN)", "Y1·24 | ours: guarded 95% (imitation GNN)"),
        ("Learning to Fix on the LP relaxation (no learning)", "Y1·24 | LtF on LP relaxation eps=1%"),
        ("LP-integral fixing + guards (no learning)", "Y1·24 | LP-integral (tol 0.2) + guards"),
        ("LP rounding + repair (no learning)", "Y1·24 | LP rounding + repair + LPs"),
    ]),
    "Congested lines (ratings × 0.6), 60 test instances": (("hard", 60), [
        ("Full MILP, 0.1 % gap (reference)", "Z3 | full MILP @0.1 %"),
        ("Full MILP, 0.5 % gap", "Z3 | full MILP @0.5 %"),
        ("Full MILP, 1 % gap", "Z3 | full MILP @1 %"),
        ("Learned pipeline, 0.5 %", "Z3 | pipeline @0.5 %"),
        ("Learned pipeline, 0.1 %", "Z3 | pipeline @0.1 %"),
        ("Learning to Fix, kNN, eps 1 %", "Z3 | LtF-kNN @0.1 %"),
        ("Learning to Fix on the LP relaxation, 0.5 % (no learning)", "Z3 | LtF-LP @0.5 %"),
        ("LP pipeline: LP values + guards, 0.5 % (no learning)", "Z3 | LP pipeline @0.5 %"),
        ("LP-integral fixing + guards, 0.5 % (no learning)", "Z3 | LP-int+g @0.5 %"),
        ("Arm B (CPU round): guided search + safety net", "B·cpu congested | guided search + safety net (theta 1 %)"),
        ("Arm B (CPU round): guided search, 16 LPs", "B·cpu congested | local search guided by rloop, +16 LPs"),
        ("LP rounding + repair (no learning)", "B·cpu congested | LP rounding + repair, 5 LPs"),
        ("Arm A (CPU round), zero-shot, 16 steps, mixed + repair", "A·cpu stretch hard | main_k16_mix5r"),
    ]),
    "PGLib-UC California (610 units, copper plate), 30 test instances": (("ca", 30), [
        ("Full MILP, 0.1 % target (stored run)", "X7 | full MILP (stored label run)"),
        ("Ours: error-cost + adequacy guard, 90 %", "X7 | harm_90"),
        ("Ours: error-cost + adequacy + LP guard, 95 %", "X7 | harm_95_lpg"),
        ("Ours: error-cost + adequacy + LP guard, 98 %", "X7 | harm_98_lpg"),
        ("Learning to Fix, kNN, eps 1 %", "X7 | ltf_knn_1"),
        ("Learning to Fix, kNN, eps 5 %", "X7 | ltf_knn_5"),
        ("Learning to Fix on our self-trained model, eps 1 %", "X7 | ltf_st_1"),
        ("RACLearn-style: self-trained p, 95 % most confident", "X7 | st_rac_95"),
        ("RACLearn-style: REINFORCE p, 95 % most confident", "X7 | rl_rac_95"),
        ("Fix the integral LP values (no learning)", "X7 | lprelax_integral"),
        ("LP rounding, 5 thresholds (no learning)", "X7 | end to end, lprelax p, screening 5 thresholds"),
        ("End to end, self-trained p, 5 thresholds", "X7 | end to end, st p, screening 5 thresholds"),
        ("End to end, REINFORCE p, 5 thresholds", "X7 | end to end, rl p, screening 5 thresholds"),
    ]),
}
SHIFT_TABLE = [("Full MILP, 0.5 % gap", "Y6 | full MILP gap 0.5%"), ("Recommended pipeline", "Y6 | pipeline: B-hg + LP guard, gap 0.5%"),
               ("Pipeline + bound check", "Z1 | pipeline + bound check"), ("Hybrid", "Y6 | hybrid"),
               ("Learning to Fix, kNN, eps 1 %", "Y4 | LtF kNN eps=1%"), ("Learning to Fix, BCE GNN, eps 1 %", "Y4 | LtF BCE eps=1%"),
               ("Error-cost + guard, 90 %", "Y4 | guarded error-cost 90%"), ("Combined, 98 %", "Y4 | combined 98%"),
               ("LP-integral fixing + guards (no learning)", "Y4 | no learning: LP-integral fixings + guards"),
               ("Learning to Fix, kNN + our LP guard", "Y4fu | LtF kNN + LP guard"),
               ("Learning to Fix on the LP relaxation (no learning)", "Y4fu | LtF LP-relaxation eps=1%"),
               ("End to end: REINFORCE, 1 LP", "Y4 | e2e RL 1 LP"), ("End to end: Lagrangian, screening", "Y4 | e2e lag screen")]


def fmt(v, p=2):
    return "–" if v is None else f"{v:.{p}f}"


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_all()
    res = dict(protocol=__doc__.split("\n\n")[1], checks=checks(), reproduce=reproduce(), tables={}, all_methods={},
               notes=REG.notes)
    L = ["# Unified re-scoring of the saved test results", "",
         "Generated by `scripts/unified_rescore.py` from the per-instance records; nothing re-solved. Protocol in the script "
         "docstring and `docs/unified_metrics.md`.", ""]
    for title, (kind, rows) in TABLES.items():
        insts = ids(kind)
        have = [(lab, n) for lab, n in rows if n in REG.methods and complete_on(n, insts)]
        miss = [n for lab, n in rows if not (n in REG.methods and complete_on(n, insts))]
        both, cs = common_served_gap([n for _, n in have], insts)
        T = []
        for lab, n in have:
            s = stats(n, insts)
            s.update(label=lab, method=n, gap_common_served=cs[n])
            T.append(s)
        T.sort(key=lambda s: s["gap_mean"])
        res["tables"][title] = dict(instances=kind, n_common_served=len(both), rows=T, missing=miss)
        L += [f"## {title}", "",
              f"Gap on the {len(both)} instances every method in this table serves: column 'common'.", "",
              "| method | gap mean [95 % CI] | median | p95 | max | > 1 % | common | served | fallback | time s | speed-up geo / median / mean | timing | run |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in T:
            ci = s["gap_ci"]
            sp = (f"{s['sp_geo']:.1f}× / {s['sp_median']:.1f}× / {s['sp_mean']:.1f}×"
                  if "sp_geo" in s and s["timing"] == "complete" else "–")
            L.append(f"| {s['label']} | {s['gap_mean']:.3f} [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] | {s['gap_median']:.3f} | "
                     f"{s['gap_p95']:.2f} | {s['gap_max']:.2f} | {s['share_gt1']:.0f} % | {fmt(s['gap_common_served'], 3)} | "
                     f"{s['served']:.1f} % | {s['fallback']:.0f} % | {fmt(s.get('time_mean')) if s['timing'] == 'complete' else '–'} | {sp} | "
                     f"{s['timing']} | {s['run']} |")
        if miss:
            L += ["", "Not complete on this instance set: " + ", ".join(miss)]
        L.append("")
    # shifted sets
    L += ["## Shifted test sets (30 instances each): mean gap · served · speed-up (geometric mean)", "",
          "| method | " + " | ".join(SHIFTS) + " |", "|---|" + "---|" * len(SHIFTS)]
    res["tables"]["shifted"] = {}
    for lab, n in SHIFT_TABLE:
        cells = []
        for sft in SHIFTS:
            insts = ids(sft)
            if n in REG.methods and complete_on(n, insts):
                s = stats(n, insts)
                res["tables"]["shifted"].setdefault(lab, {})[sft] = s
                cells.append(f"{s['gap_mean']:.2f} % · {s['served']:.0f} % · " + (f"{s['sp_geo']:.1f}×" if "sp_geo" in s else "–"))
            else:
                cells.append("–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    # every method on its largest complete set
    for name in sorted(REG.methods):
        d = REG.methods[name]["data"]
        for kind in ("tf120", "tf60"):
            if complete_on(name, ids(kind)):
                res["all_methods"][name] = dict(instances=kind, **stats(name, ids(kind)))
                break
    for title, (kind, rows) in OTHER_TABLES.items():
        insts = ids(kind)
        have = [(lab, n) for lab, n in rows if n in REG.methods and complete_on(n, insts)]
        miss = [n for lab, n in rows if not (n in REG.methods and complete_on(n, insts))]
        both, cs = common_served_gap([n for _, n in have], insts)
        T = []
        for lab, n in have:
            st = stats(n, insts)
            st.update(label=lab, method=n, gap_common_served=cs[n])
            T.append(st)
        T.sort(key=lambda z: z["gap_mean"])
        res["tables"][title] = dict(instances=list(kind), n_common_served=len(both), rows=T, missing=miss)
        L += [f"## {title}", "", f"Gap on the {len(both)} instances every method in this table serves: column 'common'.", "",
              "| method | gap mean [95 % CI] | median | p95 | max | > 1 % | common | served | fallback | time s | speed-up geo / median / mean | timing | run |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for z in T:
            ci = z["gap_ci"]
            ok = "sp_geo" in z and z["timing"] in ("complete", "unpaired reference")
            sp = f"{z['sp_geo']:.1f}× / {z['sp_median']:.1f}× / {z['sp_mean']:.1f}×" if ok else "–"
            L.append(f"| {z['label']} | {z['gap_mean']:.3f} [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] | {z['gap_median']:.3f} | "
                     f"{z['gap_p95']:.2f} | {z['gap_max']:.2f} | {z['share_gt1']:.0f} % | {fmt(z['gap_common_served'], 3)} | "
                     f"{z['served']:.1f} % | {z['fallback']:.0f} % | {fmt(z.get('time_mean')) if ok else '–'} | {sp} | "
                     f"{z['timing']} | {z['run']} |")
        if miss:
            L += ["", "Not complete on this instance set: " + ", ".join(miss)]
        L.append("")
    # seed spread: the only test results with several training seeds
    SEEDS = [("End to end: combined + screening (no MILP)", "X2 | e2e combined + screening (7 thresholds) s{}", "tf120"),
             ("Error-cost + guard, 90 %", "X2 | harm+guard @0.9 s{}", "tf60"),
             ("Error-cost + LP guard, 95 %", "X2 | harm+lp @0.95 s{}", "tf60"),
             ("Combined + LP guard, 95 %", "X2 | combo st+harm+lp @0.95 s{}", "tf60"),
             ("Combined + LP guard, 98 %", "X2 | combo st+harm+lp @0.98 s{}", "tf60"),
             ("REINFORCE ranking, 95 %", "X2 | rl rac @0.95 s{}", "tf60")]
    L += ["## Training seeds (the only test results with three seeds; gap and served per seed)", "",
          "| method | instances | seed 0 | seed 1 | seed 2 | mean gap | spread (max − min) |", "|---|---|---|---|---|---|---|"]
    res["seeds"] = {}
    for lab, pat, kind in SEEDS:
        st = [stats(pat.format(k), ids(kind)) for k in range(3) if pat.format(k) in REG.methods]
        if len(st) < 3:
            continue
        g = [x["gap_mean"] for x in st]
        res["seeds"][lab] = st
        L.append(f"| {lab} | {kind[2:]} | " + " | ".join(f"{x['gap_mean']:.2f} % · {x['served']:.0f} %" for x in st)
                 + f" | {sum(g) / 3:.2f} % | {max(g) - min(g):.2f} pp |")
    L.append("")
    # checks and reproduction
    c = res["checks"]
    L += ["## Checks", "",
          f"* LP relaxation, the same instance in different files: {c['lp_relaxation_disagreements']} disagreements over "
          f"{c['lp_relaxation_instances']} instances (exact match expected; a shifted index would break it).",
          f"* Delivered costs below the tightest bound: {c['cost_below_bound']} (0 expected).",
          f"* Bounds below the LP relaxation: {c['bounds_below_lp_relaxation']} (0 expected).",
          "* Full MILPs of the robustness study against the other runs: " + "; ".join(
              f"{t}: {v['within_0p2pct']}/{v['n']} within 0.2 % (max {100 * v['max_rel']:.2f} %)" for t, v in c["full_milp_agreement"].items()),
          "* The same rule measured in several files (identical cost / within 0.2 %):"]
    for tag, d in c["same_rule_in_several_files"].items():
        L.append(f"  * {tag}: " + "; ".join(f"{k.split(' vs ')[0]}: {v['identical']}/{v['n']} / {v['within_0p2pct']}/{v['n']}" for k, v in d.items()))
    u = c["unresolved_gap_tf"]
    L += [f"* Still unresolved (best known cost vs tightest bound) on test_fresh: mean {u['mean']:.3f} %, max {u['max']:.2f} %, "
          f"{u['above_0p1']} of {u['n']} instances above 0.1 %, {u['above_0p5']} above 0.5 % (worst: "
          + ", ".join(f"#{i} {v} %" for i, v in u["worst"]) + "); shifted sets: mean "
          f"{c['unresolved_gap_shifted']['mean']:.3f} %, {c['unresolved_gap_shifted']['above_0p1']} of {c['unresolved_gap_shifted']['n']} above 0.1 %."]
    L += [f"* Reference full MILP (0.1 %, 60 s) stopped by the time limit on {c['reference_milp_time_limited']['time_limited']} "
          f"of {c['reference_milp_time_limited']['n']} test_fresh instances (GPU-round arm A run)."]
    L += ["* How far each file's own bound is below the tightest one (mean / max, % of DB): " + "; ".join(
        f"{s}: {v['mean']:.3f} / {v['max']:.2f}" for s, v in c["own_bound_shortfall_pct"].items()), "",
        "Reproduction of published numbers with each study's own convention:", ""]
    for p, v in res["reproduce"]["gpu_round"].items():
        L.append(f"* GPU round, {p}: published {v['published']:.3f} %, recomputed {v['recomputed']:.3f} %")
    for p, v in res["reproduce"]["Y6"].items():
        L.append(f"* Y6, {p}: published {v['published_gap']:.3f} % at {v['published_sp']:.2f}×, recomputed "
                 f"{v['recomputed_gap']:.3f} % at {v['recomputed_sp']:.2f}×")
    for study in ("California", "congested", "uc24 follow-up"):
        for p, v in res["reproduce"][study].items():
            L.append(f"* {study}, {p}: published {v['published_gap']:.3f} % at {v['published_sp']:.2f}×, recomputed "
                     f"{v['recomputed_gap']:.3f} % at {v['recomputed_sp']:.2f}×")
    if REG.notes:
        L += ["", "Notes:"] + [f"* {n}" for n in REG.notes]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rescored.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    with open(os.path.join(OUT, "rescored.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
