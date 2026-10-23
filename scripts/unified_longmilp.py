"""Step 2 of docs/unified_metrics.md, bounds: one long full-MILP solve per instance whose best known cost is still more
than 0.1 % above its tightest proven bound, to tighten the bound DB that every gap is measured against.

    python3 scripts/unified_longmilp.py --plan                       # write the item list from the scored records
    python3 scripts/unified_longmilp.py --task 2 --ntasks 14         # one array task: its share of the list
    python3 scripts/unified_longmilp.py --pull --tag w3              # claim items from the list until none is left
                                                                     # (shifted instances wait until their file exists)

Each solve: the same UC MILP as every other full-MILP run of its benchmark (otsl.ood.OODModel, outages as in the
shifted sets; congested lines through otsl.hard's system), HiGHS through highspy with one thread, mip_rel_gap 1e-5,
a one-hour limit (--tl). The bound and the incumbent are recorded; times are not compared with anything.
Output: results/unified/retime/longmilp_<tag>.jsonl (resume skips finished items).
Progress: every --progress_s seconds of a solve, the solver's current dual bound (a proven lower bound at that moment)
and incumbent go to results/unified/retime/bounds_progress_<tag>.jsonl, so a solve that is killed (e.g. by the job's
memory limit) still leaves its last proven bound; scripts/unified_step2.py reads both files.
"""
import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
RES = "results/unified/retime"
ITEMS = os.path.join(RES, "longmilp_items.json")


def plan(threshold):
    """instances whose best known cost (any method, any run) is more than `threshold` % above the tightest bound"""
    import unified_step2 as S2
    import unified_rescore as U
    U.load_all()
    if os.path.isdir(RES):
        S2.ARGS = argparse.Namespace(res=RES)
        S2.load_fix(RES)
        S2.load_fix_addon(RES, "fixtune2")
        S2.load_fix_addon(RES, "fixtune3")
        S2.load_fix_addon(RES, "fixorig")
        S2.load_arm_a(RES)
        for b, kind, lab in (("uc12", "armB", "S2·B"), ("uc12", "armB0", "S2·B0"), ("uc24", "armB", "S2·B"),
                             ("hard", "armB", "S2·B"), ("line_out", "armB", "S2·B")):
            S2.load_arm_b(RES, b, kind, lab)
        S2.load_hard(RES)
        S2.load_longmilp(RES)
    best = {}
    for M in U.REG.methods.values():
        for x, v in M["data"].items():
            if U.finite(v[0]):
                best[x] = min(best.get(x, float("inf")), v[0])
    out = []
    for x, c in best.items():
        db = U.db_of(x)
        if db is None or x[0] == "ca":
            continue
        g = (c - db) / c * 100
        if g > threshold:
            bench, s = ("uc24", "tf") if x[0] == "u24" else ("hard", "tf") if x[0] == "hard" else ("uc12", x[0])
            out.append(dict(bench=bench, set=s, k=x[1], unresolved_pct=round(g, 4)))
    out.sort(key=lambda z: -z["unresolved_pct"])
    os.makedirs(RES, exist_ok=True)
    json.dump(dict(threshold_pct=threshold, n=len(out), items=out), open(ITEMS, "w"), indent=1)
    print(f"{len(out)} instances above {threshold} %: " + ", ".join(f"{z['bench']}:{z['set']}:{z['k']} {z['unresolved_pct']:.2f}%"
                                                                 for z in out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--tl", type=float, default=3600.0)
    ap.add_argument("--gap", type=float, default=1e-5)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_wait_h", type=float, default=4.0, help="pull: how long to wait for missing instance files")
    ap.add_argument("--claims", default="claims_bounds", help="pull: claims folder (a new one for a longer second pass)")
    ap.add_argument("--progress_s", type=float, default=300.0, help="seconds between progress lines (0: none)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    if a.plan:
        return plan(a.threshold)
    import numpy as np
    from otsl.b3 import load_b3
    from otsl.hard import bench_system, load_split, sc_of
    from otsl.ltfx import highs_solve
    from otsl.ood import OODModel, OODScenario
    from otsl.uc import load_rts_gmlc
    tag = a.tag or f"t{a.task}"
    out = os.path.join(RES, f"longmilp_{tag}.jsonl")
    done = {(r["bench"], r["set"], r["k"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    cache = {}
    have = lambda it: it["bench"] != "uc12" or it["set"] == "tf" or \
        os.path.exists(os.path.join("data/generated/uc12_ood", it["set"], f"{it['k']:03d}.npz"))

    def items():
        if not a.pull:
            yield from json.load(open(ITEMS))["items"][a.task::a.ntasks]
            return
        cl = os.path.join(RES, a.claims)
        os.makedirs(cl, exist_ok=True)
        t0 = time.time()
        while True:
            took = waiting = False
            for it in json.load(open(ITEMS))["items"]:                  # re-read: the list may be extended
                f = os.path.join(cl, f"{it['bench']}_{it['set']}_{it['k']}")
                if os.path.exists(f):
                    continue
                if not have(it):
                    waiting = True
                    continue
                try:
                    fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, f"{tag} {socket.gethostname()}\n".encode())
                os.close(fd)
                took = True
                yield it
                break
            if took:
                continue
            if not waiting or time.time() - t0 > a.max_wait_h * 3600:
                return
            time.sleep(60)

    prog = os.path.join(RES, f"bounds_progress_{tag}.jsonl")

    def progress(key):
        """subscribe a MIP callback that appends the current dual bound and incumbent every --progress_s seconds"""
        last = [time.time()]

        def cb(e):
            now = time.time()
            if now - last[0] < a.progress_s:
                return
            d = e.data_out
            if np.isfinite(d.mip_dual_bound):
                last[0] = now
                with open(prog, "a") as f:
                    f.write(json.dumps(dict(bench=key[0], set=key[1], k=key[2], t=round(d.running_time, 1),
                                            bound=d.mip_dual_bound, obj=d.mip_primal_bound, nodes=int(d.mip_node_count),
                                            host=socket.gethostname()), default=float) + "\n")
        return lambda h: h.cbMipInterrupt.subscribe(cb)

    for it in items():
        key = (it["bench"], it["set"], it["k"])
        if key in done:
            continue
        b, s, k = key
        if b == "hard":
            if "hard" not in cache:
                sysm, _ = bench_system()
                cache["hard"] = (OODModel(sysm, T=12), load_split("test"))
            m, d = cache["hard"]
            sc = sc_of(d, k)
        elif b == "uc24":
            if "uc24" not in cache:
                cache["uc24"] = (OODModel(load_rts_gmlc(), T=24), load_b3("data/generated/uc24/test.npz"))
            m, d = cache["uc24"]
            sc = OODScenario(load=d["load"][k], avail=d["avail"][k], u0=d["u0"][k], sr=d["sr"][k])
        else:
            if "uc12" not in cache:
                cache["uc12"] = (OODModel(load_rts_gmlc(), T=12), load_b3("data/generated/uc12/test_fresh.npz"))
            m, d = cache["uc12"]
            if s == "tf":
                sc = OODScenario(load=d["load"][k], avail=d["avail"][k], u0=d["u0"][k], sr=d["sr"][k])
            else:
                z = np.load(os.path.join("data/generated/uc12_ood", s, f"{k:03d}.npz"))
                sc = OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"], day=int(z["day"]),
                                 start=int(z["start"]), g_out=tuple(int(g) for g in z["g_out"]),
                                 l_out=tuple(int(x) for x in z["l_out"]))
        lo, hi, lb, ub = m._rhs_bounds(sc)
        t0 = time.time()
        r = highs_solve(m.c, m.A, lo, hi, lb, ub, m.integ, a.tl, a.gap, hook=progress(key) if a.progress_s > 0 else None)
        rec = dict(bench=b, set=s, k=k, status=r["status"], obj=r["obj"], bound=r["bound"], gap=r["gap"],
                   time=time.time() - t0, tl=a.tl, mip_rel_gap=a.gap, host=socket.gethostname(),
                   unresolved_before_pct=it["unresolved_pct"])
        if r["x"] is not None:
            x, T = r["x"], m.T
            get = lambda name: x[m.off[name][0]:m.off[name][0] + m.off[name][1] * T]
            rec.update(shed=float(get("shed").sum()), spill=float(get("spill").sum()), short=float(get("short").sum()))
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"{key}: {r['status']} obj {r['obj']:.1f} bound {r['bound']:.1f} ({(r['obj'] - r['bound']) / r['obj'] * 100:.3f} %) "
              f"in {rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
