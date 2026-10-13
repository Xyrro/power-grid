"""hard benchmark: the same validation scenarios at the original line ratings (paired check of the relaxation gap).

    taskset -c N python scripts/uc_hard_paired.py [--split val]

For every instance of the split (data/generated/hard/<split>_<kk>.npz: identical load, renewables, reserve and initial
status) the LP relaxation and the full MILP (HiGHS, one thread, 0.1 %, 600 s) with the original continuous ratings
(line_scale 1.0). Records (results/hard/hard_paired_orig_<split>.jsonl; resume skips finished instances): LP gap to the
MILP objective and to its dual bound, integral share, fractional and wrong integral values (otsl.hard.relax_stats),
MILP time, served share.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.b3 import solve_milp_hs  # noqa: E402
from otsl.hard import T_H, load_split, relax_stats, sc_of  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    out = f"results/hard/hard_paired_orig_{a.split}.jsonl"
    done = {json.loads(x)["i"] for x in open(out)} if os.path.exists(out) else set()
    s1 = load_rts_gmlc(line_scale=1.0)
    m1 = UCModel(s1, T=T_H, network=True)
    d = load_split(a.split)
    for i in range(len(d["obj"])):
        if i in done:
            continue
        sc = sc_of(d, i)
        t0 = time.time()
        rel = m1.solve_dispatch(sc, None, relax=True)
        sol = solve_milp_hs(m1, sc, time_limit=600.0, mip_gap=1e-3, threads=1, trace_every=0.0)
        rec = dict(i=i, day=int(d["day"][i]), start=int(d["start"][i]), c_rel=rel.obj, t_rel=rel.time, milp_time=sol["time"],
                   status=sol["status"], obj=sol["obj"], bound=sol["bound"], mip_gap=sol["gap"],
                   shed=sol.get("shed"), short=sol.get("short"),
                   lp_gap_obj=float((sol["obj"] - rel.obj) / sol["obj"] * 100), lp_gap_db=float((sol["bound"] - rel.obj) / sol["bound"] * 100),
                   derated=dict(c_rel=float(d["c_rel"][i]), obj=float(d["obj"][i]), bound=float(d["bound"][i]),
                                time=float(d["time"][i])))
        rec.update(relax_stats(rel.u, sol["u"], s1, sc.u0))
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"{i} day {rec['day']} orig: LP gap {rec['lp_gap_db']:.3f}% (derated "
              f"{(d['bound'][i] - d['c_rel'][i]) / d['bound'][i] * 100:.3f}%) MILP {sol['time']:.1f}s "
              f"(derated {d['time'][i]:.1f}s) {time.time() - t0:.0f}s", flush=True)
