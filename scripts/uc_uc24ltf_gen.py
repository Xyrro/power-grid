"""uc24ltf, step 1: give the 24-hour benchmark (uc24) what Learning to Fix needs.

    python3 scripts/uc_uc24ltf_gen.py --what val,train --n_val 40 --n_train 120

* val   : full MILPs (highspy, 0.1 % gap, 300 s limit, 1 thread; the uc24 test settings) for the validation-day
          instances of make_jobs(SPLITS["val"], n_val, seed 202). The first 30 jobs are exactly the 30 label-free
          instances of data/generated/uc24/val.npz (same day / seed draws; checked below), the rest are new
          validation-day instances. -> data/generated/uc24/uc24ltf_val.npz (+ .json)
* train : labelled training schedules for the kNN: full MILPs at a looser 0.5 % gap and a 60 s limit for the
          first n_train instances of data/generated/uc24/train.npz (same scenarios: day / start / seed re-used;
          the train order is already random). -> data/generated/uc24/uc24ltf_train_lab.npz (+ .json)

One worker process (spawn), one HiGHS thread. Partial results are re-saved every 5 instances
(<name>_partial.npz) so the run can be stopped at any time.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.b3 import generate_b3, load_b3, make_jobs, save_b3  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_gen import SPLITS  # noqa: E402

ROOT = os.path.join("data", "generated", "uc24")
BASE = {"T": 24, "start": 0, "network": True, "mode": "milp"}


def summary(data, cfg, wall, workers):
    ok = np.isfinite(data["obj"])
    return {"cfg": cfg, "n": int(len(data["obj"])), "wall_s": wall, "workers": workers,
            "milp_time_mean": float(data["time"].mean()), "milp_time_median": float(np.median(data["time"])),
            "milp_time_sum_s": float(data["time"].sum()), "milp_cpu_sum_s": float(np.nansum(data["cpu"])),
            "milp_opt_frac": float(data["opt"].mean()), "milp_gap_mean_%": float(data["gap"][ok].mean() * 100),
            "milp_gap_max_%": float(data["gap"][ok].max() * 100),
            "milp_served_%": float(((data["short"] < 1e-6) & (data["shed"] < 1e-6))[ok].mean() * 100),
            "relax_gap_%": float(((data["obj"] - data["c_rel"]) / data["obj"])[ok].mean() * 100),
            "lp_check_max_rel_err": float(np.nanmax(np.abs(data["obj_lp"] - data["obj"]) / data["obj"])),
            "loadavg_mean": float(data["loadavg"].mean())}


def run(name, cfg, jobs, workers):
    out = os.path.join(ROOT, f"{name}.npz")
    t0 = time.time()
    print(f"=== {name}: {len(jobs)} instances, cfg {cfg}", flush=True)
    data = generate_b3(cfg, jobs, workers=workers, log=lambda s: print(s, flush=True),
                       partial_path=out.replace(".npz", "_partial.npz"))
    save_b3(out, data)
    summ = summary(data, cfg, time.time() - t0, workers)
    json.dump(summ, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(summ), flush=True)
    if os.path.exists(out.replace(".npz", "_partial.npz")):
        os.remove(out.replace(".npz", "_partial.npz"))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="val,train")
    ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--train_gap", type=float, default=5e-3)
    ap.add_argument("--train_limit", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for what in a.what.split(","):
        if what == "val":
            jobs = make_jobs(SPLITS["val"], a.n_val, 202, T=24, start=0)
            old = load_b3(os.path.join(ROOT, "val.npz"))
            k = min(len(old["day"]), len(jobs))
            same = all(int(old["day"][i]) == jobs[i][0] and int(old["seed"][i]) == jobs[i][2] for i in range(k))
            print(f"first {k} val jobs identical to val.npz: {same}", flush=True)
            assert same
            run("uc24ltf_val", dict(BASE, time_limit=300.0, mip_gap=1e-3), jobs, a.workers)
        elif what == "train":
            tr = load_b3(os.path.join(ROOT, "train.npz"))
            n = min(a.n_train, len(tr["day"]))
            jobs = [(int(tr["day"][i]), int(tr["start"][i]), int(tr["seed"][i])) for i in range(n)]
            data = run("uc24ltf_train_lab", dict(BASE, time_limit=a.train_limit, mip_gap=a.train_gap), jobs, a.workers)
            err = float(np.abs(data["load"] - tr["load"][:n]).max())
            print(f"scenario check vs train.npz (max |load diff|): {err:.3e}", flush=True)
    print("done", flush=True)
