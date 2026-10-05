"""B3 data: 24-hour network-constrained UC on RTS-GMLC (config "uc24").

    python scripts/uc_b3_gen.py --splits train,val,test --n_train 300 --n_val 30 --n_test 40

* train: NO MILP. Scenario + LP relaxation (commitment, prices, flows) + repaired relaxation labels and their
  dispatch-LP costs; u = -1, obj = NaN.
* val / test: the same plus the full MILP (highspy, 1 thread per solve, 2 solves in parallel) with its
  incumbent trajectory (for time-to-quality), final MIP gap, dual bound and optimality flag.

Days follow scripts/uc_gen.py SPLITS (test = day % 5 == 0, val = day % 5 == 2, train = the rest); start hour 0.
Output: data/generated/uc24/{split}.npz (+ .json summary). Partial results are re-saved every 5 instances.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.b3 import generate_b3, make_jobs, save_b3  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc_gen import SPLITS  # noqa: E402

B3_CONFIGS = {
    # chosen from the probe (results/uc24/b3_probe_T24.json; 4 instances, 900 s limit): two instances reached
    # 0.1 % in ~80 s, two hit 900 s at 0.14 / 0.23 % (0.16 / 0.31 % at 300 s). See docs/methods/b3.md
    "uc24": {"T": 24, "start": 0, "network": True, "time_limit": 300.0, "mip_gap": 1e-3},
    "uc24_dbg": {"T": 24, "start": 0, "network": True, "time_limit": 20.0, "mip_gap": 1e-3},   # code debugging only
}
SEEDS = {"train": 101, "val": 202, "test": 303}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="uc24")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_mode", default="lp", choices=["lp", "milp"],
                    help="lp: val without full MILPs (label-free selection); milp: with full MILPs")
    ap.add_argument("--out_name", default="", help="file name suffix, e.g. _milp for a val set with MILPs")
    a = ap.parse_args()
    cfg = dict(B3_CONFIGS[a.cfg])
    if a.limit is not None:
        cfg["time_limit"] = a.limit
    if a.gap is not None:
        cfg["mip_gap"] = a.gap
    root = os.path.join("data", "generated", a.cfg)
    os.makedirs(root, exist_ok=True)
    for split in a.splits.split(","):
        n = {"train": a.n_train, "val": a.n_val, "test": a.n_test}[split]
        c = dict(cfg, mode={"train": "lp", "val": a.val_mode, "test": "milp"}[split])
        jobs = make_jobs(SPLITS[split], n, SEEDS[split], T=c["T"], start=c["start"])
        t0 = time.time()
        print(f"=== {split}: {n} instances, mode {c['mode']}, cfg {c}", flush=True)
        out = os.path.join(root, f"{split}{a.out_name}.npz")
        data = generate_b3(c, jobs, workers=a.workers, log=lambda s: print(s, flush=True),
                           partial_path=out.replace(".npz", "_partial.npz"))
        save_b3(out, data)
        summ = {"cfg": c, "n": n, "wall_s": time.time() - t0, "workers": a.workers,
                "label_s_mean": float(data["t_label"].mean()), "relax_s_mean": float(data["t_rel"].mean()),
                "c_lf_over_relax_%": float(((data["c_lf"] - data["c_rel"]) / data["c_rel"]).mean() * 100),
                "lf_label_served_%": float(((data["short_lf"] < 1e-6) & (data["shed_lf"] < 1e-6)).mean() * 100),
                "blk_label_served_%": float(((data["short_blk"] < 1e-6) & (data["shed_blk"] < 1e-6)).mean() * 100)}
        if c["mode"] == "milp":
            ok = np.isfinite(data["obj"])
            gap_ = lambda x: ((x - data["obj"]) / data["obj"])[ok] * 100
            summ.update({"milp_time_mean": float(data["time"].mean()), "milp_time_median": float(np.median(data["time"])),
                         "milp_opt_frac": float(data["opt"].mean()), "milp_gap_mean_%": float(data["gap"][ok].mean() * 100),
                         "milp_gap_max_%": float(data["gap"][ok].max() * 100),
                         "relax_gap_%": float((-gap_(data["c_rel"])).mean()),
                         "lf_label_gap_mean_%": float(gap_(data["c_lf"]).mean()),
                         "lf_label_gap_median_%": float(np.median(gap_(data["c_lf"]))),
                         "blk_label_gap_median_%": float(np.median(gap_(data["c_blk"]))),
                         "milp_served_%": float(((data["short"] < 1e-6) & (data["shed"] < 1e-6))[ok].mean() * 100),
                         "lp_check_max_rel_err": float(np.nanmax(np.abs(data["obj_lp"] - data["obj"]) / data["obj"])),
                         "loadavg_mean": float(data["loadavg"].mean())})
        json.dump(summ, open(out.replace(".npz", ".json"), "w"), indent=1)
        print(json.dumps(summ), flush=True)
        if os.path.exists(out.replace(".npz", "_partial.npz")):
            os.remove(out.replace(".npz", "_partial.npz"))
    print("done", flush=True)
