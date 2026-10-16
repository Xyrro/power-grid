"""rloop (arm B) selection on validation only -> results/recursive/rloop_select.json (read by rec_rloop_eval.py).

    python3 scripts/rec_rloop_select.py

Rules (fixed before any test run):
* whole-schedule recursion: among the validated (model, decoding bias) runs `rloop_val_<model>[_b<bias>].npz`
  (360 instances, K <= 16), the lowest mean gap at K = 16; K_sel = the smallest K whose mean gap is within 0.02 pp of
  that run's best; 24 h: 2 x K_sel.
* guided search: lam per prior = the lowest mean gap after 16 LPs on validation instances 0-119
  (`rloop_val_guided_<prior>_lam<lam>_0-120.npz`); the guiding network = the one with the lowest full-validation
  mean gap at its lam (`rloop_val_guided_net-<model>_lam<lam>.npz`).
* main learned pipeline: whichever of the two has the lower full-validation mean gap.
* m1x thresholds for 1 / 3 / 5 LPs: greedy forward selection on validation (rloop_val_summary.json).
* safety-net threshold theta: 1 % (pre-declared; the test reports 0.5 / 1 / 2 / 4 %).
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rec_rloop_common as C  # noqa: E402


def gap(c, db):
    return (c - db) / c * 100


def main():
    os.chdir(C.ROOT)
    out = {}
    # whole-schedule runs (exclude seeded and guided files)
    best = None
    for f in sorted(glob.glob(os.path.join(C.RES, "rloop_val_*.npz"))):
        name = os.path.basename(f)[10:-4]
        if name.startswith(("baselines", "guided_")) or "start-" in name or name.endswith("_round"):
            continue
        R = dict(np.load(f))
        if R["cost"].shape[1] != 360:
            continue
        curve = [float(gap(R["cost"][:k + 1].min(0), R["db"]).mean()) for k in range(R["cost"].shape[0])]
        if best is None or curve[-1] < best[1][-1]:
            best = (name, curve)
    if best:
        name, curve = best
        k_sel = next(k + 1 for k, g in enumerate(curve) if g <= min(curve) + 0.02)
        m = re.match(r"(.+?)(?:_b([0-9.]+))?$", name)
        out["whole"] = dict(run=name, model=m.group(1), bias=float(m.group(2) or 0.0), K=k_sel, val_gap=curve[k_sel - 1],
                            curve=curve)
    # guided search
    lam = {}
    for f in sorted(glob.glob(os.path.join(C.RES, "rloop_val_guided_*_0-120.npz"))):
        name = os.path.basename(f)[len("rloop_val_guided_"):-len("_0-120.npz")]
        if "seed-" in name or name.startswith("net_once"):
            continue
        prior, lm = name.rsplit("_lam", 1)
        g = float(gap(np.load(f)["cost"][-1], np.load(f)["db"]).mean())
        if prior not in lam or g < lam[prior][1]:
            lam[prior] = (float(lm), g)
    out["lam"] = {k: v[0] for k, v in lam.items()}
    full = {}
    for f in sorted(glob.glob(os.path.join(C.RES, "rloop_val_guided_*.npz"))):
        name = os.path.basename(f)[len("rloop_val_guided_"):-4]
        if re.search(r"_\d+-\d+$", name):
            continue
        R = dict(np.load(f))
        full[name] = [float(gap(R["cost"][k], R["db"]).mean()) for k in range(R["cost"].shape[0])]
    out["guided_full"] = full
    nets = {k: v for k, v in full.items() if k.startswith("net-") and "seed-" not in k}
    if nets:
        gname = min(nets, key=lambda k: nets[k][-1])
        gm = gname[len("net-"):].rsplit("_lam", 1)
        out["guided"] = dict(run=gname, model=gm[0], lam=float(gm[1]), val_gap=nets[gname][-1])
    vs = json.load(open(os.path.join(C.RES, "rloop_val_summary.json")))
    ths = vs.get("m1x_thresholds", {})
    out["m1x_thresholds"] = {k: v["th"] for k, v in ths.items() if k in ("1", "3", "5")}
    # main pipeline and fields read by rec_rloop_eval.py
    main_g = out.get("guided", {}).get("val_gap", np.inf)
    main_w = out.get("whole", {}).get("val_gap", np.inf)
    out["main"] = "rloop_guided" if main_g < main_w else "rloop"
    out["model"] = out["whole"]["model"] if "whole" in out else out["guided"]["model"]
    out["bias"] = out.get("whole", {}).get("bias", 0.0)
    k = out.get("whole", {}).get("K", 8)
    out["K"] = {"uc12": k, "hard": k, "line_out": k, "uc24": 2 * k}
    out["guide_model"] = out.get("guided", {}).get("model", out["model"])
    out["lam_net"] = out.get("guided", {}).get("lam", 0.5)
    out["lam_m1x"] = out["lam"].get("m1x", 0.5)
    out["theta"] = 1.0
    json.dump(out, open(os.path.join(C.RES, "rloop_select.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("guided_full",)}, indent=1))


if __name__ == "__main__":
    main()
