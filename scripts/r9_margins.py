"""Round 9 (docs/methods/agg.md): the probing-margin feature (otsl.probe.margin_feature) of every training and validation
instance of the temporal GNN (scripts/uc_m1x_train.py), from a fresh LP relaxation (its bus and reserve prices), saved
next to each data file as <file>.margin.npz (key margin [n, T, G]).

    python3 scripts/r9_margins.py --files data/generated/uc12/train.npz data/generated/uc12_m1x/extra_0.npz ...
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.probe import margin_feature  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    sysm = load_rts_gmlc()
    m = UCModel(sysm, T=12)
    for f in a.files:
        out = f.replace(".npz", ".margin.npz")
        if os.path.exists(out):
            print(f"{out} exists")
            continue
        d = np.load(f)
        n = len(d["load"])
        M = np.zeros((n, 12, sysm.G), np.float32)
        t0 = time.time()
        for i in range(n):
            sc = UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])
            rel = m.solve_dispatch(sc, None, relax=True)
            M[i] = margin_feature(sysm, sc.u0, rel.lmp, rel.mu, rel.obj)
        np.savez_compressed(out + ".tmp.npz", margin=M)
        os.replace(out + ".tmp.npz", out)
        print(f"{out}: {n} instances in {time.time() - t0:.0f}s", flush=True)
