"""Paired comparison of arm A models on the shared validation set, one pipeline at a time.

Every arm A validation run stores, per pipeline, a (360, 3+) array of (cost, shed, short[, extra LPs]) for the same
360 uc12 validation instances (data/generated/recursive/val/<tag>.npz, written by scripts/rec_trm_val.py). Comparing
models under *different* pipelines, or by unpaired means, can reverse a conclusion: the instance-to-instance spread
of the gap is far larger than the differences between models. This script holds the pipeline fixed and reports the
paired difference with a bootstrap CI, plus the same comparison restricted to instances both models serve (so that
a feasibility effect is not mistaken for a cost effect).

    python3 scripts/rec_trm_pair.py --a w1_trm --b w1b_oneshot_best w1_oneshot w1b_deep_best w1_deep \
        --pipelines mix5_rep__rep mix5_plain__plain

Positive differences mean model A is better (lower gap).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl import trm  # noqa: E402

BOOT = 10000
VAL_DIR = os.path.join("data", "generated", "recursive", "val")


def load(tag, pipe, ref):
    d = np.load(os.path.join(VAL_DIR, f"{tag}.npz"))
    if pipe not in d.files:
        return None
    a = d[pipe]
    gap = (a[:, 0] - ref) / ref * 100
    srv = (a[:, 1] < 1e-6) & (a[:, 2] < 1e-6)
    return gap, srv


def ci(x, rng):
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--pipelines", nargs="+", default=["mix5_rep__rep", "mix5_plain__plain"])
    p.add_argument("--n", type=int, default=360)
    p.add_argument("--out", default="")
    a = p.parse_args()
    ref = trm.subset(trm.val_data(), np.arange(a.n))["obj"]
    rng = np.random.default_rng(0)
    res = {}
    for pipe in a.pipelines:
        A = load(a.a, pipe, ref)
        if A is None:
            print(f"{pipe}: missing for {a.a}")
            continue
        ga, sa = A
        print(f"\n== {pipe}   A = {a.a}: mean {ga.mean():.3f}  median {np.median(ga):.3f}  served {sa.mean() * 100:.1f}%")
        print(f"{'B':24s} {'B mean':>7s} {'B med':>6s} {'B srv':>6s} | {'diff (B-A)':>10s} {'95% CI':>18s} "
              f"| {'n both':>6s} {'diff both-served':>16s} {'95% CI':>18s} | {'A<B / tie / A>B':>16s}")
        for tb in a.b:
            B = load(tb, pipe, ref)
            if B is None:
                print(f"{tb:24s} missing")
                continue
            gb, sb = B
            d = gb - ga
            lo, hi = ci(d, rng)
            both = sa & sb
            db = d[both]
            lo2, hi2 = ci(db, rng) if both.sum() > 1 else (float("nan"), float("nan"))
            win = int((d > 1e-9).sum()); loss = int((d < -1e-9).sum()); tie = len(d) - win - loss
            print(f"{tb:24s} {gb.mean():7.3f} {np.median(gb):6.3f} {sb.mean() * 100:5.1f}% | {d.mean():+10.3f} "
                  f"[{lo:+7.3f}, {hi:+7.3f}] | {int(both.sum()):6d} {db.mean():+16.3f} [{lo2:+7.3f}, {hi2:+7.3f}] "
                  f"| {win:4d} / {tie:3d} / {loss:4d}")
            res.setdefault(pipe, {})[tb] = dict(
                a_mean=float(ga.mean()), a_median=float(np.median(ga)), a_served=float(sa.mean() * 100),
                b_mean=float(gb.mean()), b_median=float(np.median(gb)), b_served=float(sb.mean() * 100),
                diff_mean=float(d.mean()), diff_ci=[lo, hi], n_both=int(both.sum()),
                diff_both_mean=float(db.mean()), diff_both_ci=[lo2, hi2], a_better=win, ties=tie, b_better=loss)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(a=a.a, b=a.b, boot=BOOT, n=a.n, results=res), f, indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
