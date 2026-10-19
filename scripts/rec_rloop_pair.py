"""Paired comparison of two refinement-loop runs on the shared validation set.

Both arms roll out over the same 360 uc12 validation instances and store a (K, n) cost matrix, so the
difference between them can be taken per instance. That matters here: the instance-to-instance spread of
the cost gap is far larger than the gap between the two arms, and an unpaired comparison of two means
drowns the effect. §3 U7 of the research log records the same lesson for the proxy metrics.

    python3 scripts/rec_rloop_pair.py --a w2_rloop_rl --b w1_rloop_nofb_rl

Reports, at each reported k, the best-of-k gap of each arm, the paired difference with a bootstrap CI,
and the share of instances the arm serves without shedding. Also reports the anytime behaviour of the
halting head, which is what a dispatcher would actually run: stop at the step the model itself picks
rather than at a fixed k.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.rec_rloop_common as C  # noqa: E402

KS = (1, 2, 4, 8, 12, 16, 20, 24)
BOOT = 10000


def load(tag):
    d = dict(np.load(os.path.join(C.RES, f"rloop_val_{tag}.npz")))
    cost, db = d["cost"], d["db"]
    best = np.minimum.accumulate(cost, 0)                   # best-of-k cost per instance
    gap = (best - db) / best * 100
    K, n = cost.shape
    cf = np.where(np.isfinite(cost), cost, np.inf)
    arg = np.argmin(cf, 0)                                  # step the overall best-of-K came from
    srv = (d["shed"] < 1e-6) & (d["short"] < 1e-6)
    # served at the best-of-k step, so the number describes the schedule the gap was measured on
    srv_best = np.stack([srv[np.argmin(cf[:k + 1], 0), np.arange(n)] for k in range(K)])
    return dict(gap=gap, db=db, cost=cost, halt=d["halt"], srv=srv, srv_best=srv_best, arg=arg)


def boot_ci(d, rng, lo=2.5, hi=97.5):
    """percentile CI of the mean of a paired difference, resampling instances"""
    n = len(d)
    idx = rng.integers(0, n, size=(BOOT, n))
    m = d[idx].mean(1)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def halt_step(h, thr=0.5):
    """first step whose halting probability clears thr, else the last step"""
    K, n = h.shape
    over = h >= thr
    first = np.where(over.any(0), over.argmax(0), K - 1)
    return first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="tag under test")
    p.add_argument("--b", required=True, help="control tag")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    A, B = load(a.a), load(a.b)
    assert np.allclose(A["db"], B["db"]), "the two runs were evaluated on different instances"
    n = A["gap"].shape[1]
    rng = np.random.default_rng(a.seed)
    rows, out = [], {"a": a.a, "b": a.b, "n": n, "boot": BOOT, "k": {}}

    for k in KS:
        i = k - 1
        ga, gb = A["gap"][i], B["gap"][i]
        d = gb - ga                                   # positive: the arm under test is better
        lo, hi = boot_ci(d, rng)
        wins = float((ga < gb - 1e-9).mean() * 100)
        # A mean gap over all instances mixes two different effects: how cheap a served schedule is, and
        # whether a served schedule was found at all (an unserved one carries VOLL at $10k/MWh and so sits
        # far out in the tail). Restricting to the instances BOTH arms serve isolates the first effect.
        both = A["srv_best"][i] & B["srv_best"][i]
        db_ = d[both]
        blo, bhi = boot_ci(db_, rng) if both.sum() > 2 else (float("nan"),) * 2
        rec = dict(a_mean=float(ga.mean()), a_med=float(np.median(ga)),
                   b_mean=float(gb.mean()), b_med=float(np.median(gb)),
                   diff_mean=float(d.mean()), ci=[lo, hi], win_pct=wins,
                   tie_pct=float((np.abs(d) <= 1e-9).mean() * 100),
                   loss_pct=float((ga > gb + 1e-9).mean() * 100),
                   a_served=float(A["srv_best"][i].mean() * 100),
                   b_served=float(B["srv_best"][i].mean() * 100),
                   both_served_n=int(both.sum()),
                   both_a_mean=float(ga[both].mean()), both_b_mean=float(gb[both].mean()),
                   both_diff=float(db_.mean()), both_ci=[blo, bhi])
        out["k"][k] = rec
        rows.append((k, rec))

    # anytime behaviour: stop where each model's own halting head says to stop
    for name, D in ((a.a, A), (a.b, B)):
        hs = halt_step(D["halt"])
        g = D["gap"][hs, np.arange(n)]
        out.setdefault("halt", {})[name] = dict(
            mean=float(g.mean()), med=float(np.median(g)), steps=float(hs.mean() + 1),
            served=float(D["srv_best"][hs, np.arange(n)].mean() * 100))

    w = max(len(a.a), len(a.b), 7)
    print(f"paired on {n} instances; CI = {BOOT} bootstrap resamples of the per-instance difference\n")
    print(f"{'k':>3}  {a.a:>{w}s}  {a.b:>{w}s}  {'diff':>7s}  {'95% CI':>16s}  {'wins':>6s}  served")
    for k, r in rows:
        ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        print(f"{k:3d}  {r['a_mean']:{w}.3f}  {r['b_mean']:{w}.3f}  {r['diff_mean']:+7.3f}  {ci:>16s}  "
              f"{r['win_pct']:5.1f}%  {r['a_served']:5.1f}/{r['b_served']:.1f}")
    print(f"\n{'median gap':>12s}  " + "  ".join(f"k={k}: {r['a_med']:.3f} vs {r['b_med']:.3f}"
                                                 for k, r in rows if k in (1, 8, 16, 24)))
    print(f"\nwin/tie/loss per instance:  " + "  ".join(
        f"k={k}: {r['win_pct']:.0f}/{r['tie_pct']:.0f}/{r['loss_pct']:.0f}" for k, r in rows if k in (1, 8, 16, 24)))
    print("\nrestricted to instances BOTH arms serve (isolates cost quality from feasibility):")
    print(f"{'k':>3}  {'n':>4}  {a.a:>{w}s}  {a.b:>{w}s}  {'diff':>7s}  {'95% CI':>16s}")
    for k, r in rows:
        ci = f"[{r['both_ci'][0]:+.3f},{r['both_ci'][1]:+.3f}]"
        print(f"{k:3d}  {r['both_served_n']:4d}  {r['both_a_mean']:{w}.3f}  {r['both_b_mean']:{w}.3f}  "
              f"{r['both_diff']:+7.3f}  {ci:>16s}")
    print("\nhalting head (the model picks its own stopping step):")
    for name, h in out["halt"].items():
        print(f"  {name:>{w}s}  mean {h['mean']:.3f}  median {h['med']:.3f}  "
              f"served {h['served']:.1f}%  mean steps {h['steps']:.1f}")

    path = a.out or os.path.join(C.RES, f"rloop_pair_{a.a}_vs_{a.b}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
