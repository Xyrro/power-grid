"""Seed-matched comparison of the loop-native attention residual variants of arm B against the plain model.

    python3 scripts/rec_loopres_compare.py --prefix cs            # CPU-scale screen (scripts/pace/pgu_rloop_screen.sbatch)
    python3 scripts/rec_loopres_compare.py --prefix g3 --ref w2_rloop_rl   # GPU scale (scripts/pace/plans/loopres_gpu.json)

Reads results/recursive/rloop_val_<prefix>_<variant>_s<seed>_rl.npz (per-step cost / shed / short on the 360
validation instances; gap to the stored MILP dual bound). Metric: best-of-K mean gap (keep the cheapest of K steps).
Equal seeds share initial weights and data order, so each seed is a matched pair. Rule (fixed before the runs): a
variant beats the plain model only if the seed-averaged per-instance difference has a 95 % bootstrap CI above zero
AND the variant is better in every seed. Also reported: served, the difference on instances both serve, the drift of
the latest schedule from the best one so far, and a paired comparison with --ref models (one run each).
Output: results/recursive/loopres_compare_<prefix>.json.
"""
import argparse
import json
import os

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "recursive")
BOOT = 10000


def load(tag, K):
    d = np.load(os.path.join(RES, f"rloop_val_{tag}.npz"))
    if d["cost"].shape[0] < K:
        raise ValueError(f"{tag} has {d['cost'].shape[0]} steps, fewer than K = {K}")
    c = d["cost"][:K]
    best = np.minimum.accumulate(c, 0)
    gap = (best - d["db"]) / best * 100
    srv = (d["shed"][:K] < 1e-6) & (d["short"][:K] < 1e-6)
    sb = srv[np.argmin(c, 0), np.arange(c.shape[1])]
    lat = (c[-1] - d["db"]) / c[-1] * 100
    drift = float((best[:-1] < c[1:] * (1 - 1e-4)).mean() * 100)
    return dict(gap=gap[-1], served=sb, latest=lat, drift=drift)


def ci(x, rng):
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=["lp", "paper"])
    ap.add_argument("--base", default="plain")
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--ref", nargs="*", default=[])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tag = lambda v, s: f"{a.prefix}_{v}_s{s}_rl"
    have = {v: [s for s in a.seeds if os.path.exists(os.path.join(RES, f"rloop_val_{tag(v, s)}.npz"))]
            for v in [a.base] + a.variants}
    out = dict(prefix=a.prefix, K=a.K, seeds=a.seeds, models={}, compare={})
    print(f"best-of-{a.K} mean gap on 360 validation instances (served at the best schedule; latest-step gap; drift)")
    for v in [a.base] + a.variants:
        for s in have[v]:
            r = load(tag(v, s), a.K)
            out["models"][tag(v, s)] = dict(gap=float(r["gap"].mean()), served=float(r["served"].mean() * 100),
                                            latest=float(r["latest"].mean()), drift=r["drift"])
            print(f"  {tag(v, s):22s} {r['gap'].mean():6.3f} %  served {r['served'].mean() * 100:5.1f} %  "
                  f"latest {r['latest'].mean():6.3f} %  drift {r['drift']:4.1f} %")
    for v in a.variants:
        seeds = [s for s in have[v] if s in have[a.base]]
        if not seeds:
            continue
        per, diffs, both = {}, [], []
        for s in seeds:
            b, x = load(tag(a.base, s), a.K), load(tag(v, s), a.K)
            d = b["gap"] - x["gap"]                                   # > 0: the variant is better
            m = b["served"] & x["served"]
            per[s] = dict(diff=float(d.mean()), ci=ci(d, rng), both=float(d[m].mean()), both_ci=ci(d[m], rng),
                          n_both=int(m.sum()), served=[float(x["served"].mean() * 100), float(b["served"].mean() * 100)])
            diffs.append(d)
            both.append(np.where(m, d, np.nan))
        pooled = np.mean(diffs, 0)
        pb = np.nanmean(np.stack(both), 0)
        pb = pb[np.isfinite(pb)]
        res = dict(seeds=seeds, per_seed=per, pooled=float(pooled.mean()), pooled_ci=ci(pooled, rng),
                   pooled_both=float(pb.mean()), pooled_both_ci=ci(pb, rng),
                   wins=sum(per[s]["diff"] > 0 for s in seeds))
        res["beats_base"] = bool(res["pooled_ci"][0] > 0 and res["wins"] == len(seeds) and len(seeds) == len(a.seeds))
        out["compare"][f"{v} vs {a.base}"] = res
        print(f"\n{v} vs {a.base} (positive = {v} better), seeds {seeds}:")
        for s in seeds:
            p = per[s]
            print(f"  seed {s}: {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  served {p['served'][0]:.1f} vs "
                  f"{p['served'][1]:.1f}  | both served (n {p['n_both']}): {p['both']:+.3f} "
                  f"[{p['both_ci'][0]:+.3f}, {p['both_ci'][1]:+.3f}]")
        print(f"  seed-averaged: {res['pooled']:+.3f} [{res['pooled_ci'][0]:+.3f}, {res['pooled_ci'][1]:+.3f}], "
              f"better in {res['wins']}/{len(seeds)} seeds; both served {res['pooled_both']:+.3f} "
              f"[{res['pooled_both_ci'][0]:+.3f}, {res['pooled_both_ci'][1]:+.3f}]  ->  "
              f"{'BEATS' if res['beats_base'] else 'does not beat'} {a.base} by the pre-set rule")
    for ref in a.ref:
        try:
            r = load(ref, a.K)
        except (FileNotFoundError, ValueError) as err:
            print(f"\nreference {ref} skipped: {err}")
            continue
        print(f"\nagainst {ref} ({r['gap'].mean():.3f} %, served {r['served'].mean() * 100:.1f} %; one run):")
        for v in [a.base] + a.variants:
            if not have[v]:
                continue
            g = np.mean([load(tag(v, s), a.K)["gap"] for s in have[v]], 0)
            d = r["gap"] - g
            c = ci(d, rng)
            out["compare"][f"{v} (seed mean) vs {ref}"] = dict(diff=float(d.mean()), ci=c, seeds=have[v])
            print(f"  {v:6s} seed mean {g.mean():.3f} %: {d.mean():+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]  (positive = {v} better)")
    with open(os.path.join(RES, f"loopres_compare_{a.prefix}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
