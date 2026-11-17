"""Summary of round 8 exploration records (scripts/agg_probe.py): per method, the mean and max gap to the tightest bound
of the instance's own runs, the share served, the mean time, the geometric-mean speed-up over `ref` in the same
process (with a 95 % bootstrap interval), and for certified pipelines how the answer was proven.

    python3 scripts/agg_probe_report.py results/agg/explore/e1 [--vs acert:s0:0.02]
"""
import argparse
import glob
import json
import os

import numpy as np


def load(folders):
    recs = {}
    for d in folders:
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                recs.setdefault((r.get("set", "val"), r["k"]), []).append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--vs", default="", help="paired comparison of every method against this one")
    a = ap.parse_args()
    recs = load(a.folders)
    rows = {}
    rng = np.random.default_rng(0)
    pair = {}
    for key, rs in recs.items():
        bounds = [r["zlp"] for r in rs]
        for r in rs:
            for x in r["runs"].values():
                for b in ("bound", "cert_bound", "q_bound"):
                    if x.get(b) is not None and np.isfinite(x[b]):
                        bounds.append(x[b])
        db = max(bounds)
        for r in rs:
            ref = r["runs"].get("ref")
            for name, x in r["runs"].items():
                gap = (x["cost"] - db) / x["cost"]
                sp = ref["time"] / x["time"] if ref else np.nan
                rg = (x["red_cost"] - db) / x["red_cost"] if x.get("red_cost") is not None else np.nan
                rows.setdefault(name, []).append((gap, x["served"], x["time"], sp, x.get("cert"), rg,
                                                  x.get("n_fixed", np.nan)))
                pair.setdefault(key, {})[name] = (gap, x["time"])
    print(f"{len(recs)} instances")
    print(f"{'method':28s} {'n':>4s} {'gap%':>7s} {'max%':>7s} {'served':>7s} {'time s':>7s} {'speed-up [95% CI]':>22s} "
          f"{'cert lp/milp/sw':>16s} {'red gap%':>9s} {'fixed':>6s}")
    for name in sorted(rows, key=lambda n: -np.exp(np.nanmean(np.log([r[3] for r in rows[n]])))):
        R = rows[name]
        g = np.array([r[0] for r in R]) * 100
        sp = np.array([r[3] for r in R])
        ls = np.log(sp[np.isfinite(sp)])
        ci = ""
        if len(ls) > 2:
            bs = [np.exp(rng.choice(ls, len(ls)).mean()) for _ in range(1000)]
            ci = f"{np.exp(ls.mean()):5.1f}x [{np.percentile(bs, 2.5):4.1f}, {np.percentile(bs, 97.5):4.1f}]"
        certs = [r[4] for r in R]
        cs = f"{certs.count('lp')}/{certs.count('milp')}/{certs.count('sandwich')}" if any(certs) else ""
        rg = np.array([r[5] for r in R]) * 100          # the reduced schedule alone (no certificate)
        red = f"{np.nanmean(rg):.3f}" if np.isfinite(rg).any() else ""
        print(f"{name:28s} {len(R):4d} {g.mean():7.3f} {g.max():7.2f} {np.mean([r[1] for r in R]) * 100:6.1f}% "
              f"{np.mean([r[2] for r in R]):7.2f} {ci:>22s} {cs:>16s} {red:>9s} {np.nanmean([r[6] for r in R]):6.0f}")
    if a.vs:
        print(f"\npaired against {a.vs}: gap difference (pp) [95% CI], speed-up of the method over it [95% CI]")
        for name in sorted(rows):
            if name == a.vs:
                continue
            d, s = [], []
            for key, mm in pair.items():
                if name in mm and a.vs in mm:
                    d.append((mm[name][0] - mm[a.vs][0]) * 100)
                    s.append(np.log(mm[a.vs][1] / mm[name][1]))
            if len(d) < 3:
                continue
            d, s = np.array(d), np.array(s)
            bd = [rng.choice(d, len(d)).mean() for _ in range(1000)]
            bsx = [np.exp(rng.choice(s, len(s)).mean()) for _ in range(1000)]
            print(f"  {name:28s} n={len(d):3d}  {d.mean():+.3f} [{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]  "
                  f"{np.exp(s.mean()):.2f}x [{np.percentile(bsx, 2.5):.2f}, {np.percentile(bsx, 97.5):.2f}]")


if __name__ == "__main__":
    main()
