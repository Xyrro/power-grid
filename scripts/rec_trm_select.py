"""Fix the test plan of the TRM study from validation only (written before the test set is read).

    python3 scripts/rec_trm_select.py --final main main_rl --controls oneshot

* final TRM pipeline: among --final tags, the lowest validation mean gap of the 1-LP + LP-repair pipeline (selected
  decoder and threshold at the training step count); every --final and --controls tag is evaluated on test with its
  own validation-selected decoders, thresholds and screening thresholds (plain and repaired pipelines);
* m1x baseline: its validation-selected decoder / threshold (1 LP) and 5 screening thresholds, plain and repaired;
* safety net g* per TRM tag: the smallest grid value of the bound gap (C - z_LP) / C at which the fallback of the
  repaired 1-LP pipeline (still shed / reserve shortfall, or bound gap >= g*) fires on at most FIRE_MAX of the
  validation instances.
Output: results/recursive/trm_select.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl import trm  # noqa: E402

RES = os.path.join("results", "recursive")
OUT = os.path.join("data", "generated", "recursive")
G_GRID = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1)
FIRE_MAX = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", nargs="+", required=True)
    ap.add_argument("--controls", nargs="*", default=[])
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    va = trm.val_data()
    plan = dict(trm={}, gstar={}, notes={})
    one = {}

    def pick(s):
        return dict(decoder=s["decoder"], th=s["th"], screen_th=s["screen_th"], val_one_lp=s["one_lp"],
                    val_screen=s["screen_th_metrics"])

    for tag in a.final + a.controls:
        v = json.load(open(os.path.join(RES, f"trm_val_{tag}.json")))
        s = v["selection"]
        kd = s["rep"]["k_def"]
        plan["trm"][tag] = dict(ckpt=tag, k_def=kd, plain=pick(s["plain"]), rep=pick(s["rep"]))
        one[tag] = s["rep"]["one_lp"]["gap_mean"]
        st = np.load(os.path.join(OUT, "val", f"{tag}.npz"))
        c = st[f"{s['rep']['decoder']}__{s['rep']['th']}__k{kd}__rep"]
        g = (c[:, 0] - va["c_rel"]) / c[:, 0]
        bad = (c[:, 1] > 1e-6) | (c[:, 2] > 1e-6)
        rates = {gg: float(np.mean(bad | (g >= gg))) for gg in G_GRID}
        ok = [gg for gg in G_GRID if rates[gg] <= FIRE_MAX]
        plan["gstar"][tag] = ok[0] if ok else None
        plan["notes"][f"fire_rates_{tag}"] = rates
    plan["final"] = min(a.final, key=lambda t: one[t])
    plan["notes"]["final_rule"] = ("lowest validation mean gap of the 1-LP + LP-repair pipeline among " +
                                   ", ".join(a.final))
    m = json.load(open(os.path.join(RES, "trm_val_m1x.json")))["selection"]
    plan["m1x"] = dict(plain=pick(m["plain"]), rep=pick(m["rep"]))
    with open(os.path.join(RES, "trm_select.json"), "w") as f:
        json.dump(plan, f, indent=1, default=float)
    print(json.dumps(plan, indent=1, default=float))


if __name__ == "__main__":
    main()
