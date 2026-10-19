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
    ap.add_argument("--val_dirs", nargs="+", default=[RES],
                    help="where to look for trm_val_<tag>.json, first match wins (e.g. a round's own folder first)")
    ap.add_argument("--out", default=os.path.join(RES, "trm_select.json"))
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    va = trm.val_data()
    plan = dict(trm={}, gstar={}, notes={})
    one = {}

    def val_json(tag):
        for d_ in a.val_dirs:
            p_ = os.path.join(d_, f"trm_val_{tag}.json")
            if os.path.exists(p_):
                plan["notes"][f"source_{tag}"] = p_
                return json.load(open(p_))
        raise SystemExit(f"no trm_val_{tag}.json in {a.val_dirs}")

    def pick(s):
        return dict(decoder=s["decoder"], th=s["th"], screen_dec=s.get("screen_dec", s["decoder"]),
                    screen_th=s.get("screen_th", []), mix_th=s.get("mix_th"), val_one_lp=s["one_lp"],
                    val_screen=s.get("screen_th_metrics"), val_mix=s.get("mix_metrics"))

    for tag in a.final + a.controls:
        v = val_json(tag)
        s = v["selection"]
        if "checkpoint" in v:
            plan["notes"][f"checkpoint_{tag}"] = v["checkpoint"]
        kd = s["rep"]["k_def"]
        plan["trm"][tag] = dict(ckpt=tag, k_def=kd, plain=pick(s["plain"]), rep=pick(s["rep"]))
        hp = os.path.join(RES, f"trm_halt_{tag}.json")
        if os.path.exists(hp):          # post-hoc halting head fitted on training instances
            h = json.load(open(hp))
            plan["trm"][tag].update(halt_w=h["w"], halt_b=h["b"], halt_K=h["K"])
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
    # headline pipeline of the final model: the lowest validation mean gap among its repaired pipelines
    v = val_json(plan["final"])["selection"]
    kd = v["rep"]["k_def"]
    cand = {"_1r": v["rep"]["one_lp"], "_t5r": v["rep"].get("screen_th_metrics"), "_mix5r": v["rep"].get("mix_metrics"),
            "_s5r": v.get(f"screen_steps_k{kd}_rep")}
    cand = {k: c["gap_mean"] for k, c in cand.items() if c}
    plan["headline"] = plan["final"] + min(cand, key=cand.get)
    plan["notes"]["headline_rule"] = "lowest validation mean gap among the final model's repaired pipelines: " + \
        json.dumps({k: round(x, 3) for k, x in cand.items()})
    m = val_json("m1x")["selection"]
    plan["m1x"] = dict(plain=pick(m["plain"]), rep=pick(m["rep"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1, default=float)
    print(json.dumps(plan, indent=1, default=float))


if __name__ == "__main__":
    main()
