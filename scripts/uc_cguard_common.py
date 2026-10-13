"""Cost-aware check study: shared helpers. Rebuilds the recommended pipeline's fixings exactly as
scripts/uc_m1x_pipeline.py does (B-hg: mean of the 5 temporal GNNs -> validation-tuned thresholds -> adequacy guard +
min up/down rows + LP-relaxation guard -> fixings of unavailable units forced OFF), and loads instances.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.hybrid import apply_guards  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.ood import OODScenario, avail_view, fixed_share, force_out, override_probs  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from uc_m1x_train import featurizer, load_net  # noqa: E402

ROOT, OODD, RES = "data/generated/uc12", "data/generated/uc12_ood", "results/uc12"
SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
R_FULL, R_FULL5, R_PIPE = "full MILP", "full MILP gap 0.5%", "pipeline: B-hg + LP guard, gap 0.5%"


def timed(f, *a, **k):
    t0 = time.time()
    r = f(*a, **k)
    return r, time.time() - t0


class BHG:
    """the pipeline's fixings for one instance"""

    def __init__(self, sysm):
        torch.set_num_threads(1)
        self.s = sysm
        feat = featurizer(sysm)
        reg = json.load(open(os.path.join(RES, "m1x_sources.json")))
        find = lambda p: p if os.path.exists(p) else os.path.join(RES, "m1x_" + os.path.basename(p))
        self.nets = [load_net(find(p), kind, sysm, feat) for p, kind in reg["pol_n2000_gnnt_ens5"]]
        tu = json.load(open(os.path.join(RES, "m1x_tune_hg_pol_n2000_gnnt_ens5.json")))
        self.lo, self.hi = np.array(tu["lo"]), np.array(tu["hi"])

    def warm(self, m, sc):
        rel = m.solve_dispatch(sc, None, relax=True)
        di = self.inputs(sc, rel)
        for n in self.nets:
            n.predict(di)

    @staticmethod
    def inputs(sc, rel):
        return dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], start=np.array([sc.start]),
                    u_rel=rel.u[None], lmp_rel=rel.lmp[None], flow_rel=rel.flow[None])

    def run(self, m, sc, rel, guards=("adeq", "rows", "lp")):
        """returns (fix_pre, fix_final, info) with info: probs, pre_s (inference + thresholds), guard_s, guard info"""
        g_out = tuple(sc.g_out)
        sv = avail_view(self.s, g_out)
        di = self.inputs(sc, rel)
        p, t_p = timed(lambda: override_probs(np.mean([n.predict(di)[0].astype(np.float64) for n in self.nets], 0), g_out))
        fix0, t_f = timed(fix_dict, p, self.lo, self.hi)
        (fix, ginfo), guard_s = timed(apply_guards, m, sv, sc, dict(fix0), guards)
        fix = force_out(fix, g_out, m.T)
        ginfo = {k: v for k, v in ginfo.items() if k != "guard_s"}
        return fix0, fix, dict(p=p, pre_s=t_p + t_f, guard_s=guard_s, **ginfo)


def shifted_instance(s, k):
    z = np.load(os.path.join(OODD, s, f"{k:03d}.npz"))
    return OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"], day=int(z["day"]), start=int(z["start"]),
                       g_out=tuple(int(g) for g in z["g_out"]), l_out=tuple(int(x) for x in z["l_out"]))


def data_instance(d, k):
    return OODScenario(load=d["load"][k], avail=d["avail"][k], u0=d["u0"][k], sr=d["sr"][k], day=int(d["day"][k]),
                       start=int(d["start"][k]))


def pipeline_records():
    """the m1x pipeline's records {(set, k): {rule: record}}"""
    import glob
    R = {}
    for p in sorted(glob.glob(os.path.join(RES, "m1x_pipeline_eval_s*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            R.setdefault((r["set"], r["k"]), {})[r["rule"]] = r
    return R


def test_jobs():
    """test_fresh 0-119 and the six shifted sets (30 each), interleaved"""
    return [(s, k) for k in range(120) for s in ("tf",) + SHIFTS if s == "tf" or k < 30]


def slack(r):
    return (r.get("shed", 0) or 0) + (r.get("spill", 0) or 0) + (r.get("short", 0) or 0)


__all__ = ["BHG", "shifted_instance", "data_instance", "pipeline_records", "test_jobs", "slack", "timed", "fixed_share",
           "load", "ROOT", "OODD", "RES", "SHIFTS", "R_FULL", "R_FULL5", "R_PIPE"]
