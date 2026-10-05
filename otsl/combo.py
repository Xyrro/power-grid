"""Combined pipelines for 12-hour UC (B2) and their multi-seed confirmation.

End-to-end pipeline (one LP, or k LPs with screening):
    predictor probabilities p -> threshold th (calibrated on val) -> block adequacy repair (min up/down-aware,
    otsl.constrained.adequacy_repair_blocks) -> min up/down repair -> dispatch LP.
    Screening: the same for every threshold of a small grid, best schedule by the exact LP.

Solver-acceleration pipeline:
    probabilities p (self-trained / REINFORCE / MILP-label BCE) -> learned error-cost ranking
    (otsl.fixpolicy harm model, features computed from p) -> fix the target share to round(p) -> adequacy guard
    -> min up/down row check -> LP-relaxation guard -> reduced MILP.

Only composition lives here; every component is imported from its own module.
"""
from __future__ import annotations

import numpy as np
import torch

from .constrained import adequacy_repair_blocks
from .fixpolicy import (FixFeaturizer, HarmEnsemble, HarmModel, fix_from_ranking, lp_guard,  # noqa: F401
                        release_conflicting_rows)
from .uc import repair_min_updown

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ----------------------------------------------------------------------------------- end-to-end decoding
def block_decode(p, d, sysm, th=0.5):
    """threshold -> block adequacy repair -> min up/down repair, for every instance of d. p: [N, T, G]"""
    out = np.empty(p.shape, np.int8)
    for i in range(len(p)):
        u = (p[i] > th).astype(np.int8)
        u = adequacy_repair_blocks(u, d["u0"][i], d["load"][i], d["avail"][i], d["sr"][i], sysm)
        out[i] = repair_min_updown(u, d["u0"][i], sysm.min_up, sysm.min_dn)
    return out


def mud_decode(p, d, sysm, th=0.5):
    """threshold -> min up/down repair only (the framework's one-shot decoder)"""
    return np.array([repair_min_updown((p[i] > th).astype(np.int8), d["u0"][i], sysm.min_up, sysm.min_dn)
                     for i in range(len(p))])


def served_mask(shed, short):
    return (np.asarray(shed) < 1e-6) & (np.asarray(short) < 1e-6)


def e2e_metrics(cost, shed, short, ref, units=None):
    """served share, median gap, mean gap on served instances, mean gap (all in %), vs reference cost ref"""
    gap = (np.asarray(cost) - ref) / ref * 100
    hard = served_mask(shed, short)
    out = {"served": float(hard.mean() * 100), "gap_median": float(np.median(gap)),
           "gap_mean_served": float(gap[hard].mean()) if hard.any() else float("nan"),
           "gap_mean": float(gap.mean()), "n": int(len(gap))}
    if units is not None:
        out["units_on"] = float(units)
    return out


# ----------------------------------------------------------------------------------- harm model I/O
def harm_from_file(path):
    obj = torch.load(path, weights_only=False)
    return HarmEnsemble.from_states(obj["d_in"], obj["hidden"], obj["members"])


def harm_scores(ens, ff, p, d, idx):
    """expected cost of fixing every decision of instances idx, with features computed from probabilities p
    (p[k] belongs to instance idx[k])"""
    X = np.stack([ff(p[k], d, i) for k, i in enumerate(idx)])
    return ens.score(X)


# ----------------------------------------------------------------------------------- fixing rules
def adequacy_guard(fix, load_, avail, sr, sysm, margin=0.05):
    """scripts/uc_fixing.adequacy_guard (copied verbatim to keep otsl free of script imports): unfix OFF-fixed
    units (cheapest first) until, every hour, the units not fixed off cover net load + reserve with a margin."""
    avg = (sysm.c_nl + (sysm.seg_c * sysm.seg_w).sum(1)) / sysm.pmax
    order = np.argsort(avg)
    released = 0
    for t in range(load_.shape[0]):
        need = (1 + margin) * (load_[t].sum() - avail[t].sum() + sr[t])
        off = {g for (tt, g), v in fix.items() if tt == t and v == 0}
        cap = sum(sysm.pmax[g] for g in range(sysm.G) if g not in off)
        for g in order:
            if cap >= need:
                break
            if g in off:
                del fix[(t, int(g))]
                cap += sysm.pmax[g]
                released += 1
    return fix, released


def rule_fixings(rule, ratio, p, sc, sysm, harm=None):
    """Fixings {(t, g): v} of one instance before any solver-based guard.
    rule: 'rac'  RACLearn: largest |p - 0.5|
          'asym' min(p, 1-p) with OFF decisions x 10, + adequacy guard
          'harm' learned expected error cost (harm [T, G]), + adequacy guard
    Values are always round(p) of the same probabilities."""
    yhat = (p > 0.5).astype(int)
    err = np.minimum(p, 1 - p)
    if rule == "rac":
        return fix_from_ranking(err, yhat, ratio)
    if rule == "asym":
        fix = fix_from_ranking(err * np.where(p < 0.5, 10.0, 1.0), yhat, ratio)
        return adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)[0]
    if rule == "harm":
        fix = fix_from_ranking(harm, yhat, ratio)
        return adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)[0]
    raise ValueError(rule)


def solver_guards(m, sysm, sc, fix):
    """row feasibility (min up/down) + LP-relaxation guard of otsl.fixpolicy. Returns (fix, info, LP seconds)."""
    fix, rel_rows = release_conflicting_rows(fix, sysm, sc.u0)
    fix, rel_lp, secs = lp_guard(m, sysm, sc, fix)
    return fix, {"released_rows": rel_rows, "released_lp": rel_lp, "lp_guard_s": secs}, secs
