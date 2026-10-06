"""Hybrid fixing helpers and guards (otsl.hybrid, otsl.combo, otsl.fixpolicy, otsl.uc24ltf): score transform,
fixing <-> mask conversion, cut serialisation, adequacy guard, min up/down row release, LP-relaxation guard."""
import numpy as np

from otsl.combo import adequacy_guard, rule_fixings
from otsl.fixpolicy import fix_from_ranking, lp_guard, relaxed_reduced, release_conflicting_rows, repair_row_forced
from otsl.hybrid import (apply_guards, cut_from_json, cut_to_json, fix_to_masks, harm_norm, harm_to_score,
                         hybrid_fixings, load_cuts, masks_to_fix)
from otsl.ltfx import fix_dict, fix_masks
from otsl.uc24ltf import adequacy_guard as adequacy_guard_24
from otsl.uc24ltf import lp_guard_soft


def _need(sc, t, margin=0.05):
    return (1 + margin) * (sc.load[t].sum() - sc.avail[t].sum() + sc.sr[t])


# ----------------------------------------------------------------------------- score transform (error-cost scores)
def test_harm_to_score_keeps_the_rounding_direction_and_orders_by_error_cost():
    rng = np.random.default_rng(0)
    p = rng.random((6, 9))
    h = np.exp(rng.normal(-10, 2, p.shape))
    mu, sd = harm_norm(h)
    s = harm_to_score(h, p, mu, sd)
    assert np.all(s[p > 0.5] > 0.5) and np.all(s[p <= 0.5] < 0.5) and np.all((s > 0) & (s < 1))
    # lo = hi = 0.5 fixes every decision by rounding
    off, on = fix_masks(s, np.full(9, 0.5), np.full(9, 0.5))
    assert np.array_equal(on, p > 0.5) and np.array_equal(off, p <= 0.5)
    # monotone in the error cost: a costlier OFF prediction moves towards 0.5 (fixed later), an ON one too
    p1 = np.array([0.2, 0.8])
    lo_h, hi_h = harm_to_score(np.array([1e-6, 1e-6]), p1, mu, sd), harm_to_score(np.array([1e-3, 1e-3]), p1, mu, sd)
    assert lo_h[0] < hi_h[0] < 0.5 < hi_h[1] < lo_h[1]


def test_harm_norm_is_mean_and_std_of_log():
    h = np.array([1e-4, 1e-2, 1.0])
    mu, sd = harm_norm(h)
    assert np.isclose(mu, np.log(h).mean()) and np.isclose(sd, np.log(h).std(), rtol=1e-9)


def test_mask_fix_roundtrip_and_hybrid_fixings_without_guards():
    rng = np.random.default_rng(1)
    s = rng.random((5, 7))
    lo, hi = np.full(7, 0.3), np.full(7, 0.7)
    off, on = fix_masks(s, lo, hi)
    fx = masks_to_fix(off, on)
    o2, n2 = fix_to_masks(fx, 5, 7)
    assert np.array_equal(o2, off) and np.array_equal(n2, on)
    fx2, info = hybrid_fixings(s, lo, hi, guards=())
    assert fx2 == fix_dict(s, lo, hi) and info["n_fixed_pre_guard"] == len(fx2)


def test_cut_json_roundtrip(tmp_path):
    import json
    conj = [({0: 0.25, 5: 0.0}, {}), ({}, {3: 0.75}), ({1: 0.1}, {1: 0.9})]
    back = cut_from_json(json.loads(json.dumps(cut_to_json(conj))))
    assert back == conj
    f = tmp_path / "cuts.jsonl"
    f.write_text("\n".join(json.dumps(cut_to_json(c)) for c in (conj, conj[:1])) + "\n")
    assert load_cuts(str(f)) == [conj, conj[:1]]


def test_fix_from_ranking_fixes_the_lowest_scores():
    score = np.array([[0.5, 0.1], [0.3, 0.2]])
    yhat = np.array([[1, 0], [1, 1]])
    assert fix_from_ranking(score, yhat, 0.5) == {(0, 1): 0, (1, 1): 1}


# ----------------------------------------------------------------------------- guards (12-hour RTS-GMLC instance)
def test_adequacy_guard_restores_capacity_every_hour(rts, uc12):
    _, sc = uc12
    T, G = sc.load.shape[0], rts.G
    fix = {(t, g): 0 for t in range(T) for g in range(G)}                  # everything off
    fix[(0, 5)] = 1
    out, rel = adequacy_guard(dict(fix), sc.load, sc.avail, sc.sr, rts)
    assert rel > 0 and out[(0, 5)] == 1
    for t in range(T):
        cap = sum(rts.pmax[g] for g in range(G) if out.get((t, g)) != 0)
        assert cap >= _need(sc, t) - 1e-9
    # minimal in merit order: without the last released unit the margin would be missed
    avg = (rts.c_nl + (rts.seg_c * rts.seg_w).sum(1)) / rts.pmax
    t = 0
    released = [g for g in np.argsort(avg) if fix.get((t, g)) == 0 and (t, g) not in out]
    cap_wo_last = sum(rts.pmax[g] for g in range(G) if out.get((t, g)) != 0) - rts.pmax[released[-1]]
    assert cap_wo_last < _need(sc, t)
    # the copy in otsl.uc24ltf is identical
    out24, rel24 = adequacy_guard_24(dict(fix), sc.load, sc.avail, sc.sr, rts)
    assert out24 == out and rel24 == rel
    # nothing to do when nothing is fixed off
    on_only = {(t, g): 1 for t in range(T) for g in range(3)}
    assert adequacy_guard(dict(on_only), sc.load, sc.avail, sc.sr, rts) == (on_only, 0)


def test_release_conflicting_rows_releases_only_infeasible_rows(rts):
    g = int(np.where(rts.min_up >= 4)[0][0])                               # a unit with a long minimum up time
    h = int(np.where(rts.min_up == 1)[0][0]) if (rts.min_up == 1).any() else None
    u0 = np.zeros(rts.G, np.int8)
    fix = {(1, g): 1, (2, g): 0, (0, 7 if g != 7 else 8): 0}               # start at t = 1, stop at t = 2: conflict
    out, rel = release_conflicting_rows(dict(fix), rts, u0)
    assert rel == 2 and (1, g) not in out and (2, g) not in out and len(out) == 1
    ok = {(1, g): 1, (1 + int(rts.min_up[g]), g): 0}                        # stop after the minimum up time: fine
    assert release_conflicting_rows(dict(ok), rts, u0) == (ok, 0)
    assert repair_row_forced(np.zeros(6, np.int8), 0, int(rts.min_up[g]), int(rts.min_dn[g]), {1: 1, 2: 0}) is None
    if h is not None:
        flip = {(0, h): 1, (1, h): 0}
        if rts.min_dn[h] == 1:
            assert release_conflicting_rows(dict(flip), rts, u0)[1] == 0


def test_lp_guard_removes_penalised_slack_of_the_relaxed_reduced_problem(rts, uc12):
    m, sc = uc12
    T, G = m.T, rts.G
    rel = m.solve_dispatch(sc, None, relax=True)
    fix = {(6, g): 0 for g in range(G)}                                    # every unit off in hour 6: shedding
    fix.update({(t, g): 1 for t in (0, 1) for g in range(G) if rel.u[t, g] > 1 - 1e-9})
    _, slack0, _ = relaxed_reduced(m, sc, fix)
    assert slack0[6] > 1e-6 and slack0.sum() == slack0[6]
    out, released, secs = lp_guard(m, rts, sc, fix)
    assert released > 0 and secs > 0 and set(out) <= set(fix)
    _, slack1, _ = relaxed_reduced(m, sc, out)
    assert slack1.sum() <= 1e-6
    # fixings that agree with an integral LP relaxation are left alone
    integral = {(t, g): int(round(rel.u[t, g])) for t in range(T) for g in range(G) if abs(rel.u[t, g] - round(rel.u[t, g])) < 1e-9}
    assert lp_guard(m, rts, sc, integral)[1] == 0
    # the soft variant is a no-op on an infeasible relaxation (min up/down conflict) and equal otherwise
    out_s, rel_s, _, inf_s = lp_guard_soft(m, rts, sc, fix)
    assert out_s == out and rel_s == released and not inf_s


def test_apply_guards_order_and_info(rts, uc12):
    m, sc = uc12
    T, G = m.T, rts.G
    fix = {(t, g): 0 for t in range(T) for g in range(G)}
    out, info = apply_guards(m, rts, sc, fix, ("adeq", "rows"))
    assert set(info) >= {"released_adeq", "released_rows", "guard_s"} and "released_lp" not in info
    assert info["released_adeq"] > 0 and len(out) == len(fix) - info["released_adeq"] - info["released_rows"]
    assert fix == {(t, g): 0 for t in range(T) for g in range(G)}          # the input is not modified
    out2, info2 = apply_guards(m, rts, sc, fix, ("adeq", "rows", "lp"))
    assert out2.items() <= out.items() and "lp_guard_s" in info2               # the LP guard only releases


def test_rule_fixings_ranks_and_guards(rts, uc12):
    _, sc = uc12
    rng = np.random.default_rng(2)
    p = rng.random((12, rts.G))
    fx = rule_fixings("rac", 0.9, p, sc, rts)
    assert len(fx) == int(round(0.9 * p.size)) and all(v == int(p[t, g] > 0.5) for (t, g), v in fx.items())
    conf = np.minimum(p, 1 - p)
    worst_fixed = max(conf[t, g] for t, g in fx)
    free = [conf[t, g] for t in range(12) for g in range(rts.G) if (t, g) not in fx]
    assert worst_fixed <= min(free) + 1e-12
    h = rng.random(p.shape)
    fh = rule_fixings("harm", 0.9, p, sc, rts, harm=h)
    assert len(fh) <= len(fx) and all(v == int(p[t, g] > 0.5) for (t, g), v in fh.items())


# ----------------------------------------------------------------------------- guard-aware tuners
def _one_instance(m, sc, obj):
    T, G = m.T, m.dims["G"]
    return dict(load=sc.load[None], avail=sc.avail[None], u0=sc.u0[None], sr=sc.sr[None], obj=np.array([obj]),
                u=np.zeros((1, T, G), np.int8))


def test_hybrid_tuner_checks_the_guarded_fixings_and_logs_cuts(rts, uc3, uc3_milp, tmp_path):
    from otsl.hybrid import HybridTuner, load_cuts
    m, sc = uc3
    sol, rel = uc3_milp
    d = _one_instance(m, sc, sol["obj"])
    pi = rel.u[None].astype(float)
    log = tmp_path / "cuts.jsonl"
    tu = HybridTuner(m, rts, d, pi, 0.01, guards=("adeq", "rows"), cut_log=str(log), Q=5, log=lambda s: None)
    off = np.ones((m.T, rts.G), bool)                                       # everything off
    on = np.zeros_like(off)
    po, pn = tu.post(0, off, on)
    ref, _ = apply_guards(m, rts, sc, masks_to_fix(off, on), ("adeq", "rows"))
    assert masks_to_fix(po, pn) == ref and po.sum() < off.sum()
    tu.post(0, off, on)
    assert tu.stats["n_guard"] == 1 and tu.stats["n_guard_cached"] == 1      # memoised per pre-guard set
    conj = [({0: 0.1}, {}), ({1: 0.2}, {1: 0.8})]
    tu.master.add_cut(conj)
    assert load_cuts(str(log)) == [conj] and len(tu.master.cuts) == 1
    plain = HybridTuner(m, rts, d, pi, 0.01, guards=(), Q=5, log=lambda s: None)
    assert plain.post(0, off, on) == (off, on)                              # no guards: the faithful tuner


def test_guard_fix_conflict_release_and_soft_lp_guard(rts, uc12):
    from otsl.uc24ltf import guard_fix
    m, sc = uc12
    g = int(np.where((rts.min_up >= 4) & (sc.u0 == 0))[0][0])
    fix = {(1, g): 1, (2, g): 0}                                            # min up/down conflict
    fix.update({(6, h): 0 for h in range(rts.G)})                           # and an hour with everything off
    # without the conflict release first, the relaxed reduced problem is infeasible: the soft LP guard leaves it
    out, info = guard_fix(m, rts, sc, dict(fix), use_adequacy=False, use_lp=True)
    assert info["lp_infeasible"] and out == fix
    # conflict release first: the LP guard then sees a feasible relaxation and removes the shortage of hour 6
    out, info = guard_fix(m, rts, sc, dict(fix), use_adequacy=False, use_lp=True, conflict_first=True)
    assert not info["lp_infeasible"] and info["released_conflict"] == 3 and info["released_lp"] > 0
    assert (1, g) not in out and (2, g) not in out and (6, g) not in out    # the conflicting unit's whole row
    _, slack, _ = relaxed_reduced(m, sc, out)
    assert slack.sum() <= 1e-6
