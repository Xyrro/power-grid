"""Paper metrics (Learning to Fix, Sec. IV-C) and bootstrap helpers wherever they live (otsl.base and the report
scripts), on hand-made records with known answers; and the no-learning baselines of otsl.base."""
import json

import numpy as np

from otsl.base import (boot_ci, gap_db, gap_within, incumbent_at, lp_integral_fixings, lp_round_candidates,
                       lp_round_screen, paired, paper_stats, solve_reduced, time_to_gap)

# --------------------------------------------------------------------------------------------- toy records
DB = np.array([100.0, 200.0, 50.0])
T_FULL = np.array([10.0, 20.0, 40.0])
COST = [101.0, None, 51.0]                          # instance 1 infeasible
T_M = np.array([2.0, 5.0, 4.0])


def test_gap_db_and_paper_stats():
    g = gap_db(COST, DB)
    assert np.isclose(g[0], 1 / 101 * 100) and np.isnan(g[1]) and np.isclose(g[2], 1 / 51 * 100)
    st = paper_stats([True, False, True], COST, DB, T_M, T_FULL, fixed=[0.8, 0.9, 0.7], served=[True, True, False])
    assert st["n"] == 3 and np.isclose(st["feasible"], 200 / 3) and st["n_infeasible"] == 1
    assert np.isclose(st["gap_mean"], (100 / 101 + 100 / 51) / 2) and np.isclose(st["gap_max"], 100 / 51)
    assert np.isclose(st["speedup_mean"], (5 + 10) / 2) and np.isclose(st["speedup_median"], 7.5)
    assert np.isclose(st["speedup_ratio_of_means"], 25 / 3)            # ratio of mean times differs from the mean ratio
    assert np.isclose(st["fixed_mean"], 75.0) and np.isclose(st["served"], 50.0)
    assert np.isclose(st["time_mean"], 3.0)
    assert paper_stats([False], [None], [1.0], [1.0], [1.0])["feasible"] == 0.0


def test_boot_ci_and_paired_are_deterministic_and_sensible():
    x = np.random.default_rng(0).normal(1.0, 0.5, 200)
    lo, hi = boot_ci(x)
    assert lo < x.mean() < hi and hi - lo < 0.3 and boot_ci(x) == [lo, hi]
    assert boot_ci(np.full(5, 2.0)) == [2.0, 2.0] and np.isnan(boot_ci([])[0])
    assert np.allclose(boot_ci([1.0, np.nan, 3.0]), boot_ci([1.0, 3.0]))    # non-finite values are dropped
    p = paired(x + 0.25, x)
    assert np.isclose(p["diff"], 0.25) and np.allclose(p["ci"], 0.25) and p["n"] == 200
    p = paired(np.array([1.0, np.nan, 3.0]), np.array([0.0, 1.0, np.inf]))
    assert p["n"] == 1 and p["diff"] == 1.0


def test_incumbent_trace_helpers():
    inc = [[1.0, 150.0, 80.0], [3.0, 110.0, 90.0], [8.0, 101.0, 99.0]]
    assert incumbent_at(inc, 0.5) == np.inf and incumbent_at(inc, 3.0) == 110.0 and incumbent_at(inc, 100) == 101.0
    assert np.isnan(gap_within(inc, 0.9, 100.0))
    assert np.isclose(gap_within(inc, 5.0, 100.0), 10 / 110 * 100)
    trace = [[0.5, np.inf, 50.0], [2.0, 150.0, 90.0], [4.0, 110.0, 100.0], [9.0, 101.0, 100.5]]
    assert time_to_gap(trace, 0.10) == 4.0 and time_to_gap(trace, 0.01) == 9.0 and np.isnan(time_to_gap(trace, 1e-4))


# --------------------------------------------------------------------------------------------- report scripts
def _write(tmp_path, recs):
    f = tmp_path / "eval.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return str(f)


def test_hybrid_report_per_instance_and_stats(tmp_path):
    import uc_hybrid_report as H
    recs = []
    for i in range(3):
        recs.append(dict(i=i, rule="full MILP", feasible=True, obj=DB[i] / (1 - 0.001), mip_gap=0.001, time=T_FULL[i],
                         pre_s=0.0, relax_s=0.0, guard_s=0.0, shed=0.0, short=0.0, fixed_share=0.0))
        ok = COST[i] is not None
        r = dict(i=i, rule="rule A", feasible=ok, obj=COST[i] if ok else np.inf, mip_gap=0.0, time=T_M[i] - 0.5,
                 pre_s=0.1, relax_s=0.3, guard_s=0.1, shed=0.0, short=0.0, fixed_share=0.9)
        if not ok:
            r.update(fb_obj=DB[i] * 1.001, fb_time=T_FULL[i], fb_shed=0.0, fb_short=0.0)
        recs.append(r)
    d = {"obj": np.array([r["obj"] for r in recs if r["rule"] == "full MILP"])}
    out, common, t_full = H.per_instance(_write(tmp_path, recs), d)
    assert common == [0, 1, 2] and np.allclose(t_full, T_FULL)
    a = out["rule A"]
    assert np.allclose(a["t_m"], T_M) and a["feas"].tolist() == [True, False, True]
    assert np.allclose(a["gap_db"][[0, 2]], gap_db(COST, DB)[[0, 2]]) and np.isnan(a["gap_db"][1])
    st = H.stats(a, t_full)
    ref = paper_stats(a["feas"], COST, DB, T_M, T_FULL)
    assert np.isclose(st["gap_mean"], ref["gap_mean"]) and np.isclose(st["speedup_mean"], ref["speedup_mean"])
    assert np.isclose(st["feasible"], ref["feasible"])
    # our earlier convention: the infeasible instance pays the fallback (full MILP) time and gets its cost
    assert np.isclose(a["t_fb"][1], T_M[1] + T_FULL[1]) and np.isclose(a["g_ref"][1], (1.001 * (1 - 0.001) - 1) * 100)
    p = H.paired(a["gap_db"], out["full MILP"]["gap_db"], "gap_db")
    assert p["n"] == 2 and np.isclose(p["diff"], np.nanmean(a["gap_db"] - out["full MILP"]["gap_db"]))
    assert H.boot([1.0, 1.0]) == [1.0, 1.0]


def test_papereval_stats_matches_the_definitions():
    import uc_papereval_score as P
    recs = [dict(obj=COST[k], shed=0.0, short=0.0, time=T_M[k] - 1.0, feasible=COST[k] is not None, fixed=0.9,
                 obj_any=COST[k] if COST[k] is not None else 100.5, time_any=T_M[k] + (0 if COST[k] else T_FULL[k]))
            for k in range(3)]
    ov = np.ones(3)
    st = P.stats(recs, DB, T_FULL, ov, DB, "rule")
    ref = paper_stats([r["feasible"] for r in recs], COST, DB, T_M, T_FULL)
    for k in ("gap_mean", "gap_median", "gap_max", "speedup_mean", "speedup_median", "feasible", "time_mean"):
        assert np.isclose(st[k], ref[k]), k
    assert np.isclose(st["speedup_mean_no_overhead"], np.mean(T_FULL[[0, 2]] / (T_M[[0, 2]] - 1.0)))
    assert np.isclose(st["fixed_mean"], 90.0) and st["served"] == 100.0


def test_uc24ltf_and_pglib_report_helpers():
    import uc_pglib_report as G
    import uc_uc24ltf_report as U
    inc = [(1.0, 120.0, 0.0), (2.0, 100.0, 0.0)]
    assert U.first_below(inc, 120.0) == 1.0 and U.first_below(inc, 100.0) == 2.0 and np.isnan(U.first_below(inc, 99.0))
    x = np.arange(10.0)
    assert U.boot_ci(x) == boot_ci(x)                                   # same resampling (seed 0, 2000 draws)
    assert G.tq([1.0, 2.0], [120.0, 100.0], 110.0, 50.0) == 2.0 and G.tq([1.0], [120.0], 110.0, 50.0) == 50.0
    assert G.hard_ok(dict(feasible=True, shed=0.0, short=5e-5)) and not G.hard_ok(dict(feasible=True, shed=1e-3, short=0.0))
    assert not G.hard_ok(dict(feasible=False, shed=0.0, short=0.0))
    recs = [dict(bound=100.0, rules=dict(a=dict(feasible=True, obj=102.0, shed=0.0, short=0.0),
                                         b=dict(feasible=True, obj=101.0, shed=0.0, short=0.0))),
            dict(bound=100.0, rules=dict(a=dict(feasible=True, obj=103.0, shed=1.0, short=0.0),
                                         b=dict(feasible=True, obj=101.0, shed=0.0, short=0.0)))]
    p = G.paired(recs, "a", "b")
    assert p["n"] == 1 and np.isclose(p["mean"], 2 / 102 * 100 - 1 / 101 * 100)


# --------------------------------------------------------------------------------------------- no-learning baselines
def test_lp_integral_fixings_tolerance():
    u = np.array([[0.0, 1.0, 0.5, 1e-7], [0.999, 0.02, 1.0 - 1e-9, 0.3]])
    assert lp_integral_fixings(u, 1e-6) == {(0, 0): 0, (0, 3): 0, (0, 1): 1, (1, 2): 1}
    assert lp_integral_fixings(u, 0.05) == {(0, 0): 0, (0, 3): 0, (1, 1): 0, (0, 1): 1, (1, 0): 1, (1, 2): 1}
    assert len(lp_integral_fixings(u, 0.5)) == u.size                 # tol 0.5 rounds everything (0.5 -> OFF)
    assert lp_integral_fixings(u, 0.5)[(0, 2)] == 0


def test_lp_round_screen_picks_the_cheapest_repaired_rounding(rts, uc12):
    from otsl.uc import violates_min_updown
    m, sc = uc12
    rel = m.solve_dispatch(sc, None, relax=True)
    cands = lp_round_candidates(rel.u, sc, rts)
    assert [th for th, _ in cands] == [0.001, 0.05, 0.2, 0.5, 0.8]
    need = sc.load.sum(1) - sc.avail.sum(1) + sc.sr
    for _, u in cands:
        assert violates_min_updown(u, sc.u0, rts.min_up, rts.min_dn) == 0
        assert np.all((u * rts.pmax).sum(1) >= need - 1e-9)
    best = lp_round_screen(m, rts, sc, rel.u)
    assert best["n_lp"] == len({u.tobytes() for _, u in cands}) and len(best["costs"]) == 5
    assert np.isclose(best["cost"], min(best["costs"])) and best["cost"] >= rel.obj - 1e-6
    k = [th for th, _ in cands].index(best["th"])
    assert np.array_equal(best["u"], cands[k][1])
    assert np.isclose(m.solve_dispatch(sc, best["u"]).obj, best["cost"])


def test_solve_reduced_record(uc3, uc3_milp):
    m, sc = uc3
    sol, rel = uc3_milp
    fix = lp_integral_fixings(rel.u, 1e-6)
    r = solve_reduced(m, sc, fix, time_limit=30, mip_gap=1e-4, trace_every=0.01)
    assert r["feasible"] and r["n_fixed"] == len(fix) and r["obj"] >= sol["bound"] - 1e-6 * sol["bound"]
    assert r["inc"] and isinstance(r["inc"][0], list) and json.dumps(r)
    full = solve_reduced(m, sc, None, time_limit=30, mip_gap=1e-4)
    assert full["n_fixed"] == 0 and abs(full["obj"] - sol["obj"]) <= 1e-6 * sol["obj"]
