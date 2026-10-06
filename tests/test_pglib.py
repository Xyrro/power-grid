"""PGLib-UC matrix model (otsl.pglib) on a tiny synthetic fleet: MILP = brute force over every commitment, LP
relaxation <= MILP, dispatch LP reproduces the MILP objective, continuous start-up categories = binary ones, reduced
MILP fixings, initial conditions, repairs, time-to-reach."""
import itertools

import numpy as np
import pytest

from otsl.pglib import PGModel, PGScenario, PGSystem, check_no_carryover, rep_minud, time_to_reach

T = 3


def tiny_system():
    """4 units: one must-run, three free ones with 2-3 piecewise points, hot / cold start-up categories, minimum
    up / down times up to 3 h, ramp limits that bind; one wind unit."""
    return PGSystem(
        names_all=np.array(["a", "b", "c", "d"]), must_run_all=np.array([1, 0, 0, 0]),
        pmax_all=np.array([100.0, 80.0, 50.0, 30.0]), pmin_all=np.array([40.0, 20.0, 10.0, 5.0]),
        ru_all=np.array([60.0, 25.0, 50.0, 30.0]), rd_all=np.array([60.0, 25.0, 50.0, 30.0]),
        su_all=np.array([100.0, 30.0, 50.0, 30.0]), sd_all=np.array([100.0, 30.0, 50.0, 30.0]),
        ut_all=np.array([1, 3, 2, 1]), dt_all=np.array([1, 2, 2, 1]),
        pw_mw=[np.array([40.0, 70.0, 100.0]), np.array([20.0, 50.0, 80.0]), np.array([10.0, 30.0, 50.0]), np.array([5.0, 30.0])],
        pw_cost=[np.array([800.0, 1400.0, 2100.0]), np.array([500.0, 1100.0, 1900.0]), np.array([300.0, 900.0, 1700.0]),
                 np.array([400.0, 1500.0])],
        st_cost=[np.array([0.0]), np.array([300.0, 600.0]), np.array([100.0, 150.0]), np.array([20.0])],
        st_lag=[[1], [1, 3], [1, 2], [1]],
        u0_all=np.array([1, 1, 0, 0]), up0_all=np.array([10, 5, 0, 0]), dn0_all=np.array([0, 0, 5, 5]),
        p0_all=np.array([60.0, 40.0, 0.0, 0.0]), ren_names=["w"])


@pytest.fixture(scope="module")
def tiny():
    s = tiny_system()
    m = PGModel(s, T=T)
    sc = PGScenario(load=np.array([[150.0], [185.0], [215.0]]), avail=np.array([[20.0], [10.0], [5.0]]),
                    avail_min=np.zeros((T, 1)), sr=np.array([5.0, 6.0, 7.0]), u0=s.u0.copy())
    return s, m, sc, m.solve_milp(sc, time_limit=30, mip_gap=1e-9)


def test_system_view_of_the_free_units(tiny):
    s, m, sc, _ = tiny
    assert s.G == 3 and s.G_all == 4 and list(s.free) == [1, 2, 3] and list(s.mr) == [0]
    assert check_no_carryover(s)
    assert np.allclose(s.seg_w[0], [30.0, 30.0]) and np.allclose(s.seg_c[0], [20.0, 800.0 / 30.0])
    assert np.allclose(s.c_su, [600.0, 150.0, 20.0]) and s.identical_groups() == []
    assert m.dims["G"] == 3 and m.nv == m.col_one + 1


def test_milp_equals_brute_force_over_all_commitments(tiny):
    s, m, sc, sol = tiny
    assert sol["status"] == "Optimal" and sol["shed"] < 1e-9 and sol["short"] < 1e-9
    best = np.inf
    for bits in itertools.product((0, 1), repeat=T * s.G):
        u = np.array(bits, float).reshape(T, s.G)
        r = m.solve_dispatch(sc, u)
        if r.ok:
            best = min(best, r.obj)
    assert abs(best - sol["obj"]) <= 1e-7 * abs(best)


def test_relaxation_dispatch_and_binary_start_categories(tiny):
    s, m, sc, sol = tiny
    rel = m.solve_dispatch(sc, None, relax=True)
    assert rel.obj <= sol["obj"] + 1e-7
    assert abs(m.solve_dispatch(sc, sol["u"]).obj - sol["obj"]) <= 1e-7 * sol["obj"]
    mb = PGModel(s, T=T, delta_binary=True)
    solb = mb.solve_milp(sc, time_limit=30, mip_gap=1e-9)
    assert abs(solb["obj"] - sol["obj"]) <= 1e-7 * sol["obj"]
    # the relaxed reduced problem at the relaxation's own integral values has the relaxation's cost
    fix = {(t, g): int(round(rel.u[t, g])) for t in range(T) for g in range(s.G) if abs(rel.u[t, g] - round(rel.u[t, g])) < 1e-9}
    c, slack, _ = m.relaxed_reduced(sc, fix)
    assert abs(c - rel.obj) <= 1e-7 * abs(rel.obj) and slack.sum() < 1e-9


def test_reduced_milp_respects_fixings_and_initial_conditions(tiny):
    s, m, sc, sol = tiny
    # unit b (free index 0) is on at t0 with its minimum up time served; unit c (index 1) must stay off for
    # nothing: dn0 = 5 >= DT = 2. Fix c on in every hour and d off.
    fix = {(t, 1): 1 for t in range(T)}
    fix.update({(t, 2): 0 for t in range(T)})
    r = m.solve_milp(sc, time_limit=30, mip_gap=1e-9, z_fix=fix)
    assert r["u"] is not None and all(r["u"][t, g] == v for (t, g), v in fix.items())
    assert r["obj"] >= sol["obj"] - 1e-7 * sol["obj"]
    # an initial minimum down time still to serve forbids an early start
    s2 = tiny_system()
    s2.dn0_all = np.array([0, 0, 1, 5])
    s2.__post_init__()
    assert not check_no_carryover(s2)
    m2 = PGModel(s2, T=T)
    _, _, lb, ub = m2._rhs_bounds(sc)
    U = m2.blk("u")
    assert ub[U[0, 1]] == 0.0 and ub[U[1, 1]] == 1.0           # DT = 2, off for 1 h: off in hour 1 only


def test_rep_minud_and_time_to_reach(tiny):
    s, _, sc, _ = tiny
    u = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.int8)
    v = rep_minud(u, s)
    from otsl.uc import violates_min_updown
    assert violates_min_updown(v, s.u0, s.min_up, s.min_dn) == 0
    t = np.array([0.5, 1.0, 3.0, np.nan])
    o = np.array([10.0, 8.0, 7.0, np.nan])
    assert time_to_reach(t, o, 8.0) == 1.0 and time_to_reach(t, o, 7.5) == 3.0 and np.isnan(time_to_reach(t, o, 6.0))
