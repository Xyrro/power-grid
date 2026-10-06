"""otsl.b3: the highspy MILP path with incumbent log, time-to-reach, incumbent packing, and the schedule repairs."""
import numpy as np

from otsl.b3 import inc_from_arrays, pack_inc, rep_adequacy, rep_block, rep_minud, solve_milp_hs, time_to_reach
from otsl.uc import violates_min_updown


def test_time_to_reach_first_crossing_tolerance_and_never():
    inc = [(1.0, 120.0, 90.0), (2.5, 105.0, 95.0), (4.0, 100.0, 99.0)]
    assert time_to_reach(inc, 130.0) == 1.0
    assert time_to_reach(inc, 105.0) == 2.5                    # equal counts
    assert time_to_reach(inc, 104.99999) == 2.5                # within the relative tolerance (1e-6)
    assert time_to_reach(inc, 104.9) == 4.0
    assert np.isnan(time_to_reach(inc, 99.0))
    assert np.isnan(time_to_reach([], 1e9))


def test_pack_inc_keeps_first_and_last_and_roundtrips():
    inc = [(float(i), 100.0 - i, 0.0) for i in range(10)]
    tt, oo = pack_inc([inc, inc[:3]], k=4)
    assert tt.shape == (2, 4)
    assert tt[0].tolist() == [0.0, 1.0, 2.0, 9.0] and oo[0].tolist() == [100.0, 99.0, 98.0, 91.0]
    assert np.isnan(tt[1, 3]) and tt[1, :3].tolist() == [0.0, 1.0, 2.0]
    back = inc_from_arrays(tt[1], oo[1])
    assert [(t, o) for t, o, _ in back] == [(t, o) for t, o, _ in inc[:3]]
    # the last incumbent survives truncation, so time_to_reach of the final objective is unchanged
    assert time_to_reach(inc_from_arrays(tt[0], oo[0]), 91.0) == time_to_reach(inc, 91.0)


def _d(sc):
    return {k: np.asarray(getattr(sc, k))[None] for k in ("load", "avail", "u0", "sr")}


def test_repairs_return_min_updown_feasible_schedules(rts, uc12):
    _, sc = uc12
    rng = np.random.default_rng(3)
    d = _d(sc)
    for q in (0.05, 0.3, 0.7):
        u = (rng.random((12, rts.G)) < q).astype(np.int8)
        for rep in (rep_minud(u, sc.u0, rts), rep_adequacy(u, d, 0, rts), rep_block(u, d, 0, rts)):
            assert rep.shape == u.shape and rep.dtype == np.int8
            assert violates_min_updown(rep, sc.u0, rts.min_up, rts.min_dn) == 0
    # a feasible schedule is a fixed point of the min up/down repair
    ok = rep_minud((rng.random((12, rts.G)) < 0.5).astype(np.int8), sc.u0, rts)
    assert np.array_equal(rep_minud(ok, sc.u0, rts), ok)


def test_block_repair_covers_net_load_and_reserve(rts, uc12):
    _, sc = uc12
    d = _d(sc)
    u = np.zeros((12, rts.G), np.int8)                          # nothing on
    v = rep_block(u, d, 0, rts)
    need = sc.load.sum(1) - sc.avail.sum(1) + sc.sr
    assert np.all((v * rts.pmax).sum(1) >= need - 1e-9)


def test_solve_milp_hs_matches_the_scipy_path_and_logs_incumbents(uc3, uc3_milp):
    m, sc = uc3
    sol, rel = uc3_milp
    ref = m.solve_uc(sc, time_limit=60, mip_gap=1e-4)
    assert sol["status"] == "optimal" and ref.status == "optimal"
    assert abs(sol["obj"] - ref.obj) <= 2e-4 * ref.obj
    assert rel.obj <= sol["obj"] * (1 + 1e-9) and sol["bound"] <= sol["obj"] * (1 + 1e-9)
    assert sol["inc"] and abs(sol["inc"][-1][1] - sol["obj"]) <= 1e-6 * sol["obj"]
    assert all(a[0] <= b[0] and a[1] >= b[1] - 1e-6 for a, b in zip(sol["inc"], sol["inc"][1:]))   # improving log
    assert time_to_reach(sol["inc"], sol["obj"]) <= sol["time"]
    assert abs(m.solve_dispatch(sc, sol["u"]).obj - sol["obj"]) <= 1e-6 * sol["obj"]           # dispatch LP of the schedule


def test_solve_milp_hs_respects_fixings(uc3, uc3_milp):
    m, sc = uc3
    sol, _ = uc3_milp
    G = m.dims["G"]
    flip = [g for g in range(G) if sol["u"][0, g] == 0][:3]
    fix = {(t, g): 1 for t in range(m.T) for g in flip}         # force three units on that the optimum keeps off
    fix.update({(t, g): int(sol["u"][t, g]) for t in range(m.T) for g in range(G) if g not in flip and g % 2 == 0})
    r = solve_milp_hs(m, sc, time_limit=60, mip_gap=1e-4, z_fix=fix)
    assert r["u"] is not None and all(r["u"][t, g] == v for (t, g), v in fix.items())
    assert r["obj"] >= sol["bound"] - 1e-6 * abs(sol["bound"])
    # contradictory fixings with min up/down make the reduced problem infeasible
    g = int(np.where((m.s.min_up >= 3) & (sc.u0 == 0))[0][0])
    bad = solve_milp_hs(m, sc, time_limit=30, mip_gap=1e-4, z_fix={(0, g): 1, (1, g): 0})
    assert bad["u"] is None and bad["status"] == "infeasible"
