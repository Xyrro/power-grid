"""otsl.hard: scenario builder, relaxation statistics, minimal conflict release."""
import numpy as np

from otsl.hard import make_scenario_h, nearest_row, relax_stats, release_min_conflicts, split_days
from otsl.uc import make_scenario, violates_min_updown


def test_scenario_matches_make_scenario(rts):
    for T, start in ((12, 5), (24, 0)):
        a = make_scenario(rts, 100, np.random.default_rng(3), T=T, start=start, reserve_frac=0.05)
        b = make_scenario_h(rts, 100, np.random.default_rng(3), T=T, start=start, reserve_frac=0.05)
        for k in ("load", "avail", "u0", "sr"):
            assert np.array_equal(getattr(a, k), getattr(b, k))


def test_two_day_window_continues_next_day(rts):
    a = make_scenario(rts, 100, np.random.default_rng(3), T=24, start=0)
    b = make_scenario_h(rts, 100, np.random.default_rng(3), T=48, start=0)
    assert b.load.shape[0] == 48 and np.allclose(a.load, b.load[:24])
    ratio = b.load[24:].sum(1) / (rts.area_load[101].sum(1))       # next day's profile with the same noise draws
    assert ratio.std() < 0.05 * ratio.mean()
    d = split_days(rts, 48)
    assert d["train"].max() <= rts.area_load.shape[0] - 2


def test_relax_stats_counts(rts):
    T, G = 3, rts.G
    u_star = np.zeros((T, G), np.int8)
    u_star[:, 0] = 1
    u_rel = u_star.astype(float).copy()
    lone = [g for g in range(G) if not any(g in grp for grp in rts.identical_groups())]
    u_rel[0, 1] = 0.4          # fractional
    u_rel[1, lone[0]] = 1.0    # integral but wrong (MILP says 0) on a unit without identical copies
    s = relax_stats(u_rel, u_star, rts, np.zeros(G, np.int8))
    assert s["n_frac"] == 1 and s["n_wrong"] == 1 and s["n_wrong_one"] == 1
    assert abs(s["int_share"] - (T * G - 1) / (T * G)) < 1e-12


def test_nearest_row_and_release(rts):
    g = int(np.argmax(rts.min_up))          # a unit with a long minimum up time
    U, D = int(rts.min_up[g]), int(rts.min_dn[g])
    T = 12
    forced = {0: 0, 1: 1, 2: 0}             # on for one hour only: infeasible if U > 1
    row = nearest_row(forced, 0, U, D, T)
    assert not violates_min_updown(row[:, None], np.zeros(1), np.array([U]), np.array([D]))
    assert sum(row[t] != v for t, v in forced.items()) == 1
    fix = {(t, g): v for t, v in forced.items()}
    out, rel = release_min_conflicts(fix, rts, np.zeros(rts.G, np.int8), T)
    assert rel == 1 and len(out) == 2
