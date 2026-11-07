"""otsl.screen: PTDF flows reproduce the B-theta network's, and the screened MILP solves the network UC."""
import numpy as np

from otsl.ood import OODModel, OODScenario
from otsl.screen import ScreenedUC, certified, confidence_fixings, monitored_from_lp
from otsl.uc import load_rts_gmlc, make_scenario


def _setup(T=4):
    s = load_rts_gmlc()
    sc = make_scenario(s, 100, np.random.default_rng(3), T=T, start=8)
    return s, sc, ScreenedUC(s, T=T), OODModel(s, T=T, network=True)


def test_ptdf_flows_match_btheta():
    s, sc, scr, m = _setup()
    rel = m.solve_dispatch(sc, None, relax=True)
    # the network LP's solution, read as a copper-plate solution: same columns, so its flows must come out equal
    lo, hi, lb, ub = m._rhs_bounds(sc)
    from otsl.ltfx import highs_solve
    r = highs_solve(m.c, m.A, lo, hi, lb, ub, None, 60, 1e-3)
    f = scr.flows(r["x"], sc)
    assert np.allclose(f, rel.flow, atol=1e-6)


def test_ptdf_with_line_out():
    s, sc, scr, m = _setup()
    sco = OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, l_out=(5,))
    lo, hi, lb, ub = m._rhs_bounds(sco)
    from otsl.ltfx import highs_solve
    r = highs_solve(m.c, m.A, lo, hi, lb, ub, None, 60, 1e-3)
    f = scr.flows(r["x"], sco)
    fx = r["x"][m.off["f"][0]:m.off["f"][0] + m.off["f"][1] * m.T].reshape(m.T, -1)
    keep = np.arange(s.L) != 5
    assert np.allclose(f[:, keep], fx[:, keep], atol=1e-6)


def test_screened_milp_is_feasible_for_the_network_and_as_good():
    s, sc, scr, m = _setup()
    rel = m.solve_dispatch(sc, None, relax=True)
    z = scr.solve(sc, monitored_from_lp(rel.flow, s.fmax), gap=1e-4)
    assert z["ok"]
    lp = m.solve_dispatch(sc, z["u"])                    # the schedule priced on the full network
    assert abs(lp.obj - z["obj"]) <= 1e-6 * abs(lp.obj)  # no line overloaded: same cost as on the copper plate
    full = m.solve_uc(sc, mip_gap=1e-4)
    assert z["obj"] <= full.obj * (1 + 2e-4) and z["bound"] <= full.obj * (1 + 1e-6)


def test_certificate_and_confidence_rule():
    s, sc, scr, m = _setup()
    rel = m.solve_dispatch(sc, None, relax=True)
    p = np.clip(rel.u, 0, 1)
    fix = confidence_fixings(p, s, sc.u0, 0.05)
    assert fix and all(v == int(p[t, g] > 0.5) for (t, g), v in fix.items())
    cand = scr.solve(sc, monitored_from_lp(rel.flow, s.fmax), fix=fix, gap=5e-3)
    x, info = certified(scr, sc, cand, rel.obj, eps=5e-3)
    C = m.solve_dispatch(sc, scr.u_of(x)).obj
    assert C <= cand["obj"] * (1 + 1e-9)
    bound = rel.obj if info["cert"] == "lp" else info["cert_bound"]
    assert (C - bound) / C <= 5e-3 + 1e-9
