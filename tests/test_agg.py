"""otsl.agg and otsl.probe: aggregating identical units is exact, and Lagrangian probing only fixes what no cheaper
schedule changes."""
import numpy as np

from otsl.agg import AggScreenedUC, agg_certified_pipeline
from otsl.ood import OODModel, OODScenario
from otsl.probe import hourly_on_cost, lp_relaxation_duals, probe_fixings, unit_margins
from otsl.screen import ScreenedUC, monitored_from_lp
from otsl.uc import load_rts_gmlc, make_scenario, violates_min_updown


def _setup(T=6, day=100, seed=3, start=8):
    s = load_rts_gmlc()
    sc = make_scenario(s, day, np.random.default_rng(seed), T=T, start=start)
    return s, sc, OODModel(s, T=T, network=True)


def test_aggregation_is_exact():
    s, sc, m = _setup()
    ag, scr = AggScreenedUC(s, T=m.T), ScreenedUC(s, T=m.T)
    assert sum(len(g) for g in ag.base_groups) == 56          # 24 groups of copies on RTS-GMLC
    rel = m.solve_dispatch(sc, None, relax=True)
    mon = monitored_from_lp(rel.flow, s.fmax)
    za = ag.solve(sc, mon, gap=1e-5)
    zu = scr.solve(sc, mon, gap=1e-5)
    assert za["ok"] and violates_min_updown(za["u"], sc.u0, s.min_up, s.min_dn) == 0
    assert abs(za["obj"] - zu["obj"]) <= 2e-5 * zu["obj"]      # same optimum
    lp = m.solve_dispatch(sc, za["u"])                         # the disaggregated schedule, priced on the full network
    assert lp.obj <= za["obj"] * (1 + 1e-7)
    assert za["bound"] <= zu["obj"] * (1 + 1e-6)               # a valid bound for the per-unit problem


def test_fixings_become_count_bounds_and_units_out():
    s, sc, m = _setup()
    ag = AggScreenedUC(s, T=m.T)
    gr = ag.base_groups[0]
    sco = OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, g_out=(gr[1],))
    b = ag.model(sco.g_out)
    assert all(gr[1] not in g for g in b["groups"])            # a unit out leaves its group
    rel = m.solve_dispatch(sco, None, relax=True)
    z = ag.solve(sco, monitored_from_lp(rel.flow, s.fmax), fix={(0, gr[0]): 1}, gap=1e-4)
    assert z["u"][:, gr[1]].sum() == 0 and z["u"][0, gr[0]] + sum(z["u"][0, g] for g in gr[2:]) >= 1


def test_probing_is_exact():
    s, sc, m = _setup()
    rel = m.solve_dispatch(sc, None, relax=True)
    L = lp_relaxation_duals(m, sc)
    assert np.allclose(rel.mu, L["mu"], atol=1e-6) and np.allclose(rel.lmp, L["lmp"], atol=1e-6)
    on = hourly_on_cost(s, rel.lmp, rel.mu)
    best, sched, margin = unit_margins(s, sc.u0, on)
    # each unit's DP optimum equals its part of the LP relaxation (the unit's LP is integral): the margins are exact
    x, T, G, S = L["x"], m.T, s.G, s.seg_w.shape[1]
    get = lambda n: x[m.off[n][0]:m.off[n][0] + m.off[n][1] * T].reshape(T, -1)
    lam = rel.lmp[:, s.gbus]
    part = (get("u") * (s.c_nl - lam * s.pmin)).sum(0) + (get("v") * s.c_su).sum(0) \
        + ((s.seg_c[None] - lam[:, :, None]) * get("ps").reshape(T, G, S)).sum((0, 2)) - (rel.mu[:, None] * get("res")).sum(0)
    assert np.allclose(best, part, atol=1e-5 * np.abs(part).max())
    opt = m.solve_uc(sc, mip_gap=1e-6)
    fix, _, _ = probe_fixings(s, sc, rel.lmp, rel.mu, rel.obj, opt.obj)
    assert fix and all(opt.u[t, g] == v for (t, g), v in fix.items())


def test_aggregated_certified_pipeline():
    s, sc, m = _setup()
    ag = AggScreenedUC(s, T=m.T)
    rel = m.solve_dispatch(sc, None, relax=True)
    p = np.clip(rel.u, 0, 1)
    for probe in (False, True):
        u, info = agg_certified_pipeline(ag, sc, p, rel, 0.05, counts=False, probe=probe)
        C = m.solve_dispatch(sc, u).obj
        bound = rel.obj if info["cert"] == "lp" else info["cert_bound"]
        assert (C - bound) / C <= 5e-3 + 1e-9
