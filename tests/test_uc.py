"""Sanity checks for the unit-commitment model: MILP / dispatch-LP consistency, exact LP sensitivities,
the differentiable dispatch physics (Model 2) and the schedule repairs."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.uc import (UCModel, adequacy_repair, load_rts_gmlc, make_scenario, repair_min_updown,  # noqa: E402
                     violates_min_updown)
from otsl.ucml import UCPhysics  # noqa: E402

SYS = load_rts_gmlc()


def _solved(T=1, day=40, start=8, seed=0):
    sc = make_scenario(SYS, day, np.random.default_rng(seed), T=T, start=start)
    m = UCModel(SYS, T=T, network=True)
    return m, sc, m.solve_uc(sc, time_limit=60, mip_gap=1e-6)


def test_dispatch_lp_reproduces_milp_and_relaxation_bounds_it():
    m, sc, sol = _solved()
    d = m.solve_dispatch(sc, sol.u)
    assert abs(d.obj - sol.obj) <= 1e-5 * abs(sol.obj)
    assert m.solve_dispatch(sc, None, relax=True).obj <= sol.obj * (1 + 1e-7)


def test_dispatch_gradient_matches_finite_differences():
    m, sc, sol = _solved()
    u = np.clip(0.6 * sol.u + 0.2, 0, 1).astype(float)              # fractional, away from kinks
    c0, g = m.dispatch_gradient(sc, u)
    assert abs(c0 - m.solve_dispatch(sc, u).obj) <= 1e-6 * abs(c0)
    eps = 1e-5
    for k in np.random.default_rng(0).choice(SYS.G, 6, replace=False):
        up, dn = u.copy(), u.copy()
        up[0, k] += eps; dn[0, k] -= eps
        fd = (m.dispatch_gradient(sc, up)[0] - m.dispatch_gradient(sc, dn)[0]) / (2 * eps)
        assert abs(fd - g[0, k]) <= 1e-3 * max(1.0, abs(fd)), (k, fd, g[0, k])


def test_physics_layer_reproduces_dispatch_flows_and_cost():
    m, sc, sol = _solved()
    d = m.solve_dispatch(sc, sol.u)
    P = UCPhysics(SYS)
    t = lambda a: torch.as_tensor(np.asarray(a, float))[None]
    va = P.angles(t(d.p), t(d.r), t(sc.load))
    assert torch.allclose(P.flows(va)[0], torch.as_tensor(d.flow), atol=1e-5)
    if d.shed < 1e-9 and d.short < 1e-9:
        c = P.cost(t(d.p), t(sol.u), torch.as_tensor(sc.u0, dtype=torch.float64)[None], torch.zeros(1, 1))
        assert abs(c.item() - d.obj) <= 1e-6 * abs(d.obj)


def test_repairs_return_feasible_adequate_schedules():
    rng = np.random.default_rng(1)
    sc = make_scenario(SYS, 100, rng, T=12, start=4)
    for _ in range(5):
        u = (rng.random((12, SYS.G)) < 0.4).astype(np.int8)
        assert violates_min_updown(repair_min_updown(u, sc.u0, SYS.min_up, SYS.min_dn), sc.u0, SYS.min_up, SYS.min_dn) == 0
        v = adequacy_repair(u, sc.load, sc.avail, sc.sr, SYS)
        need = sc.load.sum(1) - sc.avail.sum(1) + sc.sr
        assert np.all((v * SYS.pmax).sum(1) >= need - 1e-9)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
