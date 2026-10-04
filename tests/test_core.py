"""Sanity checks: LP/MILP consistency and the differentiable DC physics layer."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otsl.case import load_case  # noqa: E402
from otsl.models import DCPhysics  # noqa: E402
from otsl.opt import DCModel  # noqa: E402


def _setup(name="case30_ieee"):
    c = load_case(name)
    return c, DCModel(c, budget=3, fixed_closed=c.bridges())


def test_lp_matches_milp_with_fixed_topology():
    c, m = _setup()
    o = m.solve_ots(c.pd, mip_gap=1e-6)
    lp = m.solve_lp(c.pd, o.z)
    assert lp.ok and abs(lp.obj - o.obj) / o.obj < 1e-4
    assert m.solve_lp(c.pd).obj >= o.obj - 1e-6          # switching never hurts the optimum


def test_physics_layer_reproduces_lp_flows():
    c, m = _setup()
    z = m.solve_ots(c.pd).z                                   # a feasible switched topology
    lp = m.solve_lp(c.pd, z)
    P = DCPhysics(c)
    pg = torch.as_tensor(lp.pg)[None]
    pd = torch.as_tensor(c.pd)[None]
    zt = torch.as_tensor(z, dtype=torch.float64)[None]
    va = P.angles(pg, pd, zt)
    f = P.flows(va, zt)[0].numpy()
    assert np.allclose(f, lp.flow, atol=1e-5)
    v = P.violations(pg, va, pd, zt)
    assert v["kcl_max"].item() < 1e-4 and v["line_max"].item() < 1e-6


def test_balance_repair_is_exact_and_within_limits():
    c, _ = _setup("case118_ieee")
    P = DCPhysics(c)
    torch.manual_seed(0)
    pg = P.pmin + (P.pmax - P.pmin) * torch.rand(16, c.n_gen, dtype=torch.float64)
    pd = torch.as_tensor(c.pd)[None].repeat(16, 1) * torch.linspace(0.8, 1.2, 16, dtype=torch.float64)[:, None]
    out = P.balance(pg, pd)
    assert torch.allclose(out.sum(1), pd.sum(1), atol=1e-9)
    assert (out >= P.pmin - 1e-12).all() and (out <= P.pmax + 1e-12).all()


def test_dual_sensitivity_ranks_good_switches():
    """-gamma*f (first-order cost change of opening a line) should correlate with the true change and
    one of its top-3 lines should reduce cost. (Its top line can be the congested line itself, whose
    removal is infeasible - the reason LP verification of several candidates is needed.)"""
    c, m = _setup()
    s = m.solve_lp(c.pd)
    est = np.where(~m.fixed_closed, -s.gamma * s.flow, np.inf)
    true = np.full(c.n_line, np.nan)
    for l in np.where(~m.fixed_closed)[0]:
        z = np.ones(c.n_line, np.int8)
        z[l] = 0
        s2 = m.solve_lp(c.pd, z)
        true[l] = s2.obj - s.obj if s2.ok else np.nan
    fin = np.isfinite(true)
    assert np.corrcoef(est[fin], true[fin])[0, 1] > 0.5
    assert np.nanmin(true[np.argsort(est)[:3]]) < 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
