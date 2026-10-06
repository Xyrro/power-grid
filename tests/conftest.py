"""Shared fixtures: the RTS-GMLC system, small UC models and scenarios (built once per test session).
Everything runs single-threaded (HiGHS threads = 1 where the code sets it, torch via OTSL_THREADS)."""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("OTSL_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="session")
def rts():
    from otsl.uc import load_rts_gmlc
    return load_rts_gmlc()


@pytest.fixture(scope="session")
def uc3(rts):
    """3-hour network UC model and one scenario (fast MILPs)"""
    from otsl.uc import UCModel, make_scenario
    m = UCModel(rts, T=3, network=True)
    sc = make_scenario(rts, 200, np.random.default_rng(0), T=3, start=14)
    return m, sc


@pytest.fixture(scope="session")
def uc12(rts):
    """12-hour network UC model and one scenario (LP relaxation and dispatch LPs only)"""
    from otsl.uc import UCModel, make_scenario
    m = UCModel(rts, T=12, network=True)
    sc = make_scenario(rts, 100, np.random.default_rng(1), T=12, start=4)
    return m, sc


@pytest.fixture(scope="session")
def uc3_milp(uc3):
    """the full MILP of the 3-hour scenario through the highspy path, with its LP relaxation"""
    from otsl.b3 import solve_milp_hs
    m, sc = uc3
    sol = solve_milp_hs(m, sc, time_limit=60, mip_gap=1e-4, threads=1)
    rel = m.solve_dispatch(sc, None, relax=True)
    return sol, rel
