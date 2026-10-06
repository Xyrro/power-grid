"""Learning to Fix (otsl.ltfx): fixing rule and thresholds, the Appendix A master with OR-cuts, Algorithm 1 + 2 on a
toy oracle, and the check / relaxation MILPs on a small UC instance."""
import numpy as np
import pytest

from otsl.ltfx import (FIX_TOL, InstanceProblems, LtFTuner, Master, constant_thresholds, fix_dict, fix_masks,
                       quantile_bins, worst_case_thresholds)


# ----------------------------------------------------------------------------- fixing rule (eq. 5) and thresholds
def test_fix_masks_strict_generator_specific_thresholds():
    pi = np.array([[0.0, 0.3, 0.5, 0.7, 1.0],
                   [0.2, 0.2, 0.2, 0.2, 0.2]])
    lo = np.array([0.1, 0.3, 0.5, 0.5, 0.0])
    hi = np.array([0.9, 0.3, 0.5, 0.6, 1.0])
    off, on = fix_masks(pi, lo, hi)
    assert off.tolist() == [[True, False, False, False, False], [False, True, True, True, False]]
    assert on.tolist() == [[False, False, False, True, False], [False, False, False, False, False]]
    # a probability exactly on a threshold stays free (FIX_TOL), lo = 0 never fixes OFF, hi = 1 never fixes ON
    assert not off[:, 4].any() and not on[:, 4].any()
    off2, on2 = fix_masks(np.array([[0.3 - 2 * FIX_TOL]]), np.array([0.3]), np.array([0.3]))
    assert off2[0, 0] and not on2[0, 0]


def test_fix_dict_matches_masks():
    rng = np.random.default_rng(0)
    pi = rng.random((4, 6))
    lo, hi = np.full(6, 0.3), np.full(6, 0.6)
    fx = fix_dict(pi, lo, hi)
    off, on = fix_masks(pi, lo, hi)
    assert set(k for k, v in fx.items() if v == 0) == set(zip(*map(lambda a: a.tolist(), np.where(off))))
    assert set(k for k, v in fx.items() if v == 1) == set(zip(*map(lambda a: a.tolist(), np.where(on))))
    assert all(isinstance(t, int) and isinstance(g, int) for t, g in fx)


def test_constant_thresholds():
    lo, hi = constant_thresholds(3, 0.05)
    assert np.allclose(lo, 0.05) and np.allclose(hi, 0.95)


def test_worst_case_thresholds_never_fix_a_validation_decision_wrongly():
    rng = np.random.default_rng(1)
    n, T, G = 30, 4, 5
    u = (rng.random((n, T, G)) < 0.4).astype(np.int8)
    pi = np.clip(u * 0.7 + rng.normal(0.15, 0.2, u.shape), 0, 1)
    u[:, :, 3] = 0                                          # generator 3 never on, generator 4 always on
    u[:, :, 4] = 1
    lo, hi = worst_case_thresholds(pi, u)
    off, on = fix_masks(pi, lo, hi)
    assert not (off & (u == 1)).any() and not (on & (u == 0)).any()
    assert np.all(lo <= hi + 1e-12)
    # separable generators get lo = hi = midpoint: everything of the generator is fixed
    for g in range(G):
        p_on, p_off = pi[..., g][u[..., g] == 1], pi[..., g][u[..., g] == 0]
        if len(p_on) and len(p_off) and p_on.min() > p_off.max():
            assert np.isclose(lo[g], (p_on.min() + p_off.max()) / 2) and np.isclose(lo[g], hi[g])
    # never on: lo = 1 before the midpoint rule, so the interval is the midpoint of 1 and the largest OFF probability
    assert np.isclose(lo[3], (1 + pi[..., 3].max()) / 2)


# ----------------------------------------------------------------------------- Appendix A master
def _pi_val(seed=2, n=12, T=2, G=3):
    return np.random.default_rng(seed).random((n, T, G))


def test_quantile_bins_shapes_floor_and_split():
    pi = _pi_val()
    pi[:, 0, 0] = 0.0                                       # a point mass: zero-width bins are floored
    W, low = quantile_bins(pi, Q=5)
    assert W.shape == (2, 3, 5) and low.shape == (2, 3, 5)
    assert np.all(W >= 1e-6) and np.allclose(W[0, 0], 1e-6)
    edges = np.quantile(pi, np.linspace(0, 1, 6), axis=0)
    assert np.array_equal(low, (edges[:-1] <= 0.5).transpose(1, 2, 0))


def test_master_without_cuts_collapses_intervals():
    m = Master(_pi_val(), Q=5)
    lo, hi, info = m.solve()
    assert np.all(lo <= hi + 1e-9) and np.allclose(lo, hi, atol=1e-7)
    assert info["n_bin"] == 0 and 0.0 < info["approx_fixed_share"] <= 1.0 + 1e-9


def test_master_respects_or_cuts_and_cuts_only_shrink_the_fixed_mass():
    m = Master(_pi_val(), Q=5)
    _, _, i0 = m.solve()
    m.add_cut([({0: 0.2}, {})])                             # lo_0 <= 0.2
    lo, hi, i1 = m.solve()
    assert lo[0] <= 0.2 + 1e-7 and i1["n_bin"] == 1
    # OR of two conjunctions: (lo_1 <= 0.1) or (lo_2 <= 0.05 and hi_2 >= 0.95)
    m.add_cut([({1: 0.1}, {}), ({2: 0.05}, {2: 0.95})])
    lo, hi, i2 = m.solve()
    assert lo[0] <= 0.2 + 1e-7
    assert lo[1] <= 0.1 + 1e-7 or (lo[2] <= 0.05 + 1e-7 and hi[2] >= 0.95 - 1e-7)
    assert i2["n_bin"] == 3
    # minimisation (of minus the fixed mass): every cut can only worsen the optimum (MIP gap 1e-5)
    assert i1["obj"] >= i0["obj"] - 1e-4 * abs(i0["obj"]) and i2["obj"] >= i1["obj"] - 1e-4 * abs(i1["obj"])


# ----------------------------------------------------------------------------- Algorithm 1 + 2 on a toy oracle
class _Sys:
    G = 3

    @staticmethod
    def identical_groups():
        return []


class ToyTuner(LtFTuner):
    """Algorithm 1 with a solver-free oracle: instance i fails iff it fixes one of its 'harmful' decisions; ADD CUTS
    returns exactly those decisions as one release set (a minimal release set)."""

    def __init__(self, pi, harmful, **kw):
        n, T, G = pi.shape
        d = dict(u=np.zeros((n, T, G), np.int8), u0=np.zeros((n, G), np.int8))
        super().__init__(None, _Sys(), d, pi, 0.01, Q=5, log=lambda s: None, **kw)
        self.harmful = harmful

    def check(self, i, off, on):
        bad = [(t, g) for t, g, v in self.harmful[i] if (off if v == 0 else on)[t, g]]
        return (not bad), dict(status="toy", u=None)

    def add_cuts(self, i, off, on):
        R = [(t, g) for t, g, v in self.harmful[i] if (off if v == 0 else on)[t, g]]
        alo, bhi = {}, {}
        for t, g in R:
            if off[t, g]:
                alo[g] = min(alo.get(g, 1.0), float(self.pi[i][t, g]))
            else:
                bhi[g] = max(bhi.get(g, 0.0), float(self.pi[i][t, g]))
        return [(alo, bhi)], [R], []


def test_algorithm1_terminates_with_thresholds_that_release_every_harmful_fixing():
    pi = _pi_val(seed=3, n=6, T=2, G=3)
    harmful = {i: [] for i in range(6)}
    harmful[1] = [(0, 0, 0 if pi[1, 0, 0] < 0.5 else 1)]
    harmful[4] = [(1, 2, 0 if pi[4, 1, 2] < 0.5 else 1), (0, 1, 0 if pi[4, 0, 1] < 0.5 else 1)]
    tu = ToyTuner(pi, harmful)
    lo, hi = tu.run()
    assert tu.history[-1]["result"] == "all instances feasible"
    off, on = fix_masks(pi, lo, hi)
    for i, H in harmful.items():
        for t, g, v in H:
            assert not (off if v == 0 else on)[i, t, g]
    assert len(tu.master.cuts) >= 2 and (off | on).mean() > 0.3


def test_check_cache_uses_monotonicity():
    pi = _pi_val(seed=4, n=1, T=2, G=3)
    tu = LtFTuner(None, _Sys(), dict(u=np.zeros((1, 2, 3), np.int8), u0=np.zeros((1, 3), np.int8)), pi, 0.01, Q=5,
                  log=lambda s: None)
    big = np.ones((2, 3), bool)
    small = np.zeros((2, 3), bool)
    small[0, 0] = True
    tu.passed[0].append((big, np.zeros((2, 3), bool), np.zeros((2, 3), np.int8)))
    ok, info = tu.check(0, small, np.zeros((2, 3), bool))           # subset of a passed set: passes from the cache
    assert ok and info["cached"] and tu.stats["n_check_cached"] == 1
    tu.failed[0].append((small, np.zeros((2, 3), bool)))
    ok, info = tu.check(0, big, np.ones((2, 3), bool))              # superset of a failed set: fails from the cache
    assert not ok and info["cached"]


# ----------------------------------------------------------------------------- check / relaxation MILPs (3-hour UC)
@pytest.fixture(scope="module")
def inst(uc3, uc3_milp):
    m, sc = uc3
    sol, _ = uc3_milp
    return InstanceProblems(m, sc, sol["obj"], eps=0.01), sol


def test_check_passes_without_fixings_and_at_the_optimum(inst):
    P, sol = inst
    T, G = P.T, P.G
    ok, info = P.check(np.zeros((T, G), bool), np.zeros((T, G), bool))
    assert ok and info["obj"] <= P.cap * (1 + 1e-7)
    u = sol["u"].astype(bool)
    ok, info = P.check(~u, u)                                        # every decision fixed to the optimum
    assert ok and abs(info["obj"] - P.c_star) <= 1e-6 * P.c_star


def test_relaxation_finds_a_release_set_that_repairs_a_failing_fixing(inst):
    P, sol = inst
    T, G = P.T, P.G
    off = np.zeros((T, G), bool)
    off[0] = sol["u"][0] == 1                                        # the units running in hour 1 fixed off: shedding
    on = np.zeros((T, G), bool)
    ok, _ = P.check(off, on)
    assert not ok
    x0, c0 = P.label_start(sol["u"])
    assert c0 <= P.cap
    R, info = P.relaxation(off, on, [], time_limit=30.0, start_x=x0)
    assert R and info["cost"] <= P.cap * (1 + 1e-6)
    assert all(off[t, g] for t, g in R)
    off2 = off.copy()
    for t, g in R:
        off2[t, g] = False
    ok, _ = P.check(off2, on)
    assert ok
    # a no-good constraint excludes the same release set
    R2, _ = P.relaxation(off, on, [R], time_limit=30.0)
    assert R2 is None or set(R2) != set(R)


# ----------------------------------------------------------------------------- Table II features and the kNN (eq. 13)
def test_knn_probabilities_inverse_distance_weighting(rts):
    from otsl.ltfx import KNNProb, table2_features
    from otsl.uc import make_scenario
    from otsl.ucml import canonical_labels
    rng = np.random.default_rng(5)
    scs = [make_scenario(rts, int(day), rng, T=4, start=6) for day in (10, 50, 90, 130, 170, 210)]
    n, T, G = len(scs), 4, rts.G
    d = dict(load=np.array([s.load for s in scs]), avail=np.array([s.avail for s in scs]),
             u0=np.array([s.u0 for s in scs]), sr=np.array([s.sr for s in scs]), start=np.full(n, 6),
             u=(rng.random((n, T, G)) < 0.3).astype(np.int8))
    Z = table2_features(rts, d)
    assert Z.shape == (n, 17 * T + G) and np.all(np.isfinite(Z))
    assert np.allclose(Z[:, -G:], np.where(d["u0"] > 0, 1.0, -1.0))
    y = canonical_labels(rts, d["u"], d["u0"])
    assert np.allclose(KNNProb(rts, d, k=1).predict(d), y)                  # each instance is its own neighbour
    p = KNNProb(rts, d, k=n).predict(d)
    assert np.all((p >= 0) & (p <= 1))
    q = {k: v[:1] for k, v in d.items()}
    q["load"] = q["load"] * 1.01                                            # a new instance, not in the training set
    kn = KNNProb(rts, d, k=3)
    Zq = (table2_features(rts, q) - kn.mu) / kn.sd
    dist = np.sqrt(((kn.X - Zq) ** 2).sum(1))
    nb = np.argsort(dist)[:3]
    w = 1 / dist[nb]
    assert np.allclose(kn.predict(q)[0], np.einsum("k,ktg->tg", w / w.sum(), y[nb]))
