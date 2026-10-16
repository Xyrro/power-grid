"""rloop (arm B): exact LP feedback (highspy path against UCModel), Viterbi decoding against the Hamming repair,
count targets, and the length / topology-agnostic network."""
import numpy as np
import pytest
import torch

from otsl import rloop
from otsl.ood import OODScenario
from otsl.uc import repair_min_updown, violates_min_updown


def _inst(m, sc):
    rel = m.solve_dispatch(sc, None, relax=True)
    return dict(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, u_rel=rel.u, lmp_rel=rel.lmp, flow_rel=rel.flow,
                c_rel=float(rel.obj))


def test_fastlp_matches_ucmodel_cost_prices_and_gradient(rts, uc12):
    m, sc = uc12
    u = (m.solve_dispatch(sc, None, relax=True).u > 0.5).astype(np.int8)
    ref = m.solve_dispatch(sc, u)
    c_ref, g_ref = m.dispatch_gradient(sc, u)
    lp = rloop.FastLP(rts, 12)
    for _ in range(2):                                  # cold, then warm-started
        sol = lp.solve(OODScenario(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr), u)
        assert abs(sol["obj"] - ref.obj) <= 1e-6 * abs(ref.obj)
        assert abs(sol["obj"] - c_ref) <= 1e-6 * abs(c_ref)
    assert np.allclose(sol["grad"], g_ref, atol=1e-3 * np.abs(g_ref).max())


def test_viterbi_is_feasible_hamming_nearest_and_most_likely(rts):
    rng = np.random.default_rng(0)
    for _ in range(10):
        u0 = (rng.random(rts.G) < 0.4).astype(np.int8)
        u = (rng.random((12, rts.G)) < 0.4).astype(np.int8)
        v = rloop.viterbi_minud(u.astype(float), u0, rts.min_up, rts.min_dn)
        h = repair_min_updown(u, u0, rts.min_up, rts.min_dn)
        assert violates_min_updown(v, u0, rts.min_up, rts.min_dn) == 0
        assert ((v != u).sum(0) == (h != u).sum(0)).all()
        p = np.clip(rng.random((12, rts.G)), 0.01, 0.99)
        w = rloop.viterbi_minud(p, u0, rts.min_up, rts.min_dn)
        hr = repair_min_updown((p > 0.5).astype(np.int8), u0, rts.min_up, rts.min_dn)
        ll = lambda s: np.sum(np.where(s > 0, np.log(p), np.log(1 - p)))
        assert violates_min_updown(w, u0, rts.min_up, rts.min_dn) == 0 and ll(w) >= ll(hr) - 1e-9


def test_count_targets_keep_counts_and_are_nested(rts):
    gi = rloop.GroupInfo(rts)
    assert gi.NG == 24 + (rts.G - sum(len(g) for g in rts.identical_groups()))
    rng = np.random.default_rng(1)
    y = (rng.random((12, rts.G)) < 0.5).astype(np.int8)
    u0 = (rng.random(rts.G) < 0.5).astype(np.int8)
    st = gi.stacked(y, u0)
    assert (gi.counts(st) == gi.counts(y)).all()
    pos = gi.positions(u0)
    for gr in gi.groups:
        order = sorted(gr, key=lambda g: pos[g])
        assert all((st[:, a] >= st[:, b]).all() for a, b in zip(order, order[1:]))   # copy j on => copy j-1 on
        assert all(u0[a] >= u0[b] for a, b in zip(order, order[1:]))                 # initially-on copies first


@pytest.mark.parametrize("T", [12, 24])
def test_network_runs_on_any_horizon_and_with_lines_out(rts, T):
    from otsl.uc import UCModel, make_scenario
    m = UCModel(rts, T=T)
    sc = make_scenario(rts, 100, np.random.default_rng(2), T=T, start=0)
    inst = _inst(m, sc)
    feats = rloop.StaticFeats(rts, [inst])
    torch.manual_seed(0)
    net = rloop.build_net(rts, feats, dict(hidden=16, n_latent=1))
    out = (inst, dict(inst, l_out=np.array([5, 11])))
    b = rloop.Batch(feats, list(out), "cpu")
    assert b.gate[1, 5] == 0 and b.gate[0, 5] == 1
    xe = net.embed_static(b.ux)
    lg, hf, st = net.step(xe, b.bx, b.lx, b.gate, b.pos, *rloop.empty_feedback(b), net.init_state(xe))
    assert lg.shape == (2, T, rts.G) and torch.isfinite(lg).all()
    pool = rloop.LPPool(workers=0)
    res = pool.run([(("t", j), x, torch.sigmoid(lg[j]).detach().numpy(), 1.0, "viterbi") for j, x in enumerate(out)])
    fb = rloop.stack_feedback(b, res, np.full(2, np.inf), rts.gbus)
    lg2, _, _ = net.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, st)
    assert torch.isfinite(lg2).all()
    for r in res:
        assert violates_min_updown(r["s"], inst["u0"], rts.min_up, rts.min_dn) == 0
