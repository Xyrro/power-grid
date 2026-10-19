"""Tests of otsl.trm: symmetry over identical units, the Viterbi decoder, targets, training and inference paths."""
import itertools

import numpy as np
import pytest
import torch

from otsl import trm
from otsl.uc import violates_min_updown


@pytest.fixture(scope="module")
def small(rts, uc12):
    """six 12-hour scenarios with their LP relaxation (features need it)"""
    from otsl.uc import make_scenario
    m, _ = uc12
    rows = []
    for k in range(6):
        sc = make_scenario(rts, 60 + 7 * k, np.random.default_rng(k), T=12, start=k)
        rel = m.solve_dispatch(sc, None, relax=True)
        rows.append(dict(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, u_rel=rel.u, lmp_rel=rel.lmp,
                         flow_rel=rel.flow, c_rel=rel.obj, y=(rel.u > 0.5).astype(np.int8)))
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def _model(rts, d, **over):
    cfg = trm.TRMConfig(hidden=16, n=2, T=2, n_sup=2, batch=3, steps=4, warmup=1, amp="none", **over)
    feats = trm.TRMFeatures(rts, lp=cfg.lp_feats, sym=cfg.sym).fit(d)
    return cfg, feats, trm.build(cfg, rts, feats)


@pytest.mark.parametrize("unit_mix,time_mix", [("pool", "conv"), ("attn", "attn")])
def test_identical_units_get_identical_outputs(rts, small, unit_mix, time_mix):
    """sym=count: no rank feature, every mixer equivariant -> copies with the same initial status get the same
    probability; permuting copies permutes the output (augmentation is the identity)"""
    cfg, feats, model = _model(rts, small, unit_mix=unit_mix, time_mix=time_mix)
    xt, xb = feats(small)
    P, _ = trm.run_steps(model, xt, xb, 2, bs=6)
    for i, sg in enumerate(trm.subgroups(rts, small["u0"])):
        for mem in sg:
            assert np.allclose(P[i, -1][:, mem], P[i, -1][:, mem[:1]], atol=1e-5)
    # swap two copies with the same initial status in the raw data: outputs swap
    sg = trm.subgroups(rts, small["u0"])[0][0]
    a, b = sg[0], sg[1]
    perm = np.arange(rts.G)
    perm[a], perm[b] = b, a
    d2 = dict(small)
    d2["u0"] = small["u0"][:, perm]
    d2["u_rel"] = small["u_rel"][:, :, perm]
    xt2, xb2 = feats(d2)
    P2, _ = trm.run_steps(model, xt2, xb2, 2, bs=6)
    assert np.allclose(P2[0, -1][:, perm], P[0, -1], atol=1e-5)


def test_rank_mode_breaks_symmetry(rts, small):
    cfg, feats, model = _model(rts, small, sym="rank")
    xt, xb = feats(small)
    assert xt.shape[-1] == feats.dims[0]
    P, _ = trm.run_steps(model, xt, xb, 1, bs=6)
    mem = trm.subgroups(rts, small["u0"])[0][0]
    assert not np.allclose(P[0, 0][:, mem[0]], P[0, 0][:, mem[1]])


def test_viterbi_matches_brute_force(rts):
    from otsl.uc import violates_min_updown as viol
    rng = np.random.default_rng(0)
    T = 7
    for U, D, u0 in [(3, 2, 0), (2, 4, 1), (1, 1, 0), (4, 3, 1), (8, 8, 0)]:
        for _ in range(5):
            w = rng.normal(size=T)
            u = trm.viterbi(w, u0, U, D)
            best = -np.inf
            for bits in itertools.product((0, 1), repeat=T):
                b = np.array(bits)[:, None]
                if viol(b, np.array([u0]), np.array([U]), np.array([D])) == 0:
                    best = max(best, float((b[:, 0] * w).sum()))
            assert viol(u[:, None], np.array([u0]), np.array([U]), np.array([D])) == 0
            assert np.isclose((u * w).sum(), best)


def test_allocation_and_decode(rts, small):
    """water-filling gives copy j clip(count - j, 0, 1); decoded schedules respect min up/down"""
    sg = trm.subgroups(rts, small["u0"])[0]
    p = np.full((12, rts.G), 0.1)
    mem = sg[0]
    p[:, mem] = 1.6 / len(mem)
    q = trm.allocate(p, sg)
    assert np.allclose(q[:, mem[0]], 1.0) and np.allclose(q[:, mem[1]], 0.6)
    assert np.allclose(q[:, mem].sum(1), 1.6)
    for how in ("viterbi", "threshold"):
        u = trm.decode(small["u_rel"][0], small, 0, rts, 0.5, how, sg)
        assert violates_min_updown(u, small["u0"][0], rts.min_up, rts.min_dn) == 0


def test_count_targets_and_wrong_count(rts, small):
    y = small["y"].copy()
    t = trm.targets(rts, small, y, "count")
    S = trm.subgroup_mean_matrix(rts, small["u0"])
    for i, sg in enumerate(trm.subgroups(rts, small["u0"])):
        for mem in sg:
            assert np.allclose(t[i][:, mem].sum(1), y[i][:, mem].sum(1))         # counts preserved
    # a prediction equal to a permuted label has zero wrong decisions
    i, mem = 0, trm.subgroups(rts, small["u0"])[0][0]
    yp = y[i].astype(float).copy()
    yp[:, mem] = yp[:, mem[::-1]]
    lg = torch.logit(torch.as_tensor(np.clip(yp, 1e-4, 1 - 1e-4))[None], eps=1e-6).float()
    w = trm.wrong_count(lg, torch.as_tensor(t[i:i + 1]), torch.as_tensor(S[i:i + 1]))
    assert float(w) == 0.0


@pytest.mark.parametrize("mode", ["trm", "oneshot", "deep"])
def test_training_smoke_and_resume(rts, small, mode, tmp_path):
    cfg = trm.TRMConfig(hidden=16, n=2, T=2, n_sup=2, batch=3, steps=3, warmup=1, amp="bf16", mode=mode)
    d = dict(small)
    va = dict(small, u=small["y"])
    ck = str(tmp_path / "m.pt")
    model, ema, feats, log = trm.train(cfg, rts, d, va, ck, log_every=1, val_every=3, logf=lambda *_: None)
    assert len(log["train"]) == 3 and np.isfinite(log["train"][-1]["bce"])
    # an interrupted 5-step run (e.g. a wall-time limit at step 3): same checkpoint, configured length 5
    c = torch.load(ck, weights_only=False)
    c["cfg"]["steps"] = 5
    torch.save(c, ck)
    cfg.steps = 5
    model, ema, feats, log = trm.train(cfg, rts, d, va, ck, log_every=1, val_every=5, logf=lambda *_: None)
    assert len(log["train"]) == 5
    # resuming under a CHANGED configuration (here a longer schedule) is refused: it would restart the learning-rate
    # schedule mid-run and overwrite the checkpoint earlier results came from
    cfg.steps = 7
    with pytest.raises(SystemExit, match="refusing to resume"):
        trm.train(cfg, rts, d, va, ck, log_every=1, val_every=5, logf=lambda *_: None)
    cfg.steps = 5
    m2, f2, c2, _ = trm.load_model(ck, rts)
    xt, xb = f2(d)
    P, Q = trm.run_steps(m2, xt, xb, 3, bs=2)
    assert P.shape == (6, 3, 12, rts.G) and np.isfinite(P).all()


def test_lp_init_and_cost_weights(rts, small):
    cfg, feats, model = _model(rts, small, y0="lp")
    xt, xb = feats(small)
    y, z = model.init_state(6, 12, rts.G, xt)
    assert float(y[0, 0].std(0).max()) > 1e-3          # differs across units with different u_rel
    g = np.random.default_rng(0).normal(size=small["y"].shape) * 1e3
    w = trm.cost_weights(g, small["y"], small["c_rel"], 1.0)
    assert (w >= 1).all() and w.shape == small["y"].shape


def test_micro_batch_matches_full_batch(rts, small, tmp_path):
    """gradient accumulation over slices of the carry batch gives the same parameters as the full batch"""
    d = dict(small)
    va = dict(small, u=small["y"])
    states = []
    for mb in (0, 1):
        cfg = trm.TRMConfig(hidden=16, n=2, T=2, n_sup=2, batch=3, steps=2, warmup=1, amp="none", micro_batch=mb,
                            ema=0.5)
        model, ema, feats, log = trm.train(cfg, rts, d, va, str(tmp_path / f"mb{mb}.pt"), log_every=1, val_every=100,
                                           logf=lambda *_: None)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    for k in states[0]:
        assert torch.allclose(states[0][k], states[1][k], atol=1e-5), k


def test_blend_limits_and_decoder(rts, small):
    p = np.random.default_rng(0).uniform(0.05, 0.95, size=(12, rts.G))
    prior = small["u_rel"][0]
    assert np.allclose(trm.blend(p, prior, 1.0), p, atol=1e-9)
    b0 = trm.blend(p, prior, 0.0)
    assert np.allclose(b0, trm.blend(1 - p, prior, 0.0)) and np.allclose(b0, np.clip(prior, 0.02, 0.98), atol=1e-3)
    sg = trm.subgroups(rts, small["u0"])[0]
    u = trm.decode(p, small, 0, rts, 0.5, "blend0.5", sg)
    assert violates_min_updown(u, small["u0"][0], rts.min_up, rts.min_dn) == 0


def test_lp_repair_never_worse(rts, uc12, small):
    """the LP-guided repair keeps a served schedule and never returns a costlier one"""
    from otsl.uc import UCScenario
    m, _ = uc12
    i = 0
    sc = UCScenario(load=small["load"][i], avail=small["avail"][i], u0=small["u0"][i], sr=small["sr"][i])
    u = trm.decode(small["u_rel"][i], small, i, rts, 0.5, "viterbi")
    u_bad = u.copy()
    u_bad[:, np.argsort(-rts.pmax)[:6]] = 0                     # drop the six largest units: shortfall
    from otsl.uc import repair_min_updown
    u_bad = repair_min_updown(u_bad, sc.u0, rts.min_up, rts.min_dn)
    s0 = m.solve_dispatch(sc, u_bad)
    u2, s2, n, _ = trm.lp_repair(m, sc, u_bad, rts, s0)
    assert s2.obj <= s0.obj + 1e-6
    assert violates_min_updown(u2, sc.u0, rts.min_up, rts.min_dn) == 0
    ok = m.solve_dispatch(sc, u)
    if ok.shed < 1e-6 and ok.short < 1e-6:
        u3, s3, n3, _ = trm.lp_repair(m, sc, u, rts, ok)
        assert n3 == 0 and np.array_equal(u3, u)
