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
    cfg.steps = 5
    model, ema, feats, log = trm.train(cfg, rts, d, va, ck, log_every=1, val_every=5, logf=lambda *_: None)
    assert len(log["train"]) == 5
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
