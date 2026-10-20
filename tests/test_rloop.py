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


def _two_insts(rts, uc12):
    from otsl.uc import make_scenario
    m, sc = uc12
    return [_inst(m, sc), _inst(m, make_scenario(rts, 140, np.random.default_rng(5), T=12, start=3))]


def test_loopres_write_is_the_exact_decayed_attention(rts, uc12):
    """the streaming write equals h_t = sum_i a_{t,i} c_i, a_{t,i} ~ exp(s_i - lambda (t - i)), exactly"""
    insts = _two_insts(rts, uc12)
    feats = rloop.StaticFeats(rts, insts)
    net = rloop.build_net(rts, feats, dict(hidden=8, loopres=True))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        net.lr_q.copy_(torch.randn(net.lr_q.shape, generator=g))
        net.lr_w.copy_(torch.randn(net.lr_w.shape, generator=g))
        net.lr_rho.fill_(0.3)
    cands = [torch.randn(2, 3, 4, 8, generator=g) for _ in range(6)]
    phis = [torch.zeros(2, rloop.F_I)] + [torch.randn(2, rloop.F_I, generator=g) for _ in range(5)]
    scores = [net._lr_score(0, c, p) for c, p in zip(cands, phis)]
    h, logd = cands[0], scores[0]
    for c, s in zip(cands[1:], scores[1:]):
        h, logd = net._lr_write(0, h, logd, c, s)
    lam = torch.nn.functional.softplus(net.lr_rho[0])
    t = len(cands) - 1
    w = torch.stack([torch.exp(s - lam * (t - i)) for i, s in enumerate(scores)])          # [t+1, B, T, G]
    ref = (w[..., None] * torch.stack(cands)).sum(0) / w.sum(0)[..., None]
    assert torch.allclose(h, ref, atol=1e-5) and torch.allclose(logd, torch.log(w.sum(0)), atol=1e-5)


def test_loopres_without_decay_memory_is_carry_last_and_its_gate_learns(rts, uc12):
    """with beta -> 0 the aggregate is the newest state: the same logits as the model without it, step by step; with
    the default decay the gate's parameters receive gradients (also through the gradient-free first cycle)"""
    insts = _two_insts(rts, uc12)
    feats = rloop.StaticFeats(rts, insts)
    cfg = dict(hidden=16, n_latent=2, n_cycles=2)
    torch.manual_seed(0)
    plain = rloop.build_net(rts, feats, cfg)
    torch.manual_seed(0)
    lr = rloop.build_net(rts, feats, dict(cfg, loopres=True))
    missing, unexpected = lr.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and set(missing) == {"lr_q", "lr_rho", "lr_w"}
    with torch.no_grad():
        lr.lr_rho.fill_(40.0)
        lr.lr_q.normal_()
        lr.lr_w.normal_()
    b = rloop.Batch(feats, insts, "cpu")
    pool = rloop.LPPool(workers=0)
    with torch.no_grad():
        xe = plain.embed_static(b.ux)
        sa, sb, fb = plain.init_state(xe), lr.init_state(xe), rloop.empty_feedback(b)
        for k in range(3):
            la, _, sa = plain.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, sa)
            lb, _, sb = lr.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, sb)
            assert torch.allclose(la, lb, atol=1e-4), k
            res = pool.run([(("t", j), x, torch.sigmoid(la[j]).numpy(), 1.0, "viterbi") for j, x in enumerate(insts)])
            fb = rloop.stack_feedback(b, res, np.full(2, np.inf), rts.gbus)
    assert len(sb) == 6 and len(sa) == 2
    with torch.no_grad():
        lr.lr_rho.fill_(float(np.log(np.expm1(-np.log(0.55)))))
    lr.train()
    xe = lr.embed_static(b.ux)
    _, _, st = lr.step(xe, b.bx, b.lx, b.gate, b.pos, *rloop.empty_feedback(b), lr.init_state(xe))
    st = lr.detach_state(st)
    assert all(not t.requires_grad for t in st)
    lg, _, _ = lr.step(xe, b.bx, b.lx, b.gate, b.pos, *fb, st)
    lg.pow(2).mean().backward()
    for name in ("lr_q", "lr_rho", "lr_w"):
        gr = getattr(lr, name).grad
        assert gr is not None and torch.isfinite(gr).all() and (gr.abs().sum(-1) > 0).all(), name


def test_loopres_trains_in_both_stages(rts, uc12):
    """deep supervision and REINFORCE run with the attention residual (state carried and detached between steps)"""
    insts = _two_insts(rts, uc12)
    feats = rloop.StaticFeats(rts, insts)
    labels = np.stack([(x["u_rel"] > 0.5).astype(np.int8) for x in insts])
    best = np.array([x["c_rel"] * 1.01 for x in insts])
    cfg = dict(rloop.PRESETS["cpu"], hidden=16, n_latent=1, n_cycles=2, epochs=1, batch=2, k_train=2, p_ext=0.5,
               rl_updates=1, rl_batch=2, rl_samples=2, rl_k=2, loopres=True)
    torch.manual_seed(0)
    net = rloop.build_net(rts, feats, cfg)
    pool = rloop.LPPool(workers=0)
    q0 = net.lr_q.detach().clone()
    net, ema = rloop.train_supervised(net, feats, rts, insts, labels, best, cfg, pool, "cpu", log=lambda *_: None)
    net, ema = rloop.train_reinforce(net, feats, rts, insts, labels, best, cfg, pool, "cpu", log=lambda *_: None)
    assert not torch.equal(net.lr_q.detach(), q0)
    steps = rloop.rollout_batch(ema.shadow, feats, rts, insts, 3, pool, "cpu", [("v", 0), ("v", 1)])
    assert len(steps) == 3 and all(np.isfinite(c) for st in steps for c, *_ in st)
