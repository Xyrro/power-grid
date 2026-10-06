"""Better Model 1 probabilities for the 12-hour UC benchmark (B2): data scaling without full MILPs, deep ensembles
with a disagreement filter, temporal mixing across hours, and probability-quality metrics for fixing.

* Scenarios without MILPs. `scenario_jobs` draws (day, start, seed) exactly as otsl.ucdata.generate does (same
  calendar split, same window and noise distributions); `lf_instance` solves only the LP relaxation (the GNN's input
  features) and builds the label-free label of V4 (rounded relaxation -> adequacy repair -> min up/down repair),
  priced once with the dispatch LP.
* Teacher-polished labels (one round of self-training, W2, with a better teacher). The teacher is the mean of the
  MILP-label BCE GNNs; its most confident decisions (asymmetric ranking: OFF errors x 10) are fixed to its rounded
  prediction, the adequacy guard and the min up/down row release are applied, and a reduced MILP with a short time
  limit fills in the rest. The schedule is priced by the dispatch LP and kept if cheaper than the label-free label.
  No full MILP is solved.
* CommitGNNT: the RACLearn-style GNN of otsl.ucml plus a residual temporal head, a dilated 1-D convolution over the
  hours of each unit on per-hour inputs (relaxed commitment, net load, bus load / renewables / relaxed price) and the
  unit's GNN embedding. Zero-initialised output: it starts as the plain GNN.
* Metrics: log-loss, Brier score, ROC AUC, top-label calibration error, and the share of unit-hours that can be fixed
  at a given precision (pooled confidence ranking), against MILP labels aligned to the prediction inside groups of
  identical units (a swapped copy is not an error).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from . import models as _models  # noqa: F401  (keeps torch single-threaded)
from .combo import adequacy_guard
from .fixpolicy import align_to_prediction, fix_from_ranking, release_conflicting_rows
from .models import EdgeGNN, mlp
from .uc import UCScenario, adequacy_repair, make_scenario, repair_min_updown
from .ucml import UCModel1

RELAX_KEYS = ("load", "avail", "u0", "sr", "day", "start", "u_rel", "lmp_rel", "c_rel", "flow_rel")


# ============================================================================ scenarios without MILPs
def scenario_jobs(n, seed, days, T=12):
    """(day, start, seed) per instance, drawn exactly as otsl.ucdata.generate draws them"""
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(n):
        day = int(rng.choice(days))
        start = int(rng.integers(0, 24 - T + 1))
        jobs.append((day, start, int(rng.integers(1 << 31))))
    return jobs


def lf_label(u_rel, sc, sysm):
    """label-free label (V4): rounded LP relaxation -> adequacy repair -> min up/down repair"""
    u = adequacy_repair((u_rel > 0.5).astype(np.int8), sc.load, sc.avail, sc.sr, sysm)
    return repair_min_updown(u, sc.u0, sysm.min_up, sysm.min_dn)


def lf_instance(m, sysm, job, T=12):
    """one scenario: LP relaxation (model input) + label-free label priced by the dispatch LP"""
    day, start, seed = job
    t0 = time.time()
    sc = make_scenario(sysm, day, np.random.default_rng(seed), T=T, start=start)
    rel = m.solve_dispatch(sc, None, relax=True)
    t_rel = time.time() - t0
    y = lf_label(rel.u, sc, sysm)
    lp = m.solve_dispatch(sc, y)
    return dict(load=sc.load, avail=sc.avail, u0=sc.u0, sr=sc.sr, day=day, start=start, u_rel=rel.u,
                lmp_rel=rel.lmp, c_rel=rel.obj, flow_rel=rel.flow, y_lf=y, c_lf=lp.obj, shed_lf=lp.shed,
                short_lf=lp.short, t_rel=t_rel, t_lf=time.time() - t0)


def scenario(d, i):
    return UCScenario(load=d["load"][i], avail=d["avail"][i], u0=d["u0"][i], sr=d["sr"][i])


# ============================================================================ teacher-polished labels
def teacher_fixings(p, sc, sysm, ratio, rank="asym"):
    """fixings of the reduced MILP that polishes a label: the ratio share of the teacher's most confident decisions
    (asym: OFF errors x 10, as in V3), fixed to round(p), then the adequacy guard and the min up/down row release"""
    yhat = (p > 0.5).astype(int)
    err = np.minimum(p, 1 - p)
    if rank == "asym":
        err = err * np.where(p < 0.5, 10.0, 1.0)
    fix = fix_from_ranking(err, yhat, ratio)
    fix, _ = adequacy_guard(fix, sc.load, sc.avail, sc.sr, sysm)
    fix, _ = release_conflicting_rows(fix, sysm, sc.u0)
    return fix


def polish(m, sysm, sc, p, y_lf, c_lf, ratio=0.9, time_limit=5.0, gap=1e-3, rank="asym", lp_guard_fn=None):
    """one reduced MILP around the teacher's confident decisions; returns the label (the reduced-MILP schedule if the
    dispatch LP prices it below the label-free label, else the label-free label) and a record"""
    t0 = time.time()
    fix = teacher_fixings(p, sc, sysm, ratio, rank)
    if lp_guard_fn is not None:
        fix = lp_guard_fn(fix)
    sol = m.solve_uc(sc, time_limit=time_limit, mip_gap=gap, z_fix=fix or None)
    rec = dict(n_fixed=len(fix), milp_s=sol.time, status=sol.status)
    if sol.u is None:
        rec.update(cost=np.inf, used="lf", total_s=time.time() - t0)
        return y_lf, rec
    lp = m.solve_dispatch(sc, sol.u)
    rec.update(cost=lp.obj, shed=lp.shed, short=lp.short, milp_obj=sol.obj, mip_gap=sol.gap)
    if lp.obj < c_lf:
        rec.update(used="milp", total_s=time.time() - t0)
        return sol.u.astype(np.int8), rec
    rec.update(used="lf", total_s=time.time() - t0)
    return y_lf, rec


# ============================================================================ temporal mixing
def feature_layout(feat, T):
    """column indices of the per-hour inputs in the featurizer's bus / unit features (relax=True, sym=True)"""
    assert feat.relax and feat.sym
    ns = feat.g_static.shape[1]
    net = ns + 1 + np.arange(T)
    urel = ns + 1 + T + 2 + np.arange(T)
    bload, bren, blmp = np.arange(T), T + np.arange(T), 2 * T + 1 + np.arange(T)
    assert urel[-1] == feat.gen_dim - 1 and blmp[-1] == feat.bus_dim - 1
    return np.stack([net, urel]), np.stack([bload, bren, blmp])


class CommitGNNT(nn.Module):
    """CommitGNN (unit encoder -> bus GNN -> per-unit decoder of T logits) plus a residual temporal head: for each
    (unit, hour) the unit's GNN embedding and the hour's inputs are projected, mixed across hours by residual dilated
    1-D convolutions (dilations 1, 2, 4: every hour of the 12-hour window sees every other), and added to the GNN's
    logit. The head's output layer starts at zero, so training starts from the plain GNN."""

    def __init__(self, sysm, feat, T, hidden=64, layers=4, th=32, dilations=(1, 2, 4), kernel=3, dropout=0.0):
        super().__init__()
        self.register_buffer("gbus", torch.as_tensor(sysm.gbus, dtype=torch.long))
        self.N, self.T = sysm.n_bus, T
        ig, ib = feature_layout(feat, T)
        self.register_buffer("ig", torch.as_tensor(ig, dtype=torch.long))
        self.register_buffer("ib", torch.as_tensor(ib, dtype=torch.long))
        self.gen_enc = mlp(feat.gen_dim, hidden, hidden)
        self.gnn = EdgeGNN(sysm.f_bus, sysm.t_bus, sysm.n_bus, feat.bus_dim + hidden, feat.edge_dim, hidden, layers)
        self.dec = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, T))
        self.t_in = nn.Linear(2 * hidden + len(ig) + len(ib), th)
        self.hour = nn.Parameter(torch.zeros(T, th))
        self.convs = nn.ModuleList([nn.Conv1d(th, th, kernel, padding=d * (kernel // 2), dilation=d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(th) for _ in dilations])
        self.t_out = nn.Linear(th, 1)
        nn.init.zeros_(self.t_out.weight)
        nn.init.zeros_(self.t_out.bias)

    def forward(self, xb, xg, xe):
        B, G = xg.shape[0], xg.shape[1]
        hg = self.gen_enc(xg)                                              # [B, G, H]
        agg = torch.zeros(B, self.N, hg.shape[-1]).index_add_(1, self.gbus, hg)
        h, _ = self.gnn(torch.cat([xb, agg], -1), xe)
        z = torch.cat([h[:, self.gbus], hg], -1)                          # [B, G, 2H]
        base = self.dec(z)                                                 # [B, G, T]
        pg = xg[:, :, self.ig]                                             # [B, G, 2, T]
        pb = xb[:, self.gbus][:, :, self.ib]                               # [B, G, 3, T]
        per = torch.cat([pg, pb], 2).transpose(2, 3)                      # [B, G, T, 5]
        x = torch.cat([z.unsqueeze(2).expand(-1, -1, self.T, -1), per], -1)
        x = self.t_in(x) + self.hour                                       # [B, G, T, th]
        x = x.reshape(B * G, self.T, -1)
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(torch.nn.functional.silu(norm(x)).transpose(1, 2)).transpose(1, 2)
        delta = self.t_out(x).reshape(B, G, self.T)
        return (base + delta).transpose(1, 2)                              # [B, T, G]


def build_model(kind, sysm, feat, T, seed=0):
    """kind: gnn (CommitGNN, the current model), gnnt (CommitGNNT), mlp (CommitMLP of otsl.ucml)"""
    from .ucml import build_uc_model1
    if kind in ("gnn", "mlp"):
        return build_uc_model1(sysm, feat, T, kind, seed=seed)
    torch.manual_seed(seed)
    return UCModel1(CommitGNNT(sysm, feat, T), feat, "gnn")


# ============================================================================ metrics
def aligned_labels(sysm, u, u0, p):
    """MILP labels aligned to the rounded prediction inside groups of identical units (cost-equivalent schedules)"""
    groups = sysm.identical_groups()
    return np.stack([align_to_prediction(sysm, u[i], (p[i] > 0.5).astype(np.int8), u0[i], groups)
                     for i in range(len(u))])


def roc_auc(p, y):
    from scipy.stats import rankdata
    p, y = p.reshape(-1), y.reshape(-1).astype(bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib_error(p, y, bins=15):
    """top-label expected calibration error: confidence max(p, 1-p) vs accuracy of round(p), equal-width bins"""
    p, y = p.reshape(-1), y.reshape(-1)
    conf = np.maximum(p, 1 - p)
    corr = ((p > 0.5) == (y > 0.5)).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    b = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    out = 0.0
    for k in range(bins):
        s = b == k
        if s.any():
            out += s.mean() * abs(corr[s].mean() - conf[s].mean())
    return float(out)


def fixable_at_precision(conf, wrong, precs=(0.99, 0.999)):
    """largest share of decisions that can be fixed, most confident first, with precision >= prec (cut points only
    between distinct confidence values)"""
    conf, wrong = conf.reshape(-1), wrong.reshape(-1).astype(float)
    o = np.argsort(-conf, kind="stable")
    c, w = conf[o], np.cumsum(wrong[o])
    n = np.arange(1, len(c) + 1)
    last = np.r_[c[1:] != c[:-1], True]                                  # end of each tie group
    prec = 1 - w / n
    out = {}
    for P in precs:
        ok = np.where(last & (prec >= P))[0]
        out[P] = float((ok.max() + 1) / len(c)) if len(ok) else 0.0
    return out


def prob_metrics(p, y_aligned, y_canon=None, std=None):
    """quality of probabilities p [N, T, G] against aligned MILP labels (and canonical labels for the log-loss)"""
    clip = lambda q: np.clip(q, 1e-6, 1 - 1e-6)
    ll = lambda q, y: float(-(y * np.log(clip(q)) + (1 - y) * np.log(1 - clip(q))).mean())
    wrong = (p > 0.5) != (y_aligned > 0.5)
    conf = np.maximum(p, 1 - p)
    fx = fixable_at_precision(conf, wrong)
    out = {"logloss": ll(p, y_aligned), "brier": float(((p - y_aligned) ** 2).mean()), "auc": roc_auc(p, y_aligned),
           "ece": calib_error(p, y_aligned), "wrong_per_inst": float(wrong.sum((1, 2)).mean()),
           "fix99": fx[0.99], "fix999": fx[0.999]}
    if y_canon is not None:
        out["logloss_canon"] = ll(p, y_canon)
    if std is not None:            # disagreement-adjusted confidence (k = 1, 2): |p - 0.5| - k * std
        for k in (1.0, 2.0):
            fk = fixable_at_precision(np.abs(p - 0.5) - k * std, wrong)
            out[f"fix99_dis{k:g}"], out[f"fix999_dis{k:g}"] = fk[0.99], fk[0.999]
    return out
