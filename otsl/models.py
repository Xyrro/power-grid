"""Neural models: a topology-aware GNN (Model 1 and Model 2 backbone), MLP baselines,
and the physics-consistent DC decoder for Model 2."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# small graphs: intra-op threading only adds contention (37x slower with 4 threads next to LP workers)
torch.set_num_threads(int(os.environ.get("OTSL_THREADS", 1)))


def mlp(i, h, o, n=2, ln=True):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h) if ln else nn.Identity(), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class EdgeGNN(nn.Module):
    """Interaction-network style message passing on the (fixed-index) bus-branch graph.

    Edges are directed (from -> to) so the model can represent signed flows; node updates
    aggregate outgoing and incoming edge states separately. ``edge_gate`` ([B, L] in [0,1])
    multiplies messages, so an open line (gate 0) carries no information - the encoder sees
    the topology it is asked about.
    """

    def __init__(self, f_bus, t_bus, n_bus, node_in, edge_in, hidden=64, layers=6):
        super().__init__()
        self.register_buffer("f", torch.as_tensor(f_bus, dtype=torch.long))
        self.register_buffer("t", torch.as_tensor(t_bus, dtype=torch.long))
        self.n_bus = n_bus
        self.node_enc = mlp(node_in, hidden, hidden)
        self.edge_enc = mlp(edge_in, hidden, hidden)
        self.edge_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(layers)])
        self.node_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(layers)])

    def forward(self, x, e, edge_gate=None):
        h, g = self.node_enc(x), self.edge_enc(e)
        B, H = h.shape[0], h.shape[-1]
        gate = None if edge_gate is None else edge_gate.unsqueeze(-1)
        for eu, nu in zip(self.edge_up, self.node_up):
            g = g + eu(torch.cat([g, h[:, self.f], h[:, self.t]], -1))
            m = g if gate is None else g * gate
            out = torch.zeros(B, self.n_bus, H, device=h.device, dtype=h.dtype).index_add_(1, self.f, m)
            inn = torch.zeros(B, self.n_bus, H, device=h.device, dtype=h.dtype).index_add_(1, self.t, m)
            h = h + nu(torch.cat([h, out, inn], -1))
        return h, g


class SwitchGNN(nn.Module):
    """Model 1: per-line logit of being switched OFF."""

    def __init__(self, f_bus, t_bus, n_bus, node_in, edge_in, hidden=64, layers=6):
        super().__init__()
        self.gnn = EdgeGNN(f_bus, t_bus, n_bus, node_in, edge_in, hidden, layers)
        self.head = mlp(3 * hidden, hidden, 1)

    def forward(self, x, e):
        h, g = self.gnn(x, e)
        f, t = self.gnn.f, self.gnn.t
        return self.head(torch.cat([g, h[:, f] + h[:, t], (h[:, f] - h[:, t]).abs()], -1)).squeeze(-1)


class SwitchMLP(nn.Module):
    """Model 1 baseline: flat demand (+ optional features) -> per-line logits."""

    def __init__(self, d_in, n_line, hidden=256, layers=3):
        super().__init__()
        self.net = mlp(d_in, hidden, n_line, n=layers)

    def forward(self, x_flat):
        return self.net(x_flat)


# ---------------------------------------------------------------------------------------------
# Model 2
# ---------------------------------------------------------------------------------------------
class DCPhysics(nn.Module):
    """Differentiable DC power-flow utilities for a fixed case (batched over topologies)."""

    def __init__(self, case, reg=1e-6):
        super().__init__()
        t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)
        self.register_buffer("A", t(case.A))                 # [L, N]
        self.register_buffer("Cg", t(case.Cg))               # [N, G]
        self.register_buffer("b", t(case.b))
        self.register_buffer("fmax", t(case.fmax))
        self.register_buffer("pmin", t(case.pmin))
        self.register_buffer("pmax", t(case.pmax))
        self.register_buffer("cost", t(case.cost))
        self.register_buffer("cost_q", t(case.cost_q))
        keep = [i for i in range(case.n_bus) if i != case.ref]
        self.register_buffer("keep", torch.as_tensor(keep))
        self.ref, self.N, self.reg = case.ref, case.n_bus, reg

    def balance(self, pg, pd):
        """E2ELR-style proportional repair: shift every unit toward its upper (or lower) limit by the
        same fraction of its headroom so that sum(pg) == sum(pd) exactly (if the total is reachable)."""
        D = pd.sum(-1, keepdim=True)
        P = pg.sum(-1, keepdim=True)
        up = (self.pmax - pg).sum(-1, keepdim=True).clamp_min(1e-9)
        dn = (pg - self.pmin).sum(-1, keepdim=True).clamp_min(1e-9)
        a_up = ((D - P) / up).clamp(0, 1)
        a_dn = ((P - D) / dn).clamp(0, 1)
        return pg + a_up * (self.pmax - pg) - a_dn * (pg - self.pmin)

    def angles(self, pg, pd, z):
        """Solve B(z) theta = Cg pg - pd with theta_ref = 0.  z may be fractional (relaxed)."""
        inj = pg @ self.Cg.T - pd                           # [B, N]
        w = self.b * z                                      # [B, L]
        Lap = torch.einsum("ln,bl,lm->bnm", self.A, w, self.A)
        k = self.keep
        Lr = Lap[:, k][:, :, k] + self.reg * torch.eye(len(k), dtype=Lap.dtype, device=Lap.device)
        th = torch.linalg.solve(Lr, inj[:, k].unsqueeze(-1)).squeeze(-1)
        va = torch.zeros_like(inj)
        va[:, k] = th
        return va

    def flows(self, va, z):
        return self.b * z * (va @ self.A.T)

    def gen_cost(self, pg):
        return (pg * self.cost).sum(-1) + (pg ** 2 * self.cost_q).sum(-1)

    def violations(self, pg, va, pd, z):
        """Per-sample constraint violations of a (pg, va) pair for topology z (all in p.u.)."""
        f = self.flows(va, z)
        kcl = (pg @ self.Cg.T - f @ self.A - pd).abs()      # A^T f = net outflow per bus
        gen = F.relu(self.pmin - pg) + F.relu(pg - self.pmax)
        line = F.relu(f.abs() - self.fmax * z)
        return {"kcl_max": kcl.amax(-1), "kcl_sum": kcl.sum(-1), "gen_max": gen.amax(-1),
                "line_max": line.amax(-1), "line_sum": line.sum(-1), "line_n": (line > 1e-4).sum(-1).double()}


class DispatchNet(nn.Module):
    """Model 2.  mode='direct': regress (pg, va) independently (the original framework).
    mode='physics': predict pg in [pmin, pmax] -> balance repair -> va from DC power flow.
    The physics mode satisfies KCL, Ohm's law and generator limits by construction; only
    thermal limits can be violated (and are penalised in training)."""

    def __init__(self, case, node_in, edge_in, mode="physics", hidden=64, layers=6):
        super().__init__()
        self.mode = mode
        self.phys = DCPhysics(case)
        self.gnn = EdgeGNN(case.f_bus, case.t_bus, case.n_bus, node_in, edge_in + 1, hidden, layers)
        self.register_buffer("gen_bus", torch.as_tensor(case.gen_bus, dtype=torch.long))
        self.pg_head = mlp(hidden, hidden, 1)
        if mode == "direct":
            self.va_head = mlp(hidden, hidden, 1)

    def forward(self, x, e, z, pd):
        """x,e: normalised features (float32); z: [B,L] topology; pd: [B,N] p.u. (float64)."""
        h, _ = self.gnn(x, torch.cat([e, z.float().unsqueeze(-1)], -1), edge_gate=z.float())
        raw = self.pg_head(h[:, self.gen_bus]).squeeze(-1).double()
        P = self.phys
        if self.mode == "direct":
            pg = P.pmin + (P.pmax - P.pmin) * raw          # unconstrained linear head
            va = self.va_head(h).squeeze(-1).double() * 0.3
            va = va - va[:, P.ref:P.ref + 1]
            return pg, va
        pg = P.pmin + (P.pmax - P.pmin) * torch.sigmoid(raw)
        pg = P.balance(pg, pd)
        va = P.angles(pg, pd, z.double())
        return pg, va
