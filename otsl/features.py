"""Graph features for the GNNs.

Base features use only what the original framework uses (demand + static grid data).
``duals=True`` adds the solution of ONE all-lines-closed DC-OPF (a ~5 ms LP): flows, line
loading, LMPs, flow-limit duals and the first-order switching sensitivity -gamma_l * f_l.
This is cheap, always available at inference time, and is exactly the information that
classic switching heuristics (Fuller et al. 2012) rank lines by.
"""
from __future__ import annotations

import numpy as np
import torch


class Featurizer:
    def __init__(self, case, train: dict, duals: bool = False, fixed_closed=None, topo: bool = False):
        self.case, self.duals, self.topo = case, duals, topo
        N, L = case.n_bus, case.n_line
        Cg = case.Cg
        pmax_bus = Cg @ case.pmax
        has_gen = (Cg.sum(1) > 0).astype(float)
        cmin = np.array([case.cost[case.gen_bus == i].min() if has_gen[i] else 0.0 for i in range(N)])
        is_ref = np.zeros(N)
        is_ref[case.ref] = 1
        self.node_static = np.c_[pmax_bus, cmin / max(case.cost.max(), 1e-9), has_gen, is_ref]
        fixed = np.zeros(L) if fixed_closed is None else fixed_closed.astype(float)
        self.edge_static = np.c_[np.log(case.b), case.fmax, fixed]
        self.stats = None
        xn, xe = self._raw(train)
        self.stats = (xn.reshape(-1, xn.shape[-1]).mean(0), xn.reshape(-1, xn.shape[-1]).std(0) + 1e-6,
                      xe.reshape(-1, xe.shape[-1]).mean(0), xe.reshape(-1, xe.shape[-1]).std(0) + 1e-6)
        self.node_dim, self.edge_dim = xn.shape[-1], xe.shape[-1]

    def _raw(self, d: dict, idx=None):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        pd = sel(d["pd"])
        B = pd.shape[0]
        N, L = self.case.n_bus, self.case.n_line
        node = [pd[..., None], np.broadcast_to(self.node_static, (B, N, self.node_static.shape[1]))]
        edge = [np.broadcast_to(self.edge_static, (B, L, self.edge_static.shape[1]))]
        if self.topo:  # base-case in-service flag (1 = in service)
            base = sel(d["base"]) if "base" in d else np.ones((B, L))
            edge.append(base[..., None].astype(float))
        if self.duals:
            lmp, mu, gam, fl, va, pg = (sel(d[k]) for k in ["lmp0", "mu0", "gamma0", "flow0", "va0", "pg0"])
            node += [lmp[..., None], va[..., None], (pg @ self.case.Cg.T)[..., None]]
            dC = -gam * fl                                    # first-order cost change of opening line l
            dl = lmp[:, self.case.f_bus] - lmp[:, self.case.t_bus]
            edge += [(fl / self.case.fmax)[..., None], np.abs(fl / self.case.fmax)[..., None], mu[..., None],
                     dC[..., None], np.sign(dC)[..., None] * np.log1p(np.abs(dC))[..., None], dl[..., None],
                     (dl * fl)[..., None]]
        return np.concatenate(node, -1), np.concatenate(edge, -1)

    def __call__(self, d: dict, idx=None):
        xn, xe = self._raw(d, idx)
        mn, sn, me, se = self.stats
        if self.topo:  # keep the in-service flag binary even if it was constant in training
            me, se = me.copy(), se.copy()
            me[self.edge_static.shape[1]], se[self.edge_static.shape[1]] = 0.0, 1.0
        return (torch.as_tensor((xn - mn) / sn, dtype=torch.float32),
                torch.as_tensor((xe - me) / se, dtype=torch.float32))

    def flat(self, d: dict, idx=None):
        """Flattened features for the MLP baseline."""
        xn, xe = self(d, idx)
        return torch.cat([xn.flatten(1), xe.flatten(1)], 1)
