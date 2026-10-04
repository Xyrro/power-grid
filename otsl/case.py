"""MATPOWER / PGLib-OPF case parsing and DC network data.

Only what the DC model needs is kept: bus loads, generator limits and costs,
branch reactances, thermal limits and angle-difference limits. Everything is
converted to per-unit on the case base MVA.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

CASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cases")


def _parse_matrix(text: str, name: str) -> np.ndarray:
    m = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
    if m is None:
        raise ValueError(f"mpc.{name} not found")
    rows = []
    for line in m.group(1).split("\n"):
        line = line.split("%")[0].strip().rstrip(";").strip()
        if not line:
            continue
        rows.append([float(v) for v in line.replace(";", " ").split()])
    width = max(len(r) for r in rows)
    return np.array([r + [0.0] * (width - len(r)) for r in rows])


@dataclass
class DCCase:
    name: str
    base_mva: float
    n_bus: int
    n_gen: int
    n_line: int
    ref: int                      # reference bus (0-based)
    pd: np.ndarray                # [N] nominal active load (p.u.)
    qd: np.ndarray                # [N] nominal reactive load (p.u.) - unused by DC model, kept as a feature
    gen_bus: np.ndarray           # [G] bus index of each generator
    pmin: np.ndarray              # [G] (p.u.)
    pmax: np.ndarray              # [G] (p.u.)
    cost: np.ndarray              # [G] linear cost ($/p.u.-h), i.e. c1 * base_mva
    cost_q: np.ndarray            # [G] quadratic coefficient ($/(p.u.)^2-h); PWL-approximated in the LP
    f_bus: np.ndarray             # [L]
    t_bus: np.ndarray             # [L]
    b: np.ndarray                 # [L] series susceptance 1/(x*tap) (p.u.)
    fmax: np.ndarray              # [L] flow limit (p.u.) incl. angle-difference limit b*dtheta_max
    dtheta_max: np.ndarray        # [L] angle-difference limit (rad)
    bus_ids: np.ndarray = field(default=None)

    # ---- derived matrices -------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        """Branch-bus incidence [L, N]: +1 at from bus, -1 at to bus."""
        A = np.zeros((self.n_line, self.n_bus))
        A[np.arange(self.n_line), self.f_bus] = 1.0
        A[np.arange(self.n_line), self.t_bus] = -1.0
        return A

    @property
    def Cg(self) -> np.ndarray:
        """Generator-bus incidence [N, G]."""
        C = np.zeros((self.n_bus, self.n_gen))
        C[self.gen_bus, np.arange(self.n_gen)] = 1.0
        return C

    def bridges(self) -> np.ndarray:
        """Boolean mask of lines whose removal disconnects the graph."""
        import networkx as nx

        g = nx.MultiGraph()
        g.add_nodes_from(range(self.n_bus))
        for l, (i, j) in enumerate(zip(self.f_bus, self.t_bus)):
            g.add_edge(int(i), int(j), key=l)
        # networkx bridges() ignores parallel edges correctly only on Graph; handle multi-edges manually
        simple = nx.Graph()
        simple.add_nodes_from(range(self.n_bus))
        mult = {}
        for l, (i, j) in enumerate(zip(self.f_bus, self.t_bus)):
            key = (min(i, j), max(i, j))
            mult[key] = mult.get(key, 0) + 1
            simple.add_edge(int(i), int(j))
        br = set()
        for (i, j) in nx.bridges(simple):
            key = (min(i, j), max(i, j))
            if mult[key] == 1:
                br.add(key)
        return np.array([(min(i, j), max(i, j)) in br for i, j in zip(self.f_bus, self.t_bus)])


def load_case(name: str, line_limit_scale: float = 1.0) -> DCCase:
    """Load a PGLib-OPF case, e.g. ``"case118_ieee"`` or ``"case118_ieee__api"``."""
    path = name if os.path.exists(name) else os.path.join(CASE_DIR, f"pglib_opf_{name}.m")
    text = open(path).read()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text).group(1))
    bus = _parse_matrix(text, "bus")
    gen = _parse_matrix(text, "gen")
    branch = _parse_matrix(text, "branch")
    gencost = _parse_matrix(text, "gencost")

    bus_ids = bus[:, 0].astype(int)
    idx = {b: i for i, b in enumerate(bus_ids)}
    ref = int(np.where(bus[:, 1] == 3)[0][0])
    pd = (bus[:, 2] + bus[:, 4]) / base   # Gs treated as constant-power load, as MATPOWER does in DC
    qd = bus[:, 3] / base

    on = gen[:, 7] > 0
    gen, gencost = gen[on], gencost[on]
    gen_bus = np.array([idx[int(b)] for b in gen[:, 0]])
    pmax = gen[:, 8] / base
    pmin = np.maximum(gen[:, 9], 0.0) / base
    # polynomial cost (model 2): ... c2 c1 c0 in the last columns
    ncoef = gencost[:, 3].astype(int)
    c1 = np.array([row[4 + n - 2] if n >= 2 else 0.0 for row, n in zip(gencost, ncoef)])
    c2 = np.array([row[4 + n - 3] if n >= 3 else 0.0 for row, n in zip(gencost, ncoef)])
    # drop generators that can never produce (synchronous condensers)
    keep = pmax > 1e-9
    gen_bus, pmin, pmax, c1, c2 = gen_bus[keep], pmin[keep], pmax[keep], c1[keep], c2[keep]

    br_on = branch[:, 10] > 0
    branch = branch[br_on]
    f_bus = np.array([idx[int(b)] for b in branch[:, 0]])
    t_bus = np.array([idx[int(b)] for b in branch[:, 1]])
    tap = np.where(branch[:, 8] == 0, 1.0, branch[:, 8])
    b = 1.0 / (branch[:, 3] * tap)
    rate = branch[:, 5] / base
    rate = np.where(rate <= 0, 99.0, rate) * line_limit_scale
    if branch.shape[1] > 12:
        dth = np.deg2rad(np.minimum(np.abs(branch[:, 11]), np.abs(branch[:, 12])))
        dth = np.where(dth <= 0, np.pi / 3, dth)
    else:
        dth = np.full(len(branch), np.pi / 3)
    fmax = np.minimum(rate, b * dth)

    return DCCase(
        name=name, base_mva=base, n_bus=len(bus), n_gen=len(gen_bus), n_line=len(branch), ref=ref,
        pd=pd, qd=qd, gen_bus=gen_bus, pmin=pmin, pmax=pmax, cost=c1 * base, cost_q=c2 * base ** 2,
        f_bus=f_bus, t_bus=t_bus, b=b, fmax=fmax, dtheta_max=dth, bus_ids=bus_ids,
    )
