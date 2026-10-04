# Learning DC Optimal Transmission Switching — critique of the two-stage framework and new methods

*Working research log. Code: this repository. Literature details: [`literature.md`](literature.md).
Raw numbers: `results/<case>/*.json`.*

<!-- RESULTS-TLDR -->

## 1. The framework as studied

```
PD ──► Model 1 (GNN) ──► z ∈ {0,1}^L ──► LP: DC-OPF(PD | z) ──► labels for Model 2
 │          ▲ BCE vs MILP z*                    │
 │          └─────────── (dashed) Model 2 output feeds Model 1's loss
 └──► MILP: DC-OTS(PD) ──► z*, PG*, VA*   (Model 1 labels; testing reference: MSE)
Model 2: (PD, z) ──► (PG, VA), MSE vs LP solution;  validation: violations + optimality gap
```

Assumptions made here (the diagram does not pin them down):

* "switching status" = **transmission line switching** (DC-OTS, Fisher, O'Neill & Ferris 2008),
  with a **switching budget** K = 3 and lines whose removal islands the grid (bridges) kept closed.
* DC model, so **QD is not used** anywhere (it is only meaningful in an AC model).
* Benchmarks: PGLib-OPF v23.07 `case118_ieee` (main) and `case30_ieee` (small, large switching
  benefit). Loads: global factor U[0.85, 1.15] × per-bus factor U[0.9, 1.1] (PGLearn-style).
* MILP: big-M formulation on HiGHS, |θ| ≤ π/3, big-M tightened by a K-edge-disjoint-path angle
  bound (valid under the budget), relative MIP gap 1e-4 (118) / 1e-5 (30).

## 2. Is it new? — No, the core idea is published; here is what is not

The closest published work is **DA-DNN** (Kim & Kim 2025, arXiv 2507.17194; extended Dec 2025):
a network predicts relaxed line states that pass through an *exact differentiable DC-OPF layer*,
trained on generation cost without MILP labels — i.e. Model 1 + the dashed arrow, with the LP
instead of a learned Model 2. **OptiGridML** (Meng, Haider & Van Hentenryck 2025) is a two-GNN
topology-predictor + flow-surrogate design (export maximisation). kNN-over-past-topologies + LP
(Johnson et al. 2020) and learning-assisted fixing + reduced MILP (Pineda et al. 2024) are the
standard non-deep baselines. See [`literature.md`](literature.md) for the full table.

What we did **not** find in the literature, and what this repository therefore investigates:

1. A quantification of **label ambiguity** in DC-OTS and its effect on imitation learning, with
   two fixes (parsimonious labels via a switching cost; equivalence-aware soft labels).
2. **Learned switching values**: a GNN trained on exact per-line LP cost changes — unique, dense,
   MILP-free labels — decoded by LP-verified greedy/beam search.
3. **Dual-informed GNN features** from one all-lines-closed DC-OPF.
4. A **physics-consistent Model 2** (PG → exact power-balance repair → VA from the switched
   B(z) matrix) versus the framework's direct (PG, VA) regression, and Model 2's actual value as a
   *screener* of candidate topologies.
5. An empirical test of the **dashed arrow** (training Model 1 through a learned Model 2) for
   surrogate exploitation, versus REINFORCE with the exact LP as critic.
6. A critique of the **testing metric** (MSE to the MILP solution).

<!-- RESULTS-BODY -->
