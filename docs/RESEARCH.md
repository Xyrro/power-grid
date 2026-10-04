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


## 3. Findings so far (IEEE 30; IEEE 118 in §4)

IEEE 30 (PGLib, 2 generators, 41 lines, K = 3): switching saves **21.3 %** of generation cost on
average — a strongly congested, "switching matters" regime. 1,269 train / 170 val / 347 test
scenarios, MILP 0.6 s per instance.

### F1. MILP labels are ambiguous — by construction, not by solver noise

| | plain DC-OTS labels (framework) | + switching cost 0.02 % of OPF cost per line |
|---|---|---|
| test scenarios with an **exactly tied** alternative topology | **91.6 %** | 64.6 % |
| scenarios that open the full budget K | 88.8 % | 44.1 % |
| distinct topologies in the training labels | 56 | 23 |
| mean benefit (switching cost included) | 21.36 % | 21.31 % |

The ties are structural: once line 2 is opened, lines 4 and 5 are in series, so opening either has
identical flows and cost; the MILP labels say {2,4} 128 times and {2,5} 54 times, arbitrarily.
A small per-switch cost ("parsimonious labels") removes the "free" third switch, cuts the number of
label classes by 2.4×, and costs only 0.05 pp of benefit — but it cannot remove series-line ties.

### F2. One-shot decoding of a per-line classifier is fragile; LP-verified screening fixes it

| Model 1 variant (IEEE 30) | top-1 decode → 1 LP: captured / feasible | candidate screening (~26 LPs): captured |
|---|---|---|
| MLP, BCE on MILP labels (framework) | 53.2 % / 57.1 % | 99.9 % |
| GNN, BCE (framework) | 50.5 % / 52.2 % | 99.8 % |
| GNN + dual features | 68.9 % / 69.5 % | 99.8 % |
| GNN + dual features + equivalence-aware soft labels | 68.2 % / 69.2 % | 99.8 % |
| GNN + dual features, raw (no switching cost) labels | 90.4 % / 91.6 % | 99.8 % |
| **GNN + dual features + REINFORCE (LP critic)** | **96.4 % / 98.3 %** | 99.6 % |
| baselines: dual-sensitivity greedy (no learning) | 83.0 % with 16 LPs | |
| kNN-LP, k = 20 (Johnson et al. 2020) | **100.0 % with 6 LPs** | |
| GNN-guided partial fixing (10 free lines) + MILP | 100.0 %, 0.08 s vs 0.63 s full MILP (8×) | |

*captured = (C_allclosed − C_method) / (C_allclosed − C_MILP); infeasible predictions fall back to all-closed.*

Why one-shot decoding fails: **97 % of the infeasible top-1 decodes open both equivalent lines 4 and 5
(predicted p ≈ 0.93 each)**. A factorised per-line Bernoulli output cannot represent "open 4 *or* 5";
class-reweighted BCE pushes both marginals above 0.5, and opening both overloads the network.
Equivalence-aware soft labels do not help (they make the two marginals *more* equal). What helps:
(i) scoring a set of candidate topologies with the exact LP (always feasible thanks to the
all-closed fallback), or (ii) a cost-aware loss — REINFORCE with the LP as critic learns the joint
consequence and breaks the symmetry (69 % → 98 % feasible with one LP).

### F3. Cost-aware training of Model 1: use the exact LP, not Model 2 (the dashed arrow)

| how Model 1's "cost-aware" loss is computed (IEEE 30) | top-1 captured | lines opened |
|---|---|---|
| BCE only (reference) | 68.9 % | 2.9 |
| through frozen Model 2 = physics decoder, relaxed z | 35.2 % | 1.4 |
| through frozen Model 2 = physics decoder, straight-through | 42.7 % | 1.4 |
| through frozen Model 2 = direct (PG,VA) regression (framework) | 31.7 % | 0.9 |
| through frozen Model 2 = direct regression, straight-through | −0.1 % (never switches) | 0.1 |
| **REINFORCE with exact LP critic (BCE warm start), 300 steps / 25 s** | **96.4 %** | 2.3 |
| REINFORCE from scratch, no MILP labels at all (1,500 steps, 7.4 k LPs, 91 s) | 79.6 % (97.6 % with screening) | 2.0 |

Gradients through a learned Model 2 drive Model 1 to switch *less*: the surrogate's cost error
(0.3–1.5 %) is of the same order as what switching saves on most grids, and the relaxed/straight-through
gradient mostly sees the overload penalty. The exact LP is cheap (9 ms) and unbiased, so the
dashed arrow should go to the LP (REINFORCE here; a differentiable LP layer as in DA-DNN is the
alternative). MILP labels remain useful as a warm start: label-free RL plateaus at 79.6 % (it
learns a safe two-line policy) while imitation → RL reaches 96.4 %.

### F4. Model 2: predict PG only and let physics produce VA

| Model 2 (IEEE 30, 1,190 test (demand, topology) pairs) | worst KCL mismatch | fully feasible | cost error |
|---|---|---|---|
| direct (PG, VA) regression (framework) | 0.19 p.u. (19 MW) | **0 %** | 1.55 % |
| direct + post-hoc repair (clip, balance, VA from B(z)θ = P) | 0 | 69 % | 0.68 % |
| physics decoder (PG → balance repair → VA from DC power flow), trained end-to-end | 0 | 68 % | 0.39 % |
| physics decoder + thermal-overload penalty | 0 | **75 %** | 0.40 % |

Given z and PG, VA is *determined* (θ = B(z)⁻¹(C_g PG − PD)), so predicting VA separately only
adds violations: the direct model's KCL residual summed over buses is 42 % of total demand. The
physics decoder satisfies power balance, Ohm's law and generator limits by construction; only line
limits remain (mean worst overload 0.2 MW). Training *through* the decoder halves the cost error
versus repairing afterwards.

### F5. Model 2 is not worth it as a screener for DC-OTS

Ranking ~64 candidate topologies per scenario by Model 2's predicted cost and LP-verifying the top 3:
87 % captured (physics decoder; Spearman 0.94) vs 76 % (direct; 0.74) vs 62 % (random) — while
simply LP-verifying all 64 candidates gives 100 % and costs ~0.6 s on one core. For DC-OTS the LP is
not the bottleneck; Model 2 only pays off where the per-topology problem is expensive (AC-OPF,
security-constrained, multi-period).

### F6. Learned switching values: MILP-free supervision that works

Labels = exact LP cost change of opening each line at the current topology (112 LPs ≈ 0.4 CPU-s per
scenario vs 0.6 s for one MILP on this small grid; unique and dense — no ambiguity). A topology-gated GNN predicts them
from one LP's duals.

| (IEEE 30) | captured | LPs / scenario |
|---|---|---|
| analytic first-order estimate −γ_l f_l: best line ranked first | 0.7 % of states | |
| learned values: best line ranked first / in top 3 | 69 % / 80 % of states | |
| exhaustive greedy (exact LP for every line, every step) | 83.1 % | 112 |
| learned-value greedy, 1 LP check per step | 84.0 % | 4 |
| **learned-value beam search (B = 3, R = 3)** | **99.4 %** | 21 |
| learned-value beam search (B = 5, R = 5) | 99.5 % (95 % of scenarios match/beat MILP) | 49 |

The analytic dual estimate almost always ranks the congested line itself first (opening it is
infeasible); the learned value corrects this. Greedy is myopic on this grid (the best pair is not
built from the best single line), beam search fixes it — reaching MILP-imitation quality **without
solving a single MILP**.

### F7. The testing metric (MSE to the MILP solution) mis-ranks solutions

With screening, 78 % of test scenarios are solved to within 0.001 % of the MILP cost; in 27 % of
all scenarios this is achieved with a *different* topology. Those solutions have identical
dispatch (PG MSE ≈ 1e-15) but VA MSE up to 8.8e-3 rad² (≈ 5° RMS), so the framework's MSE metric
would mark equally optimal answers as wrong; MSE and the optimality gap have rank correlation of
only 0.51. Report the optimality gap, benefit captured, feasibility rate and LP/MILP calls instead,
and compare PG only if a distance is wanted.

<!-- RESULTS-118 -->

