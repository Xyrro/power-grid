# Learning DC Optimal Transmission Switching — critique of the two-stage framework and new methods

> **Side study.** This report assumed "switching status" meant *transmission line* switching (DC-OTS).
> The framework is about *generator* on/off (unit commitment); the main report is
> [`../RESEARCH.md`](../RESEARCH.md). The line-switching results are kept because they cover the
> setting RACLearn names as future work (topology changes).

*Working research log. Code: this repository. Literature details: [`literature.md`](../literature.md).
Raw numbers: `results/<case>/*.json`.*


## TL;DR

* **The core idea is published.** DA-DNN (Kim & Kim 2025) already trains switching predictions
  through an exact DC-OPF layer on generation cost, i.e. Model 1 + the dashed arrow. OptiGridML
  (Meng et al. 2025) is a two-GNN topology + flow design. A new paper must therefore position
  itself on what those works do not cover — see §2 and the findings below.
* **The labels are ambiguous.** 55–92 % of DC-OTS test instances have an *exactly* tied alternative
  topology (series lines, interchangeable switches). Plain DC-OTS also switches the full budget 89–98 %
  of the time, even when it saves nothing. A tiny per-switch cost ("parsimonious labels") removes
  much of this at 0.05 pp of benefit; equivalence-aware soft labels lift one-shot quality on IEEE 118
  from 61 % to 85 % of the gap closed.
* **One-shot "Model 1 → LP" is fragile**: 50–66 % gap closed; on IEEE 30 30–48 % of the predicted
  topologies are infeasible. Factorised per-line outputs open both of two mutually exclusive lines.
* **Fixes that work** (all keep the LP as the final step, so the answer is feasible and never worse
  than DC-OPF):
  1. *LP-verified candidate screening* (~26 LPs): 99.6–101 % on both grids.
  2. *REINFORCE with the exact LP as critic*, warm-started from imitation: **near-MILP quality with one
     LP** — 95–100 % of the gap closed on IEEE 118 over three seeds (mean 97.8 %), 97 % on IEEE 30.
     300 steps: 25–70 s of training.
  3. *GNN-guided partial fixing + MILP*: optimal-or-better on IEEE 118 in 0.23 s vs 7.0 s (30×).
* **The dashed arrow should not go through a learned Model 2**: through the framework's direct
  Model 2, Model 1 got worse (IEEE 30: 66 % → 31 % gap closed; IEEE 118: 61 % → −116 %, i.e. worse
  than never switching). A physics-consistent Model 2 helps a little but is fragile (IEEE 118:
  76 % relaxed, 0 % straight-through). The exact LP as critic gives 95–100 %.
* **Model 2 should predict PG only.** VA follows from DC power flow on the switched network. The
  framework's direct (PG, VA) regression had 0 % fully feasible outputs on both grids (worst KCL
  mismatch 19 MW on IEEE 30, 455 MW on IEEE 118); the physics decoder satisfies balance, Ohm's law
  and generator limits by construction (75–80 % fully feasible with an overload penalty). As a
  screener of topologies it is beaten by simply running the cheap LPs.
* **Strong simple baselines.** kNN over stored MILP topologies + LP check (Johnson et al. 2020) is
  near-optimal with ≤ 6 LPs whenever the base topology is fixed; the learned pipeline only pulls ahead
  under topology change (99.4 % vs 93.3 % with unseen outages).
* **Switching benefits are complementary** on IEEE 118 (a line that saves only with two others
  costs on its own), so greedy / beam / one-step value learning plateau at ~50 %, and label-free RL
  never discovers the good sets (0 %). MILP labels are still needed as a warm start there.
* **MSE to the MILP solution is the wrong test metric** (rank correlation with the optimality gap
  0.18–0.51; equally optimal answers have VA MSE up to 0.011 rad²). Report the gap closed,
  feasibility rate and solver calls.


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
standard non-deep baselines. If "switching status" means **generator on/off** rather than line
status, the closest work is **RACLearn** (Park, Chen, Han, Tanneau & Van Hentenryck, IEEE TPS 2024,
arXiv 2211.15755): a GNN predicts commitments and active line constraints, epistemic uncertainty
selects high-confidence commitments to fix, a polynomial-time restoration makes them feasible, and the
MILP finishes the job (2–4× faster on MISO's 6,708-bus system). See [`literature.md`](../literature.md)
for the full table.

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


## 3. Findings on IEEE 30 (IEEE 118 and topology shift in §4)

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
limits remain (mean worst overload 0.2 MW). On IEEE 30, training *through* the decoder also halves
the cost error versus repairing afterwards (on IEEE 118 it does not, see F13).

### F5. Model 2 is not worth it as a screener for DC-OTS

Ranking ~64 candidate topologies per scenario by Model 2's predicted cost and LP-verifying the top 3:
87 % captured (physics decoder; Spearman 0.94) vs 76 % (direct; 0.74) vs 62 % (random) — while
simply LP-verifying all 64 candidates gives 100 % and costs ~0.6 s on one core. For DC-OTS the LP is
not the bottleneck; Model 2 only pays off where the per-topology problem is expensive (AC-OPF,
security-constrained, multi-period).

### F6. Learned switching values: MILP-free supervision that works on IEEE 30 (not on IEEE 118, see F10)

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


## 4. IEEE 118 (a "does switching pay at all?" regime) and topology shift

PGLib `case118_ieee`, nominal ratings, K = 3, 800 train / 120 val / 200 test scenarios. MILP: 6.5 s
mean (HiGHS, 20 s limit, 5.5 % of instances stop at the limit, 0.01 % gap). Switching saves only
**0.40 %** on average, the median scenario saves 0.05 %, and 25 % of scenarios save < 0.01 %.
Because of this the per-scenario "benefit captured" ratio explodes, so the headline metric below is
**gap closed** = 1 − mean gap(method) / mean gap(all-closed) (all-closed gap = 0.405 %). Values above
100 % mean the method beats the time/gap-limited reference MILP.

### F8. Ambiguity persists at scale; raw labels always use the whole budget

| | raw DC-OTS labels | + switching cost |
|---|---|---|
| exactly tied alternative topology (≤ 1e-6) | 55 % | 37.5 % |
| alternative within MILP tolerance (≤ 1e-4) | 66 % | 54.5 % |
| scenarios opening the full budget | **98 %** | 54 % |
| training scenarios with an equally optimal single-line swap | – | 23 % |

(Raw: 100 test scenarios; 17 % of their MILPs hit the time limit under CPU contention, so these
ambiguity numbers are lower bounds.)

### F9. Model 1 on IEEE 118

| method (IEEE 118) | gap closed | LPs / scenario |
|---|---|---|
| dual-sensitivity greedy (no learning) | 28.5 % | 11 |
| kNN-LP, k = 5 / k = 20 (Johnson et al.) | 100.8 % / **101.1 %** | 3 / 5 |
| MLP-BCE (framework), top-1 → LP | 65.7 % | 1 |
| GNN-BCE (framework), top-1 → LP | 61.0 % | 1 |
| GNN-BCE + duals, top-1 → LP | 60.8 % | 1 |
| GNN-BCE + duals + **equivalence-aware labels**, top-1 → LP | **84.9 %** | 1 |
| **GNN + duals + REINFORCE (LP critic)**, top-1 → LP | **100.3 %** | **1** |
| any of the above + candidate screening | 100.4 – 101.2 % | ~26 |
| GNN-guided partial fixing (10 free lines) + MILP | 102.3 %, **0.23 s vs 7.0 s** full MILP (30×) | – |
| GNN-guided partial fixing (25 free lines) + MILP | 102.4 %, 0.84 s (8×) | – |

* The framework's one-shot pipeline closes only ~61–66 % of the gap; its models **over-switch**
  (2.8–3.0 lines opened vs 1.9 for the MILP) because class-weighted BCE inflates positives and
  extra switches are never penalised by the loss.
* **Equivalence-aware soft labels** help here (61 % → 85 %), unlike on IEEE 30 where ties were
  between mutually exclusive lines.
* **Seed check** (top-1, one LP; seeds 0 / 1 / 2): BCE 60.8 / 56.4 / 67.6 % (mean 61.6 %);
  equivalence-aware labels 84.9 / 69.5 / 72.9 % (mean 75.8 %, better on every seed);
  BCE → REINFORCE 100.3 / 94.8 / 98.3 % (mean 97.8 %). The ordering is stable; the single-seed
  "100 %" for REINFORCE is the top of a 95–100 % range.
* **REINFORCE with the exact LP as critic** (300 steps, BCE warm start) makes one-shot decoding
  MILP-quality: −0.0014 % mean gap with **one** LP per scenario, 78.5 % of scenarios matching or
  beating the reference MILP, and it learns to open 1.96 lines (MILP: 1.9).
* **GNN-guided partial fixing** (fix all but the 10 most likely lines closed, solve the small MILP)
  is the best accuracy/time trade-off when a MILP solver is available: optimal-or-better in 0.23 s.
* **kNN-LP is again near-optimal with ≤ 5 LPs.** With a fixed base topology and only load variation,
  the optimal topologies come from a small set (43 distinct in training), so a lookup + LP check is
  a very strong, training-free baseline that any learned Model 1 must be compared against.

### F10. Switching benefits are complementary — one-line-at-a-time methods cannot find them

| (IEEE 118) | gap closed | LPs / scenario |
|---|---|---|
| exhaustive greedy (exact LP for every line and step) | 49.9 % | 423 |
| learned-value greedy, R = 1 / 4 | 44.3 % / 46.5 % | 3 / 10 |
| learned-value beam search B = 5, R = 5 | 47.4 % | 45 |

The most frequent optimal set {61, 64, 105} illustrates why. In every test scenario where it is
optimal, opening any *single* one of these lines **raises** cost (+0.01 %, +0.12 %, +0.62 %), most
pairs raise it as well, but the triple **saves 0.8–1.4 %**. Greedy, beam search and one-step value
learning are structurally unable to reach such sets; set-level methods (MILP imitation + LP
screening, kNN over stored sets) are. Learned one-step values work on IEEE 30 (99.4 % with beam
search) but not here — a learned *cost-to-go* (multi-step RL / tree search) would be needed.

### F11. MSE to the MILP solution is nearly uninformative on IEEE 118

With screening, 89.5 % of test scenarios are within 0.001 % of the MILP cost, 30.5 % of them with
a different topology. Equally optimal solutions differ by up to 0.011 rad² in VA and even 0.076 p.u.²
in PG (generators with identical cost are interchangeable). Rank correlation between the MSE and the
optimality gap: **0.18**.


### F12. Topology shift: learned + LP screening transfers better than kNN; no GNN-over-MLP advantage yet

IEEE 30, every test scenario has one random non-bridge line out of service (546 scenarios, unseen
combinations of load and outage). Models get the in-service flag as an edge feature (GNN: also as
message gating) and dual features from one DC-OPF on the *actual* outaged topology.

| trained on → | intact grid only | mix with outages (68 % of scenarios) |
|---|---|---|
| kNN-LP, k = 20 | 95.3 % (87.8 % captured) | 97.6 % (93.3 %), 16 LPs |
| MLP + duals, screening | 97.4 % (94.6 %) | **99.8 % (99.4 %)** |
| GNN + duals, screening | 95.1 % (91.8 %) | 99.6 % (98.9 %) |
| MLP + duals, top-1 | 51.3 % (47 % feasible) | 67.0 % (65 % feasible) |
| GNN + duals, top-1 | 52.8 % (55 % feasible) | 59.0 % (57 % feasible) |
| dual-sensitivity greedy (no learning) | 82.3 % | |

*gap closed (benefit captured)*; on the intact test set all screening variants reach 99.9 %.

* kNN degrades under topology change (its stored topologies may not fit the outaged grid); a learned
  proposal + LP check degrades less, and training on topology-varied data closes most of the gap.
* At this scale the GNN shows **no transfer advantage over an MLP that sees the same features** —
  the dual features (flows/prices of an OPF on the actual topology) already encode the outage. The
  case for a GNN ("GNN team") therefore has to be made on larger grids, multi-outage shifts, or
  transfer *across grids*, which an MLP cannot do at all; this is untested here.


### F13. Model 2 and the dashed arrow on IEEE 118

| Model 2 (IEEE 118, 787 test (demand, topology) pairs) | worst KCL mismatch | fully feasible | cost error |
|---|---|---|---|
| direct (PG, VA) regression (framework) | 4.55 p.u. (455 MW) | **0 %** | 4.36 % |
| direct + post-hoc repair | 0 | 47 % | 1.07 % |
| physics decoder | 0 | 57 % | 1.41 % |
| physics decoder + overload penalty | 0 | **80 %** | 1.87 % |

On the larger grid the direct model's angle errors turn into huge flow errors (KCL residual summed
over buses = 177 % of demand, 7 overloaded lines per sample). Physics-consistent decoding removes
all balance violations; the overload penalty buys feasibility at some cost accuracy, and here a
post-hoc repair of a direct model is slightly *more* cost-accurate than end-to-end decoding (unlike
IEEE 30) — the advantage of end-to-end training is feasibility, not cost accuracy.

| use of Model 2 (IEEE 118) | gap closed |
|---|---|
| screener: LP-verify all ~64 candidates (no Model 2) | 101.3 % |
| screener: physics decoder picks top 3 → LP (Spearman 0.65) | 66.1 % |
| screener: direct regression picks top 3 → LP (Spearman 0.04) | 1.9 % |
| critic: Model 1 trained through physics decoder, relaxed z | **76.5 %** (from 60.8 %) |
| critic: physics decoder, straight-through | 0.0 % (stops switching) |
| critic: direct regression, relaxed z | **−116.5 %** (worse than never switching) |
| critic: direct regression, straight-through | −22.1 %, 9.5 % feasible |
| critic: exact LP (REINFORCE), three seeds | **94.8–100.3 %** |

A physics-consistent Model 2 can serve as a critic in its relaxed form (+16 pp) but is fragile;
the framework's direct Model 2 is actively harmful as a critic (surrogate exploitation: its cost
error on the topologies Model 1 ends up choosing is 4.9–8.1 %). The exact LP remains the right
critic.


### F14. RACLearn-style confidence-aware fixing transfers to line switching; MC-dropout adds nothing here

RACLearn (Park et al., IEEE TPS 2024; see [`raclearn_comparison.md`](../raclearn_comparison.md)) fixes the
most confident binaries by MC-dropout (1/σ) and solves the reduced MILP. Same recipe on IEEE 118
(100 test scenarios, full MILP 7.0 s):

| fixing strategy | lines fixed | gap closed | time (speed-up) |
|---|---|---|---|
| MC-dropout, fix 50 % / 90 % / 97 % / all | 50 / 90 / 97 / 99 % | 101.8 / **101.7** / 68.3 / 60.8 % | 4.07 s (1.7×) / **0.34 s (20×)** / 0.06 s / 0.03 s |
| probability margin, fix 50 % / 90 % / 97 % | | 101.5 / 101.6 / 68.1 % | 4.55 / 0.40 / 0.07 s |
| keep the 10 most likely-to-open lines free | 94 % | **102.2 %** | **0.33 s (21×)** |

MC-dropout and the probability margin rank decisions almost identically (≈ 1 % of line decisions are
"open"), and quality falls off a cliff between 90 % and 97 % fixed, where the lines that should be
opened sit. For sparse switching decisions, choose the *free* set by probability of being opened
rather than fixing by a symmetric confidence.

## 5. Recommended changes to the framework

| box in the diagram | change | evidence |
|---|---|---|
| MILP solver (labels) | add a small per-switch cost; store equivalent / near-optimal topologies (no-good cuts or single-swap LP checks) | F1, F8, F9 |
| Model 1 loss | BCE (on equivalence-aware targets) as warm start, then **REINFORCE with the exact LP** as the cost-aware critic | F2, F3, F9 |
| Model 1 features | add one all-closed DC-OPF solve (flows, LMPs, flow-limit duals) and the in-service flags | F2, F12 |
| Model 1 output → LP | decode a *set* of candidates (top subsets + samples + all-closed) and keep the cheapest LP; or partial-fix + small MILP | F2, F9 |
| dashed arrow | point it at the LP (policy gradient or a differentiable LP layer), not at Model 2 | F3 |
| Model 2 | predict PG only → balance repair → VA = B(z)⁻¹P; penalise line overloads; use it only where the per-topology problem is expensive | F4, F5 |
| validation / testing | gap closed, feasibility before fallback, solver calls, wall time; never MSE to one MILP solution | F7, F11 |
| baselines | always report all-closed DC-OPF, kNN-LP, partial-fix MILP, and the MILP time | F9 |
| benchmarks | report the share of instances with non-trivial benefit; include topology-shifted test sets | F9, F12 |

## 6. Research backlog (ranked)

1. **Cost-to-go learning for complementary switching** (F10): Q-learning / MCTS over switching sets
   with the LP as the environment, warm-started from MILP imitation; compare with the one-shot
   REINFORCE policy on grids where good switching sets are complementary.
2. **Autoregressive / set-valued Model 1**: a decoder that picks lines sequentially, conditioned on
   earlier picks, can represent "open 4 *or* 5" (F2) — a cheaper alternative to screening.
3. **Scale and transfer**: PGLib 300 / 1354 / 2869-bus cases and *cross-grid* training (one GNN for
   several grids) — the setting where a GNN can beat an MLP or kNN (F12). The case still needs a
   faster MILP (Gurobi, tighter big-M, or partial fixing for the labels themselves).
4. **Differentiable LP layer vs REINFORCE** (DA-DNN-style cvxpylayers vs policy gradient): gradient
   bias and variance, wall time, local optima.
5. **Guarantees**: conformal calibration of the number of candidates so that P(gap ≤ ε) ≥ 1 − α;
   the all-closed fallback already guarantees "never worse than DC-OPF".
6. **Harder per-topology problems** where Model 2 is worth it: security-constrained (N-1) OTS,
   multi-period OTS with switching transitions, AC feasibility of DC-OTS topologies.
7. **Benchmark contribution**: an OTS learning dataset with solution pools / equivalence classes
   (no public OPF dataset — PGLearn, OPFData — has switching labels).

## 7. Reproducibility and caveats

* Everything runs on CPU with open-source HiGHS (`scripts/gen_all.sh`, `scripts/run_all_118.sh`,
  `scripts/topology_chain.sh`); ~6 h on 4 cores in total. Seeds are fixed; single training seed per
  model (no confidence intervals yet — differences of a few points on top-1 decoding are within noise).
* Reference MILPs use a 0.01 % (IEEE 118) / 0.001 % (IEEE 30) relative gap and a 20 s / 30 s limit;
  5.5 % of IEEE 118 instances hit the limit, which is why some methods "beat" the MILP (gap closed > 100 %).
* Wall times were measured with several jobs sharing 4 cores; LP / MILP call counts are the more
  reliable cost measure. One fixed-topology DC-OPF takes ~5 ms (IEEE 30) / ~10 ms (IEEE 118) on one core.
* DC model only; the switching decisions have not been checked for AC feasibility.




