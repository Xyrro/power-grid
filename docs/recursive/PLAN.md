# Recursive reasoning for end-to-end unit commitment

*Branch `recursive-uc` (from `unit-commitment`). Question: can a small network applied recursively, in the style of
the Tiny Recursive Model (TRM) and its relatives, produce good unit-commitment schedules end to end, without a MILP?*

## Why recursion fits unit commitment

Unit commitment is a constraint-satisfaction problem with an economic objective. The one-shot Model 1 of the
`unit-commitment` branch predicts every on/off decision in one pass and gets the joint structure wrong: without repairs
it serves only 10–17 % of 12-hour instances (§4 V2 of `docs/RESEARCH.md`), and even with repairs, REINFORCE fine-tuning and
candidate screening the best end-to-end pipeline is a 2.6 % mean gap, behind a learning-free baseline (LP rounding +
repair + 5 LPs, 1.6 %; §7 Y1). The same failure, a single pass that cannot correct itself, is what recursive reasoning
models address on Sudoku, mazes and ARC:

* **TRM** (Jolicoeur-Martineau 2025) keeps an answer embedding *y* and a latent *z*; a 2-layer network updates *z*
  *n* times from (*x*, *y*, *z*), then updates *y* from (*y*, *z*); this repeats *T* times, and the whole process is
  repeated for up to 16 *deep-supervision* steps, each supervised towards the answer, carrying (*y*, *z*) forward. A
  learned halting head stops early. 87 % on Sudoku-Extreme from 1,000 training puzzles with a 5M-parameter model.
* **HRM** (Wang et al. 2025): the two-network predecessor; deep supervision is the main source of its gains.
* **Recurrent Relational Networks**, **NeuroSAT**, graph-colouring GNNs: recurrent message passing on the constraint
  graph, with a loss at every step; more iterations at test time solve harder or larger instances.
* **Easy-to-hard recurrent networks**, **Deep Thinking without overthinking**: recurrent networks trained on small
  problems extrapolate to larger ones by running more iterations, if the input is re-injected at every step ("recall")
  and training uses a progressive loss.
* **Universal Transformers**, **ACT**, **recurrent-depth latent reasoning**, **looped transformers**: weight-shared depth
  and adaptive computation; **Deep Equilibrium Models**: the fixed-point view (TRM found it unnecessary).
* **SATNet**: a differentiable relaxation layer inside the network.

For unit commitment, recursion adds three things a one-shot model lacks: (1) it can correct its own joint
inconsistencies (adequacy across hours, min up/down, start-up economics), as a Sudoku solver corrects conflicting
cells; (2) shared-weight message passing over the network graph reaches far beyond the few hops of a fixed-depth GNN,
which matters for congestion; (3) test-time compute scales: harder instances, longer horizons or larger systems can be
given more iterations.

## Design

**Tokens.** One token per (unit, hour); unit features (costs, limits, min up/down, initial status), hourly features
(load, renewables, reserve) and, optionally, the LP-relaxation values and prices that Model 1 already uses. Mixing
along three axes: across units in the same hour (adequacy, reserve), across hours for the same unit (min up/down,
ramping, start-ups), and over the network graph through the units' buses (congestion).

**Recursion (arm A, pure TRM).** One tiny shared network; *z* ← net(*x*, *y*, *z*) *n* times, *y* ← net(*y*, *z*);
*T* − 1 cycles without gradient and one with; up to *N*sup deep-supervision steps; a halting head trained to predict
"the current schedule is within ε of the optimum"; EMA of the weights.

**Recursion with the solver in the loop (arm B).** Between deep-supervision steps, round the current *y*, run the
exact dispatch LP (milliseconds) and feed back its signals: hourly shortfall, reserve shortfall, prices, line
loadings, unit dispatch. The model refines its schedule with what the physics says is wrong. Slower per step, but each
step sees exact feedback, and every step's schedule is a priced candidate.

**Training signal.** Deep supervision on cheap near-optimal labels (the 2,000 polished labels of §7 Y3; no full MILP),
with symmetry-aware targets for identical units (34–62 % of multi-hour labels are arbitrary between them; §3 U1, §4
V1); then cost-aware fine-tuning of the final answer with the exact LP as critic (REINFORCE, §3 U4). Augmentation:
permutations of identical units, as TRM shuffles Sudoku digits.

**Inference.** Run the recursion, round, apply the existing min up/down and block-adequacy repairs, solve one dispatch
LP (or screen the last few steps' schedules with ≤ 5 LPs).

## Tokenization and positional embeddings

**Serialize the grid as a variable-length set of typed tokens**: unit-hour tokens (the decisions), bus tokens (loads
and renewables per hour), line tokens (limits; an outaged line is simply a missing token) and hour tokens (system load,
reserve requirement). The length follows the number of units, buses, lines and hours, so one model can run on 12 or
24 hours, on RTS-GMLC or a larger system, and on a changed topology without retraining its shape. On RTS-GMLC 12 h
this is about 876 unit-hour + 73 bus + 120 line + 12 hour tokens: full attention on a GPU, factorized (axial)
attention over time and over units on a CPU.

**Positional embeddings that carry physics, not indices:**
* **Time**: relative, not absolute (rotary or an ALiBi-style bias on the hour difference). Min up/down, ramping and
  start-up economics are relations between hours, and relative encodings extrapolate to longer horizons.
* **Grid position**: electrical, not graph-index. An attention bias from the electrical distance between buses
  (from the DC susceptance matrix: effective resistance or bus-impedance entries, in the spirit of Graphormer's
  shortest-path bias), and as node encodings each bus's PTDF signature (how an injection there loads every line) or
  sign-invariant random-walk / Laplacian eigenvector encodings.
* **No unit-ID embeddings**: a unit is described by its parameters, so identical units stay exactly symmetric and
  the model can transfer to other systems. Token-type embeddings distinguish unit, bus, line and hour tokens.

## Risks and how to address them

**1. Many equally good answers.** 56 of the 73 RTS-GMLC units are in 24 groups of identical units, every group at a
single bus, so they are exactly interchangeable; 34–62 % of MILP labels are arbitrary between them. A
permutation-equivariant network also gives identical units identical outputs, so it cannot say "two of these five".
* Predict, per identical group and hour, **how many** units are on (a count), and assign which ones with a
  deterministic dynamic program that respects each unit's initial status and min up/down. This removes the exact
  symmetry and shrinks the output; augmentation by permuting identical units comes for free.
* For near-ties between different units: **set-valued targets**, i.e. several near-optimal schedules per instance
  (polished labels from different teachers or seeds) and a loss that takes the best match in the set.
* Cost-based objectives (below) do not care which optimum is reached.

**2. The goal is cost and feasibility, not exact match.**
* **Cost-weighted deep supervision**: weight each decision's loss by the LP-priced cost of getting it wrong (the
  error-cost models of §5 W3 of `docs/RESEARCH.md`).
* **Reward each step for improving the schedule**: REINFORCE with the exact LP as critic and a leave-one-out
  baseline, on every deep-supervision step, so the model learns to improve rather than to imitate.
* **Decision-focused gradients**: with relaxed commitments the dispatch LP's duals give d(cost)/d(commitment)
  (sensitivities already checked against finite differences in `tests/test_uc.py`).
* **Halting head** trained to predict "within 0.5 % of the best known cost", not exact match.
* **Use the steps as candidates**: price each deep-supervision step's schedule with the LP and keep the cheapest.

**3. No feasibility guarantee.**
* **Feasible by construction for unit-level constraints**: the network outputs scores; each unit's (or each group's)
  schedule is decoded by a dynamic program over an on/off state machine with min up/down and the initial status
  (Viterbi). Train it with a structured loss (perturbed or smoothed DP, or a structured hinge).
* **Adequacy**: the existing block adequacy repair, or price each hour's demand inside the DPs. This is classical
  Lagrangian relaxation of unit commitment: the network can predict the hourly prices, unit DPs respond, and the
  recursion becomes a learned dual ascent.
* **Network**: the dispatch LP with priced slacks always returns a dispatch; arm B feeds its shortfalls and prices
  back into the recursion.
* **Safety net**: if the LP shows shedding or reserve shortfall, or the bound gap of §8 Z1 is large, fall back to a
  reduced MILP around the model's schedule, or to the full MILP. Report the fallback rate: the end-to-end speed holds
  if fallbacks are rare.

**4. A high bar on little compute.**
* **Data is not scarce here**: scenarios are unlimited and polished labels cost about 5 s of CPU, so 50–100k
  labelled instances take 70–140 CPU-hours as cluster array jobs, far more than TRM's 1,000 Sudokus.
* **Start from a good answer**: initialize *y* from the one-shot ensemble or the LP relaxation, so the recursion
  learns corrections.
* **Avoid overthinking and allow extrapolation**: recall and a progressive loss (Bansal et al. 2022), EMA; train on
  12 h and run longer on 24 h.
* **GPU scale** on PACE ICE (job scripts in `scripts/pace/`); the CPU runs here decide what is worth scaling.

## Protocol (shared by both arms)

* **Data**: uc12 (RTS-GMLC, 73 units, 12 h, DC network); training on the existing training sets and polished labels;
  validation on val + val_extra + val_extra2 (360); test once on test_fresh (120).
* **Metrics**: share of instances served (no shedding, no reserve shortfall), gap to the full MILP's dual bound (mean,
  median, max), wall time including every overhead, speed-up against the full MILP; paired bootstrap CIs.
* **Baselines on the same instances**: LP rounding + repair + 5 LPs (no learning); the one-shot temporal-GNN ensemble
  of §7 Y3 used end to end with the same repairs and LPs (equal-compute comparison); the REINFORCE end-to-end pipeline;
  the full MILP at 0.1 % and 0.5 %; the recommended MILP pipeline (0.30 % at ~10×) as the ceiling.
* **Questions**: does recursion beat a one-shot model of the same size? Does test-time compute help (more steps, better
  schedules)? Does the solver feedback (arm B) help? Does it extrapolate: train on 12 h, run on 24 h with more steps;
  train on the original ratings, run on the congested benchmark (§8 Z3)?
* **Compute**: four CPU cores, no GPU. Models are scaled down (well under 1M parameters) and training budgets are
  hours, not GPU-days; this is a feasibility study, not a replication of TRM's scale.

## What would count as success

An end-to-end model that serves ≥ 95 % of test instances at ≤ 0.5 % mean gap in well under a second (one network
pass and one or a few LPs) would be 50–100× faster than the full MILP and several times faster than the best
MILP-based pipeline, at similar quality. Beating the learning-free LP rounding baseline (1.6 %) and the one-shot
model is the minimum bar.

## Added after the literature check (2026-10-17)

**A differentiable feasibility layer.** Xu et al. 2026 (GATCN-PICC, IJEPES 177; see
[`../literature_uc.md`](../literature_uc.md)) put a deterministic min up/down + ramp + balance projection
*inside* the forward pass, so the constraint layer is on the gradient path and the model trains end to end
against it. Here the equivalent repairs (Viterbi, block adequacy, dispatch LP) sit outside the gradient,
which is why cost-aware fine-tuning uses REINFORCE. A differentiable relaxation of the min up/down
projection would allow direct gradient training through the repair, and is the most concrete untried idea
from the surrounding literature. Their evaluation is not comparable with ours (dispatch labels; the speed
reference is a QP repair, not a MILP solve), but the architectural point stands on its own.
