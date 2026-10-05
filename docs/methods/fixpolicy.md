# Fixing policy trained on solver outcomes (B2, 12-hour network-constrained UC)

Code: [`otsl/fixpolicy.py`](../../otsl/fixpolicy.py), `scripts/uc_fixpolicy_{crossfit,label,train,eval,report}.py`.
Results: [`results/uc12/fixpolicy_results.md`](../../results/uc12/fixpolicy_results.md) (+ `.json`, `fixpolicy_pareto.png`).

## Question

Predict-and-fix (RACLearn-style) sets a share of the (unit, hour) commitment decisions to the rounded
prediction of Model 1 and solves the reduced MILP. Which decisions should be fixed? Published rules use the
predictor's confidence (RACLearn) or per-generator confidence thresholds calibrated on the cost impact of
fixing errors (Learning to Fix). Here the choice is learned from solver outcomes: a model of the
**expected cost of fixing each decision**, trained on counterfactual dispatch-LP solves, ranks the decisions,
and the cheapest-to-fix share is fixed.

## Method

**Ranking.** For decision j = (t, g) with Model 1 probability p_j and rounded value ŷ_j = 1[p_j > 0.5],

    r_j = P(fixing u_j := ŷ_j is harmful | x_j) · exp(E[log harm_j | harmful, x_j]),

and the share q of decisions with the smallest r_j is fixed (then the adequacy guard of `scripts/uc_fixing.py`,
which releases OFF-fixed units hour by hour until the units not fixed off cover net load + reserve + 5 %).

* **Harm head.** logit P(harmful) = α · logit(min(p, 1 − p)) + β + c(x): a learned correction of the
  probability model's own error odds. c is zero-initialised, so the policy starts at the RACLearn ranking and
  keeps its resolution among very confident decisions (a free-form classifier could not resolve error
  probabilities of 1e-4 vs 1e-6 and fixed 3× more wrong decisions at 80 % in a first attempt). A second MLP
  regresses log harm on the harmful decisions. Both: 2 × 32 SiLU, AdamW (lr 5e-4, wd 1e-2), 20 epochs, early
  stopping on the validation criterion below, 3 seeds averaged geometrically.
* **Features (42 per decision).** p, |p − 0.5|, logit, ŷ; LP-relaxation value u_rel, |u_rel − ŷ|, fractionality;
  hour, first/last hour; u0, ŷ at t − 1 and t + 1, distance to the nearest predicted switch, switches in the row,
  min / mean confidence of the unit's row, neighbour confidences; net load, reserve requirement, committed-capacity
  slack (as predicted, and if this unit were off / on), minimum-output floor slack, number of uncertain decisions
  in the hour; relaxed LMP at the unit's bus and its margin over the unit's average / first-segment cost; unit
  data (pmax, pmin/pmax, average cost, no-load, start-up, min up/down, ramp, type, identical-group size).

**Labels from solver outcomes** (`otsl/fixpolicy.py: harm_labels, compensated_harms`).

1. *Deployment-like errors.* The probability model is retrained in 4 day-grouped folds of the train set (same
   architecture, features, canonical labels and 80 epochs as `uc_model1_4.pt`), so every train instance gets
   out-of-fold probabilities: 15.8 wrong decisions per instance (val: 14.8, test: 13.6; in-sample: 8.8).
2. *Symmetry.* The MILP optimum u* is aligned to the prediction inside groups of identical units with the same
   initial status (assignment problem), so a swapped copy is not an error.
3. *Counterfactual harm.* For every decision with ŷ_j ≠ u*_j (aligned): force u_j := ŷ_j in u*, repair the unit's
   row to the nearest schedule that satisfies min up/down with that entry fixed (constrained DP), and price it
   with the exact dispatch LP: harm_j = (LP(u') − LP(u*)) / LP(u*). Decisions the model gets right have harm 0.
4. *Compensation.* A wrong OFF fix forces reserve shortfall or shedding only if nothing can replace the unit;
   in a reduced MILP free units can. For wrong OFF fixes the lost capacity is replaced by units off in u*
   ((a) merit-order greedy, (b) the cheapest single unit at least as large; their rows repaired), and the cheapest
   of uncompensated / (a) / (b) is the label. This cuts the median OFF-error harm from 9.3 % to 1.6 %.

Cost: 560 instances (500 train out-of-fold + 60 val), 26,743 dispatch LPs, ≈ 1.6 core-hours; no MILP is solved
for labels. Training the harm model: ≈ 1 min.

**Selection on validation only.** The harm models are early-stopped on the summed measured harm of the decisions
they would fix at 80/90/95/97 % on the 60 val instances (compensated labels: 0.17 vs 0.36 for the confidence
ranking). Label variant and guard were chosen with reduced MILPs on the first 10 val instances (table at the end
of the results file): uncompensated labels make the policy leave free almost only OFF decisions of large units
(94 % of the free set), which is slower and worse (mean gap 1.4 / 4.2 % at 90 / 95 %); compensated labels gave
0.21 / 1.6 %; the guard changes nothing at 90–95 % and removes the one shortfall outlier at 97 %.

**Post-hoc extension: solver-aware guard** (`release_conflicting_rows`, `lp_guard`). Added after the first
test pass showed a few catastrophic outliers (load shedding + reserve shortfall at 95–97 % fixed) that the
capacity guard cannot see (min down-time timing, network, ramping). Two parameter-free checks before the MILP:
(i) release all fixings of a unit whose fixed entries cannot be completed under min up/down (the reduced MILP
would be infeasible and fall back to the full MILP); (ii) solve the LP relaxation of the reduced problem (fixed
decisions at their values, free ones in [0, 1]); its cost bounds the reduced MILP's from below, so if even the
relaxation pays penalty-priced shedding / over-generation / reserve shortfall, the fixings force such penalties
on the reduced MILP. Then release the OFF fixes in [t − min_dn, t] of every hour t with slack (then ON fixes in
[t − min_up, t] if still needed), re-check, up to 4 rounds. The LP time
(≈ 0.15–0.5 s per check) is counted in the rule's time. It is applied to RACLearn as well, to separate the effect
of the guard from that of the ranking. It was checked on 10 val instances before the test run; nothing was tuned.

## Baselines (re-implemented on the same instances and the same probability model)

* **RACLearn** (Park et al. 2024): fix the share with the largest |p − 0.5|. The saved network has no dropout
  (p = 0), so MC-dropout confidence reduces to the probability margin.
* **Learning to Fix** (Fritz et al. 2026), from the abstract: generator-specific confidence thresholds from a
  per-generator decomposition subject to a cost tolerance on calibration instances. Implementation: for every
  generator g and threshold θ, the impact of fixing g's decisions with max(p, 1 − p) ≥ θ is measured with the
  dispatch LP (all other units at the aligned optimum, g's row repaired, conflicts with min up/down = infinite);
  θ_g is the lowest threshold whose mean relative impact over the 560 calibration instances (train
  out-of-fold + val) stays ≤ τ for every higher threshold too; τ is set so the val fixed share hits the target.
  Same solver-measured data as the proposed policy; what differs is the policy class (one context-free
  threshold per generator vs a per-decision, context-dependent expected cost). Calibrating on val only (60) looked
  better in-sample but is out-of-sample no better than RACLearn on the label proxy, so the pooled calibration is
  used. What we cannot reproduce: their kNN predictor and the paper's exact decomposition algorithm (only the
  abstract is reachable).
* **Asymmetric + adequacy guard** (ours, U5/V3): rank by min(p, 1 − p), OFF decisions × 10, then the guard.
* **REINFORCE probabilities** (ours, U4): rank and fix by the LP-critic fine-tuned model `uc_model1_rl.pt`.

## Results (first 60 B2 test instances)

Protocol: each worker solves the full MILP and then every reduced MILP of the same instance back-to-back
(60 s limit, 0.1 % gap, as in the dataset); 2 workers on a machine shared with two other jobs (load ≈ 6 on
4 cores), so absolute times are ≈ 1.4× those of the earlier study (full MILP 39.6 s here vs 28.3 s) and only
ratios are comparable. Gaps are to `test['obj']`. All baseline numbers reproduce the earlier study exactly
(e.g. RACLearn 1.384 / 38.509 %, asymmetric 0.413 / 14.004 %, REINFORCE 3.150 / 5.501 % mean gap at 90 / 95 %);
only the speed-ups moved with the load.

**Pass A — 90 / 95 / 97 % fixed** (full MILP 39.6 s, serves 90.0 %):

| fixed | rule | mean gap | median gap | > 10 % gap | served | speed-up |
|---|---|---|---|---|---|---|
| 90 % | RACLearn (margin) | 1.38 % | 0.000 % | 2 | 86.7 % | 2.2× |
| 90 % | Learning to Fix | 7.81 % | 0.339 % | 3 | 86.7 % | 2.5× |
| 90 % | asymmetric + guard (ours, previous best) | **0.41 %** | 0.055 % | 0 | **96.7 %** | 3.6× |
| 90 % | REINFORCE probabilities | 3.15 % | 0.474 % | 6 | 91.7 % | **7.8×** |
| 90 % | **harm policy + guard (proposed)** | 0.47 % | **0.002 %** | 0 | 90.0 % | 3.7× |
| 95 % | RACLearn (margin) | 38.5 % | 0.087 % | 7 | 80.0 % | 5.2× |
| 95 % | Learning to Fix | 49.4 % | 1.514 % | 14 | 63.3 % | 4.0× |
| 95 % | asymmetric + guard | 14.0 % | 0.268 % | 4 | 90.0 % | 7.4× |
| 95 % | REINFORCE probabilities | **5.5 %** | 1.482 % | 10 | **91.7 %** | **17.7×** |
| 95 % | **harm policy + guard** | 13.9 % | **0.052 %** | 4 | 88.3 % | 7.8× |
| 97 % | RACLearn | 54.0 % | 0.398 % | 10 | 63.3 % | 7.2× |
| 97 % | asymmetric + guard | 32.1 % | 0.468 % | 6 | 85.0 % | 12.2× |
| 97 % | REINFORCE | **7.4 %** | 2.426 % | 14 | 81.7 % | **14.8×** |
| 97 % | **harm policy + guard** | 20.1 % | **0.245 %** | 7 | 78.3 % | 8.2× |

(Learning to Fix was not run at 97 %: dominated on val, 64 % mean gap at 21.5 s.)

PASS_BC_PLACEHOLDER

## Verdict

VERDICT_PLACEHOLDER

## Limitations

* **Mean gaps are dominated by a few instances with load shedding / reserve shortfall** (VOLL $10,000/MWh): at
  95 % one instance contributes 11 of the policy's 13.9 % (2.6 % without it). The capacity guard misses them; the
  LP-relaxation guard removes them but releases many fixings (actual fixed share far below the target), because
  it also fires in hours where even the full relaxation pays reserve shortfall — comparing against the full
  relaxation's per-hour slack would release less (not tested).
* **Labels are single-error counterfactuals priced by an LP with heuristic compensation**, not reduced-MILP
  outcomes; interactions between several wrong fixes are only handled by the guards. A policy-gradient /
  bandit stage on reduced-MILP rewards was not run (a reduced MILP costs 1–60 s here; the label budget was 26.7k
  LPs ≈ 1.6 core-hours instead).
* **Speed is not modelled.** The policy optimises the expected cost of errors at a given share; its reduced MILPs
  are about as fast as the asymmetric rule's at the same share and slower than REINFORCE's, which over-commits
  and makes the MILP easy. At ≥ 10× speed-up REINFORCE ranking remains the best of the tested rules.
* Learning to Fix is re-implemented from its abstract (per-generator decomposition, cost tolerance) on our impact
  measurements; its published kNN predictor and exact algorithm may behave differently. Our impact measure fixes
  all other units at the optimum (no compensation), which makes its thresholds conservative for large units and
  its reduced MILPs slow (rows of risky units stay entirely free).
* Selection used only 10 val instances with reduced MILPs (plus 60 for early stopping on LP labels); one probability
  model, one system (RTS-GMLC, 73 units), one horizon (12 h). The machine was shared, so timings are noisy;
  every rule saw the same load per instance.

## Prior work and what is new

| work | which decisions are fixed | learned from | granularity |
|---|---|---|---|
| RACLearn — Park, Chen, Van Hentenryck, IEEE TPS 2024 ([arXiv 2211.15755](https://arxiv.org/abs/2211.15755)) | most confident (MC dropout) | prediction uncertainty | per decision, cost-agnostic |
| Learning to Fix — Fritz, Makrides, Fetanat, Pinson 2026 ([arXiv 2609.39396](https://arxiv.org/abs/2609.39396)) | confidence ≥ generator threshold | cost impact of fixing errors on validation (calibrated after training) | one threshold per generator |
| Xavier, Qiu, Ahmed, INFORMS JoC 2021 ([arXiv 1902.01697](https://arxiv.org/abs/1902.01697)) | per-variable classifier with precision / recall thresholds | error frequency | per variable |
| Yang, Li, Chen, Zheng, Appl. Sci. 15:4498, 2025 ([MDPI](https://www.mdpi.com/2076-3417/15/8/4498)) | "stable" variables: dual threshold on GNN confidence and root LP | branch-and-bound histories | per variable |
| He et al., IET GTD 2026 ([doi:10.1049/gtd2.70405](https://doi.org/10.1049/gtd2.70405)) | reliable / repairable split from root-relaxation records + confidence; PPO repair with LP dispatch feedback | LP cost feedback for the *repair* | per decision |
| Wang, Wu, Weng, Zhang 2026 ([arXiv 2604.02788](https://arxiv.org/abs/2604.02788)) | structurally stable binaries, solver-certified restriction | structure | per variable |
| Neural Diving — Nair et al. 2020 ([arXiv 2012.13349](https://arxiv.org/abs/2012.13349)) | SelectiveNet coverage head chooses which variables to assign | solution matching | per variable |
| Predict-and-Search (Han et al., ICLR 2023, [arXiv 2302.05636](https://arxiv.org/abs/2302.05636)); Apollo-MILP (ICLR 2025, [arXiv 2503.01129](https://arxiv.org/abs/2503.01129)) | trust region around the prediction; uncertainty-based error bound (UEBO) for fixing | prediction / uncertainty | per variable |
| Learned LNS: Sonnerat et al. 2021 ([arXiv 2107.10201](https://arxiv.org/abs/2107.10201)); CL-LNS, Huang et al., ICML 2023 ([PMLR](https://proceedings.mlr.press/v202/huang23g/huang23g.pdf)) | which variables to *free* around an incumbent, iteratively | imitation of local branching (solver outcomes) | per variable |

What is new here, precisely:

1. **The fixing score is a learned expected cost of a fixing error per decision**, P(error | x) · E[cost | error, x],
   conditioned on instance context (hour, adequacy slack, LP-relaxation agreement, schedule position, relaxed
   prices). Learning to Fix also uses the cost impact of errors, but as one confidence threshold per generator
   chosen after training; RACLearn, Neural Diving, Apollo-MILP and Xavier et al. use confidence, coverage,
   uncertainty or error frequency, which are blind to what an error costs.
2. **Labels are counterfactual solver outcomes on deployment-like errors**: one exact dispatch LP per wrong
   decision of an out-of-fold predictor, with a constrained min up/down repair, symmetry alignment of identical
   units, and a compensated variant for OFF errors. No MILP is needed to label (26.7k LPs, ≈ 1.6 core-hours).
   Learned-LNS methods also learn from solver outcomes, but choose neighbourhoods around an incumbent inside an
   iterative search by imitating local branching; here the policy acts once, before the MILP, without an incumbent.
3. **The error head is a correction of the predictor's own error log-odds**, so it inherits the predictor's
   resolution among confident decisions (necessary: a free-form head was worse than the confidence ranking).
4. A like-for-like comparison against RACLearn and a Learning-to-Fix re-implementation that share the probability
   model, the instances, the solver-measured impact data and back-to-back timing.

Not new: fixing confident predictions and solving the reduced MILP (RACLearn, Neural Diving); the adequacy guard
and the asymmetric ranking (U5); the use of LP dispatch cost as feedback (He et al. 2026, our U4).
