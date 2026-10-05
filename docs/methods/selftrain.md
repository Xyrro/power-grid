# Self-training with the solver as teacher (B2, 12-hour UC)

*Code: [`otsl/selftrain.py`](../../otsl/selftrain.py), [`scripts/uc_selftrain.py`](../../scripts/uc_selftrain.py),
[`scripts/uc_selftrain_report.py`](../../scripts/uc_selftrain_report.py). Raw numbers:
[`results/uc12/selftrain_results.md`](../../results/uc12/selftrain_results.md) / `.json`, log `results/uc12/selftrain_run.log`.*

**Summary.** Starting from the label-free model (imitation of the repaired LP relaxation), three rounds of
"predict → fix 80 % of the label-consistent decisions → 15 s reduced MILP → keep the schedule if the dispatch LP says
it is cheaper → retrain" turn the 500 training labels from 35 % served / 12 % median gap into 91 % served / 0.08 %
median gap to the MILP (35 % at or below the MILP's cost), **without any full-MILP solve and at 37 % of the label
compute of MILP labels (1.7 vs 4.7 core-hours)**. On the B2 test set (120 instances for LP decoders, 40 for
fixing) the self-trained model matches the MILP-label model as a one-shot / screening predictor and is a much better
fixing ranker at 95 % fixed (1.5 % vs 20.6 % mean gap, 12×). Followed by REINFORCE it is the best one-LP pipeline so
far (87.5 % served one-shot, 90 % with repair or screening), **but self-training does not remove the REINFORCE
over-commitment** (+1.8 units per hour, +8.5 % on served instances). Adding the solver labels to the REINFORCE group
(self-imitation) cuts that excess by 80 % and gives the best label-free 90–95 % fixing (0.52 % at 4.1×, 1.38 % at
6.4×), at the price of one-shot coverage. Learning to Fix's < 0.5 % at > 20× is not reached.

## 1. Method

**Goal.** Train Model 1 (the LP-feature GNN of V2/V4 in [`RESEARCH.md`](../RESEARCH.md)) to MILP-label quality without
solving a single full MILP for training, and check whether better labels remove the over-commitment that REINFORCE
fine-tuning introduces on B2.

**Round 0** (exactly `scripts/uc_label_free.py`, stage 1, same seed): label = rounded LP relaxation after adequacy and
min up/down repair; BCE, 80 epochs. Every label is scored once with the exact dispatch LP (its cost, shedding and
reserve shortfall).

**Round r = 1..R** (R = 3). The 500 training instances are split at random into R chunks; round r relabels chunk r with
the model of round r − 1:

1. predict probabilities `p` on the chunk;
2. among the decisions on which the rounded prediction **agrees with the current label**, fix the q most confident
   ones (|p − 0.5|), to the label's value; then the adequacy guard of `scripts/uc_fixing.py` releases OFF-fixed units
   (merit order) in any hour where the units not fixed off cannot cover net load + reserve + 5 %. Because every
   fixed value equals the current label, **the current label stays feasible in the reduced MILP** (the neighbourhood
   is the RINS idea, Danna et al. 2005, with the model's prediction in place of the LP relaxation);
3. solve the reduced MILP (HiGHS, gap 1e-3, **15 s limit**), score its schedule with the dispatch LP;
4. replace the label if it is cheaper (strictly); otherwise keep it;
5. retrain Model 1 from scratch on the whole label pool (BCE, 80 epochs, fixed schedule, final weights — no
   validation labels are needed, so validation stays free of solver labels).

q is chosen once, by a pilot on 30 training instances of chunk 1 (q = 0.8 vs 0.9, round-0 model): the setting with the
better label quality wins, the cheaper one within 0.1 pp. Both pilot runs count as label cost.

**End with REINFORCE** (120 steps, same settings as the label-free baseline): (a) plain RLOO with the dispatch-LP
critic; (b) **with self-imitation of the solver labels** (`train_reinforce_sil`): the instance's pool label is added
to its group of sampled commitments with its known LP cost; samples get the usual leave-one-out advantage, the label
gets its advantage clipped at zero (it is pushed up only when it beats the policy's own samples, as in self-imitation
learning, Oh et al. 2018). The anchor is in the units of the policy-gradient term, so there is no weight to tune.

**Selection without MILP information.** Validation metrics use the LP-relaxation bound `c_rel` as reference (no
validation MILP objective, no validation labels). The final self-trained model is the last round (fixed in advance).
Training MILP objectives appear only as a diagnostic of label quality (*diag* columns).

**Test protocol.** One LP per decoder on all 120 test instances: top-1 (+ min up/down repair), + adequacy repair,
candidate screening (thresholds 0.2..0.8 + 8 samples, best by LP). Fixing + reduced MILP on the first 40 test
instances (60 planned; cut after a container restart): each worker solves the full MILP (60 s, 0.1 %) and then every reduced MILP of the same instance, back to
back, in the same process (2 workers; the machine was shared with two other jobs, so compare ratios). Rankings:
symmetric (RACLearn, |p − 0.5|) and asymmetric (OFF errors × 10) + adequacy guard, as in `scripts/uc_fixing.py`.

## 2. Prior work and what is new here

Searched 2026-10-05 (abstracts and snippets; arXiv and publisher full texts were not reachable).

| work | what it does | difference from this study |
|---|---|---|
| DAgger (Ross, Gordon & Bagnell 2011, [1011.0686](https://arxiv.org/abs/1011.0686)) | the expert labels the states the learner visits; aggregate the data; retrain | instances here are i.i.d. (no trajectory, no covariate shift); what the learner "visits" is the neighbourhood its confidence defines, and the "expert" is a time-limited reduced MILP that does not know the optimum |
| Expert iteration (Anthony, Tian & Barber 2017, [1705.08439](https://arxiv.org/abs/1705.08439)); AlphaZero | a search guided by the policy produces better targets, the policy imitates them | the search is a MILP solver restricted by the policy's confidence; a target replaces the old one only if the exact dispatch LP says it is cheaper (monotone label pool) |
| Self-improvement for neural CO (Pirnay & Grimm, TMLR 2024, [2403.15180](https://arxiv.org/abs/2403.15180); Luo et al. 2024, [2403.19561](https://arxiv.org/abs/2403.19561)) | sample from the policy (or reconstruct locally), keep the best solutions as pseudo-labels; TSP / CVRP / JSSP | no solver in the loop; constructive sequence models; no constraints beyond the routing structure |
| Neural Diving (Nair et al. 2020, [2012.13349](https://arxiv.org/abs/2012.13349)); Predict-and-Search (Han et al., ICLR 2023, [2302.05636](https://arxiv.org/abs/2302.05636)); ConPaS (Huang et al., ICML 2024, [PMLR](https://proceedings.mlr.press/v235/huang24f.html)) | learn a (partial) assignment from **full-solver** solutions (ConPaS: positive and negative solutions collected with the solver); at test time fix / trust region + solve the sub-MILP | same test-time use (reduced MILP); the training labels here never come from a full solve |
| Neural LNS (Sonnerat et al. 2021, [2107.10201](https://arxiv.org/abs/2107.10201)); SPL-LNS (2025, [2508.16171](https://arxiv.org/abs/2508.16171)) | learn which variables to free in large-neighbourhood search: imitation of a local-branching expert (Neural LNS) or self-generated, hindsight-relabelled search data (SPL-LNS) | they learn a test-time neighbourhood-selection policy; here neighbourhood search is used offline to manufacture labels for a one-shot / fixing predictor |
| Apollo-MILP (Liu et al., ICLR 2025, [2503.01129](https://arxiv.org/abs/2503.01129)) | test-time alternation: predict, trust-region search gives a reference solution, fix variables on which prediction and reference agree | the same "fix where the model and an incumbent agree" neighbourhood, but used at training time to improve labels; Apollo's predictor is trained on solver solutions |
| RINS (Danna, Rothberg & Le Pape 2005, Math. Prog. 102) | sub-MIP fixing the variables on which the incumbent and the LP relaxation agree | the model's prediction replaces the LP relaxation |
| Cheap Thrills (Nguyen, Ellinas, Bhagavathula & Donti 2026, [2603.05495](https://arxiv.org/abs/2603.05495)) | cheap imperfect labels for supervised pre-training, then self-supervised refinement; up to 59x lower offline cost (incl. power-grid operation) | the two-stage logic of V4 here (LP-relaxation labels + LP-critic REINFORCE); this study adds a third ingredient, solver-improved labels, on a discrete problem |
| Self-imitation learning (Oh et al. 2018, [1806.05635](https://arxiv.org/abs/1806.05635)) | imitate one's own past good trajectories, weighted by the clipped advantage | used here with the solver's labels inside the RLOO group |
| UC: RACLearn (Park et al. 2024, [2211.15755](https://arxiv.org/abs/2211.15755)); Learning to Fix (Fritz et al. 2026, [2609.39396](https://arxiv.org/abs/2609.39396)); Qin & Yu 2023 ([2311.15216](https://arxiv.org/abs/2311.15216)); Yang, Li & Jian 2025 ([2505.14408](https://arxiv.org/abs/2505.14408)); He et al. 2026 ([doi](https://doi.org/10.1049/gtd2.70405)); few-sample learning to branch for UC ([Appl. Sci. 15:3366](https://doi.org/10.3390/app15063366)) | all train on full-MILP solutions or solver traces: confidence fixing (RACLearn), cost-aware per-unit thresholds (Learning to Fix), neural diving + branching (Qin & Yu), GNN initial commitment + learned neighbourhood for test-time LNS (Yang et al.), behaviour cloning + PPO repair with LP dispatch feedback (He et al.), imitation of branching decisions (Appl. Sci.) | no UC work was found that trains the commitment predictor **without any full-MILP solution** by iterated reduced-MILP relabelling. The "DAgger-style" label collection attributed to Qin & Yu in `RESEARCH.md` §2 could not be confirmed from the abstract (it probably concerns their neural-branching data) |

**What is new** (to the extent abstracts show): (1) for UC, a commitment predictor trained with no full-MILP label,
whose labels are improved by reduced MILPs around the model's own confident and label-consistent decisions, so that
each label is a feasible point of its reduced MILP and the label pool improves monotonically; (2) the label-cost
accounting against full-MILP labels on the same instances; (3) the solver labels used inside the REINFORCE group as a
self-imitation anchor, aimed at the over-commitment of LP-critic fine-tuning. **Not new**: expert iteration / DAgger,
RINS-style agreement neighbourhoods, confidence fixing, self-imitation.


## 3. Results

Single seed (0). Timings come from a machine shared with two other experiments (compare ratios, not seconds). A
container restart interrupted the first ST + REINFORCE run at step ~100; it was rerun from the saved round-3 model
with the same seed (the RL trajectory is deterministic: the logged sampled gaps at steps 25–75 match the interrupted
run exactly).

### 3.1 Labels

Label pool on the 500 training instances (the *diag* gap uses the training MILP objective as a diagnostic only):

| label pool | served (no shed / shortfall) | diag. median gap to MILP | diag. mean gap | diag. mean gap, served | label ≤ MILP cost | units on (MILP 13.39) |
|---|---|---|---|---|---|---|
| round 0: repaired LP relaxation | 35.4 % | 12.2 % | 155 % | 0.89 % | 7.0 % | 13.12 |
| after round 1 | 53.6 % | 1.72 % | 108 % | 0.68 % | 13.0 % | 13.16 |
| after round 2 | 69.4 % | 0.46 % | 53 % | 0.49 % | 22.4 % | 13.23 |
| **after round 3 (all 500 relabelled once)** | **90.6 %** | **0.080 %** | 5.6 % | **0.41 %** | **34.8 %** | 13.32 |

Per round, on the chunk that round relabels (each chunk is a random third of the training set):

| round | model used | solved | label replaced | mean reduced-MILP time | hit the 15 s limit | chunk served before → after | chunk diag. median gap before → after | chunk labels ≤ MILP after |
|---|---|---|---|---|---|---|---|---|
| pilot (30, q = 0.8 and 0.9) | round 0 | 60 | 87–93 % | 10.6–11.6 s | 50–53 % | 26.7 → 93.3 % (q = 0.8), 76.7 % (q = 0.9) | 14.6 → 0.39 % (0.8), 0.52 % (0.9) | 23 % / 20 % |
| 1 | round 0 | 137 | 84 % | 9.8 s | 45 % | 47.9* → 90.4 % | 4.5* → 0.23 % | 27 % |
| 2 | round 1 | 167 | 90 % | 11.5 s | 55 % | 41.3 → 88.6 % | 8.9 → 0.080 % | 32 % |
| 3 | round 2 | 166 | 93 % | 11.4 s | 54 % | 28.9 → 92.8 % | 15.4 → **0.011 %** | **45 %** |

\* chunk 1 includes the 30 pilot instances, already relabelled. q = 0.8 was chosen by the pilot (4.2 % vs 10.2 % mean
gap to the LP bound). Reading: **one reduced MILP per instance turns a poor label into a near-MILP label**, and the
labels get better from round to round although each chunk starts from equally poor round-0 labels: a better model
proposes a better neighbourhood (0.23 → 0.080 → 0.011 % median; 27 → 32 → 45 % at or below the MILP's cost). 52
instances (10 %) kept their round-0 label (no cheaper schedule found). Unit-hour agreement with the MILP labels
does not change (98.4 % → 98.3 %): the label errors that matter are a handful of unit-hours (a missing unit for
reserve, a peaker on too long). Exact agreement with the canonicalised MILP schedule rises from 5.4 % to 24.4 %, and
the self-trained labels are less arbitrary under identical-unit symmetry than the MILP's (41.8 % vs 57.8 % change
when canonicalised).

**Label cost** (solver core-seconds): LP relaxations 177, scoring the round-0 labels 121, pilot 683, round reduced
MILPs 5,143, round dispatch LPs 118: **6,243 core-s (1.73 core-h) = 36.7 % of the 17,018 core-s (4.73 core-h) of
the 500 full-MILP labels (2.7× cheaper)**. Half of the reduced MILPs stopped at the 15 s limit, so the saving per
label is ~3× (11 s vs 34 s), not the 87× of pure relaxation labels. REINFORCE adds ~11,500 dispatch LPs per
120-step run (training compute, not labels; 23–25 min wall each with 2 workers).

### 3.2 Validation (no MILP information; gap to the LP-relaxation bound)

| model | top-1: served / median gap | + adequacy repair: served / median gap | units on |
|---|---|---|---|
| LF-BCE (round 0) | 11.7 % / 35.8 % | 33.3 % / 19.7 % | 12.3 |
| ST round 1 / 2 / 3 | 18.3 / 25.0 / 26.7 % | 40.0 / 50.0 / 46.7 % (median 9.2 / 4.6 / 5.3 %) | 12.3–12.5 |
| LF + REINFORCE | 90.0 % / 5.10 % | 90.0 % / 5.10 % | 14.0 |
| ST + REINFORCE | 88.3 % / 5.06 % | 91.7 % / 4.64 % | 14.0 |
| ST + REINFORCE + SIL | 61.7 % / 3.52 % | 63.3 % / 2.69 % | 12.9 |

The fixed-epoch BCE trainer used in rounds 1–3 gives the same validation numbers as round 0's early-stopped trainer
(control row in `selftrain_results.md`), so round-to-round changes come from the labels.

### 3.3 Test, Model 1 → LP (120 instances; MILP: 92.5 % served, 14.20 units on per hour)

| Model 1 (B2) | MILP solves in training | top-1: served / median gap / mean gap served | + adequacy repair: served / median gap | screening (~14 LPs): served / median gap / mean gap served | units on − MILP (top-1) |
|---|---|---|---|---|---|
| (i) MILP-label BCE (`uc_model1_4`, RACLearn-style predictor) | 500 | 16.7 % / 57 % / 0.62 % | 43.3 % / 5.27 % | 62.5 % / 1.65 % / 1.86 %† | −0.18 |
| LF-BCE (round 0) | 0 | 20.8 % / 18.8 % / 0.60 % | 39.2 % / 7.35 % | 53.3 % / 1.59 % / 1.34 %† | −0.21 |
| **ST round 3 (self-trained BCE)** | **0** | 25.8 % / 16.9 % / 0.97 % | 42.5 % / 6.30 % | **66.7 % / 1.46 %** / 1.50 % | −0.14 |
| (iii) MILP-label + REINFORCE (`uc_model1_rl`) | 500 | 80.8 % / 5.27 % / 9.07 % | 86.7 % / 4.53 % | 87.5 % / 2.06 % / 4.67 %† | +1.81 |
| (ii) LF + REINFORCE | 0 | 84.2 % / 4.45 % / 9.22 % | 86.7 % / 4.10 % | 88.3 % / 1.87 % / 4.76 %† | +1.69 |
| **ST + REINFORCE** | **0** | **87.5 % / 3.95 %** / 8.53 % | **90.0 % / 3.92 %** | **90.0 %** / 1.91 % / 4.74 % | +1.77 |
| **ST + REINFORCE + SIL** | **0** | 55.8 % / **3.04 % / 2.45 %** | 71.7 % / **1.99 %** / 2.22 % | 81.7 % / **1.23 % / 2.62 %** | **+0.36** |

† screening rows of the baselines are from `results/uc12/uc_label_free_results.md` (same decoder, a different random
draw of the 8 sampled candidates); all other rows are from this run (the reproduced baseline rows match the earlier
runs to the digit).

* **Self-trained BCE ≈ MILP-label BCE, with zero MILP labels**: one-shot 25.8 vs 16.7 %, with adequacy repair 42.5 vs
  43.3 %, screening 66.7 % / 1.46 % vs 62.5 % / 1.65 %. Imitation-trained models still under-commit by ~0.15 units
  per hour and shed in most instances; better labels do not change that.
* **Self-training + REINFORCE is the best one-LP pipeline so far on B2** (87.5 % one-shot, 90.0 % with adequacy repair or
  screening; median gap 3.9 %), slightly ahead of both REINFORCE baselines — but single seed, 120 instances: +3.3 pp
  over LF + REINFORCE one-shot is 4 instances.
* **It does not fix the over-commitment**: ST + REINFORCE commits +1.77 units per hour above the MILP and costs 8.5 %
  more on served instances, like both REINFORCE baselines (+1.69 to +1.81 units, 9.1–9.2 %). The over-commitment
  comes from the critic stage (the stochastic policy hedges against VOLL), not from the quality of the imitation start.
* **Self-imitation of the solver labels (SIL) cuts the over-commitment by ~80 %** (+0.36 units; mean gap on served
  instances 2.2–2.6 % instead of 8.4–9.2 %; best median gaps: 1.99 % with adequacy repair, 1.23 % with screening)
  **but gives back coverage** (55.8 / 71.7 / 81.7 % served). It moves along the cost–coverage trade-off rather
  than beating the REINFORCE models on both axes.

### 3.4 Test, fixing + reduced MILP

First **40** test instances (time budget after a container restart; the earlier B2 fixing study used 60). Each
worker solved the full MILP and then all 18 reduced MILPs of an instance back to back. The full MILP took 31.7 s on
average, serves 87.5 % of these instances and reproduced the reference objective on all 40. "Symmetric" means
RACLearn's |p − 0.5| ranking; "asym + guard" means OFF errors × 10 plus the adequacy guard.

| fixed | model (ranking) | MILP labels | mean gap | median gap | instances > 1 % / > 5 % | worst | served | time | speed-up |
|---|---|---|---|---|---|---|---|---|---|
| – | full MILP (60 s, 0.1 %) | – | 0 | 0 | – | – | 87.5 % | 31.7 s | 1× |
| 80 % | (i) MILP-label BCE, symmetric (RACLearn) | 500 | **0.006 %** | 0 | 1 / 0 | 1.7 % | 87.5 % | 17.1 s | 1.9× |
| 80 % | ST round 3, symmetric | 0 | 0.45 % | 0 | 4 / 2 | 6.9 % | 87.5 % | 14.3 s | 2.2× |
| 90 % | (i) MILP-label BCE, symmetric (RACLearn) | 500 | 1.91 % | 0 | 4 / 3 | 55 % | 82.5 % | 11.1 s | 2.8× |
| 90 % | (i) MILP-label BCE, asym + guard | 500 | **0.38 %** | 0.032 % | 6 / 0 | 2.9 % | 95.0 % | 7.3 s | 4.3× |
| 90 % | (iii) MILP-label + REINFORCE, symmetric | 500 | 2.90 % | 0.45 % | 14 / 6 | 32 % | 92.5 % | 2.5 s | 12.8× |
| 90 % | (ii) LF + REINFORCE, symmetric | 0 | 2.41 % | 0.51 % | 15 / 6 | 18 % | 95.0 % | 2.5 s | 12.7× |
| 90 % | ST round 3, symmetric | 0 | 0.95 % | 0 | 6 / 3 | 18 % | 87.5 % | 9.7 s | 3.3× |
| 90 % | ST round 3, asym + guard | 0 | 0.62 % | 0.016 % | 7 / 1 | 8.4 % | 92.5 % | 5.9 s | 5.3× |
| 90 % | ST + REINFORCE, symmetric | 0 | 2.57 % | 0.41 % | 14 / 10 | 17 % | 95.0 % | 2.7 s | 12.0× |
| 90 % | **ST + REINFORCE + SIL, symmetric** | 0 | **0.52 %** | **0.007 %** | 4 / 1 | 14 % | 87.5 % | 7.7 s | 4.1× |
| 95 % | (i) MILP-label BCE, symmetric (RACLearn) | 500 | 53.0 % | 0.037 % | 8 / 5 | 759 % | 77.5 % | 4.4 s | 7.2× |
| 95 % | (i) MILP-label BCE, asym + guard | 500 | 20.6 % | 0.34 % | 11 / 5 | 565 % | 85.0 % | 3.1 s | 10.4× |
| 95 % | (iii) MILP-label + REINFORCE, symmetric | 500 | 5.54 % | 1.60 % | 23 / 13 | 49 % | 92.5 % | 0.62 s | **51×** |
| 95 % | (ii) LF + REINFORCE, symmetric | 0 | 4.46 % | 1.00 % | 20 / 10 | 29 % | 95.0 % | 0.82 s | 39× |
| 95 % | ST round 3, symmetric | 0 | 10.9 % | 0.022 % | 9 / 4 | 392 % | 82.5 % | 4.5 s | 7.1× |
| 95 % | **ST round 3, asym + guard** | 0 | **1.52 %** | 0.20 % | 12 / 5 | 18 % | 90.0 % | 2.7 s | **11.9×** |
| 95 % | ST + REINFORCE, symmetric | 0 | 5.33 % | 1.53 % | 23 / 12 | 29 % | 92.5 % | 1.0 s | 30× |
| 95 % | **ST + REINFORCE + SIL, symmetric** | 0 | **1.38 %** | **0.027 %** | 10 / 2 | 19 % | 87.5 % | 5.0 s | 6.4× |

(SIL at 95 %: one instance needed the fallback re-solve without fixings; its time is included.)

* **At 95 % fixed, self-training removes the catastrophic tail of the BCE ranking**: with the same asymmetric
  ranking and guard, the mean gap is 1.52 % instead of 20.6 % (worst instance 18 % instead of 565 %) at the same
  speed-up (11.9× vs 10.4×). With symmetric ranking it is 10.9 % vs 53 %. The SIL model gives 1.38 % (median
  0.027 %) at 6.4×. (The 60-instance study in `RESEARCH.md` V3 had 38.5 % / 14.0 % for the two BCE rankings.) They are also cheaper in
  mean gap than the REINFORCE rankings (4.5–5.5 %), which remain the fastest (30–51×) but are wrong by > 1 % on half
  of the instances.
* **At 90 % fixed, nothing beats the MILP-label BCE with asym + guard** (0.38 %, 4.3×). The best label-free rows are
  SIL symmetric (0.52 %, median 0.007 %, 4.1×) and ST asym + guard (0.62 %, 5.3×), so they match it within noise
  (one instance moves the mean by several tenths of a percent), without MILP labels.
* **At 80 %, the MILP-label BCE is better** (0.006 % vs 0.45 %; two ST instances lose 5–7 %).
* The over-committing REINFORCE models make the fastest reduced MILPs (their confident decisions are mostly ON,
  which leaves a small, easy problem) but the costliest schedules. ST + REINFORCE behaves like the two REINFORCE
  baselines here too.


## 4. Verdict

* **Against the MILP-label pipeline (RACLearn-style predictor, baseline (i))**: self-training reaches the same
  one-shot / screening quality **with no full-MILP label, at 37 % of the label compute (2.7× cheaper)**, and its
  confidence ranks fixings better at aggressive ratios: 1.5 % instead of 20.6 % mean gap at 95 % fixed (same rule,
  11.9×). Up to 90 % the MILP-label model is as good or better (80 %: 0.006 % vs 0.45 %).
* **Against Learning to Fix** (Fritz et al. 2026: < 0.5 % mean gap at > 20×, with MILP labels and cost-aware per-unit
  thresholds): not reached. The best label-free operating points are 0.52 % at 4.1× (SIL, 90 %), 0.62 % at 5.3×
  (ST, 90 %, asym + guard) and 1.4–1.5 % at 6–12× (95 %). The two ideas are complementary: Learning to Fix's
  per-unit thresholds could be calibrated on self-trained models (our reduced-MILP machinery already measures fixing
  errors), not tested.
* **Against our current best** (label-free + REINFORCE, baseline (ii)): ST + REINFORCE is a small step up for the
  one-LP pipeline (87.5 vs 84.2 % one-shot, 90.0 vs 86.7 % with adequacy repair, 90.0 vs 88.3 % with screening;
  single seed). **Self-training does not fix the over-commitment** (+1.77 units per hour, +8.5 % cost on served
  instances, the same as before). Adding the solver labels to the REINFORCE group (SIL) cuts the excess units by 80 %
  and the served-instance cost gap from 8.5 % to 2.2–2.6 %, at the price of coverage (55.8 % one-shot, 81.7 % with
  screening). On the hybrid, SIL is the most reliable aggressive-fixing ranker tested (95 %: 1.38 % mean, 0.027 %
  median).
* **Recommendation**: if MILP labels are unaffordable, self-train (3 rounds, q = 0.8, 15 s reduced MILPs) and use the
  self-trained model for fixing (asym + guard at 90–95 %), and the SIL fine-tune when cheap schedules matter more than
  one-LP coverage. Do not expect self-training to remove the REINFORCE over-commitment; that needs a change in
  the critic stage (SIL or a constrained formulation).


## 5. Limitations

* **Single seed, small test sets** (120 instances for LP decoders, 40 for fixing): differences of a few points
  are within noise; no confidence intervals were computed.
* **Timings** are from a 4-core machine shared with two other experiments (6+ busy processes); speed-ups are ratios
  of back-to-back solves in the same process, which cancels most but not all of the load variation.
* **Label cost is only 2.7× below MILP labels**, because half of the 15 s reduced MILPs hit the limit. Shorter limits,
  q = 0.85–0.9 in later rounds, or a warm start from the current label (not available in SciPy's HiGHS interface)
  would cut it; not tested. Rounds relabelled each instance once; revisiting instances was not tried.
* **Hyper-parameters not tuned**: q chosen by a 30-instance pilot on training data; rounds, time limit, epochs,
  REINFORCE steps and the SIL weighting were fixed in advance; the final self-trained model is the last round.
  The REINFORCE variant is not selected on validation either: plain REINFORCE is better on validation coverage, SIL on
  validation cost, and both are reported.
* The reference for test gaps is the time-limited full MILP (82.5 % proven optimal), so negative gaps are real.
* Training MILP objectives were read only to report label quality; the method itself never needs them.

