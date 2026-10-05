# Self-training with the solver as teacher (B2, 12-hour UC)

*Code: [`otsl/selftrain.py`](../../otsl/selftrain.py), [`scripts/uc_selftrain.py`](../../scripts/uc_selftrain.py),
[`scripts/uc_selftrain_report.py`](../../scripts/uc_selftrain_report.py). Raw numbers:
[`results/uc12/selftrain_results.md`](../../results/uc12/selftrain_results.md) / `.json`, log `results/uc12/selftrain_run.log`.*

SUMMARY_PLACEHOLDER

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
candidate screening (thresholds 0.2..0.8 + 8 samples, best by LP). Fixing + reduced MILP on the first 60 test
instances: each worker solves the full MILP (60 s, 0.1 %) and then every reduced MILP of the same instance, back to
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


RESULTS_PLACEHOLDER
