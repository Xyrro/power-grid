# rloop: recursion with the dispatch LP in the loop (arm B)

*Code: [`otsl/rloop.py`](../../otsl/rloop.py), `scripts/rec_rloop_{common,train,val,eval,report}.py`, Slurm jobs
`scripts/pace/rloop_*.sbatch`. Results: [`results/recursive/rloop_results.md`](../../results/recursive/rloop_results.md)
(+ `.json`), raw records `results/recursive/rloop_eval_<bench>_s<shard>.jsonl`, validation
`results/recursive/rloop_val_*.npz` and `rloop_val_summary.json`, training histories `results/recursive/rloop_train_<tag>.json`.
Checkpoints and logs (git-ignored): `data/generated/recursive/rloop/`.*

## Status

*Kept current for resuming after a container restart. Every step resumes: rerun the same command.*

- 23:45 smoke tests passed (training one epoch, RL, worker pool with in-flight batches, test evaluation on one
  instance each of uc12 and line_out).
- 23:51 validation baselines done (`scripts/rec_rloop_val.py --baselines --report`).
- 23:52 `main` (core 2) and 23:54 `nofb` (core 3) supervised training started, 10 epochs each (~5.3 min/epoch):
  `OTSL_THREADS=1 taskset -c 2 python3 scripts/rec_rloop_train.py --tag main --preset cpu`,
  `... -c 3 ... --tag nofb --preset cpu --no_feedback`; logs `data/generated/recursive/rloop/<tag>.{log,out}`.
- 00:15 diagnosis: under imitation the per-step training errors are identical (≈ 9.6 wrong decisions per instance at
  steps 0, 1, 2, with and without feedback); the model alone is worse than LP rounding on validation (epoch 4, 60
  instances: 3.43 % vs 1.86 %), seeded with the LP-rounding schedule it improves it slightly (1.75 %). Added: seeded
  refinement (inference `--start lpround`, RL `--p_seed`), an anchor skip from the priced schedule to the logits
  (`--anchor`), a sampling temperature for RL (`--rl_temp`), cost-weighted deep supervision (`--w_wrong`, `--w_grad`).
- 00:30 queues (scratchpad `rloop_q2b.sh`, `rloop_q3.sh`; each job resumes or no-ops if done):
  core 2: main → `main_rl` (RL) → `main_rls` (RL with seeded batches, anchor 2, temperature 0.5) → `norecall`;
  core 3: nofb → `nods` → `nofb_rl`. Commands: `python3 scripts/rec_rloop_train.py --tag main_rl --stage rl --init main`;
  `... --tag main_rls --stage rl --init main --anchor --anchor_init 2.0 --rl_temp 0.5 --p_seed 0.5`;
  `... --tag norecall --preset cpu --no_recall`; `... --tag nods --preset cpu --no_deep_sup`;
  `... --tag nofb_rl --stage rl --init nofb`.
- 00:55 `nods` (back-propagation through all 3 outer steps) reached 5.0 GB at batch 32: restarted from its epoch-2
  checkpoint with batch 16 × gradient accumulation 2 (same effective batch; `--accum`), ~2.8 GB. Core-2 queue had
  stalled (its wait pattern matched its own parent shell) and was restarted; `main_rl` ran 01:00–01:23.
- 01:10 full validation (360): whole-schedule recursion, supervised: `main` 8.05 % (K = 1) → 6.17 % (K = 16), `nofb`
  7.63 % → 6.95 %; seeded with LP rounding (2.14 %): `main` 1.87 %, `nofb` 1.88 %. Learning-free LP-gradient local
  search from LP rounding: 1.64 % with 16 LPs. Added: model-guided LP-verified search (`rloop.guided_search`,
  `rec_rloop_val.py --guide none|m1x|net --lam`), selection script `scripts/rec_rloop_select.py`.
- 01:30 λ sweep of the guided search on validation 0–119 (scratchpad `rloop_sweep1.sh`), whole-schedule validation
  of `main_rl`, `main_rls` and decoding biases (scratchpad `rloop_valq2.sh`).

## Summary

(to be written)

## Method

**Loop.** One outer step is one pass of a small recurrent network followed by one exact dispatch LP:

```
step 0:  (y, z) <- recursion(x, no feedback)              -> scores_0 -> s_0 = decode(scores_0) -> LP(s_0)
step k:  (y, z) <- recursion(x, feedback(LP(s_{k-1})))    -> scores_k -> s_k                    -> LP(s_k)
output:  the cheapest priced schedule among s_0 .. s_{K-1} (shedding and reserve shortfall priced in)
```

Every step's schedule is a priced candidate, so the output never gets worse with more steps; a step whose decoded
schedule was already priced costs no LP.

**Tokens and mixing.** One token per (unit, hour), carrying physics rather than indices: unit parameters (limits, cost
curve, start-up cost, min up/down, ramp, type), initial status, identical-group size, hourly system data (load, net
load, reserve requirement, renewables), the distance to the horizon's start and end (clipped at 12 h), and the LP
relaxation's commitment and price at the unit's bus. There are no unit-ID embeddings and no absolute hour embeddings.
One shared recursion block mixes along three axes: across hours for each unit (residual dilated 1-D convolutions,
dilations 1, 2, 4, kernel 3, length-agnostic), across units in the same hour (DeepSets mean pool), and over the bus
graph per hour (units averaged into their buses, interaction-network message passing over the 120 lines with an edge
gate that switches off lines out of service, gathered back to the units). Bus tokens carry load, renewables, capacity,
the relaxed price and a PTDF signature (the bus's column of the PTDF matrix projected on the 16 leading singular
vectors of the intact network's PTDF, recomputed when lines are out); line tokens carry susceptance, rating, relaxed
loading and an in-service flag.

**Recursion (TRM).** State per token: an answer y and a latent z (64-d). One outer step runs `n_latent = 2` latent
updates `z <- core(z + x_emb + fb_emb + W y)` and one answer update `y <- y + MLP(y, z)`. *Recall*: the input
embedding x_emb is re-injected at every latent update (Bansal et al. 2022); without recall it only initialises z.
(y, z) is carried across outer steps.

**Count heads for identical units.** 56 of the 73 units sit in 24 groups of identical copies at one bus. Per group and
hour the network outputs ordinal logits l_j = logit P(count > j), j = 0 .. m−1, from the group's mean-pooled answer
state, monotone by construction (l_j = l_{j−1} − softplus(·)). The copies are stacked in a fixed order (initially-on
units first, then by index): copy j is on iff the count exceeds j, so l_j is copy j's logit and its target is
[count* > j], with count* the label's count. This removes the exact symmetry (34–62 % of labels are arbitrary between
copies) without a rank feature; singletons are groups of size 1. Ablation: per-unit heads with a symmetry-rank input
and labels aligned to the prediction inside groups (Hungarian matching per group, as in `otsl.fixpolicy`).

**Decoding, feasible by construction for unit constraints.** Per unit, a Viterbi pass over the on/off state machine
(on for 1..U hours, off for 1..D hours; switching needs U on-hours or D off-hours; initial status without carry-over,
as in the MILP) finds the most likely schedule under the network's probabilities; it is vectorised over units with
equal (U, D) and takes 2.6 ms for 12 h. Then the block (min up/down-aware) adequacy repair of §5 W1 and the Hamming
min up/down repair as a safety net (a no-op after Viterbi unless the adequacy repair breaks a block). On 0/1 input
the Viterbi pass is the Hamming-nearest feasible schedule (checked against `otsl.uc.repair_min_updown`). Ablation:
rounding at 0.5 + repairs. An inference-time logit shift (decoding bias) is tuned on validation only.

**LP feedback.** The dispatch LP of the decoded schedule (`otsl.uc.UCModel`'s model, solved with highspy on a solver
object that is warm-started across one instance's steps; outages as in `otsl.ood.OODModel`) returns, as inputs of the
next step: per unit-hour the priced schedule, what decoding and repair changed relative to the rounding, output and
reserve relative to the limits, at-limit flags, the LMP at the unit's bus and the exact sensitivity d cost / d u from
the bound duals including the start-up and shut-down variables (the envelope-theorem gradient of
`UCModel.dispatch_gradient`, checked to match it exactly); per bus-hour LMP, shed and spill; per line-hour loading,
the flow-limit dual and a binding flag; per hour energy shed, spill, reserve shortfall, reserve price and capacity /
minimum-output margins; per instance the certified gaps of the current and best schedule to the LP relaxation, and
served / is-best flags. Prices and duals enter as signed logs. Ablation *no feedback*: every LP-derived signal is
zeroed; the priced schedule and the repair changes stay, so the recursion and inputs are otherwise identical.

**Training, stage 1: deep supervision with LP feedback of the model's own schedule.** 2,000 training instances (500 with
MILP labels, 1,500 with the polished labels of §7 Y3; no further MILP). Per batch, K_train = 3 outer steps on policy:
the network's step-k probabilities are decoded and priced by the LP, and the feedback feeds step k+1. Every step is
supervised (BCE to the stacked count targets), the optimizer steps after every outer step and (y, z) are detached in
between (TRM deep supervision). *Progressive training*: with probability 0.25 a batch starts from an arbitrary
schedule and its LP feedback instead of the model's own step 0 (the LP relaxation rounded at a random threshold, or the
label with ~3 units removed and ~3 added for random blocks), so the network learns to improve from any start.
A halting head (a probe on detached pooled answer states) is trained to predict whether the step's schedule will be
priced within 0.5 % of the best known cost (the label's LP cost). AdamW 1e-3, cosine schedule with warm-up, gradient
clipping, EMA of the weights (0.995) used for evaluation; batch 32, 10 epochs. Ablation *no deep supervision*: the loss
of the last step only, back-propagated through all three steps (no detach).

**Training, stage 2: cost-aware fine-tuning with the exact LP as critic.** REINFORCE on every outer step: the
trajectory follows the deterministic decoding (as at inference); at each step 4 Bernoulli samples of the step's
probabilities are decoded and priced; reward = log(c_best_before / c_sample), the step's improvement over the best
schedule so far, which is exactly what keep-best inference gains (clipped below at −5 %; step 0: −log(c / c_relax));
leave-one-out baseline; plus 0.5 × the supervised loss as an anchor; halting loss on the greedy schedule's cost.

**LP budget and parallelism.** LPs (decoding + repair + LP + feedback) run in a pool of CPU worker processes or in
process; with `inflight` > 1, several batches are kept in flight so the network trains on one while the workers price
another (the GPU configuration). On this container each training run is one single-threaded process on one core with
the LPs in process.

**Safety net.** If the best schedule after K steps sheds load or misses reserve, or its certified gap to the LP
relaxation exceeds θ, a reduced MILP (0.5 % gap) around it is solved: the decisions whose last probabilities are within
0.05 of 0 or 1 and agree with the schedule are fixed (min up/down row release as in the hybrid pipeline), and the
cheaper of the two schedules is kept. θ is chosen on validation; the fallback rate is reported.

## Setup

* **Compute.** Cores 2 and 3 of a 4-core container (cores 0–1 belong to the parallel arm A), one single-threaded
  process per core (torch 1 thread, HiGHS 1 thread), < 3.5 GB each. Training runs and evaluation shards are pinned with
  `taskset`.
* **Data.** uc12 (RTS-GMLC, 73 units, 12 h, DC network). Training: `train.npz` (500, MILP labels) + `uc12_m1x`
  extra_0–5 with polish_0–5 (1,500, polished labels). Validation: val + val_extra + val_extra2 (360), gap to the stored
  MILP's dual bound. Test once: test_fresh (120).
* **Selection (validation only).** Model variant, decoding bias, number of outer steps K and the safety-net threshold θ
  are chosen on the 360 validation instances; the m1x baseline's rounding thresholds for 1, 3 and 5 LPs are chosen there
  too (greedy forward selection on the mean gap). Pre-declared rule for K: the smallest K ≤ 16 whose validation mean
  gap is within 0.02 pp of the best; for 24 h (zero-shot) 2 × that K, fixed before seeing any 24-hour result.
* **Test protocol (paired).** Per instance, in one process on one core and back to back: the full MILP at 0.1 % (60 s
  limit; 300 s on 24 h, 600 s on the congested benchmark, as in their studies) and at 0.5 %; the LP relaxation (its
  time is counted in every method); LP rounding + block repair + min up/down repair, best of 5 LPs (no learning, the
  thresholds of `otsl.base`); the m1x 5-member temporal-GNN ensemble (§7 Y3) rounded at the validation-selected
  thresholds with the same repairs, 1 / 3 / 5 LPs, and decoded with Viterbi, 1 LP; rloop with K outer steps; the
  safety net. All dispatch LPs through the same `FastLP` code (one solver object per method and instance).
* **Metrics.** Served share (no shedding, no reserve shortfall), gap (C − DB) / C to the dual bound of the full MILP
  solved in the same process (C = the schedule's dispatch-LP cost, penalties included), mean / median / max, wall time
  including every overhead, speed-up T_full(0.1 %) / T_method (mean, median, geometric mean), paired instance
  bootstrap 95 % CIs (2,000 resamples).

## Results

(to be written)

## Ablations

(to be written)

## Extrapolation

(to be written)

## Verdict

(to be written)

## Caveats

(to be written)

## Running on PACE ICE

Everything that is cluster-specific is a marked placeholder in `scripts/pace/rloop_*.sbatch` (partition, QOS,
account, GPU type, module and environment names); check the PACE ICE documentation and fill them in, or pass them to
`sbatch` on the command line. Nothing about PACE ICE was verified from this container.

1. **Data.** The datasets are git-ignored and are regenerated on the cluster by the shared data job of the recursive
   study (see `scripts/pace/` for the job the other arm provides). rloop needs `data/generated/uc12` (train, val,
   val_extra, val_extra2, test_fresh), `data/generated/uc12_m1x` (extra_0–5 + polish_0–5) and, for the extrapolation
   tests, `data/generated/{uc24,hard,uc12_ood}`. More training data: polished-label chunks in the uc12_m1x format
   (`extra_<k>.npz` with the LP-relaxation fields, `polish_<k>.npz` with the label `y`) in any directory, passed with
   `--data_dir DIR` (repeatable) and `--n_train N`.
2. **Training** (`scripts/pace/rloop_train.sbatch`): one GPU for the network and the job's other CPU cores as LP
   workers. Stage 1 (deep supervision with LP feedback, preset `gpu`: hidden 160, 3 latent updates × 2 cycles per outer
   step, 2 message-passing layers, dilations 1–8, K_train = 6, batch 64, 60 epochs), then stage 2 (REINFORCE, 1,500
   updates, 8 samples, 6 steps), then validation. Knobs: `TAG`, `LP_WORKERS` (default: CPUs − 4), `INFLIGHT` (batches in
   flight, default 4), `N_TRAIN`, `EXTRA` (any `rec_rloop_train.py` flags, e.g. `--no_feedback`, `--data_dir ...`).
   The LP workers never touch the GPU; with `INFLIGHT` ≥ 2 the trainer runs the network on one batch while the workers
   decode and price another, so the GPU does not wait for the LPs as long as `LP_WORKERS` keeps up (rule of thumb:
   one worker solves ~30–40 12-hour LPs per second including decoding; one outer step of a 64-instance batch needs
   64 LPs).
3. **Evaluation** (`scripts/pace/rloop_eval.sbatch`, CPU only): an array of single-core tasks, each solving the full
   MILPs, baselines and the model back to back per instance (paired timing). `BENCH=uc12|uc24|hard|line_out`; model and
   K from `results/recursive/rloop_select.json` or `MODEL=… K_SEL=…`. Then `python3 scripts/rec_rloop_report.py`.
4. **Resume.** Training checkpoints every epoch (stage 1) and every 25 updates (stage 2); resubmitting the same job
   resumes. Evaluation skips finished instances.
