# Arm A: a tiny recursive model (TRM) for end-to-end unit commitment

*Branch `recursive-uc`, plan in [`PLAN.md`](PLAN.md). Code: [`otsl/trm.py`](../../otsl/trm.py) (features, model,
deep-supervision training, REINFORCE stage, decoders), `scripts/rec_trm_{train,val,select,eval,report,data}.py`,
`scripts/rec_trm_queue.sh`, tests [`tests/test_trm.py`](../../tests/test_trm.py), PACE ICE jobs `scripts/pace/trm_*`.
Results: [`results/recursive/trm_results.md`](../../results/recursive/trm_results.md) (+ `.json`), raw records
`results/recursive/trm_eval_test_fresh_s*.jsonl`, validation `results/recursive/trm_val_*.json`, training logs
`results/recursive/trm_train_*.json`. Data and weights (git-ignored): `data/generated/recursive/`.*

## Status

*Kept current for resuming after a container restart. Every job is resumable: rerun the same command.*

- 23:49 main run started on core 0 (`bash scripts/rec_trm_queue.sh 0 "train:main:--preset cpu"`), one-shot controls on
  core 1. Logs: `data/generated/recursive/logs/` (`train_<tag>.log`, queue outputs `queue_core*.out`).
- 00:03 one-shot control done (13 min); validation pass of it running.

## Summary

(pending)

## Method

**Task.** End-to-end 12-hour UC on RTS-GMLC (73 thermal units, 73 buses, 120 lines, DC network, min up/down, ramping,
start-ups, 3 % spinning reserve): predict all 876 on/off decisions, decode a feasible schedule, solve one dispatch LP
(or screen ≤ 5). No MILP in the pipeline, except the optional safety net below.

**Tokens and mixing.** One token per (unit, hour) with the unit's static data (capacities, cost segments, start-up
cost, min up/down, ramp, type), its initial status, the size of its identical-copy subgroup, the hour's totals (load,
renewables, net load, reserve, net load + reserve), a bounded position encoding (first hour, e^−t/4, e^−(T−1−t)/4),
its bus's load, renewables and PTDF encoding (8 principal components of the bus's PTDF column), and optionally the LP
relaxation (u_rel, price at the bus, largest incident line loading). Bus inputs (load, renewables, unit capacity, PTDF
encoding, price, loading) enter through the graph mixer. One tiny network of 2 layers; each layer has four residual
sub-blocks with a post RMS norm (as TRM): (1) **across units within an hour**: mean and capacity-weighted pools and a
merit-order prefix pool (the capacity-weighted state of the cheaper units), or attention with an electrical-distance
bias (option `unit_mix=attn`); (2) **across hours for a unit**: a depthwise convolution (kernel 5), or attention with a
relative hour-difference bias (`time_mix=attn`); (3) **over the network through the buses**: unit → bus averaging plus
the bus inputs, two steps of the normalised bus adjacency and one step of an electrical-distance kernel
(exp(−R/τ) of the effective reactance distance from the DC susceptance matrix), bus → unit; (4) a SwiGLU MLP. No
unit-ID embeddings: a unit is its parameters. This is the factorized (axial) CPU form of the graph-to-token design of
PLAN.md; bus states are recomputed in every layer rather than carried as tokens, and lines enter as bus encodings
(PTDF, loading) rather than as tokens.

**Identical units: counts, not copies.** 56 of the 73 units are in 24 groups of exact copies at one bus. With the
symmetry-rank feature removed, every mixer is equivariant over copies, so copies with the same initial status get the
same probability; permuting copies permutes the output (tested), i.e. augmentation by permuting identical units is
built in. The target of a unit is its subgroup's mean label per hour (the on-count divided by the size), which BCE
fits by the count. The LP-relaxation input is symmetrised the same way (the relaxation splits copies arbitrarily).
Decoding turns the expected count of a subgroup into per-copy scores by water-filling (copy j gets
clip(Σp − j, 0, 1)), so the assignment is deterministic. The earlier recipe (rank feature + canonical labels) is an
ablation (`sym=rank`).

**Recursion (TRM).** Answer embedding y and latent z (fixed truncated-normal initial vectors); z ← net(x + y + z)
n times, then y ← net(y + z) (the bus inputs enter only the z updates); T − 1 such cycles without gradient and one
with; a linear head reads one logit per token from y, a halting head reads the mean of y. **Deep supervision**: each
training instance stays in the batch for up to N_sup steps, (y, z) carried detached, with BCE on every step's answer
plus 0.5 × BCE of the halting head on "the rounded answer has ≤ 2 wrong unit-hours" (counts for copies); an instance
leaves the batch when the halting head fires (after a random minimum number of steps with probability 0.1) or at
N_sup, and is replaced by the next one (TRM's single-pass ACT; batch slots start staggered so that a batch mixes
supervision steps). AdamW (β = 0.9, 0.95), linear warm-up, cosine decay, gradient clipping 1, EMA of the weights 0.999
(evaluation uses the EMA weights), bf16 autocast.

**Controls with the same blocks.** *One-shot*: the same 2-layer network applied once to x + y_init + z_init (equal
parameters), trained with plain BCE. *Deep*: T(n + 1) separate 2-layer networks applied once (the compute of one
supervision step without weight sharing; ~10× the parameters).

**Second stage: cost-aware fine-tuning.** REINFORCE with the exact dispatch LP as critic on the final answer, as in
`train_uc_reinforce`: k − 1 supervision steps without gradient, step k with; 6 Bernoulli schedules per instance, each
decoded as at inference (Viterbi + block adequacy repair) and priced by the LP; reward −log(cost / LP-relaxation cost);
leave-one-out baseline; plus 0.5 × BCE to the targets as an anchor (REINFORCE alone over-commits, RESEARCH.md §4 V4).
With `random_step`, k is uniform in 1..N_sup, so every step learns to improve the schedule (under the leave-one-out
baseline, rewarding step k's cost and rewarding its improvement over step k − 1 are the same).

**Decoding (feasible by construction for unit constraints).** Probabilities after k steps → water-filling over copies →
per-unit **Viterbi** DP over the on/off state machine with min up/down and the initial status, maximising
Σ_t u_t (logit q_t − logit θ) (θ = decision threshold) → block (min up/down-aware) adequacy repair → min up/down
repair (a no-op after Viterbi) → dispatch LP. The plain threshold decoder (p > θ, then the same repairs) is the
alternative; decoder and θ are chosen on validation per model.

**LP-guided repair (added after diagnosing the first validation runs).** Most failing schedules miss reserve although
committed capacity exceeds net load + reserve: congestion or ramping forces the LP to curtail renewables, which the
per-hour capacity check counts as available. If the dispatch LP sheds load or misses reserve, the failing hours' reserve
requirement is raised by the curtailed renewables plus 1.1 × the deficit, the block adequacy repair is rerun and the
new schedule priced (≤ 2 rounds, kept only if cheaper). It is applied identically to every end-to-end method
("+ LP repair" rows), with that pipeline's decoder and threshold selected separately on validation.

**Screening (≤ 5 LPs).** (a) The schedules of the last 5 supervision steps; (b) 5 thresholds of the selected decoder,
chosen by greedy forward selection on validation; the cheapest by the LP is kept (then repaired if it fails).

**Safety net.** On top of the repaired 1-LP pipeline: if it still sheds or misses reserve, or its bound gap
(C − z_LP) / C against the LP relaxation (§8 Z1) is ≥ g* (the smallest grid value firing on ≤ 20 % of the validation
instances), solve a reduced MILP (0.5 % gap, 60 s) with the model's schedule fixed on its 90 % most confident
decisions (plus the adequacy guard and the min up/down row release) and keep the cheaper schedule. The fallback rate
is reported.

## Setup

* **Data.** Training: the m1x set, 500 instances of `train.npz` (full-MILP labels) + 1,500 teacher-polished labels
  (`data/generated/uc12_m1x/polish_0..5`, one reduced MILP around a teacher's confident decisions, no full MILP), the
  same data as the m1x temporal-GNN ensemble. Validation: val + val_extra + val_extra2 (360), against the stored MILP
  objective C* (60 s, 0.1 %); the 60 of `val.npz` also give the monitoring loss during training. Test: test_fresh
  (120), read once.
* **CPU preset** (`otsl.trm.PRESETS["cpu"]`): hidden 64, 2 layers, n = 4, T = 2, N_sup = 8, batch 16, 3,000 optimiser
  steps, lr 1e-3 (warm-up 200, cosine to 10 %), weight decay 0.1, EMA 0.999, bf16 autocast; 135 k parameters.
  A step costs ~1.5 s on one core (~75 min per run); n = 6, T = 3 (TRM's setting) costs ~2.2× per step. Inference
  on one instance: ~56 ms per supervision step.
* **Selection on validation only** (`scripts/rec_trm_val.py`, `scripts/rec_trm_select.py`): per model the decoder and
  threshold of each pipeline, the 5 screening thresholds, the safety-net threshold g*, and the final TRM among the
  candidates by the validation mean gap of the 1-LP + repair pipeline.
* **Test protocol** (`scripts/rec_trm_eval.py`): per instance, in one process pinned to one core (cores 0 and 1, one
  shard each; HiGHS and torch single-threaded): full MILP at 0.1 % (dual bound DB and speed-up denominator) and at
  0.5 % (highspy, 60 s), the LP relaxation (re-solved and timed), then every pipeline. Method time includes every
  overhead (LP relaxation where used, features, network passes on one instance, decoding and repairs, LPs).
* **Metrics.** Gap (C − DB) / C (the LP prices shedding at $10,000/MWh and reserve shortfall at $1,000/MWh, so mean
  gaps include those penalties), served share (no shedding, no reserve shortfall), speed-up T_MILP / T_method per
  instance (mean, median, geometric mean); paired instance-bootstrap 95 % CIs (2,000 resamples) of Δ gap and Δ log
  speed-up.

## Results

(pending)

## Ablations

(pending)

## Verdict

(pending)

## Caveats

(pending)

## Running on PACE ICE

(pending)
