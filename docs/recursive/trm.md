# Arm A: a tiny recursive model (TRM) for end-to-end unit commitment

*Branch `recursive-uc`, plan in [`PLAN.md`](PLAN.md). Code: [`otsl/trm.py`](../../otsl/trm.py) (features, model,
deep-supervision training, REINFORCE stage, decoders), `scripts/rec_trm_{train,val,select,eval,report,data}.py`,
`scripts/rec_trm_queue.sh`, tests [`tests/test_trm.py`](../../tests/test_trm.py), PACE ICE jobs `scripts/pace/trm_*`.
Results: [`results/recursive/trm_results.md`](../../results/recursive/trm_results.md) (+ `.json`), raw records
`results/recursive/trm_eval_test_fresh_s*.jsonl`, validation `results/recursive/trm_val_*.json`, training logs
`results/recursive/trm_train_*.json`. Data and weights (git-ignored): `data/generated/recursive/`.*

## Status

*Kept current for resuming after a container restart. Every job is resumable: rerun the same command (training resumes
from its checkpoint, a finished job is skipped because its result file exists).*

- Queues (each pins its jobs to one core, single-threaded): `bash scripts/rec_trm_queue.sh <core> <job> ...` with jobs
  `train:<tag>:<flags>`, `val:<tag>`, `cmd:<shell>`; logs in `data/generated/recursive/logs/` (`train_<tag>.log`,
  `q_*.out`, `queue_core*.out`). Checkpoints: `data/generated/recursive/models/<tag>.pt`.
- 02:00 container restart (all jobs killed; files intact). 02:04 queues relaunched; 02:30 reordered so that the test
  runs as soon as the final candidates are validated (`<scratchpad>/trm_q0g.sh`, `trm_q1g.sh`):
  - core 1: `deep` (resumed at step 1970) → `val:main` → `oneshot_rl` (REINFORCE on the one-shot control:
    `--stage rl --base oneshot --rl_steps 150 --rl_bs 16 --rl_samples 6 --rl_lr 1e-4 --rl_anchor 0.5`) → `val:oneshot_rl`
    → `val:deep` → (once `results/recursive/trm_val_main_act.json` exists) `python3 scripts/rec_trm_select.py --final
    main main_rl main_act --controls oneshot oneshot_rl deep` → `python3 scripts/rec_trm_halt.py --tag <final>` →
    select again → flag `data/generated/recursive/select.done` → test shard 1 (`python3 scripts/rec_trm_eval.py --split
    test_fresh --shard 1 --nshards 2`) → `oneshot_long`, `oneshot_nolp`, `oneshot_rank`, `oneshot_costw`,
    `oneshot_attn`, `n2T1` → their quick validations;
  - core 0: `main_act` (ACT, halt_tol 8) → `val:main_act` → (after the flag) test shard 0 → `nods`, `nolp`, `y0lp`
    → their quick validations.
  Resume after a restart: rerun the corresponding lines (finished trainings skip, trainings resume from checkpoints,
  the test evaluation skips finished instances, validations rerun).
- Done: training of `oneshot`, `main`, `main_rl`, `main_act`, `deep`, `oneshot_rl`; validation of `lpround`, `m1x`,
  `oneshot`, `oneshot_rl`, `main`, `main_rl`, `main_act`, `deep`; selection (`results/recursive/trm_select.json`: final
  `main`, headline pipeline `main_mix5r`); post-hoc halting head of `main` (`results/recursive/trm_halt_main.json`);
  zero-shot 24 h for `main`, `main_rl`, `oneshot` (`results/recursive/trm_stretch_uc24.json`).
- 04:18 test running: shard 0 on core 0, shard 1 on core 1 (`results/recursive/trm_eval_test_fresh_s{0,1}.jsonl`,
  logs `data/generated/recursive/logs/eval_s{0,1}.out`); afterwards the remaining ablations start automatically.
- After the queues: report (`python3 scripts/rec_trm_report.py`), congested stretch
  (`python3 scripts/rec_trm_stretch.py --bench hard --tags main main_rl oneshot`).

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

### Validation (360 instances; mean / median gap to the stored MILP objective, served share)

All models are tuned and compared on validation only; the full tables (BCE, wrong unit-hours, every pipeline, the
test-time compute curve) are in `results/recursive/trm_results.md`.

| model (parameters, training) | 1 LP | 1 LP + LP repair | 5 thresholds + repair | + 4 LP roundings + repair |
|---|---|---|---|---|
| LP rounding, 5 thresholds (no learning) | – | – | 2.08 / 0.93 % (97.5 %) | – |
| m1x temporal-GNN ensemble (5 × 180 k) | 9.48 % (86 %) | 3.97 / 1.10 % (95 %) | 3.48 / 0.61 % (95 %) | 1.96 / 0.87 % (97.2 %) |
| one-shot, same net as the TRM (135 k, 13 min) | 9.44 % (91 %) | 3.68 / 0.96 % (95 %) | 3.19 / 0.66 % (96 %) | 1.89 / 0.63 % (97.2 %) |
| deep non-recursive, 10 nets (1.3 M, 153 min) | 5.69 % (90 %) | 3.84 / 1.24 % (98 %) | 2.70 / 0.66 % (98 %) | 1.84 / 0.55 % (97.2 %) |
| **TRM `main`** (135 k, n 4, T 2, N_sup 8, 73 min) | 10.67 % (85 %) | 4.08 / 1.45 % (96 %) | 4.50 / 0.79 % (96 %) | 1.87 / 0.63 % (96.4 %) |
| TRM, ACT halting at ≤ 8 wrong (`main_act`) | 9.70 % (86 %) | 4.26 / 1.44 % (96 %) | 4.13 / 0.78 % (95 %) | 1.87 / 0.59 % (96.7 %) |
| TRM + REINFORCE (`main_rl`, +34 min) | 5.67 % (96 %) | 4.20 / 1.22 % (96 %) | 2.94 / 0.77 % (97 %) | 1.81 / 0.63 % (97.2 %) |
| one-shot + REINFORCE (`oneshot_rl`, +21 min) | 6.61 % (95 %) | 3.51 / 1.15 % (98 %) | 3.40 / 0.70 % (98 %) | 1.89 / 0.57 % (97.2 %) |

(1-LP columns: mean gap (served); the others mean / median (served). The TRM's "last 5 supervision steps screened +
repair": `main` 3.64 %, `main_rl` 3.99 %, `main_act` 4.08 %.)

* **No learned model's own schedule beats rounding the LP relaxation.** With one LP and the LP-guided repair the models
  are at 3.5–4.3 % mean (median 1.0–1.5 %) against 2.08 % (median 0.93 %) for LP rounding with 5 LPs; even the median is
  worse, and the models lose to LP rounding on about 76 % of the instances. The models make as many wrong decisions as the
  rounded relaxation (12.4 count-aware wrong unit-hours per instance for the TRM, 11.9 for the one-shot, 12.4 for the
  rounded LP), but their errors are cost-blind, whereas the relaxation's are economically mild.
* **The useful combination is model + LP rounding**: screening the model's schedule together with 4 LP roundings gives
  1.81–1.96 % (median 0.55–0.87 %) for every learned model, slightly better than LP rounding alone. The model's role is
  a diverse candidate; which model it is matters little (1.81–1.96 %).
* **The LP-guided repair is the largest single gain for every model** (TRM 10.7 → 4.1 %, one-shot 9.4 → 3.7 %): most
  failing schedules miss reserve because congestion or ramping forces renewable curtailment that the per-hour capacity
  check does not see.
* **Cost-aware fine-tuning** fixes the unrepaired pipeline (TRM 10.7 → 5.7 %, one-shot 9.4 → 6.6 %, served 96 % / 95 %),
  but once the LP repair is applied it changes little, except for the TRM's threshold screening (4.50 → 2.94 %).


## Ablations

**Which items of PLAN.md were tested, and how** (numbers below and in `results/recursive/trm_results.md`):

| PLAN item | tested here | how |
|---|---|---|
| recursion vs one-shot at equal parameters | yes | `oneshot`: the same 2-layer net applied once, same data and steps; `oneshot_long`: same wall-clock as `main` |
| recursion vs a deeper non-recursive net at matched compute | yes | `deep`: T(n + 1) = 10 unshared 2-layer nets applied once (one supervision step's compute, 1.24 M parameters) |
| deep supervision on / off | yes | `nods`: N_sup = 1 (one supervision step from the initial state), same steps |
| n and T | partly | `n2T1` (3 net calls per step) vs `main` (n = 4, T = 2: 10 calls); TRM's n = 6, T = 3 (21 calls) not run (2.2× the cost of a run) |
| test-time compute (1, 2, 4, 8, 16, 32 steps) | yes | every recursive model, validation and test |
| LP-relaxation features on / off | yes | `nolp` (TRM) and `oneshot_nolp` |
| group counts for identical units + DP assignment (risk 1) | yes | count targets + water-filling + Viterbi in every model; `oneshot_rank`: the earlier rank feature + canonical labels |
| feasible-by-construction decoding (risk 3) | yes | per-unit Viterbi DP vs threshold decoder, chosen per model on validation |
| graph-to-token transformer (tokenization section) | partly | factorized form (pools / convolution / bus graph with an electrical-distance kernel, PTDF encodings, no unit IDs) in `main`; axial attention with an electrical-distance bias over units and a relative hour bias in `oneshot_attn`; a full typed-token transformer (bus, line, hour tokens) was not built |
| cost-weighted deep supervision (risk 2) | yes | `oneshot_costw`: BCE weighted by the dispatch-LP sensitivity at the label (first-order cost of flipping each decision) |
| reward each step's improvement (risk 2) | yes | `main_rl`: REINFORCE at a random supervision step with the exact LP as critic |
| halting on "within 0.5 % of the best known cost" (risk 2) | yes | post-hoc halting head fitted on LP-priced steps of training instances (`scripts/rec_trm_halt.py`) |
| use the steps as candidates (risk 2) | yes | screening the last 5 steps' schedules |
| start from a good answer (risk 4) | partly | `y0lp`: y initialised from the LP-relaxation commitment; from the one-shot ensemble: not tested |
| safety net (risk 3) | yes | reduced-MILP fallback on the repaired 1-LP pipeline, test only |
| more data, recall / progressive loss, 24 h, congested benchmark | partly | input re-injected at every z update (recall) as in TRM; no progressive loss; zero-shot 24 h and congested runs (stretch) |
| ACT for data coverage (not in PLAN; found necessary) | yes | `main_act`: halting target ≤ 8 wrong unit-hours instead of ≤ 2 |

(results pending)


## Verdict

(pending)

## Caveats

(pending)

## Running on PACE ICE

The CPU runs here are a feasibility study; this section is the recipe for scaling on a GPU cluster. Cluster-specific
values in `scripts/pace/trm_*` (account, partitions, QOS, GPU type, module names, storage paths) are **placeholders
marked `PLACEHOLDER`**: check the PACE ICE documentation and fill them in before the first submission. Nothing in this
section was run on PACE.

**Environment** (`scripts/pace/trm_env.sh`, sourced by every job): load the cluster's Python module (placeholder), create a
virtual environment with `torch numpy scipy pandas highspy pytest` (the study used torch 2.14, scipy 1.17 with its
bundled HiGHS, highspy 1.12), `cd` to the clone of this repository (branch `recursive-uc`), single-threaded HiGHS and
BLAS. Submit from the repository root after `mkdir -p logs`; run `python3 -m pytest -q tests/test_trm.py` once (CPU,
~30 s).

**Data regeneration order** (the datasets are git-ignored and must be rebuilt from scratch; all CPU):

1. `sbatch scripts/pace/trm_gen_base.sbatch`: the uc12 splits with full MILPs: train 500 (seed 11), val 60 (12),
   val_extra 120 (31), val_extra2 180 (41), test_fresh 120 (23); ~980 MILPs ≈ 8 core-hours (1 node × 8 cores, ~1 h).
2. `sbatch scripts/pace/trm_teachers.sbatch`: the label teacher (3 MILP-label BCE GNNs, `scripts/uc_hybrid_train.py`),
   the m1x extra scenarios and 1,500 polished labels (`scripts/uc_m1x_data.py`, seed 111) and the 5-member m1x
   temporal-GNN ensemble (the baseline); ≈ 4–5 core-hours, 6 cores.
3. `sbatch --array=0-399%100 scripts/pace/trm_gen_data_array.sbatch` (`N_TOTAL=100000`): 100,000 more scenarios with
   teacher-polished labels, one chunk of 250 per task (`scripts/rec_trm_data.py`, seed 211, the m1x polishing setting);
   ≈ 5.5 s per instance → ≈ 150 core-hours, 3 GB per task, ~25 min per task.
4. Training on GPU (`scripts/pace/trm_train_gpu.sbatch`), then the CPU evaluation (`scripts/pace/trm_eval_cpu.sbatch`:
   validation passes and selection as one job, the paired test evaluation as an array of single-core shards), then
   `python3 scripts/rec_trm_report.py`.

**Commands.**

```bash
mkdir -p logs
sbatch scripts/pace/trm_gen_base.sbatch
sbatch scripts/pace/trm_teachers.sbatch                         # after step 1
N_TOTAL=100000 sbatch --array=0-399%100 scripts/pace/trm_gen_data_array.sbatch   # after step 2
TAG=gpu_small PRESET=gpu_small N_TRAIN=50000 sbatch scripts/pace/trm_train_gpu.sbatch
TAG=paper PRESET=paper N_TRAIN=100000 EXTRA="--batch 768 --micro_batch 128 --steps 20000" RL=1 \
    sbatch scripts/pace/trm_train_gpu.sbatch
TAGS="paper paper_rl" CONTROLS="gpu_small" sbatch scripts/pace/trm_eval_cpu.sbatch          # validation + selection
sbatch --array=0-7 --export=ALL,STAGE=test scripts/pace/trm_eval_cpu.sbatch                  # paired test, 8 shards
python3 scripts/rec_trm_report.py
```

Every knob of `otsl.trm.TRMConfig` is a flag of `scripts/rec_trm_train.py` (`--hidden --layers --n --T --n_sup --batch
--micro_batch --ema --lr --weight_decay --warmup --beta1 --beta2 --unit_mix --time_mix --heads ...`); `--device cuda`
selects the GPU. Presets: `cpu` (this study), `gpu_small` (hidden 128, attention mixers, batch 256), `paper` (TRM's
scale: hidden 512, 2 layers, n = 6, T = 3, N_sup = 16, batch 768, EMA 0.999, AdamW β = (0.9, 0.95), lr 1e-4, weight
decay 1.0, 2,000 warm-up steps, attention mixers). Jobs resume from their checkpoint when resubmitted with the same tag.

**Resource estimates** (analytical: per token and layer, attention mixers and SwiGLU cost ≈ 19 D² + 400 D
multiply-adds; one net call = 2 layers over 876 tokens; a supervision step = T(n + 1) calls forward and n + 1 with
backward; activation memory measured on CPU, ≈ 0.5 GB per training sample at hidden 512 and bf16):

| job | config | compute | memory | estimate |
|---|---|---|---|---|
| data, step 1 | 980 full MILPs | CPU | 2 GB / process | ≈ 8 core-h |
| data, step 2 | teachers, 1,500 polished labels, m1x ensemble | CPU | 3 GB / process | ≈ 5 core-h |
| data, step 3 | 50,000 / 100,000 polished labels | CPU array | 3 GB / task | ≈ 75 / 150 core-h |
| train | `gpu_small` (hidden 128, n 6, T 3, N_sup 16, batch 256), 30k steps | 1 GPU | ≈ 35 GB (micro-batch 128 on a 40 GB card) | ≈ 2–4 GPU-h (A100 / H100) |
| train | same blocks, one-shot control (`--mode oneshot`), equal epochs | 1 GPU | < 10 GB | < 1 GPU-h |
| train | `paper` (hidden 512, batch 768), 20k steps | 1 GPU, `--micro_batch 96–128` | ≈ 60–75 GB per micro-batch | ≈ 7–9 GPU-h (H100) / 22–27 (A100) |
| fine-tune | REINFORCE stage, 400 steps × 32 instances × 8 samples | 1 GPU + 6–8 CPU cores (LPs) | as training | ≈ 1 GPU-h + 6–8 core-h |
| evaluate | validation (360) + paired test (120), 6 models | CPU (1 core per shard); TRM inference `--trm_device cuda` for the paper scale | 4–6 GB | ≈ 10–15 core-h |

At the paper scale the CPU forward pass is ≈ 15 s per supervision step for one instance (measured), slower than the
full MILP, so an end-to-end speed claim at that scale needs GPU inference (`scripts/rec_trm_eval.py --trm_device cuda`;
LPs and MILPs stay on the CPU core).

**Which configurations are worth running at scale** (from the CPU study; see the Verdict): see the "PACE plan" at the end
of the Verdict.
