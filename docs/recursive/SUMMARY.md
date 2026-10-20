# Recursive reasoning for end-to-end unit commitment: summary

*First round: two arms run in parallel on 4 CPU cores (no GPU), following [`PLAN.md`](PLAN.md). Details:
[`trm.md`](trm.md) (arm A, a TRM-style recursive model) and [`rloop.md`](rloop.md) (arm B, recursion with the exact
dispatch LP in the loop). uc12 (RTS-GMLC, 73 units, 12 h); selection on 360 validation instances; one pass over the 120
test_fresh instances; the full MILP and every pipeline are timed back to back on one core per instance. Gap is to the
dual bound of the full MILP at a 0.1 % gap. One seed per configuration.*

## GPU round (PACE ICE, 50,000 labels): validation and test results

The decisive runs of the plan below were made on PACE ICE (H200 GPUs; 50,000 polished labels instead of 2,000);
details and every table in [`gpu_round.md`](gpu_round.md). Selection on the 360 validation instances, then one
test pass on the 120 test_fresh instances; all comparisons paired per instance:

* **Arm A fails its go/no-go.** On validation, at their final checkpoints a same-size one-shot network (−0.24 pp [−0.39, −0.12]) and a
  deeper unshared network (−0.51 pp [−0.75, −0.29]) are both significantly *better* than the recursive model
  (1.731 % mean gap). Their validation loss rises late in training, but their schedules keep improving, so
  best-validation-loss checkpoints understate them. Training the recursive model twice as long did not help.
* **Arm B's LP feedback buys feasibility, not cost.** On validation, with feedback the loop serves 97.5 % of instances at 16 steps
  against 92.2 % without, and the mean gap is 0.62 pp [0.23, 1.08] better; on the instances both serve, feedback is
  not better (−0.08 pp [−0.22, +0.05]). The feedback model's halting head stops earlier (5.0 steps against 6.5) and
  lands better (1.63 % against 2.26 %). Training it took about 23 GPU-hours of jobs, against 17.5 for the control.
* **Neither arm displaces the MILP pipeline** of `docs/RESEARCH.md` §7–§8 (0.30 % at ~10×); the best end-to-end
  validation numbers are 1.0–1.2 % mean gap. Every learned model now beats learning-free LP rounding (2.11 %).
* **The test pass confirms all three** (gap to the tightest proven bound, paired timing on one core). Arm A: the
  recursive model loses to the final one-shot network on both of the gate's pipelines (−0.36 pp [−0.65, −0.08] with
  1 LP + repair, −0.26 pp [−0.40, −0.12] with 5-threshold screening). Its 16 CPU steps take 8 s per instance, slower
  than the full MILP at 0.5 % on a typical instance. Arm B: feedback +0.55 pp [+0.01, +1.28] overall (99.2 % served
  against 94.2 %), −0.05 pp [−0.22, +0.12] on the instances both serve; with the safety net it serves every instance
  at 0.342 %, 0.08 pp worse than the 0.5 % MILP and 1.35× faster. Best arm A pipeline: the deep control with a safety
  net, 0.753 % at 20× (mean). One seed per model; 120 instances.

* **After the round, an InfiLoop-style gate on arm B failed its pilot** ([`infiloop_pilot.md`](infiloop_pilot.md)):
  starting each step from a learned, LP-verdict-scored average of earlier states instead of the newest one made the
  CPU-scale model worse on the mean (−0.91 pp [−1.79, −0.16], fewer instances served) though slightly cheaper where both
  serve (+0.15 pp [+0.03, +0.29]) — within the spread of two runs of the plain model, so no gain but no proven loss.
  A seed-matched comparison of the plain model, this gate and the paper's own placement is running at CPU and GPU scale.

## CPU round (4 cores, 2,000 labels)

### Bottom line

* **Neither arm meets the success criterion** of [`PLAN.md`](PLAN.md) (≥ 95 % served at ≤ 0.5 % mean gap in well under
  a second), and no end-to-end pipeline beats the MILP-based pipeline of `docs/RESEARCH.md` §7–§8 (0.30 % at ~10×).
* **Latent recursion alone (arm A) does not help**: a TRM does not beat a one-shot network of the same size, and its
  own schedule is worse than learning-free LP rounding. Deep supervision makes extra steps safe but not useful; quality
  is flat after about 4 steps.
* **Recursion with exact LP feedback (arm B) does help**: with feedback, more steps keep improving the schedule
  (3.5 % after 1 step, 1.8 % after 16, 1.6 % after 32), and feedback beats the same recursion without it by 0.8–1.3 pp.
  But the network's whole-schedule output is only level with LP rounding; the best arm-B pipeline is a local search in
  which the LP prices every move and the network only ranks the moves, and most of its gain comes from the search.
* **The LP relaxation carries most of the answer**: without LP-relaxation inputs every model degrades to 6–8 %.

### Test results (uc12 test_fresh, 120 instances)

Both arms re-ran the shared baselines on their own cores; their numbers differ slightly because arm A added an
LP-guided adequacy repair to every pipeline and arm B used the existing repairs.

| pipeline | served | mean gap [95 % CI] | speed-up vs full MILP (mean) |
|---|---|---|---|
| full MILP at 0.5 % | 96.7 % | 0.23 % [0.18, 0.29] | 3.9× |
| *MILP pipeline of §7–§8 (ensemble fixings, LP guard, 0.5 % gap; quoted)* | *97 %* | *0.30 %* | *~10×* |
| LP rounding + repair, 5 LPs (no learning) | 98.3–99.2 % | 1.71–1.75 % | 43–62× |
| one-shot temporal-GNN ensemble + repairs, 5 LPs | 95.0–99.2 % | 1.45–2.00 % | 34–55× |
| **A**: TRM schedule + 4 LP roundings + repair | 99.2 % | 1.48 % [1.10, 1.98] | 19× |
| A: one-shot (same net) + REINFORCE + reduced-MILP safety net | 100 % | 1.17 % [0.98, 1.37] | 48× |
| **B**: whole-schedule recursion with LP feedback, 16 steps | 95.8 % | 1.77 % [1.33, 2.30] | 33× |
| B: local search from LP rounding, no learning, 16 LPs | 99.2 % | 1.29 % [1.03, 1.57] | 28× |
| **B**: same search, moves ranked by the recursive network | 99.2 % | **1.14 % [0.93, 1.39]** | 21× |
| B: … + reduced-MILP safety net (25 % fall back) | 100 % | 0.64 % [0.55, 0.74] | 13× |
| B: … with a stricter safety net (52 % fall back) | 99.2 % | 0.46 % [0.38, 0.56] | 8.5× |

* Arm A against its controls: TRM + LP roundings vs LP rounding −0.23 pp [−0.37, −0.12] but 2.3× slower; vs the
  one-shot network in the same mixed pipeline −0.12 pp [−0.24, −0.02] but 2× slower; vs a deeper non-recursive network
  at matched compute +0.03 pp [−0.06, +0.14]. The TRM's own schedule (1 LP + repair) is +0.84 pp [+0.30, +1.49]
  worse than LP rounding.
* Arm B: the guided search beats LP rounding by −0.61 pp [−0.95, −0.33] and the learning-free search by −0.15 pp
  [−0.28, −0.03]; it is level with the same search ranked by the one-shot ensemble (−0.02 pp [−0.06, +0.02]).

### What the ablations say (validation, 360 instances)

| question | answer |
|---|---|
| recursion vs one-shot at equal size (A) | no gain: one-shot BCE 0.041, TRM 0.046; same mixed-pipeline gap |
| more steps at test time (A) | 4 steps help a little, then flat to 32; no overthinking, no further gain |
| deep supervision (A, B) | without it A collapses at 32 steps; in B no significant effect at 16 steps |
| exact LP feedback in the loop (B) | **yes**: −0.78 pp [−1.54, −0.07] after supervised training, −1.26 pp [−2.05, −0.62] after REINFORCE |
| REINFORCE on LP-priced improvement (A, B) | large gain without repairs (A: 10.7 → 5.7 %), −1.38 pp for B's recursion |
| group counts + Viterbi decoding, cost weighting, LP-initialised answer, axial attention with electrical-distance bias (A, B) | no gain over simpler variants |
| halting heads (A, B) | not useful: about 30 % of steps qualify as good at every step |
| LP-relaxation inputs (A) | essential: without them TRM 7.96 %, one-shot 7.24 % |

### Zero-shot extrapolation (nothing re-tuned)

| benchmark | LP rounding | learned schedules (A, B) | B: guided search | B: + safety net |
|---|---|---|---|---|
| line outages (30) | 1.23 % | collapse, ~50 % served | 0.84 % | 0.47 % at 18× |
| congested lines × 0.6 (60) | 2.37 % | collapse (8–24 %), ~50 % served | 1.06 % | 0.51 % at 27× |
| 24 hours (40) | 1.77–1.78 % | A + LP roundings 1.59 %; B 3.18 % | 1.13 % | 0.67 % at 24× |

The networks' own schedules shed load on about half of the shifted instances; the LP-in-the-loop search stays robust.

### Why it did not work better, and what would test it properly

* **Too few epochs for the TRM**: deep supervision keeps each instance for 8 steps, so in the CPU budget the TRM saw
  3 epochs against the one-shot's 24.
* **Small models**: well under 1M parameters, against TRM's 5–7M at batch 768.
* **The relaxation already holds the answer**: models and rounded LP both make ~12 wrong unit-hours per instance, but
  the models' errors are cost-blind, while the LP feedback in arm B is not.
* **Small data**: 2,000 polished labels; 50–100k are cheap to generate on a cluster.

These are exactly what GPU time can change. The decisive experiments:

1. **Arm A go/no-go**: TRM vs one-shot vs deep non-recursive at equal epochs on 50k polished labels, with and without
   LP features. Go only if the TRM beats the one-shot by more than ~0.2 pp.
2. **Arm B**: a larger recursive network with LP feedback, trained with REINFORCE on 50k instances, against the same
   without feedback. The feedback effect grew with REINFORCE, so this is the run that could change the
   whole-schedule verdict.
3. **Cheap, either way**: the one-shot + REINFORCE + safety-net pipeline on 50k labels.

## PACE ICE plan (stages 1–2 run; stage 3 not run because arm A's gate failed; see [`scripts/pace/README.md`](../../scripts/pace/README.md))

| stage | job | resources (estimates) |
|---|---|---|
| 1 | rebuild base splits and teachers (`trm_gen_base`, `trm_teachers`) | ≈ 13 CPU core-h |
| 1 | 50k polished labels (`trm_gen_data_array`, 200 tasks × 250) | ≈ 75 CPU core-h, 3 GB per task (100k: ≈ 150) |
| 1 | extrapolation test sets (`rloop_data_extrap`) | a few CPU core-h |
| 2 | arm A go/no-go (`trm_train_gpu`, preset `gpu_small`): TRM, one-shot, deep, ± LP features | TRM ≈ 3–4 GPU-h, each control < 1 GPU-h; ≈ 35 GB GPU memory (40 GB+ card) |
| 2 | one-shot + REINFORCE + safety net on 50k | < 1 GPU-h + ≈ 2 CPU core-h |
| 2 | arm B with and without feedback (`rloop_train`, preset `gpu`, 50k, REINFORCE) | ≈ 25 + 6 GPU-h, ≈ 500 + 120 CPU core-h of LP workers (16–24 CPU cores beside the GPU), ≈ 8 GB GPU, 24–32 GB host |
| 2 | evaluation (`trm_eval_cpu`, `rloop_eval`), paired on dedicated cores | ≈ 10–15 CPU core-h per arm |
| 3 | only if arm A is a go: paper-scale TRM (hidden 512, batch 768 via micro-batches, 20k steps) + REINFORCE | ≈ 7–9 H100-h, 60–75 GB per micro-batch |

Stages 1–2 total roughly **40–45 GPU-hours and 750–850 CPU core-hours**. Any 40 GB+ GPU is enough for stage 2; stage 3
wants an H100 (80 GB) or H200.

**ICE limits (PACE notice, September 2026)**: per user, 32 GPU-hours of running capacity (GPUs × remaining walltime
of running jobs) and 50 queued + running GPU jobs, shared with the user's other projects. `scripts/pace/pgu_guard.py`
reads all of the user's jobs before each submission and keeps this project within a configurable share (default 16
GPU-hours, 4 GPU jobs, 16 CPU tasks, ≤ 8 h per job), so stage 2 runs as at most two 1-GPU jobs at a time, the long
arm-B run as resumable 7-hour segments, and the data array as 16 tasks (8 at a time).
