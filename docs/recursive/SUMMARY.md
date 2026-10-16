# Recursive reasoning for end-to-end unit commitment: CPU round, summary

*Two arms run in parallel on 4 CPU cores (no GPU), following [`PLAN.md`](PLAN.md). Details:
[`trm.md`](trm.md) (arm A, a TRM-style recursive model) and [`rloop.md`](rloop.md) (arm B, recursion with the exact
dispatch LP in the loop). uc12 (RTS-GMLC, 73 units, 12 h); selection on 360 validation instances; one pass over the 120
test_fresh instances; the full MILP and every pipeline are timed back to back on one core per instance. Gap is to the
dual bound of the full MILP at a 0.1 % gap. One seed per configuration.*

## Bottom line

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

## Test results (uc12 test_fresh, 120 instances)

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

## What the ablations say (validation, 360 instances)

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

## Zero-shot extrapolation (nothing re-tuned)

| benchmark | LP rounding | learned schedules (A, B) | B: guided search | B: + safety net |
|---|---|---|---|---|
| line outages (30) | 1.23 % | collapse, ~50 % served | 0.84 % | 0.47 % at 18× |
| congested lines × 0.6 (60) | 2.37 % | collapse (8–24 %), ~50 % served | 1.06 % | 0.51 % at 27× |
| 24 hours (40) | 1.77–1.78 % | A + LP roundings 1.59 %; B 3.18 % | 1.13 % | 0.67 % at 24× |

The networks' own schedules shed load on about half of the shifted instances; the LP-in-the-loop search stays robust.

## Why it did not work better, and what would test it properly

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

## PACE ICE plan (nothing submitted; cluster values are placeholders in `scripts/pace/`)

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
