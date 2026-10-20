# Arm B with a loop-native attention residual (InfiLoop): pilot

*After the GPU round. The idea comes from Li et al., "Scaling to Tens of Thousands of Test-Time Iterations with
Loop-Native Attention Residuals" (arXiv 2610.11570, October 2026): replace TRM's carry-last rule — each step continues
from the newest state — with a running weighted average of earlier states, scored by content and aged by a learned
decay, computed by an exact streaming update. On Sudoku-Extreme it lifts a 7M TRM-style model from 74.7 % to 97.9 %
and stops accuracy from falling as loops are added.*

## Why arm B

On the GPU-round validation runs the arm B feedback model wanders away from its best schedule: at 24 steps its latest
schedule is 1.565 % from the bound against 1.037 % for the best one seen so far, and 27 % of (step, instance) pairs
are worse than an earlier step. The no-feedback control shows the paper's failure outright: its latest schedule
degrades past its 6 training steps (3.14 % at 4 steps, 5.57 % at 24; served 88.6 % → 83.9 %). Arm A does not degrade
on average (flat from 8 to 32 steps), so the connection was tried on arm B, where the network also sees the exact LP
verdict of every step — the "is this state good?" signal the paper's scorer has to learn on its own.

## What was built

`RLoopNet(loopres=True)` in [`otsl/rloop.py`](../../otsl/rloop.py), flag `--loopres` of
`scripts/rec_rloop_train.py`. Each outer step starts from h_k = Σ a_{k,i} c_i over the final states c_i of earlier steps
(y and z, separate parameters), a_{k,i} ∝ exp(s_i − λ(k − i)), by the paper's exact streaming recurrence. A state is
written once, at the start of the next step, when its schedule's LP verdict is known; its score is q · RMSNorm(c) per
token plus w · φ, φ the per-instance verdict (certified gap of the schedule and of the best so far, served, is-best),
clipped to [−8, 8]; the decay is learned from β₀ = 0.55. The step's own answer, decoding, pricing and keep-the-best
output are unchanged. With β → 0 the model reproduces the carry-last model step by step (tested); the gate's gradient
passes the gradient-free first cycle as an identity (the 1-step gradient of TRM / HRM).

## Pilot (CPU scale, decided before the run)

The CPU-round arm B model (preset `cpu`, 2,000 labels, seed 0) trained twice on PACE, identical except `--loopres`:
supervision, REINFORCE, then 24 steps on the 360 validation instances (`scripts/pace/pgu_rloop_pilot.sbatch`). Rule,
fixed in advance: continue to the GPU run only if the gated model's best-of-24 mean gap beats the plain model's with a
paired 95 % CI excluding zero.

| | gated | plain | difference (gated better > 0) |
|---|---|---|---|
| best of 24 steps, mean gap | 4.495 % | 3.588 % | **−0.907 pp [−1.786, −0.162]** |
| served (best schedule) | 88.6 % | 90.3 % | |
| median gap | 0.945 % | 1.031 % | |
| on the 314 instances both serve | 1.544 % | 1.694 % | +0.150 pp [+0.032, +0.286] |
| latest schedule at step 24 | 6.059 % | 4.484 % | |
| steps worse than the best so far | 13.9 % | 12.9 % | |

**How much of this is noise.** Two runs of the *plain* model differ by about as much. The CPU round's arm B model
(same preset and seed, another machine, an earlier code version) against this pilot's plain run, at 16 steps:
0.48 pp worse [−0.27, +1.28], and significantly cheaper where both serve (−0.16 pp [−0.30, −0.03]). Against that run
the gated model is 0.38 pp worse [−0.31, +1.17], not significant. The bootstrap intervals above resample instances
only, not training runs, and the seed then fixed only data order and sampling — the initial weights were unseeded
in every arm B run so far (fixed since). So the pilot shows no gain, but it does not show that the gate makes the
model worse.

**The pilot fails the rule.** On this single pair the gated model is worse on the mean, through feasibility (it serves
fewer instances), while its schedules are slightly cheaper where both models serve — the mirror image of what the LP
feedback itself does. It does not reduce the drift it was meant to fix: its latest schedule is further from its best
one, not closer. The halting head also stops later (17.8 steps against 14.1).

**The GPU run was stopped** after 7 of 60 supervised epochs (1.5 GPU-hours; job 6408754, checkpoint kept, resumable).
On the 120-instance training monitor it was ahead of the original GPU-round model at 5 of those 7 epochs (best-of-16
gap 2.06 % against 2.69 % at epoch 6). That curve is unpaired, from one seed, and the original's moved by ±0.3 pp
between epochs, so it does not overturn the paired pilot; it is recorded because it points the other way.

Caveats: one seed per model at CPU scale (1 recursion cycle per step, 3 training steps), where the drift being
targeted is smaller (13 % of steps) than at GPU scale (27 %); only the LP-verdict variant was tried, not the paper's
state-only scores (`--loopres_no_lp`) or its within-step axes. Records: `results/recursive/rloop_val_pilot_*_rl.npz`,
`rloop_pair_pilot_lr_rl_vs_pilot_base_rl.json`, `rloop_train_pilot_*.json`, `rloop_train_w3_rloop_lr.json`.

## Seed-matched comparison at both scales

The question that matters is whether either version beats the existing models, so both are now run with seeds:
the plain model, this LP-step gate, and the paper's own placement (`loopres_mode="paper"`: an inner axis over each
cycle's latent updates and an outer axis over every answer update, state-only scores), seeds 1–3, with seeded initial
weights so that equal seeds give equal starting weights and data order. CPU scale: `scripts/pace/pgu_rloop_screen.sbatch`.
GPU scale (the GPU-round configuration, 50,000 labels): `scripts/pace/plans/loopres_gpu.json`, run through
`scripts/pace/pgu_queue.py` within the project's GPU share. Rule, fixed before the runs (`scripts/rec_loopres_compare.py`):
a variant beats the plain model only if its seed-averaged per-instance advantage on validation has a 95 % CI above zero
and it is better in every seed; only then does it go to the test set, against the plain seeds, the GPU-round model and
arm A's best pipelines.

### CPU scale: neither version beats the plain model

Best of 24 steps on the 360 validation instances, mean gap to the bound (served at the best schedule in brackets);
`python3 scripts/rec_loopres_compare.py --prefix cs`, record `results/recursive/loopres_compare_cs.json`.

| seed | plain | LP-step gate | paper's placement |
|---|---|---|---|
| 1 | 3.757 % (88.9 %) | 4.314 % (87.2 %) | 2.598 % (93.1 %) |
| 2 | 3.056 % (91.7 %) | 3.770 % (89.7 %) | 4.282 % (88.3 %) |
| 3 | 2.872 % (91.7 %) | 4.057 % (87.5 %) | 3.444 % (88.9 %) |
| **against plain, seed-averaged** (better > 0) | | **−0.819 pp [−1.282, −0.410]**, better in 0/3 seeds | **−0.213 pp [−0.693, +0.276]**, better in 1/3 |
| on the instances both serve | | −0.051 pp [−0.118, +0.008] | +0.131 pp [+0.060, +0.204] |
| steps worse than the best so far | 16.7 / 22.0 / 35.7 % | 14.6 / 15.0 / 23.2 % | 31.8 / 29.3 / 19.4 % |

* **The LP-step gate is worse**, in every seed and outside the interval: it serves fewer instances (87–90 % against
  89–92 %) and is no cheaper where both serve. It does reduce the drift it was built for — fewer steps fall behind
  the best one in every seed — but its best schedule is worse, so less drift buys nothing.
* **The paper's placement is a wash on the mean** and the seeds disagree (+1.16, −1.23, −0.57 pp), all through
  feasibility (served +4.2, −3.4, −2.8 points). Where both models serve, it is cheaper in all three seeds
  (+0.08, +0.19, +0.09 pp) — the one consistent signal of the study, and a small one. It does not reduce drift.
* The plain model's own seeds span 2.87–3.76 %: the run-to-run spread is as large as any effect here.

**At this scale the gates hardly learned.** After training, the decay is β = 0.55–0.57 in every gated model (from 0.55),
the score vectors |q| = 0.15–0.31 and the LP weights |w| = 0.06–0.12 (from 0): scores near zero, so each step starts from a near-fixed
running average of earlier states that gives the newest state weight ≈ 0.45. The CPU screen therefore tests that
structure more than a learned gate. At GPU scale the gate does learn: the seed-1 LP-step model after 9 of 60 epochs
has β ≈ 0.71 / 0.74, |q| ≈ 2.1 / 2.4, |w| ≈ 0.9 / 1.0 (the stopped `w3_rloop_lr` after 7 epochs: β ≈ 0.64 / 0.62), so
the GPU runs are the real test. They are queued plain and paper's placement first (seeds 1–3), then the remaining
LP-step seeds.
