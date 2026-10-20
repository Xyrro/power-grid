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

**The pilot fails the rule.** The gated model is significantly worse on the mean, through feasibility (it serves
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
