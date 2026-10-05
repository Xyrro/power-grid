# Fixing policy learned from solver outcomes — B2 (uc12), first 60 test instances

Every reduced MILP is solved in the same worker process right after the full MILP of the same instance (60 s limit, 0.1 % MIP gap, the dataset settings), 2 workers, machine shared with other jobs. Gaps are to `test['obj']` (negative = cheaper than the dataset MILP); served = no load shedding and no reserve shortfall. Same probability model (saved BCE GNN `uc_model1_4.pt`) for RACLearn, Learning to Fix, the asymmetric rule and the proposed policy; the REINFORCE rule uses `uc_model1_rl.pt`.

## 90 / 95 / 97 % fixed (pass A, 60 instances)

Full MILP in this pass: mean 39.6 s, serves 90.0 %, mean gap 0.025 % to the dataset MILP.

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s | speed-up |
|---|---|---|---|---|---|---|---|---|---|---|
| full MILP (same run) |  | 0.0 | 0.025 | 0.000 | 0.037 | 0.57 | 90.0 | 88.3 | 39.60 | 1.0x |
| RACLearn (confidence margin) | 90 | 90.0 | 1.384 | 0.000 | 0.584 | 55.53 | 86.7 | 55.0 | 17.63 | 2.2x |
| Learning to Fix (generator thresholds) | 90 | 90.1 | 7.814 | 0.339 | 3.981 | 391.89 | 86.7 | 25.0 | 16.15 | 2.5x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 90 | 89.9 | 0.413 | 0.055 | 1.447 | 3.77 | 96.7 | 40.0 | 11.09 | 3.6x |
| REINFORCE probabilities (ours, previous) | 90 | 90.0 | 3.150 | 0.474 | 9.520 | 37.18 | 91.7 | 20.0 | 5.08 | 7.8x |
| harm policy (compensated) + adequacy guard | 90 | 89.9 | 0.469 | 0.002 | 0.584 | 8.81 | 90.0 | 48.3 | 10.69 | 3.7x |
| RACLearn (confidence margin) | 95 | 95.0 | 38.509 | 0.087 | 38.454 | 759.25 | 80.0 | 35.0 | 7.60 | 5.2x |
| Learning to Fix (generator thresholds) | 95 | 95.0 | 49.394 | 1.514 | 90.820 | 845.81 | 63.3 | 15.0 | 9.90 | 4.0x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 95 | 94.9 | 14.004 | 0.268 | 3.781 | 565.33 | 90.0 | 21.7 | 5.35 | 7.4x |
| REINFORCE probabilities (ours, previous) | 95 | 95.0 | 5.501 | 1.482 | 20.151 | 49.05 | 91.7 | 6.7 | 2.24 | 17.7x |
| harm policy (compensated) + adequacy guard | 95 | 94.9 | 13.909 | 0.052 | 3.615 | 678.59 | 88.3 | 35.0 | 5.09 | 7.8x |
| RACLearn (confidence margin) | 97 | 97.0 | 53.955 | 0.398 | 48.898 | 1286.93 | 63.3 | 26.7 | 5.47 | 7.2x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 97 | 96.9 | 32.101 | 0.468 | 10.599 | 758.68 | 85.0 | 16.7 | 3.25 | 12.2x |
| REINFORCE probabilities (ours, previous) | 97 | 97.0 | 7.383 | 2.426 | 17.266 | 67.82 | 81.7 | 3.3 | 2.68 | 14.8x |
| harm policy (compensated) + adequacy guard | 97 | 96.9 | 20.086 | 0.245 | 12.290 | 697.58 | 78.3 | 25.0 | 4.83 | 8.2x |

## Pareto data and mean gap at equal speed-up

Points marked on the frontier are not dominated (lower-or-equal mean gap and higher-or-equal speed-up) by any other rule/share on the 60 instances covered by all passes: BCE, OFF x10 + adequacy guard (ours, previous) @ 90 %, harm policy (compensated) + adequacy guard @ 90 %, REINFORCE probabilities (ours, previous) @ 90 %, REINFORCE probabilities (ours, previous) @ 95 %.

Best mean gap (%) a rule attains with a share whose speed-up is at least the column value (blank = no share of that rule is that fast):

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 1.384 | 38.509 | 38.509 | 38.509 | 53.955 |  |  |  |  |
| Learning to Fix (generator thresholds) | 7.814 | 49.394 |  |  |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.413 | 0.413 | 14.004 | 14.004 | 14.004 | 32.101 |  |  |  |
| REINFORCE probabilities (ours, previous) | 3.150 | 3.150 | 3.150 | 3.150 | 3.150 | 5.501 | 5.501 |  |  |
| harm policy (compensated) + adequacy guard | 0.469 | 0.469 | 13.909 | 13.909 | 13.909 |  |  |  |  |

Best median gap (%) at a speed-up of at least the column value:

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 0.000 | 0.087 | 0.087 | 0.087 | 0.398 |  |  |  |  |
| Learning to Fix (generator thresholds) | 0.339 | 1.514 |  |  |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.055 | 0.055 | 0.268 | 0.268 | 0.268 | 0.468 |  |  |  |
| REINFORCE probabilities (ours, previous) | 0.474 | 0.474 | 0.474 | 0.474 | 0.474 | 1.482 | 1.482 |  |  |
| harm policy (compensated) + adequacy guard | 0.002 | 0.002 | 0.052 | 0.052 | 0.052 |  |  |  |  |

## Model selection on validation (first 10 val instances, no full MILP in this run; selection between label variants and the guard)

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 90 | 90.0 | 1.965 | 0.107 | 2.534 | 17.83 | 80.0 | 30.0 | 22.94 |
| Learning to Fix (generator thresholds) | 90 | 90.6 | 1.823 | 0.322 | 6.115 | 8.01 | 100.0 | 40.0 | 27.27 |
| harm policy, compensated labels (proposed) | 90 | 90.0 | 0.206 | 0.085 | 0.689 | 0.78 | 90.0 | 30.0 | 17.09 |
| harm policy, uncompensated labels | 90 | 90.0 | 1.445 | 0.287 | 2.263 | 10.86 | 80.0 | 30.0 | 15.28 |
| harm policy (compensated) + adequacy guard | 90 | 90.0 | 0.206 | 0.085 | 0.689 | 0.78 | 90.0 | 30.0 | 16.44 |
| RACLearn (confidence margin) | 95 | 95.0 | 16.215 | 0.728 | 37.114 | 112.10 | 70.0 | 20.0 | 12.25 |
| Learning to Fix (generator thresholds) | 95 | 95.3 | 45.971 | 0.330 | 139.778 | 326.59 | 70.0 | 30.0 | 22.84 |
| harm policy, compensated labels (proposed) | 95 | 95.0 | 1.605 | 0.118 | 6.789 | 6.83 | 80.0 | 30.0 | 5.17 |
| harm policy, uncompensated labels | 95 | 95.0 | 4.153 | 0.458 | 7.608 | 33.72 | 90.0 | 40.0 | 16.91 |
| harm policy (compensated) + adequacy guard | 95 | 95.0 | 1.605 | 0.118 | 6.789 | 6.83 | 80.0 | 30.0 | 5.54 |
| RACLearn (confidence margin) | 97 | 97.0 | 34.176 | 1.535 | 112.459 | 118.42 | 50.0 | 20.0 | 4.84 |
| Learning to Fix (generator thresholds) | 97 | 96.8 | 63.945 | 0.355 | 195.376 | 326.59 | 60.0 | 20.0 | 21.54 |
| harm policy, compensated labels (proposed) | 97 | 97.0 | 26.663 | 0.087 | 49.574 | 233.67 | 70.0 | 30.0 | 8.13 |
| harm policy, uncompensated labels | 97 | 97.0 | 4.211 | 0.322 | 7.692 | 34.57 | 90.0 | 50.0 | 21.41 |
| harm policy (compensated) + adequacy guard | 97 | 96.9 | 5.039 | 0.087 | 18.604 | 29.12 | 70.0 | 30.0 | 8.39 |

## LP-relaxation guard on validation (first 10 val instances, sanity check before pass C)

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn + LP-relaxation guard (post hoc) | 95 | 85.8 | 0.413 | 0.082 | 1.404 | 1.98 | 100.0 | 30.0 | 23.78 |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 95 | 85.5 | 0.884 | 0.118 | 1.580 | 6.78 | 100.0 | 30.0 | 9.82 |
| RACLearn + LP-relaxation guard (post hoc) | 97 | 75.4 | 0.237 | 0.047 | 0.665 | 0.69 | 100.0 | 20.0 | 19.38 |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 97 | 83.5 | 0.797 | 0.107 | 2.941 | 3.30 | 100.0 | 10.0 | 7.81 |
| RACLearn + LP-relaxation guard (post hoc) | 99 | 66.3 | 0.313 | 0.107 | 0.755 | 1.42 | 100.0 | 10.0 | 18.29 |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 99 | 76.5 | 1.132 | 0.147 | 3.827 | 4.15 | 100.0 | 10.0 | 8.82 |
