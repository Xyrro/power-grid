# Fixing policy learned from solver outcomes — B2 (uc12), first 33 test instances

Every reduced MILP is solved in the same worker process right after the full MILP of the same instance (60 s limit, 0.1 % MIP gap, the dataset settings), 2 workers, machine shared with other jobs. Gaps are to `test['obj']` (negative = cheaper than the dataset MILP); served = no load shedding and no reserve shortfall. Same probability model (saved BCE GNN `uc_model1_4.pt`) for RACLearn, Learning to Fix, the asymmetric rule and the proposed policy; the REINFORCE rule uses `uc_model1_rl.pt`.

## 90 / 95 / 97 % fixed (pass A, 33 instances)

Full MILP in this pass: mean 38.6 s, serves 87.9 %, mean gap 0.014 % to the dataset MILP.

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s | speed-up |
|---|---|---|---|---|---|---|---|---|---|---|
| full MILP (same run) |  | 0.0 | 0.014 | 0.000 | 0.029 | 0.27 | 87.9 | 87.9 | 38.56 | 1.0x |
| RACLearn (confidence margin) | 90 | 90.0 | 2.289 | 0.000 | 1.582 | 55.53 | 78.8 | 57.6 | 15.69 | 2.5x |
| Learning to Fix (generator thresholds) | 90 | 89.8 | 12.543 | 0.296 | 3.050 | 391.89 | 90.9 | 30.3 | 12.95 | 3.0x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 90 | 90.0 | 0.427 | 0.046 | 1.513 | 2.94 | 93.9 | 42.4 | 10.84 | 3.6x |
| REINFORCE probabilities (ours, previous) | 90 | 90.0 | 3.163 | 0.444 | 10.939 | 32.07 | 90.9 | 24.2 | 3.48 | 11.1x |
| harm policy (compensated) + adequacy guard | 90 | 89.9 | 0.429 | 0.000 | 0.722 | 8.81 | 90.9 | 60.6 | 9.22 | 4.2x |
| RACLearn (confidence margin) | 95 | 95.0 | 64.115 | 0.042 | 208.081 | 759.25 | 72.7 | 42.4 | 6.35 | 6.1x |
| Learning to Fix (generator thresholds) | 95 | 94.8 | 60.905 | 1.216 | 191.257 | 845.81 | 69.7 | 15.2 | 7.16 | 5.4x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 95 | 94.9 | 24.802 | 0.409 | 11.522 | 565.33 | 81.8 | 21.2 | 4.76 | 8.1x |
| REINFORCE probabilities (ours, previous) | 95 | 95.0 | 5.726 | 1.609 | 18.265 | 49.05 | 93.9 | 6.1 | 0.97 | 39.9x |
| harm policy (compensated) + adequacy guard | 95 | 94.9 | 22.493 | 0.006 | 4.642 | 678.59 | 84.8 | 48.5 | 3.55 | 10.9x |
| RACLearn (confidence margin) | 97 | 97.0 | 90.496 | 0.344 | 279.985 | 1286.93 | 63.6 | 33.3 | 5.82 | 6.6x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 97 | 96.9 | 52.265 | 0.793 | 37.414 | 758.68 | 78.8 | 21.2 | 2.73 | 14.1x |
| REINFORCE probabilities (ours, previous) | 97 | 97.0 | 7.202 | 2.563 | 14.377 | 52.01 | 81.8 | 6.1 | 4.14 | 9.3x |
| harm policy (compensated) + adequacy guard | 97 | 96.9 | 29.315 | 0.026 | 5.272 | 697.58 | 78.8 | 39.4 | 6.31 | 6.1x |

## Pareto data and mean gap at equal speed-up

Points marked on the frontier are not dominated (lower-or-equal mean gap and higher-or-equal speed-up) by any other rule/share (passes covering the same instances as pass A): BCE, OFF x10 + adequacy guard (ours, previous) @ 90 %, harm policy (compensated) + adequacy guard @ 90 %, REINFORCE probabilities (ours, previous) @ 90 %, REINFORCE probabilities (ours, previous) @ 95 %.

Best mean gap (%) a rule attains with a share whose speed-up is at least the column value (blank = no share of that rule is that fast):

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 2.289 | 64.115 | 64.115 | 64.115 |  |  |  |  |  |
| Learning to Fix (generator thresholds) | 12.543 | 60.905 | 60.905 | 60.905 |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.427 | 0.427 | 24.802 | 24.802 | 24.802 | 52.265 |  |  |  |
| REINFORCE probabilities (ours, previous) | 3.163 | 3.163 | 3.163 | 3.163 | 3.163 | 3.163 | 5.726 | 5.726 | 5.726 |
| harm policy (compensated) + adequacy guard | 0.429 | 0.429 | 0.429 | 22.493 | 22.493 | 22.493 |  |  |  |

Best median gap (%) at a speed-up of at least the column value:

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 0.000 | 0.042 | 0.042 | 0.042 |  |  |  |  |  |
| Learning to Fix (generator thresholds) | 0.296 | 1.216 | 1.216 | 1.216 |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.046 | 0.046 | 0.409 | 0.409 | 0.409 | 0.793 |  |  |  |
| REINFORCE probabilities (ours, previous) | 0.444 | 0.444 | 0.444 | 0.444 | 0.444 | 0.444 | 1.609 | 1.609 | 1.609 |
| harm policy (compensated) + adequacy guard | 0.000 | 0.000 | 0.000 | 0.006 | 0.006 | 0.006 |  |  |  |

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
