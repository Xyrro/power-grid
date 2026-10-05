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

## 80 % fixed (pass B; first 12 instances, one back-to-back full MILP per instance shared by both tables: mean 42.2 s, serves 66.7 %, mean gap 0.031 %)

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s | speed-up |
|---|---|---|---|---|---|---|---|---|---|---|
| full MILP (same run) |  | 0.0 | 0.031 | 0.000 | 0.080 | 0.27 | 66.7 | 75.0 | 42.17 | 1.0x |
| RACLearn (confidence margin) | 80 | 80.0 | -0.125 | -0.000 | 0.041 | 0.30 | 66.7 | 75.0 | 27.47 | 1.5x |
| Learning to Fix (generator thresholds) | 80 | 79.3 | 0.009 | 0.000 | 0.106 | 1.59 | 83.3 | 58.3 | 26.17 | 1.6x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 80 | 80.0 | -0.108 | 0.000 | 0.099 | 0.27 | 75.0 | 58.3 | 23.40 | 1.8x |
| REINFORCE probabilities (ours, previous) | 80 | 80.0 | 1.017 | 0.117 | 2.480 | 5.99 | 91.7 | 16.7 | 11.50 | 3.7x |
| harm policy (compensated) + adequacy guard | 80 | 80.0 | 0.666 | 0.005 | 1.231 | 8.22 | 66.7 | 41.7 | 21.31 | 2.0x |

## Post-hoc extension: row feasibility check + LP-relaxation guard (pass C; same run as pass B)

Added after pass A showed catastrophic shedding / shortfall outliers that the capacity guard misses; no parameter, checked on 10 val instances before this run; the guard's LP time is included.

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s | speed-up |
|---|---|---|---|---|---|---|---|---|---|---|
| RACLearn + LP-relaxation guard (post hoc) | 95 | 84.3 | 0.138 | 0.000 | 0.313 | 1.94 | 75.0 | 58.3 | 16.10 | 2.6x |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 95 | 92.9 | 0.535 | 0.024 | 1.618 | 3.47 | 91.7 | 41.7 | 8.16 | 5.2x |
| RACLearn + LP-relaxation guard (post hoc) | 97 | 83.7 | 0.318 | 0.111 | 1.356 | 1.94 | 75.0 | 33.3 | 14.51 | 2.9x |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 97 | 80.5 | 0.319 | 0.009 | 1.182 | 2.03 | 75.0 | 50.0 | 19.48 | 2.2x |
| RACLearn + LP-relaxation guard (post hoc) | 99 | 74.0 | 0.873 | 0.269 | 1.966 | 5.98 | 83.3 | 16.7 | 19.74 | 2.1x |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 99 | 72.0 | 1.204 | 0.309 | 4.197 | 5.98 | 83.3 | 16.7 | 14.13 | 3.0x |

## Pareto data and mean gap at equal speed-up

Points marked on the frontier are not dominated (lower-or-equal mean gap and higher-or-equal speed-up) by any other rule/share on the 12 instances covered by all passes: harm policy (compensated) + adequacy guard @ 95 %, REINFORCE probabilities (ours, previous) @ 95 %, BCE, OFF x10 + adequacy guard (ours, previous) @ 80 %, harm policy + adequacy + LP-relaxation guard (post hoc) @ 95 %, RACLearn (confidence margin) @ 80 %, RACLearn + LP-relaxation guard (post hoc) @ 95 %, RACLearn + LP-relaxation guard (post hoc) @ 97 %.

Best mean gap (%) a rule attains with a share whose speed-up is at least the column value (blank = no share of that rule is that fast):

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 5.659 | 122.349 | 122.349 | 122.349 | 122.349 |  |  |  |  |
| Learning to Fix (generator thresholds) | 0.969 | 95.474 |  |  |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.537 | 0.537 | 66.241 | 66.241 | 66.241 | 66.241 | 128.499 | 128.499 |  |
| REINFORCE probabilities (ours, previous) | 1.017 | 1.017 | 5.023 | 5.023 | 5.023 | 9.032 | 9.032 | 9.032 | 9.032 |
| harm policy (compensated) + adequacy guard | 0.772 | 0.772 | 1.208 | 1.208 | 1.208 | 1.208 |  |  |  |
| RACLearn + LP-relaxation guard (post hoc) | 0.138 |  |  |  |  |  |  |  |  |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 0.319 | 0.535 | 0.535 | 0.535 |  |  |  |  |  |

Best median gap (%) at a speed-up of at least the column value:

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 0.003 | 0.030 | 0.030 | 0.030 | 0.030 |  |  |  |  |
| Learning to Fix (generator thresholds) | 0.303 | 0.921 |  |  |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.032 | 0.032 | 0.524 | 0.524 | 0.524 | 0.524 | 1.281 | 1.281 |  |
| REINFORCE probabilities (ours, previous) | 0.117 | 0.117 | 0.565 | 0.565 | 0.565 | 3.739 | 3.739 | 3.739 | 3.739 |
| harm policy (compensated) + adequacy guard | 0.003 | 0.003 | 0.024 | 0.024 | 0.024 | 0.024 |  |  |  |
| RACLearn + LP-relaxation guard (post hoc) | 0.000 |  |  |  |  |  |  |  |  |
| harm policy + adequacy + LP-relaxation guard (post hoc) | 0.009 | 0.024 | 0.024 | 0.024 |  |  |  |  |  |

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
