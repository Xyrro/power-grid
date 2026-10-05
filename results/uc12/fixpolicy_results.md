# Fixing policy learned from solver outcomes — B2 (uc12), first 47 test instances

Every reduced MILP is solved in the same worker process right after the full MILP of the same instance (60 s limit, 0.1 % MIP gap, the dataset settings), 2 workers, machine shared with other jobs. Gaps are to `test['obj']` (negative = cheaper than the dataset MILP); served = no load shedding and no reserve shortfall. Same probability model (saved BCE GNN `uc_model1_4.pt`) for RACLearn, Learning to Fix, the asymmetric rule and the proposed policy; the REINFORCE rule uses `uc_model1_rl.pt`.

## 90 / 95 / 97 % fixed (pass A, 47 instances)

Full MILP in this pass: mean 38.0 s, serves 89.4 %, mean gap 0.010 % to the dataset MILP.

| rule | share | fixed % | mean gap % | median gap % | p90 gap % | max gap % | served % | matches MILP % | time s | speed-up |
|---|---|---|---|---|---|---|---|---|---|---|
| full MILP (same run) |  | 0.0 | 0.010 | 0.000 | 0.000 | 0.27 | 89.4 | 91.5 | 37.98 | 1.0x |
| RACLearn (confidence margin) | 90 | 90.0 | 1.636 | 0.000 | 0.468 | 55.53 | 85.1 | 57.4 | 15.82 | 2.4x |
| Learning to Fix (generator thresholds) | 90 | 90.0 | 9.009 | 0.265 | 2.996 | 391.89 | 89.4 | 27.7 | 13.49 | 2.8x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 90 | 90.0 | 0.324 | 0.009 | 1.422 | 2.94 | 95.7 | 46.8 | 10.51 | 3.6x |
| REINFORCE probabilities (ours, previous) | 90 | 90.0 | 2.730 | 0.446 | 9.760 | 32.07 | 93.6 | 23.4 | 3.70 | 10.3x |
| harm policy (compensated) + adequacy guard | 90 | 89.9 | 0.315 | 0.000 | 0.496 | 8.81 | 91.5 | 57.4 | 10.23 | 3.7x |
| RACLearn (confidence margin) | 95 | 95.0 | 45.268 | 0.018 | 23.428 | 759.25 | 78.7 | 42.6 | 7.25 | 5.2x |
| Learning to Fix (generator thresholds) | 95 | 94.9 | 46.368 | 1.639 | 119.680 | 845.81 | 59.6 | 12.8 | 7.45 | 5.1x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 95 | 95.0 | 17.514 | 0.225 | 4.876 | 565.33 | 87.2 | 25.5 | 4.77 | 8.0x |
| REINFORCE probabilities (ours, previous) | 95 | 95.0 | 5.121 | 1.589 | 15.462 | 49.05 | 93.6 | 6.4 | 1.23 | 30.8x |
| harm policy (compensated) + adequacy guard | 95 | 94.9 | 15.839 | 0.007 | 2.104 | 678.59 | 89.4 | 44.7 | 4.81 | 7.9x |
| RACLearn (confidence margin) | 97 | 97.0 | 64.171 | 0.344 | 91.381 | 1286.93 | 61.7 | 31.9 | 5.56 | 6.8x |
| BCE, OFF x10 + adequacy guard (ours, previous) | 97 | 97.0 | 36.851 | 0.344 | 5.696 | 758.68 | 85.1 | 21.3 | 2.83 | 13.4x |
| REINFORCE probabilities (ours, previous) | 97 | 97.0 | 6.710 | 2.404 | 15.260 | 52.01 | 83.0 | 4.3 | 3.07 | 12.4x |
| harm policy (compensated) + adequacy guard | 97 | 96.9 | 20.777 | 0.074 | 4.846 | 697.58 | 80.9 | 31.9 | 5.27 | 7.2x |

## Pareto data and mean gap at equal speed-up

Points marked on the frontier are not dominated (lower-or-equal mean gap and higher-or-equal speed-up) by any other rule/share on the 47 instances covered by all passes: harm policy (compensated) + adequacy guard @ 90 %, REINFORCE probabilities (ours, previous) @ 90 %, REINFORCE probabilities (ours, previous) @ 95 %.

Best mean gap (%) a rule attains with a share whose speed-up is at least the column value (blank = no share of that rule is that fast):

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 1.636 | 45.268 | 45.268 | 45.268 |  |  |  |  |  |
| Learning to Fix (generator thresholds) | 9.009 | 46.368 | 46.368 | 46.368 |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.324 | 0.324 | 17.514 | 17.514 | 17.514 | 36.851 |  |  |  |
| REINFORCE probabilities (ours, previous) | 2.730 | 2.730 | 2.730 | 2.730 | 2.730 | 2.730 | 5.121 | 5.121 | 5.121 |
| harm policy (compensated) + adequacy guard | 0.315 | 0.315 | 15.839 | 15.839 | 15.839 |  |  |  |  |

Best median gap (%) at a speed-up of at least the column value:

| rule | 2x | 3x | 4x | 5x | 7x | 10x | 15x | 20x | 30x |
|---|---|---|---|---|---|---|---|---|---|
| RACLearn (confidence margin) | 0.000 | 0.018 | 0.018 | 0.018 |  |  |  |  |  |
| Learning to Fix (generator thresholds) | 0.265 | 1.639 | 1.639 | 1.639 |  |  |  |  |  |
| BCE, OFF x10 + adequacy guard (ours, previous) | 0.009 | 0.009 | 0.225 | 0.225 | 0.225 | 0.344 |  |  |  |
| REINFORCE probabilities (ours, previous) | 0.446 | 0.446 | 0.446 | 0.446 | 0.446 | 0.446 | 1.589 | 1.589 | 1.589 |
| harm policy (compensated) + adequacy guard | 0.000 | 0.000 | 0.007 | 0.007 | 0.007 |  |  |  |  |

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
