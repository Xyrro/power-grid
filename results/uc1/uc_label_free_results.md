| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | matches_or_beats_milp_% | unit_hour_accuracy_% | LPs_per_instance |
|---|---|---|---|---|---|---|---|
| LF: imitate repaired LP relaxation (no MILP): top-1 -> LP | 79.500 | 0.021 | 3.833 | 150.624 | 48.800 | 97.940 | 1.000 |
| LF: imitate repaired LP relaxation (no MILP): top-1 + adequacy repair -> LP | 88.000 | 0.012 | 5.848 | 104.479 | 49.200 | 97.827 | 1.000 |
| LF: imitate repaired LP relaxation (no MILP): candidate screening -> LP | 88.200 | 0.000 | 4.004 | 81.428 | 51.400 | 97.908 | 1.787 |
| LF + REINFORCE (LP critic, no MILP): top-1 -> LP | 94.100 | 0.070 | 6.413 | 8.692 | 45.500 | 97.851 | 1.000 |
| LF + REINFORCE (LP critic, no MILP): top-1 + adequacy repair -> LP | 97.400 | 0.065 | 6.958 | 7.933 | 45.800 | 97.818 | 1.000 |
| LF + REINFORCE (LP critic, no MILP): candidate screening -> LP | 95.500 | 0.041 | 6.367 | 7.886 | 47.500 | 97.863 | 1.136 |
| REINFORCE from scratch (400 steps, no MILP): top-1 -> LP | 94.400 | 0.657 | 8.548 | 15.930 | 28.900 | 97.451 | 1.000 |
| REINFORCE from scratch (400 steps, no MILP): top-1 + adequacy repair -> LP | 97.000 | 0.651 | 9.120 | 11.577 | 29.000 | 97.418 | 1.000 |
| REINFORCE from scratch (400 steps, no MILP): candidate screening -> LP | 98.600 | 0.288 | 8.067 | 8.585 | 36.800 | 97.552 | 3.460 |
| [MILP labels] GNN + symmetry + LP-relaxation features: top-1 -> LP | 74.000 | 0.012 | 2.046 | 272.639 | 49.000 | 98.149 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features: top-1 + adequacy repair -> LP | 87.700 | 0.000 | 4.647 | 165.755 | 50.900 | 97.970 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features: candidate screening -> LP | 94.600 | -0.000 | 1.286 | 11.958 | 70.100 | 98.440 | 5.210 |
| [MILP labels] GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP | 93.900 | 0.051 | 5.645 | 17.395 | 46.600 | 97.948 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features + REINFORCE: top-1 + adequacy repair -> LP | 96.600 | 0.048 | 5.725 | 13.377 | 46.900 | 97.945 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP | 97.400 | 0.005 | 4.001 | 6.246 | 49.700 | 98.027 | 1.737 |

- lp_relaxation_s_per_label: 0.026952600479125975
- milp_s_per_label: 0.4907413218617439
- labels_agree_with_milp_unit_hour_%: 97.95890410958904
- lf_bce_train_s: 789.1864466667175
- lf_rl_s: 57.96251606941223
- scratch_rl_s: 397.1527910232544
