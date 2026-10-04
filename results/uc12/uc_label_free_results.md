| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | matches_or_beats_milp_% | unit_hour_accuracy_% | LPs_per_instance |
|---|---|---|---|---|---|---|---|
| LF: imitate repaired LP relaxation (no MILP): top-1 -> LP | 20.833 | 18.848 | 0.604 | 167.321 | 4.167 | 98.571 | 1.000 |
| LF: imitate repaired LP relaxation (no MILP): top-1 + adequacy repair -> LP | 39.167 | 7.354 | 0.870 | 112.033 | 4.167 | 98.556 | 1.000 |
| LF: imitate repaired LP relaxation (no MILP): candidate screening -> LP | 53.333 | 1.591 | 1.342 | 43.508 | 5.833 | 98.369 | 13.783 |
| LF + REINFORCE (LP critic, no MILP): top-1 -> LP | 84.167 | 4.447 | 9.215 | 14.953 | 0.833 | 96.732 | 1.000 |
| LF + REINFORCE (LP critic, no MILP): top-1 + adequacy repair -> LP | 86.667 | 4.102 | 9.169 | 11.029 | 0.833 | 96.729 | 1.000 |
| LF + REINFORCE (LP critic, no MILP): candidate screening -> LP | 88.333 | 1.872 | 4.760 | 5.476 | 0.833 | 97.681 | 14.358 |
| REINFORCE from scratch (200 steps, no MILP): top-1 -> LP | 83.333 | 67.285 | 66.242 | 227.577 | 0.000 | 66.110 | 1.000 |
| REINFORCE from scratch (200 steps, no MILP): top-1 + adequacy repair -> LP | 90.000 | 63.955 | 65.136 | 156.601 | 0.000 | 66.885 | 1.000 |
| REINFORCE from scratch (200 steps, no MILP): candidate screening -> LP | 87.500 | 43.890 | 78.323 | 74.878 | 0.000 | 75.971 | 14.633 |
| [MILP labels] GNN + symmetry + LP-relaxation features: top-1 -> LP | 16.667 | 57.462 | 0.615 | 447.087 | 1.667 | 98.343 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features: top-1 + adequacy repair -> LP | 43.333 | 5.268 | 0.791 | 220.461 | 2.500 | 98.318 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features: candidate screening -> LP | 62.500 | 1.651 | 1.858 | 58.105 | 3.333 | 98.040 | 14.575 |
| [MILP labels] GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP | 80.833 | 5.271 | 9.074 | 14.538 | 0.000 | 96.567 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features + REINFORCE: top-1 + adequacy repair -> LP | 86.667 | 4.532 | 8.709 | 13.075 | 0.000 | 96.553 |  |
| [MILP labels] GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP | 87.500 | 2.060 | 4.666 | 5.765 | 0.833 | 97.520 | 14.775 |

- lp_relaxation_s_per_label: 0.3892935156822205
- milp_s_per_label: 34.035398604393
- labels_agree_with_milp_unit_hour_%: 98.35662100456621
- lf_bce_train_s: 135.52767610549927
- lf_rl_s: 950.5874164104462
- scratch_rl_s: 1632.7970461845398
