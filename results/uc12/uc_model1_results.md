| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | matches_or_beats_milp_% | unit_hour_accuracy_% | exact_match_% | LPs_per_instance | time_s |
|---|---|---|---|---|---|---|---|---|---|
| persistence (keep units that were running) | 15.833 | 5029.244 | 23.990 | 8342.817 | 0.833 | 94.521 | 0.833 |  |  |
| rounded LP relaxation | 21.667 | 13.398 | 0.329 | 157.727 | 8.333 | 98.591 | 5.000 |  |  |
| rounded LP relaxation + adequacy repair | 36.667 | 6.038 | 0.544 | 126.164 | 9.167 | 98.564 | 5.833 |  |  |
| persistence + adequacy repair | 47.500 | 26.911 | 9.673 | 688.474 | 1.667 | 94.244 | 1.667 |  |  |
| merit-order priority list (no learning) | 19.167 | 307.055 | 2.007 | 2176.692 | 0.000 | 94.413 | 0.000 |  |  |
| LP relaxation (lower bound) | 100.000 | -0.295 | -0.806 | -0.806 | 100.000 |  |  |  |  |
| kNN-LP k=5 (Xavier et al. 2021 style) | 64.167 | 11.287 | 11.671 | 212.614 | 0.000 | 95.207 | 0.000 | 5.000 |  |
| kNN-LP k=20 (Xavier et al. 2021 style) | 79.167 | 7.186 | 9.466 | 21.727 | 0.833 | 95.570 | 0.833 | 20.000 |  |
| MLP, BCE on MILP labels (framework): top-1 -> LP | 15.833 | 230.222 | 1.877 | 1106.430 | 0.000 | 97.480 | 0.000 |  |  |
| MLP, BCE on MILP labels (framework): top-1 + adequacy repair -> LP | 40.000 | 18.634 | 1.208 | 626.959 | 0.000 | 97.498 | 0.000 |  |  |
| MLP, BCE on MILP labels (framework): candidate screening -> LP | 70.000 | 3.734 | 6.941 | 63.951 | 0.000 | 96.905 | 0.000 | 14.925 |  |
| GNN, BCE on MILP labels (framework): top-1 -> LP | 10.000 | 247.586 | 2.551 | 1073.161 | 0.000 | 97.794 | 0.000 |  |  |
| GNN, BCE on MILP labels (framework): top-1 + adequacy repair -> LP | 40.833 | 14.555 | 1.662 | 493.844 | 0.000 | 97.691 | 0.000 |  |  |
| GNN, BCE on MILP labels (framework): candidate screening -> LP | 63.333 | 3.916 | 3.820 | 192.168 | 0.833 | 97.525 | 0.833 | 14.783 |  |
| GNN + symmetry rank + canonical labels: top-1 -> LP | 8.333 | 331.886 | 0.897 | 1167.529 | 0.000 | 97.817 | 0.000 |  |  |
| GNN + symmetry rank + canonical labels: top-1 + adequacy repair -> LP | 40.833 | 14.848 | 1.204 | 536.673 | 0.833 | 97.696 | 0.833 |  |  |
| GNN + symmetry rank + canonical labels: candidate screening -> LP | 60.833 | 4.789 | 3.211 | 208.146 | 0.833 | 97.540 | 0.833 | 14.800 |  |
| GNN + symmetry + LP-relaxation features: top-1 -> LP | 16.667 | 57.462 | 0.615 | 447.087 | 1.667 | 98.343 | 1.667 |  |  |
| GNN + symmetry + LP-relaxation features: top-1 + adequacy repair -> LP | 43.333 | 5.268 | 0.791 | 220.461 | 2.500 | 98.318 | 1.667 |  |  |
| GNN + symmetry + LP-relaxation features: candidate screening -> LP | 62.500 | 1.651 | 1.858 | 58.105 | 3.333 | 98.040 | 2.500 | 14.575 |  |
| GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP | 80.833 | 5.271 | 9.074 | 14.538 | 0.000 | 96.567 | 0.000 |  |  |
| GNN + symmetry + LP-relaxation features + REINFORCE: top-1 + adequacy repair -> LP | 86.667 | 4.532 | 8.709 | 13.075 | 0.000 | 96.553 | 0.000 |  |  |
| GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP | 87.500 | 2.060 | 4.666 | 5.765 | 0.833 | 97.520 | 0.833 | 14.775 |  |
| confidence fixing 50 % + MILP (RACLearn-style) | 88.333 | 0.000 | -0.048 | -0.053 | 85.000 | 99.216 | 26.667 |  | 20.687 |
| confidence fixing 80 % + MILP (RACLearn-style) | 88.333 | 0.000 | -0.004 | -0.009 | 71.667 | 99.111 | 21.667 |  | 14.781 |
| confidence fixing 90 % + MILP (RACLearn-style) | 86.667 | 0.000 | 0.154 | 1.384 | 55.000 | 98.876 | 23.333 |  | 9.939 |
| confidence fixing 95 % + MILP (RACLearn-style) | 80.000 | 0.087 | 0.181 | 38.509 | 35.000 | 98.575 | 16.667 |  | 4.329 |

- n_train: 500
- n_test: 120
- milp_time_mean_s: 28.62916024128596
- milp_optimal_%: 82.5
- frac_labels_changed_by_canonicalisation: 0.625
- train_s[MLP, BCE on MILP labels (framework)]: 17.196396350860596
- train_s[GNN, BCE on MILP labels (framework)]: 141.64505577087402
- train_s[GNN + symmetry rank + canonical labels]: 147.54911136627197
- train_s[GNN + symmetry + LP-relaxation features]: 150.134916305542
- rl_time_s: 752.1045939922333
- note_mse: computed in uc_model2.py from dispatch solutions
