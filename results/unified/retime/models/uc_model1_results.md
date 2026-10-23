| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | matches_or_beats_milp_% | unit_hour_accuracy_% | exact_match_% | LPs_per_instance | time_s |
|---|---|---|---|---|---|---|---|---|---|
| persistence (keep units that were running) | 12.500 | 4356.429 | 16.996 | 7274.833 | 1.667 | 95.131 | 1.667 |  |  |
| rounded LP relaxation | 24.167 | 13.879 | 0.642 | 86.144 | 4.167 | 98.606 | 2.500 |  |  |
| rounded LP relaxation + adequacy repair | 45.833 | 4.440 | 0.844 | 62.387 | 4.167 | 98.564 | 2.500 |  |  |
| persistence + adequacy repair | 42.500 | 39.456 | 6.768 | 542.360 | 2.500 | 94.561 | 2.500 |  |  |
| merit-order priority list (no learning) | 16.667 | 482.489 | 3.044 | 1575.770 | 0.000 | 94.554 | 0.000 |  |  |
| LP relaxation (lower bound) | 100.000 | -0.233 | -0.559 | -0.559 | 100.000 |  |  |  |  |
| kNN-LP k=5 (Xavier et al. 2021 style) | 70.833 | 10.836 | 11.329 | 192.273 | 0.000 | 94.727 | 0.000 | 5.000 |  |
| kNN-LP k=20 (Xavier et al. 2021 style) | 83.333 | 5.567 | 7.981 | 51.390 | 0.000 | 95.089 | 0.000 | 20.000 |  |
| MLP, BCE on MILP labels (framework): top-1 -> LP | 20.833 | 189.884 | 1.480 | 909.805 | 1.667 | 97.578 | 0.833 |  |  |
| MLP, BCE on MILP labels (framework): top-1 + adequacy repair -> LP | 40.000 | 31.578 | 1.510 | 491.235 | 1.667 | 97.563 | 0.833 |  |  |
| MLP, BCE on MILP labels (framework): candidate screening -> LP | 66.667 | 3.079 | 4.486 | 62.643 | 1.667 | 97.013 | 0.833 | 14.900 |  |
| GNN, BCE on MILP labels (framework): top-1 -> LP | 10.833 | 373.334 | 1.457 | 952.987 | 1.667 | 97.972 | 1.667 |  |  |
| GNN, BCE on MILP labels (framework): top-1 + adequacy repair -> LP | 36.667 | 52.143 | 1.335 | 500.502 | 2.500 | 97.805 | 2.500 |  |  |
| GNN, BCE on MILP labels (framework): candidate screening -> LP | 61.667 | 4.707 | 3.470 | 188.596 | 1.667 | 97.566 | 1.667 | 14.458 |  |
| GNN + symmetry rank + canonical labels: top-1 -> LP | 13.333 | 275.095 | 1.082 | 1035.684 | 1.667 | 97.905 | 1.667 |  |  |
| GNN + symmetry rank + canonical labels: top-1 + adequacy repair -> LP | 37.500 | 37.199 | 1.166 | 451.310 | 2.500 | 97.704 | 2.500 |  |  |
| GNN + symmetry rank + canonical labels: candidate screening -> LP | 63.333 | 3.652 | 2.763 | 200.760 | 1.667 | 97.468 | 1.667 | 14.675 |  |
| GNN + symmetry + LP-relaxation features: top-1 -> LP | 15.833 | 23.253 | 1.319 | 241.826 | 2.500 | 98.311 | 1.667 |  |  |
| GNN + symmetry + LP-relaxation features: top-1 + adequacy repair -> LP | 42.500 | 7.430 | 1.145 | 133.110 | 2.500 | 98.280 | 1.667 |  |  |
| GNN + symmetry + LP-relaxation features: candidate screening -> LP | 70.000 | 1.787 | 1.884 | 22.421 | 5.833 | 97.989 | 4.167 | 14.358 |  |
| GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP | 85.833 | 5.948 | 9.519 | 13.185 | 0.833 | 96.104 | 0.833 |  |  |
| GNN + symmetry + LP-relaxation features + REINFORCE: top-1 + adequacy repair -> LP | 90.833 | 5.580 | 9.183 | 12.520 | 0.833 | 96.099 | 0.833 |  |  |
| GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP | 93.333 | 2.527 | 5.173 | 5.377 | 1.667 | 97.039 | 1.667 | 14.333 |  |

- n_train: 500
- n_test: 120
- milp_time_mean_s: 21.042403348286946
- milp_optimal_%: 88.33333333333333
- frac_labels_changed_by_canonicalisation: 0.5583333333333333
- train_s[MLP, BCE on MILP labels (framework)]: 10.765319347381592
- train_s[GNN, BCE on MILP labels (framework)]: 87.74035477638245
- train_s[GNN + symmetry rank + canonical labels]: 73.93893051147461
- train_s[GNN + symmetry + LP-relaxation features]: 66.02249050140381
- rl_time_s: 366.0406975746155
- note_mse: computed in uc_model2.py from dispatch solutions
