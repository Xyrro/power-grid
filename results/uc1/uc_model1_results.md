| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | matches_or_beats_milp_% | unit_hour_accuracy_% | exact_match_% | LPs_per_instance | time_s |
|---|---|---|---|---|---|---|---|---|---|
| persistence (keep units that were running) | 63.300 | 1.902 | 3.881 | 1343.859 | 19.900 | 97.127 | 19.900 |  |  |
| rounded LP relaxation | 65.000 | 0.049 | 1.594 | 242.458 | 47.100 | 97.996 | 45.800 |  |  |
| rounded LP relaxation + adequacy repair | 86.500 | 0.004 | 4.957 | 111.936 | 49.700 | 97.864 | 48.100 |  |  |
| persistence + adequacy repair | 81.400 | 1.840 | 10.906 | 565.148 | 19.900 | 96.523 | 19.900 |  |  |
| merit-order priority list (no learning) | 72.500 | 0.852 | 9.173 | 995.148 | 24.200 | 96.547 | 23.800 |  |  |
| LP relaxation (lower bound) | 100.000 | -0.159 | -1.761 | -1.761 | 100.000 |  |  |  |  |
| kNN-LP k=5 (Xavier et al. 2021 style) | 88.300 | 0.250 | 7.717 | 94.849 | 39.600 | 97.170 | 37.300 | 5.000 |  |
| kNN-LP k=20 (Xavier et al. 2021 style) | 96.600 | -0.000 | 3.852 | 14.284 | 58.800 | 97.818 | 54.900 | 20.000 |  |
| MLP, BCE on MILP labels (framework): top-1 -> LP | 64.000 | 0.373 | 1.889 | 488.583 | 39.000 | 98.082 | 38.500 |  |  |
| MLP, BCE on MILP labels (framework): top-1 + adequacy repair -> LP | 84.000 | 0.149 | 5.608 | 352.234 | 42.200 | 97.864 | 41.400 |  |  |
| MLP, BCE on MILP labels (framework): candidate screening -> LP | 97.200 | -0.000 | 3.815 | 6.175 | 64.300 | 98.086 | 56.700 | 6.573 |  |
| GNN, BCE on MILP labels (framework): top-1 -> LP | 62.200 | 0.778 | 2.880 | 836.474 | 33.400 | 97.653 | 33.400 |  |  |
| GNN, BCE on MILP labels (framework): top-1 + adequacy repair -> LP | 81.200 | 0.435 | 7.641 | 424.892 | 35.400 | 97.316 | 35.400 |  |  |
| GNN, BCE on MILP labels (framework): candidate screening -> LP | 94.900 | -0.000 | 3.436 | 15.741 | 54.400 | 97.638 | 50.000 | 7.497 |  |
| GNN + symmetry rank + canonical labels: top-1 -> LP | 59.400 | 1.309 | 1.999 | 846.026 | 31.200 | 97.597 | 30.400 |  |  |
| GNN + symmetry rank + canonical labels: top-1 + adequacy repair -> LP | 80.900 | 0.471 | 6.955 | 446.872 | 34.400 | 97.282 | 33.400 |  |  |
| GNN + symmetry rank + canonical labels: candidate screening -> LP | 92.900 | -0.000 | 3.780 | 29.158 | 54.000 | 97.629 | 49.700 | 7.666 |  |
| GNN + symmetry + LP-relaxation features: top-1 -> LP | 74.000 | 0.012 | 2.046 | 272.639 | 49.000 | 98.149 | 45.000 |  |  |
| GNN + symmetry + LP-relaxation features: top-1 + adequacy repair -> LP | 87.700 | 0.000 | 4.647 | 165.755 | 50.900 | 97.970 | 46.900 |  |  |
| GNN + symmetry + LP-relaxation features: candidate screening -> LP | 94.600 | -0.000 | 1.286 | 11.958 | 70.100 | 98.440 | 60.300 | 5.210 |  |
| GNN + symmetry + LP-relaxation features + REINFORCE (LP critic): top-1 -> LP | 93.900 | 0.051 | 5.645 | 17.395 | 46.600 | 97.948 | 45.200 |  |  |
| GNN + symmetry + LP-relaxation features + REINFORCE: top-1 + adequacy repair -> LP | 96.600 | 0.048 | 5.725 | 13.377 | 46.900 | 97.945 | 45.500 |  |  |
| GNN + symmetry + LP-relaxation features + REINFORCE: candidate screening -> LP | 97.400 | 0.005 | 4.001 | 6.246 | 49.700 | 98.027 | 47.300 | 1.737 |  |
| confidence fixing 50 % + MILP (RACLearn-style) | 98.000 | 0.000 | 0.002 | 0.002 | 99.500 | 99.438 | 84.000 |  | 0.341 |
| confidence fixing 80 % + MILP (RACLearn-style) | 98.000 | -0.000 | 0.015 | 0.014 | 97.500 | 99.384 | 83.000 |  | 0.171 |
| confidence fixing 90 % + MILP (RACLearn-style) | 98.000 | -0.000 | 0.100 | 0.098 | 91.500 | 99.075 | 78.000 |  | 0.095 |
| confidence fixing 95 % + MILP (RACLearn-style) | 91.500 | -0.000 | 0.369 | 18.122 | 80.000 | 98.870 | 72.500 |  | 0.058 |

- n_train: 4000
- n_test: 1000
- milp_time_mean_s: 0.5090833415985108
- milp_optimal_%: 100.0
- frac_alt_commitment_within_1e-6: 0.336
- frac_alt_commitment_within_1e-4: 0.363
- frac_labels_changed_by_canonicalisation: 0.214
- train_s[MLP, BCE on MILP labels (framework)]: 40.85504174232483
- train_s[GNN, BCE on MILP labels (framework)]: 731.536039352417
- train_s[GNN + symmetry rank + canonical labels]: 736.5800457000732
- train_s[GNN + symmetry + LP-relaxation features]: 756.2235383987427
- rl_time_s: 64.27088379859924
- note_mse: computed in uc_model2.py from dispatch solutions
