| method | gap_mean_% | gap_p95_% | benefit_captured_% | beats_or_ties_milp_% | feasible_% | z_exact_match_% | n_open | LPs_per_scenario | time_ms |
|---|---|---|---|---|---|---|---|---|---|
| all-closed DC-OPF | 27.793 | 37.302 | 0.000 | 0.000 | 100.000 | 0.000 | 0.000 | 1.000 | 10.590 |
| dual-greedy heuristic (R=5, no learning) | 5.366 | 21.569 | 82.969 | 49.280 | 100.000 | 32.277 | 2.222 | 16.000 | 95.900 |
| kNN-LP k=5 (Johnson et al.) | 0.112 | 0.179 | 99.355 | 91.931 | 100.000 | 60.807 | 2.432 | 3.807 | 8.008 |
| kNN-LP k=20 (Johnson et al.) | 0.000 | 0.000 | 99.999 | 99.424 | 100.000 | 67.147 | 2.441 | 5.968 | 6.045 |
| MLP-BCE: top-1 decode -> LP | 12.667 | 36.149 | 53.235 | 37.752 | 57.061 | 24.496 | 2.882 | 1.000 | 0.010 |
| MLP-BCE: candidate screening -> LP | 0.008 | 0.000 | 99.923 | 98.559 | 100.000 | 67.435 | 2.441 | 26.277 | 53.864 |
| GNN-BCE: top-1 decode -> LP | 13.999 | 36.295 | 50.511 | 29.971 | 52.161 | 19.885 | 2.890 | 1.000 | 0.384 |
| GNN-BCE: candidate screening -> LP | 0.041 | 0.043 | 99.787 | 94.524 | 100.000 | 65.706 | 2.424 | 26.357 | 28.713 |
| GNN-BCE +duals: top-1 decode -> LP | 9.334 | 34.477 | 68.887 | 48.415 | 69.452 | 29.683 | 2.885 | 1.000 | 0.377 |
| GNN-BCE +duals: candidate screening -> LP | 0.044 | 0.000 | 99.795 | 98.271 | 100.000 | 67.435 | 2.435 | 26.046 | 13.461 |
| GNN-BCE +duals +equiv-labels: top-1 decode -> LP | 9.296 | 34.450 | 68.216 | 46.974 | 69.164 | 32.565 | 2.896 | 1.000 | 0.355 |
| GNN-BCE +duals +equiv-labels: candidate screening -> LP | 0.040 | 0.000 | 99.813 | 98.559 | 100.000 | 68.588 | 2.438 | 26.277 | 13.950 |
| GNN-BCE +duals [raw MILP labels]: top-1 decode -> LP | 2.475 | 25.945 | 90.429 | 32.277 | 91.643 | 17.867 | 3.000 | 1.000 | 0.313 |
| GNN-BCE +duals [raw MILP labels]: candidate screening -> LP | 0.034 | 0.000 | 99.812 | 97.695 | 100.000 | 65.706 | 2.435 | 26.159 | 17.216 |
| GNN +duals +REINFORCE: top-1 decode -> LP | 0.851 | 2.154 | 96.426 | 54.755 | 98.271 | 34.582 | 2.256 | 1.000 | 0.000 |
| GNN +duals +REINFORCE: candidate screening -> LP | 0.098 | 0.672 | 99.561 | 78.098 | 100.000 | 50.720 | 2.383 | 26.029 | 0.000 |
| GNN +duals: partial-fix MILP (free=10) | 0.000 | 0.000 | 100.000 | 100.000 | 100.000 | 68.000 | 2.450 | nan | 77.296 |
| GNN +duals: partial-fix MILP (free=25) | -0.000 | 0.000 | 100.000 | 100.000 | 100.000 | 59.000 | 2.450 | nan | 324.140 |
