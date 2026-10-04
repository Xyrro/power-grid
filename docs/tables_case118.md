# Result tables: case118

All-closed DC-OPF mean gap to the MILP: 0.4049 %

## Model 1 study

| method | mean gap % | gap closed % | feasible % | matches/beats MILP % | lines opened | LPs / scenario |
|---|---|---|---|---|---|---|
| all-closed DC-OPF | 0.405 | 0.000 | 100.000 | 21.500 | 0.000 | 1.000 |
| dual-greedy heuristic (R=5, no learning) | 0.289 | 28.535 | 100.000 | 30.000 | 1.380 | 11.375 |
| kNN-LP k=5 (Johnson et al.) | -0.003 | 100.767 | 100.000 | 90.000 | 1.885 | 2.990 |
| kNN-LP k=20 (Johnson et al.) | -0.004 | 101.054 | 100.000 | 97.000 | 1.910 | 4.930 |
| MLP-BCE: top-1 decode -> LP | 0.139 | 65.655 | 100.000 | 29.000 | 3.000 | 1.000 |
| MLP-BCE: candidate screening -> LP | -0.004 | 101.013 | 100.000 | 92.500 | 1.905 | 26.735 |
| GNN-BCE: top-1 decode -> LP | 0.158 | 60.977 | 100.000 | 30.000 | 2.925 | 1.000 |
| GNN-BCE: candidate screening -> LP | -0.002 | 100.379 | 100.000 | 88.500 | 1.880 | 26.925 |
| GNN-BCE +duals: top-1 decode -> LP | 0.159 | 60.776 | 100.000 | 40.000 | 2.805 | 1.000 |
| GNN-BCE +duals: candidate screening -> LP | -0.003 | 100.694 | 100.000 | 94.000 | 1.915 | 26.345 |
| GNN-BCE +duals +equiv-labels: top-1 decode -> LP | 0.061 | 84.898 | 100.000 | 44.000 | 2.800 | 1.000 |
| GNN-BCE +duals +equiv-labels: candidate screening -> LP | -0.005 | 101.167 | 100.000 | 96.000 | 1.920 | 26.470 |
| GNN +duals +REINFORCE: top-1 decode -> LP | -0.001 | 100.346 | 100.000 | 78.500 | 1.960 | 1.000 |
| GNN +duals +REINFORCE: candidate screening -> LP | -0.002 | 100.611 | 100.000 | 88.500 | 1.900 | 26.035 |
| GNN +duals: partial-fix MILP (free=10) | -0.009 | 102.298 | 100.000 | 93.000 | 1.880 | nan |
| GNN +duals: partial-fix MILP (free=25) | -0.010 | 102.368 | 100.000 | 95.000 | 1.880 | nan |

## Learned switching values

| method | mean gap % | gap closed % | feasible % | matches/beats MILP % | lines opened | LPs / scenario |
|---|---|---|---|---|---|---|
| exhaustive greedy (exact LP for all lines) | 0.203 | 49.883 | 100.000 | 48.000 | 1.920 | 422.925 |
| learned-value greedy (R=1 LP checks/step) | 0.226 | 44.253 | 100.000 | 40.500 | 1.755 | 3.285 |
| learned-value greedy (R=2 LP checks/step) | 0.221 | 45.458 | 100.000 | 42.000 | 1.825 | 5.640 |
| learned-value greedy (R=4 LP checks/step) | 0.216 | 46.546 | 100.000 | 42.000 | 1.885 | 10.420 |
| learned-value beam search (B=3, R=3) | 0.217 | 46.489 | 100.000 | 42.000 | 1.885 | 19.910 |
| learned-value beam search (B=5, R=5) | 0.213 | 47.359 | 100.000 | 42.000 | 1.890 | 45.365 |
