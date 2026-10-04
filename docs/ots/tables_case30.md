# Result tables: case30

All-closed DC-OPF mean gap to the MILP: 27.7931 %

## Model 1 study

| method | mean gap % | gap closed % | feasible % | matches/beats MILP % | lines opened | LPs / scenario |
|---|---|---|---|---|---|---|
| all-closed DC-OPF | 27.793 | 0.000 | 100.000 | 0.000 | 0.000 | 1.000 |
| dual-greedy heuristic (R=5, no learning) | 5.366 | 80.694 | 100.000 | 49.280 | 2.222 | 16.000 |
| kNN-LP k=5 (Johnson et al.) | 0.112 | 99.597 | 100.000 | 91.931 | 2.432 | 3.807 |
| kNN-LP k=20 (Johnson et al.) | 0.000 | 99.999 | 100.000 | 99.424 | 2.441 | 5.968 |
| MLP-BCE: top-1 decode -> LP | 12.667 | 54.424 | 57.061 | 37.752 | 2.882 | 1.000 |
| MLP-BCE: candidate screening -> LP | 0.008 | 99.969 | 100.000 | 98.559 | 2.441 | 26.277 |
| GNN-BCE: top-1 decode -> LP | 13.999 | 49.633 | 52.161 | 29.971 | 2.890 | 1.000 |
| GNN-BCE: candidate screening -> LP | 0.041 | 99.854 | 100.000 | 94.524 | 2.424 | 26.357 |
| GNN-BCE +duals: top-1 decode -> LP | 9.334 | 66.417 | 69.452 | 48.415 | 2.885 | 1.000 |
| GNN-BCE +duals: candidate screening -> LP | 0.044 | 99.842 | 100.000 | 98.271 | 2.435 | 26.046 |
| GNN-BCE +duals +equiv-labels: top-1 decode -> LP | 9.296 | 66.554 | 69.164 | 46.974 | 2.896 | 1.000 |
| GNN-BCE +duals +equiv-labels: candidate screening -> LP | 0.040 | 99.857 | 100.000 | 98.559 | 2.438 | 26.277 |
| GNN-BCE +duals [raw MILP labels]: top-1 decode -> LP | 2.475 | 91.096 | 91.643 | 32.277 | 3.000 | 1.000 |
| GNN-BCE +duals [raw MILP labels]: candidate screening -> LP | 0.034 | 99.879 | 100.000 | 97.695 | 2.435 | 26.159 |
| GNN +duals +REINFORCE: top-1 decode -> LP | 0.851 | 96.936 | 98.271 | 54.755 | 2.256 | 1.000 |
| GNN +duals +REINFORCE: candidate screening -> LP | 0.098 | 99.647 | 100.000 | 78.098 | 2.383 | 26.029 |
| GNN +duals: partial-fix MILP (free=10) | 0.000 | 100.000 | 100.000 | 100.000 | 2.450 | nan |
| GNN +duals: partial-fix MILP (free=25) | -0.000 | 100.000 | 100.000 | 100.000 | 2.450 | nan |

## Learned switching values

| method | mean gap % | gap closed % | feasible % | matches/beats MILP % | lines opened | LPs / scenario |
|---|---|---|---|---|---|---|
| exhaustive greedy (exact LP for all lines) | 5.348 | 80.760 | 100.000 | 54.755 | 2.398 | 112.000 |
| learned-value greedy (R=1 LP checks/step) | 4.979 | 82.084 | 100.000 | 55.043 | 2.372 | 3.997 |
| learned-value greedy (R=2 LP checks/step) | 5.400 | 80.572 | 100.000 | 53.314 | 2.398 | 7.000 |
| learned-value greedy (R=4 LP checks/step) | 5.399 | 80.575 | 100.000 | 53.314 | 2.398 | 13.000 |
| learned-value beam search (B=3, R=3) | 0.100 | 99.640 | 100.000 | 89.625 | 2.421 | 21.493 |
| learned-value beam search (B=5, R=5) | 0.092 | 99.669 | 100.000 | 95.101 | 2.438 | 48.588 |

## Label-free REINFORCE

| method | mean gap % | gap closed % | feasible % | matches/beats MILP % | lines opened | LPs / scenario |
|---|---|---|---|---|---|---|
| BCE imitation of MILP labels (reference): top-1 decode -> LP | 9.334 | 66.417 | 69.452 | 48.415 | 2.885 | 1.000 |
| BCE imitation of MILP labels (reference): candidate screening -> LP | 0.044 | 99.842 | 100.000 | 98.271 | 2.435 | 26.046 |
| REINFORCE from scratch, no MILP labels (1500 steps): top-1 decode -> LP | 2.379 | 91.442 | 80.692 | 48.703 | 2.000 | 1.000 |
| REINFORCE from scratch, no MILP labels (1500 steps): candidate screening -> LP | 0.483 | 98.262 | 100.000 | 48.703 | 2.078 | 26.003 |

## Model 2 accuracy

| Model 2 | PG MSE | worst KCL (p.u.) | total KCL / demand % | worst overload (p.u.) | feasible % | cost error % |
|---|---|---|---|---|---|---|
| direct (PG,VA) regression [framework] | 0.001 | 0.191 | 41.556 | 0.014 | 0.000 | 1.548 |
| direct + post-hoc repair (clip, balance, VA from DC power flow) | 0.000 | 0.000 | 0.001 | 0.003 | 69.160 | 0.680 |
| physics decoder | 0.000 | 0.000 | 0.001 | 0.002 | 68.403 | 0.392 |
| physics decoder + overload penalty | 0.000 | 0.000 | 0.001 | 0.002 | 75.042 | 0.399 |

## Model 2 as screener

| method | mean gap % | gap closed % | LPs / scenario | Spearman |
|---|---|---|---|---|
| LP-verify ALL candidates | 0.000 | 100.000 | 64.046 |  |
| Model 2 [direct (PG,VA) regression [framework]] picks top-1 -> LP | 6.496 | 76.626 | 2.000 | 0.739 |
| Model 2 [direct (PG,VA) regression [framework]] picks top-3 -> LP | 3.914 | 85.917 | 4.000 | 0.739 |
| Model 2 [physics decoder] picks top-1 -> LP | 2.978 | 89.285 | 2.000 | 0.939 |
| Model 2 [physics decoder] picks top-3 -> LP | 1.379 | 95.037 | 4.000 | 0.939 |
| Model 2 [physics decoder + overload penalty] picks top-1 -> LP | 2.860 | 89.711 | 2.000 | 0.943 |
| Model 2 [physics decoder + overload penalty] picks top-3 -> LP | 1.382 | 95.029 | 4.000 | 0.943 |
| random top-3 -> LP | 11.459 | 58.772 | 4.000 |  |

## Training Model 1 through Model 2

| method | mean gap % | gap closed % | feasible % | lines opened |
|---|---|---|---|---|
| BCE-trained Model 1 (reference) | 9.334 | 66.417 | 69.452 | 2.885 |
| M1 trained via frozen Model 2 [physics decoder + overload penalty] st=False | 18.018 | 35.170 | 95.101 | 1.409 |
| M1 trained via frozen Model 2 [direct (PG,VA) regression [framework]] st=False | 19.084 | 31.336 | 100.000 | 0.867 |
| M1 trained via frozen Model 2 [physics decoder + overload penalty] st=True | 16.863 | 39.325 | 100.000 | 1.403 |
| M1 trained via frozen Model 2 [direct (PG,VA) regression [framework]] st=True | 27.822 | -0.105 | 100.000 | 0.127 |
