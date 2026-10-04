| method | gap_mean_% | gap_p95_% | benefit_captured_% | beats_or_ties_milp_% | feasible_% | n_open | LPs_per_scenario | time_ms |
|---|---|---|---|---|---|---|---|---|
| exhaustive greedy (exact LP for all lines) | 0.203 | 0.850 | 71.145 | 48.000 | 100.000 | 1.920 | 422.925 | 2215.603 |
| learned-value greedy (R=1 LP checks/step) | 0.226 | 0.807 | 61.771 | 40.500 | 100.000 | 1.755 | 3.285 | 25.905 |
| learned-value greedy (R=2 LP checks/step) | 0.221 | 0.804 | 64.846 | 42.000 | 100.000 | 1.825 | 5.640 | 34.119 |
| learned-value greedy (R=4 LP checks/step) | 0.216 | 0.804 | 66.413 | 42.000 | 100.000 | 1.885 | 10.420 | 60.261 |
| learned-value beam search (B=3, R=3) | 0.217 | 0.804 | 66.918 | 42.000 | 100.000 | 1.885 | 19.910 | 131.053 |
| learned-value beam search (B=5, R=5) | 0.213 | 0.804 | 68.069 | 42.000 | 100.000 | 1.890 | 45.365 | 282.420 |

- label_time_s_per_scenario: 2.505
- label_LPs_per_scenario: 411.975
- train_time_s: 332.357
- val_best_line_top1_%: 37.063
- val_best_line_in_top3_%: 52.098
- val_spearman: 0.793
- dual_estimate_best_line_top1_%: 0.000
- dual_estimate_best_line_in_top3_%: 16.434
