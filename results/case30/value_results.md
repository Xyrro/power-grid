| method | gap_mean_% | gap_p95_% | benefit_captured_% | beats_or_ties_milp_% | feasible_% | n_open | LPs_per_scenario | time_ms |
|---|---|---|---|---|---|---|---|---|
| exhaustive greedy (exact LP for all lines) | 5.348 | 21.559 | 83.116 | 54.755 | 100.000 | 2.398 | 112.000 | 260.499 |
| learned-value greedy (R=1 LP checks/step) | 4.979 | 21.448 | 84.049 | 55.043 | 100.000 | 2.372 | 3.997 | 11.713 |
| learned-value greedy (R=2 LP checks/step) | 5.400 | 21.564 | 82.851 | 53.314 | 100.000 | 2.398 | 7.000 | 17.286 |
| learned-value greedy (R=4 LP checks/step) | 5.399 | 21.559 | 82.855 | 53.314 | 100.000 | 2.398 | 13.000 | 33.452 |
| learned-value beam search (B=3, R=3) | 0.100 | 0.138 | 99.440 | 89.625 | 100.000 | 2.421 | 21.493 | 53.894 |
| learned-value beam search (B=5, R=5) | 0.092 | 0.000 | 99.514 | 95.101 | 100.000 | 2.438 | 48.588 | 127.876 |

- label_time_s_per_scenario: 0.206
- label_LPs_per_scenario: 112.000
- train_time_s: 400.772
- val_best_line_top1_%: 69.111
- val_best_line_in_top3_%: 80.222
- val_spearman: 0.733
- dual_estimate_best_line_top1_%: 0.667
- dual_estimate_best_line_in_top3_%: 40.444
