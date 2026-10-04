| method | gap_mean_% | gap_p95_% | benefit_captured_% | beats_or_ties_milp_% | feasible_% | n_open | LPs_per_scenario |
|---|---|---|---|---|---|---|---|
| BCE imitation of MILP labels (reference): top-1 decode -> LP | 9.334 | 34.477 | 68.887 | 48.415 | 69.452 | 2.885 | 1.000 |
| BCE imitation of MILP labels (reference): candidate screening -> LP | 0.044 | 0.000 | 99.795 | 98.271 | 100.000 | 2.435 | 26.046 |
| REINFORCE from scratch, no MILP labels (1500 steps): top-1 decode -> LP | 2.379 | 10.648 | 79.625 | 48.703 | 80.692 | 2.000 | 1.000 |
| REINFORCE from scratch, no MILP labels (1500 steps): candidate screening -> LP | 0.483 | 1.910 | 97.590 | 48.703 | 100.000 | 2.078 | 26.003 |

- milp_label_cpu_s_total: 857.9875695705414
- n_train: 1269
- rl_scratch_time_s: 90.74881339073181
- rl_scratch_lp_solves: 7428
