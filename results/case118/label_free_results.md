| method | gap_mean_% | gap_p95_% | gap_closed_% | benefit_captured_% | beats_or_ties_milp_% | feasible_% | n_open | LPs_per_scenario |
|---|---|---|---|---|---|---|---|---|
| BCE imitation of MILP labels (reference): top-1 decode -> LP | 0.159 | 0.902 | 60.776 | -99.965 | 40.000 | 100.000 | 2.805 | 1.000 |
| BCE imitation of MILP labels (reference): candidate screening -> LP | -0.003 | 0.002 | 100.694 | 100.827 | 94.000 | 100.000 | 1.915 | 26.345 |
| REINFORCE from scratch, no MILP labels (600 steps): top-1 decode -> LP | 0.405 | 1.296 | 0.000 | 0.000 | 21.500 | 100.000 | 0.000 | 1.000 |
| REINFORCE from scratch, no MILP labels (600 steps): candidate screening -> LP | 0.399 | 1.269 | 1.524 | 0.937 | 21.500 | 100.000 | 0.460 | 26.000 |

- milp_label_cpu_s_total: 5125.197247743607
- n_train: 800
- rl_scratch_time_s: 114.99345278739929
- rl_scratch_lp_solves: 2105
