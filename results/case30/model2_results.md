## A. Model 2 accuracy and constraint satisfaction (test topologies)

| method | pg_mse | va_mse | kcl_max_pu | kcl_total_rel_% | gen_viol_max_pu | line_viol_max_pu | lines_overloaded | feasible_% | cost_abs_err_% | time_ms_per_sample |
|---|---|---|---|---|---|---|---|---|---|---|
| direct (PG,VA) regression [framework] | 0.001 | 0.000 | 0.191 | 41.556 | 0.001 | 0.014 | 0.545 | 0.000 | 1.548 | 0.387 |
| direct + post-hoc repair (clip, balance, VA from DC power flow) | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.003 | 0.334 | 69.160 | 0.680 | 0.418 |
| physics decoder | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.002 | 0.363 | 68.403 | 0.392 | 0.377 |
| physics decoder + overload penalty | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.002 | 0.289 | 75.042 | 0.399 | 0.377 |

LP solve time (HiGHS, 1 core): 9.0 ms/sample

## B. Model 2 as a candidate screener

| method | gap_mean_% | benefit_captured_% | beats_or_ties_milp_% | LPs_per_scenario | spearman |
|---|---|---|---|---|---|
| LP-verify ALL candidates | 0.000 | 100.000 | 100.000 | 64.046 |  |
| Model 2 [direct (PG,VA) regression [framework]] picks top-1 -> LP | 6.496 | 67.859 | 28.242 | 2.000 | 0.739 |
| Model 2 [direct (PG,VA) regression [framework]] picks top-3 -> LP | 3.914 | 75.782 | 65.706 | 4.000 | 0.739 |
| Model 2 [physics decoder] picks top-1 -> LP | 2.978 | 80.819 | 70.605 | 2.000 | 0.939 |
| Model 2 [physics decoder] picks top-3 -> LP | 1.379 | 87.003 | 84.726 | 4.000 | 0.939 |
| Model 2 [physics decoder + overload penalty] picks top-1 -> LP | 2.860 | 81.580 | 68.876 | 2.000 | 0.943 |
| Model 2 [physics decoder + overload penalty] picks top-3 -> LP | 1.382 | 87.141 | 83.573 | 4.000 | 0.943 |
| random top-3 -> LP | 11.459 | 62.297 | 8.646 | 4.000 |  |

## C. Training Model 1 through Model 2 (dashed arrow)

| method | gap_mean_% | benefit_captured_% | feasible_% | n_open | critic_cost_err_on_own_choice_pct |
|---|---|---|---|---|---|
| BCE-trained Model 1 (reference) | 9.334 | 68.887 | 69.452 | 2.885 |  |
| M1 trained via frozen Model 2 [physics decoder + overload penalty] st=False | 18.018 | 35.156 | 95.101 | 1.409 | 0.353 |
| M1 trained via frozen Model 2 [direct (PG,VA) regression [framework]] st=False | 19.084 | 31.702 | 100.000 | 0.867 | 1.501 |
| M1 trained via frozen Model 2 [physics decoder + overload penalty] st=True | 16.863 | 42.711 | 100.000 | 1.403 | 0.275 |
| M1 trained via frozen Model 2 [direct (PG,VA) regression [framework]] st=True | 27.822 | -0.081 | 100.000 | 0.127 | 0.820 |
