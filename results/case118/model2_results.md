## A. Model 2 accuracy and constraint satisfaction (test topologies)

| method | pg_mse | va_mse | kcl_max_pu | kcl_total_rel_% | gen_viol_max_pu | line_viol_max_pu | lines_overloaded | feasible_% | cost_abs_err_% | time_ms_per_sample |
|---|---|---|---|---|---|---|---|---|---|---|
| direct (PG,VA) regression [framework] | 0.191 | 0.003 | 4.553 | 177.115 | 0.000 | 2.649 | 6.957 | 0.000 | 4.355 | 1.939 |
| direct + post-hoc repair (clip, balance, VA from DC power flow) | 0.171 | 0.005 | 0.000 | 0.000 | 0.000 | 0.094 | 1.013 | 46.887 | 1.072 | 2.786 |
| physics decoder | 0.183 | 0.004 | 0.000 | 0.000 | 0.000 | 0.066 | 0.645 | 56.925 | 1.411 | 2.574 |
| physics decoder + overload penalty | 0.228 | 0.004 | 0.000 | 0.000 | 0.000 | 0.029 | 0.252 | 80.305 | 1.874 | 2.370 |

LP solve time (HiGHS, 1 core): 19.7 ms/sample

## B. Model 2 as a candidate screener

| method | gap_mean_% | gap_closed_% | benefit_captured_% | beats_or_ties_milp_% | LPs_per_scenario | spearman |
|---|---|---|---|---|---|---|
| LP-verify ALL candidates | -0.005 | 101.264 | 101.153 | 99.500 | 64.345 |  |
| Model 2 [direct (PG,VA) regression [framework]] picks top-1 -> LP | 0.401 | 1.073 | 17.554 | 33.500 | 2.000 | 0.040 |
| Model 2 [direct (PG,VA) regression [framework]] picks top-3 -> LP | 0.397 | 1.927 | 23.683 | 37.500 | 4.000 | 0.040 |
| Model 2 [physics decoder] picks top-1 -> LP | 0.181 | 55.188 | 46.098 | 30.500 | 2.000 | 0.653 |
| Model 2 [physics decoder] picks top-3 -> LP | 0.137 | 66.077 | 60.330 | 42.500 | 4.000 | 0.653 |
| Model 2 [physics decoder + overload penalty] picks top-1 -> LP | 0.202 | 50.080 | 32.342 | 25.000 | 2.000 | 0.504 |
| Model 2 [physics decoder + overload penalty] picks top-3 -> LP | 0.166 | 59.042 | 46.070 | 33.500 | 4.000 | 0.504 |
| random top-3 -> LP | 0.284 | 29.977 | 33.395 | 27.500 | 4.000 |  |

## C. Training Model 1 through Model 2 (dashed arrow)

| method | gap_mean_% | gap_closed_% | benefit_captured_% | feasible_% | n_open | critic_cost_err_on_own_choice_pct |
|---|---|---|---|---|---|---|
| BCE-trained Model 1 (reference) | 0.159 | 60.776 | -99.965 | 100.000 | 2.805 |  |
| M1 trained via frozen Model 2 [physics decoder + overload penalty] st=False | 0.095 | 76.479 | 28.602 | 100.000 | 1.865 | 2.024 |
| M1 trained via frozen Model 2 [direct (PG,VA) regression [framework]] st=False | 0.877 | -116.478 | -599.177 | 100.000 | 2.855 | 4.880 |
| M1 trained via frozen Model 2 [physics decoder + overload penalty] st=True | 0.405 | 0.000 | 0.000 | 100.000 | 0.000 | 1.803 |
| M1 trained via frozen Model 2 [direct (PG,VA) regression [framework]] st=True | 0.494 | -22.115 | -9.631 | 9.500 | 3.000 | 8.099 |
