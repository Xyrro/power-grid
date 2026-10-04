200 test instances, uc1

| method | fixed_share_ | gap_mean_% | gap_median_% | no_shed_no_shortfall_% | matches_or_beats_milp_% | time_s | speedup_x | released_per_instance | fallback_ |
|---|---|---|---|---|---|---|---|---|---|
| full MILP (same run, for timing) | 0.000 | 0.000 | 0.000 | 98.000 | 100.000 | 0.633 |  | 0.000 | 0.000 |
| 80 %: BCE, per-unit calibrated | 79.452 | 0.015 | -0.000 | 98.000 | 97.500 | 0.219 | 2.891 | 0.000 | 0.000 |
| 80 %: BCE, per-unit calibrated + adequacy guard | 79.425 | 0.015 | -0.000 | 98.000 | 97.500 | 0.215 | 2.948 | 0.020 | 0.000 |
| 80 %: REINFORCE, per-unit calibrated | 79.452 | 0.085 | -0.000 | 98.000 | 93.500 | 0.193 | 3.276 | 0.000 | 0.000 |
| 90 %: BCE, per-unit calibrated | 89.041 | 0.446 | -0.000 | 97.500 | 92.000 | 0.118 | 5.351 | 0.000 | 0.000 |
| 90 %: BCE, per-unit calibrated + adequacy guard | 88.973 | 0.446 | -0.000 | 97.500 | 92.000 | 0.120 | 5.282 | 0.050 | 0.000 |
| 90 %: REINFORCE, per-unit calibrated | 89.041 | 0.336 | -0.000 | 98.500 | 86.500 | 0.116 | 5.442 | 0.000 | 0.000 |
| 95 %: BCE, per-unit calibrated | 94.521 | 69.941 | -0.000 | 86.500 | 83.500 | 0.073 | 8.678 | 0.000 | 0.000 |
| 95 %: BCE, per-unit calibrated + adequacy guard | 94.240 | 41.236 | -0.000 | 91.500 | 83.500 | 0.076 | 8.307 | 0.205 | 0.000 |
| 95 %: REINFORCE, per-unit calibrated | 94.521 | 1.984 | -0.000 | 97.500 | 77.000 | 0.070 | 8.992 | 0.000 | 0.000 |
| 98 %: BCE, per-unit calibrated | 97.260 | 173.496 | -0.000 | 82.500 | 70.500 | 0.049 | 12.999 | 0.000 | 0.000 |
| 98 %: BCE, per-unit calibrated + adequacy guard | 96.705 | 126.712 | -0.000 | 89.500 | 71.500 | 0.058 | 10.863 | 0.405 | 0.000 |
| 98 %: REINFORCE, per-unit calibrated | 97.260 | 3.110 | -0.000 | 98.000 | 70.000 | 0.051 | 12.480 | 0.000 | 0.000 |
