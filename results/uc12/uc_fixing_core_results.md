60 test instances, uc12

| method | fixed_share_ | gap_mean_% | gap_median_% | no_shed_no_shortfall_% | matches_or_beats_milp_% | time_s | speedup_x | released_per_instance | fallback_ |
|---|---|---|---|---|---|---|---|---|---|
| full MILP (time from data generation) |  |  |  |  |  | 28.297 |  |  |  |
| 90 %: BCE, symmetric | 89.954 | 1.384 | 0.000 | 86.667 | 55.000 | 9.898 | 2.859 | 0.000 | 0.000 |
| 90 %: BCE, asymmetric k=10 + adequacy guard | 89.939 | 0.413 | 0.055 | 96.667 | 40.000 | 5.960 | 4.748 | 0.133 | 0.000 |
| 90 %: REINFORCE, symmetric | 89.954 | 3.150 | 0.474 | 91.667 | 20.000 | 2.565 | 11.033 | 0.000 | 0.000 |
| 95 %: BCE, symmetric | 94.977 | 38.509 | 0.087 | 80.000 | 35.000 | 4.094 | 6.911 | 0.000 | 0.000 |
| 95 %: BCE, asymmetric k=10 + adequacy guard | 94.937 | 14.004 | 0.268 | 90.000 | 21.667 | 2.673 | 10.587 | 0.350 | 0.000 |
| 95 %: REINFORCE, symmetric | 94.977 | 5.501 | 1.482 | 91.667 | 6.667 | 1.041 | 27.178 | 0.000 | 0.000 |
