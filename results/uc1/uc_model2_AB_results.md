## A. Model 2 accuracy (test (demand, commitment) pairs)

| method | fully_feasible_% | worst_KCL_MW | worst_overload_MW | worst_gen_limit_MW | worst_ramp_MW | cost_abs_err_% |
|---|---|---|---|---|---|---|
| direct (PG, VA) regression [framework] | 0.000 | 1012.512 | 280.327 | 26.288 | 0.000 | 7.471 |
| physics decoder | 69.667 | 6.737 | 14.896 | 0.000 | 0.000 | 7.700 |
| physics decoder + overload penalty | 74.083 | 6.737 | 9.234 | 0.000 | 0.000 | 12.819 |

Dispatch LP: 31 ms on one core

## B. Model 2 as screener

| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | LPs | spearman |
|---|---|---|---|---|---|---|
| LP-check all 20 kNN candidates | 97.500 | -0.000 | 3.949 | 19.031 | 20.000 |  |
| Model 2 [direct (PG, VA) regression [framework]], PG cost: top-3 -> LP | 22.000 | 1791.267 | 2.994 | 3180.227 | 3.000 | -0.304 |
| Model 2 [direct (PG, VA) regression [framework]], PG cost + implied shed/reserve slack: top-3 -> LP | 80.250 | 0.231 | 4.634 | 403.328 | 3.000 | 0.557 |
| Model 2 [physics decoder], PG cost: top-3 -> LP | 24.750 | 1613.948 | 2.661 | 3079.986 | 3.000 | -0.265 |
| Model 2 [physics decoder], PG cost + implied shed/reserve slack: top-3 -> LP | 81.750 | 0.356 | 4.134 | 234.127 | 3.000 | 0.727 |
| Model 2 [physics decoder + overload penalty], PG cost: top-3 -> LP | 22.750 | 1655.223 | 2.945 | 3119.775 | 3.000 | -0.285 |
| Model 2 [physics decoder + overload penalty], PG cost + implied shed/reserve slack: top-3 -> LP | 89.250 | 5.001 | 16.609 | 293.590 | 3.000 | 0.537 |
| random 3 -> LP | 83.000 | 2.062 | 14.732 | 236.131 | 3.000 |  |

## C. Dashed arrow


## D. MSE between tied optimal solutions

{}