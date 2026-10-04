## A. Model 2 accuracy (test (demand, commitment) pairs)

| method | fully_feasible_% | worst_KCL_MW | worst_overload_MW | worst_gen_limit_MW | worst_ramp_MW | cost_abs_err_% |
|---|---|---|---|---|---|---|
| direct (PG, VA) regression [framework] | 0.000 | 1012.512 | 280.327 | 26.288 | 0.000 | 7.471 |
| physics decoder | 69.667 | 6.737 | 14.896 | 0.000 | 0.000 | 7.700 |
| physics decoder + overload penalty | 74.083 | 6.737 | 9.234 | 0.000 | 0.000 | 12.819 |

Dispatch LP: 30 ms on one core

## B. Model 2 as screener

| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | LPs | spearman |
|---|---|---|---|---|---|---|
| LP-check all 20 kNN candidates | 97.500 | -0.000 | 3.949 | 19.031 | 1000000.000 |  |
| Model 2 [direct (PG, VA) regression [framework]] picks top-3 -> LP | 22.000 | 1791.267 | 2.994 | 3180.227 | 3.000 | -0.304 |
| Model 2 [physics decoder] picks top-3 -> LP | 24.750 | 1613.948 | 2.661 | 3079.986 | 3.000 | -0.265 |
| Model 2 [physics decoder + overload penalty] picks top-3 -> LP | 22.750 | 1655.223 | 2.945 | 3119.775 | 3.000 | -0.285 |
| random 3 -> LP | 83.000 | 2.062 | 14.732 | 236.131 | 3.000 |  |

## C. Dashed arrow

| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | unit_hour_accuracy_% | units_on |
|---|---|---|---|---|---|---|
| imitation (BCE, canonical labels) reference | 77.500 | 0.000 | 1.266 | 199.991 | 98.342 | 14.932 |
| M1 trained through frozen Model 2 [physics decoder + overload penalty] st=False | 71.000 | 0.472 | 6.443 | 187.807 | 97.774 | 14.578 |
| M1 trained through frozen Model 2 [physics decoder + overload penalty] st=True | 98.250 | 24.049 | 97.440 | 96.642 | 95.897 | 16.378 |
| M1 trained through frozen Model 2 [direct (PG, VA) regression [framework]] st=False | 76.000 | 24.968 | 23.598 | 1080.033 | 91.592 | 17.890 |
| M1 trained through frozen Model 2 [direct (PG, VA) regression [framework]] st=True | 75.500 | 155.475 | 136.524 | 1261.778 | 72.541 | 26.538 |
| M1 fine-tuned with exact LP sensitivities (dual gradient) | 77.750 | 0.004 | 0.580 | 144.340 | 98.373 | 14.610 |

## D. MSE between tied optimal solutions

{
 "n_tied_instances": 300,
 "pg_mse_tied_mean": 0.013226645202054651,
 "pg_mse_tied_max": 0.3452739729042466,
 "va_mse_tied_mean": 0.0003379013328438303,
 "commitment_hamming_tied_mean": 2.1866666666666665
}