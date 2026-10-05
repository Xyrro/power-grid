uc24 val, 6 instances, cfg {'T': 24, 'time_limit': 300.0, 'mip_gap': 0.001}

| method | n | fixed_share_% | gap_mean_% | gap_median_% | gap_max_% | served_% | time_mean_s | speedup_mean_x | speedup_median_x | cpu_speedup_mean_x | ttq_reached_% | ttq_speedup_median_x | limit_hit_% | fallback_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LP relaxation (reference; no full MILP) | 6 |  | 0.000 | 0.000 |  | 0.000 | nan |  |  |  |  |  |  |  |
| 80%|bce_g | 6 | 79.966 | 0.666 | 0.693 | 1.347 | 83.333 | 54.909 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 80%|bce_asym_g | 6 | 79.966 | 0.636 | 0.745 | 1.140 | 100.0 | 45.802 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 80%|rl_g | 6 | 79.966 | 0.761 | 0.797 | 1.159 | 100.0 | 43.160 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 90%|bce_g | 6 | 89.869 | 0.715 | 0.784 | 1.258 | 83.333 | 32.978 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 90%|bce_asym_g | 6 | 89.954 | 1.040 | 1.120 | 1.465 | 83.333 | 29.243 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 90%|rl_g | 6 | 89.945 | 2.597 | 1.101 | 10.370 | 100.0 | 19.815 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 95%|bce_g | 6 | 92.209 | 0.983 | 0.797 | 2.077 | 100.0 | 35.826 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 95%|bce_asym_g | 6 | 94.901 | 1.505 | 1.325 | 2.650 | 100.0 | 13.594 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
| 95%|rl_g | 6 | 93.084 | 4.220 | 3.265 | 11.674 | 100.0 | 9.197 | nan | nan | nan | 0.000 | nan | 0.000 | 0.000 |
