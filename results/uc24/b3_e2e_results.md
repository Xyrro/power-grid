40 test instances (uc24, T=24); gaps vs the full MILP reference

| method | no_shed_no_shortfall_% | gap_median_% | gap_mean_served_% | gap_mean_% | matches_or_beats_milp_% | units_on | LPs_per_instance |
|---|---|---|---|---|---|---|---|
| full MILP (reference) | 97.500 | 0.000 | 0.000 | 0.000 | 100.000 | 13.682 | 0.000 |
| rounded LP relaxation + adequacy + min up/down repair (training label) | 17.500 | 9.452 | 0.248 | 55.087 | 2.500 | 13.546 | 1.000 |
| rounded LP relaxation + block repair | 75.000 | 1.522 | 1.760 | 18.539 | 2.500 | 13.860 | 1.000 |
| LF-BCE (imitation, no MILP): top-1 -> LP | 2.500 | 361.939 | 2.472 | 655.534 | 0.000 | 13.601 | 1.000 |
| LF-BCE (imitation, no MILP): top-1 + block repair -> LP | 65.000 | 3.262 | 2.664 | 35.876 | 0.000 | 14.338 | 1.000 |
| LF-BCE (imitation, no MILP): threshold 0.3 (val) + block repair -> LP | 75.000 | 4.546 | 4.119 | 17.106 | 0.000 | 14.688 | 1.000 |
| LF-BCE (imitation, no MILP): candidate screening (block repair) -> LP | 85.000 | 2.189 | 3.165 | 6.159 | 0.000 | 14.373 | 15.000 |
| LF-BCE + REINFORCE (LP critic, no MILP): top-1 -> LP | 75.000 | 11.946 | 13.995 | 28.794 | 0.000 | 16.781 | 1.000 |
| LF-BCE + REINFORCE (LP critic, no MILP): top-1 + block repair -> LP | 87.500 | 10.341 | 13.709 | 13.862 | 0.000 | 16.835 | 1.000 |
| LF-BCE + REINFORCE (LP critic, no MILP): threshold 0.5 (val) + block repair -> LP | 87.500 | 10.341 | 13.709 | 13.862 | 0.000 | 16.835 | 1.000 |
| LF-BCE + REINFORCE (LP critic, no MILP): candidate screening (block repair) -> LP | 92.500 | 4.283 | 5.879 | 6.383 | 0.000 | 15.047 | 14.950 |
