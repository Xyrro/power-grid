| method | gap_mean_% | gap_closed_% | benefit_captured_% | beats_or_ties_milp_% | feasible_% | LPs_per_scenario |
|---|---|---|---|---|---|---|
| dual-greedy (no learning) [intact grid] | 5.366 | 80.694 | 82.969 | 49.280 | 100.000 | 16.000 |
| kNN-LP k=20, trained intact [intact grid] | 0.000 | 99.999 | 99.999 | 99.424 | 100.000 | 5.968 |
| kNN-LP k=20, trained with outages [intact grid] | -0.203 | 100.732 | 101.283 | 97.406 | 100.000 | 15.971 |
| dual-greedy (no learning) [unseen outage] | 4.509 | 82.294 | 82.575 | 54.212 | 100.000 | 15.158 |
| kNN-LP k=20, trained intact [unseen outage] | 1.196 | 95.303 | 87.754 | 78.205 | 100.000 | 5.947 |
| kNN-LP k=20, trained with outages [unseen outage] | 0.620 | 97.565 | 93.304 | 84.799 | 100.000 | 16.018 |
| MLP+duals, trained intact: top-1 [intact grid] | 11.020 | 60.348 | 60.369 | 48.703 | 62.536 | 1.000 |
| MLP+duals, trained intact: screening [intact grid] | 0.006 | 99.980 | 99.949 | 98.847 | 100.000 | 26.300 |
| MLP+duals, trained intact: top-1 [unseen outage] | 12.396 | 51.321 | 44.683 | 23.077 | 47.436 | 1.000 |
| MLP+duals, trained intact: screening [unseen outage] | 0.663 | 97.395 | 94.650 | 77.106 | 100.000 | 26.310 |
| GNN+duals, trained intact: top-1 [intact grid] | 10.126 | 63.566 | 66.621 | 54.467 | 67.147 | 1.000 |
| GNN+duals, trained intact: screening [intact grid] | 0.020 | 99.930 | 99.910 | 98.271 | 100.000 | 26.037 |
| GNN+duals, trained intact: top-1 [unseen outage] | 12.029 | 52.761 | 46.513 | 26.007 | 54.945 | 1.000 |
| GNN+duals, trained intact: screening [unseen outage] | 1.243 | 95.120 | 91.801 | 74.359 | 100.000 | 26.136 |
| MLP+duals, trained with outages: top-1 [intact grid] | 11.235 | 59.575 | 58.174 | 48.415 | 63.977 | 1.000 |
| MLP+duals, trained with outages: screening [intact grid] | 0.012 | 99.955 | 99.918 | 98.271 | 100.000 | 26.311 |
| MLP+duals, trained with outages: top-1 [unseen outage] | 8.391 | 67.046 | 60.053 | 48.901 | 65.385 | 1.000 |
| MLP+duals, trained with outages: screening [unseen outage] | 0.059 | 99.767 | 99.401 | 93.407 | 100.000 | 26.249 |
| GNN+duals, trained with outages: top-1 [intact grid] | 12.505 | 55.006 | 53.948 | 31.988 | 54.467 | 1.000 |
| GNN+duals, trained with outages: screening [intact grid] | 0.019 | 99.933 | 99.887 | 96.830 | 100.000 | 26.133 |
| GNN+duals, trained with outages: top-1 [unseen outage] | 10.434 | 59.025 | 56.172 | 35.714 | 57.326 | 1.000 |
| GNN+duals, trained with outages: screening [unseen outage] | 0.114 | 99.552 | 98.874 | 90.110 | 100.000 | 26.123 |
