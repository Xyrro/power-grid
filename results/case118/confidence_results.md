| method | fixed_pct | feasible_before_repair_pct | gap_mean_% | gap_closed_% | beats_or_ties_milp_% | time_s | milp_time_s_reference |
|---|---|---|---|---|---|---|---|
| MC-dropout (RACLearn), fix 50 % | 49.718 | 100.000 | -0.008 | 101.819 | 95.000 | 4.070 | 7.018 |
| MC-dropout (RACLearn), fix 90 % | 89.831 | 100.000 | -0.007 | 101.695 | 96.000 | 0.344 | 7.018 |
| MC-dropout (RACLearn), fix 97 % | 96.802 | 100.000 | 0.133 | 68.327 | 59.000 | 0.063 | 7.018 |
| MC-dropout (RACLearn), fix 100 % | 98.876 | 100.000 | 0.165 | 60.776 | 41.000 | 0.032 | 7.018 |
| probability margin, fix 50 % | 49.718 | 100.000 | -0.006 | 101.456 | 95.000 | 4.552 | 7.018 |
| probability margin, fix 90 % | 89.831 | 100.000 | -0.007 | 101.596 | 94.000 | 0.400 | 7.018 |
| probability margin, fix 97 % | 96.864 | 100.000 | 0.134 | 68.141 | 58.000 | 0.065 | 7.018 |
| probability margin, fix 100 % | 98.876 | 100.000 | 0.165 | 60.776 | 41.000 | 0.030 | 7.018 |
| keep 10 most likely lines free (ours) | 94.350 | 100.000 | -0.009 | 102.235 | 94.000 | 0.331 | 7.018 |


Accuracy of the most-confident subset (switchable lines):

{
 "mc_dropout": {
  "0.3": {
   "accuracy": 0.9996233521657251,
   "share_of_opened_lines_in_subset": 0.010582010582010581
  },
  "0.5": {
   "accuracy": 0.9996610169491525,
   "share_of_opened_lines_in_subset": 0.037037037037037035
  },
  "0.7": {
   "accuracy": 0.9995964487489911,
   "share_of_opened_lines_in_subset": 0.1111111111111111
  },
  "0.9": {
   "accuracy": 0.9992467043314501,
   "share_of_opened_lines_in_subset": 0.455026455026455
  },
  "0.95": {
   "accuracy": 0.9985727029438002,
   "share_of_opened_lines_in_subset": 0.6084656084656085
  },
  "0.99": {
   "accuracy": 0.9863607829709524,
   "share_of_opened_lines_in_subset": 0.9841269841269841
  },
  "1.0": {
   "accuracy": 0.9833333333333333,
   "share_of_opened_lines_in_subset": 1.0
  }
 },
 "margin": {
  "0.3": {
   "accuracy": 0.9996233521657251,
   "share_of_opened_lines_in_subset": 0.010582010582010581
  },
  "0.5": {
   "accuracy": 0.999774011299435,
   "share_of_opened_lines_in_subset": 0.010582010582010581
  },
  "0.7": {
   "accuracy": 0.9997578692493947,
   "share_of_opened_lines_in_subset": 0.026455026455026454
  },
  "0.9": {
   "accuracy": 0.9994978028876333,
   "share_of_opened_lines_in_subset": 0.2328042328042328
  },
  "0.95": {
   "accuracy": 0.9992268807612251,
   "share_of_opened_lines_in_subset": 0.5026455026455027
  },
  "0.99": {
   "accuracy": 0.9871026650687668,
   "share_of_opened_lines_in_subset": 0.9735449735449735
  },
  "1.0": {
   "accuracy": 0.9833333333333333,
   "share_of_opened_lines_in_subset": 1.0
  }
 }
}