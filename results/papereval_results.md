# Our results scored with the Learning to Fix protocol, next to its Table I

*Fritz, Makrides, Fetanat & Pinson, "Learning to Fix: Optimisation-Aware Machine Learning for Accelerated Unit
Commitment", arXiv 2609.39396 ("the paper"). Method and code: [`docs/methods/papereval.md`](../docs/methods/papereval.md);
all rows and extra statistics (bootstrap intervals, seed ranges, gaps on proven-reference instances, best-of-k curve):
`results/papereval_results.json`; new raw solves: `results/papereval_fresh_runs.jsonl`; overheads:
`results/papereval_overhead.json`. Produced by `scripts/uc_papereval_{run,overhead,score}.py`.*

## Summary

* **Protocol.** Every per-instance record of our studies was re-scored with the paper's metrics. The gap is
  (C − DB) / C, where DB is the best dual bound of the full MILP. Speed-up is the mean of the per-instance ratios
  T_MILP / T_m. Statistics are taken over feasible instances only, and a fallback re-solve counts as infeasible.
  Runtimes include every part of the pipeline. Served share and medians are added because shedding and reserve
  shortfall are soft in our model.
* **Runtime audit.** Our earlier fixing runs did not time four things: the LP relaxation used as GNN input
  (0.26–0.28 s per instance), the GNN pass (3 ms), the error-cost scoring (2.5 ms), and the adequacy guard / row
  release (2 ms). The end-to-end pipelines stored no times at all. All of these were measured per instance and added.
  They were scaled to the load of the earlier runs with a factor of 1.60 (fresh) and 1.18 (original); the factor was
  calibrated by recomputing the LP guard, which gave identical outcomes on 20 of 20 instances.
* **The paper's metrics are kinder to us than our own.**
  * Per-instance mean speed-ups are 1.3–4× larger than our ratio of mean times. For example, the combined pipeline
    at 98 % moves from 7.0× to 12.6×, and the error-cost rule with both guards (95 %) from 4.6× to 8.0×.
  * The gap formula caps every instance below 100 %. RACLearn at 95 % therefore falls from 49 % to 10.4 % mean gap.
  * Guarded rules gain about +0.1 pp of gap, because our reference MILP is itself 0.14–0.25 % above its bound.
* **Side by side**, as the best mean gap at a mean speed-up of at least x (feasibility ≥ 95 %):

  | | ≥ 5× | ≥ 10× | ≥ 20× |
  |---|---|---|---|
  | paper (72 h, copper plate, Gurobi) | 0.48 % (20.8×, ε = 1 %) | 0.48 % | 0.48 % |
  | B2 fresh, 12 h | 0.56 % (8.0×) | 0.68 % (11.7×; 0.84 % at 12.6× for the combined pipeline) | 2.70 % (28.8×) |
  | B2 original, 12 h | 0.44 % (5.9×) | 0.63 % (10.4×) | 1.96 % (21.5×) |
  | B3, 24 h | 0.67 % (11.5×) | 0.67 % (11.5×) | 3.06 % (36.6×) |

  At about 0.6–0.8 % mean gap we reach 8–13×; the paper reaches 0.48 % at 20.8×. Beyond 20× we pay 2–3 % (up to 8 %).
* **The paper's baselines on our benchmark** (new runs, fresh B2 set):
  * The kNN with k = 50 at constant thresholds fixes the same shares as in the paper (87 / 81 / 71 % against
    86 / 82 / 74 %) and is 100 % feasible, as in the paper.
  * It is worse on both axes. [0.1, 0.9]: 4.5 % at 5.9× (paper 2.7 % at 13.9×). [0.05, 0.95]: 2.7 % at 2.6×
    (paper 0.58 % at 6.8×). [0.01, 0.99]: 0.50 % at 2.0× (paper 0.21 % at 4.1×).
  * The cost-ranked kNN: 12.0 % at 5.4× (10.4 % at 8.7× on all 120 instances); paper 5.4 % at 7.1×.
  * So B2 is a harder benchmark for these methods, and its fast MILP (median 18 s) caps the speed-ups.
  * Measured against those same baselines, our rules improve at least as much as Learning to Fix does in the paper.
    The error-cost rule with both guards reaches about the gap of kNN [0.01, 0.99] (0.59 vs 0.50 %) at 4× its
    speed-up. In the paper, ε = 1 % vs [0.05, 0.95] is 0.48 vs 0.58 % at 3× the speed-up.

## 1. Metric definitions (paper, Sec. IV-C) and their implementation

| quantity | paper | here |
|---|---|---|
| gap | (C_m − DB) / C_m, DB = best dual bound of the full MILP, same for all methods | uc12: DB = obj · (1 − mip_gap) of the dataset's MILP. On the fresh set it is replaced by the bound of this study's back-to-back MILP where that is larger (7 instances; 3 of them proved there). uc24: the stored bound. C_m includes penalty costs |
| feasibility | the reduced problem has a solution | the same. Fallback re-solves (fixings in conflict with min up/down) are infeasible |
| runtime | total: inference + downstream optimisation | recorded solve + guard times + measured overheads (§2) |
| speed-up | mean over instances of T_MILP / T_m (max also reported) | the same, plus the median. T_MILP comes from the full MILP solved back to back in the same worker (fixing runs), or from the dataset's MILP run (end-to-end rows, uc24) |
| statistics | over feasible instances | the same; seeded rules are averaged over 3 seeds |
| (added) | – | served share (no shedding, over-generation or reserve shortfall), median gap, 95 % instance-bootstrap intervals (JSON) |

## 2. What our runtimes did and did not include

| part | uc12 fixing | uc24 fixing | end-to-end | measured here, per instance (uc12 / uc24) |
|---|---|---|---|---|
| reduced MILP + fallback | timed | timed | – | – |
| LP-relaxation guard | timed | timed | – | – |
| min up/down row release | not timed | timed | – | 1.4–1.5 ms |
| LP relaxation (GNN / error-cost features) | **not timed** | timed (`t_rel`, 1.23 s) | **not timed** | 0.26–0.28 s / stored |
| GNN forward pass, one instance | not timed | not timed | not timed | 3.2–3.3 ms / 3.5 ms |
| error-cost features + scoring | not timed | – | – | 2.5 ms |
| ranking, adequacy guard | not timed | not timed | – | 1.0 ms / 1.5 ms |
| block repair, dispatch LP(s) | – | – | **not timed** | 4.5 ms + 96 ms per LP / 11 ms + 198 ms per LP |

* **Load scaling.** The earlier runs were timed under load 6–8 on 4 cores. The LP guard of RACLearn + LP guard
  (95 %) was recomputed on 20 instances per set: same fixings, same guard outcome on 20 of 20. The recorded times are
  1.60× (fresh) and 1.18× (original) the current ones. On uc24 the stored LP-relaxation times are 1.40× the current
  ones. Measured overheads are multiplied by these factors.
* **Size of the correction.**
  * The unaccounted overhead is ~0.33–0.42 s per instance, almost all of it the LP relaxation.
  * It barely moves the slow rules: error-cost at 90 % goes from 4.84× to 4.10× (mean of ratios).
  * It halves the speed-ups of the fastest ones: MILP-label REINFORCE ranking at 95 % goes from 75× to 33×.
  * kNN rules need no LP relaxation (1 ms of neighbour search).
* **End-to-end pipelines.** Their costs were recomputed and match the stored per-instance arrays exactly. The 24-hour
  end-to-end runs were recomputed from the saved probabilities and reproduce `b3_e2e_results` exactly.
* **New runs.** The new runs of this study were timed under the current load and are not scaled.

## 3. What changes against our earlier conventions (fresh B2 set, first 60 instances)

| rule | earlier: mean gap to reference, all instances / ratio of mean times | gap to DB, feasible | ratio of means + overheads | mean of ratios, no overheads | **paper: mean of ratios + overheads** | gap to DB, proven-reference instances only |
|---|---|---|---|---|---|---|
| error-cost + adequacy guard, 90 % | 0.28 % / 3.16× | 0.40 % | 3.03× | 4.84× | **4.10×** | 0.27 % |
| error-cost + both guards, 95 % | 0.46 % / 4.57× | 0.59 % | 4.28× | 10.6× | **8.04×** | 0.49 % |
| combined pipeline, 98 % | 0.72 % / 6.99× | 0.84 % | 6.35× | 18.8× | **12.6×** | 0.72 % |
| RACLearn-style, 95 % | 49.2 % / 8.15× | 10.4 % | 7.30× | 31.1× | **16.3×** | 10.4 % |
| REINFORCE ranking (label-free, 3 seeds), 95 % | 2.93 % / 10.6× | 2.70 % (95 % feasible) | 17.8× | 63.9× | **28.8×** | 2.16 % |
| REINFORCE ranking (MILP-label model), 95 % | 3.61 % / 30.5× | 3.37 % | 21.3× | 75.1× | **32.5×** | 2.94 % |
| earlier LtF reconstruction, kNN, τ = 1 % | 37.7 % / 2.32× | 10.3 % | 2.32× | 8.28× | **8.17×** | 11.2 % |

The gap barely changes for guarded rules: it rises by about the reference's own gap, 0.12 pp. It shrinks 3–5× for
rules with catastrophic instances, because (C − DB) / C < 100 % per instance. The mean of per-instance ratios is
dominated by the instances on which the full MILP is slow (or hits its 60 s limit) and the reduced problem is fast.
On B2 that makes it 1.3–4× the ratio of mean times; the medians lie close to the ratio of means.

## 4. Side by side

Columns as in the paper's Table I, plus medians and served share. Runtimes in seconds are not comparable across
machines and solvers (§6); compare ratios.

### 4.1 Paper, Table I (kNN; Irish system, copper plate, 72 h; Gurobi 11, 8 logical CPUs; 525 test instances)

| method | feasible % | gap mean % | gap max % | runtime mean s | runtime max s | speed-up mean | speed-up max | fixed mean % | fixed max % |
|---|---|---|---|---|---|---|---|---|---|
| full MILP (0.25 % MIP gap) | 100.00 | 0.19 | 0.25 | 60.35 | 947.21 | 1.00 | 1.00 | 0.00 | 0.00 |
| cost-ranked kNN (k = 50) | 100.00 | 5.40 | 89.99 | 9.86 | 18.91 | 7.13 | 184.73 | 100.00 | 100.00 |
| τ = 0.5 | 11.05 | 78.84 | 99.40 | 0.16 | 1.91 | 421.17 | 8175.31 | 100.00 | 100.00 |
| [0.1, 0.9] | 100.00 | 2.73 | 96.04 | 10.16 | 138.56 | 13.88 | 214.68 | 86.10 | 91.94 |
| [0.05, 0.95] | 100.00 | 0.58 | 46.62 | 20.94 | 215.25 | 6.75 | 91.67 | 81.62 | 88.18 |
| [0.01, 0.99] | 100.00 | 0.21 | 1.13 | 31.25 | 393.99 | 4.06 | 60.59 | 74.42 | 81.35 |
| worst-case thresholds | 99.81 | 0.20 | 0.59 | 52.90 | 858.13 | 2.17 | 52.29 | 20.09 | 40.06 |
| suboptimality-constrained, ε = 10 % | 98.67 | 1.56 | 74.35 | 1.28 | 14.07 | 60.33 | 972.85 | 87.46 | 91.99 |
| suboptimality-constrained, ε = 5 % | 98.86 | 1.05 | 45.45 | 2.51 | 43.76 | 34.25 | 1395.66 | 82.25 | 87.12 |
| **suboptimality-constrained, ε = 1 %** | 99.81 | **0.48** | 6.63 | 4.70 | 96.28 | **20.82** | 240.81 | 78.81 | 84.37 |

### 4.2 B2, 12 hours, fresh test set (fixing: first 60 instances; end-to-end: all 120)

| method | n | feasible % | gap mean % | gap median % | gap max % | runtime mean s | runtime max s | speed-up mean | speed-up median | speed-up max | fixed mean % | fixed max % | served % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full MILP, reference run (0.1 %, 60 s) | 60 | 100.0 | 0.14 | 0.090 | 1.66 | 25.24 | 60.3 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | – |
| full MILP, back to back (this study) | 60 | 100.0 | 0.15 | 0.088 | 2.33 | 21.27 | 60.1 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | 100.0 |
| *paper's baselines, run here* | | | | | | | | | | | | | |
| cost-ranked kNN, k = 50 (best of 50 LPs) | 60 | 100.0 | 12.02 | 3.414 | 83.74 | 4.35 | 7.3 | 5.39 | 3.86 | 20.5 | 100.0 | 100.0 | 78.3 |
| kNN (k = 50), τ = 0.5 | 60 | 86.7 | 63.10 | 88.298 | 99.23 | 0.21 | 0.3 | 99.65 | 53.54 | 326.7 | 100.0 | 100.0 | 19.2 |
| kNN (k = 50), [0.1, 0.9] | 60 | 100.0 | 4.54 | 0.329 | 74.56 | 10.39 | 34.3 | 5.86 | 2.11 | 118.6 | 86.6 | 94.9 | 86.7 |
| kNN (k = 50), [0.05, 0.95] | 60 | 100.0 | 2.65 | 0.125 | 73.35 | 13.76 | 60.1 | 2.60 | 1.64 | 27.6 | 81.2 | 92.5 | 95.0 |
| kNN (k = 50), [0.01, 0.99] | 60 | 100.0 | 0.50 | 0.090 | 19.59 | 16.17 | 60.0 | 2.01 | 1.40 | 16.4 | 70.9 | 88.4 | 96.7 |
| MILP-label BCE GNN, τ = 0.5 | 60 | 65.0 | 24.47 | 7.412 | 92.71 | 0.44 | 0.5 | 40.34 | 25.50 | 130.0 | 100.0 | 100.0 | 23.1 |
| MILP-label BCE GNN, [0.1, 0.9] | 60 | 100.0 | 9.03 | 0.185 | 78.23 | 3.32 | 17.2 | 14.33 | 6.59 | 67.0 | 95.3 | 99.4 | 73.3 |
| MILP-label BCE GNN, [0.05, 0.95] | 60 | 100.0 | 4.10 | 0.118 | 70.89 | 7.25 | 60.4 | 7.90 | 2.95 | 56.4 | 92.6 | 98.5 | 86.7 |
| MILP-label BCE GNN, [0.01, 0.99] | 60 | 100.0 | 0.15 | 0.086 | 1.38 | 12.19 | 60.4 | 2.64 | 1.61 | 14.8 | 79.4 | 90.5 | 96.7 |
| *earlier rules* | | | | | | | | | | | | | |
| earlier LtF reconstruction (kNN, τ = 1 %) | 60 | 100.0 | 10.32 | 0.867 | 85.46 | 12.89 | 40.0 | 8.17 | 2.42 | 104.3 | 89.7 | 95.9 | 75.0 |
| earlier LtF thresholds on REINFORCE GNN, τ = 1 % | 60 | 100.0 | 0.68 | 0.437 | 3.54 | 6.78 | 29.3 | 11.67 | 6.10 | 66.9 | 85.9 | 90.6 | 98.3 |
| RACLearn-style, 90 % | 60 | 100.0 | 3.36 | 0.097 | 78.23 | 10.97 | 33.9 | 5.42 | 2.32 | 80.5 | 90.0 | 90.0 | 91.7 |
| RACLearn-style, 95 % | 60 | 100.0 | 10.35 | 0.177 | 90.25 | 4.07 | 17.8 | 16.26 | 7.18 | 87.1 | 95.0 | 95.0 | 68.3 |
| RACLearn-style + LP guard, 95 % | 60 | 100.0 | 0.58 | 0.114 | 5.56 | 10.23 | 61.0 | 7.00 | 3.54 | 45.5 | 87.0 | 95.0 | 96.7 |
| error-cost + adequacy guard, 90 % (3 seeds) | 60 | 100.0 | 0.40 | 0.099 | 4.67 | 9.84 | 42.1 | 4.10 | 2.83 | 17.8 | 89.9 | 90.0 | 98.9 |
| error-cost + both guards, 95 % (3 seeds) | 60 | 100.0 | 0.59 | 0.158 | 6.17 | 7.01 | 53.7 | 8.04 | 4.22 | 39.1 | 92.8 | 95.0 | 98.3 |
| combined pipeline, 95 % (3 seeds) | 60 | 100.0 | 0.71 | 0.218 | 5.40 | 6.12 | 35.7 | 7.21 | 4.73 | 42.9 | 93.9 | 95.0 | 96.7 |
| **combined pipeline, 98 %** (3 seeds) | 60 | 100.0 | **0.84** | 0.373 | 5.98 | 4.74 | 44.2 | **12.59** | 6.68 | 64.1 | 92.4 | 97.9 | 97.8 |
| REINFORCE ranking, 95 % (label-free, 3 seeds) | 60 | 95.0 | 2.70 | 1.099 | 29.38 | 1.63 | 10.3 | 28.79 | 18.27 | 87.9 | 95.0 | 95.0 | 88.3 |
| REINFORCE ranking, 95 % (MILP-label model) | 60 | 100.0 | 3.37 | 1.446 | 21.60 | 1.40 | 11.4 | 32.53 | 20.86 | 96.8 | 95.0 | 95.0 | 93.3 |
| *end-to-end and LP-only (120 instances; T_MILP = dataset run)* | | | | | | | | | | | | | |
| full MILP, reference run | 120 | 100.0 | 0.30 | 0.096 | 3.17 | 34.32 | 60.3 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | 98.3 |
| cost-ranked kNN, k = 50 | 120 | 100.0 | 10.41 | 4.171 | 94.66 | 4.22 | 7.3 | 8.71 | 9.09 | 20.5 | 100.0 | 100.0 | 81.7 |
| REINFORCE + block repair, 1 LP (3 seeds) | 120 | 100.0 | 6.49 | 3.410 | 38.44 | 0.58 | 0.9 | 58.79 | 60.26 | 153.6 | 100.0 | 100.0 | 95.0 |
| combined, 1 LP (lag_D, θ = 0.6, block repair) | 120 | 100.0 | 4.82 | 1.839 | 77.64 | 0.58 | 0.9 | 58.79 | 60.26 | 153.6 | 100.0 | 100.0 | 92.2 |
| combined + screening (5.8 LPs) | 120 | 100.0 | 3.01 | 1.147 | 40.34 | 1.35 | 1.8 | 25.33 | 26.03 | 75.7 | 100.0 | 100.0 | 93.3 |

### 4.3 B2, 12 hours, original test set (fixing: first 60; end-to-end: 120)

| method | n | feasible % | gap mean % | gap median % | gap max % | runtime mean s | runtime max s | speed-up mean | speed-up median | speed-up max | fixed mean % | fixed max % | served % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full MILP, reference run | 60 | 100.0 | 0.25 | 0.092 | 3.60 | 28.30 | 60.1 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | – |
| full MILP, back to back (combo run) | 60 | 100.0 | 0.24 | 0.092 | 3.60 | 23.83 | 60.1 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | 88.3 |
| earlier LtF reconstruction (kNN, τ = 1 %) | 60 | 100.0 | 4.64 | 0.674 | 57.56 | 11.22 | 37.2 | 8.15 | 2.89 | 103.2 | 89.2 | 95.0 | 78.3 |
| RACLearn-style, 90 % | 60 | 100.0 | 1.27 | 0.098 | 35.77 | 8.48 | 26.4 | 4.60 | 2.34 | 85.0 | 90.0 | 90.0 | 86.7 |
| RACLearn-style, 95 % | 60 | 100.0 | 7.93 | 0.214 | 88.57 | 3.79 | 13.9 | 10.58 | 5.29 | 70.8 | 95.0 | 95.0 | 80.0 |
| RACLearn-style + LP guard, 95 % | 60 | 100.0 | 0.44 | 0.124 | 2.84 | 7.21 | 60.6 | 5.95 | 4.06 | 30.5 | 90.1 | 95.0 | 91.7 |
| error-cost + adequacy guard, 90 % (3 seeds) | 60 | 100.0 | 0.83 | 0.103 | 17.34 | 5.76 | 21.6 | 5.24 | 3.82 | 21.4 | 89.9 | 90.0 | 88.3 |
| **error-cost + both guards, 95 %** (3 seeds) | 60 | 100.0 | **0.63** | 0.198 | 4.46 | 4.17 | 48.7 | **10.39** | 6.63 | 55.4 | 91.9 | 95.0 | 96.7 |
| combined pipeline, 95 % (3 seeds) | 60 | 100.0 | 0.90 | 0.216 | 14.66 | 3.69 | 44.1 | 12.05 | 6.46 | 78.1 | 93.1 | 95.0 | 93.3 |
| combined pipeline, 98 % (3 seeds) | 60 | 100.0 | 1.06 | 0.313 | 11.33 | 2.69 | 31.1 | 18.71 | 10.97 | 113.9 | 91.4 | 97.9 | 93.9 |
| self-trained, asym + adequacy guard, 95 % (3 seeds) | 60 | 100.0 | 1.90 | 0.414 | 28.77 | 2.82 | 13.0 | 19.68 | 7.84 | 137.4 | 94.9 | 95.0 | 87.8 |
| REINFORCE ranking, 95 % (label-free, 3 seeds) | 60 | 96.7 | 2.98 | 0.985 | 20.59 | 1.12 | 6.5 | 31.89 | 20.27 | 142.9 | 95.0 | 95.0 | 87.9 |
| REINFORCE ranking, 95 % (MILP-label model) | 60 | 100.0 | 4.79 | 1.591 | 33.39 | 1.15 | 9.2 | 35.60 | 27.05 | 146.4 | 95.0 | 95.0 | 91.7 |
| full MILP, reference run (120) | 120 | 100.0 | 0.23 | 0.096 | 3.60 | 28.63 | 60.1 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | 92.5 |
| earlier kNN-20 (repaired neighbours, best of 20 LPs) | 120 | 100.0 | 11.72 | 6.789 | 80.35 | 2.29 | 2.6 | 12.68 | 11.71 | 29.0 | 100.0 | 100.0 | 79.2 |
| REINFORCE + block repair, 1 LP (3 seeds) | 120 | 100.0 | 7.58 | 3.265 | 54.71 | 0.44 | 0.6 | 64.61 | 62.32 | 156.5 | 100.0 | 100.0 | 93.6 |
| combined, 1 LP | 120 | 100.0 | 4.28 | 1.573 | 58.23 | 0.44 | 0.6 | 64.61 | 62.32 | 156.5 | 100.0 | 100.0 | 94.4 |
| combined + screening (6.1 LPs) | 120 | 100.0 | 2.87 | 0.912 | 53.01 | 1.06 | 1.3 | 27.24 | 25.95 | 76.0 | 100.0 | 100.0 | 95.8 |

### 4.4 B3, 24 hours (40 test instances; T_MILP = dataset run, 0.1 %, 300 s limit)

| method | feasible % | gap mean % | gap median % | gap max % | runtime mean s | runtime max s | speed-up mean | speed-up median | speed-up max | fixed mean % | fixed max % | served % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full MILP (30 % at the 300 s limit) | 100.0 | 0.16 | 0.099 | 0.77 | 154.31 | 300.2 | 1.00 | 1.00 | 1.0 | 0.0 | 0.0 | 97.5 |
| RACLearn-style, 80 % (no guard) | 100.0 | 0.48 | 0.294 | 3.59 | 45.78 | 122.3 | 4.85 | 2.84 | 37.7 | 80.0 | 80.0 | 92.5 |
| RACLearn-style, 90 % | 100.0 | 9.22 | 0.696 | 83.77 | 28.57 | 75.0 | 9.21 | 5.51 | 60.0 | 90.0 | 90.0 | 67.5 |
| RACLearn-style, 95 % | 100.0 | 23.39 | 2.657 | 95.08 | 10.72 | 59.2 | 38.77 | 13.14 | 184.7 | 95.0 | 95.0 | 40.0 |
| val-selected guarded rule, 80 % | 100.0 | 0.68 | 0.439 | 4.42 | 33.02 | 77.1 | 8.90 | 4.28 | 82.7 | 80.0 | 80.0 | 95.0 |
| val-selected guarded rule, 90 % target | 100.0 | 0.72 | 0.489 | 4.05 | 32.58 | 74.8 | 7.49 | 4.47 | 58.2 | 87.4 | 90.0 | 92.5 |
| **val-selected guarded rule, 95 % target** | 100.0 | **0.67** | 0.459 | 2.32 | 32.40 | 302.3 | **11.55** | 4.08 | 61.7 | 85.4 | 95.0 | 95.0 |
| REINFORCE ranking, 90 % | 100.0 | 3.06 | 1.202 | 16.61 | 15.18 | 55.0 | 36.64 | 9.34 | 190.7 | 90.0 | 90.0 | 92.5 |
| REINFORCE ranking, 95 % | 100.0 | 7.58 | 5.184 | 34.34 | 5.93 | 42.4 | 60.72 | 26.97 | 227.3 | 94.9 | 95.0 | 92.5 |
| REINFORCE ranking + guards, 95 % target | 100.0 | 7.59 | 5.184 | 34.34 | 5.76 | 41.5 | 51.61 | 25.50 | 174.0 | 90.3 | 95.0 | 97.5 |
| end-to-end: REINFORCE + block repair, 1 LP | 100.0 | 11.44 | 9.621 | 37.76 | 1.53 | 2.0 | 100.57 | 60.31 | 236.3 | 100.0 | 100.0 | 87.5 |
| end-to-end: REINFORCE + screening (15 LPs) | 100.0 | 5.84 | 4.411 | 21.26 | 5.69 | 6.2 | 27.01 | 16.81 | 57.1 | 100.0 | 100.0 | 92.5 |

### 4.5 Equal speed-up: best mean gap at a mean speed-up of at least x (paper metrics, feasibility ≥ 95 %)

| benchmark | ≥ 2× | ≥ 5× | ≥ 10× | ≥ 20× | ≥ 50× |
|---|---|---|---|---|---|
| paper, Table I | 0.20 % (2.2×, worst-case) | 0.48 % (20.8×, ε = 1 %) | 0.48 % (ε = 1 %) | 0.48 % (ε = 1 %) | 1.56 % (60.3×, ε = 10 %) |
| B2 fresh | 0.15 % (2.6×, BCE GNN [0.01, 0.99]) | 0.56 % (8.0×, error-cost + both guards) | 0.68 % (11.7×, LtF thresholds on REINFORCE GNN); 0.84 % (12.6×, combined 98 %) | 2.70 % (28.8×, REINFORCE ranking) | 4.82 % (58.8×, end-to-end, 1 LP) |
| B2 original | 0.44 % (5.9×, RACLearn + LP guard) | 0.44 % (5.9×) | 0.63 % (10.4×, error-cost + both guards) | 1.96 % (21.5×, self-trained asym + guard) | 4.28 % (64.6×, end-to-end, 1 LP) |
| B3, 24 h | 0.48 % (4.8×, RACLearn 80 %) | 0.67 % (11.5×, guarded 95 %) | 0.67 % (11.5×) | 3.06 % (36.6×, REINFORCE 90 %) | 7.58 % (60.7×, REINFORCE 95 %) |

Instance-bootstrap 95 % intervals of the mean (JSON) are wide. For the combined pipeline at 98 % on the fresh set the
gap interval is 0.56–1.16 % and the speed-up interval 9.2–16.2×. For the error-cost rule with both guards it is
0.35–0.89 % and 5.9–10.5×.

## 5. The paper's baselines on our benchmark (fresh B2 set)

Run with `scripts/uc_papereval_run.py`, back to back with a fresh full MILP:

* **kNN, k = 50, inverse-distance weights (the paper's eq. 13).**
  * Setup: our system-level features; labels canonicalised inside identical-unit groups.
  * Fixed shares and feasibility match the paper: 86.6 / 81.2 / 70.9 % fixed at [0.1, 0.9] / [0.05, 0.95] /
    [0.01, 0.99] (paper 86.1 / 81.6 / 74.4 %), 100 % feasible.
  * The gap shrinks and the speed-up falls as the grey zone widens, as in the paper. Both are worse here: 4.5 / 2.7 /
    0.50 % at 5.9 / 2.6 / 2.0×, against 2.7 / 0.58 / 0.21 % at 13.9 / 6.8 / 4.1×.
  * The reduced MILPs with 71–87 % fixed still take 10–16 s against 21 s for the full MILP. HiGHS spends most of the
    time in the 12-hour network LP, and our MILP is fast (median 18 s, at most 60 s), which caps the per-instance
    ratios.
* **Hard threshold τ = 0.5.** 86.7 % feasible (paper 11 %), 63 % mean gap, 19 % served. The only infeasibility in our
  model is a min up/down conflict with the initial status, presumably rarer over 12 hours than over 72.
* **Cost-ranked kNN (Pineda & Morales 2022).**
  * 38.8 of the 50 neighbour schedules are min up/down-feasible for the test instance; one LP takes 0.084 s.
  * Mean gap: 12.0 % on the first 60 instances (5.4×) and 10.4 % on all 120 (8.7×); the median is 3.4–4.2 % and
    78–82 % of instances are served. The paper reports 5.4 % at 7.1×.
  * Best of the first k neighbours (120 instances): k = 1 62 % (96.7 % feasible), k = 5 31 %, k = 10 22 %,
    k = 20 15 %, k = 50 10.4 %.
  * Our earlier kNN-20 (min up/down-repaired neighbours, original set) gives 11.7 % at 12.7×.
* **Our GNN with the same constant thresholds** is far better than the kNN at the wide grey zone. [0.01, 0.99]:
  0.15 % at 2.6×, the lowest gap of any rule at ≥ 2×. At the narrow zones it is better on speed and worse on gap:
  [0.1, 0.9] fixes 95 % (9.0 % at 14×).

## 6. What is and is not comparable

| aspect | paper | ours |
|---|---|---|
| system | EPRI competition data, simplified Irish system: 51 thermal units, 15 hydro, 13 batteries, 1 pumped storage, aggregated wind / solar | RTS-GMLC: 73 thermal units (many identical copies), renewables as availability |
| network | copper plate | DC network, 120 branches (B2 and B3) |
| horizon | 72 h | 12 h (B2), 24 h (B3) |
| size | 19,296 binaries per instance | 2,628 binaries (876 commitment decisions) in B2; 5,256 (1,752) in B3 |
| soft constraints | the paper attributes the infeasibilities of its tuned rules to min up/down violations; balance and reserve violations are presumably penalised (UnitCommitment.jl's formulation has penalty slacks) | shedding / over-generation at VOLL, priced reserve shortfall; infeasible only through min up/down conflicts. Same notion of feasibility |
| solver | Gurobi 11, 8 logical CPUs (AMD EPYC node) | HiGHS (SciPy for B2, highspy 1.15 for B3), 1 thread per solve, 2 solves in parallel on a shared 4-core VM (load 4–8) |
| MILP tolerance | 0.25 % | 0.1 % with a 60 s (B2) / 300 s (B3) limit. A looser tolerance would shorten T_MILP and reduce our speed-ups; the time limit truncates T_MILP on 8–20 % (B2) and 30 % (B3) of instances and does the opposite |
| full-MILP time | 60 s mean, 947 s max | B2: 21–30 s mean, median 18 s, max 60 s; B3: 154 s mean, median 98 s, max 300 s |
| dual bound | the reference's own gap is 0.19 % mean, 0.25 % max | 0.14–0.25 % mean (comparable), but 1.7–3.6 % max on the instances the MILP did not prove. On proven-reference instances the guarded rules' gaps are 0.1–0.5 pp lower, e.g. 0.40 → 0.27 % (JSON `gap_mean_proven`). Part of that difference is instance selection: the unproven instances are the hard ones |
| data | 2,621 instances, 60/20/20 split: ~1,570 train, ~525 validation, 525 test | B2: 500 train, 60 validation, 120 test (fixing on 60) per test set; B3: 300 train, 30 validation (no MILP), 40 test |
| decision rules | thresholds tuned on ~525 validation instances (logic-based Benders) | guards are parameter-free; targets and probability sources chosen on 15–60 validation instances |
| classifier | kNN (k = 50); CatBoost in the appendix | GNNs with LP-relaxation features (MILP-label BCE, self-trained, REINFORCE, Lagrangian); kNN only as a baseline |
| timing | dedicated node | shared machine; only back-to-back ratios are meaningful |

The speed-up statistic is the least comparable quantity. A mean of per-instance ratios grows with the tail of the
full-MILP time distribution (the paper's goes to 947 s; ours is truncated at 60 or 300 s). With 60 instances,
one or two slow instances move it by 10–30 % (bootstrap intervals in the JSON).

## 7. Assessment

1. **Absolute numbers.** Under the paper's own metrics, our best guarded rules reach 0.6–0.8 % mean gap at 8–13×
   mean speed-up on all three of our test sets:
   * fresh B2: 0.59 % at 8.0×, 0.84 % at 12.6×;
   * original B2: 0.63 % at 10.4×;
   * B3: 0.67 % at 11.5×;
   * 95–98 % served, 100 % feasible.

   Learning to Fix reports 0.48 % at 20.8× (99.81 % feasible). We do not reach that point. Beyond 20× our rules cost
   2–3 % mean gap on B2 and 3–8 % on B3.
2. **Relative to the paper's baselines on the same benchmark**, the picture is reversed.
   * On B2 the paper's own baselines are 1.7–4.6× worse in gap than in the paper and, except the cost-ranked kNN on
     all 120 instances, 1.3–2.6× slower. Our system, network and short MILP times make fixing harder and limit the
     attainable speed-up.
   * Against those baselines on the same instances our rules gain about as much as Learning to Fix does in the
     paper, or more:
     * at the gap of kNN [0.01, 0.99], about 4× its speed-up (paper: ε = 1 % is 3× faster than [0.05, 0.95] at a
       similar gap);
     * at the speed of cost-ranked kNN, a 20× lower mean gap.

   Whether the full Learning to Fix method (threshold tuning with joint validation cuts) would do better on B2 is
   the subject of the separate faithful implementation (`ltfx` study). Our earlier reconstruction (10.3 % at 8.2×
   under these metrics) is superseded and is not evidence either way.
3. **Our earlier conventions were conservative.**
   * Ratio-of-means speed-ups understate the paper-style figure by 1.3–4×.
   * Gaps to the reference objective omit the reference's own 0.1–0.25 % gap.
   * Gaps to the reference over all instances let a few penalty-priced instances dominate: up to 500 % per instance,
     against < 100 % under (C − DB) / C.

   Conversely, the paper's gap hides catastrophic schedules: a schedule that sheds load counts at most ~100 %.
   Served share and median should be reported next to it.
4. **The runtime audit matters for fast rules only.** The untimed LP relaxation (0.3–0.4 s) is negligible against
   10–40 s reduced MILPs. It halves the per-instance speed-up of REINFORCE-ranked fixing and is the largest part of the
   one-LP end-to-end runtime.

## 8. Files

* `scripts/uc_papereval_run.py` produces `results/papereval_fresh_runs.jsonl` (660 records: 60 back-to-back full
  MILPs, 8 constant-threshold reduced problems per instance, cost-ranked kNN on 120 instances).
* `scripts/uc_papereval_overhead.py` produces `results/papereval_overhead.json` (per-instance overheads, load
  calibration, recomputed end-to-end costs and times; all checks exact).
* `scripts/uc_papereval_score.py` produces `results/papereval_results.json` (every row with all statistics) and the
  tables above.
