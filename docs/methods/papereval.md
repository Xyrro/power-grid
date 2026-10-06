# Our results under the evaluation protocol of Learning to Fix

*Code: [`scripts/uc_papereval_run.py`](../../scripts/uc_papereval_run.py) (missing Table I rows on the fresh 12-hour
test set), [`scripts/uc_papereval_overhead.py`](../../scripts/uc_papereval_overhead.py) (untimed overheads, load
calibration, end-to-end recomputation on 24 h), [`scripts/uc_papereval_score.py`](../../scripts/uc_papereval_score.py)
(re-scoring of every per-instance record). Results: [`results/papereval_results.md`](../../results/papereval_results.md)
(+ `.json`), raw records `results/papereval_fresh_runs.jsonl`, overheads `results/papereval_overhead.json`.*

## 1. Why

Our earlier studies (§5–§6 of [`RESEARCH.md`](../RESEARCH.md)) reported the gap to the reference MILP objective over
all instances (fallback re-solves included), and the speed-up as a ratio of mean times without inference. Learning to
Fix (Fritz, Makrides, Fetanat & Pinson, arXiv 2609.39396; "the paper") uses different definitions (its Sec. IV-C). To
put our numbers next to its Table I, every per-instance record we have was re-scored with the paper's definitions, the
untimed parts of our pipelines were measured and added, and the paper's baselines that we had not run (cost-ranked
kNN, constant confidence thresholds with a kNN classifier, hard thresholding) were run on the fresh 12-hour test set.

## 2. The paper's protocol and how it is applied here

| paper (Sec. IV-C) | here |
|---|---|
| optimality gap (C_m − DB) / C_m, DB = best dual bound of the full MILP, the same for every method | uc12: DB = obj · (1 − mip_gap) of the dataset's reference MILP run (scipy/HiGHS define mip_gap = (primal − dual) / primal; checked on uc24, where the bound is stored: identical to 1e-16); uc24: the stored bound. C_m is the method's objective with its penalty costs |
| speed-up T_MILP / T_m per instance; mean and max reported | the same, plus the median. T_MILP: the full MILP solved back to back in the same worker (fixing runs), or the dataset's MILP run (end-to-end rows, uc24, instances without a back-to-back solve) |
| T_m = total runtime incl. inference and downstream optimisation | recorded solver time + guard LPs (already timed) + the untimed parts measured by `uc_papereval_overhead.py` (§3) |
| feasibility rate reported separately; gap, runtime, speed-up, fixed share over feasible instances only | feasible = the reduced problem returned a solution. A fallback re-solve (fixings in conflict with min up/down, so the reduced MILP is infeasible and the full MILP is solved instead) counts as infeasible for the method |
| — | added: served share (no shedding, over-generation or reserve shortfall in the method's solution) and medians, because our shedding / reserve shortfall are soft and penalty-priced |

Seeded configurations (3 training seeds) are scored per seed and the statistics averaged.

## 3. Runtime audit: what the earlier evaluations timed

Read from `scripts/uc_combo_eval.py`, `uc_ltf_eval.py`, `uc_fixpolicy_eval.py`, `uc_b3_fix.py`, `otsl/fixpolicy.py`:

| component | uc12 fixing runs | uc24 fixing runs | end-to-end |
|---|---|---|---|
| reduced MILP (+ fallback re-solve) | timed | timed | – |
| LP-relaxation guard LPs | timed (`lp_guard_s`, incl. building the bounds) | timed (`t_guard`) | – |
| min up/down row release | **not timed** | timed (in `t_guard`) | – |
| LP relaxation used as GNN / error-cost input | **not timed** (precomputed in the dataset) | timed (`t_rel` added) | **not timed** |
| GNN forward pass | **not timed** (batched before the workers) | not timed (0.004 s batched) | not timed |
| error-cost features + harm-ensemble scoring | **not timed** | – | – |
| ranking, adequacy guard | **not timed** | **not timed** | – |
| block / min up/down repair, dispatch LPs | – | – | **not timed** (no times stored) |

The missing parts were measured per instance with `scripts/uc_papereval_overhead.py`: one instance at a time in a
single process (as in deployment, no batching), on all 120 instances of each 12-hour test set and all 40 uc24 test
instances. The earlier runs were timed on a busier machine (load 6–8 on 4 cores), so the measurements are scaled to
those conditions by a **load factor**: the LP-relaxation guard of RACLearn + LP guard (95 % target) was recomputed
on the first 20 instances of each 12-hour set (same fixings, same LPs; the guard outcome must match the record) and the
median ratio recorded / now is the factor; for uc24 the LP relaxation was re-timed on 10 instances against the stored
`t_rel`. The new runs of this study (`uc_papereval_run.py`) were timed under the current, lighter load (load average
4–5), like the overhead measurement, and are not scaled.

## 4. New runs (fresh 12-hour test set)

`scripts/uc_papereval_run.py`, first 60 `test_fresh` instances, 2 spawn workers, each instance back to back: full MILP
(60 s, 0.1 %), then reduced MILPs for

* **kNN, k = 50, inverse-distance weights** (the paper's classifier and eq. 13) on the system-level features of our
  reconstruction (`otsl.ltf.KNNCommit(kind="sys")`: per-hour area loads, renewable availability by type, initial
  status), labels canonicalised inside groups of identical units; constant thresholds [0.1, 0.9], [0.05, 0.95],
  [0.01, 0.99] and the hard threshold 0.5 (all fixed);
* the same thresholds on the **MILP-label BCE GNN** (RACLearn's predictor);
* the **cost-ranked kNN** (Pineda & Morales 2022; the paper's "cost-ranked" row): the 50 nearest training instances
  by the same distance, each one's MILP schedule fixed and priced by the dispatch LP of the test instance (a schedule
  that violates min up/down given the instance's initial status makes the LP infeasible and is discarded), the
  cheapest kept; runtime = kNN search + all 50 LPs. Run on all 120 instances (instances 60–119 without a back-to-back
  MILP; scored against the dataset's MILP time).

No fallback re-solve in these runs: a reduced problem without solution is infeasible.

## 5. Checks

* All recorded costs re-read under the old conventions reproduce the published tables exactly: RACLearn at 90 % is
  12.43 % / 2.82×, and error-cost at 90 % is 0.28 % / 3.16×.
* The end-to-end costs recomputed for timing match the stored per-instance arrays exactly. The 24-hour end-to-end
  costs, recomputed from the saved probabilities and the original random-number order, reproduce
  `b3_e2e_results.json` exactly (87.5 % / 10.34 % and 92.5 % / 4.28 %).
* The LP relaxation reproduces the stored `c_rel` exactly.
* The load calibration recomputed the LP guard on 2 × 20 instances; all 40 had identical released counts.

## 6. Results and verdict

Full tables: [`results/papereval_results.md`](../../results/papereval_results.md). In short:

* **Speed-ups.** The paper's metrics raise our speed-ups. The mean of per-instance ratios is 1.3–4× our ratio of mean
  times; adding the untimed overheads (mainly the 0.3–0.4 s LP relaxation) takes back part of that, most for the
  fastest rules.
* **Gaps.** The DB-based gap adds the reference's own 0.1–0.25 % to guarded rules. It compresses heavy tails, because
  each instance stays below 100 %.
* **Best guarded rules, paper metrics.** 0.59 % at 8.0× and 0.84 % at 12.6× (fresh B2), 0.63 % at 10.4× (original
  B2), 0.67 % at 11.5× (24 h), with 100 % feasibility and 95–98 % served. The paper reports 0.48 % at 20.8×. Beyond
  20× our rules cost 2–8 % mean gap.
* **The paper's baselines on fresh B2.**
  * kNN, k = 50, with constant thresholds: fixed shares as in the paper, but gaps 1.7–4.6× larger and speed-ups
    2–2.6× smaller.
  * Cost-ranked kNN: 10–12 % mean gap at 5–9×; the paper reports 5.4 % at 7.1×.
  * Against these baselines our rules improve by at least the margin Learning to Fix shows over them in the paper.

## 7. Limitations

* **Dual bound.** On 8–20 % of the 12-hour instances the reference MILP stops at its 60 s limit, so its bound is
  weaker than the paper's (whose full MILP closes to 0.25 %). The report gives gaps restricted to proven instances
  as well.
* **Load scaling.** The overhead scaling uses one factor per set, from the LP-guard timings of the combo run. The
  older runs (fixpolicy, ltf) had similar but not identical loads.
* **Old fixing runs.** None was repeated. Their solver times are used as recorded, under a 0.1 % gap and 60 s limit,
  not the paper's 0.25 %.
* **Sample sizes.** 60 fixing instances per 12-hour set and 40 instances on 24 h; bootstrap intervals are in the JSON.
* **Constant-threshold and cost-ranked runs.** These run on the fresh set only, with one kNN configuration (our
  features, canonical labels for the probabilities, raw neighbour schedules for the cost ranking).

## 8. Reproduce

```bash
python3 scripts/uc_papereval_run.py --n_fix 60 --n_rank 120 --workers 2    # ~50 min with 2 workers
python3 scripts/uc_papereval_overhead.py                                   # ~7 min, one process
python3 scripts/uc_papereval_score.py                                      # seconds; writes results/papereval_results.json
```
