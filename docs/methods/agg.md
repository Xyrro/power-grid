# Round 8: identical units aggregated, Lagrangian probing, and certified pipelines built on them

Code: [`otsl/agg.py`](../../otsl/agg.py) (aggregated screened model, count fixings, the aggregated certified pipeline,
the sandwich certificate), [`otsl/probe.py`](../../otsl/probe.py) (Lagrangian probing),
[`scripts/agg_probe.py`](../../scripts/agg_probe.py) (exploration), [`scripts/agg_eval.py`](../../scripts/agg_eval.py),
[`scripts/agg_run.py`](../../scripts/agg_run.py), [`scripts/pace/pgu_agg.sbatch`](../../scripts/pace/pgu_agg.sbatch),
[`scripts/pace/pgu_agg_probe.sbatch`](../../scripts/pace/pgu_agg_probe.sbatch), [`scripts/agg_report.py`](../../scripts/agg_report.py);
PGLib-UC: [`scripts/agg_pglib_eval.py`](../../scripts/agg_pglib_eval.py), [`scripts/agg_pglib_base.py`](../../scripts/agg_pglib_base.py),
[`scripts/agg_pglib_report.py`](../../scripts/agg_pglib_report.py), `scripts/pace/pgu_pglib{,_eval,_base}.sbatch`; round 9:
[`scripts/r9_margins.py`](../../scripts/r9_margins.py), `scripts/uc_m1x_train.py --margin`, `scripts/pace/pgu_r9.sbatch`;
tests [`tests/test_agg.py`](../../tests/test_agg.py).
Results: `results/agg/` (exploration `explore/e1`–`e4`, validation `val/`, `val2/` and round 9's `val3/`, choices
`select.json` and `select9.json`, test `test/`, report `report.{md,json}`; PGLib `pglib/` (30 perturbed test instances,
with Learning to Fix's re-tuned thresholds `pglib_ltf_knn_1_retuned.json`) and `pglib_base/` (the 20 instances as
distributed), each with its `report.md`); round 9's network metrics `results/uc12/m1x_train_r9.json`.

## Status

* Exploration (2026-11-17, validation instances only): local probes on an Apple M4, then four batches on PACE ICE over
  the first 120 validation instances (`results/agg/explore/e1`–`e4`).
* Validation: **done** (2026-11-17, 360 instances in two passes, PACE ICE, none failed). Choices: `acert:ens5+x:0.05`
  (learned) and `acert:lag0.001+x:0.4` (learning-free), `results/agg/select.json`.
* Test: **done** (2026-11-17, 400 instances read once, four 4-core jobs pinned to Xeon Gold 6226 nodes, HiGHS 1.15.1,
  none failed). Report: `python3 scripts/agg_report.py` → `results/agg/report.{md,json}`.
* PGLib-UC California: **done** (30 perturbed test instances with Learning to Fix re-tuned; the 20 instances as
  distributed). Round 9 (probing margins as a network input): validated, **not adopted** (no faster).

## Summary

**Three exact ideas, no new model, and a certified pipeline that is 7× faster than Learning to Fix and 5× more accurate.**
(1) Identical units become integer counts: the MILP loses its symmetry, exactly. (2) Lagrangian probing: at the LP
relaxation's prices each unit's DP gives the exact cost of flipping any of its decisions; decisions that no cheaper
schedule can change are fixed in the proof step, and the same margins rank decisions for fixing without any learning.
(3) Round 7's certificate keeps every answer proven within 0.5 %.

* **12 h test (120): 0.103 % (max 0.47 %) at 17.5×**, every answer proven; against **Learning to Fix as published
  (same process): 7.22× [5.69, 9.13] faster and 0.557 pp more accurate [−0.765, −0.360]** (0.660 %, max 5.85 %, 2.4×);
  against round 7's certified pipeline 1.36× [1.25, 1.47] faster at the same accuracy; against the full MILP at 0.5 %,
  8.31× faster and 0.095 pp more accurate.
* **Without learning: 0.102 % at 13.0× on 12 h, 13.7× on 24 h, 6.4× on the congested lines**, 1.5–2.2× faster than round
  7's learning-free pipeline everywhere and 5.38× faster than Learning to Fix on 12 h.
* **Exact, no fixing at all: the aggregated MILP is 3.1× faster at 0.1 % (more accurate) and 5.3× at 0.5 %** on 12 h
  (1.43× over round 7's screened MILP; 1.54–1.58× on 24 h and the congested lines, where round 7's screen alone gave
  nothing).
* **Robust**: 0.07–0.12 % on all six shifted sets at 8.7–15.5×; Learning to Fix there averages 5.7 % (25.9 % under line
  outages).
* **PGLib-UC California (public, 610 units)**: the certified pipeline on the LP values (no learning) beats Learning to Fix
  in its own regime by the geometric mean (1.69× [1.07, 2.57]) and the ratio of mean times (11.5× vs 6.3×), with 0.13 %
  against 3.9 % and no failure; on the 20 instances as distributed (48 h) it is 11.0× faster than the full MILP (20.8×
  by the mean of ratios) at 0.082 %, proven within 0.5 %.
* Negative results, all on validation: the sandwich certificate, count fixings, a ladder of reduced MILPs, the reduced
  MILP at 0.1 %, cheapest-first rounding on RTS-GMLC, an economic veto or override of the network, solver heuristics
  off, and round 9's networks with the margins as input (better calibrated, not faster).


## Plan, fixed before the validation pass

**Methods on validation** (the 360 instances of round 7, every method of an instance in one single-threaded process):
the full MILP at 0.1 % (speed-up reference), the aggregated screened full MILP at 0.5 %, round 7's chosen certified
pipeline (`cert:s0:0.02`), and the aggregated certified pipelines `acert:<pred>[+x][+r]:<delta>` of
`scripts/agg_eval.py --acert` (default grid): learned predictors s0 (round 7's network) and ens5 (its five networks)
and kNN (k = 50, Learning to Fix's predictor); learning-free predictors lp (LP values) and lag (Lagrangian probing
margins); +x exact probing fixings in the proof step; +r the reduced MILP at 0.1 %.

**Choices.** Every certified pipeline proves its answer within 0.5 % (unless the proof step hits its time limit), so
speed decides: (1) the learned pipeline is the learned variant with the highest geometric-mean speed-up over the full
MILP at 0.1 % on the 360 instances, the single network preferred if within 5 % of the best; (2) the learning-free
pipeline is the lag / lp variant with the highest geometric-mean speed-up. Selection: `scripts/agg_report.py --select`.

**Test, read once** (round 7's 400 instances): the full MILP at 0.1 % and 0.5 %, round 7's screened MILP and its two
certified pipelines, the aggregated full MILP at 0.1 % and 0.5 %, Learning to Fix as published (kNN, ε = 1 %, step 2's
tuning; 12 h only, the other benchmarks have no labelled training set here), the two chosen pipelines and the
aggregated certified pipeline without probing as ablation. Two operating points come from the same records without
re-solving: the reduced schedule alone (confidence fixing without a certificate, RACLearn's rule) and an LP certificate
at 1 %.

## Method

Round 7's certified pipeline spends most of its time proving: on 60 % of the instances that need the proof step the LP
relaxation is more than 0.5 % below the optimum, so no fixing rule can avoid it, and that proof is a branch-and-bound
run whose root cut loop dominates (HiGHS ends at the root node on most instances). Three ideas make the proof and the
reduced MILP cheaper without approximating anything, and one makes the learned part unnecessary for most of the speed.

1. **Identical units as counts (exact).** 56 of RTS-GMLC's 73 thermal units form 24 groups of exact copies (same bus,
   costs, limits, minimum times). Their schedules can be permuted at no cost, so branch-and-bound must close every
   permuted copy of a subtree, and a learned model trained on one of the tied optima sees arbitrary labels. Each group
   becomes one integer count with the group's summed rows (Knueven, Ostrowski & Watson 2018): a relaxation by
   construction (valid bounds), and exact because every integer solution splits back into member schedules of the same
   cost (stop the units on longest, start those off longest). Units out of service leave their group. Per-unit fixings
   become bounds on counts. The screened network of round 7 is kept (the copies share a bus, so their PTDF columns are
   equal).
2. **Lagrangian probing (exact).** At the LP relaxation's prices (bus prices and the reserve price, both read from the
   relaxation already solved) each unit is its own small problem; a forward-backward DP over its states gives every
   unit-hour's *margin*, the extra cost of the cheapest schedule in the other state. Any schedule that flips (t, g) costs
   at least z_LP + margin(t, g), so given a schedule of cost C, every decision with margin > C - z_LP holds in every
   cheaper schedule and can be fixed in the proof step. Unlike a single variable's LP reduced cost, the margin prices the
   whole minimum-up or -down block a flip forces. Checked: each unit's DP optimum equals its share of the LP relaxation to
   1e-6 (the unit's LP is integral), and no probed fixing ever contradicts the MILP optimum.
3. **The margins as a ranking (no learning).** The same margins rank decisions by economic robustness: fixing every
   decision whose margin is at least tau x z_LP is a learning-free fixing rule.

## Exploration (validation instances only)

**Where round 7's time goes.** On the 120 test instances where round 7's pipeline needed the proof step, the LP
relaxation's bound is more than 0.5 % below the optimum on 60 % (so the proof is unavoidable whatever the fixings), and
on the other 40 % the reduced schedule was not good enough for the LP bound. The relaxation's gap is entirely no-load
cost: it commits fractions of units (17–35 fractional unit-hours) where the integer optimum must commit 3–12 more whole
units, steam units carrying 8–24 h minimum up times. Ramping never binds on RTS-GMLC and minimum up/down times are in
their tight form, so no reformulation of a single unit raises the bound; only branching (or HiGHS's root cuts, which
close most of it at the root node) does. HiGHS 1.15 accepts a lazy-constraint callback but never calls it, so line
limits are still added between solves.

**Two batches on PACE (first 120 validation instances, every method of an instance in one process;
`results/agg/explore/e1`, `e2`; `scripts/agg_probe_report.py`).** Speed-up: geometric mean over the full MILP at 0.1 %.

| method | gap mean | max | speed-up | proven by the LP bound | vs aggregated certified (GNN) |
|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 0.078 % | 1.21 % | 1.0× | – | |
| round 7: screened full MILP, 0.5 % | 0.103 % | 0.50 % | 3.5× | – | |
| aggregated screened full MILP, 0.5 % (no learning) | 0.092 % | 0.79 % | 5.2× | – | |
| aggregated screened full MILP, 0.1 % (no learning) | 0.062 % | 0.43 % | 3.1× | – | |
| round 7: certified pipeline (s0, δ = 0.02) | 0.106 % | 0.50 % | 11.1× | 65 of 120 | 0.80× |
| **aggregated certified pipeline (s0, δ = 0.02)** | 0.116 % | 1.38 % | **14.1×** | 65 of 120 | 1.00× |
| … five networks, δ = 0.05 | 0.109 % | 0.50 % | 14.7× | 64 | 1.05× [0.97, 1.12] |
| … + exact probing in the proof step | 0.126 % | 0.50 % | 14.7× | 65 | 1.05× [0.98, 1.11] |
| … solver heuristics off in the proof step | 0.147 % | 0.92 % | 14.3× | 65 | 1.02× [0.96, 1.08] |
| … economic veto (drop fixings the margins contradict) | 0.136 % | 1.45 % | 13.7× | 65 | 0.97× |
| … count fixings (round(sum p) per group) | 0.112 % | 0.50 % | 12.6× | 60 | 0.90× [0.81, 1.00] |
| sandwich certificate (relax the confident decisions for a bound) | 0.118 % | 1.38 % | 11.5× | 65 (+18 by the relaxation) | 0.82× [0.76, 0.88] |
| **probing ranker, τ = 0.1 % (no learning)** | 0.129 % | 1.05 % | **12.4×** | 66 | 0.85× [0.76, 0.93] |
| probing ranker, τ = 0.03 % / 0.3 % | 0.142 / 0.121 % | | 11.4× / 8.0× | | |
| LP values (no learning, round 7's rule) | 0.114 % | 1.05 % | 8.0× | 27 | 0.56× |
| kNN (Learning to Fix's predictor) | 0.101 % | 0.50 % | 8.0× | 63 | 0.57× |

* **Aggregation is the gain**: 1.47× for the exact MILP and 1.25× for the certified pipeline over round 7, with the same
  proofs (validation estimate; the test below is the measurement).
* **The probing ranker needs no training and beats round 7's learned pipeline** (12.4× against 11.1×) and the LP-value
  rule by 1.5×: the margins know which LP-integral decisions are robust, the LP values do not.
* Negative: the sandwich certificate (its relaxation leaves the confident decisions fractional and closes only part of the
  gap, at the price of a solve), count fixings (few extra fixings, some wrong), the economic veto (it never fires where it
  matters), heuristics off in the proof step (no faster, and it keeps worse schedules). Exact probing in the proof step
  helps a little (1.05×, not significant on 120 instances); it is in the validation grid.

**Third and fourth batches (`results/agg/explore/e3`, `e4`).** A *ladder* (when the LP bound does not prove the first
reduced schedule, a second reduced MILP with fewer fixings before the proof) proves only 3 more answers of 116 and is
slower (0.92–0.95× [0.88, 0.99]). Overriding the network with the probing schedule wherever the margin is large adds
fixings that are more often wrong (τ = 0.03 % / 0.1 % / 0.3 % of z_LP: 0.79× / 0.86× / 1.04× [0.97, 1.12]).

**Cheapest candidates first (`+f`).** Round the probabilities and the LP values (0.5, 0.2), repair block adequacy and
min up/down, price each with the dispatch LP, and deliver the cheapest when the LP bound proves it within 0.2 % or
0.5 %. On RTS-GMLC the best rounding is proven within 0.5 % on 42 % of 60 validation instances (0.14 s), but the four
dispatch LPs cost more on every instance than the reduced MILP they sometimes replace: it was tested in a second
validation pass (below) and loses. On PGLib-UC, where the reduced MILP of a 610-unit model takes 5–6 s and one
dispatch LP 0.7 s, the balance is the other way (section PGLib).

## Validation (360 instances, PACE ICE)

Two passes over the 360 validation instances of round 7 (`results/agg/val`: the grid of the plan; `results/agg/val2`:
the cheapest-first variants, added after the exploration, with the anchor `acert:s0+x:0.02` solved again, 17.3× there
against 17.5× in the first pass); every method's speed-up is paired with the full MILP solved in its own process; no
instance failed. Selection: `scripts/agg_report.py --select` → `results/agg/select.json`.

| method | gap mean | max | served | time s | speed-up | proven by the LP bound |
|---|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 0.105 % | 2.83 % | 96.9 % | 22.18 | 1.0× | – |
| aggregated screened full MILP, 0.5 % (no learning) | 0.114 % | 0.79 % | 97.2 % | 5.80 | 5.7× | – |
| round 7: certified pipeline (s0, δ = 0.02) | 0.130 % | 0.50 % | 95.0 % | 4.53 | 13.1× | 214 |
| aggregated certified, s0, δ = 0.02 | 0.123 % | 1.28 % | 96.1 % | 3.13 | 16.7× | 216 |
| … + exact probing in the proof step | 0.121 % | 0.50 % | 96.4 % | 3.00 | 17.5× | 216 |
| … + reduced MILP at 0.1 % | 0.105 % | 0.50 % | 96.9 % | 3.77 | 13.3× | 225 |
| … + cheapest candidates first, 0.2 % / 0.5 % (second pass) | 0.121 / 0.133 % | 0.50 % | 96.7 / 96.9 % | 3.28 / 3.25 | 11.3× / 11.7× | 216 / 218 |
| aggregated certified, five networks, δ = 0.05 | 0.127 % | 0.50 % | 94.2 % | 2.88 | 17.5× | 207 |
| **… + exact probing (learned choice)** | **0.124 %** | 1.01 % | 95.0 % | **2.60** | **18.6×** | 207 |
| … LP certificate at 1 % (proven within 1 %) | 0.137 % | 1.01 % | 94.2 % | 2.41 | 21.4× | |
| … without the certificate | 1.723 % | 78.8 % | 87.5 % | 0.73 | 32.8× | |
| aggregated certified, kNN + probing | 0.123 % | 0.50 % | 96.1 % | 4.28 | 9.7× | 195 |
| aggregated certified, LP values + probing (no learning) | 0.129 % | 0.63 % | 96.7 % | 3.82 | 10.1× | 89 |
| probing ranker τ = 0.01 % / 0.03 % + probing (no learning) | 0.136 / 0.138 % | 0.63 % | 96.4 / 96.1 % | 3.79 / 3.41 | 11.3× / 13.3× | 118 / 162 |
| **probing ranker τ = 0.1 % + probing (learning-free choice)** | **0.119 %** | 0.63 % | 96.1 % | 3.02 | **14.3×** | 213 |
| … LP certificate at 1 % | 0.129 % | 0.74 % | 95.8 % | 2.83 | 15.8× | |
| … without the certificate | 0.579 % | 16.3 % | 95.6 % | 0.96 | 23.8× | |

* Choices (rule fixed beforehand): learned **`acert:ens5+x:0.05`** (the single network with probing reaches 17.5×,
  94 % of the best, just outside the 5 % margin that would have preferred it); learning-free **`acert:lag0.001+x:0.4`**.
* Exact probing in the proof step is a small, consistent gain (1.04–1.06× in every comparison); the reduced MILP at
  0.1 % proves more answers with the LP bound but costs more than it saves; cheapest-first loses on RTS-GMLC (above).
* The learning-free pipeline (14.3×) is faster than round 7's learned pipeline (13.1×) on the same instances; the
  ensemble is 1.30× faster again, so learning still buys speed here, as in round 7.
* Without the certificate the ensemble's fixings fail badly (max 78.8 %, 87.5 % served), the probing ranker's much less
  (max 16.3 %, 95.6 % served): the margins rarely fix a needed unit off.

## PGLib-UC California: a public system, against Learning to Fix in its own regime

[`scripts/agg_pglib_eval.py`](../../scripts/agg_pglib_eval.py), [`scripts/agg_pglib_report.py`](../../scripts/agg_pglib_report.py),
[`scripts/pace/pgu_pglib.sbatch`](../../scripts/pace/pgu_pglib.sbatch) (data, models and Learning to Fix's tuning,
regenerated on PACE with the commands and seeds of [`pglib.md`](pglib.md) section 9: the same perturbed 24-hour
instances), `results/agg/pglib/report.md`. 610 units (410 free), copper plate, start-up categories, ramping; the full
MILP takes 68 s on average here. Learning to Fix as published: kNN (k = 50) on the self-training labelled set,
per-generator thresholds tuned by its Benders procedure at ε = 1 % on the 30 validation instances (converged: all 30
pass, 86.2 % fixed; the earlier study's tuning fixed 86.75 %), reduced MILP at 0.1 %. Nothing of round 8 was tuned on
PGLib: the certified pipelines use the thresholds of RTS-GMLC and every variant run is reported. Aggregation is not
used (the PGLib formulation's start-up categories and initial conditions would need their own disaggregation), and the
learned models of `pglib.md` fix needed units off (their reduced schedules shed load), so the LP values are the main
predictor here.

**30 test instances, every method of an instance in one process** (gap to the tightest bound of the instance's runs):

| method | gap mean | max | served | speed-up geo / median / mean of ratios / ratio of means | proven by the LP bound |
|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 0.029 % | 0.09 % | 100 % | 1.0× | – |
| full MILP, 0.5 % | 0.046 % | 0.22 % | 100 % | 1.2× / 1.0× / 1.5× / 1.3× | – |
| Learning to Fix (kNN, ε = 1 %, as published) | 3.863 % | 99.9 % | 97 % | 9.4× / 5.9× / **25.5×** / 6.3× | – |
| **certified, LP values, rounding first, proven within 0.5 %** | **0.133 %** | **0.41 %** | **100 %** | **15.8× / 16.5× / 22.6× / 11.5×** | 28 (26 at the rounding) |
| certified, LP values, rounding first, proven within 0.2 % | 0.077 % | 0.19 % | 100 % | 8.0× / 13.7× / 13.4× / 4.2× | 22 (18) |
| certified, LP values, rounding first, proven within 0.1 % | 0.040 % | 0.09 % | 100 % | 3.9× / 4.5× / 8.2× / 1.4× | 17 (12) |
| certified, LP values, proven within 0.5 % (no rounding step) | 0.095 % | 0.26 % | 100 % | 8.4× / 9.8× / 10.1× / 6.2× | 26 |
| certified, self-trained model, rounding first, 0.5 % | 0.145 % | 0.46 % | 100 % | 11.3× / 13.6× / 16.6× / 10.4× | 30 (18) |
| certified, kNN (Learning to Fix's predictor), 0.5 % | 0.060 % | 0.48 % | 100 % | 2.6× / 2.2× / 3.6× / 3.0× | 27 |

* **Learning to Fix reproduces its published regime** (mean of per-instance speed-ups 25.5×, the earlier study's figure
  exactly) and **fails on one instance** (a needed unit fixed off: the schedule sheds load, 99.9 % above the optimum).
* **The learning-free certified pipeline is faster by the geometric mean (1.69× [1.07, 2.57]), the median and the ratio
  of mean times (11.5× against 6.3×), 3.7 pp more accurate on average [−10.6, −0.2], never fails, and proves every
  answer within 0.5 %** (actual mean 0.13 %). By the paper's own metric, the mean of per-instance ratios, Learning to Fix
  stays slightly ahead (25.5× against 22.6×), carried by a few instances where it is very fast.
* **The same guarantee as the reference** (proven within 0.1 %, 0.040 % on average) still comes 3.9× faster (8.2× by
  the mean of ratios).
* Here the cheapest candidates come first: the LP relaxation is within 0.03 % of the optimum at the median, the
  rounded and repaired LP values are proven within 0.5 % on 26 of 30 instances in about 2 s, and the reduced MILP of a
  610-unit model costs 5–6 s. The learned predictors do not help on this system (kNN 2.6×, self-trained 11.3×).

**The 20 California instances as distributed** (48 h, unchanged; `scripts/agg_pglib_base.py`,
`results/agg/pglib_base/report.md`; learning-free methods only, nothing trained or tuned on them; the full MILP's limit
is 30 min, which 2 instances reach, so their speed-ups are lower bounds):

| method | gap mean | max | served | speed-up geo / median / mean of ratios / ratio of means | proven by the LP bound |
|---|---|---|---|---|---|
| full MILP, 0.1 % (reference; 324 s on average) | 0.049 % | 0.18 % | 100 % | 1.0× | – |
| full MILP, 0.5 % | 0.069 % | 0.38 % | 100 % | 1.6× / 1.0× / 2.9× / 2.8× | – |
| **certified, LP values, proven within 0.5 %** | **0.082 %** | 0.23 % | 100 % | **11.0× / 10.8× / 20.8× / 14.5×** | 19 of 20 |
| certified, LP values, rounding first, proven within 0.5 % | 0.082 % | 0.23 % | 100 % | 9.6× / 9.4× / 18.2× / 13.5× | 19 |
| certified, LP values, rounding first, proven within 0.2 % | 0.072 % | 0.17 % | 100 % | 7.8× / 8.4× / 12.0× / 9.5× | 14 |
| certified, LP values, rounding first, proven within 0.1 % | 0.047 % | 0.11 % | 100 % | 3.6× / 3.6× / 5.8× / 1.9× | 9 |

At 48 hours the rounding step is proven less often and the plain certified pipeline is the faster one.

## Round 9 plan: the probing margins as a network input (fixed before its validation pass)

The temporal GNN of rounds 7–8 sees the LP relaxation's commitments, prices and line loadings, but not the probing
margins, which say how robust each LP-integral decision is economically (and which, alone, already rank decisions
better than the LP values: 14.3× against 10.1× on validation). Round 9 adds one per-unit, per-hour input: the margin in
% of z_LP, signed by the state of the unit's cheapest schedule at the LP prices, clipped to ±2 %
(`otsl.probe.margin_feature`, `scripts/r9_margins.py`); it also feeds the temporal head. Five seeds are trained exactly
as round 7's networks (2,000 instances with polished labels, `scripts/uc_m1x_train.py --margin`), with no other change.
Validation (`results/agg/val3`, 360 instances, one process per instance): the full MILP at 0.1 %, round 8's learned
choice (`acert:ens5+x:0.05`), the single round-7 network with probing, and the margin networks `acert:{m0,mens5}+x:{0.02,
0.05}`. Rule (`scripts/agg_report.py --select9`): the best margin-network variant by geometric-mean speed-up replaces
round 8's choice only if its paired speed-up over it is above 1; it is then read once on the 12-hour test set and the
six shifted sets (`--phase test2`), with round 8's choice run again in the same processes.

## Round 9 result (validation, 360 instances): not adopted

The margin feature makes the networks better calibrated (validation log-loss 0.0485 → 0.0453 for one network, 0.0463 →
0.0428 for five; decisions fixable at 99 % precision 98.74 % → 98.95 %, `results/uc12/m1x_train_r9.json`), but not the
pipeline faster: the best variant, five margin networks at δ = 0.05 with probing, runs at **0.974× [0.934, 1.015]** of
round 8's learned choice in the same processes, at the same accuracy (−0.006 pp), and the LP bound proves as many
answers (208 against 207). By the rule fixed beforehand it is not adopted and not tested (`results/agg/select9.json`,
records `results/agg/val3`). It confirms the diagnosis of this round: on RTS-GMLC the remaining time is the proof of
the bound on instances whose LP relaxation is weak, which better fixings cannot remove.

## Test (400 instances, PACE ICE, Xeon Gold 6226, read once)

Scoring as round 7 (gap to the tightest bound known for the instance; served = no shed, over-generation or reserve
shortfall; speed-up paired with the full MILP at 0.1 % in the same process). Every method of an instance ran back to
back in one single-threaded process; no instance failed. Full tables: `results/agg/report.md`.

**12 h, test_fresh (120)**

| method | gap mean [95 % CI] | max | served | time s | speed-up geo |
|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 0.156 [0.088, 0.247] | 3.64 | 97.5 % | 29.73 | 1.0× |
| full MILP, 0.5 % | 0.198 [0.131, 0.280] | 3.44 | 96.7 % | 16.66 | 2.1× |
| Learning to Fix (kNN, ε = 1 %, as published) | 0.660 [0.458, 0.874] | 5.85 | 95.0 % | 18.99 | 2.4× |
| round 7: screened full MILP, 0.5 % | 0.123 [0.089, 0.164] | 1.71 | 99.2 % | 11.50 | 3.7× |
| round 7: certified pipeline | 0.108 [0.091, 0.126] | 0.47 | 97.5 % | 7.42 | 12.9× |
| round 7: certified pipeline on the LP values | 0.119 [0.098, 0.141] | 0.53 | 96.7 % | 9.65 | 6.0× |
| **aggregated screened full MILP, 0.1 % (no learning)** | **0.049 [0.040, 0.058]** | 0.38 | 95.8 % | 14.54 | **3.1×** |
| **aggregated screened full MILP, 0.5 % (no learning)** | 0.082 [0.066, 0.100] | 0.49 | 96.7 % | 8.84 | **5.3×** |
| **aggregated certified pipeline, learned (`ens5+x`)** | **0.103 [0.085, 0.122]** | **0.47** | 96.7 % | 4.79 | **17.5×** |
| … LP certificate at 1 % (proven within 1 %) | 0.117 [0.096, 0.138] | 0.51 | 96.7 % | 4.35 | 20.9× |
| … without the certificate (confidence fixing alone) | 1.008 [0.303, 1.893] | 40.74 | 93.3 % | 1.52 | 29.6× |
| aggregated certified, one network + probing (ablation) | 0.095 [0.079, 0.112] | 0.38 | 98.3 % | 5.00 | 16.6× |
| aggregated certified, one network, no probing (ablation) | 0.100 [0.084, 0.118] | 0.49 | 98.3 % | 5.33 | 15.8× |
| **aggregated certified pipeline, no learning (`lag0.001+x`)** | **0.102 [0.084, 0.121]** | 0.49 | 95.8 % | 5.57 | **13.0×** |
| … LP certificate at 1 % | 0.109 [0.089, 0.129] | 0.59 | 95.8 % | 5.30 | 14.5× |
| … without the certificate | 0.367 [0.229, 0.547] | 5.83 | 96.7 % | 1.68 | 20.1× |

Paired (A − B in pp; A's speed-up over B; 95 % bootstrap intervals):

| A | B | gap A − B | speed-up |
|---|---|---|---|
| learned (`ens5+x`) | Learning to Fix | −0.557 [−0.765, −0.360] | 7.22× [5.69, 9.13] |
| no learning (`lag0.001+x`) | Learning to Fix | −0.558 [−0.798, −0.366] | 5.38× [4.22, 6.79] |
| learned | round 7's certified pipeline | −0.006 [−0.020, +0.008] | 1.36× [1.25, 1.47] |
| learned | full MILP, 0.5 % | −0.095 [−0.174, −0.032] | 8.31× [6.96, 9.93] |
| learned | the same with one network, no probing | +0.003 [−0.012, +0.019] | 1.11× [1.03, 1.19] |
| one network, aggregated | round 7 (one network, per unit) | −0.008 [−0.021, +0.005] | 1.22× [1.15, 1.30] |
| no learning (`lag0.001+x`) | round 7 on the LP values | −0.017 [−0.042, +0.006] | 2.19× [1.84, 2.62] |
| learned | no learning | +0.001 [−0.018, +0.020] | 1.34× [1.19, 1.51] |
| aggregated full MILP, 0.5 % | round 7's screened, 0.5 % | −0.041 [−0.080, −0.010] | 1.43× [1.29, 1.59] |
| aggregated full MILP, 0.1 % | full MILP, 0.1 % | −0.107 [−0.193, −0.043] | 3.12× [2.69, 3.63] |

**Shifted 12-hour sets (30 each; nothing re-tuned):** mean gap · served · speed-up

| method | load +15 % | load −15 % | wind + solar ×1.5 | units out | lines out | across midnight |
|---|---|---|---|---|---|---|
| full MILP, 0.5 % | 0.12 · 100 · 1.7× | 0.18 · 90 · 1.8× | 0.27 · 97 · 1.6× | 0.28 · 90 · 1.7× | 0.12 · 90 · 1.9× | 0.20 · 93 · 1.8× |
| Learning to Fix | 1.94 · 97 · 1.6× | 2.08 · 97 · 2.8× | 1.50 · 93 · 2.0× | 1.65 · 93 · 2.4× | 25.92 · 60 · 3.3× | 1.21 · 90 · 1.9× |
| round 7: certified pipeline | 0.12 · 97 · 11.7× | 0.08 · 93 · 9.9× | 0.08 · 90 · 6.2× | 0.14 · 93 · 10.2× | 0.11 · 93 · 10.3× | 0.10 · 90 · 9.6× |
| **learned (`ens5+x`)** | **0.10 · 100 · 14.6×** | **0.07 · 97 · 15.5×** | **0.12 · 90 · 8.7×** | **0.08 · 97 · 13.0×** | **0.09 · 90 · 12.6×** | **0.10 · 93 · 13.6×** |
| no learning (`lag0.001+x`) | 0.09 · 100 · 11.0× | 0.12 · 93 · 11.2× | 0.09 · 90 · 7.5× | 0.13 · 90 · 11.1× | 0.09 · 87 · 16.8× | 0.08 · 87 · 10.5× |
| aggregated full MILP, 0.5 % | 0.07 · 100 · 4.9× | 0.08 · 93 · 4.1× | 1.26 · 87 · 3.8× | 2.42 · 90 · 4.7× | 0.07 · 93 · 6.0× | 0.07 · 90 · 5.0× |

Over the 180: learned against Learning to Fix −5.62 pp [−8.33, −3.27] and 5.65× [4.50, 7.04] faster; against round 7
−0.012 pp and 1.35× [1.24, 1.48]; against the full MILP at 0.5 % −0.104 pp and 7.30× [6.20, 8.59]. (The aggregated
full MILP's 1.26 % and 2.42 % are time-limited solves holding poor schedules, as round 7's screened MILP did; the
certified pipelines cannot do that, as they keep the better of the reduced and the proof step's schedules.)

**24 hours (40) and congested lines (60)**: no learned model of these benchmarks was kept; learning-free methods only.

| method | 24 h: gap · max · speed-up | congested: gap · max · speed-up |
|---|---|---|
| full MILP, 0.1 % (reference) | 0.101 · 0.39 · 1.0× | 0.082 · 0.10 · 1.0× |
| full MILP, 0.5 % | 0.190 · 0.84 · 2.6× | 0.184 · 0.47 · 2.4× |
| round 7: screened full MILP, 0.5 % | 0.118 · 0.38 · 4.4× | 0.146 · 0.46 · 2.6× |
| round 7: certified pipeline on the LP values | 0.168 · 0.45 · 9.0× | 0.146 · 0.44 · 4.3× |
| **aggregated full MILP, 0.1 %** | **0.046 · 0.10 · 3.5×** | **0.069 · 0.10 · 1.7×** |
| aggregated full MILP, 0.5 % | 0.096 · 0.36 · 7.0× | 0.120 · 0.37 · 4.0× |
| **aggregated certified pipeline, no learning** | **0.097 · 0.38 · 13.7×** | **0.128 · 0.37 · 6.4×** |
| … LP certificate at 1 % | 0.097 · 0.38 · 14.2× | 0.133 · 0.37 · 7.9× |

Paired: learning-free against round 7's on the LP values 1.53× [1.18, 1.98] (24 h, −0.071 pp) and 1.49× [1.18, 1.88]
(congested, −0.018 pp); against the full MILP at 0.5 % 5.18× and 2.63×; the aggregated full MILP against round 7's
screened one 1.58× (24 h) and 1.54× (congested), and at 0.1 % against the full MILP 3.55× and 1.68×, more accurate.

**Anatomy** (means): the learned pipeline fixes 95.4 % of the decisions on test_fresh; the LP bound proves 78 of 120
answers, the proof step 42, with 177 exact probing fixings on average in that step; time LP relaxation + inference
0.38 s, reduced MILP 1.14 s, certificate 3.27 s (round 7: 1.72 / 5.34 s). The learning-free pipeline fixes 86.7 % and
proves 80 by the LP bound.

**Checks.** The learned pipeline proves all 120 test answers and 175 of 180 shifted ones within 0.5 % (5 hit the proof
step's 60 s limit); the learning-free one 119 + 176 of 300 on 12 h and all 100 on 24 h and congested (the full MILP at
0.5 % hits its limit on 30 of the 300 12-hour instances and 4 of the 100 others). All 1,200 screened or aggregated full
solves end with every line within its rating; no delivered cost is below the tightest bound; round 7's methods re-run
here reproduce round 7's costs (395–400 of 400 within 0.1 %); one node type throughout.

## Caveats

* RTS-GMLC only for the learned pipeline; the networks are round 7's (one training run each, five seeds); the 24-hour
  and congested benchmarks have no learned model (the learning-free pipeline is the one measured there).
* Learning to Fix on RTS-GMLC uses step 2's tuning (kNN on the 500 training schedules, Benders thresholds at ε = 1 %); on
  PGLib it was re-tuned here. Its reduced MILP is the full B-theta model, as published; it does not get the screen or the
  aggregation.
* The guarantee is as strong as the proof step's time limit (60 s on 12 h); answers that hit it keep a weaker proven
  gap, as the full MILP itself does.
* Served share: the learning-free pipeline's 24-hour answers include a small reserve shortfall more often (90 % served
  against 97.5 % for the full MILP at 0.1 %); shortfall is priced, and every answer is within 0.5 % in cost.
* Speed-ups are single-threaded on one CPU type; the screen and the aggregation pay where few lines bind and units come
  in identical groups (RTS-GMLC: 56 of 73 units in 24 groups).
