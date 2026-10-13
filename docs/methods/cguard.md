# A cost-aware check for fixed commitments (B2, 12-hour UC)

Code: [`otsl/cguard.py`](../../otsl/cguard.py), `scripts/uc_cguard_{common,diag,deep,run,shiftval,report}.py`.
Results: [`results/uc12/cguard_results.md`](../../results/uc12/cguard_results.md) (+ `.json`); records `results/uc12/cguard_{diag_test,diag_val,val,test_full,shiftval}.jsonl`, `cguard_{select,select_B,retime,deep}.json`.

## Status

*Kept current for resumption.* Started 2026-10-13 06:41 UTC on core 0 (`taskset -c 0`, HiGHS and torch one thread).

* Diagnosis (LPs only, `scripts/uc_cguard_diag.py --sets test / val`): done (300 test, 360 validation).
* Validation stage done 06:51–08:25 UTC (360 instances, `scripts/uc_cguard_run.py --stage val`); selection
  (`--stage select`): d_pre = 0.75 %, d_post = 4 % (`results/uc12/cguard_select.json`).
* Timing check (`--stage retime`, 08:25–08:30): the m1x pipeline records are **not** reusable — the same solves ran
  ~30 % faster on core 0 (full MILP ratio median 0.69, pipeline 0.67; objectives identical on 14 / 14). The test pass
  therefore re-solved everything (`--stage testfull`, 08:31–16:45 with interruptions: an API spend limit at ~12:00 and
  container restarts at 12:49, 15:20 and 15:37; each time resumed, finished instances skipped). **Done**: 300 / 300.
* Failure anatomy (`scripts/uc_cguard_deep.py`, 16:45–16:55) done; shifted-validation arm (`scripts/uc_cguard_shiftval.py
  --n 20` and `--select`, 16:55–18:07) done: d_post_B = 2 % (`results/uc12/cguard_select_B.json`).
* **Study complete** (18:10 UTC). To reproduce: `scripts/uc_cguard_diag.py --sets test / val`, `scripts/uc_cguard_run.py
  --stage val / select / retime / testfull`, `scripts/uc_cguard_deep.py`, `scripts/uc_cguard_shiftval.py --n 20` and
  `--select`, `scripts/uc_cguard_report.py` (each pass resumable; all on one core with `taskset`).
* Report: `python3 scripts/uc_cguard_report.py` → `results/uc12/cguard_results.{md,json}`.

**Plan, fixed before the validation and test passes.** The diagnosis (below) showed that the relaxed-reduced-LP
increase Δ does not see the failures, while the reduced MILP's cost against the LP relaxation bound does. Candidates:
C1 pre-solve Δ check + release by reduced cost (the obvious design, kept as the negative control); C2 post-solve bound
check g = (C_red − z_LP) / C_red + gradient polish (60 fixings released by the dispatch-LP gradient at the solution,
re-solve from the incumbent at 0.5 %); C3 post-solve bound check + full MILP at 0.5 % (keep the better). Thresholds:
the smallest value of a fixed grid that fires on ≤ 5 % of the 360 original validation instances (val, val_extra,
val_extra2); nothing else is tuned. Pipeline records are reused only after re-solving 14 of them on core 0 within 10 %.

## Summary

The recommended pipeline (B-hg fixings → guards → LP-relaxation guard → reduced MILP at 0.5 %) loses to the full MILP at
0.5 % under wind + solar ×1.5 and load −15 % because some wrong commitments are expensive without shedding anything.

* **Diagnosis**: the relaxed reduced LP does not see these errors (Δ < 0.07 % on half of them), nor do per-fixing reduced
  costs or disagreement with the LP relaxation: the wrong fixings mostly *agree* with the relaxation's integral values
  (mid-size steam units it also keeps off). The reduced MILP's cost against the LP relaxation bound,
  g = (C_red − z_LP) / C_red, does (≥ 2 % on every bad instance), and costs nothing to compute.
* **Check that works (C3)**: if g exceeds 4 % (chosen on the original validation set: fires on 3.1 % there), solve the
  full MILP at 0.5 % and keep the better solution. On test it fires on 1 % of test_fresh (0.30 % at 10.1×, unchanged)
  and fixes both failures: load −15 % 1.59 → 0.40 %, wind + solar ×1.5 1.36 → 0.43 %, level with the full MILP at 0.5 %
  (−0.01 / +0.02 pp) at 1.9–2.3× its speed. No other set changes.
* **Negative results**: the pre-solve LP-cost check with reduced-cost release (C1) costs 6–17 % speed everywhere and
  leaves wind + solar significantly worse than the full MILP (+0.73 pp [+0.07, +1.64]); gradient polish after the bound
  check (C2) recovers only 15–30 % of the excess.
* **Recalibration on shifted validation data** (separate arm, validation days): threshold 2 %; also removes the units-out
  residual, every set within 0.06 pp of the full MILP at 0.5 %, for ~10 % of the in-distribution speed (9.9×).

## Question

The recommended pipeline of the m1x study (B-hg fixings → adequacy + min up/down guards → LP-relaxation guard →
reduced MILP at a 0.5 % gap; [`m1x.md`](m1x.md), last section) matches the full MILP at 0.5 % in distribution
(0.30 % at 9.7× on test_fresh 0–119) and survives line outages, but loses to the loosened full MILP under wind + solar
×1.5 (1.42 % vs 0.50 %, +0.91 pp [+0.21, +1.87]) and load −15 % (1.64 % vs 0.47 %, one instance at 27 %). There a
wrong commitment makes the reduced problem expensive without shedding anything, so neither the LP guard nor the LP
veto sees it. Is there a cost-aware check that catches these errors without giving up the in-distribution speed?

## Diagnosis: which signal separates the failures?

`scripts/uc_cguard_diag.py` rebuilds the pipeline's fixings for all 300 test instances (fixed shares identical to the
m1x records on 300 / 300) and computes candidate signals with LPs only; the pipeline's outcome comes from its records.
Ten instances are more than 1 pp worse than the full MILP at 0.5 % (load −15 %: 27.2, 5.6, 2.3 pp; wind + solar ×1.5:
8.6, 8.6, 4.4, 2.2, 1.9, 1.1 pp; units out: 1.9 pp); none of them sheds load or misses reserve.

* **The relaxed reduced LP does not see them.** Δ = (z_RR − z_LP) / z_LP, the LP relaxation with the fixings applied
  against the full relaxation, is below 0.07 % on 5 of the 10 (including both 8.6-pp instances), while in distribution
  it reaches 1.5 % on harmless instances (test_fresh 95th percentile 0.78 %); Spearman with the excess 0.26. The
  fixings are consistent with a cheap *fractional* completion; the cost appears only when the free decisions must be
  integral.
* **Per-fixing reduced costs do not separate them either.** The largest first-order saving of releasing one fixing
  (bound marginal in the relaxed reduced LP) is 4–42 % of z_LP on the bad instances and up to 42 % on good ones; the
  sums are dominated by OFF fixings of expensive units and carry no signal.
* **Disagreement with the LP relaxation's integral values does not separate them**: 6 of the 10 have no fixing that
  contradicts an integral LP value (on the 290 others: 218 none, 72 one to four). On one wind + solar instance every
  fixing agrees with the relaxation and the reduced MILP is still 2.2 pp worse: the relaxation's integral values are
  themselves wrong there (these sets have relaxation gaps of ~2 %).
* **The integer solution against the LP bound does.** g = (C_red − z_LP) / C_red, the reduced MILP's cost against the
  LP relaxation solved anyway as model input (an upper bound on the true gap, free to compute), is ≥ 2.0 % on all 10,
  and exceeds 2 % on 8 % of test_fresh instances and 4 % on 0.8 %. It cannot tell a wrong fixing from a weak relaxation
  (on the wind + solar set the relaxation gap itself is 5.4 % at the 90th percentile), so it fires more under shift,
  which is where it should.

So the check has to look at the integer solution: a post-solve bound check with a response, not a pre-solve LP check.
The pre-solve LP check (the obvious design) is kept as a control (C1).

## Candidates and calibration

Three candidates (`otsl/cguard.py`, `scripts/uc_cguard_run.py`), fixed after the diagnosis and before any validation or
test pass; z_LP is the LP relaxation already solved as model input.

* **C1, pre-solve LP-cost check** (the obvious design; control): Δ = (z_RR − z_LP) / z_LP. If Δ > d_pre, release the
  fixings with the largest first-order saving in the relaxed reduced LP (bound marginals of the fixed u; batches of 10 %
  of the fixings with positive saving, at least 5, at most 6 LPs) until Δ ≤ d_pre, then solve the reduced MILP. Its LP
  is an added cost on every instance.
* **C2, post-solve bound check + gradient polish**: g = (C_red − z_LP) / C_red of the reduced MILP's solution. If
  g > d_post: the dispatch LP's exact derivative at the solution (`otsl.solver.flip_savings`, as in the polish study,
  [`solver.md`](solver.md)) predicts the saving of flipping each decision; the 60 fixings with the largest predicted
  saving (60 = that study's uc12 setting) are released and the reduced MILP is re-solved from the incumbent at 0.5 %;
  the better solution is kept.
* **C3, post-solve bound check + full-MILP fallback**: if g > d_post, solve the full MILP at 0.5 % (cold) and keep the
  better solution. Costly where it fires, safe by construction.

**Calibration on the original validation set only** (val + val_extra + val_extra2, 360 instances, no shift; the B-hg
thresholds were tuned on the same instances): per instance the pipeline's fixings, Δ, the reduced MILP at 0.5 %, g;
where g > 1 % also C2's polish and C3's full MILP. Rule, fixed in advance: the smallest value of a fixed grid that fires
on at most 5 % of the 360 instances (the in-distribution speed budget). Result: **d_pre = 0.75 %** (fires on 3.9 %;
grid 0.25–5 %), **d_post = 4 %** (fires on 3.1 %; grid 1–10 %). On validation the pipeline has a 0.265 % mean gap to the
dataset MILP's bound; the 11 instances where the bound check fires average 1.32 % (pipeline), 1.22 % with C2's polish and
0.87 % with C3's fallback.

**Separate arm B: recalibration on shifted validation data** (`scripts/uc_cguard_shiftval.py`). The six shifts of the
robustness study ([`ood.md`](ood.md)) drawn on *validation* calendar days (day % 5 = 2) with new seeds (401–406),
20 instances each; per instance the pipeline and the full MILP at 0.5 %. Rule, fixed before that pass: d_post_B = the
largest grid value that fires on every shifted-validation instance where the pipeline is more than 1 pp above the full
MILP at 0.5 %, subject to firing on at most 10 % of the original validation instances (else the smallest value with at
most 10 %). C2's polish is run on every test instance with g > 1 %, so both thresholds are evaluated from the same test
records.

## Setup

* **Test sets**: test_fresh 0–119 and the robustness study's six shifted sets (30 each; outages handled as there:
  `otsl.ood.OODModel`, probabilities of unavailable units 0, guards on the available fleet, fixings of unavailable units
  forced OFF); 300 instances.
* **Paired timing.** The m1x pipeline records could not be reused: re-solving 14 of them on core 0 gave the same
  objectives but ~30 % shorter times (full MILP ratio median 0.69, ratio of means 0.77; pipeline 0.67 / 0.69;
  `results/uc12/cguard_retime.json`). Everything was therefore re-solved here, per instance back to back in one process
  on core 0 (HiGHS through highspy, one thread; torch one thread; the other three cores busy with other agents' jobs):
  full MILP at 0.1 % (dual bound DB, speed-up denominator), full MILP at 0.5 %, LP relaxation, B-hg inference and guards,
  the pipeline's reduced MILP at 0.5 %, C1 (relaxed reduced LP; release LPs and reduced MILP where it fires), C2's
  dispatch-LP gradient and polish MILP (where g > 1 %).
* **Method time** = inference + LP relaxation + guards (incl. the LP guard's LPs) + reduced MILP; plus C1's LPs and, where
  it fires, its new reduced MILP; plus, where the bound check fires, C2's gradient LP and polish MILP or C3's full MILP
  at 0.5 %. The bound check itself costs nothing.
* **Metrics** (as in the m1x study): gap = (C − DB) / C, mean [95 % CI] and max; served share (no shedding,
  over-generation or reserve shortfall); speed-up T_full(0.1 %) / T_method, mean and median; fixed share; firing rate.
  Paired instance-bootstrap 95 % CIs (2,000 resamples) of Δ gap, Δ log speed-up and Δ served against the pipeline and
  against the full MILP at 0.5 %. All reduced problems were feasible.

## Results

Full tables: [`results/uc12/cguard_results.md`](../../results/uc12/cguard_results.md) (+ `.json`); raw records
`results/uc12/cguard_test_full.jsonl`. The re-solved pipeline reproduces the m1x records' objectives on 300 / 300
instances (the reference full MILP on 276; the rest stopped at its 60 s limit at a different point). Every reduced
problem was feasible. Thresholds as selected on validation: d_pre = 0.75 %, d_post = 4 %.

**Mean gap % [95 % CI] (max), mean / median speed-up, firing rate of the check:**

| set | full MILP 0.5 % | pipeline | + C1 pre-solve LP check | + C2 bound check + polish | **+ C3 bound check + full-MILP fallback** |
|---|---|---|---|---|---|
| test_fresh (120) | 0.29 [0.21, 0.40] (3.97), 3.4 / 1.6× | 0.30 [0.23, 0.37] (2.71), 10.1 / 5.5× | 0.28 (2.78), 9.0 / 5.2×, fires 6 % | 0.30 (2.71), 10.1 / 5.5×, 1 % | **0.30 [0.23, 0.37] (2.71), 10.1 / 5.5×, 1 %** |
| load +15 % | 0.15 (0.51), 2.5 / 1.5× | 0.17 (0.77), 8.4 / 5.3× | 0.16, 7.6 / 5.0×, 7 % | 0.17, 8.4 / 5.3×, 0 % | 0.17 (0.77), 8.4 / 5.3×, 0 % |
| load −15 % | 0.41 [0.22, 0.63] (2.41), 3.0 / 1.2× | 1.59 [0.40, 3.49] (27.24), 8.0 / 7.1× | 0.57 [0.29, 0.90] (4.12), 6.9 / 5.5×, 17 % | 1.41 (27.24), 6.4 / 4.7×, 23 % | **0.40 [0.22, 0.59] (2.00), 6.2 / 4.1×, 23 %** |
| wind + solar ×1.5 | 0.41 [0.24, 0.58] (1.70), 3.0 / 1.0× | 1.36 [0.64, 2.35] (8.67), 7.2 / 4.0× | 1.13 [0.51, 2.05] (8.67), 6.1 / 3.5×, 20 % | 1.08 [0.44, 1.98] (8.67), 6.7 / 3.8×, 23 % | **0.43 [0.23, 0.68] (3.12), 6.5 / 3.7×, 23 %** |
| 2–3 units out | 0.49 (4.82), 2.2 / 1.6× | 0.61 (3.30), 10.0 / 6.4× | 0.58, 7.8 / 5.8×, 17 % | 0.61, 9.9 / 6.4×, 10 % | 0.61 (3.30), 9.8 / 6.4×, 10 % |
| 1–2 lines out | 0.15 (0.61), 2.9 / 2.1× | 0.22 (1.03), 10.4 / 5.3× | 0.19, 8.9 / 5.2×, 10 % | 0.22, 10.4 / 5.3×, 0 % | 0.22 (1.03), 10.4 / 5.3×, 0 % |
| across midnight | 0.22 (1.53), 2.6 / 1.6× | 0.22 (1.01), 8.3 / 5.6× | 0.22, 7.3 / 5.2×, 7 % | 0.22, 8.3 / 5.6×, 3 % | 0.22 (1.01), 8.3 / 5.6×, 3 % |

**Paired** (Δ gap pp [95 % CI]; time ratio = geometric mean of reference time / rule time, > 1 = the rule is faster):

| set | C3 − pipeline: Δ gap / Δ log speed-up | C3 − full MILP 0.5 %: Δ gap / time ratio | pipeline − full MILP 0.5 % | C1 − pipeline | C2 − pipeline |
|---|---|---|---|---|---|
| test_fresh | +0.00 / −0.01 [−0.03, 0.00] | +0.00 [−0.06, +0.06] / 2.81× | +0.00 [−0.06, +0.06] / 2.83× | −0.01 [−0.04, 0.00] / −0.08 [−0.10, −0.06] | −0.00 / −0.01 |
| load −15 % | **−1.19 [−3.10, −0.01]** / −0.43 [−0.74, −0.15] | **−0.01 [−0.14, +0.12]** / 1.92× | +1.18 [−0.00, +3.12] / 2.96× | −1.02 [−2.80, +0.01] / −0.19 | −0.18 [−0.55, −0.00] / −0.27 |
| wind + solar ×1.5 | **−0.93 [−1.91, −0.23]** / −0.33 [−0.58, −0.13] | **+0.02 [−0.08, +0.17]** / 2.26× | **+0.95 [+0.23, +1.92]** / 3.15× | −0.22 [−0.57, +0.01] / −0.17 | −0.27 [−0.52, −0.07] / −0.21 |
| 2–3 units out | 0.00 / −0.14 [−0.31, 0.00] | +0.12 [−0.07, +0.30] / 3.30× | +0.12 [−0.07, +0.30] / 3.80× | −0.03 / −0.15 | −0.00 / −0.08 |
| load +15 %, lines out, midnight | 0.00 / −0.03 to 0.00 | +0.00 to +0.06 (n.s.) / 2.69–2.89× | +0.00 to +0.06 (n.s.) | −0.02 to 0.00 / −0.06 to −0.12 | 0.00 / ≤ −0.01 |

* **C3 closes both failures at a third of the cost of the full MILP.** On load −15 % and wind + solar ×1.5 the bound check
  fires on 23 % of the instances (7 of 30 each). Over all 300 test instances, 11 have the pipeline more than 1 pp above
  the full MILP at 0.5 %; the check fires on 9 of them (every one above 2 pp except one), plus harmless instances whose
  relaxation is simply weak. With the full-MILP fallback the pipeline then matches the full MILP at 0.5 %
  (−0.01 pp and +0.02 pp, both CIs around 0; max 2.0 % and 3.1 % instead of 27.2 % and 8.7 %), and remains 1.9× and 2.3×
  faster than it (geometric mean; mean speed-up 6.2× and 6.5× against the reference full MILP, from 8.0× and 7.2×).
* **In distribution it costs nothing measurable**: the check fires on 1 of 120 test_fresh instances (g = 4.2 %, an
  instance where the pipeline was already right), 0.30 % at 10.1× mean / 5.5× median, Δ log speed-up −0.01
  [−0.03, 0.00]. It never fires on load +15 % or line outages, and on 3 % / 10 % of the midnight and unit-outage instances
  without changing a gap.
* **What C3 misses**: instances whose excess is below what g can distinguish from a weak relaxation: on units out
  1.9 pp (instance 7, g = 2.0 %) and on wind + solar 1.7 pp (instance 3, g = 3.7 %) stay. A lower threshold would catch
  them at a higher firing rate (arm B, below).
* **C2's gradient polish recovers only part** (wind + solar −0.27 pp, load −15 % −0.18 pp; the 27 % instance is not
  repaired). The failure anatomy (`scripts/uc_cguard_deep.py`, `results/uc12/cguard_deep.json`) shows why: the wrong
  fixings are mostly OFF fixings of mid-size steam units (76–155 MW) that the full MILP runs for 3–12 hours in place of
  several CT starts, and they *agree with the LP relaxation's integral values* (u_rel = 0). Among ~700 fixings the
  dispatch-LP derivative at the pipeline's solution ranks them 17th–96th (sometimes 165th–371st), so releasing 60 frees
  some and not others; the relaxed-reduced-LP ranking is no better.
* **C1, the pre-solve LP-cost check, is the wrong instrument.** It repairs the 27 % instance (its Δ was 1.6 %; releasing
  90 fixings by reduced cost gives 2.1 % instead of 27 %), so load −15 % improves (−1.02 pp, CI touching 0), but it
  leaves wind + solar significantly worse than the full MILP (+0.73 pp [+0.07, +1.64]) and costs speed everywhere
  (an extra LP on every instance and false alarms on 6–20 %: Δ log speed-up −0.06 to −0.19, all CIs below 0).

**Arm B: recalibration on shifted validation data.** On the 120 shifted validation instances (validation days) the
pipeline is more than 1 pp above the full MILP at 0.5 % on 11 (up to 21.9 pp); their g ranges from 1.4 % to 22.7 %, so
no threshold at most 10 %-firing in distribution catches all of them (1 % would, at a 20 % in-distribution firing rate).
The fallback of the rule gives **d_post_B = 2 %** (in-distribution validation firing 8.6 %, catches 9 of the 11). On test,
C3 at 2 %:

| set | gap mean [CI] (max), speed-up mean / median, fires | Δ vs pipeline (gap / log speed-up) | Δ vs full MILP 0.5 % (gap / time ratio) |
|---|---|---|---|
| test_fresh | 0.28 [0.22, 0.35] (2.71), 9.9 / 5.4×, 8 % | −0.01 [−0.03, −0.00] / −0.11 [−0.19, −0.05] | −0.01 [−0.07, +0.04] / 2.54× |
| load −15 % | 0.37 [0.19, 0.55] (2.00), 6.1 / 4.1×, 30 % | −1.23 [−3.12, −0.05] / −0.48 | −0.05 [−0.16, +0.06] / 1.82× |
| wind + solar ×1.5 | 0.36 [0.21, 0.52] (1.67), 5.9 / 3.3×, 33 % | −1.00 [−1.98, −0.29] / −0.52 | −0.05 [−0.10, +0.01] / 1.88× |
| 2–3 units out | 0.52 [0.25, 0.82] (3.30), 8.0 / 5.9×, 17 % | −0.10 [−0.25, +0.00] / −0.25 | +0.02 [−0.13, +0.16] / 2.96× |
| load +15 %, lines out, midnight | unchanged gaps, fires 0–10 % | 0.00 / −0.03 to 0.00 | +0.00 to +0.06 (n.s.) / 2.69–2.89× |

Recalibrating on shifted validation data buys the last residuals (units out, the 1.7-pp wind + solar instance) and a
slightly lower in-distribution gap, for about 10 % of the in-distribution speed (9.9× instead of 10.1×; geometric time
ratio −0.11) and ~10 % more under shift; every set is then level with the full MILP at 0.5 % while 1.8–3.0× faster than
it. C2's polish at 2 % stays partial (wind + solar 1.07 %, load −15 % 1.38 %).

## Verdict

* **A cost-aware check that works has to look at the integer solution, not the LP.** The obvious pre-solve check — the
  relaxed reduced LP against the full LP relaxation, with release by reduced cost — does not see the failures: the wrong
  fixings mostly *agree* with the LP relaxation's integral values (mid-size steam units the relaxation also keeps off),
  so the relaxed reduced problem is as cheap as the full relaxation. As C1 it costs 6–17 % speed everywhere and repairs
  only the one instance it happens to see.
* **The post-solve bound check with a full-MILP fallback (C3) does what was asked.** g = (C_red − z_LP) / C_red needs no
  extra LP (z_LP is the model input). With the threshold chosen on the original validation set only (4 %, 3.1 % firing
  there), it leaves the pipeline untouched in distribution (fires on 1 of 120 test_fresh instances; 0.30 % at 10.1× mean
  / 5.5× median, Δ log speed-up −0.01 [−0.03, 0.00]) and closes both failures: load −15 % 1.59 → 0.40 % (full MILP at
  0.5 %: 0.41 %), wind + solar ×1.5 1.36 → 0.43 % (0.41 %), each level with the loosened full MILP (−0.01 and +0.02 pp)
  and still 1.9× / 2.3× faster than it. Nothing else changes on the other shifts.
* **Cheaper responses do not suffice.** Releasing 60 fixings by the dispatch-LP gradient (C2) recovers 15–30 % of the
  excess; the culprits rank too low among ~700 fixings. The cost of C3 is paid only where it fires:
  23 % of instances under those two shifts, 1 % in distribution.
* **Recalibration on shifted validation data (arm B) helps a little more, at a price.** A 2 % threshold (8.6 % firing in
  distribution) also removes the units-out residual and brings every set to within 0.06 pp of the full MILP at
  0.5 %, at ~10 % of the in-distribution speed (9.9× vs 10.1×). Neither threshold catches everything: excesses of
  1–2 pp on instances whose relaxation gap is itself 2–4 % are indistinguishable from a weak relaxation by this signal.
* **Recommendation**: add the bound check with a full-MILP fallback (threshold 4 %, or 2 % if robustness to unknown
  shifts is worth ~10 % of the in-distribution speed) to the recommended pipeline. It turns the pipeline's two known
  failure modes into a bounded cost and keeps it 1.8–3.3× faster than the loosened full MILP on every set tested.

## Caveats

* **Small failure counts.** 11 of 300 test instances have the pipeline more than 1 pp above the full MILP at 0.5 %; the
  conclusions on the two failure sets rest on 30 instances each and a handful of large errors (CIs in the tables).
* **The bound gap g is an upper bound** that mixes a bad fixing with a weak relaxation; under shifts with large
  relaxation gaps (wind + solar ×1.5: ~5 % at the 90th percentile) it fires on harmless instances too, which costs time
  but not accuracy. The threshold was chosen on validation instances on which the B-hg thresholds were themselves tuned
  (in-sample for the fixings, not for g).
* **Timing.** The pass ran over ten hours with interruptions, under varying machine load (load average ~4 most of the
  time). Every comparison is paired within an instance (all its solves back to back on core 0), so speed-ups are
  per-instance ratios; absolute times are not comparable across instances solved at different times. The m1x
  pipeline records were not reused (they were ~30 % slower than this core now).
* **Full-MILP fallback is cold.** A warm start from the pipeline's solution would make it faster; not tried.
* **One gradient-release size** (60, the polish study's uc12 setting) and one batch rule for C1; not tuned.
* **Arm B** used 20 shifted validation instances per shift (120), with the same shift generators as the test sets
  (only the calendar days and seeds differ), so it is an optimistic test of "recalibration on shifted data": the
  shifts are the same kind as those it is tested on.

