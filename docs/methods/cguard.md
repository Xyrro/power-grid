# A cost-aware check for fixed commitments (B2, 12-hour UC)

Code: [`otsl/cguard.py`](../../otsl/cguard.py), `scripts/uc_cguard_{common,diag,deep,run,shiftval,report}.py`.
Results: `results/uc12/cguard_results.{md,json}` (when written), diagnostics `results/uc12/cguard_diag_{test,val}.jsonl`.

## Status

*Kept current for resumption.* Started 2026-10-13 06:41 UTC on core 0 (`taskset -c 0`, HiGHS and torch one thread).

* Diagnosis (LPs only, `scripts/uc_cguard_diag.py --sets test / val`): done (300 test, 360 validation).
* Validation stage done 06:51–08:25 UTC (360 instances, `scripts/uc_cguard_run.py --stage val`); selection
  (`--stage select`): d_pre = 0.75 %, d_post = 4 % (`results/uc12/cguard_select.json`).
* Timing check (`--stage retime`, 08:25–08:30): the m1x pipeline records are **not** reusable — the same solves now run
  ~30 % faster on core 0 (full MILP ratio median 0.69, pipeline 0.67; objectives identical on 14 / 14). So the test
  pass re-solves everything (`--stage testfull`, started 08:31, ~45 s per instance, ~3.8 h; resumable, log
  `results/uc12/cguard_testfull.log`). Queued after it (`cguard_chain.log`): `scripts/uc_cguard_deep.py`, then the
  shifted-validation arm (`scripts/uc_cguard_shiftval.py --n 20`, then `--select`). Resume by re-running the stage.
* Report: `python3 scripts/uc_cguard_report.py` → `results/uc12/cguard_results.{md,json}`.

**Plan, fixed before the validation and test passes.** The diagnosis (below) showed that the relaxed-reduced-LP
increase Δ does not see the failures, while the reduced MILP's cost against the LP relaxation bound does. Candidates:
C1 pre-solve Δ check + release by reduced cost (the obvious design, kept as the negative control); C2 post-solve bound
check g = (C_red − z_LP) / C_red + gradient polish (60 fixings released by the dispatch-LP gradient at the solution,
re-solve from the incumbent at 0.5 %); C3 post-solve bound check + full MILP at 0.5 % (keep the better). Thresholds:
the smallest value of a fixed grid that fires on ≤ 5 % of the 360 original validation instances (val, val_extra,
val_extra2); nothing else is tuned. Pipeline records are reused only after re-solving 14 of them on core 0 within 10 %.

## Summary

(pending)

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

