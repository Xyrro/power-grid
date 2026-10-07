# No-learning baselines and fair solver budgets on networked RTS-GMLC (12 h and 24 h)

*Code: [`otsl/base.py`](../../otsl/base.py) (LP-integral fixings, LP rounding + repair + LP screening, solver wrapper,
incumbent-log helpers, paper metrics and bootstrap), `scripts/uc_base_{val,tune,eval,report,audit,fuse,stack_report}.py`. Tests:
`tests/test_{ltfx,hybrid,b3,pglib,metrics}.py` (+ `tests/conftest.py`). Results:
[`results/uc12/base_results.md`](../../results/uc12/base_results.md), [`results/uc24/base_results.md`](../../results/uc24/base_results.md)
(+ `.json`), raw records `results/<bench>/base_eval_test.jsonl`, validation `results/<bench>/base_val.jsonl`,
`base_val_select.json`, tuning `results/<bench>/base_tune_lp_1.{json,log}`, run logs `results/<bench>/base_*.log`;
stacking pass `results/<bench>/base_stack_test.jsonl`, `base_stack.json` (+ the `stacking` key of `base_results.json`),
fused rule `results/uc12/base_tune_fused_1.{json,log}`, `_cuts.jsonl`.*

## Summary

On networked RTS-GMLC the LP relaxation is looser than on PGLib California but still 98 % integral: 0.19 % (12 h) and
0.26 % (24 h) below the MILP optimum at the median (California: 0.03 %), mean 0.47 / 0.37 %, with 7–12 wrong
integral values per instance. All timing is paired: every run of an instance was solved back to back on one dedicated
core, and the re-run references reproduce their published paper metrics. Results:

* **Fixing the LP's integral values is fast but costs ~1 %.** 12 h: 1.06 % at 12.9×; 24 h with guards: 0.76 % at
  21.9×.
* **Learning to Fix's calibration run on the LP values (no learning) is as accurate as the best learned rules.**
  12 h: 0.23 % at 2.0×, level with the hybrid (0.24 %), but the hybrid is 1.75× faster (geometric mean). 24 h:
  0.34 % at 6.8×. That is better than LtF-kNN on both axes (−0.20 pp [−0.38, −0.03], 1.6× faster) and more accurate
  than our guarded 95 % rule (−0.30 pp [−0.54, −0.09]) at a speed the test cannot separate.
* **Loosening the solver is a strong baseline.** The full MILP at a 1 % gap gives 0.21 % at 4.0× on 12 h. That is
  level with the hybrid (−0.03 pp [−0.11, +0.04]; the hybrid is 1.27× faster by geometric mean, not significant) and
  more accurate than LtF-kNN at equal speed. On 24 h the MILP at a 0.5 % gap gives 0.28 % at 4.9×, more accurate than
  LtF-kNN at equal speed.
* **Stopping the full MILP at the method's time is a poor alternative.** At each method's time it often has no
  solution yet, or a much worse one.
* **Without a MILP, LP rounding + repair + 5 LPs beats the learned end-to-end pipelines.** 12 h: 1.59 % vs 2.64 %.
  24 h: 1.83 % vs 5.65 %.
* **Fixing and a loose gap stack (§4.4).** Solving the reduced MILPs at the same 0.5 % / 1 % gap as a loosened full
  MILP keeps every fixing rule 1.7–6× faster than it (geometric mean, significant on both benchmarks). The most
  accurate rules do this at a gap the tests cannot tell apart from the loosened full MILP's: on 12 h the hybrid and a
  new fused rule (error-cost score with an LP veto) at 0.5 % give 0.28 / 0.25 % at 8.6 / 9.2× (full MILP: 0.20 % at
  2.9×); on 24 h learning-free LtF at 0.5 % gives 0.35 % at 15.7× (full MILP: 0.25 % at 5.1×). The loose gap alone is
  not the frontier.
* **Where learning still wins.** On 12 h, Learning to Fix on our BCE GNN matches the 1 %-gap MILP in gap at 1.55× its
  speed (significant). The learned rules at ~13× are 0.2–0.3 pp more accurate than LP-integral fixing (borderline).
  On 24 h no learned rule beats the best no-learning baseline.

## 1. Questions

1. On PGLib California the LP relaxation is within 0.03 % (median) of the optimum and learning-free LP rounding gives
   0.33 % at 24× (§6 X7 of [`RESEARCH.md`](../RESEARCH.md)). Is the networked RTS-GMLC benchmark (73 units, DC network,
   12 h and 24 h) equally easy, i.e. do no-learning baselines built on the LP relaxation match the learned rules
   (hybrid, Learning to Fix with kNN)?
2. Would a practitioner simply loosen the solver? The full MILP at a 0.25 / 0.5 / 1 % relative gap, and the full MILP
   stopped after exactly each method's time.

## 2. Methods

No model and no training data; the only input is the instance's LP relaxation (solved once, ~0.3 s on 12 h, ~1.2 s
on 24 h, and counted in every baseline's time).

* **(a) LP-integral fixing** (`otsl.base.lp_integral_fixings`): fix every unit-hour whose LP-relaxation value is
  within `tol` of 0 or 1 to that value, solve the reduced MILP (the "fix the integral values" rule of
  [`pglib.md`](pglib.md)). Variant **+ guards**: our test-time guards in the hybrid's order (`otsl.hybrid.apply_guards`:
  adequacy guard with a 5 % margin, min up/down row release, LP-relaxation guard). `tol` ∈ {1e-6, 0.05, 0.2} chosen
  on validation per guard setting (rule fixed before the runs: fewest infeasible reduced MILPs, then the lowest mean
  gap to the validation MILP objective C\*; ties within 0.02 pp → larger fixed share).
* **(b) Learning-free Learning to Fix** (`scripts/uc_base_tune.py`): the faithful joint threshold tuning of
  `otsl.ltfx` (Algorithm 1 + 2, Appendix A master, OR-cuts) with π = the LP-relaxation values, at ε = 1 %, with the
  settings of the faithful kNN run it is compared with (12 h: `LtFTuner`, 180 validation instances, check MILPs
  60 s, relaxation MILPs 6 s, K_max = 10, Q = 20, 75-min budget; 24 h: `uc24ltf.StartLtFTuner`, 40 validation
  instances, relaxation MILPs 8 s, 40-min budget). At test time: eq. (5) on the instance's LP relaxation, reduced MILP,
  no guards (faithful).
* **(c) LP rounding + repair + LP screening** (`otsl.base.lp_round_screen`), the PGLib baseline ported unchanged:
  round u_rel at 0.001 / 0.05 / 0.2 / 0.5 / 0.8, block (min up/down-aware) adequacy repair, min up/down repair, exact
  dispatch LP of every distinct schedule, keep the cheapest (penalties included). No MILP.

Fair solver budgets:

* **Loose gaps**: the full MILP with `mip_rel_gap` = 0.25 %, 0.5 %, 1 % (same time limit), solved in the same
  process as the reference.
* **Same budget**: the full MILP's best incumbent after exactly each method's per-instance time, read from the
  incumbent log of the re-timed 0.1 % run (HiGHS is deterministic, so a run with that time limit would have the same
  incumbent up to timing noise); and the time-to-quality (when the full MILP first reaches the method's cost).

## 3. Setup

* **Instances.** 12 h: the first 60 instances of `test_fresh` (the references' test set). 24 h: the first 20 of the 40
  uc24 test instances (subsample for compute: one back-to-back set — full MILP at four gaps, two references, four
  baselines — costs ~9 min per instance on one core). Validation: 12 h `val` (60) for the tolerance, `val` +
  `val_extra` (180) for the LtF tuning; 24 h `uc24ltf_val` (40) for both.
* **Paired timing.** Every run of an instance (LP relaxation, full MILP at 0.1 / 0.25 / 0.5 / 1 %, the reference
  rules re-run from their saved thresholds and probabilities, the baselines) is solved back to back in one process on
  one dedicated core (`taskset -c 1`, HiGHS 1 thread, torch 1 thread). The stored reference times were taken on a
  different machine state (a 5-instance probe here: full MILP 1.4–2.1× slower than in the stored 12-hour records), so
  nothing is compared with stored times. **Solver**: 12 h — highspy 1.12, i.e. the HiGHS bundled with scipy 1.17 that
  every 12-hour reference used (`scipy.optimize.milp`); on 5 instances it reproduces scipy's objectives exactly with
  run times within ~5 % (scipy's own call adds ~0.5 s of model set-up, which the stored times include and ours do
  not); 24 h — highspy 1.15.1 as in uc24ltf. Solver time is `Highs.run()`; method time adds inference, the LP
  relaxation where used, fixing rule and guards.
* **Reference rules** (fixings rebuilt from the saved files; fixed counts and reduced-MILP objectives checked against
  the stored records): 12 h — faithful LtF kNN ε = 1 % (kNN recomputed and timed here), faithful LtF on the BCE GNN
  ε = 1 %, hybrid `he_bce_s0_e1_n360`, our error-cost + adequacy rule at 90 %; 24 h — faithful LtF kNN ε = 1 %, our
  guarded 95 % rule (`95%|best=bce_g`). GNN / error-cost inference times are the stored per-instance values
  (milliseconds).
* **Metrics** (Learning to Fix, Sec. IV-C): gap (C − DB) / C to the reference dual bound (12 h: the hybrid study's
  back-to-back full MILP, obj · (1 − gap); 24 h: the dataset run's bound) — the same DB as in the reference tables
  and for every row here; feasibility rate; speed-up = per-instance T_MILP / T_method with T_MILP the re-timed 0.1 %
  run (mean, median, ratio of mean times); statistics over feasible instances; fixed share after guards; served share.
  Paired instance-bootstrap 95 % CIs (2,000 resamples) of Δ gap and Δ log speed-up against the hybrid and LtF-kNN
  (12 h) / LtF-kNN and the guarded 95 % rule (24 h), and against the full MILP at a 1 % gap (both).

## 4. Results

### 4.1 Validation

* **LP-integral tolerance** (12 h, 60 `val` instances; 24 h, 40 `uc24ltf_val`): the tolerance barely changes what is
  fixed (98.2 → 98.9 % of the decisions from tol 1e-6 to 0.2 on 12 h) and no reduced MILP was infeasible. 12 h:
  1e-6 without guards (1.36 % mean gap to C\*, max 10.1 %), 0.05 with guards (1.23 %, max 7.6 %). Already on validation
  this rule is ~1 % worse than the MILP: 20 of the 60 instances are above 1 %, 37 above 0.5 %. On the integral entries
  the relaxation disagrees with the (symmetry-aligned) MILP schedule on 6.0 OFF and 3.3 ON decisions per instance;
  the gap correlates with the wrong OFF fixes (r = 0.33) and with the instance's LP-relaxation gap (r = 0.42), not with
  adequacy, so the guards (built against shortfall) release almost nothing.
* **Learning-free Learning to Fix** (12 h, 180 instances): converged in 30 iterations (23 min), every validation
  instance within ε (witness max 0.996 %), but at **55.2 % fixed** against 66.7 % for the paper's kNN and 83.6 % for our
  BCE GNN under the same tuning. The LP values are 98 % exactly 0 or 1, so a cut that must release one wrong integral
  value of a unit (π = 0 or 1) can only do it by freeing *all* of that unit's integral values on that side: 32 units
  end with τ̲ = 0 (never fixed OFF) and 27 with τ̄ = 1 (never fixed ON). The relaxation carries no confidence within
  its integral values, which is exactly what the calibration needs.
* **24 h validation** (40 instances): LP-integral 0.79 % mean gap to C\* (tol 1e-6, no guards; max 4.5 %, 98.0 %
  fixed); with guards the rule picks tol 0.2 (0.58 %, 91.8 % fixed: the LP guard releases the fractional roundings
  that would shed or miss reserve). LP rounding + 5 LPs: 1.59 % (median 0.91 %), 92.5 % served. Learning-free LtF
  converged in 15 iterations (14 min), every instance within ε (max 0.99 %), **80.5 % fixed** — more than the kNN
  under the same tuning (68.5 %); 15 units never fixed OFF, 23 never fixed ON.

### 4.2 12 hours (first 60 of `test_fresh`)

Full MILP re-timed on this core: 26.3 s mean (stored back-to-back run 20.3 s; median ratio 1.41, the same factor for
every re-run rule, 1.36–1.39), 12 % at the 60 s limit. The re-run references reproduce their stored objectives
(95–100 % of instances identical; the rest are time-limited solves) and **their published paper metrics**: LtF-kNN
0.41 % at 4.6× (published 0.40 % at 4.6×), LtF-BCE 0.28 % at 5.4× (5.2×), hybrid 0.24 % at 5.2× (5.2×), error-cost
90 % 0.42 % at 4.2× (4.2×). LP relaxation: 0.47 % below the MILP objective on average (median 0.19 %, max 3.8 %;
validation 0.78 / 0.33 %), 0.33 % below the reference dual bound; 98.3 % of its values integral, 7.3 of them wrong per
instance. PGLib California: median 0.03 %, 99 % integral.

Paper metrics (gap to the reference dual bound, feasible instances; speed-up = re-timed T_MILP / T_method, mean
[95 % CI], median, geometric mean; full table in [`results/uc12/base_results.md`](../../results/uc12/base_results.md)):

| method | feasible | gap mean [CI] | gap median | speed-up mean [CI] | median | geo. mean | fixed |
|---|---|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 100 % | 0.16 % | 0.09 % | 1.0× | 1.0× | 1.0× | 0 % |
| full MILP, 0.25 % gap | 100 % | 0.15 % [0.09, 0.24] | 0.10 % | 1.7× [1.3, 2.3] | 1.1× | 1.4× | 0 % |
| full MILP, 0.5 % gap | 100 % | 0.20 % [0.12, 0.30] | 0.10 % | 2.8× [1.8, 4.3] | 1.3× | 1.8× | 0 % |
| **full MILP, 1 % gap** | 100 % | **0.21 % [0.14, 0.30]** | 0.10 % | **4.0× [2.6, 5.8]** | 1.6× | 2.3× | 0 % |
| LtF kNN ε = 1 % (paper's setting) | 100 % | 0.41 % [0.28, 0.56] | 0.23 % | 4.6× [2.5, 7.8] | 1.6× | 2.1× | 68 % |
| LtF on our BCE GNN ε = 1 % | 98.3 % | 0.28 % [0.17, 0.40] | 0.10 % | 5.4× [3.9, 7.3] | 2.5× | 3.6× | 84 % |
| hybrid (guard-aware LtF on error-cost scores) | 100 % | 0.24 % [0.17, 0.33] | 0.12 % | 5.2× [3.6, 7.3] | 2.2× | 2.9× | 86 % |
| ours: error-cost + adequacy, 90 % | 100 % | 0.42 % [0.25, 0.63] | 0.10 % | 4.2× [3.4, 5.0] | 2.9× | 3.3× | 90 % |
| *no learning*: LP-integral fixing (tol 1e-6) | 100 % | 1.06 % [0.80, 1.36] | 0.81 % | 12.9× [10.1, 16.3] | 9.9× | 9.2× | 98 % |
| *no learning*: LP-integral (tol 0.05) + guards | 100 % | 1.10 % [0.81, 1.41] | 0.81 % | 11.0× [9.2, 12.9] | 9.0× | 8.5× | 98 % |
| *no learning*: LtF on the LP relaxation, ε = 1 % | 100 % | **0.23 % [0.15, 0.34]** | 0.09 % | 2.0× [1.7, 2.5] | 1.5× | 1.7× | 55 % |
| *no learning*: LP rounding + repair + 5 LPs (no MILP) | 100 % | 1.59 % [1.07, 2.19] | 0.85 % | 27.6× [23.1, 32.1] | 28.8× | 19.2× | – |

Paired (method − reference, instances feasible for both; Δ log speed-up > 0: method faster):

| method | vs hybrid: Δ gap pp / Δ log speed-up | vs LtF-kNN | vs full MILP at 1 % gap |
|---|---|---|---|
| full MILP, 1 % gap | −0.03 [−0.11, +0.04] / −0.24 [−0.51, +0.03] | **−0.20 [−0.37, −0.06]** / +0.09 [−0.26, +0.45] | – |
| hybrid | – | **−0.16 [−0.33, −0.02]** / **+0.33 [+0.01, +0.62]** | +0.03 [−0.04, +0.11] / +0.24 [−0.03, +0.51] |
| LtF on our BCE GNN | +0.03 [−0.06, +0.14] / +0.19 [−0.03, +0.40] | −0.13 [−0.30, +0.02] / **+0.52 [+0.25, +0.79]** | +0.07 [−0.03, +0.19] / **+0.44 [+0.16, +0.73]** |
| LtF on the LP relaxation | −0.02 [−0.10, +0.09] / **−0.56 [−0.79, −0.33]** | **−0.18 [−0.34, −0.02]** / −0.23 [−0.51, +0.03] | +0.02 [−0.07, +0.14] / **−0.32 [−0.57, −0.08]** |
| LP-integral (tol 1e-6) | **+0.82 [+0.55, +1.09]** / **+1.15 [+0.87, +1.42]** | **+0.65 [+0.39, +0.92]** / **+1.48 [+1.17, +1.76]** | **+0.85 [+0.61, +1.10]** / **+1.39 [+1.14, +1.61]** |
| LP rounding + 5 LPs | **+1.34 [+0.83, +1.94]** / **+1.89 [+1.52, +2.25]** | **+1.18 [+0.69, +1.76]** / **+2.22 [+1.85, +2.57]** | – |

At the speed of the no-learning fixing rule (~13×), the learned rules of the stored reference runs (same instances,
same DB; objectives are deterministic, speed-ups from their own run) are only slightly more accurate: combined
pipeline 98 % 0.82 % at 13.8× and LtF on the self-trained GNN at ε = 10 % 0.85 % at 13.6×; LP-integral − these:
+0.24 [−0.04, +0.52] and +0.18 [−0.09, +0.46] pp (with guards +0.28 [+0.01, +0.56]). Without a MILP, the no-learning LP
rounding (1.59 %, median 0.85 %, 96.7 % served) is *better* than the best learned end-to-end pipeline of the earlier
study on the same 60 instances (combined + screening, 3 seeds: 2.64 %, median 1.02 %, 91.7 % served; −1.05 pp
[−2.60, +0.03]).

**Same budget.** Stopped after exactly each method's time, the full MILP has no solution yet on 7–30 % of the
instances (the 12-hour MILP finds its good incumbent at the end of the root) and, where it has one, it is 5–13 % worse
on average; every method beats it (hybrid: 0.24 % vs 8.5 %, method better on 29, full better on 15, full without a
solution on 14 of 60). Time to the same quality (median): hybrid 1.7×, LtF-BCE 2.1×, LtF-kNN 1.35×, LtF on the LP
relaxation 1.2×, LP-integral 4.8×, LP rounding 10×. **The fair cheap alternative is the loose gap, not the time
limit**: a 1 % gap stops the full MILP after the root at a median 1.6× (mean 4.0×) and still delivers 0.21 %.

### 4.3 24 hours (first 20 of the 40 uc24 test instances)

Full MILP re-timed: 150 s mean (median 88 s, 30 % at the 300 s limit), same objectives as the stored dataset run on
all 20 (median time ratio 0.98: this core runs at the dataset run's speed, ~1.27× slower than the uc24ltf re-runs).
The two references reproduce their stored objectives on 20/20 and their load-corrected published metrics: LtF-kNN
0.54 % at 5.8× (published on all 40: 0.60 % at 7.1×, 5.6× load-corrected), guarded 95 % rule 0.61 % at 11.7× (0.67 % at
14.8×, 11.6× load-corrected). LP relaxation: 0.37 % below the MILP objective (median 0.26 %, max 1.6 %), 0.21 % below
the dual bound; 97.9 % integral, 11.7 wrong integral values per instance.

| method | feasible | gap mean [CI] | gap median | speed-up mean [CI] | median | geo. mean | fixed | served |
|---|---|---|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 100 % | 0.16 % | 0.09 % | 1.0× | 1.0× | 1.0× | 0 % | 95 % |
| full MILP, 0.25 % gap | 100 % | 0.19 % [0.10, 0.29] | 0.10 % | 1.8× [1.3, 2.4] | 1.0× | 1.5× | 0 % | 90 % |
| **full MILP, 0.5 % gap** | 100 % | **0.28 % [0.15, 0.43]** | 0.15 % | **4.9× [2.4, 7.9]** | 1.9× | 2.6× | 0 % | 95 % |
| full MILP, 1 % gap | 100 % | 0.39 % [0.24, 0.55] | 0.34 % | 6.2× [3.7, 9.2] | 3.3× | 3.7× | 0 % | 80 % |
| LtF kNN ε = 1 % (paper's setting) | 100 % | 0.54 % [0.30, 0.83] | 0.28 % | 5.8× [3.0, 9.2] | 3.0× | 3.4× | 68 % | 95 % |
| ours: guarded 95 % rule (label-free GNN) | 100 % | 0.61 % [0.36, 0.90] | 0.42 % | 11.7× [6.0, 18.8] | 3.2× | 5.6× | 84 % | 95 % |
| *no learning*: LP-integral (tol 1e-6) | 100 % | 0.88 % [0.52, 1.30] | 0.59 % | 16.3× [8.4, 26.6] | 6.6× | 9.3× | 98 % | 80 % |
| *no learning*: **LP-integral (tol 0.2) + guards** | 100 % | 0.76 % [0.42, 1.19] | 0.42 % | **21.9× [12.2, 33.6]** | 8.7× | 12.2× | 96 % | 85 % |
| *no learning*: **LtF on the LP relaxation, ε = 1 %** | 95 % | **0.34 % [0.19, 0.50]** | 0.16 % | 6.8× [4.1, 10.4] | 4.7× | 4.8× | 81 % | 95 % |
| *no learning*: LP rounding + repair + 5 LPs (no MILP) | 100 % | 1.83 % [0.90, 3.08] | 0.81 % | 57.8× [38.1, 79.2] | 31.6× | 36.0× | – | 100 % |

Paired (method − reference; Δ gap pp / Δ log speed-up, > 0: method faster):

| method | vs LtF-kNN | vs guarded 95 % rule | vs full MILP at 1 % gap |
|---|---|---|---|
| full MILP, 0.5 % gap | **−0.25 [−0.47, −0.09]** / −0.28 [−0.84, +0.24] | **−0.33 [−0.60, −0.09]** / **−0.78 [−1.29, −0.24]** | **−0.11 [−0.22, −0.01]** / **−0.37 [−0.64, −0.14]** |
| full MILP, 1 % gap | −0.15 [−0.35, +0.03] / +0.09 [−0.45, +0.59] | −0.22 [−0.52, +0.05] / −0.41 [−0.88, +0.08] | – |
| guarded 95 % rule | +0.07 [−0.21, +0.35] / +0.50 [−0.13, +1.13] | – | +0.22 [−0.05, +0.52] / +0.41 [−0.08, +0.88] |
| LtF on the LP relaxation | **−0.20 [−0.38, −0.03]** / **+0.45 [+0.10, +0.80]** | **−0.30 [−0.54, −0.09]** / −0.20 [−0.80, +0.39] | −0.03 [−0.15, +0.08] / +0.26 [−0.22, +0.77] |
| LP-integral (tol 0.2) + guards | +0.23 [+0.00, +0.48] / **+1.27 [+0.62, +1.88]** | +0.16 [−0.13, +0.50] / **+0.77 [+0.40, +1.21]** | **+0.38 [+0.08, +0.71]** / **+1.18 [+0.75, +1.62]** |
| LP rounding + 5 LPs | **+1.30 [+0.55, +2.29]** / **+2.35 [+1.68, +2.95]** | **+1.23 [+0.39, +2.40]** / **+1.85 [+1.43, +2.31]** | **+1.45 [+0.59, +2.58]** / **+2.26 [+1.79, +2.75]** |

The faster learned rules are worse than the no-learning ones on the same 20 instances: the REINFORCE ranking at 95 %
(stored re-run) has 7.19 % at 64× (load-corrected), 6.3–7.0 pp worse than every no-learning baseline; the best learned
end-to-end pipeline (REINFORCE + screening, 15 candidates) has 5.65 % (median 3.37 %, 90 % served), 3.82 pp
[2.15, 5.81] worse than LP rounding (1.83 %, median 0.81 %, 100 % served).

**Same budget.** Stopped after each method's time, the 24-hour full MILP usually has a solution (85–95 %), and against
LtF-kNN it is level (0.92 % vs 0.54 % where it has one; −0.34 pp [−1.18, +0.30], 8 wins each). Against the faster
rules it is far behind (guarded 95 %: 0.61 % vs 14.1 %; LP-integral: 0.9 % vs 22 %). Time to the same quality (median):
LtF-kNN 1.5×, LtF on the LP relaxation 1.7×, guarded 95 % 2.3×, LP-integral 2.2× / 3.6× (with guards), LP rounding 10×.

### 4.4 Stacking: partial fixing on top of a loose solver gap

Every fixing rule above solves its reduced MILP at the reference gap (0.1 %). A second paired pass (same instances,
one process per benchmark) solved each rule's reduced MILP at `mip_rel_gap` 0.5 % and 1 % and re-ran the full MILP at
the same two gaps back to back with them. Gaps are to the reference dual bound; speed-ups are against the first pass's
0.1 % full MILP. The re-run loose-gap full MILPs took 0.96–0.99× their first-pass time (median; 90–100 % of instances
within 10 %), so the two passes are comparable. Paired comparisons are against the full MILP at the *same* gap. The
"Pareto" column is descriptive (chosen on test): a point is dominated if another has a lower-or-equal mean gap and a
higher-or-equal speed-up. Full tables: the "Stacking" sections of `results/<bench>/base_results.md`.

**24 hours** (20 instances; speed-up mean / geometric mean):

| rule | 0.1 % gap | 0.5 % gap | 1 % gap | vs full MILP at 0.5 %: Δ gap pp / Δ log speed-up |
|---|---|---|---|---|
| full MILP | 0.16 % · 1.0× | 0.25 % · 5.1× / 2.7× | 0.39 % · 6.5× / 3.9× | – |
| LtF kNN ε = 1 % | 0.54 % · 5.8× / 3.4× | 0.64 % · 15.9× / 9.4× | **0.70 % · 18.6× / 13.0×** (frontier) | **+0.39 [+0.17, +0.67]** / **+1.26 [+0.68, +1.86]** |
| ours: guarded 95 % rule | 0.61 % · 11.7× / 5.6× | 0.63 % · 14.6× / 8.6× | 0.72 % · 17.4× / 10.6× | **+0.38 [+0.13, +0.66]** / **+1.17 [+0.79, +1.58]** |
| LtF on the LP relaxation | **0.34 % · 6.8× / 4.8×** | **0.35 % · 15.7× / 8.1×** | **0.48 % · 18.1× / 10.9×** | +0.09 [−0.05, +0.23] / **+1.07 [+0.67, +1.50]** |
| LP-integral + guards | **0.76 % · 21.9× / 12.2×** | **0.79 % · 32.4× / 16.5×** | **0.86 % · 34.3× / 19.4×** | **+0.54 [+0.23, +0.90]** / **+1.82 [+1.42, +2.24]** |

*(bold: on the Pareto frontier of mean gap vs mean speed-up)*

* **Speed-ups stack on 24 h.** At the same loose gap every rule is 2.7–6× faster than the full MILP (geometric
  mean, all significant). The cost in gap is 0.3–0.5 pp for the learned and LP-integral rules, and none for
  learning-free LtF: at 0.5 % it is level with the 0.5 %-gap full MILP in gap (0.35 vs 0.25 %, +0.09 pp
  [−0.05, +0.23]) and 2.9× faster.
* **The frontier is mostly no-learning.**
  * The full MILP at 0.1 / 0.5 %, learning-free LtF at all three gaps, and LP-integral + guards at all three gaps are
    on it.
  * The paper's LtF-kNN reaches it only at 1 % (0.70 % at 18.6×).
  * Our learned guarded rule is dominated at every gap by learning-free LtF (e.g. 0.63 % at 14.6× vs 0.35 % at 15.7×).
* **Best setting per rule:**
  * learning-free LtF: 0.5 % (no loss against 0.1 %, 2.3× faster);
  * LP-integral + guards: 0.5–1 %;
  * LtF-kNN: 1 %;
  * the guarded rule: 0.5 % (0.63 % at 14.6×; the gain over 0.1 % is mainly in the median, 3.2× → 7.2×).

**12 hours** (60 instances; the two passes agree to a median 0.94–0.95 on the re-run loose-gap full MILPs, 73–75 %
within 10 %; the 12-hour pass was interrupted by a container restart after 9 instances and resumed in a new process,
pairing being per instance). Two fused learned + LP rules were added, both at all three gaps:

* **"LtF on the LP relaxation + guards":** the learning-free thresholds with our three guards applied at test. No
  tuning.
* **"Fused":** the hybrid's guard-aware LtF tuning (adequacy + row guards in the check, ε = 1 %, 180 validation
  instances, 34 min, converged) on a fused score, `otsl.base.lp_veto_score`. The score is the hybrid's error-cost
  score, except where the LP relaxation is integral and contradicts the learned rounding: those decisions are demoted
  to the least confident level of their direction.
  * The veto flags 5.0 decisions per validation instance. 65 % of them are actual GNN errors: 3.2 of its 14.7 wrong
    decisions per instance.
  * A literal max(learned confidence, LP integrality) would put 98 % of the scores at exactly 0 / 1. The learning-free
    LtF showed that the calibration can release such point masses only one whole unit at a time.

| rule | 0.1 % gap | 0.5 % gap | 1 % gap | vs full MILP at 0.5 %: Δ gap pp / Δ log speed-up |
|---|---|---|---|---|
| full MILP | 0.16 % · 1.0× | 0.20 % · 2.9× / 1.9× | 0.21 % · 4.2× / 2.4× | – |
| LtF kNN ε = 1 % | 0.41 % · 4.6× / 2.1× | 0.44 % · 10.3× / 4.2× | **0.48 % · 12.1× / 5.1×** | **+0.24 [+0.10, +0.42]** / **+0.79 [+0.49, +1.08]** |
| LtF on our BCE GNN | 0.28 % · 5.4× / 3.6× | 0.31 % · 8.0× / 5.3× | 0.34 % · 9.8× / 6.4× | +0.11 [−0.00, +0.23] / **+1.02 [+0.79, +1.26]** |
| hybrid | 0.24 % · 5.2× / 2.9× | 0.28 % · 8.6× / 4.6× | 0.32 % · 9.4× / 5.2× | +0.08 [−0.01, +0.17] / **+0.89 [+0.67, +1.11]** |
| ours: error-cost + adequacy 90 % | 0.42 % · 4.2× / 3.3× | 0.45 % · 6.9× / 4.7× | 0.50 % · 10.0× / 6.3× | **+0.25 [+0.13, +0.38]** / **+0.91 [+0.72, +1.09]** |
| LtF on the LP relaxation | 0.23 % · 2.0× / 1.7× | 0.27 % · 5.3× / 3.6× | 0.33 % · 7.2× / 4.5× | +0.07 [−0.02, +0.20] / **+0.64 [+0.46, +0.82]** |
| LtF on the LP relaxation + guards | 0.23 % · 2.0× / 1.7× | 0.27 % · 4.8× / 3.4× | 0.33 % · 6.4× / 4.2× | +0.07 [−0.02, +0.20] / **+0.57 [+0.40, +0.75]** |
| **fused (error-cost + LP veto)** | **0.21 % · 5.6× / 3.2×** | **0.25 % · 9.2× / 4.9×** | **0.28 % · 10.7× / 6.0×** | +0.05 [−0.02, +0.13] / **+0.95 [+0.73, +1.17]** |
| LP-integral + guards | 1.10 % · 11.0× / 8.5× | **1.12 % · 13.2× / 9.7×** | **1.13 % · 14.9× / 10.6×** | **+0.92 [+0.68, +1.15]** / **+1.63 [+1.43, +1.82]** |

*(bold cells: on the Pareto frontier of mean gap vs mean speed-up; the full MILP at all three gaps is on it too)*

* **Speed-ups stack on 12 h too.**
  * At the same loose gap every fixing rule is 1.7–5× faster than the full MILP (geometric mean, all significant).
  * The hybrid, LtF on the BCE GNN, the fused rule and learning-free LtF do so at a gap that cannot be told apart from
    the full MILP's at 0.5 % (+0.05 to +0.11 pp, CIs include 0).
  * Learned rules: the hybrid at 0.5 % gives 0.28 % at 8.6× against the full MILP's 0.20 % at 2.9×; the fused rule
    gives 0.25 % at 9.2× (2.6× faster than the full MILP at the same gap).
  * The loose gap alone is not the frontier: the fixing rules move it to roughly 2.5× the speed at +0.05–0.1 pp.
* **The fused rule is the most accurate fixing rule at every gap and on the frontier at all three.**
  * Against the hybrid: −0.03 pp [−0.08, +0.02] (0.5 %) and −0.04 [−0.11, +0.04] (1 %) at the same speed
    (+0.06 [−0.06, +0.17] / +0.13 [0.00, +0.27] log). The gain over the hybrid is within noise.
  * Against learning-free LtF it is 1.3–1.4× faster (+0.31 [+0.08, +0.52] log at 0.5 %).
  * At 0.1 % it reaches 0.21 % at 5.6× (hybrid 0.24 % at 5.2×; cross-pass).
  * Applying the guards to learning-free LtF at test changes nothing: no fixing is released, the guard LPs only add
    time.
* **Best setting per rule on 12 h:**
  * full MILP: 1 % (0.21 % at 4.2×);
  * fused and hybrid: 0.5 % (+0.04 pp for +65 % mean / +55 % geometric-mean speed against 0.1 %);
  * LtF-BCE and learning-free LtF: 0.5–1 %;
  * LtF-kNN: 1 %;
  * LP-integral: 1 %.
* **Learning versus no learning, with stacking.** On 12 h the learned rules keep their edge once both sides use a
  loose gap.
  * At 0.5 %: hybrid 0.28 % at 8.6× (geometric mean 4.6×), fused 0.25 % at 9.2× (4.9×), against learning-free LtF
    0.27 % at 5.3× (3.6×).
  * Fused vs learning-free LtF: level in gap, 1.36× faster.
  * On 24 h learning-free LtF stays the best accurate rule after stacking (0.35 % at 15.7×), ahead of every learned
    rule tested there.

## 5. Verdict

* **Is RTS-GMLC as easy for the LP relaxation as California? Partly.** The relaxation is 6–9× looser at the median
  (0.19 / 0.26 % vs 0.03 %) and its integral values hold 7–12 wrong decisions per instance. Fixing them blindly costs
  ~1 % (12 h: 1.06 %, 24 h: 0.76–0.88 %), against 0.105 % on California. RTS-GMLC is therefore a harder benchmark for
  relaxation-based fixing, and the naive no-learning rule does not match the learned rules' accuracy.
* **Does learning beat the no-learning baselines? On 12 h only narrowly, on 24 h no.** The relaxation becomes
  competitive once it is calibrated by Learning to Fix's joint ε-tuning:
  * on 12 h it equals the hybrid's accuracy, though more slowly;
  * on 24 h it beats the paper's kNN version on both axes and our guarded rule on accuracy.
  The kNN classifier of the paper never beats its learning-free counterpart. The learned gains that survive:
  * on 12 h, speed at equal accuracy (hybrid and LtF on our BCE GNN, 1.3–1.6× faster than the 1 %-gap MILP; only the
    BCE version significantly);
  * a 0.2–0.3 pp accuracy edge at ~13× (borderline).
  On 24 h LP-integral fixing with guards is level in gap with our label-free guarded rule at about 2.2× its speed
  (geometric mean), and learning-free LtF is more accurate than both.
* **Would a practitioner just loosen the solver? Yes, it is a baseline every method must beat.** A 1 % gap on 12 h
  (0.21 % at 4.0×) and a 0.5 % gap on 24 h (0.28 % at 4.9×) match or beat the paper's LtF-kNN in gap at equal speed.
  On 12 h the hybrid is level with the 1 %-gap MILP in gap and not significantly faster; only LtF on our BCE GNN is
  significantly faster at equal gap. On 24 h our guarded rule is level with the 1 %-gap MILP on both axes, and 2.2×
  faster (geometric mean) than the 0.5 %-gap MILP but 0.33 pp less accurate. Loosening to 1 % on 24 h lets the MILP
  accept reserve shortfall on 20 % of the instances (served 80 %), so 0.5 % is the practical setting there.
* **Same budget.** Stopping the 0.1 % MILP at a method's time is not the fair cheap alternative. The 12-hour MILP finds
  its good incumbent only at the end of the root, so a time-limited run has no solution or a poor one, and every
  method beats it. Time-to-quality speed-ups of the MILP-based rules are 1.2–2.3×; only the LP-only pipelines reach 5–10×.
* **Do fixing and a loose gap stack? Yes, on both benchmarks** (§4.4). Every fixing rule keeps a 1.7–6× speed-up over
  the full MILP at the same loose gap. The accurate ones (the hybrid, the fused rule and LtF on our BCE GNN on 12 h;
  learning-free LtF on both) pay at most ~0.1 pp for it, not significant at 0.5 %. The practical frontier is
  therefore "fixing rule + 0.5 % gap":
  * 12 h: fused rule 0.25 % at 9.2× (hybrid 0.28 % at 8.6×);
  * 24 h: learning-free LtF 0.35 % at 15.7×.
  Learning keeps its edge on 12 h (the fused rule is 1.36× faster than learning-free LtF at equal gap). It does not on
  24 h, where no learned rule tested reaches learning-free LtF's accuracy at any gap. The LP veto, a cheap
  learned + LP fusion, is the most accurate rule on 12 h at every gap, but its gain over the hybrid (−0.03 pp) is
  within noise.
* **What this means for the study's claims** (RESEARCH.md §6):
  * "our models make Learning to Fix better" holds against the kNN, not against the LP relaxation itself (12 h);
  * on 24 h the learning-free calibration matches or beats every learned rule tested;
  * the full MILP with a loose gap should be reported next to every speed-up, alongside time-to-quality.

## 6. Caveats

* **One core of a shared machine.** Four agents ran on the four cores (1-minute load ≈ 4.3); this study used core 1
  only and nothing else ran on it during the test passes. Every comparison is paired within an instance; the stored
  reference runs were slower by a constant factor (1.36–1.41 on 12 h for the full MILP and every re-run rule), so the
  re-timed speed-ups match the published ones.
* **Solver time** is `Highs.run()`. The stored 12-hour times came from `scipy.optimize.milp`, which also counts ~0.3–0.5 s
  of model set-up per call; excluding it here affects the full MILP and every method alike.
* **Reference dual bound** = the stored reference run's bound (the same DB as the published tables), not this run's;
  on time-limited instances the two differ slightly (`timing.own_db_minus_ref_db_mean_pct` in the JSON).
* **Same budget** uses the incumbent log of the re-timed 0.1 % run instead of separate time-limited re-solves (HiGHS
  is deterministic; a run stopped at time t has the incumbent the log shows at t, up to timing noise).
* **Loose-gap runs** keep the reference time limit (60 s / 300 s) and one thread; HiGHS prunes with the gap, so a
  loose-gap run follows a different search than the 0.1 % run after the root.
* **Small samples**: 60 (12 h) and 20 (24 h, half of the test set, chosen as the first 20 before any run) test
  instances; validation 60 / 180 (12 h) and 40 (24 h); one seed for every learned reference rule here.
* **Learning-free LtF** inherits the compute caps of the faithful runs (relaxation MILPs 6 s / 8 s, K_max = 10), so its
  thresholds are as conservative as theirs.
* **Inference times** of the GNN-based reference rules are the stored per-instance values (milliseconds), not re-timed.

## 7. Test suite

`taskset -c 1 python -m pytest` runs 53 tests (8 earlier + 45 new) in **18 s** on an otherwise idle core (36 s while a
tuning job shared it). Deterministic: fixed seeds, single-threaded solvers, small instances (3-hour RTS-GMLC
MILPs, 12-hour LPs, a synthetic 4-unit PGLib fleet).

| file | what it checks |
|---|---|
| `tests/conftest.py` | shared fixtures: RTS-GMLC, a 3-hour UC model with its MILP (highspy) and LP relaxation, a 12-hour model |
| `tests/test_ltfx.py` (12) | eq. (5) masks (strict, generator-specific, tolerance), `fix_dict`, constant and worst-case thresholds (never fix a validation decision wrongly; midpoint rule; never-on generators); Appendix A quantile bins; master: collapses intervals without cuts, honours single and OR-cuts, cuts only worsen its optimum; **Algorithm 1 + 2 on a solver-free toy oracle** (terminates with thresholds that release every harmful fixing); the monotone check cache; the check and relaxation MILPs on a 3-hour UC (a failing fixing is repaired by the release set, no-goods exclude it); Table II features and the inverse-distance kNN (eq. 13) |
| `tests/test_hybrid.py` (12) | error-cost score transform (rounding direction, monotone in the error cost, lo = hi = 0.5 rounds), `harm_norm`, mask ↔ fixing round trip, cut JSON round trip, ranking; adequacy guard (capacity restored every hour, minimal in merit order, identical copy in `uc24ltf`), min up/down row release (only infeasible rows), LP-relaxation guard (removes penalised slack, leaves LP-integral fixings alone), guard order and info, `HybridTuner` (guarded check, memoisation, cut log), `uc24ltf.guard_fix` (soft LP guard is a no-op on an infeasible relaxation; conflict release first) |
| `tests/test_b3.py` (6) | `time_to_reach` (first crossing, tolerance, never), incumbent packing (first k−1 + last), repairs (min up/down feasible, fixed point, block repair covers net load + reserve), `solve_milp_hs` = scipy path on the same model, improving incumbent log, dispatch LP reproduces the MILP, fixings respected, contradictory fixings infeasible |
| `tests/test_pglib.py` (5) | synthetic 4-unit PGLib fleet (must-run unit, hot / cold start-up categories, binding ramps, wind): free-unit view, **MILP = brute force over all 512 commitments**, LP relaxation ≤ MILP, dispatch LP = MILP, continuous = binary start-up categories, relaxed reduced LP at integral relaxation values = relaxation, reduced-MILP fixings, initial min-down rule, repairs, `time_to_reach` |
| `tests/test_metrics.py` (10) | paper metrics and bootstraps wherever they live — `otsl.base` (gap to DB, feasibility filtering, mean of per-instance speed-ups vs ratio of means, bootstrap / paired CIs, incumbent-log helpers), `scripts/uc_hybrid_report.py` (`per_instance` on synthetic records incl. the fallback convention, `stats`, `paired`, `boot`), `scripts/uc_papereval_score.py` (`stats` incl. overheads), `scripts/uc_uc24ltf_report.py` (`first_below`, `boot_ci`), `scripts/uc_pglib_report.py` (`tq`, `hard_ok`, `paired`); the baselines of `otsl.base` (tolerances, repaired roundings, screening, solver record) and the LP-veto fused score |

**Bugs found: none** in the tested modules (no behaviour was changed; every test passed against the code as it is).
Behaviours worth knowing, documented in the tests rather than changed: `fixpolicy.release_conflicting_rows` releases
the conflicting unit's *whole* row, including fixings outside the conflict; `ltfx.fix_dict` lets ON win if a pair of
thresholds with lo > hi is passed (never produced by the master, which enforces lo ≤ hi); `hybrid.harm_to_score` maps
p = 0.5 to the OFF side, consistent with `rule_fixings` (yhat = p > 0.5).

## 8. Reproducibility audit of RESEARCH.md (TL;DR and §6 X5–X8)

`scripts/uc_base_audit.py` (output `results/uc12/base_audit.json`) recomputes 148 headline numbers from the saved results (summary JSON / MD, and for X6–X8
also from the per-instance records with the documented definitions; `--lp 1` re-prices the stored MILP schedules of
B1 / B2 for the MILP's served shares). A number "matches" when the recomputed value rounds to the stated one.

**144 match, 4 do not** (all small):

| where | stated | recomputed (source) | comment |
|---|---|---|---|
| TL;DR item 3 | B2 one-shot pipeline serves "10–17 %" | 8.3–16.7 % (`results/uc12/uc_model1_results.md`) | the GNN + symmetry + canonical-label variant serves 8.3 % (V2 table); the B1 range "59–74 %" includes the same variant, so the B2 range should read 8–17 % |
| §6 X5, row "24 h: guarded rule, 95 % target" | 11.6× | 11.55× (`results/papereval_results.json`) → 11.5× | 11.6× is the load-corrected re-run of uc24ltf (11.61×, X8); the papereval record X5 cites gives 11.5× (as `methods/papereval.md` says) |
| §6 X6, hybrid − LtF-kNN | −0.16 pp [−0.33, **−0.03**] | [−0.333, −0.0246] (`results/uc12/hybrid_results.json`) → [−0.33, −0.02] | rounding of the CI's upper end (also in `methods/hybrid.md`); still excludes 0 |
| §6 X7, "ours: learned end-to-end (5 LPs)" | 23.5× | 23.59× (`results/pglib/pglib_results.json`) → 23.6× | the label-free source (0.65 %); `methods/pglib.md`'s table has 23.6×, its summary 23.5× |

Notes (match, but the wording is looser than the data): X7 / TL;DR 12 "the LP relaxation is within 0.03 % of the
optimum" is the validation *median* (0.029 %); mean 0.079 %, max 0.41 % (test: median 0.038 %, mean 0.059 %, max
0.41 %). X5 "the paper's own baselines are 1.7–4.6× worse in gap" holds for the constant-threshold kNN and the
cost-ranked kNN (1.66–4.58×); the hard-threshold kNN is better here than in the paper (ratio 0.80). Outside the
audited scope: §3 U2 says the B1 LP relaxation is "1.3 % below the MILP on average"; the dataset summary
(`data/generated/uc1/test.json`, and the LP-relaxation row of `results/uc1/uc_model1_results.md`) gives 1.76 %.

## Status

* First study **complete** (2026-10-06 22:10 UTC): validation, learning-free LtF tuning, test passes, reports, audit,
  tests (see the reproduction order below); 53 tests pass.
* **Stacking follow-up complete** (2026-10-07 02:45 UTC, core 1):
  * 24-hour stacking pass, 20/20 instances: `uc_base_eval.py --bench uc24 --idx 0-19 --full_ref 0 --gaps 0.005,0.01
    --red_gaps 0.005,0.01 --rules "LtF kNN eps=1%;ours: guarded 95% (imitation GNN);LtF on LP relaxation eps=1%;LP-integral
    (tol 0.2) + guards" --out results/uc24/base_stack_test.jsonl`.
  * 12-hour fused-rule tuning: `uc_base_fuse.py --eps 0.01 --budget_min 60` (converged, 83.2 % fixed after the guards,
    34 min).
  * 12-hour stacking pass, 60/60 instances: `uc_base_eval.py --bench uc12 --idx 0-59 --highs_path <highspy 1.12>
    --full_ref 0 --gaps 0.005,0.01 --red_gaps 0.005,0.01 --lp_guards 1 --fused results/uc12/base_tune_fused_1.json --rules
    "LtF kNN eps=1%;LtF BCE GNN eps=1%;hybrid (he_bce_s0_e1_n360);ours: error-cost + adequacy 90%;LtF on LP relaxation
    eps=1%;LP-integral (tol 0.05) + guards;LtF on LP relaxation eps=1% + guards;fused: error-cost score + LP veto,
    guard-aware LtF eps=1%" --ref_gap_rules "<the last two>" --out results/uc12/base_stack_test.jsonl`. A container
    restart at 00:18 UTC interrupted it after 9 instances; it was resumed with the same command, which skips the
    instances already recorded. Pairing is per instance, so this does not affect the comparisons.
  * Reports: `uc_base_stack_report.py --bench uc24 / uc12`; run `uc_base_report.py` first if regenerating, because it
    rewrites `base_results.{md,json}`.
  * Logs: `results/<bench>/base_stack_run.log`, `results/uc12/base_fuse_run.log`.
* Reproduction order of the first study: `uc_base_val.py --bench uc12 / uc24` (+ `--select`), `uc_base_tune.py --bench
  uc12 / uc24 --eps 0.01`, `uc_base_eval.py --bench uc12 --idx 0-59 --highs_path <highspy 1.12>`, `uc_base_eval.py
  --bench uc24 --idx 0-19`, `uc_base_report.py --bench uc12 / uc24`, `uc_base_audit.py --lp 1 --out
  results/uc12/base_audit.json`; one core (`taskset -c 1`, `OTSL_THREADS=1`). Compute: uc12 validation 21 min + tuning
  23 min + test 2.4 h; uc24 validation 55 min + tuning 14 min + test 2.9 h.
* highspy 1.12 (= scipy 1.17's HiGHS) lives outside the repository: `pip install --no-deps --target <dir>
  highspy==1.12.0`, passed as `--highs_path <dir>`.
* Not done: the 20 remaining 24-hour test instances (compute), ε = 5 % for the learning-free LtF, separate
  time-limited re-solves for the same-budget comparison (the incumbent log is used instead).
