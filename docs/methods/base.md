# No-learning baselines and fair solver budgets on networked RTS-GMLC (12 h and 24 h)

*Code: [`otsl/base.py`](../../otsl/base.py) (LP-integral fixings, LP rounding + repair + LP screening, solver wrapper,
incumbent-log helpers, paper metrics and bootstrap), `scripts/uc_base_{val,tune,eval,report,audit}.py`. Tests:
`tests/test_{ltfx,hybrid,b3,pglib,metrics}.py` (+ `tests/conftest.py`). Results:
[`results/uc12/base_results.md`](../../results/uc12/base_results.md), [`results/uc24/base_results.md`](../../results/uc24/base_results.md)
(+ `.json`), raw records `results/<bench>/base_eval_test.jsonl`, validation `results/<bench>/base_val.jsonl`,
`base_val_select.json`, tuning `results/<bench>/base_tune_lp_1.{json,log}`, run logs `results/<bench>/base_*.log`.*

## Summary

*(filled in when the test runs finish; see Status)*

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
  baselines — costs ~7 min per instance on one core). Validation: 12 h `val` (60) for the tolerance, `val` +
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
  (12 h) / LtF-kNN and the guarded 95 % rule (24 h).

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

### 4.3 24 hours

*(pending)*

## 5. Verdict

*(pending)*

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

`taskset -c 1 python -m pytest` runs 52 tests (8 earlier + 44 new) in **36 s** (measured while a tuning job shared the
core; about half that alone). Deterministic: fixed seeds, single-threaded solvers, small instances (3-hour RTS-GMLC
MILPs, 12-hour LPs, a synthetic 4-unit PGLib fleet).

| file | what it checks |
|---|---|
| `tests/conftest.py` | shared fixtures: RTS-GMLC, a 3-hour UC model with its MILP (highspy) and LP relaxation, a 12-hour model |
| `tests/test_ltfx.py` (12) | eq. (5) masks (strict, generator-specific, tolerance), `fix_dict`, constant and worst-case thresholds (never fix a validation decision wrongly; midpoint rule; never-on generators); Appendix A quantile bins; master: collapses intervals without cuts, honours single and OR-cuts, cuts only worsen its optimum; **Algorithm 1 + 2 on a solver-free toy oracle** (terminates with thresholds that release every harmful fixing); the monotone check cache; the check and relaxation MILPs on a 3-hour UC (a failing fixing is repaired by the release set, no-goods exclude it); Table II features and the inverse-distance kNN (eq. 13) |
| `tests/test_hybrid.py` (12) | error-cost score transform (rounding direction, monotone in the error cost, lo = hi = 0.5 rounds), `harm_norm`, mask ↔ fixing round trip, cut JSON round trip, ranking; adequacy guard (capacity restored every hour, minimal in merit order, identical copy in `uc24ltf`), min up/down row release (only infeasible rows), LP-relaxation guard (removes penalised slack, leaves LP-integral fixings alone), guard order and info, `HybridTuner` (guarded check, memoisation, cut log), `uc24ltf.guard_fix` (soft LP guard is a no-op on an infeasible relaxation; conflict release first) |
| `tests/test_b3.py` (6) | `time_to_reach` (first crossing, tolerance, never), incumbent packing (first k−1 + last), repairs (min up/down feasible, fixed point, block repair covers net load + reserve), `solve_milp_hs` = scipy path on the same model, improving incumbent log, dispatch LP reproduces the MILP, fixings respected, contradictory fixings infeasible |
| `tests/test_pglib.py` (5) | synthetic 4-unit PGLib fleet (must-run unit, hot / cold start-up categories, binding ramps, wind): free-unit view, **MILP = brute force over all 512 commitments**, LP relaxation ≤ MILP, dispatch LP = MILP, continuous = binary start-up categories, relaxed reduced LP at integral relaxation values = relaxation, reduced-MILP fixings, initial min-down rule, repairs, `time_to_reach` |
| `tests/test_metrics.py` (9) | paper metrics and bootstraps wherever they live — `otsl.base` (gap to DB, feasibility filtering, mean of per-instance speed-ups vs ratio of means, bootstrap / paired CIs, incumbent-log helpers), `scripts/uc_hybrid_report.py` (`per_instance` on synthetic records incl. the fallback convention, `stats`, `paired`, `boot`), `scripts/uc_papereval_score.py` (`stats` incl. overheads), `scripts/uc_uc24ltf_report.py` (`first_below`, `boot_ci`), `scripts/uc_pglib_report.py` (`tq`, `hard_ok`, `paired`); the baselines of `otsl.base` (tolerances, repaired roundings, screening, solver record) |

**Bugs found: none** in the tested modules (no behaviour was changed; every test passed against the code as it is).
Behaviours worth knowing, documented in the tests rather than changed: `fixpolicy.release_conflicting_rows` releases
the conflicting unit's *whole* row, including fixings outside the conflict; `ltfx.fix_dict` lets ON win if a pair of
thresholds with lo > hi is passed (never produced by the master, which enforces lo ≤ hi); `hybrid.harm_to_score` maps
p = 0.5 to the OFF side, consistent with `rule_fixings` (yhat = p > 0.5).

## 8. Reproducibility audit of RESEARCH.md (TL;DR and §6 X5–X8)

`scripts/uc_base_audit.py` recomputes 148 headline numbers from the saved results (summary JSON / MD, and for X6–X8
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

* Done: uc12 validation (`uc_base_val.py --bench uc12` + `--select`: tol 1e-6 without guards, 0.05 with guards) and
  uc12 learning-free LtF tuning (`uc_base_tune.py --bench uc12 --eps 0.01`: converged, 55.2 % fixed on validation,
  23 min).
* Done (core 1, in this order): `uc_base_eval.py --bench uc12 --idx 0-59 --highs_path <highspy 1.12>` →
  `uc_base_report.py --bench uc12` → `uc_base_val.py --bench uc24` (+ `--select`: tol 1e-6 / 0.2 with guards) →
  `uc_base_tune.py --bench uc24 --eps 0.01 --budget_min 40` (converged, 80.5 % fixed).
* Running since 19:15 UTC: `uc_base_eval.py --bench uc24 --idx 0-19` → `uc_base_report.py --bench uc24` (log
  `results/uc24/base_eval_run.log`; resumable per instance: rerun the same command; records
  `results/uc24/base_eval_test.jsonl`).
* highspy 1.12 (= scipy 1.17's HiGHS) lives outside the repository: `pip install --no-deps --target <dir>
  highspy==1.12.0`, passed as `--highs_path <dir>`.
* Tests and audit: done (52 passed; 148 checks, 4 mismatches).
