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

*(pending)*

## 5. Verdict

*(pending)*

## 6. Caveats

*(pending)*

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
* Running since 15:30 UTC (one queue, core 1, in this order): `uc_base_eval.py --bench uc12 --idx 0-59 --highs_path
  <highspy 1.12>` → `uc_base_report.py --bench uc12` → `uc_base_val.py --bench uc24` (+ `--select`) →
  `uc_base_tune.py --bench uc24 --eps 0.01 --budget_min 40` → `uc_base_eval.py --bench uc24 --idx 0-19` →
  `uc_base_report.py --bench uc24`. Logs `results/<bench>/base_eval_run.log`, `base_val.log`, `base_tune_run.log`.
  Resume after a restart: rerun the remaining commands in that order; eval and val are resumable per instance
  (records `base_eval_test.jsonl`, `base_val.jsonl`); a tuning run is not (rerun it whole).
* highspy 1.12 (= scipy 1.17's HiGHS) lives outside the repository: `pip install --no-deps --target <dir>
  highspy==1.12.0`, passed as `--highs_path <dir>`.
* Tests and audit: done (52 passed; 148 checks, 4 mismatches).
