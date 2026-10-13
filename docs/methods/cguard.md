# A cost-aware check for fixed commitments (B2, 12-hour UC)

Code: [`otsl/cguard.py`](../../otsl/cguard.py), `scripts/uc_cguard_{common,diag,deep,run,shiftval,report}.py`.
Results: `results/uc12/cguard_results.{md,json}` (when written), diagnostics `results/uc12/cguard_diag_{test,val}.jsonl`.

## Status

*Kept current for resumption.* Started 2026-10-13 06:41 UTC on core 0 (`taskset -c 0`, HiGHS and torch one thread).

* Diagnosis (LPs only, `scripts/uc_cguard_diag.py --sets test / val`): done (300 test, 360 validation).
* Chain (`scripts/uc_cguard_run.py --stage val → select → retime → test`, each resumable; log
  `results/uc12/cguard_chain.log`): validation stage running since 06:51 UTC (~13 s per instance). Queued after it:
  `scripts/uc_cguard_deep.py` (failure anatomy), then the shifted-validation arm (`scripts/uc_cguard_shiftval.py --n 20`,
  `--select`, then `uc_cguard_run.py --stage testB`). Resume any stage by re-running it (all skip finished instances).
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

