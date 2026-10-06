# Closing the gap between hard fixing and exact solving: trust regions, warm starts and polishing (12 h and 24 h UC)

*Code: [`otsl/solver.py`](../../otsl/solver.py) (highspy MILP with fixings, extra rows, MIP start and incumbent trace;
trust region; complete starts; local branching; gradient release), [`scripts/uc_solver_eval.py`](../../scripts/uc_solver_eval.py)
(one process, one core, every variant of an instance back to back), [`scripts/uc_solver_report.py`](../../scripts/uc_solver_report.py)
(paper metrics, time-limit truncation, bootstrap). Results: `results/uc12/solver_results.{md,json}`,
`results/uc24/solver_results.{md,json}`; raw records `results/uc12/solver_*.jsonl`, `results/uc24/solver_*.jsonl`.*

## Status

(kept current; newest first)

* 2026-10-06 16:25 — uc12 validation done (12 `val` instances, 3 passes: `results/uc12/solver_val{,_b,_c}.jsonl`).
  **Test selection fixed (below, "uc12 selection") before any test instance was read.** Next: test_fresh 0–59
  (`results/uc12/solver_test_fresh.jsonl`, resumable), then uc24 validation (10 `uc24ltf_val` instances) and test.
* 2026-10-06 15:10 — module and evaluation script written; pilot on uc12 val instances 0–5 (scratch).

## Summary

(pending)

## Method

All variants use the probabilities and scores of the existing rules — no new model is trained. On uc12: the
MILP-label BCE GNN (`uc_model1_4.pt`), its learned error cost h[t, g] (harm ensemble seed 0, `combo_harm_s0.pt`), the
self-trained GNN of the combined pipeline, the paper's kNN; on uc24: the label-free imitation GNN (`b3_lf_bce.pt`)
and the kNN of [`uc24ltf.md`](uc24ltf.md). Guards are the existing ones (`otsl.combo.adequacy_guard`,
`otsl.fixpolicy.release_conflicting_rows`, `otsl.fixpolicy.lp_guard`).

**Solver path.** Every MILP — the full MILP, the references and all new variants — is the `otsl.uc.UCModel` MILP
passed to highspy 1.15 by `otsl.solver.solve_uc_hs` (same matrices and objective as `UCModel.solve_uc` and
`otsl.b3.solve_milp_hs`): one thread, `mip_rel_gap` 1e-3 (the setting of every stored reference), time limit 60 s
(uc12) / 300 s (uc24), extra rows and a MIP start when a variant needs them, and an incumbent trace from HiGHS's
improving-solution callback. The uc12 references were originally solved with scipy's HiGHS; here they are re-run
through highspy in the same process as everything else (see Setup).

**1. Trust region (Predict-and-Search).** Han et al. (ICLR 2023) and ConPaS (Huang et al., ICML 2024) do not fix the
predicted values; they pick the k0 binaries predicted most confidently at 0 and the k1 most confidently at 1 and add
one row allowing at most Δ of them to deviate:

  Σ_{(t,g)∈S0} u[t,g] + Σ_{(t,g)∈S1} (1 − u[t,g]) ≤ Δ,

then solve the full MILP with this row. Here S0 / S1 are the shares q0 / q1 of the predicted-OFF / predicted-ON
decisions with the lowest score (uc12: learned expected error cost h of the BCE GNN, the ranking of our error-cost
rule and the hybrid; uc24: error probability min(p, 1 − p) of the imitation GNN), and the set then passes through the
benchmark's guards before it enters the row (uc12: adequacy guard + min up/down row release, the hybrid's chain; uc24:
adequacy + LP-relaxation guard + row release, the guarded rule's chain), so an OFF set that would leave too little
capacity in some hour is never part of the trust region. Three variants share each set:

* `hard:q0:q1` — the set hard-fixed (bounds), the comparator "hard fixing at the same k";
* `pas:q0:q1:Δ` — the set as a trust region, nothing hard-fixed (Predict-and-Search proper);
* `core:<rule>:q0:q1:Δ` — the fixings of an existing rule (the hybrid on uc12, the guarded rule on uc24) hard-fixed and
  the trust region over the remaining decisions of the set ("fix the confident core, soft-fix the band").

**2. Warm starts.** A MIP start must be a complete primal point. From a commitment u the start is u, the implied
start-up / shut-down indicators v, w and the dispatch LP of u for every continuous variable (`otsl.solver.dispatch_x`;
always feasible, because shedding, over-generation and reserve shortfall are priced slacks). The schedule is the
learned one-shot decoder of the combined pipeline (BCE probabilities → threshold (0.5 on uc12, the validation-chosen 0.3
on uc24) → block adequacy repair → min up/down repair; one LP). For a reduced MILP the decoded schedule is first made
consistent with the rule's fixings (fixed entries forced; a row that then breaks min up/down is replaced by the
Hamming-nearest feasible row through its forced values, `otsl.fixpolicy.repair_row_forced`). Acceptance is checked
on every run: HiGHS reports an accepted start through the improving-solution callback with the start's objective as
the first incumbent; the share of accepted starts is reported. Measured against a cold start on the same instance:
time to proof (termination at the 0.1 % gap or the limit) and time until the incumbent is within 1 % / 0.5 % of the
cold full MILP's final cost (the start counts from the moment it is available).

* `warmfull:dec` — full MILP + decoded start;
* `warmred:<rule>` — the rule's reduced MILP + the consistent decoded start;
* `ftp:<rule>` — "fix, then prove": the full MILP warm-started with the reduced MILP's solution.

**3. Fix and polish.** After the reduced MILP of a rule (hybrid, or the fast combined 98 % rule on uc12; the guarded
rule on uc24), one improvement step on the full problem around its incumbent x̂, warm-started from x̂ and
time-limited:

* `ftp:<rule>:τ` — the full MILP from x̂, stopped at τ (the solver's own heuristics, RINS among them, around x̂);
* `lb:<rule>:r:τ` — local branching (Fischetti & Lodi 2003): the full MILP plus the Hamming ball
  Σ_{û=0} u + Σ_{û=1} (1 − u) ≤ r around the incumbent's commitment û;
* `grad:<rule>:m:τ` — gradient release: the exact derivative of the dispatch-LP cost with respect to u at û
  (`UCModel.dispatch_gradient`: the reduced costs of u, v, w in the dispatch LP combined by the envelope theorem)
  predicts the saving of flipping each decision; the m fixings with the largest positive predicted saving are
  released and the reduced MILP is re-solved from x̂;
* `rins:<rule>:τ` — RINS neighbourhood (Danna, Rothberg & Le Pape 2005) with the LP relaxation that is computed anyway
  as GNN input: every decision where û equals the LP relaxation is fixed, the rest is free (the LP relaxation is
  integral on ≈ 98 % of the decisions, so this re-opens the ≈ 2–5 % where the incumbent and the relaxation disagree).

The method time of a polish variant is the whole base pipeline plus the polish step. Anytime variants (trust region,
polish, full MILP) are run once with a time limit at least as large as any τ considered, and the result at a smaller τ
is read from the incumbent trace (the last improving solution before τ; HiGHS is deterministic, so this is what a run
with time limit τ returns, up to timing noise).

**No-learning reference** (`lpfix`): fix every decision that the LP relaxation sets integrally (≈ 98 %), release
min up/down-conflicting rows, solve the reduced MILP.

**Time accounting.** Method time = inference (GNN forward pass, error-cost features and ensemble, kNN search; per
instance, unbatched) + the LP relaxation that feeds the GNN (for GNN-based rules) + fixing rule + guards (incl. their
LPs) + decoding and start LP (warm starts) + model build + solver run; for polish variants also the base pipeline,
the gradient LP or the RINS neighbourhood. The full MILP's time is its model build + solver run.

## Setup

**Machine and timing.** One process pinned to core 0 (`taskset -c 0`) of a 4-core machine shared with three other
agents' jobs (1-minute load average 4–5), HiGHS single-threaded, PyTorch single-threaded (`OTSL_THREADS=1`), < 3 GB.
**Timing is paired by construction**: for every instance the full MILP is solved cold in the same process,
immediately before every reference rule and every new variant of that instance, so each speed-up compares two runs
taken under the same conditions minutes apart. The stored reference results (scipy's HiGHS on uc12, other load
levels) are therefore not reused; the references (faithful LtF with kNN and with the BCE GNN at ε = 1 %, the hybrid
of [`hybrid.md`](hybrid.md), our guarded error-cost rule at 90 %, the combined pipeline at 98 %; on uc24 LtF-kNN at
ε = 1 % and the guarded 95 % rule) are re-run through the same highspy path with their stored thresholds, scores and
guards. Their numbers here therefore differ slightly from the published ones (another HiGHS build, another load).
Reproducibility of the timing on one core: the same instance re-solved in two separate runs differed by ≤ 4 %
(val instance 0: full MILP 11.6 / 12.0 s, error-cost rule 5.8 / 6.1 s).

**Metrics** (paper's protocol, [`papereval.md`](papereval.md)): gap = (C − DB) / C with DB the dual bound of the
cold full MILP of the same process (0.1 % gap, 60 s / 300 s limit); speed-up = per-instance T_full / T_method, mean
and median over feasible instances, plus the ratio of mean times; feasibility rate; fixed share. Paired instance
bootstrap (10,000 resamples) for differences against the hybrid and against LtF-kNN (gap in pp; log speed-up).
Time to quality (TTQ q): time until a run holds a schedule within q = 1 % / 0.5 % of the cold full MILP's final cost.

**Validation (uc12).** The first 12 instances of `val` (full MILP 33.5 s mean here — harder than the first 60 of
`test_fresh`, ≈ 20 s), three passes (`results/uc12/solver_val{,_b,_c}.jsonl`): trust-region sets
(q0, q1) ∈ {(0.97, 0.9), (1, 1)} with Δ ∈ {10, 20} (30 s limit), core + band with Δ ∈ {10, 20}, polish around the
hybrid and the combined 98 % rule (warm full MILP, local branching r ∈ {10, 30}, gradient release m ∈ {20, 60, 120},
RINS; 10 s limit, cut post hoc at τ ∈ {1, 2, 3, 5, 10} s), the warm-started full and reduced MILPs, and the
LP-integral reference. 12 instances only, because one instance with all variants costs ≈ 5 min of the single core.

| validation (12 `val` instances) | gap mean % | speed-up mean |
|---|---|---|
| hybrid / error-cost 90 % / combined 98 % (hard fixing) | 0.170 / 0.450 / 0.872 | 3.74 / 4.33 / 8.49 |
| Predict-and-Search, (0.97, 0.9), Δ = 10 / (1, 1), Δ = 20 (30 s limit) | 0.246 / 0.370 | 1.20 / 1.22 |
| hard fixing of the (0.97, 0.9) set (same k) | 5.14 (max 22.6) | 24.0 |
| core (hybrid) + band trust region, Δ = 10 / 20 | 0.174 / 0.170 | 3.31 / 3.49 |
| polish of the hybrid at τ = 1 / 10 s: RINS; local branching r = 10 | 0.162 / 0.157; 0.170 / 0.115 | 3.12 / 2.77; 2.43 / 1.53 |
| polish of combined 98 % at τ = 3 / 10 s: RINS | 0.481 / 0.466 | 5.48 / 4.98 |
| polish of combined 98 % at τ = 5 / 10 s: gradient release m = 60 | 0.655 / 0.388 | 4.51 / 3.94 |
| polish of combined 98 % at τ = 10 s: warm full MILP; local branching r = 10 / 30 | 0.631; 0.733 / 0.791 | 2.30; 2.30 / 2.29 |
| no learning: fix the LP-integral decisions | 2.147 | 11.43 |
| warm-started full MILP (decoded start), to proof / cut at τ = 20 s; cold full MILP cut at 20 s | 0.096 / 0.243; 5.06 | 0.94 / 1.72; 1.77 |
| warm-started reduced MILP (decoded start), hybrid / error-cost 90 %, cut at τ = 10 s | 0.352 / 0.482 | 4.97 / 5.24 |

**uc12 selection (fixed before the test run).**
* Trust region: `pas:0.97:0.9:10` (lower validation gap than (1, 1), Δ = 20 at the same speed), with the 30 s
  limit used on validation; its comparators `hard:0.97:0.9` (same set hard-fixed) and `core:hybrid:0.97:0.9:10`.
* Polish of combined 98 %: RINS with τ = 3 s (the best validation gap among polish points ≥ 5×) and gradient release
  m = 60 with τ = 10 s (the lowest validation gap of all polish points); local branching r = 10 is also run (curve
  only). Polish of the hybrid: RINS τ = 1 s and local branching r = 10, τ = 10 s (the only polish points that
  improved the hybrid on validation). Gradient release on the hybrid and warm-full-MILP polish of combined 98 % were
  dominated on validation and are not run on test.
* Warm starts (no parameter): `warmfull:dec`, `warmred:{hybrid, ec90, comb98}`, `ftp:hybrid` (full MILP from the
  hybrid's solution, 60 s, to proof).
* Warm-start headline points (validation pass `_c`, fixed 16:35 before reading test output): to proof, and cut at
  τ = 10 s (warm-started reduced MILPs; validation: hybrid 0.352 % at 4.97× vs 5.35 % cold at τ = 10 s, error-cost 90 %
  0.482 % at 5.24×) and τ = 20 s (warm-started full MILP; validation 0.243 % vs 5.06 % cold), each paired with the
  cold run cut at the same τ. Validation: starts accepted on 100 % of the runs; time to proof warm vs cold — full
  MILP 36.0 / 33.5 s, hybrid 23.4 / 20.8 s, error-cost 90 % 13.0 / 11.0 s, combined 98 % 8.0 / 5.8 s (no gain).
* Every polish / anytime run is reported along its whole τ curve as well; headline points are the τ above.

## Results

(pending)

## Verdict

(pending)

## Caveats

(pending)
