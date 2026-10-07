# Closing the gap between hard fixing and exact solving: trust regions, warm starts and polishing (12 h and 24 h UC)

*Code: [`otsl/solver.py`](../../otsl/solver.py) (highspy MILP with fixings, extra rows, MIP start and incumbent trace;
trust region; complete starts; local branching; gradient release), [`scripts/uc_solver_eval.py`](../../scripts/uc_solver_eval.py)
(one process, one core, every variant of an instance back to back), [`scripts/uc_solver_report.py`](../../scripts/uc_solver_report.py)
(paper metrics, time-limit truncation, bootstrap). Results: `results/uc12/solver_results.{md,json}`,
`results/uc24/solver_results.{md,json}`; raw records `results/uc12/solver_*.jsonl`, `results/uc24/solver_*.jsonl`.*

## Status

(kept current; newest first)

* 2026-10-06 20:15 — **uc12 test done** (60 instances, `results/uc12/solver_test_fresh.jsonl`; report
  `results/uc12/solver_results.{md,json}` + Pareto figure, rebuilt with
  `python3 scripts/uc_solver_report.py --spec results/uc12/solver_selection.json`). uc24 validation running
  (`results/uc24/solver_val.jsonl`, instances 30–35 of `uc24ltf_val`). Next: uc24 selection (written here before the
  test), uc24 test (`results/uc24/solver_test.jsonl`), then, if time allows, test_fresh 60–119 with the headline variants.
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

**uc24.** Base rule for the core / polish / warm-start variants: the label-free guarded rule at 95 % target
(`95%|bce_g` of [`b3.md`](b3.md): imitation-GNN confidence ranking, adequacy guard, LP-relaxation guard, min up/down
conflict release); reference LtF-kNN at ε = 1 % ([`uc24ltf.md`](uc24ltf.md)). There is no error-cost model on uc24
(it needs MILP-priced errors), so trust-region sets are ranked by the imitation GNN's error probability and pass
through the guarded rule's guard chain. Validation: instances 30–35 of `uc24ltf_val` (six of the 10 validation instances
added for Learning to Fix; instances 0–5 were used to select the guarded rule), with the stored full MILP of that file
as the reference (highspy, one thread, 300 s, the same solver path; not re-solved, to save ≈ 2 min of solver time per
instance). Local branching and warm-full-MILP polish were not carried to uc24: on uc12 validation they were dominated
by RINS / gradient release, and on 24 hours a full-size polish step costs at least one full-size root solve. Nor
was core + band: on uc12 validation it matched the hybrid's gap (0.174 / 0.170 % vs 0.170 %) at a lower speed-up
(3.3–3.5× vs 3.7×).

| validation (6 `uc24ltf_val` instances; stored full MILP 105 s mean) | gap mean % | speed-up mean |
|---|---|---|
| guarded 95 % rule (hard fixing) | 0.798 | 8.07 |
| … + decoded start | 0.805 | 7.82 |
| … + RINS, τ = 5 / 10 / 20 / 60 s | 0.753 / 0.725 / 0.610 / 0.593 | 5.47 / 4.33 / 3.55 / 3.43 |
| … + gradient release m = 100, τ = 20 / 60 s | 0.742 / 0.742 | 3.39 / 2.11 |
| … + gradient release m = 250, τ = 20 / 60 s | 0.615 / 0.490 | 3.16 / 1.97 |
| (q0, q1) = (0.97, 0.9) set hard-fixed | 1.173 | 12.77 |
| Predict-and-Search on that set, Δ = 20 (150 s limit); cut at 30 / 60 s | 0.142; 7.92 / 0.235 | 0.89; 3.26 / 1.70 |
| full MILP cut at 30 / 60 s (stored incumbent log) | 0.281 / 0.181 | 3.50 / 1.83 |

**uc24 selection (fixed 20:40, before the test run).** Same rules as on uc12: RINS at the smallest τ within 0.02 pp
of its best validation gap (τ = 20 s); gradient release at its lowest validation gap (m = 250, τ = 60 s);
Predict-and-Search with (0.97, 0.9), Δ = 20 and its 150 s limit; warm starts without parameters. Test pass A (all 40
instances, `results/uc24/solver_test.jsonl`): full MILP (300 s), LtF-kNN ε = 1 %, guarded 95 %, LP-integral fixing,
guarded 95 % + decoded start, + RINS (60 s limit, headline τ = 20 s), + gradient release m = 250 (60 s, headline
τ = 60 s). Pass B (first 12 instances only, for compute; `results/uc24/solver_test_b.jsonl`): full MILP + decoded
start (300 s), full MILP from the guarded rule's solution (300 s), the (0.97, 0.9) set hard-fixed, and
Predict-and-Search.

## Results

### uc12: first 60 instances of `test_fresh` (paired, one core)

Full tables: [`results/uc12/solver_results.md`](../../results/uc12/solver_results.md) (+ `.json`, Pareto figure
`solver_results_pareto.png`), raw records `results/uc12/solver_test_fresh.jsonl`. Full MILP (cold, same process):
26.6 s mean, 7 of 60 at the 60 s limit, 0.221 % mean gap to its own bound (max 4.3 %). Paper metrics: gap to the full
MILP's dual bound over feasible instances; speed-up = mean (median) of per-instance ratios incl. every overhead.

| method | feasible % | gap mean % [95 % CI] | gap max % | speed-up mean (median) | fixed % |
|---|---|---|---|---|---|
| *references, re-run in the same process* | | | | | |
| faithful LtF, kNN, ε = 1 % (paper's setting) | 100 | 0.424 [0.30, 0.58] | 3.59 | 3.58 (1.66) | 68 |
| faithful LtF, BCE GNN, ε = 1 % | 98.3 | 0.298 [0.19, 0.43] | 2.41 | 5.39 (2.50) | 84 |
| **hybrid** (guard-aware LtF on error-cost scores) | 100 | **0.263** [0.19, 0.35] | 1.80 | **5.74** (2.23) | 86 |
| ours: error-cost + adequacy guard, 90 % | 100 | 0.441 [0.25, 0.67] | 4.36 | 4.62 (3.16) | 90 |
| ours: combined pipeline, 98 % | 100 | 0.840 [0.57, 1.15] | 6.28 | 13.52 (7.94) | 92 |
| no learning: fix the LP-integral decisions | 100 | 1.080 [0.80, 1.39] | 6.74 | 12.44 (10.04) | 98 |
| *1. trust region* | | | | | |
| Predict-and-Search, (q0, q1) = (0.97, 0.9), Δ = 10, 30 s limit | 100 | 0.250 [0.14, 0.39] | 3.03 | 1.28 (1.16) | 0 (95 in the row) |
| the same set hard-fixed | 100 | 3.552 [1.19, 6.55] | 68.96 | 12.60 (5.69) | 95 |
| hybrid fixings hard + trust region over the rest of the set, Δ = 10 | 100 | 0.312 [0.20, 0.45] | 3.22 | 5.25 (2.23) | 86 |
| *2. warm starts* | | | | | |
| full MILP, decoded start, to proof | 100 | 0.154 [0.08, 0.26] | 2.65 | 1.17 (0.99) | – |
| full MILP, decoded start / cold, both cut at 20 s | 100 | 0.257 / 0.339 | 3.05 / 5.85 | 1.69 / 1.56 | – |
| hybrid reduced MILP, decoded start, to proof | 100 | 0.268 | 1.82 | 5.73 (2.20) | 86 |
| hybrid reduced MILP, decoded start / cold, both cut at 10 s | 100 | 0.478 / 0.333 | 9.45 / 3.15 | 6.74 / 6.75 | 86 |
| fix, then prove: full MILP from the hybrid's solution (60 s) | 100 | 0.133 [0.08, 0.19] | 1.42 | 1.26 (0.74) | – |
| *3. fix and polish (τ chosen on validation)* | | | | | |
| hybrid + RINS, τ = 1 s | 100 | 0.220 [0.15, 0.30] | 1.80 | 3.96 (1.99) | 98 in the polish |
| hybrid + local branching r = 10, τ = 10 s | 100 | 0.219 | 1.80 | 1.56 (1.18) | – |
| combined 98 % + RINS, τ = 3 s | 100 | 0.655 [0.46, 0.86] | 3.11 | 7.25 (5.64) | 98 in the polish |
| combined 98 % + gradient release m = 60, τ = 10 s | 100 | 0.585 [0.41, 0.78] | 2.61 | 4.76 (3.29) | 85 in the polish |
| combined 98 % + local branching r = 10, τ = 10 s | 100 | 0.680 | 4.28 | 2.44 (2.11) | – |

Paired against the hybrid and against LtF-kNN (variant − comparator; gap in pp, log of the per-instance speed-up;
instance bootstrap, 60 instances):

| variant | vs hybrid: Δ gap pp [CI] | Δ log speed-up [CI] | vs LtF-kNN: Δ gap pp [CI] | Δ log speed-up [CI] |
|---|---|---|---|---|
| Predict-and-Search | −0.013 [−0.107, +0.099] | −1.00 [−1.28, −0.72] | −0.174 [−0.352, −0.013] | −0.59 [−0.86, −0.34] |
| core + band | +0.049 [+0.008, +0.108] | −0.04 [−0.15, +0.07] | −0.112 [−0.294, +0.054] | +0.36 [+0.07, +0.65] |
| full MILP, decoded start | −0.109 [−0.176, −0.046] | −1.10 [−1.34, −0.85] | −0.270 [−0.441, −0.126] | −0.69 [−0.92, −0.48] |
| fix, then prove | −0.130 [−0.189, −0.080] | −1.27 [−1.45, −1.09] | −0.291 [−0.452, −0.165] | −0.86 [−1.12, −0.61] |
| hybrid + RINS, τ = 1 s | **−0.043 [−0.076, −0.015]** | −0.19 [−0.25, −0.14] | **−0.204 [−0.370, −0.065]** | +0.21 [−0.07, +0.48] |
| hybrid + local branching, τ = 10 s | −0.044 [−0.085, −0.016] | −0.87 [−1.07, −0.68] | −0.205 [−0.368, −0.069] | −0.47 [−0.73, −0.22] |
| combined 98 % + RINS, τ = 3 s | +0.392 [+0.214, +0.584] | +0.56 [+0.27, +0.86] | +0.231 [+0.029, +0.438] | +0.96 [+0.65, +1.26] |
| combined 98 % + gradient release, τ = 10 s | +0.322 [+0.149, +0.511] | +0.16 [−0.10, +0.44] | +0.161 [−0.040, +0.370] | +0.56 [+0.30, +0.82] |
| hybrid cut at 10 s (cold; τ chosen on validation) | +0.070 [+0.016, +0.140] | +0.41 [+0.27, +0.55] | −0.091 [−0.283, +0.083] | +0.81 [+0.53, +1.08] |

Warm start against cold start, same instances (time to proof incl. every overhead; TTQ = time until the incumbent is
within 1 % / 0.5 % of the cold full MILP's final cost; starts accepted on 100 % of the runs):

| comparison | time to proof mean s, cold / warm | median ratio cold/warm [CI] | TTQ 1 % median s, cold / warm | TTQ 0.5 % median s |
|---|---|---|---|---|
| full MILP, decoded start | 26.6 / 25.8 | 0.99 [0.92, 1.17] | 10.8 / **3.2** | 11.0 / 9.7 |
| hybrid reduced MILP, decoded start | 14.4 / 13.9 | 0.92 [0.91, 1.12] | 5.1 / 3.8 | 5.2 / 5.4 |
| error-cost 90 % reduced MILP, decoded start | 8.4 / 8.5 | 0.97 [0.89, 1.13] | 3.0 / 1.3 | 3.3 / 2.9 |
| combined 98 % reduced MILP, decoded start | 3.5 / 3.5 | 0.94 [0.90, 1.04] | 1.7 / 0.9 | 1.7 / 1.1 |
| full MILP from the hybrid's solution (incl. the hybrid) | 26.6 / 39.3 | 0.74 [0.72, 1.07] | 10.8 / 6.1 | 11.0 / 6.3 |

Polish: gap reduction against added time (base → polished; added time = mean method-time increase):

| polish | τ = 1 s | τ = 3 s | τ = 10 s |
|---|---|---|---|
| hybrid + RINS (0.263 %, 5.74×) | 0.220 %, +0.75 s, 3.96× | 0.200 %, +1.6 s, 3.70× | 0.186 %, +2.5 s, 3.60× |
| hybrid + local branching r = 10 | 0.263 %, +1.0 s, 3.32× | 0.226 %, +2.7 s, 2.25× | 0.219 %, +7.5 s, 1.56× |
| hybrid + full MILP from its solution | 0.263 %, +1.0 s, 3.32× | 0.240 %, +2.8 s, 2.28× | 0.211 %, +7.6 s, 1.67× |
| combined 98 % + RINS (0.840 %, 13.5×) | 0.767 %, +0.7 s, 8.90× | 0.655 %, +1.4 s, 7.25× | 0.566 %, +2.2 s, 6.78× |
| combined 98 % + gradient release m = 60 | 0.760 %, +0.9 s, 8.12× | 0.728 %, +1.9 s, 6.09× | 0.585 %, +4.2 s, 4.76× |
| combined 98 % + local branching r = 10 | 0.840 %, +1.0 s, 7.95× | 0.746 %, +2.7 s, 4.76× | 0.680 %, +7.6 s, 2.44× |

Re-timing: the full MILP here took a median 1.27× its stored dataset time (13 % of instances within ±10 %), so every
reference was re-run (see Setup and Caveats).

![uc12 Pareto](../../results/uc12/solver_results_pareto.png)

What the numbers say (uc12):

* **Trust region (Predict-and-Search) does not buy speed.** At the same set, the trust region with Δ = 10 recovers the
  quality that hard fixing loses (3.55 % → 0.25 %, the hard-fixed set has a 69 % outlier), but it is as slow as the full
  MILP (1.28× mean; it hit its 30 s limit on 24 of 60 instances). Against the hybrid: level gap, 2.7× lower speed-up
  (log −1.00). Why: the full MILP spends nearly all of its time at the root node (median 1 node on test; median 14 and
  0 nodes for the full MILP and the trust region on validation): LP solves, cut rounds and heuristics on the full-size
  network model. A trust-region row removes no variable and does not shrink the LP; only hard fixing, through
  presolve, makes the root cheap. The core + band variant (hybrid fixings hard, trust region over the rest) is never
  faster than the hybrid and slightly worse (+0.049 pp [+0.008, +0.108]).
* **Warm starts are accepted but do not shorten proofs.** HiGHS takes the start (100 %) and reports it as its first
  incumbent, but the 0.1 % proof is limited by the dual bound: time to proof is unchanged for the full MILP (median
  ratio 0.99) and for every reduced MILP (0.92–0.97). What the start changes is the anytime profile of the full MILP:
  within 1 % of the final cost after a median 3.2 s instead of 10.8 s, and 0.26 % vs 0.34 % mean gap when both are cut
  at 20 s (paired −0.08 pp [−0.21, +0.01]). For reduced MILPs the start is a mixed blessing: cut at 10 s, the hybrid
  with a start is not better than without (+0.145 pp [−0.03, +0.47]; one instance at 9.5 %), because the start changes
  HiGHS's search path. "Fix, then prove" (the full MILP warm-started with the hybrid's solution) reaches a better mean
  gap than the cold full MILP within the same 60 s limit (0.133 % vs 0.221 %, −0.088 pp [−0.228, −0.002]: the start
  helps on the instances where the cold run stops at the limit), but it costs the hybrid's time plus a full solve
  (1.26× mean, 0.74× median speed-up).
* **Polish: RINS on the LP relaxation is the only neighbourhood that pays.** Re-opening the ≈ 2 % of decisions where
  the incumbent disagrees with the LP relaxation (already computed as GNN input) is a tiny reduced MILP: it cuts the
  hybrid's gap by 0.043 pp [0.015, 0.076] in 0.75 s (0.263 → 0.220 %, 5.7× → 4.0×) and the combined 98 % rule's by
  0.185 pp in 1.4 s (0.840 → 0.655 %, 13.5× → 7.3×). Local branching and the warm full MILP need the full-size root
  (≥ 1–2 s before they find anything) and give the same or smaller reductions at 3–8× the added time. Gradient release
  (dispatch-LP reduced costs) works on the fast rule (−0.25 pp at τ = 10 s) but costs more time than RINS for the same
  gain.
* **Pareto front.** From fast to slow: combined 98 % (0.84 %, 13.5×) → combined 98 % + RINS (0.66 %, 7.3×) → hybrid
  cut at 10 s (0.33 %, 6.8×) → hybrid (0.26 %, 5.7×) → hybrid + RINS (0.22 %, 4.0×) → full MILP from the hybrid's
  solution (0.13 %, 1.3×). Learning to Fix with the paper's kNN (0.42 %, 3.6×) is dominated by every point from the
  hybrid down; the LP-integral fixing without learning (1.08 %, 12.4×) is dominated by the combined 98 % rule.

### uc24: 40 test instances (pass A), first 12 (pass B)

Full tables: [`results/uc24/solver_results.md`](../../results/uc24/solver_results.md) (+ `.json`, Pareto figure),
raw records `results/uc24/solver_test.jsonl` (pass A) and `solver_test_b.jsonl` (pass B). Full MILP (cold, same
process): 149.6 s mean, 11 of 40 at the 300 s limit, 0.152 % mean gap to its own bound. Re-timing: median ratio 0.98
to the stored dataset times, 80 % of instances within ±10 % (the stored uc24 times are consistent with core 0 today).

| method | n | feasible % | gap mean % [95 % CI] | gap max % | speed-up mean (median) | fixed % |
|---|---|---|---|---|---|---|
| *references, re-run in the same process* | | | | | | |
| faithful LtF, kNN, ε = 1 % | 40 | 100 | 0.603 [0.41, 0.82] | 2.70 | 5.12 (3.29) | 67 |
| ours: guarded rule, 95 % target | 40 | 100 | 0.668 [0.49, 0.86] | 2.29 | 11.25 (4.13) | 85 |
| no learning: fix the LP-integral decisions | 40 | 100 | 0.951 [0.70, 1.22] | 3.35 | 17.36 (8.87) | 98 |
| *warm start* | | | | | | |
| guarded 95 % + decoded start, to proof | 40 | 100 | 0.671 [0.49, 0.86] | 2.29 | 11.28 (5.29) | 85 |
| *fix and polish (τ chosen on validation)* | | | | | | |
| guarded 95 % + RINS, τ = 20 s | 40 | 100 | 0.486 [0.35, 0.64] | 1.79 | 5.68 (3.00) | 97 in the polish |
| guarded 95 % + gradient release m = 250, τ = 60 s | 40 | 100 | 0.430 [0.31, 0.57] | 2.12 | 4.75 (2.57) | 74 in the polish |
PASS_B_ROWS

Paired (variant − comparator; 40 instances unless noted):

| variant | vs guarded 95 %: Δ gap pp [CI] | Δ log speed-up [CI] | vs LtF-kNN: Δ gap pp [CI] | Δ log speed-up [CI] |
|---|---|---|---|---|
| guarded 95 % + RINS, τ = 20 s | −0.182 [−0.291, −0.092] | −0.52 [−0.69, −0.37] | −0.117 [−0.307, +0.067] | +0.15 [−0.18, +0.47] |
| guarded 95 % + RINS, τ = 10 s (curve point) | −0.134 [−0.209, −0.068] | | −0.070 [−0.257, +0.110] | +0.29 [−0.06, +0.63] |
| guarded 95 % + gradient release, τ = 60 s | −0.238 [−0.345, −0.141] | −0.69 [−0.85, −0.55] | −0.173 [−0.382, +0.019] | −0.02 [−0.36, +0.30] |
| guarded 95 % + decoded start | +0.003 [−0.000, +0.008] | −0.00 [−0.16, +0.15] | +0.068 [−0.131, +0.267] | +0.66 [+0.28, +1.04] |
| guarded 95 % (reference) | – | – | +0.065 [−0.134, +0.263] | +0.67 [+0.25, +1.08] |
| LP-integral fixing (no learning) | +0.283 [+0.064, +0.530] | +0.50 [+0.15, +0.87] | +0.348 [+0.119, +0.568] | +1.17 [+0.80, +1.52] |
PASS_B_PAIRS

Polish curve (guarded 95 %: 0.668 %, 11.25×): RINS τ = 5 / 10 / 20 / 60 s → 0.606 / 0.533 / 0.486 / 0.453 % at
7.7 / 6.6 / 5.7 / 5.3× (added 3.9 / 7.0 / 11.3 / 15.2 s); gradient release m = 250 → 0.663 / 0.633 / 0.488 / 0.430 %
at 7.3 / 6.2 / 5.4 / 4.8× (added 4.2 / 7.6 / 12.6 / 25.3 s). Warm start of the guarded rule's reduced MILP: time to
proof 30.5 / 32.0 s cold / warm (median ratio 0.93), TTQ 1 % median 10.3 / 8.5 s. Full MILP: within 1 % / 0.5 % of its
final cost after a median 25.8 / 26.3 s; 6.5 % mean gap if cut at 30 s, 0.29 % at 60 s.
PASS_B_TEXT

## Verdict

**uc12 (60 test instances, paired, paper metrics).**

* **Against the hybrid: no solver-side variant beats it at equal speed-up.** The only significant improvement of its
  gap comes from RINS polish (−0.043 pp [−0.076, −0.015] in 0.75 s), which moves along the frontier rather than past
  it (5.7× → 4.0×). Trust regions match its gap only at full-MILP speed (−0.013 pp [−0.107, +0.099], log speed-up
  −1.00 [−1.28, −0.72]); soft-fixing the band beyond the hybrid's fixings costs gap and gains nothing
  (+0.049 pp [+0.008, +0.108]); warm starts change neither its gap nor its proof time.
* **Against Learning to Fix as published (kNN, ε = 1 %): beaten on both axes.** Hybrid + RINS has a lower gap
  (−0.204 pp [−0.370, −0.065]) at a speed-up that is not lower (+0.21 log [−0.07, +0.48]; 4.0× vs 3.6× mean); the
  hybrid cut at 10 s is level in gap (−0.09 pp [−0.28, +0.08]) and 2.2× faster (+0.81 log [+0.53, +1.08]). (The hybrid
  alone already beats it: −0.161 pp [−0.333, −0.013], +0.40 log [+0.10, +0.70], reproducing [`hybrid.md`](hybrid.md).)
* **What closes the gap to exact solving is cheap polishing in a reduced space, not trust regions or warm starts.**
  Every neighbourhood that keeps the full-size model (trust region, local branching, warm-started full MILP) pays the
  full-size root again; on this benchmark that root *is* the solve. RINS around the LP relaxation, a reduced MILP of
  ≈ 2 % free decisions, adds 0.7–1.4 s and recovers a quarter to a third of the distance to the best schedules
  found here (fix-then-prove, 0.133 %): hybrid 0.263 → 0.220 %, combined 98 % 0.840 → 0.655 %.

## Caveats

* **Another HiGHS build than the stored uc12 references.** The earlier uc12 studies solved through scipy's HiGHS; here
  everything runs through highspy 1.15 with one thread on one pinned core. The re-run references reproduce their
  stored objectives (identical solutions on the instances compared), but solve times are longer: on the first test
  instances the full MILP took a median 1.7× its earlier back-to-back time and the reduced MILPs 1.3–1.6×. Speed-ups
  here are paired and internally consistent, but not interchangeable with the published ones (they are slightly
  higher for fixing rules, because the full MILP slowed down more than the reduced ones).
* **Shared machine.** Three other agents' jobs ran on the other cores (load average 4–5); one process ran on core 0 at
  a time. Back-to-back pairing removes the drift between runs, not the noise within one.
* **Post-hoc time limits.** Results at τ are read from one run's incumbent trace. HiGHS is deterministic, so a run with
  limit τ follows the same path, but its timing can differ slightly (timer checks, final clean-up).
* **Small validation sets.** 12 uc12 and 6 uc24 validation instances (one instance with every variant costs ≈ 5 min
  on uc12 and more on uc24). Parameters were chosen coarsely (two values each); a finer search could move individual
  points, but the structural results (no proof speed-up from trust regions or warm starts) did not depend on them.
* **The hybrid's thresholds were tuned on all 360 uc12 validation instances**, including the 12 used here, so its
  validation gaps are in-sample; this affects the selection of polish parameters only.
* **One seed** for every model (BCE GNN seed 0, harm ensemble seed 0, self-trained GNN seed 0, uc24 imitation GNN).
* **Trust-region scope.** Predict-and-Search was evaluated as published (one deviation budget over the most confident
  decisions, full model otherwise), not with its learned contrastive model (ConPaS); the ranking is our error cost
  (uc12) or error probability (uc24). With a commercial solver whose root processing is cheaper relative to the tree
  the balance could differ.
