# Learning to Fix, implemented from the full paper (B2, 12-hour network-constrained UC)

Code: [`otsl/ltfx.py`](../../otsl/ltfx.py) (kNN, fixing rule, Appendix A master, check / relaxation MILPs,
Algorithm 1 + 2), `scripts/uc_ltfx_{prep,tune,eval,report,diag}.py`. Results:
[`results/uc12/ltfx_results.md`](../../results/uc12/ltfx_results.md) (+ `.json`), raw solves
`results/uc12/ltfx_eval_test_fresh.jsonl`, tuning logs and thresholds `results/uc12/ltfx_tune_<model>_<eps>.{log,json}`,
probabilities `results/uc12/ltfx_probs.npz`, diagnostics `results/uc12/ltfx_diag.json`. New data:
`data/generated/uc12/val_extra.npz` (120 validation-day instances, seed 31).

**Summary.** Learning to Fix (Fritz et al. 2026) is implemented from the full paper: kNN probabilities (k = 50,
eq. 13, Table II features), generator-specific thresholds shared across hours, tuned by the logic-based Benders
decomposition of Algorithms 1–2 with the Appendix A density objective, so that every validation instance's reduced
UC — all fixings jointly — stays within ε of C*. On 180 validation instances (60 + 120 newly generated) all eight
tuning runs (kNN and self-trained GNN at ε = 10 / 5 / 1 %, REINFORCE and BCE GNNs at 1 %) converged, verified
instance by instance, for 4.0 core-hours; the one deviation is a 6 s cap on the relaxation MILPs (more
conservative thresholds). On the first 60 instances of `test_fresh` (full MILP back to back): kNN at ε = 1 % gives
0.40 % gap to the dual bound (paper 0.48 %), 100 % feasible, 68 % fixed, but only 4.7× mean per-instance
speed-up (paper 20.8×; 1.4× as a ratio of mean times). The joint check removes the combined-OFF failure of our
earlier per-generator reconstruction (0 vs 10 instances above 10 %; mean gap 0.27 vs
37.7 % on the same instances), but the ε guarantee is in-sample (kNN at ε = 10 %: 3 instances
above 10 %, 3 infeasible). Applied to our BCE GNN, LtF at ε = 1 % is the most accurate rule at 2–3×
(0.14 % mean gap at 2.7× vs 0.29 % at 3.6× for our error-cost rule; paired +0.15 pp [-0.03, +0.33], not
significant); above ≈ 4.5×
no tuned LtF rule has a point (our combined pipeline: 0.64 % at 6.4×, 0.70 % at 8.3×).

## 1. The method (Fritz, Makrides, Fetanat & Pinson, arXiv 2609.39396, §III-B, §IV-B, Appendices A–B)

* **Classifier.** kNN with k = 50, Euclidean distance; on-probability of (g, t) = inverse-distance-weighted mean of
  the neighbours' optimal commitments (eq. 13). Features (Table II): aggregate demand, renewables, net load, their
  hourly deltas, 3-step moving average and rolling maximum of the net load, net load normalised by its mean / max
  over the horizon and over the day, sin / cos of the hour, and each unit's initial on/off hours.
* **Fixing rule (eq. 5).** Generator-specific thresholds [τ̲_g, τ̄_g] shared across hours: π > τ̄_g fixes ON,
  π < τ̲_g fixes OFF, otherwise the solver decides.
* **Threshold choices.** Constant [r, 1 − r] (eq. 6); worst-case misprediction (eq. 7: τ̲_g = lowest probability of
  an optimal ON decision on validation, τ̄_g = highest probability of an optimal OFF decision, midpoint if they do not
  overlap); **suboptimality-constrained** (eq. 8): the tightest thresholds by the probability mass left free (eq. 10)
  such that every validation instance's reduced UC — *all* its fixings applied together — has a solution within
  ε of C* (8e).
* **Algorithm 1** (logic-based Benders decomposition). Master: minimise d(τ̲, τ̄) s.t. 0 ≤ τ̲ ≤ τ̄ ≤ 1 and the
  accumulated cuts Φ. Check the validation instances in order; at the first instance whose reduced problem has no
  solution within ε, call ADD CUTS, add the cut, re-solve the master, restart the checks. Stop when all pass.
* **Algorithm 2** (ADD CUTS). Up to K_max times: solve the relaxation MILP min Σ ν over the fixed decisions
  (ν = 1 releases a fixing) s.t. the UC constraints, the cost tolerance and no-good constraints against the earlier
  release sets; each release set R_k gives a conjunction of threshold conditions (τ̲_g ≤ π for a released OFF fix,
  τ̄_g ≥ π for a released ON fix); the cut is the OR of the K conjunctions, modelled with auxiliary binaries.
* **Density objective (Appendix A).** Per (g, t): split the validation probabilities into Q = 20 quantile bins
  of widths Δ^q; bin variables 0 ≤ Δτ^q ≤ Δ^q with τ̲_g ≥ Σ_q Δτ̲^q and τ̄_g ≤ 1 − Σ_q Δτ̄^q; bins up to the one
  above which all probabilities exceed 0.5 count for τ̲, the rest for τ̄; maximise Σ (1/max(Δ^q, 10⁻⁶)) Δτ^q.
* **Settings.** ε ∈ {10, 5, 1} %, K_max = 10; validation = 20 % of 2,621 instances (≈ 524).

## 2. Implementation, and where it deviates

Every element of §1 is implemented; the table lists the setting of each element here and marks what is a
**deviation** (changes the method's output), a *choice* (the paper is silent) or *exact* (saves solves only).

| element | paper | here |
|---|---|---|
| UC | UnitCommitment.jl default, Irish system, copper plate, 51 thermal units + storage / hydro, 72 h, Gurobi (0.25 %) | `otsl.uc.UCModel`: RTS-GMLC, DC network, 73 thermal units, 12 h, min up / down, ramping, start-up costs, 3 % reserve, shedding and reserve shortfall as penalties (a reduced problem is infeasible only through min up / down conflicts of the fixings); HiGHS |
| data | 2,621 instances, 60 / 20 / 20 % (≈ 524 validation) | 500 training, **180 validation** (the 60 of `val.npz` + 120 new `val_extra.npz` instances on validation days, generated for this study with `uc_gen.py --seed 31`), test: first 60 of `test_fresh` |
| C* in (8e) | optimal cost | the dataset MILP objective (60 s, 0.1 % gap; 73 % / 82.5 % of val / val_extra proven optimal, max gap 2.1 %) |
| kNN (eq. 13) | k = 50, Euclidean, inverse-distance weights, Table II features | same. *Choices*: features z-scored on train (the paper mixes MW, ratios and harmonics without stating a scaling); labels = optimal schedules canonicalised inside groups of identical units (an equally optimal schedule; RTS-GMLC has many identical copies); RES = wind + solar + hydro (hydro is a curtailable profile here); initial condition = ±1 (our data assume no carry-over of min up / down, so hours-in-state do not exist); the "daily" normalisations equal the horizon ones because a 12-h window lies within one day (kept, as listed) |
| fixing rule (eq. 5) | π > τ̄ ON, π < τ̲ OFF | same, with a 10⁻⁸ tolerance (master solutions sit exactly on cut values) |
| constant / worst-case thresholds (eqs. 6–7) | | same; worst case uses the canonical validation labels; a generator never ON (OFF) on validation gets τ̲ = 1 (τ̄ = 0) before the midpoint rule (*choice*) |
| master objective (Appendix A) | Q = 20 quantile bins per (g, t), weights 1 / max(Δ, 10⁻⁶) | same. *Choice / interpretation*: the 10⁻⁶ floor also bounds the bin variable, so that bins of zero width (point masses) still count: 65 % of the kNN probabilities are exactly 0 or 1, and with a zero bound a cut releasing one π = 0 decision (τ̲_g ≤ 0, i.e. *every* π = 0 decision of that generator freed) would cost the master nothing. *Choice*: a tie-break of weight 10⁻³ pulls thresholds whose objective is flat (generators with no validation mass between them) towards the standard rounding point 0.5; without it the master is degenerate for those generators |
| cuts | OR over K of AND of threshold conditions, auxiliary binaries | same (big-M = 1) |
| check (Algorithm 1, step 5) | feasibility of (8b)–(8g) | reduced MILP with (1 + ε)·C* as objective cut-off, stopped at the first solution below it, 60 s limit; no answer within 60 s counts as a violation (conservative) |
| relaxation MILP (Algorithm 2) | min Σ ν, solved (presumably to optimality), K_max = 10 | same MILP, K_max = 10, but **6 s per solve (deviation, for compute)**: the best release set found is used even if not minimal, and the K-loop stops at the first relaxation that finds no solution in 6 s, so cuts carry 1.4–3.0 alternatives on average instead of up to 10 (48–87 % of the relaxation solves stopped at 6 s). The first solve of an instance is warm-started from its optimal schedule aligned to the fixings inside identical-unit groups (always feasible; *exact*). ν = 1 entries whose u equals the fixed value are dropped from the release set (*exact*: the same solution satisfies those fixings) |
| caching | – | check results are reused by monotonicity: a fixing set contained in one that passed passes with the same witness schedule, one containing a failed set fails; relaxation solutions are stored as witnesses (*exact*) |
| budget | – | 75 min per tuning run (a run that hits it would stop and report how many validation instances its thresholds still violate); no run hit it |
| verification | "upon termination the thresholds admit a solution within ε for every validation instance" | after termination, every validation instance's stored witness schedule is checked against the final fixings and priced with the exact dispatch LP (min up / down rows included) |
| other classifiers | CatBoost (Appendix C) | the same tuning on our GNN probabilities: MILP-label BCE (`uc_model1_4.pt`), self-trained (`selftrain_m3.pt`), MILP-label REINFORCE (`uc_model1_rl.pt`); "our best GNN" = self-trained, the probability source chosen on validation for fixing in [`combo.md`](combo.md) (rule 3) |

**A property of the exact algorithm worth knowing.** With the probability-mass objective the master prefers
zero-width intervals (τ̲_g = τ̄_g: everything of g fixed). A cut τ̲_g ≤ π (release an OFF fix) is then satisfied by
*sliding* the collapsed interval down rather than widening it, which turns every decision of g with probability
between the old and the new position from fixed-OFF into fixed-ON; the next check of that instance (or another)
fails on those ON fixes and adds τ̄_g ≥ π. The same instance therefore fails several times and the intervals open
one side at a time. This is what Algorithm 1 does as written (checked on the master problem alone: a single-generator cut moves both
thresholds of that generator and nothing else); it costs iterations, not correctness.

## 3. Tuning on validation

180 validation instances (60 + 120 new), checked in that order. One process (one core) per run; the machine was
shared with another agent's jobs throughout, so wall times are indicative. Every run below **converged**: the final
master solution passed the check of all 180 instances, and the independent verification (each instance's witness
schedule, which respects all final fixings, priced with the exact dispatch LP) stays within ε for every instance.

| run | iterations | instances that ever failed | cut: alternatives / release-set size (mean) | relaxation MILPs (stopped at 6 s) | check MILPs solved / answered from cache | val fixed % | test_fresh fixed % (OFF / ON) | collapsed intervals (τ̲ = τ̄) | witness max cost increase % | core-h |
|---|---|---|---|---|---|---|---|---|---|---|
| kNN, ε = 10 % | 45 | 30 | 2.18 / 12.6 | 140 (104) | 421 / 1,159 | 82.9 | 83.5 (67.3 / 16.2) | 29 | 9.890 | 0.35 |
| kNN, ε = 5 % | 60 | 36 | 2.42 / 10.4 | 198 (137) | 863 / 1,279 | 75.2 | 75.8 (59.5 / 16.3) | 30 | 4.985 | 0.70 |
| kNN, ε = 1 % | 96 | 53 | 1.37 / 10.4 | 225 (195) | 414 / 3,596 | 66.7 | 67.6 (52.3 / 15.3) | 20 | 0.999 | 0.73 |
| self-trained GNN, ε = 10 % | 66 | 47 | 3.00 / 6.8 | 257 (123) | 695 / 2,841 | 84.8 | 84.6 (66.6 / 18.0) | 30 | 10.000 | 0.50 |
| self-trained GNN, ε = 5 % | 61 | 40 | 2.85 / 6.4 | 225 (130) | 431 / 2,138 | 83.7 | 83.6 (66.5 / 17.0) | 28 | 4.898 | 0.48 |
| self-trained GNN, ε = 1 % | 70 | 45 | 1.57 / 8.5 | 177 (138) | 300 / 2,493 | 75.3 | 75.7 (61.1 / 14.6) | 21 | 0.997 | 0.46 |
| REINFORCE GNN, ε = 1 % | 72 | 51 | 2.06 / 7.3 | 216 (121) | 280 / 2,798 | 70.9 | 69.2 (59.0 / 10.3) | 19 | 0.995 | 0.43 |
| BCE GNN, ε = 1 % | 76 | 49 | 1.60 / 7.9 | 195 (122) | 326 / 3,024 | 83.6 | 83.5 (67.5 / 16.1) | 23 | 0.992 | 0.39 |

For reference, the cheap thresholds on the same 180 instances fix (val): kNN worst-case 51.7 %, constant
[0.1, 0.9] / [0.05, 0.95] / [0.01, 0.99] 84.7 / 79.8 / 70.2 %; self-trained worst-case 59.9 %, constant 96.3 / 94.2 /
84.9 %. The paper's kNN fixed 87.5 / 82.3 / 78.8 % at ε = 10 / 5 / 1 % (test) — the same ordering, at a lower level
here.

**Tuning cost.** The eight runs took 4.03 core-hours in total (3.98 CPU-hours; 0.35–0.73 h per run, 2.1 h of wall time on two workers); one run costs 0.35–0.73 core-h for 180 instances, i.e. 7–15 s per validation instance, less than one full-MILP solve (≈ 20–30 s). The relaxation MILPs took 40–70 % of each run's time and the check MILPs (≈ 0.7–3 s
each) most of the rest; the master problem (Appendix A LP plus up to ~170 cut binaries) took < 4 min in total
per run. The cache answered 60–91 % of the checks (each re-solve of the master restarts the checks at instance 1).
The validation labels themselves (120 new MILPs) cost another 1.0 core-h.

**What the tuning does.** Most failures are min up / down conflicts or penalised shortfalls created by the
master's tightest-possible intervals: in 38–62 of the failing checks per run HiGHS proved that no solution below
(1 + ε)·C* exists (min up / down conflict or cost cut-off), in the other 6–42 it returned a solution above it. Because a cut on τ̲_g lets the master slide a collapsed interval rather than widen it (§2), the
same instance fails several times (kNN ε = 1 %: 96 cuts from 53 distinct instances). The final intervals are
very uneven: 20–30 generators keep a collapsed interval (everything fixed, by rounding at a generator-specific
point), while the cuts of the joint check push τ̲_g to 0 for 16 generators (kNN, ε = 1 %; 12 at 5 %, 5 at 10 %),
i.e. those units are never fixed OFF, not even at π = 0. The joint check thus protects exactly the combined OFF
errors that broke the earlier reconstruction (§5), at the price of fixing much less: 67.6 % of the test decisions
for kNN at ε = 1 % against 89.7 % for the earlier per-generator reconstruction.

## 4. Test results (first 60 instances of test_fresh, full MILP and every reduced MILP back to back)

Protocol: for each of the first 60 instances of `test_fresh` (generation seed 23, never used for any decision), one
worker solves the full MILP (60 s, 0.1 %) and then every reduced MILP back to back (2 workers, spawn pool; the other
agent's jobs shared the machine). Thresholds, rules and probability sources were all fixed on validation before
the test pass. The method's time includes inference (kNN: features and neighbour search; GNN: forward pass plus the
LP relaxation that feeds its features; our rules: also the error-cost model and the LP-relaxation guard).

Paper metrics (Section IV-C; gap = (C − DB) / C to the back-to-back full MILP's dual bound; statistics over feasible instances; speed-up = per-instance T_MILP / T_method including inference, LP relaxation for GNN features, guards; n = 60; full MILP 19.6 s on average):

| rule | feasible % | gap mean / max % | runtime mean / max s | speed-up mean / max | fixed mean / max % |
|---|---|---|---|---|---|
| full MILP (back to back) | 100.0 | 0.12 / 1.66 | 19.6 / 60.1 | 1.0 / 1.0 | 0.0 / 0.0 |
| kNN, τ = 0.5 (all fixed) | 73.3 | 51.86 / 97.92 | 0.2 / 0.3 | 109.9 / 442.7 | 100.0 / 100.0 |
| kNN, constant [0.1, 0.9] | 100.0 | 6.84 / 83.60 | 8.9 / 26.7 | 6.3 / 100.3 | 85.4 / 93.5 |
| kNN, constant [0.05, 0.95] | 100.0 | 1.78 / 73.35 | 12.9 / 60.0 | 2.4 / 24.3 | 80.3 / 91.0 |
| kNN, constant [0.01, 0.99] | 100.0 | 0.94 / 48.09 | 14.4 / 60.0 | 2.3 / 27.6 | 70.8 / 84.1 |
| kNN, worst case (eq. 7) | 100.0 | 0.19 / 1.78 | 16.9 / 60.1 | 1.8 / 12.6 | 52.2 / 62.0 |
| **LtF kNN, ε = 10 %** | 95.0 | 4.52 / 71.79 | 9.8 / 60.0 | 7.0 / 69.8 | 83.5 / 90.3 |
| **LtF kNN, ε = 5 %** | 100.0 | 1.12 / 5.90 | 9.4 / 49.8 | 8.3 / 97.9 | 75.8 / 83.0 |
| **LtF kNN, ε = 1 %** (paper's headline setting) | 100.0 | 0.40 / 3.60 | 14.0 / 60.1 | 4.7 / 78.7 | 67.6 / 75.9 |
| self-trained GNN, constant [0.1, 0.9] | 100.0 | 3.52 / 69.07 | 2.9 / 19.9 | 13.9 / 64.3 | 96.3 / 99.4 |
| self-trained GNN, constant [0.05, 0.95] | 100.0 | 1.71 / 40.60 | 3.7 / 24.6 | 10.9 / 61.0 | 94.2 / 99.4 |
| self-trained GNN, constant [0.01, 0.99] | 100.0 | 1.20 / 40.56 | 8.9 / 60.4 | 3.7 / 30.1 | 85.5 / 93.2 |
| self-trained GNN, worst case | 100.0 | 0.20 / 3.02 | 12.4 / 60.4 | 2.1 / 9.2 | 60.2 / 65.5 |
| **LtF self-trained GNN, ε = 10 %** | 96.7 | 0.85 / 5.07 | 3.5 / 18.1 | 13.5 / 61.7 | 84.6 / 90.2 |
| **LtF self-trained GNN, ε = 5 %** | 98.3 | 1.03 / 16.23 | 5.2 / 29.8 | 11.3 / 61.6 | 83.6 / 88.5 |
| **LtF self-trained GNN, ε = 1 %** | 98.3 | 0.33 / 2.72 | 8.5 / 55.4 | 4.0 / 33.7 | 75.7 / 79.5 |
| LtF REINFORCE GNN, ε = 1 % | 95.0 | 0.39 / 4.30 | 9.9 / 60.4 | 4.0 / 21.9 | 69.2 / 78.2 |
| LtF BCE GNN, ε = 1 % | 98.3 | 0.28 / 2.37 | 6.9 / 26.4 | 5.2 / 28.1 | 83.5 / 88.8 |
| ours: error-cost ranking + adequacy guard, 90 % (BCE GNN) | 100.0 | 0.42 / 4.34 | 5.4 / 20.9 | 4.2 / 17.2 | 89.9 / 90.0 |
| ours: combined pipeline (self-trained + error-cost + adequacy + LP guard), 95 % | 100.0 | 0.75 / 6.75 | 3.1 / 22.0 | 9.5 / 71.6 | 94.0 / 95.0 |
| ours: combined pipeline, 98 % | 100.0 | 0.82 / 6.28 | 2.4 / 18.9 | 13.6 / 65.4 | 92.1 / 97.9 |

Our earlier convention (gap to the dataset MILP objective over all instances; an infeasible reduced MILP falls back to the full MILP and pays both times; served = no shedding / spill / reserve shortfall; speed-up = ratio of mean times; mean-gap CI: instance bootstrap):

| rule | mean gap % [95 % CI] | median % | # > 1 % / > 10 % | served % | speed-up (ratio of means) | infeasible (fallback) |
|---|---|---|---|---|---|---|
| full MILP (back to back) | -0.02 [-0.05, -0.00] | 0.000 | 0 / 0 | 100.0 | 1.00 | 0 |
| kNN, τ = 0.5 (all fixed) | 523.56 [290.56, 785.83] | 18.502 | 43 / 33 | 40.0 | 2.59 | 16 |
| kNN, constant [0.1, 0.9] | 22.41 [5.14, 46.20] | 0.027 | 11 / 7 | 86.7 | 2.20 | 0 |
| kNN, constant [0.05, 0.95] | 5.03 [0.17, 14.37] | 0.000 | 5 / 2 | 93.3 | 1.52 | 0 |
| kNN, constant [0.01, 0.99] | 1.55 [-0.03, 4.67] | -0.000 | 1 / 1 | 96.7 | 1.36 | 0 |
| kNN, worst case (eq. 7) | 0.05 [-0.02, 0.14] | 0.000 | 2 / 0 | 96.7 | 1.16 | 0 |
| **LtF kNN, ε = 10 %** | 10.91 [0.97, 23.43] | 0.425 | 20 / 3 | 86.7 | 1.97 | 3 |
| **LtF kNN, ε = 5 %** | 1.02 [0.67, 1.39] | 0.412 | 14 / 0 | 98.3 | 2.08 | 0 |
| **LtF kNN, ε = 1 %** (paper's headline setting) | 0.27 [0.16, 0.44] | 0.099 | 4 / 0 | 96.7 | 1.40 | 0 |
| self-trained GNN, constant [0.1, 0.9] | 6.64 [1.20, 15.05] | 0.424 | 19 / 5 | 80.0 | 6.78 | 0 |
| self-trained GNN, constant [0.05, 0.95] | 2.10 [0.51, 4.81] | 0.206 | 14 / 3 | 86.7 | 5.25 | 0 |
| self-trained GNN, constant [0.01, 0.99] | 1.54 [0.23, 3.96] | 0.024 | 10 / 1 | 93.3 | 2.21 | 0 |
| self-trained GNN, worst case | 0.06 [-0.02, 0.16] | -0.000 | 2 / 0 | 95.0 | 1.58 | 0 |
| **LtF self-trained GNN, ε = 10 %** | 0.72 [0.51, 0.97] | 0.452 | 13 / 0 | 98.3 | 4.43 | 2 |
| **LtF self-trained GNN, ε = 5 %** | 0.95 [0.42, 1.74] | 0.174 | 12 / 1 | 96.7 | 3.82 | 1 |
| **LtF self-trained GNN, ε = 1 %** | 0.19 [0.10, 0.30] | 0.018 | 3 / 0 | 96.7 | 2.35 | 1 |
| LtF REINFORCE GNN, ε = 1 % | 0.24 [0.12, 0.41] | 0.012 | 4 / 0 | 98.3 | 2.00 | 3 |
| LtF BCE GNN, ε = 1 % | 0.14 [0.05, 0.26] | 0.023 | 2 / 0 | 98.3 | 2.68 | 1 |
| ours: error-cost ranking + adequacy guard, 90 % (BCE GNN) | 0.29 [0.14, 0.46] | 0.010 | 7 / 0 | 98.3 | 3.62 | 0 |
| ours: combined pipeline (self-trained + error-cost + adequacy + LP guard), 95 % | 0.64 [0.36, 0.95] | 0.096 | 12 / 0 | 96.7 | 6.36 | 0 |
| ours: combined pipeline, 98 % | 0.70 [0.44, 1.02] | 0.219 | 16 / 0 | 98.3 | 8.28 | 0 |

Paired against our rule closest in speed (per-instance gap difference, ours − rule, bootstrap 95 % CI):

| rule (speed-up) | our rule (speed-up) | difference, pp [95 % CI] |
|---|---|---|
| kNN, τ = 0.5 (all fixed) (2.6×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -523.27 [-785.51, -290.30] |
| kNN, constant [0.1, 0.9] (2.2×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -22.12 [-46.11, -4.81] |
| kNN, constant [0.05, 0.95] (1.5×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -4.74 [-14.12, +0.17] |
| kNN, constant [0.01, 0.99] (1.4×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -1.26 [-4.46, +0.42] |
| kNN, worst case (eq. 7) (1.2×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | +0.24 [+0.07, +0.44] |
| LtF kNN, ε = 10 % (2.0×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -10.62 [-23.16, -0.72] |
| LtF kNN, ε = 5 % (2.1×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -0.73 [-1.15, -0.35] |
| LtF kNN, ε = 1 % (paper's headline setting) (1.4×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | +0.02 [-0.21, +0.24] |
| self-trained GNN, constant [0.1, 0.9] (6.8×) | combined pipeline (self-trained + error-cost + adequacy + LP guard), 95 % (6.4×) | -6.00 [-14.39, -0.67] |
| self-trained GNN, constant [0.05, 0.95] (5.2×) | combined pipeline (self-trained + error-cost + adequacy + LP guard), 95 % (6.4×) | -1.47 [-4.04, +0.07] |
| self-trained GNN, constant [0.01, 0.99] (2.2×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -1.25 [-3.65, +0.07] |
| self-trained GNN, worst case (1.6×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | +0.23 [+0.09, +0.38] |
| LtF self-trained GNN, ε = 10 % (4.4×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -0.43 [-0.68, -0.20] |
| LtF self-trained GNN, ε = 5 % (3.8×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | -0.66 [-1.44, -0.15] |
| LtF self-trained GNN, ε = 1 % (2.4×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | +0.10 [-0.06, +0.27] |
| LtF REINFORCE GNN, ε = 1 % (2.0×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | +0.05 [-0.10, +0.22] |
| LtF BCE GNN, ε = 1 % (2.7×) | error-cost ranking + adequacy guard, 90 % (BCE GNN) (3.6×) | +0.15 [-0.03, +0.33] |

**Paper vs here (kNN).** The paper (Table I, kNN): ε = 10 / 5 / 1 % → feasible 98.67 / 98.86 / 99.81 %, mean gap to the
dual bound 1.56 / 1.05 / 0.48 %, mean speed-up 60.3 / 34.3 / 20.8×, fixed 87.5 / 82.3 / 78.8 %; constant [0.01, 0.99]
0.21 % at 4.1×; worst case 0.20 % at 2.2×. Here: feasible 95.0 / 100.0 / 100.0 %, mean gap to the dual bound 4.52 / 1.12 / 0.40 % (max 71.79 / 5.90 / 3.60 %), mean per-instance speed-up 7.0× / 8.3× / 4.7×, fixed 83.5 / 75.8 / 67.6 %; constant [0.01, 0.99] 0.94 % at 2.3×; worst case 0.19 % at 1.8×. **The ordering and the solution quality reproduce** (gap falls with ε, sub-0.5 % at ε = 1 %, the same min up / down infeasibility mode at ε = 10 %), **the speed-up does not**: the faithful tuning fixes 68 % instead of 79 % at ε = 1 %, and a 12-hour UC that the full MILP solves in 20 s on average leaves little to gain (ratio of mean times 1.4× at ε = 1 %, 2.1× at 5 %). As in the paper, the worst-case thresholds are safe but slow and constant [0.1, 0.9] is fast but has catastrophic instances (max 83.60 %); unlike the paper, even [0.01, 0.99] has one (max 48.09 %).

**Classifier matters.** The tuning applies unchanged to the GNN probabilities; BCE and self-trained fix more than kNN at the same validated ε. At ε = 1 %: BCE 0.28 % gap at 5.2× (84 % fixed, 98 % feasible), self-trained 0.33 % at 4.0× (76 %), REINFORCE 0.39 % at 4.0× (69 %), kNN 0.40 % at 4.7× (68 %). Self-trained at ε = 10 / 5 %: 0.85 / 1.03 % at 13.5× / 11.3×. In our convention (all instances, fallback, dataset reference): LtF-BCE ε = 1 % has a 0.14 % mean gap (max 2.36 %) at 2.7× — the lowest mean gap of any rule faster than 2× here.

**Failures.** Reduced problems made infeasible by the fixings (min up / down conflicts; the paper's only failure mode): kNN ε = 10 / 5 / 1 %: 3 / 0 / 0 of 60; self-trained 2 / 1 / 1; REINFORCE 3; BCE 1; kNN τ = 0.5 16; constant / worst-case rows and our rules 0 (the combined pipeline releases conflicting rows before solving). Instances above 10 % (our convention): kNN ε = 10 % 3, ε = 5 / 1 % 0 / 0; self-trained ε = 10 / 5 / 1 % 0 / 1 / 0; the validated ε is an in-sample guarantee.

## 5. Faithful version vs the earlier per-generator reconstruction

The earlier reconstruction ([`ltf.md`](ltf.md), written from search snippets only) and the paper's method differ in
four places, and the last two are what mattered:

| | earlier reconstruction | paper (implemented here) |
|---|---|---|
| classifier | kNN, k = 21, per-hour area loads + renewables by type + initial status, chosen on val log-loss | kNN, k = 50, Table II features, eq. 13 |
| rule | symmetric: fix round(p) where max(p, 1 − p) ≥ θ_g | asymmetric interval [τ̲_g, τ̄_g]; OFF and ON sides tuned separately |
| what is checked | per generator and threshold level: a dispatch-LP **upper bound** of the cost of *that generator's* fixings alone, other units compensating heuristically | per validation instance: does the reduced MILP with **all** fixings of all generators have a solution within ε (MILP) |
| constraint | Σ_g mean impact_g ≤ τ: an **average** over instances, additive over generators | **every** instance within ε: a worst case over instances, joint over generators |

**Does the joint check fix the combined-OFF failure?** On validation, yes, and by construction. Applied to the earlier
reconstruction's kNN thresholds (τ = 1 %), the paper's joint check fails on **11 of the 60** instances they were
calibrated on (6.8 wrong OFF fixes per failing instance, 3.1 per passing one; `ltfx_diag.json`): its additive budget
let several individually cheap OFF errors coexist. The faithful algorithm turns each such instance into cuts; at
ε = 1 % they push τ̲_g to 0 for 16 kNN generators (no OFF fixing at all for those units) and leave 66.7 % of the
validation decisions fixed instead of the earlier 88–90 %. On the same 60 test instances, LtF-kNN at ε = 1 % carries 2.0 wrong OFF fixes per instance against 5.9 for the earlier reconstruction, has 0 instances above 10 % (earlier: 10) and a mean gap of 0.27 % against 37.7 % (paired difference -37.4 pp [-70.2, -12.3]). **The combined-OFF failure of the earlier reconstruction is gone at ε = 1 %.** On REINFORCE probabilities, where the earlier calibration already worked, the faithful ε = 1 % thresholds give 0.24 % against 0.55 % (paired -0.31 pp [-0.47, -0.15]).

**What the joint check does not fix.** (i) The guarantee is in-sample: kNN at ε = 10 % has 3 of 60 test instances above 10 %
(up to 72 % to the dual bound) and self-trained at ε = 5 % one at 16 % (§4) — failure modes absent from the 180
validation instances. (ii) Min up / down infeasibility of the
reduced problem remains possible on test (the paper reports the same, 0.2–1.3 % of its test instances), because a
validation-feasible threshold pair can still produce an inconsistent ON–OFF–ON pattern on a new instance.
(iii) The price is speed: the joint, worst-case constraint fixes far less than the earlier average-case budget, and
on a 12-hour UC whose full MILP takes ≈ 20 s the reduced problems at 65–85 % fixed are only a few times faster.

## 6. Verdict

* **Implemented as written, with one compute deviation.** Algorithm 1 + 2, the Appendix A objective, the kNN of eq. 13
  and the Table II features are implemented from the text; the relaxation MILPs are capped at 6 s (non-minimal release
  sets, 1–3 alternatives per cut instead of up to 10), which makes the thresholds more conservative than the exact
  algorithm's. All eight tuning runs converged, and the ε constraint holds **jointly for every one of the 180
  validation instances** (verified independently by pricing each instance's witness schedule). Tuning cost:
  4.0 core-hours for eight runs (0.35–0.73 per run, i.e. 7–15 s per validation instance, less than one full-MILP solve).
* **The paper's solution quality reproduces, its speed-up does not.** kNN at ε = 1 %: 0.40 % mean gap to the dual
  bound (paper 0.48 %), 100 % feasible (99.81 %), but 4.7× mean per-instance speed-up (20.8×) and
  1.4× as a ratio of mean times, at 68 % fixed (78.8 %). On B2 the joint, worst-case
  constraint leaves a third of the decisions free and the 12-hour MILP is already fast (≈ 20 s).
* **The joint check fixes the combined-OFF failure of the earlier reconstruction.** Same instances, ε = 1 %: 0 vs
  10 instances above 10 %, 2.0 vs 5.9 wrong OFF fixes per instance, mean gap
  0.27 vs 37.7 %; the earlier thresholds violate the paper's joint check on 11 of their own 60
  calibration instances. The guarantee is in-sample, though: at ε = 10 % kNN has 3 test instances above
  10 % and 3 reduced problems made infeasible by min up / down conflicts (the paper's failure mode).
* **Against our rules.** In our convention (ratio of mean times), LtF on our BCE GNN at ε = 1 % is the most accurate
  rule faster than 2×: 0.14 % mean gap (max 2.36 %) at 2.7×, against 0.29 % at 3.6× for our error-cost
  ranking + adequacy guard at 90 % — paired, ours − LtF +0.15 pp [-0.03, +0.33], not significant, with ours
  35 % faster. The fastest tuned LtF point, self-trained at ε = 10 % (0.72 % at 4.4×,
  2 infeasible), is dominated by our combined pipeline at 95 % (0.64 % at 6.4×); beyond ≈ 4.5× only our rules
  (0.70 % at 8.3× at 98 %) and the paper's constant thresholds (2.10 / 6.64 % mean gaps on the same
  self-trained probabilities, max 222.84 %) have points. By the paper's own metric (mean of per-instance speed-ups,
  feasible instances only) LtF-self-trained at ε = 10 % (13.5×, 0.85 % to the dual bound, 96.7 % feasible) and
  our combined pipeline at 98 % (13.6×, 0.82 %, 100 % feasible) are level. LtF with the paper's own kNN is
  dominated: at 1.4× it matches the gap our 90 % rule reaches at 3.6× (+0.02 pp [-0.21, +0.24]).
* **Net.** The faithful Learning to Fix is a sound, safe calibration method — the joint, per-instance check is exactly
  what the earlier reconstruction lacked — and on good probabilities (our BCE GNN) it gives the most accurate
  fixing measured on B2 at 2–3×, statistically level with our error-cost rule at 3.6×. It does not reach the speeds of our learned error-cost rules, and the paper's 20×
  is not attainable on a 12-hour, 73-unit UC with ≈ 20 s full solves.

## 7. Limitations

* **Relaxation MILPs capped at 6 s** (§2): most cuts carry 1–3 alternatives instead of up to 10, and many release
  sets are not minimal (the first relaxation of an instance is warm-started from its optimal schedule, which releases
  every disagreeing fixing; 6 s of improvement does not always reach the minimum). Non-minimal release sets widen the
  intervals more than the exact algorithm would, so the faithful-in-structure thresholds here are **more
  conservative** than the paper's algorithm would produce on the same data: the exact method would fix more, run
  faster, and — given that its guarantee does not transfer perfectly to test (ε = 10 %) — fail somewhat more often.
* **180 validation instances instead of ≈ 524.** Fewer instances mean fewer cuts (looser fit to rare failure modes);
  the ε guarantee is in-sample only.
* **C\* is the dataset MILP's incumbent** (0.1 % gap, 60 s; up to 2.1 % gap on time-limited instances), so "within ε of
  C*" is slightly looser than within ε of the optimum on hard instances.
* **System and horizon differ from the paper** (§2): a 12-hour RTS-GMLC instance with a DC network solves in ≈ 15–30 s,
  not ≈ 60 s for 72 hours; the room for speed-up is smaller, and inference / guard time weighs more.
* **One seed, one test pass.** Thresholds are deterministic given the data; the GNNs are the existing single-seed
  checkpoints; the harm model of our rules is seed 0 of the 3-seed ensemble. Timings come from a 4-core machine shared
  with another agent (load ≈ 2–5), so per-instance speed-ups are noisy; the paper's mean of per-instance ratios is
  dominated by instances where the full MILP is slow and the reduced one fast, which is why both it and the ratio of
  mean times are reported.
* **Best GNN chosen a priori** (self-trained, from the combo study's validation rule) and tuned at all three ε; BCE and
  REINFORCE were tuned at ε = 1 % only. By the paper's own criterion — validation fixed share at the same guarantee —
  BCE would have been the best GNN at ε = 1 % (83.6 % vs 75.3 % self-trained, 70.9 % REINFORCE); its ε = 5 / 10 %
  thresholds were not tuned for lack of time.
* Feature scaling, canonical labels, the bin-floor reading of Appendix A and the tie-break are implementation choices
  where the paper is silent (§2).

## 8. Reproduce

```bash
python3 scripts/uc_gen.py --cfg uc12 --split val_extra --n 120 --seed 31 --workers 2   # 120 validation-day instances
python3 scripts/uc_ltfx_prep.py                       # kNN (paper settings) and GNN probabilities on val, val_extra, test_fresh
python3 scripts/uc_ltfx_tune.py --jobs knn:0.01,st:0.01,knn:0.1,st:0.1,knn:0.05,st:0.05,rl:0.01,bce:0.01 \
    --workers 2 --relax_tl 6 --budget_min 75         # Algorithm 1 + 2, one job per process; also constant / worst-case
python3 scripts/uc_ltfx_eval.py --split test_fresh --n 60 --workers 2
python3 scripts/uc_ltfx_report.py && python3 scripts/uc_ltfx_diag.py --joint_check_old 60
```
Worker pools use 2 spawn processes; HiGHS runs single-threaded in the tuning, PyTorch single-threaded.
