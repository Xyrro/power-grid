# Learning to Fix, reconstructed, against our fixing rules (B2, 12-hour network-constrained UC)

Code: [`otsl/ltf.py`](../../otsl/ltf.py), `scripts/uc_ltf_{prep,curves,calib,seq,eval,proxycheck,diag,report}.py`.
Results: [`results/uc12/ltf_results.md`](../../results/uc12/ltf_results.md) (+ `.json`), raw solves
`results/uc12/ltf_eval_*.jsonl`, thresholds `results/uc12/ltf_thresholds{,_seq}.json`, probabilities
`results/uc12/ltf_probs.npz`, impact curves `results/uc12/ltf_curves.pkl`, logs `results/uc12/ltf_*.log`.

**Summary.** Learning to Fix (Fritz et al. 2026; EPRI 2025 UC competition winner) is reconstructed from its
abstract: a kNN commitment classifier, generator-specific confidence thresholds from a decomposition under a 1 %
validation cost tolerance, the impact of fixing errors measured with dispatch-LP upper bounds of single-generator
reduced MILPs. On the first 60 instances of a fresh B2 test set (12-hour network-constrained UC, RTS-GMLC), with full
and reduced MILPs solved back to back, **the published configuration fails on B2**: 37.7 % mean gap (0.66 % median,
75 % of instances served without shedding / reserve shortfall) at 2.3×, and 56 % [38, 71] over three bootstrap
calibration seeds; larger tolerances are catastrophic. Generator-wise impact estimates miss joint OFF errors that
together remove the reserve. **The calibration is worth more than the kNN**: applied to our REINFORCE probabilities it
gives 0.55 % mean / 98 % served at 4.7× and 0.80 % at 6.7×. **Our rules beat Learning to Fix at every speed-up** —
best mean gap at ≥ 2–3× 0.29 % (learned error-cost + adequacy guard) vs 0.55 % for the best LtF variant and 37.7 % for
LtF as published; at ≥ 5× 0.72 % (self-trained ranking + LP-relaxation guard, 100 % served) vs 0.80 %; at ≥ 10–20× only
our REINFORCE ranking has points (3.6 % mean at 33×) — but against the strongest LtF hybrid at matched 4.6–6.9× the
advantage (−0.07 to −0.12 pp) is **not significant** on the fresh set. On the original test set (60 instances) the
ordering is the same and every LtF point is significantly worse than our closest-speed rule (best mean gap at ≥ 2–5×:
0.40 % ours vs 0.76 % best LtF vs 7.97 % LtF-kNN). The paper's 0.48 % at 20.8× is not reproduced here.

## 1. What is known about the paper

Fritz, Makrides, Fetanat & Pinson, *Learning to Fix: Optimisation-Aware Machine Learning for Accelerated Unit
Commitment*, arXiv [2609.39396](https://arxiv.org/abs/2609.39396) (2026). The full text was **not reachable**
(arXiv, Semantic Scholar, ResearchGate, Hugging Face and archive.org are blocked by the network policy of this
environment; a direct fetch of `arxiv.org/html/2609.39396` was refused). Everything below comes from web-search
snippets of the abstract / HTML page (searched 2026-10-05):

| fact | source (search snippet) |
|---|---|
| UC is a MILP with many binary commitment decisions; ML predicts a subset of them to shrink the solver's search space | arXiv abstract / HTML |
| "an optimisation-aware framework that yields **generator-specific confidence thresholds** based on the impact of fixing errors on Unit Commitment solution quality" | abstract |
| existing confidence-based approaches fix variables with user-defined probability thresholds that are agnostic to the downstream optimisation | abstract |
| "a **decomposition algorithm** determines generator-specific confidence thresholds subject to a prescribed **cost tolerance on validation instances**, which controls the trade-off between problem reduction and solution quality" | HTML snippet |
| thresholds account for "the effects of binary variable fixing on UC **feasibility and operating cost**"; conservative thresholds limit savings, aggressive fixing can raise cost or make the reduced UC infeasible | HTML snippet |
| "Using a **k-nearest-neighbour (kNN) classifier and a 1 % validation cost tolerance**, the method achieves an average speed-up of 20.8, an average optimality gap of 0.48 %, and feasible solutions for 99.81 % of test instances" | abstract |
| first place in the EPRI 2025 *AI-ccelerating Unit Commitment* competition; "mean optimality gap below 0.5 % ... average speed-up of more than 20×" | abstract |
| "a general mechanism for translating machine learning predictions into controlled reductions of mixed-integer search spaces" | abstract |
| competition data (as described by search snippets of this paper and of [arXiv 2604.21891](https://arxiv.org/abs/2604.21891); competition page [epri.com](https://www.epri.com/ai-ccelerating-unit-commitment-competition), not fetched): simplified Irish system, **copper plate** (no network), 51 thermal units plus hydro, storage and aggregated wind / solar, **72-hour horizon**, hourly, 2,621 instances, UnitCommitment.jl formulation (19,296 binaries per instance) | search snippets |

Not known: the classifier's features and k; how the impact of a fixing error is measured (reduced MILP, LP, which
reference); what the decomposition algorithm decomposes and how it combines generators; whether the tolerance
bounds each generator or the total; whether there is any feasibility repair; the size of the validation set.

## 2. Reconstruction and assumptions

Pipeline: kNN on-probabilities → per-generator impact curves of fixing errors on validation instances →
thresholds θ_g by a decomposition under a cost tolerance τ → fix u[t, g] := round(p[t, g]) wherever
max(p, 1 − p) ≥ θ_g → reduced MILP.

* **A1 — classifier.** kNN over instance features: per-hour area loads (3 × 12), per-hour renewable availability
  by type (PV, rooftop PV, wind, hydro: 4 × 12) and the initial status of the 73 units; each block standardised and
  scaled to unit total variance (initial-status block weighted by w). p[t, g] = (1/distance-weighted) mean of the k
  nearest training instances' MILP schedules, **canonicalised inside groups of identical units** (RTS-GMLC has many
  identical copies; without it neighbours disagree only by swapped copies). Grid on val (log-loss against the
  canonical val labels): feature set {system-level, bus-level} × k ∈ {5, 11, 21, 41, 81} × w ∈ {0.5, 1, 2} ×
  {uniform, distance}. Chosen: system-level features, k = 21, w = 1, distance-weighted (`ltf_knn_select.json`).
* **A2 — calibration set.** The 60 validation instances (the paper calibrates on "validation instances"; ours is
  small, so the sensitivity to the calibration sample is measured with bootstrap resamples, see A6).
* **A3 — impact of fixing errors.** For generator g, threshold θ and calibration instance i: take the MILP optimum
  (aligned to the prediction inside identical-unit groups, so a swapped copy is not an error), force all of g's
  decisions with confidence ≥ θ to the prediction, repair g's row to the nearest schedule that satisfies min
  up/down with these entries fixed (none → impact = ∞, the reduced problem would be infeasible), and price the
  commitment with the exact dispatch LP; impact = relative cost increase over the optimum. *Compensated* variant:
  also price (a) merit-order replacement of the capacity g loses by units that are off in the optimum, (b) the
  cheapest single unit at least as large, (c) switching off the most expensive unit made redundant by a wrong ON
  fix, and keep the cheapest. Every priced schedule is feasible for the reduced MILP in which only g's decisions are
  fixed, so the impact **upper-bounds** that reduced MILP's cost increase (up to its 0.1 % MIP gap). This is the
  documented cheap proxy for "solve the reduced MILP with g's fixings": a reduced MILP with only one generator
  fixed is as hard as the full MILP (≈ 30 s), and one per (instance, generator, threshold level) would be ≈ 3,000
  MILPs per model. The curve is evaluated at the confidence levels of g's errors (the step points).
* **A4 — decomposition.** Mean impact over the calibration instances per generator and threshold gives one curve
  per generator. Two readings of "decomposition algorithm ... subject to a prescribed cost tolerance":
  *budget* (declared primary before any test run): maximise the mean fixed share subject to Σ_g impact_g(θ_g) ≤ τ —
  a Lagrangian decomposition (for a multiplier μ each generator independently maximises share_g − μ·impact_g; the
  feasible μ with the largest share is kept, then greedily filled to the budget). The headline 0.48 % test gap under a
  1 % tolerance fits a tolerance on the total; *each*: every generator's own impact ≤ τ at its threshold and at every
  higher one (a per-generator tolerance; also the reading of our earlier re-implementation in
  [`fixpolicy.md`](fixpolicy.md), which however tuned τ to hit a target share).
* **A5 — tolerance.** τ = 1 % as in the paper, plus a sweep τ ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8} % on val to trace the
  curve; the points tested are chosen on val.
* **A6 — seeds.** The kNN, the calibration on val and the thresholds are deterministic. The random element added
  is the calibration sample: three bootstrap resamples of the 60 validation instances (seeds 0, 1, 2), each giving
  its own thresholds, all evaluated on the test instances. (A variant adding out-of-fold training instances to the
  calibration set is coded in `uc_ltf_calib.py` but was not run.) The GNN checkpoints are the existing single-seed
  models (retraining them was out of the time budget).
* **A7 — the calibration is the contribution.** The same impact curves, decomposition and τ are applied to our GNN
  probabilities: MILP-label BCE (`uc_model1_4.pt`), self-trained (`selftrain_m3.pt`) and REINFORCE
  (`uc_model1_rl.pt`).
* Not reconstructed: any repair or fallback specific to the paper; their competition system (copper plate, 72 h,
  51 units, storage) — here RTS-GMLC with a DC network, 12 h, 73 units, soft shedding / reserve shortfall.

## 3. Calibration on validation (60 instances), and what was selected there

**Classifier.** kNN (system-level features, k = 21, distance weights) makes 28.2 wrong rounded decisions per val
instance against the canonical MILP labels (BCE GNN 14.8, self-trained 17.4, REINFORCE 31.0; log-loss 0.082 vs
0.055 / 0.071 / 0.106). Impact curves: 2,330 dispatch LPs for the kNN, ~9,000 for all four models (≈ 0.2 s each).

**Thresholds → fixed share** (compensated LP impacts; calibration impact = the decomposition's own estimate; share on
val / on test_fresh):

| model | budget τ = 0.1 % | 0.5 % | **1 %** | 2 % | 4 % | each τ = 0.1 % | each 1 % |
|---|---|---|---|---|---|---|---|
| kNN | 69.6 / 70.2 | 80.2 / 81.2 | **88.2 / 89.4** | 93.3 / 94.1 | 95.7 / 96.0 | 90.8 / 91.4 | 98.5 / 98.5 |
| BCE GNN | 88.0 / 88.1 | 93.9 / 93.9 | **95.9 / 95.7** | 98.0 / 97.8 | 99.0 / 99.0 | 93.5 / 93.3 | 99.6 / 99.6 |
| self-trained GNN | 81.3 / 81.5 | 89.2 / 89.5 | **92.7 / 92.9** | 96.0 / 96.2 | 98.9 / 98.9 | 90.6 / 90.8 | 99.8 / 99.9 |
| REINFORCE GNN | 74.7 / 74.7 | 82.2 / 82.2 | **85.8 / 85.9** | 89.7 / 89.8 | 95.7 / 95.9 | 82.2 / 82.1 | 96.8 / 97.0 |

The "each" reading lets every generator use the whole tolerance (summed estimate at τ = 1 %: 14.6 % for kNN), so it
fixes almost everything; "budget" is the conservative reading.

**Reduced MILPs on val** (in-sample: these instances calibrated the thresholds; first 9 of 20 planned instances,
stopped early to free the CPU; speed-ups against the dataset's MILP times, not back to back):

| kNN rule | fixed | mean gap | median gap | served | speed-up |
|---|---|---|---|---|---|
| budget, τ = 1 % | 87.5 % | 0.73 % | 0.16 % | 9/9 | 1.2× |
| budget, τ = 2 % | 92.7 % | 380 % | 0.27 % | 6/9 | 3.3× |
| budget, τ = 4 % | 95.4 % | 621 % | 2.9 % | 5/9 | 6.2× |
| each, τ = 0.1 % | 90.0 % | 1.07 % | 0.35 % | 9/9 | 1.7× |
| each, τ = 1 % | 98.4 % | 1,471 % | 328 % | 4/9 | 25.7× |
| budget, uncompensated LP impacts, τ = 4 % | 84.7 % | 0.23 % | 0.05 % | 9/9 | 1.2× |
| BCE GNN, budget τ = 2 % / self-trained 2 % / REINFORCE 2 % | 97.2 / 95.2 / 89.1 % | 115 / 65 / 0.51 % | 2.5 / 0.11 / 0.35 % | 4 / 7 / 9 of 9 | 9.2 / 5.3 / 3.4× |

* At the paper's τ = 1 % the separable budget decomposition keeps its promise in-sample (0.73 % ≤ 1 %). Above it,
  the generator-wise estimates break down: several "cheap" OFF errors of different generators, each compensated by
  other units when priced alone, are fixed together and the reduced MILP must shed load or miss reserve. A
  decomposition that prices generators separately cannot see this, whatever the impact measure.
* **Joint check (tested, not used for test runs).** Pricing the *joint* fixings with the same LP upper bound (all
  fixings forced together into the aligned optimum; free units compensating lost or surplus capacity in merit order)
  and bisecting the internal tolerance until the joint bound is ≤ τ (`otsl/ltf.py: JointCalibrator`) is safe but very
  conservative: the bound is loose by an order of magnitude (separable budget at 1 %: joint bound 35 % on val vs 0.73 %
  measured with reduced MILPs). At τ = 1 % it fixes only 60 % (kNN, budget + joint check), 70 % (kNN, sequential
  generator-by-generator decomposition with the joint check), 82 % (BCE), 90 % (self-trained), 76 % (REINFORCE). The
  joint-checked variants were not run on test (at 60–70 % fixed they cannot be faster than ≈ 1.2–1.5×; dropped for time).
* **Chosen for test** (val only): the separable budget decomposition with compensated impacts at τ = 1 % (the
  paper's tolerance) for all four models; τ = 2 % for kNN (the most aggressive point before val failures dominate, also
  with the LP guard) and for REINFORCE (its val result at 2 % was clean); in the second pass, "each" at τ = 0.1 %
  (the alternative reading at the largest tolerance that was clean on val) and three bootstrap resamples of the
  calibration instances (seeds 0–2) at τ = 1 %.

## 4. Results on the fresh test set (first 60 instances of `test_fresh`, seed 23)

Protocol: each worker solves the full MILP and then every reduced MILP of the same instance back to back (60 s
limit, 0.1 % gap, as in the dataset); 2 workers on a 4-core machine shared with two other agents' jobs (load ≈ 6–8), so
only ratios are comparable. Full MILP: 29.9 s on average, 8 of 60 at the time limit. Gaps are to the dataset MILP
objective; speed-up = mean full time / mean rule time (median of per-instance ratios in the results file). Every rule
uses the same instances; RACLearn, asymmetric, learned error-cost and LtF-BCE use the same BCE GNN.

| rule (B2, test_fresh 0–59) | fixed | mean gap | median gap | # gap > 10 % | served | speed-up |
|---|---|---|---|---|---|---|
| full MILP (back to back) | 0 | 0.03 % | 0.000 % | 0 | 100 % | 1× |
| **Learning to Fix, kNN, τ = 1 % (as published)** | 89.7 % | 37.7 % | 0.66 % | 10 | 75.0 % | 2.3× |
| Learning to Fix, kNN, τ = 2 % | 94.2 % | 500 % | 7.7 % | 27 | 48.3 % | 5.7× |
| Learning to Fix, kNN, τ = 2 % + LP guard | 76.5 % | 0.83 % | 0.32 % | 0 | 98.3 % | 2.1× |
| Learning to Fix on BCE GNN, τ = 1 % | 95.6 % | 21.1 % | 0.11 % | 9 | 76.7 % | 6.1× |
| Learning to Fix on self-trained GNN, τ = 1 % | 93.0 % | 1.06 % | 0.13 % | 2 | 91.7 % | 5.4× |
| Learning to Fix on REINFORCE GNN, τ = 1 % | 85.9 % | 0.55 % | 0.28 % | 0 | 98.3 % | 4.7× |
| Learning to Fix on REINFORCE GNN, τ = 2 % | 89.9 % | 0.80 % | 0.48 % | 0 | 93.3 % | 6.7× |
| RACLearn (BCE confidence), 90 % | 90.0 % | 12.4 % | 0.014 % | 4 | 91.7 % | 2.8× |
| RACLearn, 95 % | 95.0 % | 49.2 % | 0.079 % | 12 | 68.3 % | 8.1× |
| asymmetric + adequacy guard (ours), 90 % | 89.9 % | 3.91 % | 0.016 % | 1 | 95.0 % | 4.1× |
| asymmetric + adequacy guard, 95 % | 94.9 % | 16.9 % | 0.165 % | 6 | 85.0 % | 9.5× |
| learned error-cost + adequacy guard (ours), 90 % | 89.9 % | **0.29 %** | **0.010 %** | 0 | 98.3 % | 3.2× |
| learned error-cost + adequacy guard, 95 % | 94.9 % | 3.93 % | 0.048 % | 3 | 93.3 % | 5.7× |
| learned error-cost + adequacy + LP guard, 95 % target | 93.0 % | 0.43 % | 0.071 % | 0 | 98.3 % | 4.6× |
| self-trained, asym + guard (ours), 95 % | 94.9 % | 2.41 % | 0.40 % | 3 | 90.0 % | 8.6× |
| self-trained + LP guard, 95 % target | 93.0 % | 0.72 % | 0.29 % | 0 | **100 %** | 6.9× |
| REINFORCE probabilities (ours), 95 % | 95.0 % | 3.61 % | 1.40 % | 8 | 93.3 % | **33.1×** |
| REINFORCE probabilities, 97 % | 97.0 % | 5.12 % | 1.83 % | 11 | 85.0 % | 15.9× |
| REINFORCE + LP guard, 97 % target | 94.2 % | 4.41 % | 1.53 % | 9 | **100 %** | 27.9× |

**Equal speed-up** (best mean gap a family reaches with a point at least that fast; point, speed-up, served):

| family | ≥ 2× | ≥ 3× | ≥ 5× | ≥ 10× | ≥ 20× |
|---|---|---|---|---|---|
| Learning to Fix, kNN as published | 37.7 % (τ 1 %, 2.3×, 75 %) | 500 % (τ 2 %, 5.7×, 48 %) | 500 % | – | – |
| Learning to Fix, any model / τ / guard | 0.55 % (REINFORCE GNN, τ 1 %, 4.7×, 98 %) | 0.55 % | 0.80 % (REINFORCE GNN, τ 2 %, 6.7×, 93 %) | – | – |
| RACLearn | 12.4 % (90 %, 2.8×) | 49.2 % (95 %, 8.1×) | 49.2 % | – | – |
| ours, any ranking / guard | **0.29 %** (error-cost, 90 %, 3.2×, 98 %) | **0.29 %** | **0.72 %** (self-trained + LP guard, 6.9×, 100 %) | **3.61 %** (REINFORCE, 95 %, 33×, 93 %) | **3.61 %** |

Best **median** gap at ≥ 2 / 3 / 5 / 10 / 20×: ours 0.010 / 0.010 / 0.048 / 1.39 / 1.39 %; Learning to Fix (any variant)
0.11 / 0.11 / 0.11 % / – / – (kNN as published 0.66 / 7.7 / 7.7 % / – / –).

**Paired, at matched speed-up** (per-instance gap of our closest-speed rule minus Learning to Fix's, bootstrap 95 % CI
over the 60 instances): vs LtF-kNN τ 1 % (2.3×; ours: error-cost 90 %, 3.2×) −37.4 pp [−67.1, −13.8]; vs LtF-kNN τ 2 % +
LP guard (2.1×; same rule of ours) −0.54 pp [−0.90, −0.22]; vs LtF on REINFORCE τ 1 % (4.7×; ours: error-cost + LP guard,
4.6×) −0.12 pp [−0.31, +0.08]; vs LtF on REINFORCE τ 2 % (6.7×; ours: self-trained + LP guard, 6.9×) −0.07 pp [−0.34,
+0.21]; vs LtF on BCE (6.1×) −17.2 pp [−46.4, +1.4]; vs LtF on self-trained (5.4×; ours: error-cost 95 % without LP guard,
5.7×) +2.9 pp [−0.5, +9.0].

**Where it fails.** Instances with a > 10 % gap under LtF-kNN τ = 1 % carry 10.9 wrong OFF fixes and 0.1 wrong ON fixes
(against the aligned MILP solution), instances with ≤ 1 % gap 3.5 / 1.2 (`results/uc12/ltf_diag_fresh.json`). The
same signature holds for RACLearn (8.4 OFF) and the BCE variant (9.1 OFF): several individually compensable OFF errors
together remove the reserve. LtF on REINFORCE probabilities has no failure: the REINFORCE model over-commits (its
wrong fixes are mostly ON, 4.9 per instance), and the tolerance keeps its uncertain OFF decisions free.

**Seeds (calibration sample).** Second pass on the same 60 instances (own back-to-back full MILP, 30.0 s): kNN
thresholds calibrated on three bootstrap resamples of the 60 validation instances (seeds 0, 1, 2), budget, τ = 1 %:
fixed 89.0 % [87.1, 90.7], mean gap **56.1 % [38.3, 70.9]**, median 0.95 % [0.39, 1.48], served 75.0 % [71.7, 78.3],
speed-up 2.5× [2.2, 2.7] (mean [min, max] over seeds); 9–13 instances above 10 % for every seed. The full-val
calibration (37.7 %) is at the favourable end of that range; the failure of LtF-kNN on B2 does not depend on the
calibration sample. The alternative "each" reading at τ = 0.1 % (91.5 % fixed): 71.9 % mean, 1.00 % median, 78.3 %
served, 2.5×.

**Original test set (first 60 instances, separate pass with its own back-to-back full MILP, 27.6 s).** Same rules.
The re-run reproduces the earlier studies exactly where they overlap (RACLearn 1.384 / 38.509 %, asymmetric 0.413 /
14.004 %, learned error-cost 0.469 / 13.909 % mean gap at 90 / 95 %, REINFORCE 5.501 % at 95 %, error-cost + LP guard
0.398 %).

| rule (original test 0–59) | fixed | mean gap | median gap | # > 10 % | served | speed-up |
|---|---|---|---|---|---|---|
| full MILP | 0 | 0.00 % | 0.000 % | 0 | 88.3 % | 1× |
| Learning to Fix, kNN, τ = 1 % | 89.2 % | 7.97 % | 0.36 % | 7 | 78.3 % | 2.5× |
| Learning to Fix, kNN, τ = 2 % | 94.0 % | 118 % | 1.75 % | 19 | 61.7 % | 5.7× |
| Learning to Fix, kNN, τ = 2 % + LP guard | 81.1 % | 1.60 % | 0.13 % | 2 | 96.7 % | 3.5× |
| Learning to Fix on BCE / self-trained GNN, τ = 1 % | 96.3 / 93.2 % | 32.1 / 15.2 % | 0.47 / 0.32 % | 13 / 4 | 60.0 / 78.3 % | 7.5 / 5.6× |
| Learning to Fix on REINFORCE GNN, τ = 1 % / 2 % | 85.9 / 89.8 % | 0.76 / 1.02 % | 0.21 / 0.29 % | 0 / 1 | 90.0 / 93.3 % | 3.1 / 3.9× |
| RACLearn 90 / 95 % | 90 / 95 % | 1.38 / 38.5 % | 0.000 / 0.087 % | 2 / 7 | 86.7 / 80.0 % | 2.7 / 6.3× |
| asymmetric + adequacy guard (ours), 90 % | 89.9 % | 0.41 % | 0.055 % | 0 | 96.7 % | 4.1× |
| learned error-cost + adequacy guard (ours), 90 % | 89.9 % | 0.47 % | 0.002 % | 0 | 90.0 % | 4.3× |
| learned error-cost + adequacy + LP guard, 95 % target | 92.1 % | **0.40 %** | 0.052 % | 0 | 96.7 % | 5.6× |
| self-trained + LP guard, 95 % target | 93.0 % | 0.81 % | 0.095 % | 0 | 96.7 % | 7.1× |
| self-trained, asym + guard, 95 % | 94.9 % | 1.99 % | 0.28 % | 2 | 86.7 % | 10.4× |
| REINFORCE probabilities, 95 % | 95.0 % | 5.50 % | 1.48 % | 10 | 91.7 % | 26.0× |
| REINFORCE + LP guard, 97 % target | 93.7 % | 5.60 % | 2.17 % | 13 | 96.7 % | 30.9× |

Best mean gap at ≥ 2 / 3 / 5 / 10 / 20×: **ours 0.40 / 0.40 / 0.40 / 1.99 / 5.50 %**; Learning to Fix, any variant 0.76 /
0.76 / 15.2 % / – / –; kNN as published 7.97 / 118 / 118 % / – / –. Paired at matched speed-up, ours minus LtF: vs
LtF on REINFORCE τ = 1 % (3.1×; asymmetric 90 %, 4.1×) −0.34 pp [−0.62, −0.11]; τ = 2 % (3.9×) −0.60 pp [−1.08,
−0.23]; vs LtF-kNN τ = 1 % (2.5×) −7.6 pp [−14.5, −1.6]; vs LtF-kNN τ = 2 % + LP guard −1.19 pp [−2.44, −0.23]; all
LtF points are worse with a CI excluding zero. Failure signature as on the fresh set: LtF-kNN instances with > 10 % gap
carry 7.9 wrong OFF fixes (1.8 on instances with ≤ 1 %).

**How good is the LP impact proxy?** (`scripts/uc_ltf_proxycheck.py`, `results/uc12/ltf_proxycheck.json`) For 24
random calibration cells (val instance, generator, threshold level) of the kNN curves with a compensated LP impact
> 0.1 %, the reduced MILP with only that generator's fixings (all else free; 60 s, 0.1 %) was solved. Mean impact:
LP without compensation 182 %, with compensation 3.6 %, reduced MILP 0.51 %; median ratio MILP / compensated LP 0.21;
in 75 % of cells the MILP impact is below half the LP bound; rank correlation 0.10. The LP bound is valid but loose
and nearly uninformative about which generator's errors cost most once other units may react. Measuring impacts
with reduced MILPs (≈ 2,300 solves per model here, ≈ 20 core-hours) would give *smaller* per-generator impacts, hence
*more* fixing at the same τ — and, since the failures are joint, probably more catastrophic instances, not fewer.
This is the main untested difference to the paper.

## 5. Verdict

* **Learning to Fix as published (kNN + per-generator cost-tolerance thresholds) does not carry over to B2.** At the
  paper's τ = 1 % it fixes 90 % of the decisions, reaches 2.3× and has a 37.7 % mean gap (0.66 % median, 10 of 60
  instances above 10 %, 75 % served); any larger tolerance is catastrophic. The paper's 0.48 % at 20.8× on the EPRI
  system is not reproduced: our kNN is twice as wrong as the GNN (28 vs 14 wrong decisions per instance), and on a
  network-constrained system with soft reserve, fixing errors that are harmless one generator at a time combine into
  reserve shortfall and shedding, which a generator-wise decomposition cannot see.
* **Our methods beat it at every speed-up.** Best mean gap at ≥ 2–3×: 0.29 % (learned error-cost + adequacy guard,
  90 %) vs 37.7 % (LtF-kNN) — paired difference −37 pp, CI excluding zero; at ≥ 5×: 0.72 % (self-trained + LP guard,
  100 % served) vs 500 %; at ≥ 10× and ≥ 20× only our REINFORCE ranking has points (3.6 % mean, 1.4 % median at 33×;
  4.4 % mean, 100 % served at 28× with the LP guard).
* **The calibration is worth more than the kNN.** Applied to our REINFORCE probabilities, the same cost-tolerance
  thresholds give the best Learning-to-Fix points: 0.55 % mean / 0.28 % median / 98 % served at 4.7× (τ = 1 %), 0.80 %
  at 6.7× (τ = 2 %), with no instance above 10 %. Against these, our best rules at the same speed are better on average
  but **not significantly** (−0.12 pp [−0.31, +0.08] at 4.7×; −0.07 pp [−0.34, +0.21] at 6.9×), with lower medians
  (0.07 vs 0.28 %; 0.29 vs 0.48 %). The LtF calibration on the BCE GNN (21 % mean) and the self-trained GNN (1.06 %) is
  worse; the probability model matters as much as the threshold rule.
* **Second test set.** On the original test set's first 60 instances the picture is the same and sharper: best mean
  gap at ≥ 2–5× 0.40 % (ours, error-cost + LP guard, 5.6×) vs 0.76 % at 3.1× (LtF on REINFORCE) and 7.97 % (LtF-kNN);
  every LtF point is significantly worse than our closest-speed rule (e.g. −0.34 pp [−0.62, −0.11] against LtF on
  REINFORCE τ = 1 %).
* **Net:** on B2 our rules dominate Learning to Fix as published (kNN) on both test sets and at every speed-up. Against
  the strongest hybrid we could build from it (its calibration on our cost-aware REINFORCE probabilities) they are
  better on both test sets — significantly on the original one, within noise on the fresh one at 4.5–7× — and they are
  the only rules beyond ~7×.

## 6. Where this reconstruction may differ from the paper

1. **Impact measure.** Dispatch-LP upper bounds with heuristic compensation instead of (probably) reduced-MILP solves
   per generator and threshold. The bound is tight when no compensation is possible and loose otherwise (see the proxy
   check above). A tighter, MILP-measured impact would allow *more* fixing (higher speed), and would not cure the joint
   failures, which come from combining generators.
2. **Decomposition.** "Budget" (sum of generator impacts ≤ τ, Lagrangian) is our reading; the paper's algorithm may
   be sequential or include a joint validation check. Our joint-checked variants are safe but fix only 60–70 % (kNN)
   because the LP bound of the joint fixings is loose; with reduced-MILP measurements a joint check would be the
   natural fix and would land between the two.
3. **Classifier.** Features, k and label canonicalisation are ours; the paper's kNN may use other features (on the
   EPRI data there is no network, so system-level load / wind / solar series are natural).
4. **System and horizon.** EPRI: copper-plate Irish system, 51 thermal units plus storage and hydro, 72 h, 2,621
   instances. Here: RTS-GMLC with a DC network, 73 units, 12 h, 500 training and 60 validation instances. A 72-hour
   copper-plate MILP is far slower than our 30 s, so 20× is attainable there; here the full MILP takes ≈ 30 s and the
   fixed share needed for 20× (≥ 95 %) is beyond what any calibrated threshold rule fixes safely.
5. **Feasibility.** The paper reports 99.81 % feasible reduced problems; with soft balance and reserve every reduced
   MILP here is feasible unless fixings conflict with min up/down times (fallback to the full MILP, counted in time).
   LtF-kNN was 100 % feasible, LtF on BCE 93 %.

## 7. Limitations

* Selection used 9 in-sample validation instances with reduced MILPs (the run was cut to free the CPU) plus the LP
  calibration on all 60; the test tolerances were fixed before the test runs, but they are few.
* One seed for every GNN (existing checkpoints); the seeded part is the calibration sample (bootstrap of the 60 val
  instances). The learned error-cost ranking is the existing 3-seed ensemble.
* Timings come from a shared machine (load 6–8 on 4 cores, two other agents); ratios of means are dominated by slow
  instances (e.g. REINFORCE 97 %: 15.9× mean ratio, 57.6× median ratio). The LP-guard time is included.
* Mean gaps are driven by a few penalty-priced instances (VOLL $10,000/MWh); medians, served shares and the count of
  > 10 % gaps are reported for that reason.
* The fresh test set shares calendar days with the original test set (different random scenarios). The original
  test set was used by earlier studies whose rules (asymmetric, error-cost, LP guard) were partly designed after
  seeing it; the fresh set is the clean comparison.

## 8. Reproduce

```bash
python3 scripts/uc_ltf_prep.py                      # kNN grid on val, probabilities of kNN / BCE / self-trained / REINFORCE
python3 scripts/uc_ltf_curves.py --sets knn_va,bce_va,st_va,rl_va   # per-generator impact curves (LP), ~9,000 LPs
python3 scripts/uc_ltf_calib.py                     # separable thresholds (budget / each), tau grid, bootstrap seeds
python3 scripts/uc_ltf_seq.py --plan knn:budgetJ:0.01 knn:seq:0.01 knn:budgetJ:0.02,0.04 bce:budgetJ:0.01 \
    st:budgetJ:0.01 rl:budgetJ:0.01                 # joint-checked variants (val only)
python3 scripts/uc_ltf_eval.py --split val --n 20 --tag val --rules ltf:knn:budget:comp:0.01,...   # val check (9 done)
R=ltf:knn:budget:comp:0.01,ltf:knn:budget:comp:0.02,ltf:bce:budget:comp:0.01,ltf:st:budget:comp:0.01,\
ltf:rl:budget:comp:0.01,ltf:rl:budget:comp:0.02,rac@0.9,rac@0.95,asym@0.9,asym@0.95,harm@0.9,harm@0.95,st@0.95,\
rl@0.95,rl@0.97,harm@0.95+lp,st@0.95+lp,rl@0.97+lp,ltf:knn:budget:comp:0.02+lp
python3 scripts/uc_ltf_eval.py --split test_fresh --n 60 --full --tag fresh --rules $R
python3 scripts/uc_ltf_eval.py --split test_fresh --n 60 --full --tag fresh_b \
    --rules ltf:knn_bs0:budget:comp:0.01,ltf:knn_bs1:budget:comp:0.01,ltf:knn_bs2:budget:comp:0.01,ltf:knn:each:comp:0.001
python3 scripts/uc_ltf_proxycheck.py --n 24         # LP impact vs single-generator reduced MILP
python3 scripts/uc_ltf_eval.py --split test --n 60 --full --tag test --rules $R
python3 scripts/uc_ltf_diag.py && python3 scripts/uc_ltf_report.py
```
Worker pools use 2 spawn workers; PyTorch runs single-threaded. Rule names are documented in `scripts/uc_ltf_eval.py`.
