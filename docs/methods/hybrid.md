# Hybrid fixing rules: Learning to Fix's calibration with our probabilities, error-cost scores and guards (B2)

Code: [`otsl/hybrid.py`](../../otsl/hybrid.py) (guard-aware tuner, error-cost score transform, test-time rule; it
subclasses `otsl.ltfx.LtFTuner` and imports the guards of `otsl.combo` / `otsl.fixpolicy` unchanged),
`scripts/uc_hybrid_{train,prep,tune,eval,report,run}.py`. Results:
[`results/uc12/hybrid_results.md`](../../results/uc12/hybrid_results.md) (+ `.json`), raw solves
`results/uc12/hybrid_eval_test_fresh.jsonl`, tuning logs, thresholds and cuts `results/uc12/hybrid_tune_<tag>.{log,json}`,
`hybrid_tune_<tag>_cuts.jsonl`, held-out checks `hybrid_holdout_*.json`, run log `hybrid_run.log`. New data:
`data/generated/uc12/val_extra2.npz` (180 validation-day instances, seed 41).

__SUMMARY__

## 1. Question

On the 12-hour benchmark (uc12, first 60 instances of `test_fresh`, paper metrics) the faithful Learning to Fix
(LtF; Fritz et al. 2026, [`ltfx.md`](ltfx.md)) and our own fixing rules were level: LtF on our BCE GNN at ε = 1 %
0.28 % gap at 5.2×, LtF-kNN 0.40 % at 4.7×, our error-cost rule + adequacy guard 0.42 % at 4.2×; at ≈ 13.5× LtF on
the self-trained GNN (ε = 10 %) 0.85 % and our combined pipeline (98 %) 0.82 %. Can the paper's joint, ε-guaranteed
threshold calibration and our components (better probabilities, error-cost scores, guards) be combined into a rule
that beats LtF at equal speed-up?

## 2. Hybrid rules

A hybrid rule is: score s[t, g] → generator-specific grey zone [τ̲_g, τ̄_g] (eq. 5) → fixings → guards → reduced
MILP. The thresholds are tuned by the paper's Algorithm 1 + 2 with its Appendix A objective and the same settings as
the faithful runs (Q = 20, K_max = 10, relaxation MILPs capped at 6 s, check MILPs 60 s with the (1 + ε)·C* cut-off,
C* = dataset MILP objective), so that **every validation instance's reduced UC, all fixings applied jointly and after
the guards, keeps a solution within ε of C***.

**(a) Guard-aware tuning.** `HybridTuner` (a subclass of the faithful tuner; nothing in `otsl/ltfx.py` changes)
applies the test-time guards inside the check: the fixings of a candidate threshold vector are passed through the
guards, in the order and with the code used at test time, and both the check MILP and the release-set MILPs of
Algorithm 2 see the guarded fixings. A cut therefore only asks the thresholds to release what the guards do not
already release, and the termination guarantee holds for the guarded rule. Guard outputs are memoised per instance and
pre-guard fixing set; the monotone check cache of the faithful tuner works on the guarded sets; cuts are logged
so that a run can be warm-started on a larger validation set (each cut comes from one instance and stays valid).

Two guard sets in the check were tried:

* **All guards (adequacy + min up/down rows + LP relaxation)**, as first planned ("hn" runs). A 6-instance pilot
  showed the problem: the master maximises the probability mass fixed *before* the guards, so it collapses the grey
  zones (99.8 % fixed before the guards) and lets the LP-relaxation guard rescue every instance; that guard releases
  every OFF fix in the min-down window of each hour with penalised slack, often hundreds of fixings per instance
  (up to 787 of 870 on one instance), and only 76.7 % stayed fixed — less than the faithful thresholds on the same
  instances (83.6 % on 180). The master cannot see what the guards release.
* **Targeted guards: adequacy + min up/down rows** ("hg", "he" runs; the main design). Both release few, specific
  fixings (2–7 OFF fixes for adequacy, one 12-hour row per min up/down conflict at fully collapsed thresholds), so
  the master's objective stays close to the fixed share after the guards. The row release is the "projection onto
  the generator polytope" the paper proposes for its own infeasible test instances (Sec. IV-C). The LP-relaxation
  guard can be added at test time ("+lp"): it only releases fixings, so the validation guarantee is kept.

**(b) Error-cost scores.** The thresholded score is the learned expected cost of a wrong fix h[t, g] (harm
ensemble of [`fixpolicy.md`](fixpolicy.md), seed s for probability seed s, features computed from the probabilities)
mapped to a pseudo-probability that keeps the rounding direction: s = 0.5·q(h) for predicted OFF and 1 − 0.5·q(h) for
predicted ON, with q(h) = sigmoid((log h − μ) / σ) (μ, σ over the first 180 validation instances). q is strictly
increasing, so τ̲_g is a unit-specific error-cost level below which OFF predictions are fixed, τ̄_g the same for
ON predictions, and τ̲ = τ̄ = 0.5 fixes everything by rounding. Generator-specific grey zones on the expected error
cost, calibrated jointly per validation instance.

**(c) Probabilities.** The paper's selection criterion for a classifier (largest validation fixed share at the same
guarantee) applied to the faithful runs on 180 instances gives the MILP-label BCE GNN (83.6 % at ε = 1 %; self-trained
75.3 %, REINFORCE 70.9 %, kNN 66.7 %); it also has the lowest validation log-loss (0.055–0.056 vs 0.064–0.075 for the
three self-trained seeds). The BCE GNN was retrained with seeds 1 and 2 (recipe of `uc_model1_4.pt`, which is seed 0:
GNN + symmetry + LP-relaxation features, canonical labels, 80 epochs, early stopping on val). The self-trained model
already has 3 seeds (`combo_st_s*.pt`). The REINFORCE GNN was not retrained (≈ 30 min per seed on the one LP worker
allowed here, and it is the weakest source by the paper's criterion).

## 3. Protocol

* **Validation (≈ 360 instances).** val (60) + val_extra (120) + val_extra2 (180 new, `uc_gen.py --cfg uc12 --split
  val_extra2 --n 180 --seed 41`): validation days, 60 s / 0.1 % reference MILPs. Runs on 180 instances use the first
  two parts (the faithful runs' set); runs on 360 are warm-started with the cuts of the 180 run of the same rule, and
  the 180-instance thresholds are first checked on the 180 new instances (held-out violation rate).
* **Selection, fixed before any result** (`scripts/uc_hybrid_run.py`): the family (hg: guard-aware LtF on
  probabilities; he: on error-cost scores) with the larger validation fixed share after the guards at ε = 1 %, seed 0,
  180 instances (the paper's criterion), gets seeds 1 and 2, the 360-instance runs and an ε = 5 % run; the other family
  keeps seed 0. Time-boxed: optional runs were skipped when they would not finish before the test pass.
* **Test.** First 60 instances of `test_fresh` (never used for a decision). One worker solves the full MILP (60 s,
  0.1 %) and then every reduced MILP back to back; the faithful LtF rules (kNN and BCE at ε = 1 %, self-trained at
  ε = 10 %), the faithful BCE thresholds with the guards added at test only, and our combined pipeline (98 %) are
  re-run in the same worker, so every speed-up is against the same full-MILP solve. Method time = reduced MILP +
  inference (kNN search, GNN forward pass, error-cost features and ensemble) + the LP relaxation that feeds the GNN +
  guards.
* **Metrics.** Paper metrics (gap to the back-to-back full MILP's dual bound, feasibility, mean / max per-instance
  speed-up over feasible instances, fixed share after guards); our convention (gap to the dataset MILP objective over
  all instances with a full-MILP fallback for infeasible reduced problems, served share, ratio of mean times);
  per-seed statistics with mean [min, max] over seeds; paired instance-bootstrap CIs against faithful LtF-kNN and
  LtF-BCE; best gap at speed-up ≥ 3 / 5 / 8 / 12 / 20×.
* **Compute.** One core for all MILP / LP work (the machine was shared with two other agents' MILP jobs, load 4–6 on
  4 cores), HiGHS and PyTorch single-threaded; every heavy job ran alone, in the order of `hybrid_run.log`.

__RESULTS__
