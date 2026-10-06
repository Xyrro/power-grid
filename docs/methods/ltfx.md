# Learning to Fix, implemented from the full paper (B2, 12-hour network-constrained UC)

Code: [`otsl/ltfx.py`](../../otsl/ltfx.py) (kNN, fixing rule, Appendix A master, check / relaxation MILPs,
Algorithm 1 + 2), `scripts/uc_ltfx_{prep,tune,eval,report,diag}.py`. Results:
[`results/uc12/ltfx_results.md`](../../results/uc12/ltfx_results.md) (+ `.json`), raw solves
`results/uc12/ltfx_eval_test_fresh.jsonl`, tuning logs and thresholds `results/uc12/ltfx_tune_<model>_<eps>.{log,json}`,
probabilities `results/uc12/ltfx_probs.npz`, diagnostics `results/uc12/ltfx_diag.json`. New data:
`data/generated/uc12/val_extra.npz` (120 validation-day instances, seed 31).

SUMMARY_PLACEHOLDER

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

Everything in §1 is implemented as written; the table lists the setting of each element here and marks what is a
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
| relaxation MILP (Algorithm 2) | min Σ ν, solved (presumably to optimality), K_max = 10 | same MILP, K_max = 10, but **6 s per solve (deviation, for compute)**: the best release set found is used even if not minimal, and the K-loop stops at the first relaxation that finds no solution in 6 s, so most cuts carry 1–2 alternatives instead of up to 10. The first solve of an instance is warm-started from its optimal schedule aligned to the fixings inside identical-unit groups (always feasible; *exact*). ν = 1 entries whose u equals the fixed value are dropped from the release set (*exact*: the same solution satisfies those fixings) |
| caching | – | check results are reused by monotonicity: a fixing set contained in one that passed passes with the same witness schedule, one containing a failed set fails; relaxation solutions are stored as witnesses (*exact*) |
| budget | – | 75 min per tuning run; a run that hits it stops, re-solves the master and reports how many validation instances its thresholds still violate |
| verification | "upon termination the thresholds admit a solution within ε for every validation instance" | after termination, every validation instance's stored witness schedule is checked against the final fixings and priced with the exact dispatch LP (min up / down rows included) |
| other classifiers | CatBoost (Appendix C) | the same tuning on our GNN probabilities: MILP-label BCE (`uc_model1_4.pt`), self-trained (`selftrain_m3.pt`), MILP-label REINFORCE (`uc_model1_rl.pt`); "our best GNN" = self-trained, the probability source chosen on validation for fixing in [`combo.md`](combo.md) (rule 3) |

**A property of the exact algorithm worth knowing.** With the probability-mass objective the master prefers
zero-width intervals (τ̲_g = τ̄_g: everything of g fixed). A cut τ̲_g ≤ π (release an OFF fix) is then satisfied by
*sliding* the collapsed interval down rather than widening it, which turns every decision of g with probability
between the old and the new position from fixed-OFF into fixed-ON; the next check of that instance (or another)
fails on those ON fixes and adds τ̄_g ≥ π. The same instance therefore fails several times and the intervals open
one side at a time. This is what Algorithm 1 does as written (verified on the master alone:
`scratchpad` test, a single-generator cut moves both thresholds of that generator and nothing else); it costs
iterations, not correctness.

## 3. Tuning on validation

TUNING_PLACEHOLDER

## 4. Test results (first 60 instances of test_fresh, full MILP and every reduced MILP back to back)

TEST_PLACEHOLDER

## 5. Faithful version vs the earlier per-generator reconstruction

DIAG_PLACEHOLDER

## 6. Verdict

VERDICT_PLACEHOLDER

## 7. Limitations

LIMITS_PLACEHOLDER

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
