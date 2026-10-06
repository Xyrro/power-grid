# Learning to Fix on the 24-hour benchmark (uc24), with our probabilities and guards

*Code: [`otsl/uc24ltf.py`](../../otsl/uc24ltf.py) (warm-start and guard-aware tuners, soft LP guard),
`scripts/uc_uc24ltf_{gen,prep,tune,eval,report}.py`, queues `scripts/uc_uc24ltf_{chain,next,next2}.sh`. Results:
[`results/uc24/uc24ltf_results.md`](../../results/uc24/uc24ltf_results.md) (+ `.json`), raw test solves
`results/uc24/uc24ltf_eval_test.jsonl`, tuning logs / thresholds `results/uc24/uc24ltf_tune_<rule>.{log,json}`,
probabilities `results/uc24/uc24ltf_probs.{npz,json}`, run logs `results/uc24/uc24ltf_*.log`. New data:
`data/generated/uc24/uc24ltf_val.npz` (validation with full MILPs), `uc24ltf_train_lab.npz` (labelled training
subset).*

**Summary.** uc24 was given what Learning to Fix needs — full MILPs for 40 validation instances (1.18 solver
core-hours) and 0.5 %-gap MILP schedules for 120 training instances (0.92 core-h) — and the faithful method
(`otsl/ltfx.py`: kNN k = 50, Algorithm 1 + 2) was tuned on validation at ε = 1 % and 5 % (both converged), together
with LtF on our label-free imitation GNN and a guard-aware LtF (adequacy + LP-relaxation guards inside the check)
at ε = 5 % (both stopped at the 22-min budget; the ε = 1 % GNN runs were dropped for compute). On the 40 test
instances (paper metrics): **LtF-kNN ε = 1 %: 0.60 % gap to the dual bound, 100 % feasible, 7.1× mean speed-up
(5.6× load-corrected, 4.1× median), 67 % fixed; ε = 5 %: 2.14 %, 92.5 % feasible, 25.5× (20×; median 6.6×)** — the
paper's quality at ε = 1 % but a quarter of its 20.8×. Our label-free guarded rule (95 % target) is level in gap with
LtF ε = 1 % (0.67 %, paired +0.06 pp [−0.14, +0.26]) at twice its speed-up (11.6× load-corrected) and needs no
labels; REINFORCE 90 % is level with LtF ε = 5 % at a higher speed-up. The non-converged GNN hybrids fail on test
(27–38 % infeasible); with our conflict release first and the guards after it, the guard-aware hybrid is always
feasible but gives 1.08 % at 11.8× (load-corrected), worse than our guarded rule at the same speed (−0.41 pp). In time to the same quality, LtF buys
1.3× (ε = 1 %) and 0.9× (ε = 5 %); our rules 2–3×.

## 1. Why

On the 12-hour benchmark the faithful Learning to Fix (LtF; Fritz et al. 2026, [`ltfx.md`](ltfx.md)) reproduced the
paper's solution quality but not its speed-up, and the obvious explanation was the benchmark: a 20 s MILP leaves
little to gain. uc24 ([`b3.md`](b3.md)) is the benchmark where the full MILP takes minutes (154 s mean, 30 % at the
300 s limit), i.e. the paper's regime (Gurobi, 60 s mean). uc24 had been built label-free — no full MILP for
training or validation — so LtF could not run on it: its tuning needs C\* on validation instances and its kNN needs
labelled training schedules. This study adds both, tunes LtF, and evaluates it against our label-free rules.

## 2. What was added to uc24 (and what it cost)

| item | instances | settings | solve time mean / median | proven | solver core-hours (wall of the solves; CPU) |
|---|---|---|---|---|---|
| validation full MILPs (`uc24ltf_val.npz`) | 40 | 0.1 %, 300 s, 1 thread | 106 s / 65 s | 87.5 % (final gap mean 0.10 %, max 0.34 %) | **1.18 h** (1.12 h CPU) |
| kNN training labels (`uc24ltf_train_lab.npz`) | 120 | 0.5 %, 60 s, 1 thread | 28 s / 24 s | 86 % reached 0.5 % (mean gap 0.35 %, max 3.0 %) | **0.92 h** (0.90 h CPU) |
| LtF tuning, 4 runs (§4) | – | one core per run | – | – | **1.26 h** CPU (1.29 h wall) for the 4 runs; + ≈ 0.2 h for smoke tests and an aborted first kNN run |
| test evaluation (§5) | 40 | reduced MILPs, 0.1 %, 300 s | – | – | **1.39 h** of solver time (6 rules × 40 + 3 back-to-back full MILPs) + 0.29 h and 0.21 h for the two test-time conflict-release passes |
| *for comparison: label-free training of the GNNs ([`b3.md`](b3.md))* | 300 | LP relaxations + dispatch LPs | – | – | *≈ 0.8 h* |

All solves ran in one worker process (one core) while two other agents shared the 4-core machine (1-minute load
average ≈ 4 during the label runs, against ≈ 6.7 when the test set was generated).

* **Validation (40 instances, full MILP, 0.1 % gap, 300 s, 1 thread — the test settings).** The jobs are
  `make_jobs(val days, 40, seed 202)`: the first 30 are exactly the 30 label-free instances of `val.npz` (same day /
  seed draws, checked), 10 are new validation-day instances. 40 is the lower end of the requested 40–60, chosen for
  the one-worker budget (each tuning run also scales with the number of validation instances).
* **Labelled training schedules for the kNN (120 instances, full MILP at 0.5 % gap, 60 s limit).** The first 120
  of the 300 training instances (their order is already random), same scenarios (day / start / seed re-used;
  loads checked identical). The looser gap and shorter limit follow the probe of `b3.md`: the MILP's incumbent is
  within 0.5 % of its final one after a median 28 s, so a 0.5 %-gap stop gives near-optimal schedules at a fraction
  of the 0.1 % cost. *Not done*: self-training labels from reduced MILPs (`otsl/selftrain.py`) as a label-cheap kNN
  variant — a reduced MILP on uc24 costs 10–45 s, about as much as the 0.5 %-gap full MILP, so it would not have been
  cheaper here.

## 3. Methods compared

| rule | probabilities | thresholds | guards |
|---|---|---|---|
| LtF kNN ε | kNN, k = 50, Table II features, eq. 13 (`otsl.ltfx.KNNProb`) on the 120 labelled schedules | Algorithm 1 + 2 on the 40 validation instances | none (faithful) |
| LtF GNN ε | our label-free GNN (the one with the lower validation log-loss; §4) | Algorithm 1 + 2 | none |
| LtF+guards GNN ε | same GNN | Algorithm 1 + 2 with the guards **inside** every check, cut and verification (`GuardedLtFTuner`) | adequacy guard + soft LP-relaxation guard, also at test |
| … + conflict release (test only) | same | same thresholds | our min up/down conflict release added at test time (after the guards, or first: "conflict release first") |
| old: 95 % target, guarded (`95%|best=bce_g`) | imitation GNN | ranking, top 95 % | adequacy + LP guard + conflict release ([`b3.md`](b3.md)) |
| old: REINFORCE ranking 95 % (`95%|rl`) | REINFORCE GNN | ranking, top 95 % | conflict release |
| stored: RACLearn-style 80 / 90 / 95 %, other b3 rules | | | earlier records (`b3_fix_test*.jsonl`) |

**Faithful LtF** is `otsl/ltfx.py` (see [`ltfx.md`](ltfx.md) §1–2 for the method and every choice where the paper is
silent) with one addition that only changes where HiGHS starts: better warm starts for the relaxation MILP of
ADD CUTS (`StartLtFTuner`, §4). Settings on uc24: Q = 20, K_max = 10, check = reduced MILP with the (1 + ε)·C\* cut-off, stopped at
the first solution below it, 60 s limit (no answer = violation); **relaxation MILPs capped at 8 s (deviation, for
compute; 6 s on uc12)**; a budget of 22 min per tuning run (a run that hits it stops and is marked "not converged";
its thresholds then violate some validation instances). C\* = the 300 s / 0.1 % MILP objective.

**Guard-aware LtF** (the hybrid; the same idea the hybrid study tests on uc12). The fixings that a threshold pair gives
on a validation instance are first passed through our guards — the adequacy guard (OFF fixes released, cheapest
first, until the units not fixed off cover net load + reserve + 5 % in every hour) and the LP-relaxation guard (release
fixings that make the relaxed reduced problem shed load, over-generate or miss reserve) — and the check, the
relaxation MILP of ADD CUTS and the final verification all see the guarded set. The cuts stay conditions on the
thresholds, so the master problem is unchanged; the guards only release, so a guarded set is a subset of the plain
one and the tuning can accept tighter thresholds wherever the guards remove the harmful fixings instance by instance.
Two design decisions came from a 5-instance smoke test:

* *The min up/down conflict release of our earlier pipelines is not part of the chain.* With it, the very first
  master solution (all intervals collapsed: 100 % fixed by generator-specific rounding) passed every check, because
  the conflict release removed whole generator rows: 100 % fixed before the guards, 42 % after. The tuning then
  "converges" in one iteration to a slow rule. Conflicts are left to the LtF cuts, as in the plain method.
* *The LP guard is made a no-op when the relaxed reduced problem is infeasible* (`lp_guard_soft`; infeasibility can
  only come from a min up/down conflict, since shedding, spill and reserve shortfall are priced slacks). The original
  guard marks every hour bad in that case and releases every OFF fixing — the same collapse.

With these two changes the smoke run behaved like plain LtF (cuts on conflicts) while the guards released ~7 % of the
fixings that the thresholds alone would keep (98.1 % → 91.6 % fixed on the 5 instances).

**Test protocol.** 40 test instances (`test.npz`), one worker, every rule of an instance solved back to back in that
worker; reduced MILPs with the full-MILP settings (HiGHS 1.15 via highspy, 0.1 %, 300 s, 1 thread, incumbent log). No
fallback: a reduced problem without a solution is infeasible (the paper's convention; our convention charges it with
the full MILP). Method time = inference (kNN: Table II features + neighbour search; GNN: forward pass + the LP
relaxation that feeds its features, timed in the worker) + fixing rule + guards + reduced MILP. Full-MILP time = the
dataset run of the test set; the first 3 instances are re-solved back to back in the evaluation worker to measure the
machine-load drift, and the two most relevant earlier rules are re-run in the same worker so that all compared rules
share conditions.

## 4. Probabilities and tuning on validation

**Probabilities (40 validation instances, against their canonical MILP schedules).**

| model | log-loss | wrong decisions at 0.5 per instance (of 1,752) | probabilities exactly 0 or 1 | fixed by worst-case thresholds (eq. 7) |
|---|---|---|---|---|
| kNN, k = 50, 120 labelled schedules | 0.119 | 86.7 | 66 % | 61.4 % |
| our imitation GNN (`b3_lf_bce.pt`, label-free) | **0.086** | **40.0** | 0 % | 67.4 % |
| our REINFORCE GNN (`b3_rl_selected.pt`, label-free) | 0.135 | 85.0 | 0 % | 63.5 % |

The imitation GNN was chosen for Learning to Fix by the pre-declared rule (lower validation log-loss). Without a single
MILP label in its training it is better calibrated to the MILP schedules than the kNN trained on 120 of them. (kNN
log-loss by k, diagnostic only: 0.140 / 0.105 / 0.103 / 0.119 / 0.153 for k = 5 / 10 / 20 / 50 / 100; k = 50 is the
paper's and was kept.)

**Tuning runs** (one core each; 40 validation instances checked in order; relaxation MILPs 8 s; budget 22 min).

| run | converged | iterations | instances that ever failed (failure type) | release set: mean size / its relaxation's lower bound | alternatives per cut | val fixed % (OFF / ON) | generators never fixed OFF (lo = 0) | collapsed intervals | CPU h |
|---|---|---|---|---|---|---|---|---|---|
| **kNN, ε = 1 %** | **yes**, all 40 verified (max witness increase 0.995 %) | 24 | 17 (19 infeasible, 3 above cap, 1 time-out) | 42.5 / ≥ 10.1 | 1.0 | **68.5** (55.3 / 13.2) | 14 | 30 | 0.29 |
| **kNN, ε = 5 %** | **yes**, all 40 verified (max 4.90 %) | 24 | 18 (23 infeasible) | 24.3 / ≥ 4.8 | 1.9 | **80.3** (66.3 / 14.0) | 8 | 42 | 0.22 |
| GNN, ε = 5 % | **no** (budget): final thresholds violate 8 of 40 | 40 | 19 (37 infeasible, 2 above cap) | 11.3 / ≥ 4.3 | 2.3 | 87.2 (69.5 / 17.7) | 0 | 33 | 0.37 |
| GNN + guards, ε = 5 % | **no** (budget): violate 11 of 40 | 31 | 18 (29 infeasible, 1 above cap) | 12.8 / ≥ 4.6 | 3.0 | 89.3 before / 86.1 after the guards | 0 | 40 | 0.37 |
| *GNN, ε = 1 %; GNN + guards, ε = 1 %* | *not run (compute: see below)* | | | | | | | | |

* **kNN reproduces the uc12 picture.** ε = 1 % fixes 68.5 % of the validation decisions (uc12: 66.7 %; paper, test:
  78.8 %), ε = 5 % 80.3 % (uc12 75.2 %, paper 82.3 %). The joint check pushes τ̲_g to 0 for 14 (8) generators.
* **Almost every failure is a min up/down conflict** ("infeasible": the reduced problem has no schedule at all),
  not a cost violation: the master's tightest intervals collapse (τ̲_g = τ̄_g, everything of g fixed by rounding at a
  generator-specific point), and a rounded 24-hour row is often not min up/down-feasible. The cuts then open the
  intervals one side at a time (the sliding behaviour of [`ltfx.md`](ltfx.md) §2).
* **Release sets are not minimal.** The 8 s relaxation MILP almost never improves on its warm start (45 of 46
  relaxations at the limit for kNN ε = 1 %); the release sets used are 2.6–5× the relaxation's own lower bound on the
  minimum, i.e. up to that factor larger than minimal. The exact algorithm would cut less and fix more. A better warm start (`StartLtFTuner`: release
  the conflicting generator rows, solve the reduced MILP under the cap, use that solution) cut the GNN release sets to
  11–13 decisions; for kNN it rarely applied (its failures need more than the conflicting rows released).
* **A first kNN ε = 1 % run with the plain start of `otsl/ltfx.py`** (the aligned optimal schedule) was stopped after
  6 min: its release sets had 17–154 decisions. It is what led to `StartLtFTuner`; the reported kNN runs use it.
* **The GNN runs did not converge in 22 min.** Better starts let the relaxation find more alternatives (up to 6 per
  cut), each costing another 8 s MILP, and the GNN's probabilities (never exactly 0 or 1) give the master more room
  to slide; 8 and 11 of the 40 validation instances (the last ones in the check order, never reached) are violated by
  the final thresholds. They are evaluated as they stand, labelled "not converged". Because a 22-min GNN run at
  ε = 5 % did not converge, the two GNN runs at ε = 1 % (more cuts) were dropped to keep the test evaluation inside
  the time budget.
* **Guards inside the check change little here**: the guards remove 3.2 pp of the fixings on validation, and the
  failures that drive the tuning — min up/down conflicts — are exactly what the adequacy and LP-relaxation guards
  do not address (by design: §3).

## 5. Test results (40 instances)

### 5.1 Paper metrics

Gap = (C − DB) / C with DB the dataset full MILP's dual bound; speed-up = per-instance T_MILP / T_method with
T_MILP the dataset full-MILP time; statistics over feasible instances; method time includes inference, the LP
relaxation that feeds the GNN, guards and the reduced MILP. "load-corr." multiplies T_MILP by 0.78, the measured
drift between the dataset run and this study's worker (§5.2). TTQ = the full MILP's time to reach the method's cost
(stored incumbent log) / method time, median. Full tables (all rules, medians, CIs, our earlier convention):
[`uc24ltf_results.md`](../../results/uc24/uc24ltf_results.md).

| rule | feasible % | gap to DB mean / max % | speed-up mean (median) | load-corr. mean | ratio of mean times | fixed % | served % | TTQ speed-up |
|---|---|---|---|---|---|---|---|---|
| full MILP (dataset run) | 100 | 0.16 / 0.77 | 1 | – | 1 | 0 | 97.5 | 1 |
| **LtF kNN, ε = 1 %** | **100** | **0.60** / 2.70 | **7.1** (4.1) | **5.6** | 3.4 | 67.4 | 97.5 | 1.3 |
| **LtF kNN, ε = 5 %** | 92.5 | 2.14 / 13.3 | 25.5 (6.6) | 20.0 | 6.1 | 79.7 | 91.9 | 0.9 |
| LtF GNN, ε = 5 % (not converged) | 72.5 | 8.41 / 81.8 | 20.1 (16.1) | 15.7 | 12.5 | 87.0 | 75.9 | 2.9 |
| LtF + guards GNN, ε = 5 % (not converged) | 62.5 | 1.01 / 4.00 | 19.1 (13.7) | 14.9 | 10.9 | 85.5 | 96.0 | 4.6 |
| LtF GNN, ε = 5 % (n.c.) + our conflict release at test | 100 | 10.70 / 91.7 | 19.3 (14.2) | 15.1 | 12.1 | 86.6 | 70.0 | 2.7 |
| LtF + guards GNN, ε = 5 % (n.c.) + conflict release at test | 100 | 5.09 / 85.5 | 16.8 (11.1) | 13.1 | 9.8 | 85.9 | 87.5 | 3.1 |
| LtF + guards GNN, ε = 5 % (n.c.) + conflict release **first**, then the guards | 100 | 1.08 / 6.89 | 15.1 (9.4) | 11.8 | 7.8 | 84.6 | 97.5 | 3.0 |
| ours, re-run: guarded rule, 95 % target (`95%|best=bce_g`) | 100 | **0.67** / 2.32 | **14.8** (5.2) | **11.6** | 5.9 | 85.4 | 95.0 | 2.6 |
| ours, re-run: REINFORCE ranking 95 % | 100 | 7.58 / 34.3 | 77.4 (35.0) | 60.5 | 34.0 | 94.9 | 92.5 | 6.5 |
| ours, stored: REINFORCE ranking 90 % | 100 | 3.06 / 16.6 | 36.6 (9.3) | (consistent) | 10.2 | 90.0 | 92.5 | 2.8 |
| ours, stored: guarded rule (OFF × 10), 80 % target | 100 | 0.68 / 4.42 | 8.9 (4.3) | (consistent) | 4.7 | 80.0 | 95.0 | 1.4 |
| RACLearn-style confidence, 80 / 90 / 95 %, stored (means only) | 100 | 0.48 / 9.22 / 23.4 | 4.8 / 9.2 / 38.8 | (consistent) | 3.4 / 5.4 / 14.4 | 80 / 90 / 95 | 92.5 / 67.5 / 40 | 1.3 / 1.1 / 2.2 |
| *paper, LtF kNN ε = 1 % / 5 % (Irish system, 72 h, Gurobi)* | *99.8 / 98.9* | *0.48 / 1.05* | *20.8 / 34.3* | | | *78.8 / 82.3* | | |

"stored" rows are the earlier uc24 records, timed together with the dataset full MILPs under the same heavier load,
so their speed-ups need no correction; every rule timed in this study (LtF, re-runs) should be read in the
load-corrected column. Bootstrap 95 % intervals of the mean gap: LtF kNN ε = 1 % [0.42, 0.82] %, guarded rule
[0.50, 0.87] %; of the mean speed-up: [4.5, 10.2]× and [9.7, 20.4]×.

### 5.2 Timing check

| check | result |
|---|---|
| full MILP re-solved back to back in the evaluation worker (instances 0–2) | 210 / 300 / 22 s against 270 / 300 / 28 s in the dataset run (ratios 0.78 / 1.00 at the 300 s limit / 0.78), identical objectives |
| our two rules re-run on all 40 instances, same worker | 0.76× and 0.78× their stored times (median), **identical costs on 40 / 40** |
| 1-minute load average | 3–4 now, 6.7 when the dataset was generated |

So every solve in this study ran ~1.28× faster than the dataset run; speed-ups against the dataset full-MILP times
are inflated by that factor for LtF and the re-runs alike (hence the load-corrected column), while stored records and
dataset times share their conditions. The relative order of the rules is unaffected.

### 5.3 Time to quality

The full MILP's incumbent reaches LtF kNN ε = 1 %'s cost on 90 % of the instances, at a median 1.3× the LtF pipeline's
time; for ε = 5 % the full MILP gets to the same cost **faster than LtF finishes** (median 0.9×). Our guarded rule:
2.6×; REINFORCE 95 %: 6.5× (but at a 7.6 % gap). To a solution within 0.5 % of the full MILP's final cost: LtF kNN
ε = 1 % gets there on 65 % of the instances (our guarded rule 57.5 %, RACLearn 80 % 77.5 %), 2.5× faster than the
full MILP when both get there (guarded rule 2.8×). These TTQ values use the stored (slower) incumbent logs, so they
are also ~1.28× optimistic for every rule timed in this study. As in [`b3.md`](b3.md): measured against time to the
same quality, partial fixing buys about 1–3× on uc24, whoever chooses the fixings.

### 5.4 At equal speed-up (paired, gap to DB, other − LtF, instances feasible for both; 95 % bootstrap CI)

| LtF rule (load-corr. speed-up) | our rule (speed-up, same conditions) | gap difference, pp |
|---|---|---|
| kNN ε = 1 % (5.6×) | guarded 95 % rule, re-run (11.6×) | +0.06 [−0.14, +0.26] — **level in gap, ours 2.1× faster** |
| kNN ε = 1 % (5.6×) | RACLearn-style 80 %, stored (4.8×) | −0.12 [−0.36, +0.13] — level, similar speed |
| kNN ε = 5 % (20.0×, 92.5 % feasible) | REINFORCE ranking 90 %, stored (36.6×, 100 % feasible) | +0.89 [−0.37, +2.09] — level in gap, ours faster |
| kNN ε = 5 % (20.0×) | guarded 95 % rule (11.6×) | −1.46 [−2.32, −0.77] — ours better, LtF 1.7× faster |
| LtF + guards GNN ε = 5 % (14.9×, 62.5 % feasible) | guarded 95 % rule (11.6×) | −0.42 [−0.80, −0.06] on the 25 instances where the LtF rule is feasible |
| LtF + guards GNN ε = 5 %, conflict release first (11.8×, 100 % feasible) | guarded 95 % rule (11.6×) | −0.41 [−0.82, −0.07] — **ours better at the same speed** |
| same hybrid (11.8×) | LtF kNN ε = 5 % (20.0×) | LtF kNN − hybrid: +1.06 [+0.28, +1.94] (hybrid better, kNN faster) |

## 6. Verdict

**Does Learning to Fix reach its paper's regime on 24 hours? In quality yes, in speed no.** Faithful LtF (kNN,
k = 50, Algorithm 1 + 2 converged on all 40 validation instances) at ε = 1 % gives 0.60 % mean gap to the dual bound
(paper 0.48 %), 100 % feasible (paper 99.8 %), but 7.1× mean per-instance speed-up — 5.6× once the lighter machine
load is corrected for, 4.1× median, 3.4× as a ratio of mean times — against the paper's 20.8×, fixing 67 % of the
decisions (paper 79 %). At ε = 5 % it reaches the paper's speed range only through the mean of per-instance ratios
(25.5×, load-corrected 20×; median 6.6×), at twice the paper's gap (2.14 % vs 1.05 %) and with 3 of 40 reduced
problems infeasible (paper 1.1 %). The 24-hour benchmark removes the excuse of uc12 (a 20 s MILP): its MILP takes
154 s, more than the paper's 60 s, and the speed-up is still a quarter of the paper's. What limits it here:
(i) the joint worst-case check leaves 20–33 % of a 24-hour network UC free, and that reduced MILP still costs
25–45 s; (ii) almost all validation failures are min up/down conflicts of rounded 24-hour rows, which the cuts can
only fix by opening intervals (14 generators never fixed OFF at ε = 1 %); (iii) our 8 s relaxation MILPs return
release sets up to 2.6–5× larger than minimal, so the thresholds are more conservative than the exact algorithm's.
**Measured as time to the same quality, LtF gains 1.3× (ε = 1 %) and nothing (0.9×, ε = 5 %)**: HiGHS's own
incumbent reaches LtF's cost about as fast as LtF finishes.

**Do our rules beat it at equal speed-up? Yes, by about 2× in speed at equal quality, and they need no labels.** Our
label-free guarded rule (imitation GNN, 95 % target, adequacy + LP-relaxation guards; no MILP label anywhere) has the
same gap as LtF-kNN at ε = 1 % (0.67 vs 0.60 %, paired +0.06 pp [−0.14, +0.26]) at 2.1× its speed-up (11.6× vs 5.6×
load-corrected; median 5.2× vs 4.1×), and its TTQ speed-up is 2.6× vs 1.3×. At LtF ε = 5 %'s speed, our REINFORCE
90 % ranking is level in gap (+0.89 pp [−0.37, +2.09]) at a higher speed-up (36.6× vs 20×) and always feasible.
LtF needed 2.1 solver core-hours of MILP labels and 0.2–0.3 h of tuning per ε for this; our rules none.

**Do the hybrids help? Not on this budget.** Our GNN is better calibrated to the MILP schedules than the kNN trained
on 120 of them (log-loss 0.086 vs 0.119, 40 vs 87 wrong decisions per instance) and its (non-converged) LtF thresholds fix more
(87 % vs 80 % at ε = 5 %), but the tuning did not converge in 22 min (8 and 11 of 40 validation instances still violated),
and the unverified failure modes appear on test: 27.5 % (plain) and 37.5 % (guard-aware) of the reduced problems are
infeasible, and with our min up/down conflict release added at test time they become feasible but shed load on a few
instances (mean gaps 10.7 % and 5.1 %, maxima 92 % and 86 %; the guards ran before the conflict release and skipped
the conflicting instances). With the conflict release *first* and the guards after it, the guard-aware hybrid is
always feasible and serves 97.5 %: 1.08 % at 11.8× (load-corrected) — but our plain guarded rule is better at the
same speed (0.67 %, paired −0.41 pp [−0.82, −0.07]). Guards inside the tuning check change
little on uc24, because its failures are min up/down conflicts, which the adequacy and LP-relaxation guards do not
address; a guard-aware LtF would need the conflict release inside the check too, and then the master must be kept
from over-fixing (§3: with it, the first, fully collapsed master solution passes every check after the guards have
released 58 % of the fixings).

**Net.** On the 24-hour benchmark, faithful Learning to Fix is safe and accurate at ε = 1 % but slow (5–7×); our
label-free guarded rule matches its quality at twice its speed, and no rule reaches the paper's 0.5 % at 20× here.
Against the time a plain MILP needs to reach the same cost, every partial-fixing pipeline measured on uc24 — LtF
included — gives 1–3×.



## 7. Limitations

* **One core, a shared machine, 7 hours.** Every number comes from one solver process on a 4-core machine shared
  with two other agents (1-minute load average 3–4 during the evaluation, 6.7 when the test set's full MILPs were
  solved). The dataset full-MILP times are therefore ~1.28× slower than a back-to-back solve today (§5.2); speed-ups
  against them are inflated by that factor for every rule timed in this study, which is why load-corrected values
  and the re-run of our rules in the same worker are reported. HiGHS, single-threaded; the paper used Gurobi on 8
  CPUs.
* **Compute-driven deviations from the paper's tuning.** 40 validation instances (paper ≈ 524); relaxation MILPs
  capped at 8 s (release sets 2.6–5× their own lower bound, i.e. more conservative thresholds than the exact
  algorithm); 22 min per tuning run, which the two GNN runs exceeded (their thresholds violate 8 and 11 of the 40
  validation instances); the GNN runs at ε = 1 % were not run. The kNN was trained on 120 labelled schedules at a
  0.5 % gap (paper: ≈ 1,573 optimal ones) with the paper's k = 50, i.e. 42 % of the training set per prediction.
* **C\*** is the 300 s / 0.1 % MILP objective (5 of 40 validation instances stop at the limit, gaps up to 0.34 %);
  the test dual bound is the dataset run's (12 of 40 stop at 300 s, gaps up to 0.77 %), which adds up to 0.77 pp to
  every method's gap on those instances, as in the paper's protocol.
* **Small samples, one pass.** 40 test instances, one seed for every model; bootstrap intervals in the JSON. The
  back-to-back check covers 3 instances (plus the full re-run of two of our rules on all 40).
* **Not done:** self-training labels for the kNN, constant / worst-case thresholds on test, a longer GNN tuning run.

## 8. Reproduce

```bash
python3 scripts/uc_uc24ltf_gen.py --what val,train --n_val 40 --n_train 120     # 2.1 solver core-h, one worker
python3 scripts/uc_uc24ltf_prep.py                                               # seconds
python3 scripts/uc_uc24ltf_tune.py --jobs knn:0.01,gnn:0.05,gnn:0.05:g,knn:0.05 --relax_tl 8 --budget_min 22
python3 scripts/uc_uc24ltf_eval.py --rules knn_1,knn_5,gnn_5,gnn_g_5 --old "95%|best=bce_g,95%|rl" --b2b 3
python3 scripts/uc_uc24ltf_eval.py --rules gnn_5+cr,gnn_g_5+cr --old "" --b2b 0 --tag _cr
python3 scripts/uc_uc24ltf_eval.py --rules gnn_g_5+crf --old "" --b2b 0 --tag _crf
python3 scripts/uc_uc24ltf_report.py --extra uc24ltf_eval_test_cr.jsonl,uc24ltf_eval_test_crf.jsonl
```
(`scripts/uc_uc24ltf_chain.sh`, `uc_uc24ltf_next.sh` and `uc_uc24ltf_next2.sh` are the queues actually run; the
first tuning queue also contained the two GNN ε = 1 % jobs, which were stopped before they started work.)

