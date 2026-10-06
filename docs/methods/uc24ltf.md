# Learning to Fix on the 24-hour benchmark (uc24), with our probabilities and guards

*Code: [`otsl/uc24ltf.py`](../../otsl/uc24ltf.py) (guard-aware tuner, soft LP guard),
`scripts/uc_uc24ltf_{gen,prep,tune,eval,report}.py`, `scripts/uc_uc24ltf_chain.sh`. Results:
[`results/uc24/uc24ltf_results.md`](../../results/uc24/uc24ltf_results.md) (+ `.json`), raw test solves
`results/uc24/uc24ltf_eval_test.jsonl`, tuning logs / thresholds `results/uc24/uc24ltf_tune_<rule>.{log,json}`,
probabilities `results/uc24/uc24ltf_probs.{npz,json}`, run logs `results/uc24/uc24ltf_*.log`. New data:
`data/generated/uc24/uc24ltf_val.npz` (validation with full MILPs), `uc24ltf_train_lab.npz` (labelled training
subset).*

RESULTS_SUMMARY

## 1. Why

On the 12-hour benchmark the faithful Learning to Fix (LtF; Fritz et al. 2026, [`ltfx.md`](ltfx.md)) reproduced the
paper's solution quality but not its speed-up, and the obvious explanation was the benchmark: a 20 s MILP leaves
little to gain. uc24 ([`b3.md`](b3.md)) is the benchmark where the full MILP takes minutes (154 s mean, 30 % at the
300 s limit), i.e. the paper's regime (Gurobi, 60 s mean). uc24 had been built label-free — no full MILP for
training or validation — so LtF could not run on it: its tuning needs C\* on validation instances and its kNN needs
labelled training schedules. This study adds both, tunes LtF, and evaluates it against our label-free rules.

## 2. What was added to uc24 (and what it cost)

COST_TABLE

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
| LtF+guards GNN ε | same GNN | Algorithm 1 + 2 with the guards **inside** every check, cut and verification (`GuardedLtFTuner`) | adequacy guard + LP-relaxation guard, also at test |
| old: 95 % target, guarded (`95%|best=bce_g`) | imitation GNN | ranking, top 95 % | adequacy + LP guard + conflict release ([`b3.md`](b3.md)) |
| old: REINFORCE ranking 95 % (`95%|rl`) | REINFORCE GNN | ranking, top 95 % | conflict release |
| stored: RACLearn-style 80 / 90 / 95 %, other b3 rules | | | earlier records (`b3_fix_test*.jsonl`) |

**Faithful LtF** is `otsl/ltfx.py` unchanged (see [`ltfx.md`](ltfx.md) §1–2 for the method and every choice where the
paper is silent). Settings on uc24: Q = 20, K_max = 10, check = reduced MILP with the (1 + ε)·C\* cut-off, stopped at
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
dataset run of the test set; the first 4 instances are re-solved back to back in the evaluation worker to measure the
machine-load drift, and the two most relevant earlier rules are re-run in the same worker so that all compared rules
share conditions.

RESULTS_SECTIONS
