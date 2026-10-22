# One evaluation protocol for every model

The studies in `docs/RESEARCH.md` and `docs/recursive/` were scored in several ways that do not line up:

* gap to the MILP's incumbent or to a proven bound;
* median or mean;
* all instances or only those where the reduced problem had a solution;
* speed-ups as a mean of ratios, a median, or a geometric mean, measured on different machines.

Two of those choices changed which method looked better. This page fixes one protocol and shows what follows from it. It works in two steps:

* **Step 1 (done):** every saved per-instance record is re-scored under the protocol. Nothing is re-solved.
* **Step 2 (proposed):** every method is re-timed on PACE.

Files:
* Script: `scripts/unified_rescore.py` (standard library only).
* Output: `results/unified/rescored.{md,json}`.

## The protocol

| | rule |
|---|---|
| instances | Every method on the same sets: 12 h test_fresh (120) and the six shifted sets (30 each), 24 h (40), congested lines (60), PGLib-UC California (30). Tuning on validation only; one test pass. |
| cost | The penalty-inclusive objective of the schedule the method actually delivers. If a reduced MILP has no solution, the method falls back to the full MILP, and that solve's cost and time are charged. **No instance is dropped**, including instances where the method sheds load. |
| gap | (C − DB) / C, where DB is the **tightest proven lower bound** for the instance over every full-MILP solve in any study (or the LP relaxation, if higher). |
| feasibility and cost | Reported separately. Next to the mean gap: share of instances served (no shedding, over-generation or reserve shortfall in any hour), median, 95th percentile and worst gap, share above 1 %, and the gap on the instances that every method in the table serves. |
| time | Everything counts: LP relaxation, inference on the same CPU core, fixing rule, guards, every LP and MILP, and fallbacks. |
| speed-up | Paired: T(full MILP at 0.1 %, same process, same core) / T(method), per instance. The headline is the **geometric mean**, with the median, the arithmetic mean and seconds per instance alongside. A mean of ratios is pulled up by a few very fast instances. |
| machine | All methods of an instance run back to back in one process on one core, with one solver version (HiGHS, one thread). The hardware is recorded. In step 1 this holds only within a study; step 2 makes it hold across studies. |
| uncertainty | 95 % bootstrap intervals over instances; paired differences. At least 3 training seeds for headline models, with the spread across seeds reported next to the instance intervals. |

## Step 1: what the re-scoring checked

* **Instance alignment.** The LP relaxation of an instance is deterministic, so every file that stores it must agree. They do: 0 disagreements over 250 instances in 4 benchmarks. This check caught a real bug in the first version of the script, where a variable was overwritten.
* **Bounds.** No delivered cost lies below the tightest bound, and no bound lies below the LP relaxation (0 violations of either).
* **Reproduction.** Applying each study's own convention to its raw records reproduces 25 of 25 published headline numbers exactly: GPU round, Y6 pipeline, congested lines, the 24 h follow-up and California. So the records are read correctly, and every change below comes from the protocol, not from the data.
* **Repeated measurements.** The same learned rule measured in different studies gives an identical cost on 85–100 % of instances, and is within 0.2 % on all of them. Full MILPs agree less (60–93 % identical, 94–98 % within 0.2 %), because the 60 s limit stops different runs at different solutions.
* **Tightness of the bound.** The reference MILP hit its time limit on 17 of 120 test_fresh instances. On 16 instances the best known cost is still more than 0.1 % above the tightest bound (worst 0.69 %). Gaps there are upper estimates.

## What changes under one protocol

**1. Gap levels fall by about 0.06 pp everywhere; the ranking does not change.**
The tightest bound is higher than each study's own. On 12 h (120 instances):

| method | mean gap |
|---|---|
| recommended pipeline | 0.236 % (was 0.30 %) |
| full MILP at 0.5 % | 0.198 % |
| arm B + safety net | 0.320 % |
| arm B alone | 0.741 % |
| LP rounding | 1.686 % |

**2. Speed-ups shrink under the geometric mean.**

| method | geometric mean | previously reported (mean of ratios) |
|---|---|---|
| pipeline (12 h) | 5.9× | 9.7× |
| full MILP at 0.5 % (12 h) | 2.2× | 3.9× |
| Learning to Fix, 12 h, first 60 | 2.1× | 4.7× |
| learned recipe, 24 h | 7.5× | 10.7× |
| pipeline, congested lines | 6.8× | 11.0× |
| Learning to Fix, California | 9.3× | 25.6× |

The California reference time is the stored label run, which is not paired; step 2 fixes that.

**3. Dropped instances flattered some methods.**

* **Learning to Fix on California:** 0.55 % as published, 3.86 % when its one shedding instance (99.9 % gap) is counted. The median is 0.38 %.
* **RACLearn on 12 h** (90 / 95 % fixed): 3.34 % / 10.33 % mean gap, where `RESEARCH.md` §6 X2 reports 12.4 % / 49.2 %.
  * The per-instance results are the same; only the formula differs.
  * §6 X2 used (C − C_MILP) / C_MILP. That reproduces 12.4 % / 49.2 % exactly, and single instances at +358 % and +925 % dominate those means.
  * (C − DB) / C never exceeds 100 %.

**4. Pipelines with a MILP, 12 h, first 60 instances.**
These are the instances where everything was run.

| method | mean gap | speed-up (geometric mean) |
|---|---|---|
| full MILP at 0.25 % | 0.131 % | 1.4× |
| full MILP at 0.5 % | 0.176 % | 1.8× |
| full MILP at 1 % | 0.186 % | 2.3× |
| Learning to Fix's tuning on the LP relaxation, no learning | 0.205 % | 1.7× |
| hybrid | 0.219 % | 2.8× |
| pipeline | 0.220 % | 6.2× |
| Learning to Fix, kNN | 0.382 % | 2.1× |
| arm B + safety net | 0.277 % | 2.5× |

The learned pipeline is the only method that is both near the MILP's accuracy and clearly faster than loosening the MILP's gap.

**5. Shifted 12-hour sets.**
* With the bound check, the pipeline stays at or below 0.61 % in every set.
* Under line outages, Learning to Fix (26 %) and the hybrid (15.8 %) fail. Adding our LP guard to Learning to Fix brings it to 0.38 %.

**6. Training seeds.**
Only the combined-pipeline study (§6 X2) has three seeds on test. The spread across seeds is:
* 0.02–0.20 pp for the guarded fixing rules;
* 0.22 pp for the end-to-end model;
* 1.31 pp for the REINFORCE ranking.

Every other model has one seed.

## What step 1 cannot fix

* **Timing across studies.** Speed-ups are paired within each study, but studies ran on different machines and HiGHS versions. Some records cover the MILP solve only, without inference or LP-relaxation time; they are marked and their speed-ups are not shown:
  * RACLearn, asymmetric and error-cost rules of `ltf_eval_fresh`;
  * the combined fixing configurations of §6 X2.
* **24-hour Learning to Fix** has no paired reference: the full MILP was re-run on only a few instances.
* **Early one-shot models were never run on test_fresh.** These are the framework GNN, LP features, kNN-20 and the He et al.-style repair (`RESEARCH.md` §4 V2, §5 W1). Their numbers use the original test set and the median gap to the MILP's incumbent, so they cannot enter these tables without a re-run.
* **Loose bounds** on the time-limited instances (above).

## Step 2: proposed re-timing on PACE (not started; needs approval)

1. **One harness.** For each instance, one process pinned to one core runs every headline method back to back, in a random order. All runs use the same HiGHS build with one thread, and inference runs on the CPU. The full MILP at 0.1 % is the reference. CPU model, HiGHS version and load average are recorded per instance.
2. **Methods.** The headline rows of `results/unified/rescored.md`, plus the early one-shot models re-run on test_fresh.
3. **Instances.** All five sets above (12 h with its shifted sets, 24 h, congested, California).
4. **Bounds.** One long full-MILP solve (about one hour) on every instance whose best known cost is still more than 0.1 % above its bound.
5. **Seeds.** Three seeds for the headline learned models. This needs retraining and can be a separate step.

The rough cost is 40–50 CPU core-hours and no GPU, inside the project's PACE share.
