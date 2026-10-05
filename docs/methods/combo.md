# Combined pipelines and 3-seed confirmation on fresh test instances (B2)

*Code: [`otsl/combo.py`](../../otsl/combo.py) (composition only; every component is imported),
[`scripts/uc_combo_train.py`](../../scripts/uc_combo_train.py) (seed replicates),
[`scripts/uc_combo_eval.py`](../../scripts/uc_combo_eval.py) (validation decisions, end-to-end and fixing evaluation),
[`scripts/uc_combo_report.py`](../../scripts/uc_combo_report.py), [`scripts/uc_combo_queue.sh`](../../scripts/uc_combo_queue.sh)
(the exact run sequence is in `results/uc12/combo_chain.log`). Raw numbers:
[`results/uc12/combo_results.md`](../../results/uc12/combo_results.md) / `.json`, per-instance records
`results/uc12/combo_fix_test_fresh.jsonl`, `combo_e2e_*_arrays.npz`.*

## Summary

Every trainable part of the §5 frameworks was retrained with seeds 0, 1, 2 and every claim re-tested on 120 fresh
B2 instances (fixing: the first 60, full MILP back to back), with all decision rules fixed on validation first.

* **Confirmed**: the block adequacy repair (label-free + REINFORCE serves 95.0 % [92.5, 96.7] of fresh instances with
  one LP; it still over-commits, 6.2 % above the MILP on served instances); lag_D + block repair (92.5 % / 2.16 % /
  3.25 %, identical coverage for all seeds); the learned error-cost ranking at 90 % (0.28 % [0.27, 0.29] at 3.2× vs
  12.4 % for RACLearn; original test 0.66 % [0.47, 0.89], i.e. the earlier 0.47 % was the best seed); the post-hoc
  LP-relaxation guard (error-cost ranking at a 95 % target: 0.46 % at 4.6× fresh, 0.39 % at 6.3× original, seed-stable);
  REINFORCE-ranked fixing at 95 % (2.9–3.6 % mean gap at 10–30×). On the original test, seed 0 reproduces every earlier
  number exactly.
* **Weaker than claimed / not confirmed**: self-trained ranking at 95 % (2.28 % at 8.1× fresh, 1.93 % at 9.6× on all 60
  original instances; the claimed 1.52 % at 11.9× came from a favourable 40-instance subset);
  REINFORCE with the val threshold (median 1.94 %, claimed 1.18 %; the threshold changes with the seed);
  RACLearn + LP guard (0.46 %, claimed 0.19 %); "serves more instances than the MILP" (true on the original set only:
  the MILP serves 98.3 % of the fresh instances).
* **Combined end-to-end pipeline** (val-selected lag_D + val threshold + block repair; screening over 7 thresholds,
  ~6 LPs): 93.3 % served, 1.04 % median, 1.64 % served-instance gap, 3.31 % mean gap on the fresh set (original set:
  95.8 % / 0.86 % / 1.73 % / 3.57 %) — the best end-to-end result measured. With one LP: 92.2 % / 1.69 % / 2.68 %.
* **Combined acceleration pipeline** (self-trained probabilities + error-cost ranking + adequacy + LP guard):
  fresh set 0.59 % at 5.3× (95 % target) and 0.72 % at 7.0× (98 %), the best seeded rule at ≥ 5× there; original set
  0.73 % at 7.3× and 0.86 % at 10.4× (the only sub-1 % rule at ≥ 10× there). It does not beat its parts at lower speed
  (error-cost ranking on BCE probabilities with both guards: 0.46 % / 0.39 %), and the val preference for self-trained
  over BCE probabilities did not carry over to test.

## 1. Question

Section 5 of [`RESEARCH.md`](../RESEARCH.md) (W1–W4) reported, from single seeds and partly post hoc on the original
120 test instances, (i) end-to-end pipelines that serve more instances than the MILP with one LP thanks to the block
adequacy repair, and (ii) fixing rules (learned error-cost ranking, self-trained probabilities, LP-relaxation guard)
that beat RACLearn at 90–95 % fixed. This study (a) re-trains every trainable part with seeds 0, 1, 2 and re-tests the
claims on **120 fresh test instances** (`data/generated/uc12/test_fresh.npz`, test days, generation seed 23, never
used for any decision), and (b) assembles the best parts into one end-to-end and one solver-acceleration pipeline.

## 2. Pipelines

**End-to-end (one LP, or k LPs).** Model 1 probabilities p → decision threshold θ (calibrated on val) → block
adequacy repair (`otsl.constrained.adequacy_repair_blocks`: cheapest min up/down-feasible blocks until committed
capacity covers net load + reserve in every hour) → min up/down repair → dispatch LP. *Screening*: the same for every
θ in {0.3, …, 0.9}, the cheapest schedule by the exact LP (≤ 7 LPs; identical schedules are solved once).

**Solver acceleration.** Probabilities p (MILP-label BCE, self-trained BCE or label-free + REINFORCE, chosen on val)
→ features of `otsl.fixpolicy.FixFeaturizer` computed from p → learned expected error cost (harm ensemble) →
fix the target share of decisions with the lowest expected cost to round(p) → adequacy guard (release OFF fixes in
merit order until the units not fixed off cover net load + reserve + 5 %) → min up/down row check → LP-relaxation
guard (release OFF / ON fixes around hours in which the relaxed reduced problem pays penalties; its LP time is
counted) → reduced MILP (60 s, 0.1 %).

## 3. Protocol

**Seed replicates** (`scripts/uc_combo_train.py`; each recipe identical to the original, only the seed changes):

| component | recipe | seed 0 | seeds 1, 2 |
|---|---|---|---|
| label-free + REINFORCE (`rl`) | `train_uc_reinforce` from `constrained_lf_bce.pt`, 120 steps × 16 instances × 6 samples | existing `constrained_rl_lf.pt` | retrained |
| label-free + Lagrangian + KL anchor (`lag`, = `lag_D`) | `train_constrained`, ε = 0.15, KL 0.05, λ ≤ 5, MuProp, window 1, 120 steps | existing `constrained_lag_D.pt` | retrained |
| self-trained BCE (`st`) | `train_bce_fixed` on the existing round-3 label pool, 80 epochs | retrained; **identical** to `selftrain_m3.pt` (max abs. difference of val probabilities 0.0) | retrained |
| error-cost (harm) model | `uc_fixpolicy_train.py --labels comp`: 3-member ensemble; replicate s = member seeds 3s, 3s+1, 3s+2 | retrained; **identical** to `fixpolicy_policy_comp.pt` | retrained |

Not retrained (single models, reported as such): the label-free BCE start `constrained_lf_bce.pt`, the MILP-label BCE
model `uc_model1_4.pt` under the harm ranking and RACLearn, and the MILP-label REINFORCE model `uc_model1_rl.pt` (the
model behind the original "REINFORCE ranking" claim). Seed s of a pipeline uses seed s of each of its parts.

**Decision rules, fixed on validation before any test set was read** (`scripts/uc_combo_eval.py`):

1. *Threshold* of every end-to-end model: argmin over θ ∈ {0.3, …, 0.9} of the mean val cost of the block-repaired
   schedule divided by the LP-relaxation cost (label-free expected cost including shedding / reserve penalties; the
   rule of `uc_constrained.py --stage calib`, grid extended to 0.3 / 0.4).
2. *End-to-end predictor*: the family (REINFORCE, lag_D, self-trained BCE, label-free BCE) with the lowest
   seed-averaged criterion of rule 1 at the calibrated thresholds.
3. *Fixing probability source*: on the first 15 val instances (full MILP and reduced MILPs back to back), harm seed 0,
   95 % target, both guards; lowest mean gap, within 0.1 pp the faster.
4. Fixing targets of the combined pipeline: 95 % and 98 % (fixed in advance).

**Test.** End-to-end: all 120 test_fresh instances, every model and seed, one LP per decoder (the original test set
too, for comparability). Fixing: first 60 test_fresh instances; every worker solves the full MILP (60 s, 0.1 %) and then
every reduced MILP of the instance back to back (2 workers, spawn pool). Rules: the confirmations (harm + adequacy
guard at 90 %, harm + both guards at 95 %, RACLearn + LP guard at 95 %, self-trained asymmetric + guard at 95 %,
REINFORCE probabilities at 95 %; 3 seeds each), the combined pipeline (3 seeds, 95 / 98 %, with and without the LP
guard), RACLearn at 90 / 95 %, RACLearn + LP guard at 95 / 98 % and the original MILP-label REINFORCE model at 95 %
(deterministic, one run each); 29 reduced MILPs per instance. The no-LP-guard variant and RACLearn + LP guard at 98 %
were added to the plan after the validation decisions but before the test pass started. The same fixing pass was run
on the original test set afterwards, within the remaining time budget (§4.5).

## 4. Results

All numbers: mean [min, max] over seeds 0, 1, 2; gaps to the MILP reference objective of each instance; *served* =
no shedding / spill and no reserve shortfall in the exact dispatch LP. Full tables, instance-bootstrap intervals and
paired comparisons: [`results/uc12/combo_results.md`](../../results/uc12/combo_results.md).

**The fresh test set differs from the original in one important way**: the MILP's own schedules serve **98.3 %** of
the fresh instances (original test: 92.5 %; 2 vs 9 instances with priced reserve shortfall) and the reference is
slightly weaker (70 % proven optimal vs 82.5 %, mean MIP gap 0.33 % vs 0.23 %; generated under heavier CPU load).
Served shares are therefore compared with the MILP of the same set, and "serves more than the MILP" is a property of
the original set, not of the method.

### 4.1 Validation decisions

| decision | outcome (val) |
|---|---|
| thresholds (rule 1) | REINFORCE 0.9 / 0.6 / 0.8 (seeds 0 / 1 / 2), lag_D 0.6 / 0.6 / 0.6, self-trained 0.3 / 0.4 / 0.3, label-free BCE 0.3 |
| end-to-end family (rule 2) | val cost / relaxation cost: lag_D 1.0544, REINFORCE 1.0598, self-trained 1.0724, label-free BCE 1.1100 → **lag_D** |
| fixing probability source (rule 3) | 15 val instances, 95 % target: self-trained 0.47 % mean gap at 5.8×, MILP-label BCE 0.85 % at 5.9×, REINFORCE 3.29 % at 13.3× → **self-trained** |

The threshold rule is noisy with 60 validation instances: for REINFORCE seed 1 the criterion is 1.0671 at θ = 0.6
and 1.0677 at θ = 0.9, and one shedding instance lifts θ = 0.7 / 0.8 to 1.21; the REINFORCE thresholds therefore
differ by seed (0.6–0.9), the lag_D threshold does not.

### 4.2 End-to-end (test_fresh, 120 instances, one LP unless noted)

| configuration | LPs | served | median gap | mean gap, served | mean gap | units on |
|---|---|---|---|---|---|---|
| MILP commitment (reference) | MILP | 98.3 % | 0 | 0 | 0 | 14.73 |
| label-free + REINFORCE, top-1 | 1 | 83.1 % [75.8, 88.3] | 3.89 % [3.62, 4.17] | 6.05 % [4.36, 8.14] | 10.6 % [7.4, 15.9] | 15.96 |
| label-free + REINFORCE + block repair | 1 | **95.0 % [92.5, 96.7]** | 3.45 % [3.33, 3.66] | 6.15 % [4.72, 7.92] | 7.44 % [6.82, 8.38] | 16.00 |
| … val-calibrated threshold (0.9 / 0.6 / 0.8) + block repair | 1 | 91.7 % [90.0, 93.3] | 1.94 % [1.54, 2.56] | 3.03 % [2.36, 3.69] | 5.95 % [5.20, 7.04] | 15.25 |
| lag_D, top-1 | 1 | 79.2 % [78.3, 80.0] | 2.23 % [2.05, 2.42] | 2.99 % [2.82, 3.12] | 9.35 % [7.76, 11.75] | 15.25 |
| lag_D + block repair | 1 | 92.5 % [92.5, 92.5] | 2.16 % [1.93, 2.32] | 3.25 % [3.03, 3.43] | 7.45 % [6.47, 9.01] | 15.31 |
| **combined, one LP**: lag_D, val threshold 0.6 + block repair | 1 | 92.2 % [91.7, 92.5] | 1.69 % [1.49, 1.87] | 2.68 % [2.55, 2.87] | 8.31 % [6.44, 12.04] | 15.14 |
| **combined, screening**: lag_D, 7 thresholds + block repair, best by LP | 5.8 | **93.3 % [91.7, 95.0]** | **1.04 % [1.03, 1.06]** | **1.64 % [1.51, 1.84]** | **3.31 % [3.12, 3.47]** | 14.78 |
| self-trained BCE, val threshold (0.3 / 0.4 / 0.3) + block repair | 1 | 85.3 % [83.3, 86.7] | 1.51 % [1.36, 1.67] | 2.25 % [2.00, 2.39] | 14.2 % [10.5, 19.4] | 15.09 |
| label-free BCE (no fine-tuning) + block repair | 1 | 85.0 % | 1.18 % | 1.74 % | 21.0 % | 14.69 |

Instance-bootstrap 95 % intervals of the seed-averaged served share are ± 3–5 pp (e.g. REINFORCE + block repair
91.7–98.1 %, lag_D + block repair 87.5–96.7 %), so coverage differences of 1–3 pp between rows are not resolved.

Seed-matched paired comparisons (instances both configurations serve; seeds 0 / 1 / 2):

| A | B | served by both | mean gap A | mean gap B | A cheaper |
|---|---|---|---|---|---|
| lag_D + block repair | REINFORCE + block repair | 111 / 107 / 111 | 3.03 / 3.41 / 3.28 % | 7.58 / 4.74 / 5.76 % | 102 / 89 / 97 |
| lag_D, val threshold + block | REINFORCE, val threshold + block | 105 / 107 / 106 | 2.47 / 2.85 / 2.62 % | 2.29 / 3.68 / 2.79 % | 28 / 85 / 49 |
| lag_D screening (5.8 LPs) | REINFORCE, val threshold + block | 104 / 111 / 107 | 1.46 / 1.62 / 1.56 % | 2.26 / 3.70 / 2.79 % | 75 / 101 / 97 |
| lag_D, val threshold + block | self-trained, val threshold + block | 102 / 96 / 102 | 2.40 / 2.50 / 2.45 % | 2.37 / 1.97 / 2.40 % | 33 / 20 / 36 |

* **The block repair result replicates**: with plain REINFORCE it serves 95.0 % [92.5, 96.7] of fresh instances with
  one LP (+11.9 pp over top-1), every seed above 92 %. It does **not** exceed the MILP on this set (98.3 %).
* **REINFORCE still over-commits**: +1.3 units per hour over the MILP, 6.2 % [4.7, 7.9] above it on served instances
  (original seed 0: 8.9 %); the seed spread of that cost is large (4.7–7.9 %).
* **lag_D + block repair replicates** (92.5 % / 2.16 % / 3.25 % vs 93.3 % / 1.73 % / 3.09 % on the original set,
  identical coverage for all three seeds) and is cheaper than REINFORCE + block repair on 89–102 of the ~110
  instances both serve (3.0–3.4 % vs 4.7–7.6 %). The paired advantage is smaller than the original seed-0 one
  (3.10 % vs 8.79 %) because seeds 1 and 2 of REINFORCE over-commit less than seed 0.
* **The calibrated threshold for REINFORCE replicates only partly**: coverage 91.7 % (claimed 90.8 %), but the
  median gap is 1.94 % [1.54, 2.56] instead of 1.18 %; the threshold itself is not stable across seeds. Once both
  are threshold-calibrated, neither is consistently cheaper on the instances both serve (paired row 2: lag_D cheaper
  on 28 / 85 / 49 of ~106, mean 2.47 / 2.85 / 2.62 % vs 2.29 / 3.68 / 2.79 %).
* **The combined end-to-end pipeline** (rule-selected lag_D, val threshold, block repair) with one LP gives
  92.2 % / 1.69 % / 2.68 %; its mean gap (8.3 %) is set by the few unserved instances. With screening over the
  threshold grid (5.8 distinct LPs on average) it is the best end-to-end configuration tested: **93.3 % served,
  1.04 % median, 1.64 % on served instances, 3.31 % mean gap**, with the smallest seed spread of any learned row,
  and cheaper than calibrated REINFORCE on 75–101 of ~107 common instances.
* Self-trained and label-free BCE with the block repair are cheaper on what they serve (1.7–2.3 %) but leave 15 %
  of instances with shedding or shortfall (mean gap 14–21 %).

### 4.3 Fixing + reduced MILP (test_fresh, first 60 instances, full MILP back to back)

Full MILP (60 s, 0.1 %) in the same worker: 29.8 s mean, serves all 60 instances. *Target* is the share requested
from the ranking; the LP guard releases fixings, so the *fixed* column is the share actually fixed.

| rule | target | fixed | mean gap | instance bootstrap 95 % | median gap | > 1 % | served | speed-up |
|---|---|---|---|---|---|---|---|---|
| RACLearn (MILP-label BCE confidence) | 90 % | 90.0 % | 12.4 % | 0.42–30.3 | 0.014 % | 9 | 91.7 % | 2.8× |
| RACLearn | 95 % | 95.0 % | 49.2 % | 12.3–98.1 | 0.079 % | 21 | 68.3 % | 8.1× |
| RACLearn + LP guard | 95 % | 87.0 % | 0.46 % | 0.17–0.78 | 0.015 % | 7 | 96.7 % | 3.0× |
| RACLearn + LP guard | 98 % | 84.0 % | 0.50 % | 0.26–0.82 | 0.088 % | 6 | 98.3 % | 3.1× |
| error-cost ranking + adequacy guard | 90 % | 89.9 % | **0.28 % [0.27, 0.29]** | 0.13–0.45 | 0.005 % | 5.7 | 98.9 % | 3.2× [3.1, 3.2] |
| error-cost ranking + adequacy + LP guard | 95 % | 92.8 % | 0.46 % [0.43, 0.51] | 0.26–0.70 | 0.073 % | 7.7 | 98.3 % | 4.6× [4.0, 5.1] |
| self-trained, asymmetric + adequacy guard | 95 % | 94.9 % | 2.28 % [1.87, 2.55] | 0.98–4.19 | 0.41 % | 16.0 | 89.4 % | 8.1× [6.4, 9.3] |
| label-free + REINFORCE probabilities | 95 % | 95.0 % | 2.93 % [1.94, 3.55] | 1.72–4.48 | 0.97 % | 28.7 | 88.9 % | 10.6× [4.9, 14.0] |
| MILP-label + REINFORCE probabilities (original model) | 95 % | 95.0 % | 3.61 % | 2.37–4.98 | 1.40 % | 38 | 93.3 % | **30.5×** |
| **combined**: self-trained p + error cost + adequacy + LP guard | 95 % | 93.9 % | 0.59 % [0.56, 0.64] | 0.37–0.83 | 0.10 % | 13.0 | 96.7 % | 5.3× [5.0, 5.7] |
| **combined** | 98 % | 92.4 % | 0.72 % [0.63, 0.84] | 0.48–0.99 | 0.26 % | 14.3 | 97.8 % | **7.0× [5.9, 7.6]** |
| combined without the LP guard | 95 % | 94.9 % | 2.17 % [0.66, 4.90] | 0.50–5.02 | 0.10 % | 13.3 | 92.8 % | 5.2× [4.4, 5.8] |
| combined without the LP guard | 98 % | 97.8 % | 11.6 % [7.2, 17.4] | 3.28–22.2 | 0.37 % | 18.7 | 82.2 % | 8.4× [6.3, 10.6] |

(RACLearn, RACLearn + LP guard and the MILP-label REINFORCE model are deterministic: one run each. "> 1 %": number of
instances with a gap above 1 %, mean over seeds.)

**Pareto view** — best seed-averaged mean gap at a seed-averaged speed-up of at least x:

| | ≥ 2× | ≥ 3× | ≥ 5× | ≥ 10× | ≥ 20× |
|---|---|---|---|---|---|
| RACLearn (confidence) | 12.4 % (90 %) | 49.2 % (95 %) | 49.2 % (95 %) | – | – |
| RACLearn + LP guard | 0.46 % (95 %, 3.0×) | 0.46 % | – | – | – |
| **combined pipeline** (self-trained + error cost + both guards) | 0.59 % | 0.59 % | **0.59 % (95 %, 5.3×)** | – | – |
| error-cost + adequacy guard (BCE p) | **0.28 % (90 %, 3.2×)** | **0.28 %** | – | – | – |
| error-cost + both guards (BCE p) | 0.46 % | 0.46 % | – | – | – |
| self-trained, asym + guard | 2.28 % | 2.28 % | 2.28 % (8.1×) | – | – |
| label-free + REINFORCE p | 2.93 % | 2.93 % | 2.93 % | **2.93 % (10.6×)** | – |
| MILP-label + REINFORCE p (original model, one seed) | 3.61 % | 3.61 % | 3.61 % | 3.61 % | **3.61 % (30.5×)** |

* **Learned error-cost ranking at 90 %: confirmed.** 0.28 % [0.27, 0.29] mean, 0.005 % median, 3.2× (claim 0.47 %,
  3.7×), against 12.4 % for RACLearn at the same share (9 instances above 1 %, median 0.014 %: a heavy tail).
  The three harm seeds agree closely.
* **LP-relaxation guard (post hoc): confirmed.** It removes the catastrophic tail for every ranking it is applied to:
  error cost 0.46 % [0.43, 0.51] at 4.6× (claim 0.40 % at 5.4×), RACLearn 0.46 % at 3.0× (claim 0.19 % at 3.2×). Without
  it, the same rankings at 95–98 % give 2–50 %. It still releases many fixings (92.8 % / 87.0 % actually fixed at a
  95 % target), which caps the speed-up at ~3–7×.
* **Self-trained ranking at 95 %: weaker than claimed.** 2.28 % [1.87, 2.55] at 8.1× (claim 1.52 % at 11.9× on 40
  instances); still 20× better in mean gap than RACLearn at 95 % (49 %), but it is no longer a sub-2 % rule.
* **REINFORCE ranking at 95 %: confirmed on the mean gap, not on speed.** Label-free REINFORCE seeds: 2.93 %
  [1.94, 3.55], but 10.6× with a seed range of 4.9–14.0×: how fast the reduced MILP is depends on how much a seed
  over-commits. The original MILP-label REINFORCE model: 3.61 % at 30.5× (claim 5.5 % at 17.7×). This family remains
  the only one beyond ~10×, at a ~1 % median gap.
* **The combined pipeline** is the best rule at ≥ 5× (0.59 % at 5.3×; 0.72 % at 7.0× with a 98 % target), with a
  small seed spread, 96.7–97.8 % served and a 0.10–0.26 % median. It does **not** improve on its parts at lower speed:
  the error-cost ranking on BCE probabilities with the same guards is 0.46 % at 4.6× and with the adequacy guard alone
  at 90 % 0.28 % at 3.2×. The validation choice of self-trained over BCE probabilities (0.47 % vs 0.85 % on 15 val
  instances) did not carry over to test (0.59 % vs 0.46 %); the bootstrap intervals of the guarded rules overlap
  (0.17–0.83 %), so 60 instances cannot rank them. What combining buys is speed at equal robustness: the 98 % target
  reaches 7.0× at 0.72 %, where the error-cost ranking with BCE probabilities (95 %) stops at 4.6×.
* **The LP guard is what makes ≥ 95 % targets safe.** The same combined ranking without it: 2.2 % (95 %) and 11.6 %
  (98 %) mean gap, 82–93 % served.

### 4.4 Original test set, for comparability (120 instances, end-to-end)

Seed 0 reproduces every earlier single-seed number exactly (REINFORCE + block repair 94.2 % / 3.92 % / 8.91 %;
threshold 0.9 + block repair 90.8 % / 1.18 % / 2.55 %; lag_D + block repair 93.3 % / 1.73 % / 3.09 % / 5.96 %).
Across seeds:

| configuration (original test; MILP serves 92.5 %) | LPs | served | median gap | mean gap, served | mean gap |
|---|---|---|---|---|---|
| REINFORCE + block repair | 1 | 93.6 % [93.3, 94.2] | 3.30 % [2.99, 3.92] | 6.80 % [4.78, 8.91] | 9.82 % [9.09, 10.98] |
| REINFORCE, val threshold + block repair | 1 | 90.3 % [89.2, 90.8] | 1.63 % [1.18, 2.19] | 3.01 % [2.55, 3.88] | 14.0 % [9.1, 19.0] |
| lag_D + block repair | 1 | 95.0 % [93.3, 95.8] | 1.80 % [1.72, 1.95] | 3.35 % [3.09, 3.76] | 5.86 % [5.28, 6.35] |
| combined, one LP (lag_D, θ = 0.6, block repair) | 1 | 94.4 % [94.2, 95.0] | 1.52 % [1.49, 1.55] | 2.83 % [2.57, 3.17] | 5.55 % [4.79, 6.30] |
| combined, screening (6.1 LPs) | 6.1 | **95.8 % [95.8, 95.8]** | **0.86 % [0.78, 0.92]** | **1.73 % [1.50, 1.96]** | **3.57 % [3.33, 4.03]** |
| self-trained BCE, val threshold + block repair | 1 | 85.8 % [85.0, 86.7] | 1.57 % | 2.68 % | 32.2 % |

On the original set the seed-0 numbers sit at the favourable end of the seed range for REINFORCE's threshold rule
(median 1.18 % vs 1.63 % mean over seeds) and at the unfavourable end for its served-instance cost without the
threshold (8.91 % vs 6.80 %). lag_D + block repair is stable across seeds on both sets.

### 4.5 Fixing on the original test set (first 60 instances, for comparability)

Same 29 reduced MILPs per instance, full MILP back to back (23.8 s mean, serves 88.3 %). **Seed 0 reproduces every
earlier per-instance result exactly** (mean gaps 0.47 / 0.40 / 0.19 / 1.38 / 38.5 / 5.50 %, and 1.52 % for the
self-trained rule on its 40 instances); only the speed-ups move with the machine load.

| rule | target | earlier single seed | seeds 0–2: mean gap | median | served | speed-up |
|---|---|---|---|---|---|---|
| error-cost ranking + adequacy guard | 90 % | 0.47 % at 3.7× | 0.66 % [0.47, 0.89] | 0.006 % | 88.3 % | 4.4× |
| error-cost ranking + adequacy + LP guard | 95 % | 0.40 % at 5.4× | **0.39 % [0.34, 0.42]** | 0.062 % | 96.7 % | 6.3× [5.7, 7.6] |
| RACLearn + LP guard | 95 % | 0.19 % at 3.2× | 0.19 % (deterministic) | 0.014 % | 91.7 % | 3.5× |
| self-trained, asym + guard | 95 % | 1.52 % at 11.9× (40 inst.) | 1.38 % [1.19, 1.52] on those 40; **1.93 % [1.81, 2.00] on all 60** | 0.27 % | 87.8 % | 9.6× [8.5, 10.7] |
| label-free + REINFORCE probabilities | 95 % | – | 2.99 % [2.23, 3.90] | 0.89 % | 87.8 % | 14.8× [5.0, 24.9] |
| MILP-label + REINFORCE probabilities | 95 % | 5.50 % at 17.7× | 5.50 % (deterministic) | 1.48 % | 91.7 % | 29.0× |
| RACLearn | 90 / 95 % | 1.38 % / 38.5 % | same (deterministic) | | 86.7 / 80.0 % | 2.9× / 6.9× |
| **combined** (self-trained + error cost + both guards) | 95 % | – | 0.73 % [0.51, 1.15] | 0.084 % | 93.3 % | 7.3× [6.3, 9.2] |
| **combined** | 98 % | – | **0.86 % [0.73, 1.11]** | 0.19 % | 93.9 % | **10.4× [8.0, 11.8]** |
| combined without LP guard | 95 / 98 % | – | 4.4 % / 16.0 % | | 87.2 / 76.1 % | 9.2× / 10.8× |

* The error-cost ranking at 90 %: the earlier 0.47 % was the best of three seeds (0.47 / 0.63 / 0.89 %); with the LP guard
  at 95 % the result is seed-stable (0.34–0.42 %) on both sets.
* The self-trained rule's 1.52 % at 11.9× was measured on 40 instances; on all 60 it is 1.93 % at 9.6×, in line with
  the fresh set (2.28 % at 8.1×): the earlier figure was an optimistic instance subset, not a seed effect.
* On this set the combined pipeline at a 98 % target is the only rule with a sub-1 % mean gap at ≥ 10× (0.86 % at 10.4×;
  REINFORCE rankings 3.0–5.5 %); on the fresh set the same rule reaches only 7.0× (0.72 %). Whether ≥ 10× at < 1 % is
  attainable therefore depends on the instance set.


## 5. Which earlier claims held up

| claim (single seed, original test) | original test, 3 seeds | **fresh test, 3 seeds** | verdict |
|---|---|---|---|
| LF + REINFORCE + block repair: 94.2 % served, 3.92 % median, 8.91 % served-instance gap (one LP) | 93.6 % / 3.30 % / 6.80 % | 95.0 % [92.5, 96.7] / 3.45 % / 6.15 % [4.7, 7.9] | **confirmed** (coverage and over-commitment cost); "more than the MILP" holds only on the original set (MILP 98.3 % on the fresh set) |
| … + val threshold 0.9: 90.8 % / 1.18 % / 2.55 % | 90.3 % / 1.63 % [1.18, 2.19] / 3.01 % | 91.7 % / 1.94 % [1.54, 2.56] / 3.03 % | **partly**: coverage and ~3 % served gap hold; the threshold is seed-dependent (0.9 / 0.6 / 0.8) and 1.18 % was the best seed |
| lag_D + block repair (post hoc): 93.3 % / 1.73 % / 3.09 %, 5.96 % mean | 95.0 % / 1.80 % / 3.35 %, 5.86 % | 92.5 % / 2.16 % / 3.25 %, 7.45 % | **confirmed** and seed-stable; "lowest mean gap of any one-LP method" holds on the original set only (fresh: ties REINFORCE + block repair, 7.44 %) |
| lag_D cheaper than REINFORCE on commonly served instances (3.10 % vs 8.79 %) | 3.1–3.5 % vs 4.8–8.8 % | 3.0–3.4 % vs 4.7–7.6 % | **confirmed** at threshold 0.5; with both thresholds calibrated the advantage disappears |
| error-cost ranking + adequacy guard, 90 %: 0.47 % at 3.7× | 0.66 % [0.47, 0.89] at 4.4× | 0.28 % [0.27, 0.29] at 3.2× | **confirmed** (< 1 % on both sets; 0.47 % was the best seed; RACLearn on the same fresh instances: 12.4 %) |
| + LP-relaxation guard (post hoc), 95 % target: 0.40 % at 5.4× | 0.39 % [0.34, 0.42] at 6.3× | 0.46 % [0.43, 0.51] at 4.6× [4.0, 5.1] | **confirmed**, seed-stable |
| RACLearn + LP guard, 95 %: 0.19 % at 3.2× | 0.19 % at 3.5× (deterministic) | 0.46 % at 3.0× | **partly**: < 0.5 % on both sets, but on the fresh set no more accurate than the error-cost ranking with the same guards, and always ≤ 3.5× |
| self-trained ranking, 95 % (40 instances): 1.52 % at 11.9× | 1.38 % [1.19, 1.52] on the 40; 1.93 % [1.81, 2.00] at 9.6× on 60 | 2.28 % [1.87, 2.55] at 8.1× [6.4, 9.3] | **not confirmed** in size: seed-stable, but the 40-instance figure was optimistic (~2 % at 8–10× on 60 instances of either set); still far better than RACLearn at 95 % (38–49 %) |
| REINFORCE ranking, 95 %: 5.5 % at 17.7× | MILP-label model 5.50 % at 29×; LF + REINFORCE 2.99 % [2.23, 3.90] at 14.8× [5.0, 24.9] | LF + REINFORCE 2.93 % [1.94, 3.55] at 10.6× [4.9, 14.0]; MILP-label model 3.61 % at 30.5× | **confirmed** (mean gap ≤ claim); the speed-up varies 3–5× across seeds |

## 6. Verdict

* **End-to-end pipeline** (pre-declared rules: lag_D, val threshold, block repair; optional screening over the
  threshold grid). With one LP: 92.2 % served, 1.69 % median, 2.68 % on served instances on the fresh set (original:
  94.4 % / 1.52 % / 2.83 %). With screening (~6 LPs) it is the best end-to-end configuration measured: **93.3 % / 1.04 % /
  1.64 %, 3.31 % mean gap on the fresh set; 95.8 % / 0.86 % / 1.73 %, 3.57 % on the original set** (above the MILP's
  92.5 % there, below its 98.3 % on the fresh set), with seed ranges of ≤ 3.3 pp in coverage and ≤ 0.5 pp in
  served-instance gap. Most of the gain over the parts comes from the screening; with one LP the calibrated threshold
  makes lag_D + block repair slightly cheaper on served instances (2.7 vs 3.3 % fresh, 2.8 vs 3.4 % original) at
  equal coverage, but not on the mean gap.
* **Solver-acceleration pipeline** (self-trained probabilities + error-cost ranking + adequacy guard + LP-relaxation
  guard): fresh set 0.59 % mean gap at 5.3× (95 % target) and 0.72 % at 7.0× (98 % target), 97–98 % served, small
  seed spread; original set 0.73 % at 7.3× and 0.86 % at 10.4×. It is the best seeded rule at ≥ 5× on the fresh set and
  the only sub-1 % rule at ≥ 10× on the original set, and the LP guard is indispensable (without it 2–16 %). It is not
  better than its parts at lower speed (error-cost ranking on the BCE model with both guards: 0.46 % at 4.6× fresh,
  0.39 % at 6.3× original), and the val preference for self-trained over BCE probabilities did not carry over to test.
  Otherwise, beyond ~8× only REINFORCE-ranked fixing remains (2.9–5.5 % mean gap at 10–30×, ~1–1.5 % median).
* **Against RACLearn-style ranking on the same instances**: at ≥ 3× the best new rule has 0.28 % mean gap against
  49 % for RACLearn (its 90 % point, 12.4 %, reaches only 2.8×). Given the same LP-relaxation guard, RACLearn is as
  accurate (0.46 %) but tops out at 3.0–3.1×, because after the guard only 84–87 % of its decisions stay fixed (target
  95–98 %); the learned ranking keeps 92–94 % fixed and is the only guarded rule at ≥ 5×.
* **What did not hold**: the self-trained 95 % ranking's 1.5 % at 12× (a favourable 40-instance subset; ~2 % at 8–10×
  on 60 instances of either set); the REINFORCE threshold 0.9's 1.18 % median and the error-cost rule's 0.47 % at 90 %
  (both the best of three seeds); "serves more than the MILP" (set-dependent); lag_D's "lowest one-LP mean gap"
  (set-dependent); RACLearn + LP guard's 0.19 % (0.46 % on the fresh set).



## 7. Limitations

* **What the seeds cover.** Seeds vary the training of the fine-tuned predictors (REINFORCE, lag_D), the self-trained
  BCE network and the harm ensemble. They do not vary the inputs those were trained from: the label-free BCE start,
  the self-training label pool (one run of reduced-MILP relabelling), the MILP-label BCE model and the harm labels
  (one cross-fit). The seed spread therefore understates the variance of the whole pipelines. Seed 0 of REINFORCE and
  lag_D is the existing checkpoint (same recipe), not a re-run; seed 0 of the self-trained and harm models was re-run
  and reproduced the saved models exactly.
* **Test sets are small.** 120 instances for end-to-end decoders and 60 for fixing; instance-bootstrap intervals
  (`combo_results.md`) are ± 3–5 pp on served shares, and fixing mean gaps are set by a few instances with shedding
  or reserve shortfall (their bootstrap intervals span a factor of 2–10). Validation decisions used 60 instances
  (thresholds) and 15 instances with one harm seed (fixing source); both are noisy (§4.1).
* **The fresh set is not an independent distribution.** It uses the same calendar test days as the original test set
  (13 of 120 instances share a day and window start, with independent load / renewable noise). It was generated
  under heavier CPU load: the MILP reference proves optimality on 70 % of instances (original 82.5 %), so negative
  gaps occur, and the reference serves 98.3 % of instances (original 92.5 %).
* **Timing.** Speed-ups are ratios of back-to-back solves in the same worker on a 4-core machine shared with two
  other agents (load average 6–9). The back-to-back full MILP itself hits the 60 s limit on some instances, which
  caps the attainable speed-up and makes it a slightly weaker reference than the stored objective. Compare ratios
  within a table only.
* **Transfer of the error-cost model.** The harm model was trained on the errors of the cross-fitted MILP-label BCE
  model; in the combined pipeline it ranks the decisions of the self-trained model (chosen on val, not retrained).
  A harm model trained on self-trained-model errors was not tried (it needs a new cross-fit and ~27k LPs).
* **Not retrained / not tested:** self-trained + REINFORCE and the SIL variant (single seed, earlier study), He et
  al.-style repair, Learning to Fix; screening for the end-to-end pipeline uses thresholds only (no samples).


