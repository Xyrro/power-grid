# Better Model 1 probabilities: data scaling, deep ensembles, temporal mixing (B2, 12-hour UC)

Code: [`otsl/m1x.py`](../../otsl/m1x.py), `scripts/uc_m1x_{data,train,valfix,tune,eval,report}.py`, run sequence
`scripts/uc_m1x_queue.sh`. Results: [`results/uc12/m1x_results.md`](../../results/uc12/m1x_results.md) (+ `.json`).
Data (git-ignored): `data/generated/uc12_m1x/`.

## Status

*Kept current for resume after a container restart. Every step is idempotent: rerun the queue phase.*

- 14:40 phase 1 started (`scripts/uc_m1x_queue.sh phase1`, log `data/generated/uc12_m1x/queue_phase1.log`):
  3,500 extra scenarios + label-free labels (14 chunks of 250) → reference metrics → label-free learning curve
  (500/1000/2000/4000) → teacher-polished labels for 1,500 extra instances → polished curve (1000/2000).
- Pilot for the polishing setting done (`data/generated/uc12_m1x/pilot.json`): confidence ranking, 90 %, LP guard,
  5 s chosen (median label gap 0.10 %, all served, 4.9 s per label).
- 15:14 3,500 extra scenarios generated (`extra_0..13.npz`, 34 min); reference metrics written (`results/uc12/m1x_train.json`).
- 15:34 label-free curve done (500–4000); 17:33 teacher polishing of 1,500 extra instances done (4.7 s per label,
  2.0 core-h); 17:40 polished curve done (1000, 2000). The polished curve stops at 2,000 instances: 2,000 more labels
  would cost another ~2.6 core-h (scaled back for the 8–10 h budget); the label-free curve goes to 4,000.
- 17:40–18:52 phase 2 done (ensemble, temporal GNN, MLP, guarded-rule validation check of 11 sources).
- 17:55 phase 2's bash was stopped and phase 3 rewritten to add the temporal GNN on 2,000 polished instances as a
  candidate (`scripts/uc_m1x_select.py --extended`); phases 3–4 run as `scripts/uc_m1x_queue2.sh phase34`
  (log `data/generated/uc12_m1x/queue_phase34.log`).
- 19:40 run A (guard-aware LtF on error-cost scores of the 5-member GNN ensemble, 500 MILP labels) converged:
  81.4 % fixed on validation after guards (reference hybrid 83.1 %).
- 19:51 selection: winner pol_n2000_gnnt (temporal GNN, 500 MILP + 1,500 polished). 19:55 phase 3 replaced by
  `scripts/uc_m1x_queue2.sh phase3b4` (log `data/generated/uc12_m1x/queue_phase3b4.log`): seeds 1–4, ensemble, run B on
  error-cost scores ("he") and on the probabilities themselves ("hg", added because the error-cost model was trained on
  the plain GNN's errors), then the test.
- 21:26 B-he converged: 84.6 % fixed on validation; 22:03 B-hg converged: 88.5 % (selected by the paper's criterion).
- 22:03–23:27 test, first 60 instances, every rule. 23:28 instances 60–119 with the reference and the selected rule only
  (budget). 00:18 a container restart killed the run after instance 103; resumed 00:20 (finished instances skipped,
  `data/generated/uc12_m1x/queue_resume.log`).
- 00:35 main study complete (test 0–119 done, report regenerated).
- 00:39–02:15 follow-up: robustness of B-hg on the ood study's shifted sets (`scripts/uc_m1x_ood.py`, records
  `results/uc12/m1x_ood_eval.jsonl`, logs `data/generated/uc12_m1x/ood_*.log`): timing check on 14 instances (agrees
  within 10 %), then all 220 instances; report `results/uc12/m1x_ood.{md,json}` (`python3 scripts/uc_m1x_ood.py
  --report`). Done; resume = rerun the same command (finished instances skipped).
- 02:42 end-to-end test of the recommended pipeline (`scripts/uc_m1x_pipeline.py`): test_fresh 0–119 + the six
  shifted sets (300 instances), three shards on cores 0, 1, 3 (one single-threaded process each), records
  `results/uc12/m1x_pipeline_eval_s{0,1,2}.jsonl`, logs `data/generated/uc12_m1x/pipe_s*.log`. Resume: rerun
  `taskset -c N python3 scripts/uc_m1x_pipeline.py --shard i --nshards 3` (finished instances skipped), then `--report`.
- `scripts/uc_m1x_queue2.sh` was rewritten after the runs to the sequence actually executed (phase2, phase3, phase4).
- To resume after a restart: `taskset -c 3 python3 scripts/uc_m1x_eval.py --runs hg_pol_n2000_gnnt_ens5 --skip_faithful
  --start 60 --n 60` then `python3 scripts/uc_m1x_report.py` (the evaluation skips finished instances; every tuning
  run is done). Earlier phases: `scripts/uc_m1x_queue.sh phase1`, `scripts/uc_m1x_queue2.sh phase2|phase3|phase4`
  (finished steps are skipped or cheap).

## Summary

**Better probabilities do translate into more fixing, but only through a better model, not through more cheap labels
or ensembling alone.** Two changes improve Model 1's probabilities substantially: a temporal head on the GNN (dilated
1-D convolutions across the 12 hours on per-hour inputs) and more training instances labelled by one reduced MILP
around a teacher's confident decisions (4.7 s per label, no full MILP). Together, as a 5-member ensemble on 500 MILP +
1,500 polished instances, they cut the validation log-loss from 0.057 to 0.046 and raise the share of unit-hours that
can be fixed at 99.9 % precision from 0.49 (0.49–0.59 over 5 seeds) to 0.79. Learning to Fix's joint ε = 1 % tuning
on these probabilities fixes 88.5 % of the validation decisions instead of 83.1 %, and on all 120 fresh test
instances it gives **0.26 % mean gap at 6.1× against the reference hybrid's 0.28 % at 4.8×** (re-run back to back on the
same core): Δ gap −0.02 pp [−0.07, +0.03], Δ log speed-up +0.18 [+0.01, +0.36] (1.20× less time per instance,
geometric mean), +3.9 pp fixed [+3.0, +4.9] (first 60: 0.23 % at 7.0× vs 0.25 % at 5.0×). The same probabilities through the error-cost transform give the lowest gap instead (0.19 %,
−0.06 pp [−0.12, −0.01] on the first 60) but at 3.2×. Negative results: label-free labels do not scale (4,000 instances: 9 % fixable
at 99.9 % precision); a 5-member ensemble alone improves the pooled metrics but not the tuned fixed share (81.4 %) or
the test result; ensemble disagreement adds nothing as a filter; a flat MLP is worse. Single tuning run per rule,
60 test instances with all rules, 120 for the reference and the selected rule; about 10 core-hours in total. **Follow-up under distribution shift** (the robustness study's six shifted sets): B-hg keeps its advantage on
five shifts but collapses under line outages like every LtF-calibrated rule (17.7 % mean gap); adding the
LP-relaxation guard at test time repairs it (0.23 % at 6.9×) at a 5–15 % speed cost elsewhere.

## Method

**Question.** The hybrid rule (Learning to Fix's joint ε-tuning on our error-cost scores, [`hybrid.md`](hybrid.md))
gains over Learning to Fix through better probabilities, not through the tuning. Do *better* Model 1 probabilities —
from more training data, deep ensembles or temporal mixing across hours — let the tuning fix more decisions at the
same validated ε, i.e. reach the same gap at a higher speed-up?

**1. More data without full MILPs.** Extra 12-hour scenarios are drawn with the sampler of `otsl.ucdata.generate`
(training days, random 12-hour window, the same load, renewable and initial-status noise; seed 111 instead of 11), so
the distribution is that of `train.npz`. Per scenario only the LP relaxation is solved (it is the GNN's input anyway).
Two label types:

* *label-free* (V4): rounded LP relaxation → adequacy repair → min up/down repair, one dispatch LP to price it;
* *teacher-polished* (one round of self-training, W2, with a better teacher): the teacher is the mean of the three
  MILP-label BCE GNNs; its 90 % most confident decisions are fixed to its rounded prediction, the adequacy guard, the
  min up/down row release and the LP-relaxation guard release what they would release at test time, and a reduced MILP
  (5 s, 0.1 %) fills in the rest. The schedule replaces the label-free label if the dispatch LP prices it cheaper. The
  setting was chosen on 24 training instances with out-of-fold teacher probabilities (pilot table in the results).
  The 500 original instances keep their MILP labels in this regime ("pol"), so the curve is 500 MILP labels + k polished.

All labels are canonicalised inside groups of identical units (as for the existing BCE GNN). Training recipe of
`uc_model1_4.pt` (GNN + symmetry rank + LP-relaxation features, BCE, AdamW 1e-3, batch 64, cosine schedule, early
stopping on the log-loss of `val.npz`), features normalised with the statistics of the 500 original instances; epochs
80 / 60 / 40 / 30 for 500 / 1000 / 2000 / 4000 instances (more gradient steps for more data, bounded compute).

**2. Deep ensembles.** Mean probability of independently seeded models (5 members; for the 500-MILP-label GNN the
three existing seeds `uc_model1_4.pt`, `hybrid_bce_s1.pt`, `hybrid_bce_s2.pt` plus seeds 3 and 4; seed 0 retrained
here reproduces `uc_model1_4.pt` exactly). Disagreement (std over members)
was tested as an extra filter by ranking on |p̄ − 0.5| − k·std (k = 1, 2).

**3. Temporal mixing** (`otsl.m1x.CommitGNNT`). The current GNN encodes the 12 hours of a unit in one 64-d state and
decodes 12 logits with a linear layer. The new head takes, per (unit, hour), the unit's GNN embedding and the hour's
own inputs (relaxed commitment, net load, bus load, bus renewables, relaxed price at the bus), adds a learned hour
embedding and mixes across hours with three residual dilated 1-D convolutions (kernel 3, dilations 1 / 2 / 4, so every
hour sees the whole window); its output is added to the GNN's logit and starts at zero (training starts from the plain
GNN). Compared with the plain GNN and with the flat MLP of `otsl.ucml` (`CommitMLP`, 3 layers of 256) at equal data.

**4. Metrics on validation** (360 instances = val 60 + val_extra 120 + val_extra2 180, MILP labels aligned to the
prediction inside identical-unit groups so that a swapped copy is not an error): log-loss, Brier score, ROC AUC,
top-label calibration error (15 bins), wrong decisions per instance at 0.5, and the share of unit-hours that can be fixed
at 99 % / 99.9 % precision (pooled ranking by confidence). Downstream on validation: a guarded rule with reduced MILPs on
the first 60 instances of val_extra2 (error-cost ranking of the harm ensemble at 95 % + adequacy + min up/down rows +
LP-relaxation guard; gap to the stored validation MILP).

**5. Downstream.** The chosen probabilities feed the reference hybrid's pipeline unchanged: error-cost scores of the
harm ensemble `combo_harm_s0.pt` on features of the new probabilities (no retraining of the harm model), the score
transform of `otsl.hybrid.harm_to_score` (μ, σ of log h over the first 180 validation instances), guard-aware tuning
(`otsl.hybrid.HybridTuner`, adequacy + min up/down rows in the check, ε = 1 %, Q = 20, K_max = 10, relaxation MILPs 6 s)
on the same 360 validation instances ("he"). The same tuning on the probabilities themselves ("hg", the other family of
the hybrid study) was added for the selected model, because the error-cost model was trained on the plain GNN's errors.

**6. Selection (validation only).** `scripts/uc_m1x_select.py`, written before the polished-label, temporal-GNN and MLP
results: a candidate (seed 0) replaces the default family (GNN on 500 MILP labels) only if its share fixable at 99.9 %
precision exceeds the default's 5-seed mean by more than the default's seed range and its log-loss is not above the
default's mean; the best such candidate gets 5 seeds and run B. Run A (the 5-seed default ensemble) isolates the
ensemble. After phase 2 showed that both more polished data and the temporal head pass the bar, their combination
(temporal GNN on 2,000 instances) was added as a candidate before any downstream run (`--extended`); it won. Between
B-he and B-hg the paper's criterion decides (larger guarded validation fixed share). The test set was read only by the
final evaluation.

## Setup

* **Compute.** One core (core 3 of a 4-core machine shared with three other agents' jobs), HiGHS and PyTorch
  single-threaded, every job alone on the core, in the order of `scripts/uc_m1x_queue.sh` (phase 1) and
  `scripts/uc_m1x_queue2.sh` (phases 2–4).
* **Selection.** Everything is chosen on validation (probability metrics, the guarded validation rule, the tuned fixed
  share). The test set (`test_fresh`) is read once, by `scripts/uc_m1x_eval.py`.
* **Test protocol.** For each test instance one worker solves the full MILP (60 s, 0.1 %), then the LP relaxation that
  feeds the GNN, then every rule's guards and reduced MILP (60 s, 0.1 %), back to back on the same core
  (`uc_hybrid_eval._job`, unchanged). The reference hybrid (`he_bce_s0_e1_n360`, 0.24 % at 5.2× in `hybrid.md`) and
  faithful LtF on the BCE GNN (ε = 1 %) are re-run in the same worker, so every speed-up is paired. Method time =
  reduced MILP + LP relaxation + forward passes of all ensemble members + error-cost features and harm ensemble + guards.
* **Metrics.** Paper metrics (gap to the back-to-back full MILP's dual bound, mean per-instance speed-up, feasible
  instances); paired differences against the reference hybrid with instance-bootstrap 95 % CIs.

## Results

Full tables: [`results/uc12/m1x_results.md`](../../results/uc12/m1x_results.md). Single seed (0) per curve point
unless noted; the seed spread of the reference recipe (5 seeds at 500 MILP labels) is given for scale.

### Labels

| label source | cost per label | quality |
|---|---|---|
| full MILP (existing 500) | 34 s | reference |
| label-free (rounded LP relaxation + repairs) | 0.59 s (LP relaxation 0.41 s of it) | median 15.8 % above the MILP on the 24-instance pilot |
| teacher-polished (conf. 90 %, LP guard, 5 s) | 4.7 s + the 0.59 s above | pilot: median 0.10 %, mean 0.81 %, all served; 1,500 new labels: 72 % hit the 5 s limit, 89 % taken from the reduced MILP, median 9.6 % cheaper than the label-free label; they differ from the teacher's rounded prediction on 1.4 % of unit-hours |

Label cost of the extra data: 3,500 scenarios with label-free labels 0.57 core-h; 1,500 polished labels 2.0 core-h
(for comparison, 1,500 full-MILP labels ≈ 14 core-h).

### Learning curve (validation, 360 instances)

| training data | n | log-loss | Brier | AUC | wrong / inst. | fixable @99 % | fixable @99.9 % | guarded 95 % rule: Δ gap vs ref, pp [CI] / Δ fixed, pp |
|---|---|---|---|---|---|---|---|---|
| MILP labels (reference recipe), 5 seeds | 500 | 0.0563–0.0590 | 0.0139–0.0144 | 0.9945–0.9949 | 14.8–15.5 | 0.975–0.977 | 0.490–0.585 (mean 0.540) | 0 (seed 0 = ref) |
| label-free | 500 | 0.0655 | 0.0139 | 0.9877 | 13.4 | 0.972 | 0.058 | +0.27 [+0.02, +0.58] / −0.4 |
| label-free | 1000 | 0.0656 | 0.0137 | 0.9888 | 13.2 | 0.973 | 0.068 | +0.25 [+0.03, +0.53] / +0.3 |
| label-free | 2000 | 0.0652 | 0.0136 | 0.9894 | 13.2 | 0.973 | 0.072 | +0.27 [+0.12, +0.44] / +0.5 |
| label-free | 4000 | 0.0653 | 0.0134 | 0.9892 | 12.9 | 0.976 | 0.089 | +0.14 [−0.07, +0.32] / −0.3 |
| 500 MILP + polished | 1000 | 0.0532 | 0.0133 | 0.9957 | 14.2 | 0.980 | 0.651 | −0.00 [−0.20, +0.23] / +1.5 |
| 500 MILP + polished | 2000 | **0.0502** | **0.0123** | 0.9961 | 13.1 | **0.985** | 0.687 | −0.03 [−0.28, +0.22] / +1.7 |

(guarded 95 % rule: reference model 0.649 % mean / 0.044 % median gap, 92.5 % fixed, on the first 60 instances of
val_extra2.)

* **Label-free labels do not scale into better fixing probabilities.** Eight times more data lowers the wrong
  decisions per instance (13.4 → 12.9, *fewer* than the MILP-label model's 14.8) but the high-confidence region stays
  polluted: only 6–9 % of the unit-hours can be fixed at 99.9 % precision, against 49–59 % for the MILP-label model.
  The label-free labels are systematically wrong in the same places (the rounded relaxation), and the model learns
  those errors confidently. Downstream the guarded rule is 0.14–0.27 pp worse at every size.
* **Polished labels scale.** 500 MILP + 500 / 1,500 polished labels improve every metric monotonically: log-loss
  0.0568 → 0.0532 → 0.0502, fixable at 99 % 0.977 → 0.980 → 0.985, at 99.9 % 0.49 → 0.65 → 0.69 (above the 5-seed
  range of the 500-label model). A single model on 2,000 instances beats the 5-member ensemble on 500 on every
  probability metric. Downstream, at the fixed 95 % target, the gap is level (−0.03 pp, CI ±0.25) and the LP guard
  releases fewer fixings (6 vs 21 per instance), so 1.7 pp more is fixed — the direction asked for, but small and not
  significant on 60 instances.

### Deep ensembles and the disagreement filter

| source | log-loss | Brier | fixable @99 % | fixable @99.9 % | guarded 95 %: Δ gap / Δ fixed vs single seed 0 |
|---|---|---|---|---|---|
| single GNN, 5 seeds (500 MILP labels) | 0.0563–0.0590 | 0.0139–0.0144 | 0.975–0.977 | 0.490–0.585 | – |
| ensemble of 3 | 0.0553 | 0.0134 | 0.980 | 0.642 | −0.02 [−0.13, +0.08] / +0.9 |
| ensemble of 5 | 0.0543 | 0.0132 | 0.981 | 0.658 | −0.01 [−0.14, +0.11] / +1.2 |

Disagreement (std over the 5 members) as an extra filter, ranking by |p̄ − 0.5| − k·std: fixable at 99.9 % 0.658 /
0.660 / 0.658 for k = 0 / 1 / 2, at 99 % 0.981 / 0.980 / 0.980. **The filter adds nothing**: where the members
disagree the mean probability is already near 0.5, and the confident errors that limit fixing are errors all members
make together (shared label noise and shared inputs), so their disagreement is low there.

### Architecture at equal data (500 MILP labels, seed 0)

| model | params | train min | log-loss | Brier | AUC | wrong / inst. | fixable @99 % | fixable @99.9 % | guarded 95 %: Δ gap / Δ fixed |
|---|---|---|---|---|---|---|---|---|---|
| GNN (current; 5 seeds) | 165 k | 2.3 | 0.0563–0.0590 | 0.0139–0.0144 | 0.9945–0.9949 | 14.8–15.5 | 0.975–0.977 | 0.490–0.585 | 0 |
| **GNN + temporal head** | 180 k | 5.8 | **0.0503** | **0.0128** | **0.9964** | 13.8 | **0.983** | **0.749** | +0.06 [−0.11, +0.21] / +1.2 |
| MLP (flat) | 2.2 M | 0.5 | 0.0641 | 0.0177 | 0.9947 | 20.2 | 0.957 | 0.748 | +0.24 [−0.13, +0.70] / −2.0 |

* **Temporal mixing is the largest single gain at equal data**: with the same 500 labels it lowers the log-loss by 12 %
  (0.0573 seed mean → 0.0503, as much as 2,000 polished instances) and raises the share fixable at 99.9 % precision
  from 0.49–0.59 (mean 0.54) to 0.75 (+0.21, against +0.15 for 2,000 polished instances and +0.12 for a 5-member
  ensemble).
* The MLP has as clean a top 75 % as the temporal GNN but many more errors below it (20 wrong decisions per instance,
  fixable at 99 % only 0.957): its downstream guarded rule fixes less and is worse. The GNN family is the better base.
* On the guarded 95 % rule (fixed target, harm model trained on the BCE GNN's errors) the temporal head's better
  probabilities do not show up as a lower gap (median 0.215 % vs 0.044 %; mean level): the error-cost model was
  trained on the plain GNN's out-of-fold errors and is applied to a different model's probabilities without retraining.

### Downstream on validation: Learning to Fix tuning (guard-aware, ε = 1 %, 360 instances)

| run | probabilities | score | val fixed after guards (before) | OFF / ON | iterations | core-h |
|---|---|---|---|---|---|---|
| reference hybrid (`he_bce_s0_e1_n360`) | single GNN, 500 MILP labels | error cost | 83.07 % (83.77 %) | 68.8 / 14.2 | 57 + 6 (180 then 360 warm) | 0.88 |
| A `he_milp500_ens5` | 5 × GNN, 500 MILP labels | error cost | 81.43 % (82.12 %) | 67.2 / 14.2 | 70 | 0.76 |
| B-he `he_pol_n2000_gnnt_ens5` | 5 × temporal GNN, 500 MILP + 1,500 polished | error cost | 84.57 % (85.00 %) | 70.1 / 14.5 | 65 | 0.82 |
| **B-hg `hg_pol_n2000_gnnt_ens5`** | same | probability | **88.48 %** (88.61 %) | – | 66 | 0.61 |

* **The ensemble alone does not fix more** under the joint ε-tuning (81.4 % vs 83.1 %), although it is better on every
  pooled probability metric. The tuned share is set by the worst validation instances per generator (one generator-wide
  threshold must keep every one of the 360 instances within ε), and the ensemble does not remove those few confident
  errors; the 1.6 pp difference is within the seed-to-seed spread of the tuning itself (hybrid study: 83.7 % vs 81.1 %
  for two BCE seeds on 180 instances).
* **The better model does**: the temporal-GNN ensemble on 2,000 instances fixes 88.5 % with the same validated
  guarantee when the thresholds are tuned on its probabilities (+5.4 pp over the reference, i.e. a third fewer free
  decisions: 11.5 % instead of 16.9 %), and 84.6 % through the error-cost transform. The error-cost model
  (`combo_harm_s0.pt`) was trained on the plain GNN's out-of-fold errors; on a different, better-calibrated model it
  costs fixings instead of adding them. The paper's criterion (larger validation fixed share) selects **B-hg**.

### Test (test_fresh, paper metrics, full MILP re-solved back to back on the same core)

The full MILP took 25.5 s here on the first 60 instances against 20.3 s in the hybrid study (per-instance ratio median
1.37: the machine was more loaded), so stored times were not reused; every speed-up is against the full MILP solved in
the same worker, and the reference hybrid and faithful LtF were re-run there. The re-run reference reaches the same
objective as in the hybrid study on every instance (0.25 % at 5.0× here vs 0.24 % at 5.2× there).

**First 60 instances, every rule:**

| rule | feasible | gap to DB, mean [95 % CI] | gap max | speed-up mean [95 % CI] | median | ratio of means | fixed |
|---|---|---|---|---|---|---|---|
| full MILP | 100 % | 0.16 % | 2.33 % | 1.0× | 1.0× | 1.00 | 0 % |
| faithful LtF, BCE GNN, ε = 1 % | 98.3 % | 0.29 % [0.18, 0.41] | 2.37 % | 5.2× [3.8, 6.8] | 2.5× | 2.48 | 83.6 % |
| reference hybrid (he, BCE GNN) | 100 % | 0.25 % [0.18, 0.34] | 1.83 % | 5.0× [3.4, 7.0] | 2.1× | 1.93 | 85.6 % |
| A: he, 5 × GNN, 500 MILP labels | 100 % | 0.25 % [0.17, 0.34] | 1.98 % | 4.6× [3.3, 6.3] | 2.0× | 1.93 | 83.8 % |
| B-he: he, 5 × temporal GNN, 2,000 inst. | 100 % | **0.19 %** [0.14, 0.26] | **1.35 %** | 3.2× [2.5, 4.0] | 2.1× | 1.80 | 86.4 % |
| **B-hg: hg, 5 × temporal GNN, 2,000 inst. (selected)** | 100 % | 0.23 % [0.16, 0.31] | 1.45 % | **7.0×** [4.7, 9.8] | **3.0×** | **2.76** | **88.9 %** |

Paired against the reference hybrid (same 60 instances, instance bootstrap 95 % CI):

| rule | Δ gap to DB, pp | Δ mean speed-up | Δ log speed-up | time ratio ref / rule (geo. mean) | Δ fixed, pp |
|---|---|---|---|---|---|
| A | −0.004 [−0.068, +0.066] | −0.37 [−2.36, +1.61] | −0.01 [−0.25, +0.23] | 0.99× [0.78, 1.26] | −1.8 [−2.4, −1.2] |
| B-he | **−0.064 [−0.124, −0.011]** | −1.82 [−3.28, −0.64] | −0.24 [−0.43, −0.04] | 0.79× [0.65, 0.96] | +0.8 [−0.0, +1.6] |
| **B-hg** | −0.024 [−0.097, +0.045] | +1.97 [−0.56, +4.65] | **+0.27 [+0.01, +0.55]** | **1.31× [1.01, 1.73]** | **+3.3 [+2.1, +4.6]** |
| faithful LtF, BCE | +0.034 [−0.059, +0.147] | +0.09 [−1.91, +1.75] | +0.18 [−0.03, +0.39] | 1.20× [0.97, 1.47] | −2.0 [−3.5, −0.5] |

**All 120 instances, reference and selected rule** (instances 60–119 were run with these two rules only, for time):

| rule | feasible | gap to DB, mean [95 % CI] | gap max | speed-up mean [95 % CI] | median | ratio of means | fixed | served |
|---|---|---|---|---|---|---|---|---|
| full MILP (29.5 s mean) | 100 % | 0.24 % | 2.47 % | 1.0× | 1.0× | 1.00 | 0 % | 98.3 % |
| reference hybrid | 100 % | 0.28 % [0.22, 0.35] | 1.83 % | 4.8× [3.8, 6.0] | 2.3× | 1.91 | 84.8 % | 96.7 % |
| **B-hg (selected)** | 100 % | 0.26 % [0.20, 0.33] | 1.54 % | **6.1×** [4.6, 7.7] | **2.8×** | **2.52** | **88.7 %** | 95.0 % |

Paired (120 instances): Δ gap −0.018 pp [−0.067, +0.031]; Δ mean speed-up +1.23 [−0.23, +2.72]; **Δ log speed-up
+0.18 [+0.01, +0.36]** (1.20× less time per instance, geometric mean, [1.01, 1.43]); **Δ fixed +3.9 pp [+3.0, +4.9]**.
Instances 60–119 alone favour the selected rule less than the first 60 (the speed advantage shrinks from 1.31× to
1.20× over all 120). B-hg leaves a priced reserve shortfall on 2 more instances than the reference (95.0 % vs 96.7 %
served; the full MILP itself 98.3 %); the gap includes those penalties.

## Verdict

* **Against the hybrid (0.24 % at 5.2× in its own run, 0.28 % at 4.8× re-run here on 120 instances): the goal is met,
  modestly — the same gap at a higher speed-up.** The selected rule (Learning to Fix's tuning on the probabilities of a
  5-member temporal-GNN ensemble trained on 500 MILP + 1,500 polished labels) fixes 88.7 % instead of 84.8 % of the test
  decisions (+3.9 pp [+3.0, +4.9]) at an unchanged gap (0.26 % vs 0.28 %, Δ −0.02 pp [−0.07, +0.03]) and takes 1.20× less
  time per instance (geometric mean, [1.01, 1.43]; Δ log speed-up +0.18 [+0.01, +0.36]); mean per-instance speed-up
  6.1× vs 4.8× (Δ +1.2×, CI [−0.2, +2.7], not significant on its own: it is dominated by a few very fast instances).
  On the first 60 instances alone the advantage was larger (7.0× vs 5.0×, 1.31× less time). Worst instance 1.54 % vs
  1.83 %; two more instances than the reference end with a priced reserve shortfall (95.0 % vs 96.7 % served).
* **Where the gain comes from**: the temporal head (the largest single improvement at equal data) and the polished data
  (log-loss and the 99.9 % fixable share improve monotonically from 500 to 2,000 instances). Not from ensembling alone
  (run A is level with the reference on test and fixes less on validation), not from the disagreement filter, not from
  label-free data (systematically wrong labels make confident errors that more data does not remove), not from a flat MLP.
* **Error-cost scores do not transfer to a new probability model without retraining.** On the better ensemble the
  error-cost transform (B-he) gives the lowest test gap of all rules (0.19 %, −0.06 pp [−0.12, −0.01] against the
  reference) but at 3.2× (slower than the reference, Δ log speed-up −0.24 [−0.43, −0.04]); tuning on the probabilities
  themselves (B-hg) gives the speed. If a lower gap matters more than speed, B-he is the better rule; the validation
  criterion fixed in advance (larger fixed share) picked B-hg. Retraining the error-cost model on the new model's
  out-of-fold errors (≈ 1.6 core-h of dispatch LPs plus 4-fold cross-fitting) is the obvious next step.
* **The tuned fixed share responds to probability quality only weakly**: the 99.9 %-fixable share 0.54 (5-seed mean)
  → 0.79 and the log-loss −19 % buy +5.4 pp of validation fixing and +3.9 pp on test. The joint ε-check is a worst case
  over 360 instances with one threshold pair per generator, so a few confident errors per generator set the thresholds.

## Robustness under distribution shift (follow-up)

Code: `scripts/uc_m1x_ood.py`; records `results/uc12/m1x_ood_eval.jsonl`; tables
[`results/uc12/m1x_ood.md`](../../results/uc12/m1x_ood.md) (+ `.json`). Shifted sets, outage handling and the other
rules come from the robustness study ([`ood.md`](ood.md), `otsl/ood.py`, `results/uc12/ood_eval.jsonl`; used read-only).

**Setup.** The selected rule B-hg, unchanged (thresholds, models and guards from validation; nothing tuned on shifted
data), and B-hg followed by the LP-relaxation guard at test time ("+ LP guard"), on the robustness study's sets:
in-distribution (test_fresh 0–39) and 30 instances each of load +15 %, load −15 %, wind + solar ×1.5, 2–3 thermal units
out, 1–2 loaded lines out, and windows across midnight (220 instances). Outages exactly as there: the UC model forces
unavailable units off and removes outaged lines (`OODModel`), the GNN's on-probabilities of unavailable units are set to
0, guards use the available fleet (`avail_view`), fixings of unavailable units are forced OFF after the guards, and the
fixed share counts available units only; MILPs through `otsl.ood.solve_milp` (highspy, one thread, 60 s, 0.1 %).
**Timing**: 14 full MILPs (2 per set) re-solved on core 3 agree with the stored times of the robustness study (per-instance
ratio median 0.96, ratio of means 0.96, 12 / 14 within 10 %, identical objectives on all 14), so the stored full-MILP
times and dual bounds are the reference for every rule; gaps are therefore exactly paired with the study's records. The
re-timed full MILPs ran ≈ 4 % faster than the stored ones, so the m1x speed-ups may be flattered by up to ≈ 4 %.

**Results** (gap to the full MILP's dual bound, mean [95 % CI]; speed-up mean / median; fixed share; all 100 % feasible):

| set | LtF-kNN | hybrid | **B-hg** | **B-hg + LP guard** |
|---|---|---|---|---|
| in-distribution (40) | 0.42 % [0.27, 0.63], 2.9× / 1.6×, 68 % | 0.31 % [0.21, 0.43], 4.7× / 2.3×, 86 % | 0.28 % [0.18, 0.39], 7.8× / 3.8×, 89 % | 0.28 %, 6.9× / 3.7×, 89 % |
| load +15 % | 4.04 % [0.67, 10.06], 2.5× / 1.6×, 67 % | 0.29 % [0.15, 0.50], 3.4× / 1.9×, 86 % | **0.18 %** [0.11, 0.26], 6.6× / 2.6×, 89 % | 0.18 %, 5.7× / 2.3×, 89 % |
| load −15 % | 1.54 % [0.74, 2.52], 4.9× / 1.7×, 66 % | 0.97 % [0.37, 1.81], 3.6× / 1.8×, 80 % | 1.64 % [0.42, 3.54], 5.8× / 3.0×, 88 % | 1.64 %, 5.5× / 3.1×, 88 % |
| wind + solar ×1.5 | 1.15 % [0.65, 1.79], 2.3× / 1.8×, 66 % (97 % feasible) | 1.66 % [0.71, 3.13], 4.3× / 2.7×, 78 % | 1.40 % [0.68, 2.37], 6.1× / 3.6×, 86 % | 1.40 %, 5.6× / 3.5×, 86 % |
| 2–3 units out | 0.85 % [0.46, 1.33], 2.6× / 1.3×, 66 % | 0.82 % [0.35, 1.42], 6.1× / 1.9×, 79 % | 0.63 % [0.34, 0.96], 6.2× / 4.7×, 86 % | 0.63 %, 5.9× / 4.8×, 86 % |
| **1–2 lines out** | 26.0 % [14.2, 38.8], 6.0× / 1.9×, 69 % | 15.9 % [8.8, 23.6], 7.4× / 4.8×, 86 % | **17.7 %** [9.9, 26.3], 10.3× / 3.4×, 89 % | **0.23 %** [0.15, 0.32], 6.9× / 2.2×, 70 % |
| across midnight | 0.93 % [0.60, 1.29], 4.5× / 1.4×, 64 % | 0.30 % [0.20, 0.42], 6.2× / 1.8×, 84 % | 0.22 % [0.14, 0.33], 7.3× / 3.1×, 84 % | 0.22 %, 6.2× / 2.9×, 84 % |

Paired against the hybrid on the same instances (Δ gap pp / Δ log speed-up, 95 % CI): in-distribution −0.04 [−0.14, +0.06]
/ +0.50 [+0.17, +0.86]; load +15 % −0.11 [−0.26, −0.01] / +0.39 [+0.02, +0.77]; load −15 % +0.67 [−0.05, +1.77] /
+0.46 [+0.20, +0.74]; wind + solar −0.27 [−1.22, +0.62] / +0.37 [+0.17, +0.58]; units out −0.20 [−0.73, +0.21] /
+0.40 [+0.00, +0.73]; lines out +1.87 [−0.15, +4.11] / +0.17 [−0.20, +0.53]; across midnight −0.07 [−0.16, +0.01] /
+0.16 [−0.22, +0.53]. With the LP guard on line outages: −15.6 pp [−23.4, −8.6] against the hybrid, −25.8 pp
[−38.6, −14.0] against LtF-kNN, Δ log speed-up −0.29 [−0.78, +0.18] against the hybrid. Against LtF-kNN, B-hg is faster
on every set (Δ log speed-up +0.52 to +0.94, every CI above 0) and fixes 20–22 pp more; full table in the results file.

* **The better probabilities carry over to five of the six shifts.** B-hg is at least as accurate as the hybrid on
  load +15 % (better on both gap and speed), wind + solar ×1.5, units out and windows across midnight, as in
  distribution, and faster (significant on log speed-up except across midnight, +0.16 [−0.22, +0.53]);
  on load −15 % it is faster but its mean gap is higher (1.64 % vs 0.97 %, CI of the difference touches 0), driven by
  one very low-load instance (27 % above a 54 k$ optimum, no shedding: a wrong cheap-unit choice that no guard sees).
* **It collapses under line outages like every LtF-calibrated rule** (17.7 % mean, 11 of 30 instances above 10 %;
  hybrid 15.9 %, LtF-kNN 26 %). Same cause as found by the robustness study: units 30 and 31 (the two CTs at one bus,
  never on in the validation set) have collapsed thresholds at τ̲ = τ̄ = 0.5 (24 of 73 generators are collapsed), so
  they are fixed by rounding with no margin; the GNN does not see the outaged line, under-predicts them in the hours a
  re-routed flow needs them, and they are fixed OFF (units 30 / 31 account for 65 of the 127 wrong OFF fixings on this
  set, before guards). Better in-distribution probabilities do not help when the model cannot see the shift.
* **The LP-relaxation guard repairs it**: 0.23 % [0.15, 0.32] mean, max 0.97 %, 93 % served (B-hg alone 63 %),
  at 6.9× mean / 2.2× median, by releasing fixings (70 % fixed instead of 89 %). On the other sets it releases nothing
  (identical gaps) and costs 5–15 % of the speed-up (one or two extra LPs). As a default the guarded variant is the
  safer rule: it gives up about 10 % of the speed-up in distribution for robustness to topology changes.
* B-hg + LP guard on line outages is also more accurate than the robustness study's own guarded rules there
  (guarded error-cost 95 % + LP guard 0.47 % at 7.1×, 90 % rule 0.68 % at 6.5×; not paired here).

## Caveats

* **Single seeds on the learning curves** (one model per data size and label type); the 5-seed spread of the reference
  recipe (fixable at 99.9 %: 0.49–0.59) is the yardstick, and the selected family was confirmed with 5 seeds
  (0.75–0.78). The share fixable at 99.9 % precision is an extreme-tail statistic (it moves with a few dozen confident
  errors) and was not sufficient on its own: the MLP matches the temporal GNN on it but is worse on everything else.
* **Polished curve stops at 2,000** (budget); the label-free curve reaches 4,000. Polished labels depend on a teacher
  trained on the 500 MILP labels (semi-supervised self-training), so the 2,000-instance models partly distil the
  3-member teacher ensemble; the single student is better than the teacher ensemble on every metric, so it is not only
  distillation.
* **The error-cost model was not retrained** for the new probabilities (its labels come from 26,700 dispatch LPs on
  the plain GNN's out-of-fold errors); B-he is therefore a lower bound for error-cost scores on the better model.
* **Selection steps added during the study** (both validation-only, before any downstream or test result): the
  temporal GNN on 2,000 instances as a candidate, and the "hg" tuning of the selected ensemble.
* **One tuning run per configuration**: the tuned fixed share also varies with the tuning's own randomness (cut order,
  6 s relaxation MILPs); the hybrid study saw 81–84 % across two BCE seeds.
* **Shared machine**: one core of four, the other three busy with other agents' MILP jobs; absolute times are slower than
  in the hybrid study (see the timing check), so only the paired speed-ups (same core, back to back) are compared.
* **Test coverage**: every rule on test_fresh 0–59; instances 60–119 only with the full MILP, the reference hybrid and
  the selected rule (B-hg), for time. A container restart interrupted the second half after instance 103; it resumed
  with the same code and core (finished instances kept).
* Validation includes `val.npz`, which is also the early-stopping set of every model (as for the reference models).
* The GNN feature normalisation uses the statistics of the 500 original instances for every model (same distribution).
