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
- 15:34 label-free curve done (500–4000); teacher polishing of 1,500 extra instances running (4.8 s per label).
- Phases 2–4 (`scripts/uc_m1x_queue2.sh phase23`, log `data/generated/uc12_m1x/queue_phase2.log`) start automatically
  after phase 1: BCE seeds 0/3/4 at 500 MILP labels, 5-member ensemble, temporal GNN and MLP at 500, guarded-rule
  validation check of all sources (phase 2); validation-only selection (`scripts/uc_m1x_select.py`, rule written before
  the results) and the two tuning runs (phase 3); test evaluation on test_fresh 0–59 then 60–119 + report (phase 4).
- To resume after a restart: rerun `scripts/uc_m1x_queue.sh phase1` if it had not finished, then
  `scripts/uc_m1x_queue2.sh phase23` (finished steps are skipped or cheap; an interrupted tuning run restarts from scratch).

## Summary

(pending)

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

**2. Deep ensembles.** Mean probability of independently seeded models (5 members). Disagreement (std over members)
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
on the same 360 validation instances.

## Setup

* **Compute.** One core (core 3 of a 4-core machine shared with three other agents' jobs), HiGHS and PyTorch
  single-threaded, every job alone on the core, in the order of `scripts/uc_m1x_queue.sh`.
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

(pending)

## Verdict

(pending)

## Caveats

(pending)
