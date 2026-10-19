# Recursive reasoning for unit commitment: the GPU round

*The decisive experiments of [`PLAN.md`](PLAN.md), run on PACE ICE (H200, 1 GPU + 20–24 CPU cores for the LP workers)
on the full 50,000-label training set instead of the 2,000 labels and 4 CPU cores of the first round
([`SUMMARY.md`](SUMMARY.md)). Arm A is the pure TRM-style recursion ([`trm.md`](trm.md)); arm B is recursion with the
exact dispatch LP in the loop ([`rloop.md`](rloop.md)). Every choice was made on the same 360 uc12 validation
instances; the [test pass](#test-pass-120-test_fresh-instances) then ran once on the 120 test_fresh instances. Gap
definitions: arm A's validation tables measure against the stored MILP objective, (C − C\*)/C\*
(`scripts/rec_trm_val.py`); arm B's against that MILP's dual bound, (C − DB)/C; the test pass uses (C − DB)/C for both
arms, DB the tightest bound over every full-MILP solve of the instance. Levels are comparable within a table, not
across the two validation metrics. One seed per configuration.*

## Bottom line

* **The test pass confirms both verdicts** (120 test_fresh instances, every choice fixed on validation beforehand).
  The recursive model loses to the one-shot network on both of the gate's pipelines (−0.36 pp [−0.65, −0.08] with
  1 LP + repair, −0.26 pp [−0.40, −0.12] with 5-threshold screening). On one CPU core its 16 steps take 8 s per
  instance, which makes it slower than the full MILP at a 0.5 % gap on a typical instance as well as 1.15 pp worse. Arm B's feedback again
  wins on feasibility only: +0.55 pp [+0.01, +1.28] overall, −0.05 pp [−0.22, +0.12] on the instances both arms
  serve. Details in [the test section](#test-pass-120-test_fresh-instances).

The validation findings that led there:

* **Arm A's go/no-go gate is not met, and the recursive model loses to plain networks trained to the end.** In paired
  comparisons on the same instances and pipeline, the recursive model is significantly *worse* than the same-size
  one-shot network (−0.24 pp [−0.39, −0.12]) and than a deeper unshared network (−0.51 pp [−0.75, −0.29]), both at
  their final checkpoints. It reaches the 0.2 pp margin the plan asked for only against the one-shot network's
  best-validation-loss checkpoint, and only under one of the two decoding pipelines.
* **Validation loss picked the wrong checkpoints for the controls.** Their validation BCE rises late in training, which
  looked like overfitting, but their *schedules* keep getting cheaper: at the final checkpoint the one-shot network
  scores 1.49 % instead of 1.96 % and the deep network 1.22 % instead of 1.65 %. A first draft of this note drew the
  opposite conclusion from point estimates taken under mixed pipelines; the paired numbers below replace it.
* **More recursion does not help either.** Doubling training to 60,000 steps plateaued in BCE and came out worse
  downstream (1.872 % against 1.731 %, +0.14 pp [+0.02, +0.30] in favour of the 30,000-step model).
* **Arm B's LP feedback buys feasibility, not optimality.** With feedback the loop serves 97.5 % of instances at 16
  steps against the no-feedback control's 92.2 %, and the mean gap is 0.62 pp better. But **restricted to the 329
  instances both arms serve, the feedback arm is not better at all** (0.942 % against 0.857 %, a difference of
  −0.084 pp with a 95 % CI of [−0.222, +0.047]). The mean-gap win is the cost of the instances the control fails to
  serve, which carry unserved energy at $10,000/MWh and sit far out in the tail.
* **The LP relaxation as an input is worth 0.13–0.18 pp** to the recursive model, reproducing §3 U5 and the first
  round at 25× the data, though at this scale the choice of architecture and checkpoint matters more.
* **Scaling the data did not change the first round's verdict.** Every learned model now beats learning-free LP
  rounding, which was not true on CPU — but the ranking among the architectures, and the conclusion that the solver in
  the loop matters more than the recursion around it, both survive.

## Arm A: does latent recursion beat a non-recursive network?

All comparisons are **paired**: the same 360 instances, the same decoding pipeline, and a bootstrap CI (10,000
resamples) of the per-instance difference in cost gap, from `scripts/rec_trm_pair.py`
(`results/recursive/trm_pair_w1_trm.json`). Positive = the recursive model is better. Two pipelines are reported
because they rank the models differently: **with repair** (the cheapest of the model's own schedule and four LP
roundings, then the LP-guided repair if it sheds load) and **without repair** (the same choice, no repair).

The recursive model (30,000 steps; its validation loss ticks up at a few evaluations but is lowest at the last one,
so its best and final checkpoints coincide) scores **1.731 % mean / 0.461 % median / 97.5 % served** with repair and 1.797 % / 0.502 % / 97.2 % without.

| control (parameters) | checkpoint | control's gap, with / without repair | difference, with repair | difference, without repair |
|---|---|---|---|---|
| one-shot (0.63 M, same network) | best validation loss (step 5,000) | 1.959 % / 1.730 % | **+0.228** [+0.086, +0.402] | −0.067 [−0.162, +0.019] |
| one-shot (0.63 M) | final (30,000) | 1.488 % / 1.572 % | **−0.244** [−0.385, −0.122] | **−0.226** [−0.390, −0.078] |
| deep, unshared (13.1 M) | best validation loss (step 3,000) | 1.653 % / 1.654 % | −0.079 [−0.233, +0.097] | −0.143 [−0.291, +0.026] |
| deep, unshared (13.1 M) | final (30,000) | 1.218 % / 1.233 % | **−0.514** [−0.749, −0.290] | **−0.565** [−0.799, −0.344] |
| recursive, 60,000 steps | best validation loss | 1.872 % / 1.804 % | +0.140 [+0.017, +0.297] | +0.006 [−0.080, +0.103] |
| one-shot at equal instance visits | final | 1.966 % / 2.104 % | +0.234 [+0.084, +0.413] | +0.307 [+0.173, +0.482] |
| recursive, no LP-relaxation inputs | final | 1.913 % / 1.927 % | +0.182 [+0.045, +0.348] | +0.130 [+0.016, +0.245] |
| one-shot, no LP-relaxation inputs | final | 1.963 % / 1.933 % | +0.232 [+0.092, +0.402] | +0.135 [−0.008, +0.304] |

For reference: the `m1x` one-shot temporal GNN of the `unit-commitment` branch 1.957 %, and learning-free LP rounding
+ repair 2.112 % (point estimates). Every learned model beats LP rounding at this data scale.

**The gate fails.** With every model at its final checkpoint — the default, and the checkpoint the recursive model
itself is evaluated at — the recursive model is significantly worse than both controls, under both pipelines. The
0.2 pp margin of [`PLAN.md`](PLAN.md) appears only against the one-shot network's best-validation-loss checkpoint, and
only with repair; without repair the same comparison is a tie. A result that flips sign with the decoding pipeline is
not one to build on.

**Validation loss chose the wrong checkpoints.** The controls' validation BCE bottoms out early and then rises (one-shot
0.0401 at step 6,000 → 0.0444 at the end; deep 0.0408 at step 2,000 → 0.0951), which was read as overfitting and was
why they were retrained with best-loss checkpointing (the retrained runs, evaluated every 500 steps, bottom out at
steps 5,000 and 3,000). Downstream the opposite holds: the late checkpoints produce markedly cheaper
schedules. The loss presumably rises because the networks become confidently wrong on a minority of unit-hours,
while the decoded schedules — after LP-rounding candidates and repair — keep improving. The recursive model's loss does not drift
up the same way (the TRM paper's claim about overfitting reproduces on the proxy), but that robustness does not translate into better
schedules. This is the same lesson as §3 U7 of [`../RESEARCH.md`](../RESEARCH.md): proxies mislead, and only the cost
gap decides. A first draft of this section reported the opposite, from point estimates taken under different pipelines
for different models; it is superseded by the paired table.

**Where the recursion does help.** It beats the one-shot network trained on the same number of instance visits, and
both no-LP variants. But when the plain networks are simply trained for the same number of optimizer steps, they are
better.

**The deep control is not parameter-matched.** It unrolls the recursion's depth without weight sharing (13.1 M
parameters against 0.63 M), so it matches compute rather than parameters.

## Arm B: does the solver in the loop help?

Both arms roll out 24 refinement steps over the same 360 instances and store a per-instance cost at every step, so
the comparison is **paired** — which matters, because the spread across instances is an order of magnitude larger
than the difference between the arms. `scripts/rec_rloop_pair.py` does the comparison:

```
python3 scripts/rec_rloop_pair.py --a w2_rloop_rl --b w1_rloop_nofb_rl
```

Both arms were trained identically (60 supervised epochs on 50k instances, then 1,500 REINFORCE updates); the only
difference is whether the LP's signals — hourly and reserve shortfall, prices, line loadings, unit dispatch — are fed
back between steps.

| k steps | feedback | no feedback | paired diff [95 % CI] | served, feedback / none |
|---|---|---|---|---|
| 1 | 4.674 % | 3.520 % | **−1.155** [−2.206, −0.183] | 85.3 % / 88.6 % |
| 2 | 2.420 % | 2.786 % | +0.365 [−0.201, +1.016] | 94.2 % / 90.3 % |
| 4 | 1.627 % | 2.564 % | **+0.936** [+0.363, +1.615] | 97.2 % / 90.0 % |
| 8 | 1.403 % | 2.109 % | **+0.707** [+0.218, +1.293] | 97.5 % / 91.7 % |
| 16 | 1.205 % | 1.826 % | **+0.621** [+0.231, +1.078] | 97.5 % / 92.2 % |
| 24 | 1.037 % | 1.543 % | **+0.506** [+0.137, +0.964] | 98.3 % / 93.3 % |

*(positive = feedback is better; CI from 10,000 bootstrap resamples of the per-instance difference)*

### The feedback model trades its first guess for its last

At one step the feedback model is **significantly worse** (4.674 % against 3.520 %). It was trained to lean on the
loop, so its opening schedule is poor; it overtakes the control by step 4 and the advantage then settles at
0.5–0.6 pp. The k-curve makes the same point from the other side: the feedback arm falls 4.674 → 1.037 % over 24
steps, the control only 3.520 → 1.543 %. The refinement, not the first pass, is doing the work. For a model meant to
be run to convergence this is a fair trade, but it rules out using the feedback model one-shot.

### The mean-gap win is a feasibility win

The headline 0.62 pp looks like a clean victory and is not one. Two observations contradict it:

* **The median is a tie.** 0.225 % against 0.245 % at 16 steps; at 8 steps the two medians are equal to three
  decimals. Per instance the record is 36 % wins / 33 % ties / 31 % losses — near-symmetric.
* **Restricted to instances both arms serve, the advantage disappears and reverses slightly.**

| k steps | n both serve | feedback | no feedback | paired diff [95 % CI] |
|---|---|---|---|---|
| 1 | 294 | 1.657 % | 1.398 % | **−0.259** [−0.426, −0.105] |
| 2 | 318 | 1.408 % | 1.193 % | **−0.215** [−0.350, −0.087] |
| 4 | 322 | 1.251 % | 1.072 % | −0.179 [−0.398, +0.030] |
| 8 | 327 | 1.067 % | 0.933 % | −0.134 [−0.323, +0.022] |
| 16 | 329 | 0.942 % | 0.857 % | −0.084 [−0.222, +0.047] |
| 24 | 334 | 0.908 % | 0.846 % | −0.062 [−0.168, +0.049] |

So the whole mean-gap advantage of putting the solver in the loop comes from **finding a feasible schedule more
often** — 97.5 % against 92.2 % at 16 steps — and none of it from finding a cheaper one. Conditional on both arms
serving the instance, feedback is a statistically insignificant 0.06–0.08 pp *worse* at 8 steps and beyond, and
significantly worse at 1–2 steps.

This is worth stating plainly because the unconditional mean would have supported a much stronger claim. An unserved
instance prices unserved energy at $10,000/MWh, so a handful of them dominate a mean of 360 cost gaps; a method that
only fixes feasibility will look like a method that improves cost. The two effects have to be separated to see which
one happened.

That said, feasibility is the more valuable of the two in operation: a schedule that sheds load is not a usable
answer at any cost, while 0.08 pp of optimality is noise against the 0.3–0.5 pp that the MILP pipelines give up
anyway. The honest summary is that **the LP in the loop is a feasibility repair mechanism that the network learns to
drive, not a route to cheaper schedules.**

### The halting head improves too

Run each model to the step its own halting head picks, which is what a dispatcher would actually do:

| | mean gap | median gap | served | mean steps used |
|---|---|---|---|---|
| feedback | **1.632 %** | **0.526 %** | **95.8 %** | **5.0** |
| no feedback | 2.260 % | 0.616 % | 90.8 % | 6.5 |

The feedback model stops **earlier** (5.0 steps against 6.5) and still lands better on every measure. Exact feedback
tells the halting head when the schedule is actually feasible, where the no-feedback model has to guess — the same
feasibility effect as above, now showing up as better-calibrated early stopping.

### What the feedback costs

Training the feedback arm took roughly **3× the wall-clock per epoch** of the control (≈ 21 min against ≈ 7 min)
because every refinement step of every training example solves dispatch LPs: 2.4 million LP solves and 57,000 LP
seconds across the REINFORCE stage alone, carried by 20 CPU worker processes beside the GPU. The supervised stage plus
1,500 REINFORCE updates came to about 13 GPU-hours in resumable segments. That is the price of the feasibility gain,
and it is worth recording next to the gain itself: the no-feedback control reaches 92 % served for a third of the
compute.

## Test pass (120 test_fresh instances)

One pass, after every choice was fixed on validation: arm A's plan
(`results/recursive/gpu_round/trm_select_gpu.json`, from `scripts/rec_trm_select.py`) and arm B's
(`rloop_select_gpu.json`: 24 steps, safety net at the preset 1 % certified gap). For each instance, one process on
one CPU core solves the full MILP (0.1 % and 0.5 %, 60 s) and runs every pipeline back to back, so speed-ups are
paired. Gap is (C − DB)/C against the tightest proven bound of the instance over every full-MILP solve of both arms.
All numbers below come from `scripts/rec_gpu_test_report.py` (`results/recursive/gpu_round/test_results.md`, every
row and pair; per-instance records alongside).

The two arms' records line up exactly: the LP relaxation, solved by both, has identical cost on every instance, and
learning-free LP rounding, run by both, gives identical costs (1.744 %). The full MILPs differ by more than 0.2 % on
six instances, all of which hit the 60 s limit; arm A, on the faster node, found the better incumbent on all six.

### Arm A on test

The recursive model is the retrained 30,000-step `w1_trm30`, a bit-for-bit reproduction of the model validated
above (the original checkpoint had been overwritten by the 60,000-step continuation; its retrain matches every
logged loss and every validation array exactly). Each model uses its own validation-chosen decoders and thresholds.

| model (checkpoint) | mixed + repair (headline) | 1 LP + repair | 5 thresholds + repair | + safety net | speed-up (mixed), mean / geo |
|---|---|---|---|---|---|
| recursive, 16 steps (final) | 1.374 % | 2.003 % | 1.378 % | 1.126 % | 2.6× / 1.7× |
| one-shot (final) | 1.240 % | 1.645 % | 1.120 % | 0.892 % | 33.8× / 22.6× |
| deep, unshared (final) | **1.021 %** | **1.415 %** | 1.268 % | **0.753 %** | 19.7× / 13.1× |
| one-shot (best validation loss) | 1.574 % | 1.935 % | 1.325 % | 1.160 % | 36.5× / 24.6× |
| deep (best validation loss) | 1.303 % | 1.992 % | 1.469 % | 1.104 % | 19.7× / 13.2× |
| m1x temporal GNN | 1.529 % | 2.576 % | 1.578 % | — | 33.9× / 22.9× |
| LP rounding, no learning | 1.707 % | — | — | — | 40.8× / 27.8× |

Paired differences, recursive minus control (positive = the recursive model is better):

| control | mixed + repair | 1 LP + repair | 5 thresholds + repair |
|---|---|---|---|
| one-shot, final | **−0.133** [−0.241, −0.029] | **−0.358** [−0.646, −0.084] | **−0.258** [−0.396, −0.122] |
| deep, final | **−0.353** [−0.710, −0.053] | **−0.588** [−0.948, −0.222] | −0.111 [−0.431, +0.240] |
| one-shot, best loss | **+0.200** [+0.011, +0.410] | −0.068 [−0.525, +0.517] | −0.053 [−0.283, +0.188] |
| deep, best loss | −0.071 [−0.181, +0.031] | −0.011 [−0.343, +0.325] | +0.091 [−0.121, +0.320] |

* **The gate fails on test as on validation.** It asks the recursive model to beat the one-shot by more than about
  0.2 pp on the 1 LP + repair and 5-threshold pipelines ([`trm.md`](trm.md)). Against the final one-shot it is
  significantly worse on both; against the best-loss one-shot it is level on both. The only win is the one validation
  also showed — the mixed pipeline against the best-loss one-shot, +0.200 pp, now with a CI that barely clears zero.
* **The final checkpoints are again the better controls**, by 0.2–0.6 pp of mean gap in every pipeline; the
  checkpoint lesson replicates.
* **Recursion at test time buys little.** 16 steps against 1 improve the 1 LP + repair pipeline by +0.146 pp
  [−0.033, +0.324] (2.149 % → 2.003 %), not significant, and 32 steps add nothing (2.008 %). The learned halting head
  rarely stops (mean 23 steps, median the 32-step cap) and comes out at 2.093 %.
* **The recursion is expensive on a CPU.** Each supervision step is 21 evaluations of the shared network (3 recursions of 6 latent updates and 1 answer
  update), about 0.5 s on one core:
  0.8 s at one step, 8.1 s at 16, 15.9 s at 32 per instance. The full MILP takes 21.6 s at 0.1 % and 11.7 s at
  0.5 % on the same core (medians 17.8 s and 6.6 s), so the recursive model's headline pipeline runs at 2.6× the 0.1 %
  MILP and is *slower* than the 0.5 % MILP on a typical instance (−0.29 log speed-up [−0.50, −0.08]; only the hardest
  instances pull the MILP's mean above 8 s) while 1.15 pp [0.80, 1.58] worse. The one-shot and deep pipelines take
  0.4–1.1 s. A GPU would shrink the gap; it would not change the ranking.
* **Against the baselines**, the recursive model's headline beats LP rounding (+0.333 [+0.184, +0.524]) and the m1x
  network (+0.155 [+0.034, +0.325]) — as do the plain networks, by more.

### Arm B on test

| pipeline | gap mean | median | served | speed-up mean / geo |
|---|---|---|---|---|
| full MILP, 0.1 % | 0.221 % | 0.096 % | 96.7 % | 1.0× |
| full MILP, 0.5 % | 0.261 % | 0.101 % | 96.7 % | 3.6× / 2.1× |
| feedback, 24 steps + safety net | 0.342 % | 0.171 % | 100 % | 3.9× / 2.9× |
| no feedback, 24 steps + safety net | 0.320 % | 0.192 % | 100 % | 3.8× / 2.9× |
| feedback, 24 steps | 0.763 % | 0.227 % | 99.2 % | 5.4× / 3.8× |
| feedback, halting head | 1.253 % | 0.511 % | 97.5 % | 25.0× / 18.9× |
| no feedback, 24 steps | 1.316 % | 0.230 % | 94.2 % | 5.4× / 3.8× |
| no feedback, halting head | 1.608 % | 0.608 % | 94.2 % | 25.1× / 17.8× |
| LP-checked local search, no learning | 1.222 % | 0.808 % | 99.2 % | 16.6× / 11.8× |
| LP rounding, no learning | 1.744 % | 0.916 % | 98.3 % | 52.0× / 37.7× |

* **Feedback is a feasibility gain, again.** At 24 steps it is +0.553 pp [+0.010, +1.282] better overall, serving
  99.2 % against 94.2 %; on the 112 instances both serve it is −0.050 pp [−0.215, +0.123]. The medians tie
  (0.227 % against 0.230 %). The halting-head comparison is +0.356 [−0.373, +1.154], not significant on 120 instances.
* **The safety net levels the two arms** (−0.023 pp [−0.061, +0.013]): once a reduced MILP backs up the failures, the
  feasibility advantage has nothing left to do. With it the feedback model serves every instance at 0.342 %, 0.08 pp
  [0.02, 0.15] worse than the 0.5 % MILP and 1.35× faster than it (+0.30 log speed-up [+0.14, +0.46]).
* **Without the safety net** the loop beats both learning-free baselines: +0.981 pp [+0.628, +1.388] against LP
  rounding and +0.459 pp [+0.253, +0.672] against the LP-checked local search, at a tenth and a third of their speed.

### Across the arms

The headline pipelines of the two arms, on the same instances: arm B's feedback loop at 24 steps is 0.611 pp
[0.322, 0.965] better than arm A's recursive model, and faster relative to its own full MILP. Arm A's best pipeline on
test is the deep control with the safety net (0.753 %, 99.2 % served, 20× mean / 14× geometric speed-up), a different
point on the trade-off than arm B's 0.342 % at 4× / 3×. Neither reaches the MILP pipeline of
[`../RESEARCH.md`](../RESEARCH.md) §7–§8 (0.30 % at ~10×). The safety nets fired on 11–16 % of test instances in arm A,
under the 20 % cap set on validation.

**Caveats.** One seed per model and 120 test instances, so CIs of 0.1–0.4 pp; many pairs without a multiplicity
correction, so the conclusions rest on directions that agree across pipelines and with validation rather than on any
single interval. Hardware: arm A ran on the Granite Rapids node (Xeon 6972P), which has bf16 units — the TRM models run
in bfloat16 and CPUs without bf16 support emulate it many times slower — while arm B's tasks were spread over three
CPU types, its speed-up at 24 steps ranging from 2.6× (AMD EPYC 7513) to 4.2× (Xeon Gold 6226) by geometric mean.
LP rounding, identical in both arms, runs at 41× in arm A's records and 52× in arm B's, which sets the scale of
hardware effects on cross-arm speed-ups; within an arm, speed-ups are paired on one core.

## What this round settles

1. **Latent recursion is not the lever** for unit commitment, at 2k labels or at 50k. Arm A's gate failed at both
   scales, and at 50k a plain one-shot or deeper unshared network trained to the end is significantly better than the
   recursive model.
2. **The solver in the loop is the lever, and it works on feasibility.** This sharpens the first round's "feedback
   beats no feedback by 0.8–1.3 pp" into something more specific and more useful: the gain is in how often a
   non-shedding schedule is found, not in how cheap it is.
3. **Neither arm displaces the MILP pipeline** of [`../RESEARCH.md`](../RESEARCH.md) §7–§8 (0.30 % at ~10×). The best
   end-to-end numbers here are 1.0–1.2 % mean gap, and the best pipelines in the first round only got to 0.46–0.64 %
   by falling back to a reduced MILP on a quarter to a half of instances — at which point the MILP is doing the work.
4. **Fix the checkpoint rule in advance, compare paired, hold the pipeline fixed, and split feasibility from cost.**
   Each of these changed a conclusion in this round: best-validation-loss checkpoints made the controls look worse
   than they are, mixing pipelines across models turned a loss into a tie, and an unconditional mean made a
   feasibility gain look like a cost gain.
5. **The test set agrees.** The single pass on the 120 test_fresh instances reproduces every verdict above: the
   recursive model loses to final-checkpoint plain networks, feedback buys feasibility and not cost, and neither arm
   reaches the MILP pipeline.

## Reproducing

```
# arm A, one validated model per array task, on the bf16-capable CPU node (lpround first: the mixed pipeline needs it)
python3 scripts/pace/pgu_guard.py submit --kind cpu --name vala --array 0-7%8 --cpus 2 --mem 16G \
    --time 02:00:00 --constraint cpu-gnr scripts/pace/pgu_val_arma.sbatch
python3 scripts/rec_trm_pair.py --a w1_trm --b w1b_oneshot_best w1_oneshot w1b_deep_best w1_deep w1_trm_best \
    w1_os_eqep w1_trm_nolp w1_os_nolp --pipelines mix5_rep__rep mix5_plain__plain

# arm B, resumable segments (each picks up from the checkpoint): submit one segment, and the next once it finishes
python3 scripts/pace/pgu_guard.py submit --kind gpu --name w2-rloop-s1 --cpus 24 --mem 64G --time 08:00:00 \
    --export "TAG=w2_rloop,N_TRAIN=50000,EXTRA=--data_dir data/generated/recursive/pace" scripts/pace/rloop_train.sbatch
python3 scripts/rec_rloop_pair.py --a w2_rloop_rl --b w1_rloop_nofb_rl

# the single test pass (120 test_fresh instances, one core per task, paired timing), then both arms in one report
python3 scripts/rec_trm_select.py --final w1_trm30 --controls w1_oneshot w1_deep w1b_oneshot_best w1b_deep_best \
    --val_dirs results/recursive/gpu_round results/recursive --out results/recursive/gpu_round/trm_select_gpu.json
python3 scripts/pace/pgu_guard.py submit --kind cpu --name testa --array 0-7%8 --cpus 1 --mem 12G \
    --time 08:00:00 --constraint cpu-gnr scripts/pace/pgu_test_arma.sbatch
python3 scripts/pace/pgu_guard.py submit --kind cpu --name testb --array 0-7%8 --cpus 1 --mem 8G \
    --time 06:00:00 scripts/pace/pgu_test_armb.sbatch
python3 scripts/rec_gpu_test_report.py
```

Both arms' per-instance validation arrays are committed under `results/recursive/` as
`rloop_val_<tag>.npz` (a 24 × 360 cost matrix, plus shedding, shortfall and halting probabilities) and
`trm_val_<tag>.json`, so every number above can be recomputed without a GPU.
