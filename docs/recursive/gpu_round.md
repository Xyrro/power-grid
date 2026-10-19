# Recursive reasoning for unit commitment: the GPU round

*The decisive experiments of [`PLAN.md`](PLAN.md), run on PACE ICE (H200, 1 GPU + 20–24 CPU cores for the LP workers)
on the full 50,000-label training set instead of the 2,000 labels and 4 CPU cores of the first round
([`SUMMARY.md`](SUMMARY.md)). Arm A is the pure TRM-style recursion ([`trm.md`](trm.md)); arm B is recursion with the
exact dispatch LP in the loop ([`rloop.md`](rloop.md)). Selection and all numbers below are on the same 360 uc12
validation instances; gap is to the dual bound of the full MILP at a 0.1 % gap. One seed per configuration.*

## Bottom line

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

The recursive model (30,000 steps; its validation loss falls monotonically, so its best and final checkpoints
coincide) scores **1.731 % mean / 0.461 % median / 97.5 % served** with repair and 1.797 % / 0.502 % / 97.2 % without.

| control (parameters) | checkpoint | control's gap, with / without repair | difference, with repair | difference, without repair |
|---|---|---|---|---|
| one-shot (0.63 M, same network) | best validation loss (≈ step 6,000) | 1.959 % / 1.730 % | **+0.228** [+0.086, +0.402] | −0.067 [−0.162, +0.019] |
| one-shot (0.63 M) | final (30,000) | 1.488 % / 1.572 % | **−0.244** [−0.385, −0.122] | **−0.226** [−0.390, −0.078] |
| deep, unshared (13.1 M) | best validation loss | 1.653 % / 1.654 % | −0.079 [−0.233, +0.097] | −0.143 [−0.291, +0.026] |
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
0.0401 at step 6,000 → 0.0444 at the end; deep 0.0503 → 0.0951), which was read as overfitting and was why they were
retrained with best-loss checkpointing. Downstream the opposite holds: the late checkpoints produce markedly cheaper
schedules. The loss rises because the networks become confidently wrong on a minority of unit-hours, while the
decoded schedules — after LP-rounding candidates and repair — keep improving. The recursive model's loss does not rise
(the TRM paper's claim about overfitting reproduces on the proxy), but that robustness does not translate into better
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
5. **Not yet done: the test set.** Every number here is on the 360 validation instances. The protocol's single test
   pass on the 120 test_fresh instances (paired timing on CPU) is the remaining step, for the models selected here.

## Reproducing

```
# arm A, one validated model per array task, each at its best checkpoint
python3 scripts/pace/pgu_guard.py submit --kind cpu --name vala --array 0-6%7 --cpus 2 --mem 16G \
    --time 16:00:00 scripts/pace/pgu_val_arma.sbatch
python3 scripts/rec_trm_pair.py --a w1_trm --b w1b_oneshot_best w1_oneshot w1b_deep_best w1_deep w1_trm_best \
    w1_os_eqep w1_trm_nolp w1_os_nolp --pipelines mix5_rep__rep mix5_plain__plain

# arm B, resumable segments (each picks up from the checkpoint; the chain is linked with afterany)
python3 scripts/pace/pgu_guard.py submit --kind gpu --name w2-rloop-s1 --cpus 24 --mem 64G \
    --time 08:00:00 scripts/pace/rloop_train.sbatch w2_rloop
python3 scripts/rec_rloop_pair.py --a w2_rloop_rl --b w1_rloop_nofb_rl
```

Both arms' per-instance validation arrays are committed under `results/recursive/` as
`rloop_val_<tag>.npz` (a 24 × 360 cost matrix, plus shedding, shortfall and halting probabilities) and
`trm_val_<tag>.json`, so every number above can be recomputed without a GPU.
