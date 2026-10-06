# Learning unit commitment with a two-stage GNN → LP → NN framework: critique, new methods, results

*Working research log. Code: this repository. Prior work: [`literature_uc.md`](literature_uc.md),
[`raclearn_comparison.md`](raclearn_comparison.md). Raw numbers: `results/uc1/`, `results/uc12/`.
A side study that (wrongly) read "switching status" as transmission-line switching is kept in
[`ots/RESEARCH_OTS.md`](ots/RESEARCH_OTS.md).*

**TL;DR** (RTS-GMLC, 73 thermal units; B1 = one hour, B2 = twelve hours with min up/down, ramping, start-ups).

1. *Model 1 → LP* is published (Tang et al. 2023; RACLearn adds confidence fixing). What can be new is below.
2. **Many MILP labels are arbitrary** — 34 % of B1 hours have an exactly tied optimum; 21 % (B1) and 62.5 % (B2)
   of labels change when identical units are reordered — so BCE and the MSE test metric penalise correct
   answers (U1, U7, V1).
3. **The framework's one-shot pipeline under-commits**: 59–74 % of B1 hours and 10–17 % of B2 instances are
   served without shedding or reserve shortfall (MILP: 98.7 % / 92.5 %) (U3, V2).
4. **Fine-tuning Model 1 with the exact dispatch LP as critic** (REINFORCE, 1–13 min) fixes this: 93.9 % (B1)
   and 80.8 % (B2) with one LP; 97.4 % / 87.5 % with candidate screening (U4, V2).
5. **MILP labels are not needed**: imitating the repaired LP relaxation (18× / 87× cheaper per label) and then
   fine-tuning matches or beats the MILP-labelled model on both benchmarks (94.1 % / 84.2 % one-shot). The
   imitation stage is needed: REINFORCE from scratch over-commits (U8, V4).
6. **The dashed arrow hurts**: every Model 1 trained through a frozen learned Model 2 is worse than plain
   imitation; the framework's direct (PG, VA) Model 2 drives it to commit 18–27 units instead of 15. The exact
   LP is the critic to use (U6).
7. **Model 2 should be a physics decoder** (unit positions → closed-form balance → VA from DC power flow):
   70–74 % fully feasible vs 0 % for direct (PG, VA) regression; it judges commitments only if the implied
   shedding / reserve shortfall is priced, and even then the exact LP is the better screener (U6).
8. **Partial fixing + MILP**: RACLearn's confidence rule is right up to 90 % fixed on B1 (4×) and 80 % on B2
   (1.9×, slightly *cheaper* than the time-limited full MILP). Beyond that, new rankings help: on B2 at 90 %,
   penalising OFF fixes + a solver-free adequacy guard (0.41 % vs 1.38 %, 4.7×); at 95 % on both benchmarks,
   the cost-aware model's probabilities (B1 1.3 % vs 18 %; B2 5.5 % vs 38.5 %, 27×) (U5, V3).
9. **Open problem**: on B2 the fine-tuned model buys coverage with extra units (+4.7–9.1 % cost on served
   instances); learning alone does not reach MILP quality there, the hybrid with a reduced MILP does.
10. **Parallel study of three new frameworks (§5), confirmed with 3 seeds on a fresh test set (§6)**: a min
   up/down-aware block adequacy repair lets one LP serve ~95 % of B2 instances; self-training with reduced MILPs
   gives near-optimal labels at 37 % of the MILP-label cost; a fixing rule learned from LP-priced errors reaches
   0.28 % mean gap at 3.2× (RACLearn: 12.4 %). Constrained (Lagrangian) fine-tuning alone did not beat REINFORCE.
11. **Best pipelines and Learning to Fix (§6)**: end-to-end, 93.3 % served at 1.04 % median gap with ~6 LPs and no
   MILP; with a reduced MILP, 0.59 % at 8.0× and 0.84 % at 12.6× (paper metrics). Learning to Fix implemented from
   the paper reproduces its quality (0.40 % vs 0.48 %) but not its speed-up (4.7× vs 20.8×) on our 12-hour UC; on our
   GNN probabilities it is level with our best rules (0.28 % at 5.2×; 0.85 % at 13.5×) — we do not beat it, our
   models make it better. On 24 hours the MILP finds a 0.5 %-good schedule in a median 28 s.

## 1. The framework and the benchmarks

```
PD (+ previous on/off status) ──► Model 1 (GNN) ──► on/off u ∈ {0,1}^(T×G) ──► LP: dispatch | u ──► PG, VA
                                       ▲ BCE vs MILP u*                            │
                                       └──────── (dashed) Model 2 output feeds Model 1's loss
Model 2: (PD, u) ──► (PG, VA), MSE vs LP;   validation: violations + gap;   test: MSE vs MILP solution
```

**System: RTS-GMLC** (73 buses, 120 branches, 73 thermal units: 39 CT, 23 steam, 10 CC, 1 nuclear;
80 PV/wind/hydro units with hourly 2020 profiles; 3 load regions). Unit data from the RTS-GMLC tables:
min up/down times, ramp rates, hot-start costs, piecewise-linear heat-rate costs (3 segments), fuel
prices. Spinning reserve 3 % of load. DC network with angles (gives the framework's VA). Load shedding,
over-generation and reserve shortfall are **soft** (VOLL $10,000/MWh, reserve $1,000/MWh) — as in
MISO's RAC and RACLearn — so every commitment has a feasible dispatch and the "LP solver" box never fails.

| benchmark | horizon | what is decided | MILP (HiGHS) | data |
|---|---|---|---|---|
| **B1** (main) | 1 hour | on/off of 73 units given demand, renewables and the units running in the previous hour (start-up costs) | 0.5 s, all optimal (gap 1e-5) | 4,000 train / 500 val / 1,000 test |
| **B2** | 12 hours | on/off per unit and hour with min up/down, ramping, start-up costs | 1–40 s (gap 0.1 %, 60 s limit) | 500 / 60 / 120 |

Scenarios: a 2020 day (test days = every 5th day, so the test set covers all seasons and never shares
a day with training), random window, area load × U[0.92, 1.08], bus load × U[0.97, 1.03], each
renewable unit × U[0.8, 1.2]. Initial status: priority-list commitment of the previous hour.

**Metrics.** Every predicted commitment is scored by the exact dispatch LP. Because VOLL-priced
shedding dominates averages, the headline numbers are the share of commitments served **without
shedding or reserve shortfall** (the MILP itself reaches 98.7 % on B1 — it accepts a small reserve
shortfall in 1.3 % of hours) and the **median gap** to the MILP; the mean gap is reported too.

## 2. Where the framework sits in the literature

* **Model 1 → LP is published.** Tang, Bai, Weng & Wang 2023 (Energy Reports) predict unit decisions with a
  GCN and solve the remaining convex problem (13–17×). **RACLearn** (Park et al., IEEE TPS 2024) adds MC-dropout
  confidence, fixes only confident commitments, repairs and solves the reduced MILP (2–4× on a 6,708-bus
  system). Ramesh & Li (IEEE TPS 2024) add a min-up/down feasibility layer; Fritz et al. 2026 ("Learning to
  Fix") calibrate generator-specific, cost-aware fixing thresholds and won EPRI's 2025 UC competition.
* **Simple baselines are strong**: kNN over past schedules (Xavier, Qiu & Ahmed 2021; Pineda & Morales 2022,
  "Is learning for the UC problem a low-hanging fruit?").
* **Identical units** make MILP labels ambiguous and permutation-equivariant GNNs provably unable to tell
  copies apart (Knueven, Ostrowski & Watson 2018; Chen et al., ICLR 2025, which proposes orbit features).
* **Model 2**: dispatch proxies conditioned on commitment exist (Chen et al. 2022 SCED proxies; E2ELR 2024 with
  closed-form repair layers). **Using a learned dispatch model as a differentiable critic for the commitment
  model — the dashed arrow — was not found in the literature**; the obvious competitor is the exact LP's own
  sensitivities (envelope theorem), tested here.
* **Dispatch-LP feedback for training is published** for a separate *repair policy*: He et al. 2026 (IET GTD)
  clone the MILP's commitments, then train a PPO repair policy on LP dispatch cost and feasibility. U4 here
  instead fine-tunes the predictor itself, and U8 / V4 drop the MILP labels entirely. Iterative label
  collection with a solver (DAgger-style) is used for neural diving in UC by Qin & Yu 2023.


## 3. Findings on B1 (single-period network-constrained UC)

### U1. A third of the MILP labels are arbitrary

* **33.6 %** of test instances have a different commitment with *exactly* the same cost (36.3 % within
  0.01 %), found by re-solving with a no-good cut.
* **21.4 %** of labels change when the schedules of identical units (24 groups of 2–5 copies at the same bus)
  are put in canonical order — the solver picks a copy arbitrarily.
* BCE against a single MILP solution therefore penalises correct answers, and MSE against the MILP solution
  (the framework's test metric) does too (see U7).

### U2. Baselines a learned Model 1 must beat (1,000 test hours)

| method | no shed / shortfall | median gap | mean gap | LPs |
|---|---|---|---|---|
| MILP (reference) | 98.7 % | 0 | 0 | MILP 0.5 s |
| keep last hour's units (persistence) | 63.3 % | 1.90 % | 1,344 % | 1 |
| merit-order priority list | 72.5 % | 0.85 % | 995 % | 1 |
| rounded LP relaxation | 65.0 % | 0.049 % | 242 % | 1 + relaxation |
| rounded LP relaxation + adequacy repair | 86.5 % | 0.004 % | 112 % | 1 + relaxation |
| kNN over training schedules, k = 5 / 20, best by LP | 88.3 / **96.6 %** | 0.25 / **0.000 %** | 95 / 14 % | 5 / 20 |

*Adequacy repair*: per hour, switch on the cheapest available units until committed capacity covers net
load + reserve, then switch off the most expensive while minimum outputs exceed demand (polynomial, no solver).
The single-period LP relaxation is tight (1.3 % below the MILP on average), so relax-and-round is already
strong, and kNN with an LP check is near-optimal — both have to be in any comparison.

### U3. The framework's one-shot pipeline (Model 1 → one LP) under-commits

| Model 1 (B1) | one-shot: no shed / median gap | + adequacy repair | candidate screening (~5–8 LPs) |
|---|---|---|---|
| MLP, BCE on MILP labels | 64.0 % / 0.37 % | 84.0 % / 0.15 % | 97.2 % / 0.000 % |
| GNN, BCE on MILP labels (framework) | 62.2 % / 0.78 % | 81.2 % / 0.44 % | 94.9 % / 0.000 % |
| GNN + symmetry rank + canonical labels | 59.4 % / 1.31 % | 80.9 % / 0.47 % | 92.9 % / 0.000 % |
| **GNN + symmetry + LP-relaxation features** | 74.0 % / **0.012 %** | **87.7 % / 0.000 %** | 94.6 % / 0.000 % |

* A third of one-shot commitments shed load or miss reserve: BCE treats a missing unit and an extra unit as
  equally wrong, while the cost of under-commitment (VOLL) is orders of magnitude larger.
* The GNN is **not better than an MLP** on a fixed network, and the framework's GNN is worse than simply
  rounding the LP relaxation.
* **Symmetry handling did not help** here (rank feature + canonical labels, as suggested by the ICLR 2025
  orbit-feature result): one-shot quality dropped slightly (single seed). A plausible reason, not yet
  tested: identical copies sit at the same bus, so swapping them changes neither cost nor flows, and the
  shedding errors that dominate the metric come from *how many* units are on, not *which copy*.
* **LP-relaxation features** (fractional commitments and prices from one LP relaxation solve) are the most
  useful input: best one-shot median gap and best repaired result.

### U4. Cost-aware Model 1 with the exact LP as critic

REINFORCE fine-tuning of the LP-feature GNN, reward = −log(dispatch-LP cost / MILP cost), 200 steps (64 s):

| (B1) | no shed / shortfall | median gap | mean gap | LPs |
|---|---|---|---|---|
| before (BCE), one-shot | 74.0 % | 0.012 % | 273 % | 1 |
| **after REINFORCE, one-shot** | **93.9 %** | 0.051 % | 17 % | 1 |
| after REINFORCE + adequacy repair | 96.6 % | 0.048 % | 13 % | 1 |
| **after REINFORCE + candidate screening** | **97.4 %** (ceiling 98.7 %) | 0.005 % | 6 % | **1.7** |

The cost-aware loss learns the asymmetry BCE cannot see: it commits slightly more (median gap up by 0.04 pp)
and almost stops shedding. With screening it is the best learned pipeline on B1, beating kNN (96.6 % with 20
LPs) at a tenth of the LP solves.

### U5. Which decisions to fix before the MILP (RACLearn-style partial fixing)

Fix a share of the (unit, hour) decisions and solve the reduced MILP. 200 test hours; the full MILP is
re-solved in the same run (0.58 s; all timings under heavy CPU contention, so compare ratios, not seconds).
Mean gap is dominated by rare VOLL / reserve-penalty hours; "matches MILP" is the share at the MILP cost.

| fixed | ranking | mean gap | no shed / shortfall | matches MILP | speed-up |
|---|---|---|---|---|---|
| 80 % | BCE confidence (RACLearn) | **0.014 %** | 98.0 % | **97.5 %** | 2.9× |
| 80 % | REINFORCE probabilities | 0.19 % | 98.0 % | 86.0 % | 3.3× |
| 90 % | BCE confidence (RACLearn) | **0.098 %** | 98.0 % | **91.5 %** | 4.2× |
| 90 % | BCE, asymmetric (OFF errors × 10) | 0.44 % | 97.5 % | 85.5 % | 4.6× |
| 90 % | REINFORCE probabilities | 0.45 % | 98.0 % | 77.0 % | 5.6× |
| 95 % | BCE confidence (RACLearn) | 18.1 % | 91.5 % | 80.0 % | 6.9× |
| 95 % | BCE + adequacy guard | 4.2 % | 95.5 % | 80.0 % | 6.4× |
| 95 % | BCE, asymmetric + adequacy guard | 2.6 % | 96.5 % | 78.0 % | 7.1× |
| 95 % | **REINFORCE probabilities** | **1.3 %** | **99.0 %** | 69.0 % | 8.8× |
| 98 % | BCE confidence (RACLearn) | 99 % | 86.0 % | 71.5 % | 11.7× |
| 98 % | BCE, asymmetric + adequacy guard | 33 % | 94.5 % | 70.5 % | 10.6× |
| 98 % | **REINFORCE probabilities** | **2.4 %** | **98.0 %** | 66.0 % | 10.5× |

(MILP itself: 98.0 % without reserve shortfall on these hours. Full table with per-unit calibration:
`results/uc1/uc_fixing_results.md`, `uc_fixing_calib_results.md`.)

* **Up to 90 % fixed, RACLearn's rule is right**: 2.9–4.2× at 0.01–0.1 % mean gap; nothing tested beats it.
* **Beyond 90 %, rank by the cost-aware model.** The REINFORCE fine-tuned Model 1 (U4) gives a far better
  ranking for aggressive fixing: 1.3 % instead of 18 % mean gap at 95 %, 2.4 % instead of 99 % at 98 %
  (10× speed-up, no more shortfall than the MILP). Below 90 % it is worse, because it over-commits slightly.
* **Diagnosis of the failures** (95 %, BCE): two kinds of error. Wrong ON fixes of the same few expensive
  combustion turbines (units 0, 54/55, 62/63) cost 1–18 % each and are frequent; wrong OFF fixes of a 355 MW
  combined-cycle unit cause reserve shortfall (57–139 %) and are rare. A wrong ON fix is *not* cheap at the
  one-hour scale, so penalising OFF errors (asymmetric ranking) trades one error for the other: it fixes the
  tail at 95–98 % and hurts at 90 %.
* **Adequacy guard** (new; unfix OFF-fixed units in merit order until the units not fixed off cover net load
  + reserve + 5 %, no solver): cuts the 95 % mean gap from 18 % to 4.2 %, but cannot see network congestion.
* **Per-unit calibration from label frequencies** (error rate per unit, direction and confidence bin on the
  validation set, in the spirit of Fritz et al.'s generator-specific thresholds) is **worse** (95 %: 70 % mean
  gap): label disagreements are dominated by harmless ties (identical units, U1) while the costly errors are
  rare. Calibration has to be cost-aware, as in Fritz et al., not frequency-based.

### U6. Model 2: the physics decoder, and why the framework's (PG, VA) output cannot judge a commitment

**Accuracy** (1,200 test (demand, commitment) pairs: the MILP commitment plus two perturbed ones with
1–3 units flipped; 60 epochs on 4,500 pairs):

| Model 2 (B1) | fully feasible | KCL error, worst bus | line overload, worst line | gen-limit violation, worst unit | cost error |
|---|---|---|---|---|---|
| direct (PG, VA) regression (framework) | **0 %** | 1,013 MW | 280 MW | 26 MW | 7.5 % |
| physics decoder | 69.7 % | 6.7 MW | 14.9 MW | 0 | 7.7 % |
| physics decoder + overload penalty | **74.1 %** | 6.7 MW | 9.2 MW | 0 | 12.8 % |

*Physics decoder*: Model 2 predicts each committed unit's position in [pmin, pmax]; a closed-form layer
restores power balance (raise units toward pmax, or curtail renewables then lower units toward pmin) and
VA follows from the DC power flow. Units that are off produce exactly zero, limits always hold, and KCL is
exact up to shedding (the 6.7 MW is imbalance in pairs whose commitment cannot serve the load).
Violation columns are the per-pair worst value, averaged over pairs.
Direct regression produces no physically valid dispatch at all: summed over buses, its injections miss the
load by up to 1 GW.

Neither variant estimates cost well enough for the decision it is meant to inform: a 7.5–13 % cost error
against cost differences of 0.01–1 % between good commitments.

**Model 2 as a screener** (pick 3 of the 20 kNN candidates by Model 2's cost, check those with the LP; 400
test hours). The first run priced only the predicted PG, which ranks under-committed candidates as cheapest
(Spearman −0.27 to −0.30 with the true cost; worse than random). Pricing what the dispatch LP would pay for
the imbalance (VOLL) and for the reserve headroom shortfall fixes the sign:

| screener (B1) | no shed / shortfall | median gap | Spearman with true cost |
|---|---|---|---|
| LP-check all 20 candidates (20 LPs) | **97.5 %** | **0.000 %** | – |
| Model 2 cost of PG only, top-3 → LP (any variant) | 22–25 % | 1,600–1,800 % | −0.27 to −0.30 |
| direct (PG, VA) Model 2 + implied slack, top-3 → LP | 80.3 % | 0.23 % | 0.56 |
| physics Model 2 + implied slack, top-3 → LP | 81.8 % | 0.36 % | **0.73** |
| physics + overload penalty + implied slack, top-3 → LP | 89.3 % | 5.0 % | 0.54 |
| random 3 → LP | 83.0 % | 2.06 % | – |

* The framework's Model 2 output (PG, VA) does not contain what makes a commitment good or bad here — whether
  load and reserve can be covered — unless the shortfall is priced explicitly.
* Even then, ranking by Model 2 barely beats random on shedding: with the LP at ~30 ms, checking all
  candidates with the exact LP is the better screener.

**The dashed arrow** (Model 1 fine-tuned through a frozen Model 2, loss = Model 2's predicted cost incl. VOLL
on imbalance + overload penalty; relaxed u or straight-through rounding; 400 test hours, same initial Model 1):

| Model 1 critic (B1) | no shed / shortfall | median gap | mean gap, served hours | units on |
|---|---|---|---|---|
| none: imitation (BCE) reference | 77.5 % | 0.000 % | 1.27 % | 14.9 |
| frozen physics Model 2, relaxed u | 71.0 % | 0.47 % | 6.4 % | 14.6 |
| frozen physics Model 2, straight-through | 98.3 % | 24 % | 97 % | 16.4 |
| frozen direct (PG, VA) regression (framework), relaxed u | 76.0 % | 25 % | 24 % | 17.9 |
| frozen direct (PG, VA) regression (framework), straight-through | 75.5 % | 155 % | 137 % | 26.5 |
| **exact LP sensitivities** (envelope theorem at relaxed u, 300 steps, 2 min) | 77.8 % | 0.004 % | **0.58 %** | 14.6 |
| *REINFORCE with the exact dispatch LP (U4, 1,000 hours)* | *93.9 %* | *0.051 %* | *5.6 %* | |

* **Every learned critic makes Model 1 worse than no critic.** The framework's direct (PG, VA) Model 2 is the
  worst: it under-prices commitment, so Model 1 learns to switch on 18–27 units instead of 15 (25–155 % median
  gap). The physics Model 2 either barely moves (relaxed) or, with straight-through rounding, buys
  feasibility by over-committing (24 % median gap).
* The exact LP sensitivities (duals of p ≤ pmax·u, p ≥ pmin·u and the no-load term; verified against finite
  differences, `tests/test_uc.py`) are the right gradient: they halve the cost of served hours, but at a
  relaxed commitment they cannot see that a fractional unit is not a unit, so shedding does not improve.
* Only the exact LP *evaluated at sampled integer commitments* (REINFORCE, U4) fixes shedding.
  The same ordering appeared in the OTS side study.

### U7. The framework's test metric (MSE to the MILP solution) mis-ranks methods

* On the 300 test hours that have an **exactly tied** alternative optimum (U1), the two optimal commitments
  differ in 2.2 units on average, and the alternative scores PG MSE 0.0132 p.u.² (RMS 11.5 MW per unit; worst
  hour 0.35 p.u.², 59 MW) and VA MSE 3.4·10⁻⁴ rad² against the MILP solution — with zero cost difference.
* Across methods, on 300 ordinary test hours:

| method (B1) | PG MSE (p.u.²) | median gap | no shed / shortfall |
|---|---|---|---|
| rounded LP relaxation + adequacy repair | 0.034 | 0.000 % | 86.3 % |
| keep last hour's units (persistence) | **0.060** | 1.66 % | 67.0 % |
| persistence + adequacy repair | 0.077 | 1.51 % | 84.0 % |
| merit-order priority list | **0.093** | **0.57 %** | **74.7 %** |

  MSE prefers persistence to the merit-order list although the latter is cheaper and sheds less, and an
  exactly optimal answer scores 40 % of the MSE of a decent heuristic. Within one method, MSE and cost gap
  are only moderately rank-correlated (Spearman 0.58–0.66; 0.05–0.34 on the tied hours).
* **Recommendation**: test with the cost gap of the exact dispatch LP plus the share of hours without
  shedding / reserve shortfall (and violations for Model 2), not with MSE to one MILP solution.


### U8. Model 1 without MILP labels

The framework needs a MILP solve per training example. Train instead on labels that need **one LP**:
the rounded LP relaxation after adequacy repair (agrees with the MILP on 98.0 % of unit-hours; 0.027 s
instead of 0.49 s per label on B1), then fine-tune with the dispatch LP as critic (U4; the RLOO baseline
cancels the per-instance reference cost, so no MILP cost is needed either). Same GNN and features, same
1,000 test hours:

| Model 1 (B1) | MILP solves in training | one-shot: no shed / median gap | + adequacy repair | screening: no shed (LPs) |
|---|---|---|---|---|
| BCE on MILP labels | 4,000 | 74.0 % / 0.012 % | 87.7 % | 94.6 % (5.2) |
| BCE on repaired LP-relaxation labels | **0** | 79.5 % / 0.021 % | 88.0 % | 88.2 % (1.8) |
| MILP labels + REINFORCE | 4,000 | 93.9 % / 0.051 % | 96.6 % | 97.4 % (1.7) |
| **LP-relaxation labels + REINFORCE** | **0** | **94.1 % / 0.070 %** | **97.4 %** | 95.5 % (1.1) |
| REINFORCE from scratch (400 steps) | 0 | 94.4 % / 0.66 % | 97.0 % | 98.6 % (3.5), median 0.29 % |

On B1 the MILP labels buy almost nothing once the critic fine-tuning is applied. The single-period
relaxation is tight, though; B2 (below) tests whether this survives a weaker relaxation.

## 4. Findings on B2 (12-hour look-ahead UC)

12 hourly periods with min up/down times, ramping and start-up costs across hours; 500 training, 60
validation, 120 test instances. The MILP (60 s limit, 0.1 % gap) proves optimality on 82.5 % of test
instances (mean 28.6 s); the reference is therefore a strong incumbent, not always the optimum, and
methods can beat it. The MILP's own schedules leave a reserve shortfall in some hour of 7.5 % of instances,
so **92.5 % is the ceiling** for "no shed / shortfall". Min up/down repair (RACLearn's DP) is applied to every
predicted schedule.

### V1. Labels: identical units make most multi-hour labels arbitrary

**62.5 %** of B2 labels change when identical units' schedules are put in canonical order (B1: 21.4 %).

### V2. Model 1 on B2 (120 test instances)

| Model 1 (B2) | LPs | no shed / shortfall | median gap | mean gap, served | mean gap |
|---|---|---|---|---|---|
| MILP (reference) | MILP 29 s | 92.5 % | 0 | 0 | 0 |
| keep last hour's units | 1 | 15.8 % | 5,029 % | 24 % | 8,343 % |
| merit-order list + repairs | 1 | 19.2 % | 307 % | 2.0 % | 2,177 % |
| rounded LP relaxation + repairs | 1 + relaxation | 36.7 % | 6.0 % | 0.54 % | 126 % |
| kNN over training schedules, k = 20 | 20 | 79.2 % | 7.2 % | 9.5 % | 22 % |
| MLP, BCE on MILP labels, top-1 | 1 | 15.8 % | 230 % | 1.9 % | 1,106 % |
| GNN, BCE on MILP labels, top-1 (framework) | 1 | 10.0 % | 248 % | 2.6 % | 1,073 % |
| GNN + symmetry + canonical labels, top-1 | 1 | 8.3 % | 332 % | 0.90 % | 1,168 % |
| GNN + symmetry + LP-relaxation features, top-1 | 1 | 16.7 % | 57 % | 0.62 % | 447 % |
| … + adequacy repair | 1 | 43.3 % | 5.3 % | 0.79 % | 220 % |
| … + candidate screening | 14.6 | 62.5 % | **1.65 %** | 1.9 % | 58 % |
| **… + REINFORCE (exact LP critic), top-1** | **1** | **80.8 %** | 5.3 % | 9.1 % | 15 % |
| … + REINFORCE + adequacy repair | 1 | **86.7 %** | 4.5 % | 8.7 % | 13 % |
| … + REINFORCE + candidate screening | 14.8 | **87.5 %** | 2.1 % | 4.7 % | **5.8 %** |

* **Heuristics collapse on the multi-hour problem** (relax-and-round + repair: 86.5 % → 36.7 % of instances
  served), and the framework's one-shot GNN serves only 10 %: on B2 imitation-trained Model 1 is unusable
  without a correction step. The GNN is again no better than an MLP, and symmetry handling again does not help.
* **REINFORCE with the exact LP critic transfers**: 16.7 % → 80.8 % served with one LP (120 steps, 12.5 min),
  better than kNN-20 with 20 LPs; with screening 87.5 % (ceiling 92.5 %) at a 5.8 % mean gap.
* **But it buys feasibility by over-committing**: on instances it serves, it costs 4.7–9.1 % more than the MILP
  (BCE-based variants: 0.6–1.9 %). The single-hour picture (cost-aware fine-tuning nearly free) does not carry
  over; a learned Model 1 alone does not reach MILP quality on B2.

### V3. Model 1 + MILP on B2 (confidence fixing, 60 test instances)

| fixed (BCE confidence) | mean gap | no shed / shortfall | matches or beats MILP | time | speed-up |
|---|---|---|---|---|---|
| full MILP (60 s limit) | 0 | 88.3 % | 100 % | 28.3 s | 1× |
| 50 % | **−0.05 %** | 88.3 % | 85 % | 20.7 s | 1.4× |
| 80 % | **−0.01 %** | 88.3 % | 72 % | 14.8 s | 1.9× |
| 90 % | 1.4 % | 86.7 % | 55 % | 9.9 s | 2.8× |
| 95 % | 38.5 % | 80.0 % | 35 % | 4.3 s | 6.5× |

On B2 the hybrid is where learning pays: fixing half to 80 % of the decisions gives schedules *cheaper* than
the time-limited full MILP (the smaller MILP gets closer to optimality within the limit) at 1.4–1.9× speed.
Errors bite earlier than on B1 (90 % fixed already costs 1.4 %).

**Which decisions to fix on B2** (same 60 instances; full MILP 28.3 s, serves 88.3 %):

| fixed | ranking | mean gap | median gap | no shed / shortfall | time | speed-up |
|---|---|---|---|---|---|---|
| 90 % | BCE confidence (RACLearn) | 1.38 % | 0.000 % | 86.7 % | 9.9 s | 2.9× |
| 90 % | **BCE, OFF errors × 10 + adequacy guard** | **0.41 %** | 0.055 % | **96.7 %** | 6.0 s | 4.7× |
| 90 % | REINFORCE probabilities | 3.15 % | 0.47 % | 91.7 % | 2.6 s | 11× |
| 95 % | BCE confidence (RACLearn) | 38.5 % | 0.087 % | 80.0 % | 4.1 s | 6.9× |
| 95 % | BCE, OFF errors × 10 + adequacy guard | 14.0 % | 0.27 % | 90.0 % | 2.7 s | 10.6× |
| 95 % | **REINFORCE probabilities** | **5.5 %** | 1.48 % | **91.7 %** | **1.0 s** | **27×** |

* Up to 80 % fixed, RACLearn's rule is free on B2 (1.9×).
* **At 90 %, the asymmetric ranking with the adequacy guard is the best trade-off** — 0.41 % mean gap at 4.7×,
  and it serves more instances than the MILP itself (96.7 % vs 88.3 %: the MILP accepts priced reserve
  shortfall; keeping extra units fixed on avoids it). On B1 the same rule hurt at 90 % (single-hour wrong ON
  fixes of peakers are costly; over 12 hours under-commitment dominates).
* At 95 %, ranking by the cost-aware (REINFORCE) model is again the most robust (5.5 % vs 38.5 %, 27×), but
  no ranking is near-optimal there.
* The fallback for fixings that conflict with min up/down times (re-solve without fixings) never triggered.

### V4. Model 1 without MILP labels on B2

Labels from the rounded LP relaxation after adequacy and min up/down repair: 0.39 s per label instead of
34 s for the MILP (**87× cheaper**; 500 labels: 3 min instead of 4.7 h), agreeing with the MILP on 98.4 % of
unit-hours. Same GNN, features and training budget; 120 test instances:

| Model 1 (B2) | MILP solves | one-shot: no shed / median gap | + adequacy repair | screening: no shed / median gap (LPs) |
|---|---|---|---|---|
| BCE on MILP labels | 500 | 16.7 % / 57 % | 43.3 % / 5.3 % | 62.5 % / 1.65 % (14.6) |
| BCE on repaired LP-relaxation labels | **0** | 20.8 % / 19 % | 39.2 % / 7.4 % | 53.3 % / 1.59 % (13.8) |
| MILP labels + REINFORCE | 500 | 80.8 % / 5.3 % | 86.7 % / 4.5 % | 87.5 % / 2.06 % (14.8) |
| **LP-relaxation labels + REINFORCE** | **0** | **84.2 % / 4.4 %** | **86.7 % / 4.1 %** | **88.3 % / 1.87 %** (14.4) |
| REINFORCE from scratch (200 steps) | 0 | 83.3 % / 67 % | 90.0 % / 64 % | 87.5 % / 44 % (14.6) |

* **The MILP labels are not needed**: relaxation imitation + LP-critic fine-tuning matches or beats the
  MILP-labelled pipeline on every decoder, on both benchmarks — and removes the framework's most expensive step.
* **The imitation stage is needed**: REINFORCE from scratch learns to avoid shedding by committing almost
  everything (67 % median gap; unit-hour accuracy 66 %). Imitating the cheap relaxation supplies the structure,
  the LP critic supplies the cost asymmetry.

## 5. Three new frameworks on B2 (parallel study)

Three directions were implemented and tested in parallel on the 12-hour benchmark, each against
re-implementations of the relevant published methods on the same data, model and test instances. Full write-ups:
[`methods/selftrain.md`](methods/selftrain.md), [`methods/fixpolicy.md`](methods/fixpolicy.md),
[`methods/constrained.md`](methods/constrained.md); tables in `results/uc12/{selftrain,fixpolicy,constrained}_results.md`.
Single seeds; results marked *post hoc* were designed after seeing test numbers and need confirmation.

### W1. End-to-end mode: a min up/down-aware adequacy repair, and constrained fine-tuning

*Block adequacy repair* (new, no solver): before the LP, add the cheapest units in blocks that respect their
minimum up/down times until capacity covers load + reserve in every hour. The per-hour repair used so far was
partly undone by the later min up/down repair; this one is not. *Constrained fine-tuning*: operating cost is the
objective, shedding + reserve shortfall of the deployed commitment a constraint with a dual-ascent multiplier,
per-hour credit assignment, an LP-sensitivity control variate and optionally a KL anchor to the imitation policy.

| B2 test, 120 instances, one LP unless noted | MILP labels | served | median gap | mean gap, served | mean gap |
|---|---|---|---|---|---|
| MILP (reference) | – | 92.5 % | 0 | 0 | 0 |
| kNN-20 (20 LPs) | yes | 79.2 % | 7.19 % | 9.47 % | 21.7 % |
| label-free + REINFORCE (V4) | no | 84.2 % | 4.45 % | 9.22 % | 14.9 % |
| **… + block repair** | no | **94.2 %** | 3.92 % | 8.91 % | 11.0 % |
| … threshold 0.9 (chosen on val) + block repair | no | 90.8 % | **1.18 %** (best of 3 seeds; §6) | 2.55 % | 14.1 % |
| He et al.-style: behaviour cloning + RL repair (re-implemented) | yes | 79.2 % | 1.34 % | 1.99 % | 18.2 % |
| He et al.-style + block repair | yes | 90.8 % | 1.29 % | 2.36 % | 7.06 % |
| label-free imitation only + block repair | no | 84.2 % | 0.93 % | 1.99 % | 33.9 % |
| Lagrangian fine-tuning only | no | 94.2 % | 8.65 % | 13.4 % | 14.7 % |
| Lagrangian + KL anchor + block repair (*post hoc*) | no | 93.3 % | 1.73 % | 3.09 % | **5.96 %** |
| variant selected by the pre-declared val rule | no | 84.2 % | 1.30 % | 2.04 % | 56.2 % |

* **The block repair is the main gain**: it lifts every model, and with plain REINFORCE serves 94.2 % of
  instances with one LP — above the MILP's own 92.5 % (the MILP accepts some priced reserve shortfall).
* Constrained fine-tuning alone overshoots (more units than plain REINFORCE); the multiplier sat at its cap,
  so it acted as a fixed penalty. With a KL anchor and the block repair it gives the lowest mean gap of any
  one-LP method (5.96 %; on the 110 instances both serve, 3.10 % vs 8.79 % for REINFORCE + repair), but the
  validation rule did not pick it.
* Against the He et al.-style baseline neither dominates: ours serves more and has a lower mean gap without
  MILP labels; theirs is cheaper on the instances it serves.
* About 85 % of the units REINFORCE adds are ones the MILP keeps off; part is hedging against its own sampling
  noise (a higher decision threshold with the block repair recovers most of the cost).

### W2. Self-training with the solver as teacher

Round 0 is the label-free model. Each round fixes the 80 % most confident decisions that agree with the current
label (so the label stays feasible), solves a 15 s reduced MILP and keeps the schedule only if the dispatch LP
prices it cheaper; three rounds relabel the 500 training instances. No full MILP is solved.

* **Labels**: 1.7 core-hours instead of 4.7 for full-MILP labels (2.7× cheaper); served share of the labels
  35 % → 91 %, median gap to the (unused) training MILP 12 % → 0.08 %.
* **As a predictor** (120 test instances): self-trained BCE 25.8 % served one-shot (MILP-label BCE 16.7 %);
  self-trained + REINFORCE 87.5 % served, 3.9 % median gap (label-free + REINFORCE 84.2 %, 4.4 %) — a small gain
  that does not remove the over-commitment (+8.5 % cost on served instances).
* **As a fixing ranker** (first 40 test instances, full MILP back to back, 31.7 s): at 95 % fixed the
  self-trained model gives **1.52 % mean gap at 11.9×** (on 60 instances and the fresh set about 2 % at 8–10×; §6 X1), against 20.6 % (10.4×) for the MILP-label BCE model
  (RACLearn-style) and 4.5–5.5 % (30–51×) for REINFORCE probabilities. At 80 % the MILP-label model is better
  (0.006 % vs 0.45 %).
* Adding solver-polished labels to REINFORCE's comparison group cuts the extra units by 80 % (2.2–2.6 % cost on
  served instances) but serves fewer instances: a different point on the cost–coverage curve, not a win on both.

### W3. A fixing rule learned from LP-priced errors

Each (unit, hour) decision is ranked by its expected cost of a wrong fix: a corrected error probability of the
BCE model times a learned cost of that error. The cost model is trained on 26,700 dispatch LPs (no MILPs), each
pricing the MILP optimum with one wrongly predicted decision forced in, using predictions from 4-fold
cross-fitted models so the errors resemble deployment errors.

| B2, 60 test instances (full MILP back to back, 39.6 s, serves 90 %) | 90 % fixed: mean / median gap, served, speed-up | 95 % fixed |
|---|---|---|
| RACLearn (confidence), re-implemented | 1.38 % / 0.000 %, 86.7 %, 2.2× | 38.5 % / 0.087 %, 80.0 %, 5.2× |
| Learning to Fix (generator thresholds), re-implemented | 7.81 % / 0.339 %, 86.7 %, 2.5× | 49.4 % / 1.51 %, 63.3 %, 4.0× |
| asymmetric + adequacy guard (V3) | 0.41 % / 0.055 %, 96.7 %, 3.6× | 14.0 % / 0.268 %, 90.0 %, 7.4× |
| REINFORCE probabilities (V3) | 3.15 % / 0.474 %, 91.7 %, 7.8× | **5.5 %** / 1.48 %, 91.7 %, **17.7×** |
| **learned error-cost ranking** + adequacy guard | 0.47 % / **0.002 %**, 90.0 %, 3.7× | 13.9 % / **0.052 %**, 88.3 %, 7.8× |

* Against the published rules it is clearly better at 90–95 % fixed (best mean gap at ≥ 3× speed-up: 0.47 % vs
  38.5 % for RACLearn and 49.4 % for Learning to Fix). It ties our asymmetric rule on the mean with a far lower
  median; REINFORCE ranking remains the only rule beyond ~8×. The 95 % mean is set by one instance (11 of 13.9 points).
* *Post hoc*: an **LP-relaxation guard** (release fixings whenever the relaxed reduced problem already pays
  shedding / shortfall penalties; no parameters; checked on val before test) removes the catastrophic cases:
  learned ranking at a 95 % target 0.40 % mean, 0.05 % median, 96.7 % served at 5.4×; RACLearn at 95 % 0.19 % at
  3.2× (0.46 % on the fresh test set; §6 X1). It releases many fixings (actual share 92 % at a 95 % target).
* The 80 % row: every rule is under 1 %; RACLearn is best (−0.01 %, 1.7×).

### W4. Against the published methods, on the same benchmark

* **Partial fixing (RACLearn, Learning to Fix)**: at 90–95 % fixed, all three new rankings — learned error cost,
  self-trained probabilities, REINFORCE probabilities — beat both re-implementations by large margins in mean
  gap at similar or higher speed-up. Learning to Fix was re-implemented from its abstract only.
* **Prediction + RL repair (He et al. 2026)**: the block-repaired, label-free models serve more instances with a
  lower mean gap and no MILP labels; the He et al.-style model is cheaper on the instances it serves.
* **kNN (Xavier et al.; Pineda & Morales)**: beaten by every block-repaired one-LP method.
* **Not reached**: Learning to Fix's published < 0.5 % at > 20× (EPRI competition systems). On B2 nothing is below
  0.5 % beyond ~5.4×, and the MILP here takes only ~30 s, which caps the attainable speed-up.

## 6. Confirmation, combined pipelines, Learning to Fix and a 24-hour benchmark

A second parallel study. Every decision rule was fixed on validation before any test set was read; final numbers
use a **fresh B2 test set** (120 new instances, seed 23; `test_fresh`) and 3 training seeds, unless noted.
Write-ups: [`methods/combo.md`](methods/combo.md), [`methods/ltf.md`](methods/ltf.md), [`methods/b3.md`](methods/b3.md).

### X1. Which earlier claims held up (3 seeds, fresh test set)

| claim from §5 | result |
|---|---|
| block repair: ~94 % served with one LP | **confirmed**: 95.0 % [92.5, 96.7] |
| Lagrangian + KL anchor + block repair (post hoc) | **confirmed, seed-stable**: 92.5 % served, 3.25 % gap on served |
| learned error-cost ranking at 90 % fixed: 0.47 % | **confirmed and better**: 0.28 % [0.27, 0.29] at 3.2× (0.47 % was the worst of three seeds) |
| LP-relaxation guard (post hoc) | **confirmed**: 0.39–0.46 %, seed-stable |
| REINFORCE ranking at 95 % fixed | **confirmed**: 2.93 % at 10.6× (≤ the claimed 5.5 %) |
| self-trained ranking at 95 %: 1.52 % at 11.9× | **weaker**: about 2 % at 8–10× on 60 instances (the 40-instance subset was favourable) |
| RACLearn + LP guard at 95 %: 0.19 % | **weaker**: 0.46 % on the fresh set |
| REINFORCE + threshold 0.9 + block repair: 1.18 % median | **weaker**: 1.94 % [1.54, 2.56] (1.18 % was the best seed) |
| one LP serves more instances than the MILP | **only on the original set** (the fresh set's MILP serves 98.3 %) |

### X2. Combined pipelines (fresh test set, 3 seeds)

| end-to-end (no MILP) | LPs | served | median gap | mean gap, served | mean gap |
|---|---|---|---|---|---|
| MILP (reference) | – | 98.3 % | 0 | 0 | 0 |
| REINFORCE + block repair | 1 | 95.0 % | 3.45 % | 6.15 % | 7.44 % |
| combined: Lagrangian + KL, threshold 0.6, block repair | 1 | 92.2 % | 1.69 % | 2.68 % | 8.31 % |
| **combined + threshold screening** | 5.8 | **93.3 %** | **1.04 %** | **1.64 %** | **3.31 %** |

(On the original test set the screening pipeline gives 95.8 % served, 0.86 % median, 3.57 % mean gap.)

| solver acceleration (first 60 fresh instances, full MILP back to back) | mean gap | speed-up |
|---|---|---|
| RACLearn 90 % / 95 % | 12.4 % / 49.2 % | 2.8× / 8.1× |
| error-cost ranking + adequacy guard, 90 % | **0.28 %** | 3.2× |
| error-cost + adequacy + LP guard, 95 % | 0.46 % | 4.6× |
| **combined**: self-trained probabilities + error cost + both guards, 95 % / 98 % | **0.59 % / 0.72 %** | **5.3× / 7.0×** |
| REINFORCE ranking, 95 % | 2.93 % | 10.6× |

On the original test set the combined rule at 98 % gives 0.86 % at 10.4× — the only sub-1 % rule at ≥ 10× there.
The combined fixing pipeline does not beat its parts at lower speed-ups.

### X3. Learning to Fix, implemented from the paper

Implemented as published ([`methods/ltfx.md`](methods/ltfx.md)): kNN (k = 50, inverse-distance probabilities,
Table II features) and generator-specific grey zones tuned by the paper's logic-based Benders decomposition, so
that **every** validation instance, with all its fixings applied jointly, keeps a reduced-UC solution within ε of
C*. Tuned on 180 validation instances (60 + 120 newly generated; the paper used ~524), 4.0 core-hours for 8 runs.
Deviation: relaxation MILPs capped at 6 s, so cuts are less minimal and thresholds somewhat more conservative.
The same tuning was applied to our GNN probabilities (as the paper does with CatBoost).

Paper metrics on the first 60 fresh 12-hour test instances (gap to the full MILP's dual bound, mean of per-instance
speed-ups, statistics over feasible instances; full MILP 0.12 % / 19.6 s):

| rule | feasible | gap mean / max | speed-up mean / max | fixed |
|---|---|---|---|---|
| kNN, constant [0.01, 0.99] | 100 % | 0.94 % / 48 % | 2.3× / 28× | 71 % |
| kNN, Learning to Fix ε = 1 % | 100 % | 0.40 % / 3.6 % | 4.7× / 79× | 68 % |
| kNN, Learning to Fix ε = 5 % | 100 % | 1.12 % / 5.9 % | 8.3× / 98× | 76 % |
| **our BCE GNN + Learning to Fix ε = 1 %** | 98.3 % | **0.28 %** / 2.4 % | **5.2×** / 28× | 84 % |
| our self-trained GNN + Learning to Fix ε = 10 % | 96.7 % | 0.85 % / 5.1 % | 13.5× / 62× | 85 % |
| ours: error-cost + adequacy guard, 90 % | 100 % | 0.42 % / 4.3 % | 4.2× / 17× | 90 % |
| ours: combined pipeline, 98 % | 100 % | 0.82 % / 6.3 % | 13.6× / 65× | 92 % |

* **The paper's quality reproduces, its speed-up does not**: ε = 1 % gives 0.40 % (paper 0.48 %) but 4.7× (paper
  20.8×), fixing 68 % of decisions (paper 79 %). On a 12-hour network-constrained UC whose MILP takes ~20 s, 20× is
  out of reach for every method tested.
* **The joint check removes the failure of our snippet-based reconstruction** (§6 earlier version): on the same
  test instances 0 instances above 10 % gap instead of 10, and 0.27 % instead of 37.7 % mean gap (our convention).
* **Against our rules it is level, not beaten.** Learning to Fix on our BCE GNN is the most accurate rule near 5×
  (0.28 % vs our 0.42 % at 4.2×; in our convention the paired difference is not significant), and at ~13.5× it is
  level with our combined pipeline (0.85 % vs 0.82 %). Our rules reach higher speed-ups only with larger gaps
  (REINFORCE ranking: 3.4 % at 33×).
* **What our work adds to it**: better probabilities. The same calibration on our GNN (LP-relaxation features) gives
  a lower gap than on the paper's kNN at every tolerance tested, and is also faster at ε = 10 % and 5 % (at 1 % the
  BCE GNN is better on both); the self-trained model needs no MILP labels.

### X4. A 24-hour benchmark (uc24)

T = 24, network on, 0.1 % gap, 300 s limit (highspy with incumbent logging). Test MILPs: 154 s mean, 30 % hit the
limit. **Training used no full MILP** (0.8 core-hours vs ~13 for MILP labels). 40 test instances, one seed:

| | mean gap | served | speed-up | time-to-quality speed-up |
|---|---|---|---|---|
| RACLearn-style, 80 % fixed | 0.33 % | 92.5 % | 3.4× | 1.3× |
| val-selected guarded rules, 80–95 % target | 0.52–0.57 % | 92.5–95 % | 4.7–4.8× | 1.4–2.0× |
| RACLearn-style, 95 %, no guard | 138 % | 40 % | 14.4× | 2.2× |
| REINFORCE ranking, 95 % | 8.8 % | 92.5 % | 26× | 5.3× |
| end-to-end, one LP + screening | 4.3 % median | 92.5 % (MILP 97.5 %) | ~120× | – |

* The guards make aggressive fixing usable (95 %: 138 % → 0.52 % mean gap).
* **The full MILP finds a schedule within 0.5 % of its final one after a median of 28 s**; the rest of its time
  proves the bound. Against *time to the same quality*, fixing is only ~2× faster. Speed-ups against the full
  solve time — the usual way they are reported — overstate the gain for operations that accept a 0.5 % gap.

### X5. All results under the paper's metrics

Re-scored from the per-instance records ([`methods/papereval.md`](methods/papereval.md)): gap to the full MILP's dual
bound, mean of per-instance speed-ups, feasible instances only, runtimes including inference and guard LPs (the
earlier 12-hour runs had not timed the LP relaxation used as model input, 0.26 s; adding it halves the fastest rule's
speed-up and barely moves the others).

| benchmark / method | gap mean | speed-up mean | fixed |
|---|---|---|---|
| paper, Learning to Fix ε = 1 % (its system) | 0.48 % | 20.8× | 79 % |
| 12 h fresh: error-cost + both guards, 95 % | 0.59 % | 8.0× | 93 % |
| 12 h fresh: combined pipeline, 98 % | 0.84 % | 12.6× | 92 % |
| 12 h fresh: REINFORCE ranking, 95 % | 3.37 % | 32.5× | 95 % |
| 12 h original: error-cost + both guards | 0.63 % | 10.4× | 92 % |
| 24 h: guarded rule, 95 % target | 0.67 % | 11.6× | 85 % |
| 12 h fresh: paper's cost-ranked kNN (k = 50) | 12.0 % | 5.4× | 100 % |

The paper's own baselines are 1.7–4.6× worse in gap and mostly 1.3–2.6× slower on our 12-hour benchmark than on
its system (same fixed shares): our benchmark is harder for fixing and its fast MILP caps speed-ups. Relative to those
baselines our rules improve at least as much as the paper's method does. Not comparable: system (Irish copper plate,
72 h vs RTS-GMLC DC network, 12/24 h), solver (Gurobi, 8 CPUs vs single-thread HiGHS on a shared machine), MILP
tolerance, and sample sizes (~525 validation / 525 test vs our 60–180 / 40–120).

## 7. Recommendations for the framework, box by box

| box | change | evidence |
|---|---|---|
| inputs (PD, QD) | add the previous on/off status (start-up costs, min up/down depend on it), renewable availability, reserve requirement; add one LP-relaxation solve (fractional u, prices, loadings) as features. QD has no role in a DC model. | U3, V2 |
| Model 1 (GNN) | pre-train by imitation, then **fine-tune on cost with the exact dispatch LP as critic** (REINFORCE with a leave-one-out baseline; 1–15 min). A GNN is not better than an MLP on a fixed network; keep it only if topology or system size changes. Symmetry features / canonical labels did not help. | U3, U4, V2 |
| MILP labels | **optional**: imitate the repaired LP relaxation instead (18× cheaper on B1, 87× on B2) — same results after fine-tuning; self-training with reduced MILPs gives near-MILP labels at 37 % of the cost. Do not train from scratch. | U8, V4, W2 |
| decoding | min up/down repair (DP) and the **block adequacy repair** (min up/down-aware, no solver) on every prediction; check 2–15 candidates with the LP when time allows. | U3, U4, V2, W1 |
| LP solver | keep it as the last step: always feasible with priced slacks, 30 ms (B1), and the source of the training signal. | all |
| with a MILP | fix decisions RACLearn-style and solve the reduced MILP: ≤ 90 % fixed on B1 (4×, ≤ 0.1 %), ≤ 80 % on B2 (1.9×, no loss). On B2 at 90 %, rank by the learned error cost or penalise OFF fixes, with the adequacy guard (≈ 0.4 %, 3.6–4.7×); at 95 %, rank by self-trained or REINFORCE probabilities (1.5 % at 12×; 5.5 % at 18×). Test the LP-relaxation guard further. | U5, V3, W2, W3 |
| Model 2 | if a fast dispatch estimate is needed, use the physics decoder (unit positions → closed-form balance → VA from DC power flow) and price the implied shedding / reserve shortfall. Do not use it as a screener when the exact LP is affordable. | U6 |
| dashed arrow | **replace the learned critic by the exact LP** (sampled commitments scored by the LP; or the LP's sensitivities). A learned Model 2 critic made Model 1 worse in every variant. | U6 |
| validation / test | report the share of instances without shedding / reserve shortfall (against the MILP's own share) and the median cost gap from the exact LP; the mean gap is dominated by penalty-priced instances. Drop MSE to one MILP solution. | U7 |

## 8. Research backlog (not done here)

1. **Report time-to-quality, not only time-to-proof.** On 24 hours the full MILP finds a 0.5 %-good schedule in
   a median 28 s; fixing then saves ~2×. Reduced MILPs with a looser gap or a time limit, and warm-starting the full
   MILP with the learned schedule, are the natural next tests.
2. **Larger systems** where even finding a good incumbent is slow (thousands of buses, hundreds of units) — the
   regime of RACLearn's 6,708-bus case and the EPRI competition.
3. **Learning to Fix from its full text** (blocked here); the reconstruction's impact measure may differ.
4. **Seed the data side too**: the 3-seed spread in §6 varies training only, not the label pool, the error-cost
   labels or the label-free start.
5. **Contingencies / topology change**: the setting where a GNN should beat an MLP.

## 9. Reproducibility and caveats

* Code and exact runs: `README.md`; result tables in `results/uc1/`, `results/uc12/`; sanity tests
  `tests/test_uc.py` (dispatch LP reproduces the MILP; LP sensitivities match finite differences; the
  physics layer reproduces the LP's flows and cost; repairs return feasible schedules).
* **Single seeds** throughout. Timings were measured while several experiments shared four cores; compare
  ratios within a table, not seconds across tables.
* B2: 17.5 % of test MILPs stopped at the 60 s limit (0.23 % mean MIP gap), so the reference is not always
  optimal; negative gaps are real improvements over the reference.
* An early 4-instance check suggested the merit-order list was near-optimal on B1; on the full test set it
  serves 72.5 % of hours with a 0.85 % median gap (U2).
* The single-hour LP relaxation is tight, which flatters relax-and-round on B1; B2 is the harder test and
  confirms the main conclusions except that a learned Model 1 alone does not reach MILP cost on B2.
