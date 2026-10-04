# Learning unit commitment with a two-stage GNN → LP → NN framework: critique, new methods, results

*Working research log. Code: this repository. Prior work: [`literature_uc.md`](literature_uc.md),
[`raclearn_comparison.md`](raclearn_comparison.md). Raw numbers: `results/uc1/`, `results/uc12/`.
A side study that (wrongly) read "switching status" as transmission-line switching is kept in
[`ots/RESEARCH_OTS.md`](ots/RESEARCH_OTS.md).*

**TL;DR (B1, single-hour network-constrained UC on RTS-GMLC; B2 in §4).**

1. *Model 1 → LP* is published (Tang et al. 2023; RACLearn adds confidence fixing). What can be new is below.
2. **A third of the MILP labels are arbitrary** (exactly tied optima, identical units), so BCE and the MSE
   test metric penalise correct answers (U1, U7).
3. **The framework's one-shot pipeline under-commits**: 59–74 % of hours served without shedding or reserve
   shortfall (MILP: 98.7 %). Rounding the LP relaxation and a kNN-over-schedules baseline are stronger (U2, U3).
4. **Fine-tuning Model 1 with the exact dispatch LP as critic** (REINFORCE, 1 min) fixes this: 93.9 % with one
   LP, 97.4 % with 1.7 LPs (U4) — and **without any MILP labels** (imitate the repaired LP relaxation, then
   fine-tune): 94.1 % (U8).
5. **The dashed arrow hurts**: every Model 1 trained through a frozen learned Model 2 is worse than plain
   imitation; the framework's direct (PG, VA) Model 2 drives it to commit 18–27 units instead of 15 (U6).
6. **Model 2 should be a physics decoder** (unit positions → closed-form balance → VA from DC power flow):
   70–74 % fully feasible vs 0 % for direct (PG, VA) regression (U6).
7. **Partial fixing + MILP**: RACLearn's confidence rule is right up to 90 % fixed (4×); beyond that the
   cost-aware model's probabilities are the better ranking (95 % fixed: 1.3 % vs 18 % mean gap) (U5).


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

<!-- UC-B2-FIX -->

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

## 5. Recommendations for the framework, box by box

| box | change | evidence |
|---|---|---|
| inputs (PD, QD) | add the previous on/off status (start-up costs, min up/down depend on it), renewable availability, reserve requirement; add one LP-relaxation solve (fractional u, prices, loadings) as features. QD has no role in a DC model. | U3, V2 |
| Model 1 (GNN) | pre-train by imitation, then **fine-tune on cost with the exact dispatch LP as critic** (REINFORCE with a leave-one-out baseline; 1–15 min). A GNN is not better than an MLP on a fixed network; keep it only if topology or system size changes. Symmetry features / canonical labels did not help. | U3, U4, V2 |
| MILP labels | **optional**: imitate the repaired LP relaxation instead (18× cheaper on B1, 87× on B2) — same results after fine-tuning. Do not train from scratch. | U8, V4 |
| decoding | min up/down repair (DP) + adequacy repair (no solver) on every prediction; check 2–15 candidates (thresholds and samples of Model 1's probabilities) with the LP when time allows. | U3, U4, V2 |
| LP solver | keep it as the last step: always feasible with priced slacks, 30 ms (B1), and the source of the training signal. | all |
| with a MILP | fix decisions RACLearn-style and solve the reduced MILP: ≤ 90 % fixed on B1 (4×, ≤ 0.1 %), ≤ 80 % on B2 (1.9×, no loss — it beats the time-limited full MILP). Beyond that, rank decisions by the cost-aware model. | U5, V3 |
| Model 2 | if a fast dispatch estimate is needed, use the physics decoder (unit positions → closed-form balance → VA from DC power flow) and price the implied shedding / reserve shortfall. Do not use it as a screener when the exact LP is affordable. | U6 |
| dashed arrow | **replace the learned critic by the exact LP** (sampled commitments scored by the LP; or the LP's sensitivities). A learned Model 2 critic made Model 1 worse in every variant. | U6 |
| validation / test | report the share of instances without shedding / reserve shortfall (against the MILP's own share) and the median cost gap from the exact LP; the mean gap is dominated by penalty-priced instances. Drop MSE to one MILP solution. | U7 |

## 6. Research backlog (not done here)

1. **Cost-aware fixing thresholds** (Fritz et al. 2026): calibrate per unit on validation by the cost of a
   wrong fix (measured with reduced MILPs), not by label frequency (which failed, U5).
2. **Reduce over-commitment on B2**: REINFORCE buys feasibility with 4.7–9.1 % extra cost on served instances.
   Candidates: a reward that separates shedding from cost (constrained RL / Lagrangian), more steps,
   per-hour credit assignment with the LP's duals (the exact sensitivities as a control variate).
3. **Seeds and scale**: three seeds for the key B1/B2 claims; a larger system (e.g. a 500-bus or the French
   RTE system used by RACLearn) where the MILP is slow enough for the hybrid to matter.
4. **Contingencies / topology change**: the only setting where the GNN should beat an MLP; test with line
   outages and unit maintenance (the OTS side study found learned + LP checks beat kNN under topology shift).
5. **Ramp-aware adequacy repair** for B2 (the current repair checks capacity per hour only).

## 7. Reproducibility and caveats

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
