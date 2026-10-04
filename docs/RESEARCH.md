# Learning unit commitment with a two-stage GNN → LP → NN framework: critique, new methods, results

*Working research log. Code: this repository. Prior work: [`literature_uc.md`](literature_uc.md),
[`raclearn_comparison.md`](raclearn_comparison.md). Raw numbers: `results/uc1/`, `results/uc12/`.
A side study that (wrongly) read "switching status" as transmission-line switching is kept in
[`ots/RESEARCH_OTS.md`](ots/RESEARCH_OTS.md).*

<!-- UC-TLDR -->

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

<!-- UC-U6-SCREEN -->

<!-- UC-U6-ARROW -->

<!-- UC-U7 -->

