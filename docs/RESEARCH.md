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
  orbit-feature result): one-shot quality dropped slightly. With a 3 % reserve margin, which copy of an
  identical unit is on rarely matters for cost, so the ambiguity costs little once decoding is cost-aware.
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

### U5. RACLearn-style confidence fixing + MILP

Fix the X % most confident (unit, hour) decisions of the LP-feature GNN, solve the reduced MILP
(200 test hours; full MILP 0.51 s):

| fixed | mean gap | matches MILP | time | speed-up |
|---|---|---|---|---|
| 50 % | 0.002 % | 99.5 % | 0.34 s | 1.5× |
| 80 % | 0.014 % | 97.5 % | 0.17 s | 3.0× |
| 90 % | 0.098 % | 91.5 % | 0.095 s | 5.4× |
| 95 % | 18 % (some hours shed load) | 80.0 % | 0.058 s | 8.8× |

RACLearn's 2–4× speed-up at near-optimal quality reproduces (80 % fixed); past ~90 % the remaining free
decisions cannot repair a wrong fixing. On single-period UC the MILP takes only 0.5 s, so the absolute gain
is small; B2 is where fixing matters.

<!-- UC-B1-MODEL2 -->

