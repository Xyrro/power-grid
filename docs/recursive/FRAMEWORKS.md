# Framework diagrams

*Diagrams of every framework in this study, drawn to match the code. Sources: the original two-stage framework and the
MILP-based pipeline in [`../RESEARCH.md`](../RESEARCH.md) §1, §7–§9; the two recursive arms in
[`PLAN.md`](PLAN.md), [`trm.md`](trm.md) and [`rloop.md`](rloop.md); the CPU-round verdicts in
[`SUMMARY.md`](SUMMARY.md). Code: [`otsl/trm.py`](../../otsl/trm.py) (arm A) and
[`otsl/rloop.py`](../../otsl/rloop.py) (arm B). Every number below is quoted from those documents with its instance
count. Unless stated otherwise the system is RTS-GMLC, 73 thermal units, 73 buses, 120 lines, DC network, 12-hour
windows with min up/down, ramping, start-ups and a 3 % spinning reserve; selection on 360 validation instances and one
pass over the 120 `test_fresh` instances.*

## 1. The original two-stage framework

The starting point is the published two-stage framework: demand goes into Model 1, a GNN that predicts every
generator's on/off decision in one pass and is trained by BCE against full-MILP labels; the dispatch LP then solves for
power and angles given those commitments; Model 2 is a second network that regresses dispatch from demand and
commitments, and its output feeds Model 1's loss through the dashed arrow. Measured on the 12-hour benchmark, that
one-shot pipeline under-commits badly: 8–17 % of instances are served without shedding or reserve shortfall against
the MILP's 92.5 % (§1 and §4 V2). The dashed arrow hurts — every Model 1 trained through a frozen learned Model 2 was
worse than plain imitation, and the framework's direct dispatch-and-angle Model 2 drove it to commit 18–27 units
instead of 15 (§3 U6). What §9 keeps is the skeleton with better inputs and a different critic: the previous on/off
status, renewables, the reserve requirement and one LP-relaxation solve as features, imitation followed by REINFORCE
with the exact dispatch LP as critic, the min up/down and block adequacy repairs on every prediction, and the LP as
the last step. What it discards is the learned Model 2 as critic or screener, and the requirement of full-MILP labels.

```mermaid
graph LR
  IN["Demand, renewables, reserve, previous on-off status - kept and extended"] --> M1
  LPR["One LP relaxation: fractional u, prices, line loadings - added"] --> M1
  LAB["Full-MILP labels, BCE - now optional, polished labels instead"] --> M1
  M1["Model 1, GNN - kept, now a temporal-GNN ensemble"] --> P["Commitment probabilities"]
  P --> REP["Min up-down repair and block adequacy repair - added"]
  REP --> U["On-off schedule u, hours by units"]
  U --> LP["Exact dispatch LP given u - kept as the last step"]
  LP --> OUT["Cost, dispatch, angles"]
  LP -.->|"REINFORCE with the exact LP as critic - added"| M1
  U --> M2["Model 2, NN for dispatch and angles by MSE - dropped as a screener"]
  M2 -.->|"learned critic for Model 1 - dropped"| M1
```

## 2. The current best method: the MILP-based pipeline

The best method in the repository is not end to end: it is the MILP pipeline of §7 Y6 and §8 Z1. A 5-member
temporal-GNN ensemble trained on polished labels produces the probabilities; Learning-to-Fix's joint threshold tuning,
run on validation only, decides which decisions to fix; the adequacy, min up/down and LP-relaxation guards release
fixings that would starve the problem; a reduced MILP at a 0.5 % gap solves what is left; and a free post-solve check
compares the reduced cost with the LP relaxation already solved as model input and, if the bound gap is at least 4 %,
also solves the full MILP at 0.5 % and keeps the cheaper schedule. On the 120 `test_fresh` instances this gives a
0.30 % mean gap, max 2.71 %, 97 % served and a 9.7× mean speed-up with 89 % of decisions fixed, level with the full
MILP at 0.5 % and 2.85× faster; with the bound check re-run in the same process, 0.30 % at 10.1×, the check firing on
1 % of instances. The LP-relaxation guard is what removes the line-outage catastrophe that breaks every tuned rule
without it, 0.24 % against 15.9–26.0 %, and the bound check is what closes the low-load and high-renewables failures,
1.59 % and 1.36 % down to 0.40 % and 0.43 %.

```mermaid
graph TD
  X["Instance: load, renewables, reserve, initial status"] --> E["5-member temporal-GNN ensemble, polished labels"]
  X --> R["LP relaxation, also a model input"]
  E --> T["Joint threshold tuning per generator, Learning-to-Fix style, validation only"]
  T --> G1["Adequacy and min up-down conflict guards"]
  G1 --> G2["LP-relaxation guard: never fix a unit-hour off against a positive relaxed value"]
  R --> G2
  G2 --> RM["Reduced MILP at a 0.5 percent gap, about 89 percent of decisions fixed"]
  RM --> BC{"Bound gap of the reduced cost to the LP relaxation at least 4 percent"}
  R --> BC
  BC -->|"no, 99 percent of test instances"| OUT["0.30 percent mean gap, 97 percent served, 9.7x mean speed-up"]
  BC -->|"yes"| FULL["Also solve the full MILP at 0.5 percent, keep the cheaper schedule"]
  FULL --> OUT
```

## 3. Arm A: the TRM-style recursive model

Arm A asks whether a tiny network applied recursively can replace the MILP. One token per unit-hour carries the unit's
parameters, the hour's totals, its bus's PTDF encoding and the LP-relaxation values; a shared 2-layer network with
unit, time, graph and SwiGLU sub-blocks updates a latent state `z` n = 4 times from `x`, `y` and `z`, then updates the
answer `y` from `y` and `z`; that cycle runs T = 2 times per supervision step, the first without gradients, so one step
costs 10 network calls. Deep supervision keeps each instance for up to N_sup = 8 steps with `y` and `z` carried
forward detached, BCE on every step's answer plus half a BCE on a halting head that predicts "at most 2 wrong
unit-hours". The CPU preset is hidden 64, 2 layers, batch 16, 3,000 training steps, EMA 0.999, 135 k parameters,
73 min on one core. On the 120 test instances the model's own schedule with one LP is 4.96 % mean gap at 87.5 % served,
2.55 % with the LP-guided repair, and the validation-chosen headline pipeline — the model's schedule screened together
with 4 LP roundings — is 1.48 % [1.10, 1.98] at 99.2 % served and 19.1× mean speed-up, against 1.71 % for learning-free
LP rounding and 1.30 % for the same network used one-shot.

```mermaid
graph TD
  X["x: one token per unit-hour, plus bus inputs; LP relaxation included"] --> C1
  subgraph STEP["One supervision step: T = 2 cycles, 10 net calls, only the last cycle carries gradients"]
    C1["Latent recursion, n = 4 times: z from x, y and z"] --> C2["Answer update: y from y and z"]
    C2 --> C1
  end
  C2 --> RD["Linear head on y: one logit per unit-hour"]
  C2 --> HH["Halting head on the mean of y: at most 2 wrong unit-hours"]
  RD --> LOSS["BCE to subgroup count targets plus 0.5 times the halting BCE"]
  LOSS --> NEXT["Carry y and z detached into the next step, up to N_sup = 8"]
  NEXT --> C1
  HH -->|"halt fires, or N_sup reached"| OUTS
  RD --> OUTS["Answer probabilities after k steps"]
  OUTS --> W["Water-filling over identical copies"]
  W --> V["Per-unit Viterbi DP: min up-down times and initial status"]
  V --> AD["Block adequacy repair"]
  AD --> MU["Min up-down repair, a no-op after Viterbi"]
  MU --> LP["Dispatch LP"]
  LP --> CHK{"Shedding or reserve shortfall"}
  CHK -->|"yes"| LGR["LP-guided repair: raise the failing hours reserve, repair and reprice, at most 2 rounds"]
  LGR --> LP
  CHK -->|"no"| SCR["Screen the schedule against 4 LP roundings, keep the cheapest"]
  SCR --> RES["Headline: 1.48 percent mean gap, 99.2 percent served, 19.1x"]
```

## 4. Arm B: recursion with the exact dispatch LP in the loop

Arm B puts the exact dispatch LP inside the recursion. One outer step is one pass of a 166 k-parameter recurrent
network over unit-hour tokens, bus tokens and line tokens — 2 latent updates and one answer update per step, the input
re-injected at every latent update, count heads for the 24 groups of identical units — followed by Viterbi decoding,
the repairs and one exact dispatch LP, whose signals are the next step's input: hourly energy shed, spill and reserve
shortfall, bus prices, line loadings and flow duals, per-unit dispatch, reserve and the exact sensitivity of cost to
each commitment, plus the cost and the certified gap to the LP relaxation. Every step's schedule is therefore a priced
candidate and the cheapest is kept, so more steps never hurt; because the recursion often re-proposes a schedule it has
already priced, 16 steps cost 3.5 distinct LPs on average and 32 steps 4.5. The feedback is what makes the extra steps
pay: with it, going from 1 to 16 steps lowers the validation mean gap by 1.88 pp, without it by 0.68 pp, and at 16
steps feedback is worth 0.78 pp [0.07, 1.54] after imitation and 1.26 pp [0.62, 2.05] after REINFORCE. On the 120 test
instances the selected whole-schedule model reaches 1.77 % [1.33, 2.30] at 95.8 % served and 33.3× mean speed-up —
level with learning-free LP rounding at 1.75 % and half as fast.

```mermaid
graph LR
  X["Tokens: unit-hour, bus and line inputs, LP relaxation"] --> NET["Recursive network: 2 latent updates and 1 answer update, y and z carried"]
  NET --> SC["Scores from count heads per identical group"]
  SC --> DEC["Viterbi decode, block adequacy repair, min up-down repair"]
  DEC --> LP["Exact dispatch LP, warm-started within the instance"]
  LP --> FB["Feedback: hourly energy and reserve shortfall, bus prices, line loadings, unit dispatch and reserve, exact cost sensitivity, cost and certified gap"]
  FB --> NET
  LP --> CAND["Priced candidate for this step"]
  CAND --> BEST["Keep the cheapest of all K steps; K = 16 costs 3.5 distinct LPs: 1.77 percent, 95.8 percent served, 33.3x"]
```

Arm B's best result uses the same network differently: the LP prices every move and the network only ranks the moves.
The search starts from the learning-free LP-rounding schedule, best of 5 LPs; at each iteration it enumerates block
moves — switch an off unit on for a block of its minimum up time, switch a whole on-block off, trim one hour from
either end, or follow the network wherever its rounding disagrees with the current schedule — and ranks them by the
LP's first-order cost prediction from the exact bound duals minus lambda times the network's mean log-odds supporting
the new values, with lambda = 0.5 chosen on validation. The best untried move is made min up/down-feasible and
adequate, priced by the LP, and kept only if it is cheaper; the network is re-run on the LP feedback of the current
best schedule after every accepted move and every few rejected ones. With 16 LPs after the start this gives 1.14 %
[0.93, 1.39] at 99.2 % served and 20.6× on the 120 test instances: 0.61 pp better than LP rounding, 0.15 pp better than
the same search with no learning at all, and indistinguishable from the same search ranked by the static one-shot
ensemble, so most of the gain is the LP-in-the-loop search rather than the recursion. Adding the safety net — a reduced
MILP at 0.5 % when the schedule is not served or its certified gap to the LP relaxation exceeds 2 %, firing on 25 % of
instances — gives 0.64 % [0.55, 0.74] at 100 % served and 13.1× mean.

```mermaid
graph TD
  S0["Start: LP rounding plus repairs, best of 5 LPs"] --> PR["Price the current best with the dispatch LP: cost and exact cost sensitivities"]
  PR --> MV["Enumerate block moves: switch a block on, switch an on-block off, trim one hour, or follow the network where it disagrees"]
  NET["Recursive network, re-run on the LP feedback of the current best schedule"] --> RK
  MV --> RK["Rank: LP first-order cost prediction minus lambda times the network mean log-odds, lambda = 0.5"]
  RK --> AP["Apply the best untried move, make it min up-down feasible and adequate"]
  AP --> PRC["Price it with the dispatch LP"]
  PRC --> ACC{"Cheaper than the current best"}
  ACC -->|"yes: accept, then re-rank"| PR
  ACC -->|"no: next move"| RK
  PRC --> BUD{"LP budget of 16 used"}
  BUD -->|"yes"| OUT["1.14 percent mean gap, 99.2 percent served, 20.6x"]
  OUT --> SN["Safety net: not served or certified gap above 2 percent, then a reduced MILP at 0.5 percent, 25 percent fall back: 0.64 percent, 100 percent served, 13.1x"]
```

## 5. The controls

Each control changes exactly one thing against its arm's main model and keeps the data, the training steps and the
pipeline identical, so the comparison isolates one design choice. Removing the recursion from arm A leaves the
one-shot network, which is better on probabilities after a sixth of the training time, so recursion does not pay at
equal size. Removing the weight sharing instead, at the same compute per step, is level with the TRM, so the
recursion's advantage over plain depth is also absent. Removing the LP-relaxation inputs from either arm-A model costs
about 2.4 pp, which is the largest single effect in the study and says the relaxation carries most of the answer.
Zeroing every LP-derived feedback signal in arm B, while keeping the priced schedule and what the repairs changed, is
the one ablation with a clear gain for recursion.

| control | the single change | measured effect |
|---|---|---|
| arm A one-shot | the same 2-layer network applied once to the initial state, equal parameters, plain BCE | better probabilities, validation BCE 0.0414 vs 0.0463, and a better best pipeline on test, 1.30 % vs the TRM's 1.48 % |
| arm A deep non-recursive | 10 unshared 2-layer networks applied once, one supervision step's compute, 1.32 M parameters | level with the TRM, Δ gap +0.03 pp [−0.06, +0.14] on test; 5 thresholds + repair 1.34 % |
| arm A no-LP-features pair | the LP-relaxation inputs removed from the TRM and from the one-shot control | validation 1-LP + repair 6.29 % and 6.30 % against 4.42 % and 3.87 %; with no LP anywhere, 7.96 % and 7.24 % |
| arm B feedback-off | every LP-derived signal zeroed, the priced schedule and the repair changes kept | validation K = 16: 6.95 % vs 6.17 %, i.e. feedback −0.78 pp [−1.54, −0.07]; after REINFORCE −1.26 pp [−2.05, −0.62] |

```mermaid
graph LR
  M["Main model of each arm: shared network applied recursively, LP-relaxation inputs, LP feedback in arm B"]
  M -->|"remove the recursion: one pass, same parameters"| A1["Arm A one-shot: BCE 0.0414 vs 0.0463, best pipeline 1.30 vs 1.48 percent"]
  M -->|"remove the weight sharing: 10 unshared nets at matched compute"| A2["Arm A deep non-recursive: level, delta gap plus 0.03 pp"]
  M -->|"remove the LP-relaxation inputs"| A3["No-LP pair: 6.29 and 6.30 percent on validation against 4.42 and 3.87"]
  M -->|"zero every LP-derived feedback signal"| B1["Arm B feedback-off: 6.95 vs 6.17 percent at 16 steps"]
```

## 6. The evaluation protocol

Every method is scored per instance in one single-threaded process pinned to one core, with HiGHS and torch limited to
one thread, and everything solved back to back so the timings are paired. The full MILP at a 0.1 % gap runs first with
a 60-second limit, 300 s on 24-hour windows and 600 s on the congested benchmark; it supplies the dual bound DB and the
denominator of every speed-up. Then come the full MILP at 0.5 %, the LP relaxation — re-solved and timed, because its
time is charged to every method that uses it — the learning-free LP-rounding baseline with repairs and the best of 5
LPs, and finally the method under test, whose time includes featurisation, every network pass, decoding, repairs, LPs
and any MILP fallback. Every schedule is priced by the same exact dispatch LP, with shedding at $10,000/MWh and
reserve shortfall at $1,000/MWh, so the cost C already contains those penalties. Reported are the served share, the
gap to the dual bound as mean, median and max, the speed-up as mean, median and geometric mean, and paired
instance-bootstrap 95 % confidence intervals from 2,000 resamples for differences in gap and in log speed-up. All
model, decoder, threshold, step-count and safety-net choices are made on the 360 validation instances; the 120
`test_fresh` instances are read once, after the selection file is written.

```mermaid
graph TD
  SEL["360 validation instances: model, decoder, thresholds, step count K and safety-net threshold chosen here"] --> I
  I["One test instance of 120 in test_fresh, one process on one core, single-threaded HiGHS and torch"] --> F1
  F1["Full MILP at 0.1 percent, 60 s limit: dual bound DB and the time T_MILP"] --> F2["Full MILP at 0.5 percent"]
  F2 --> R["LP relaxation, re-solved and timed; its time is charged to every method that uses it"]
  R --> B["Learning-free baseline: LP rounding plus repairs, best of 5 LPs"]
  B --> MT["Method under test, timed with every overhead"]
  MT --> SCORE["Price every schedule with the exact dispatch LP: cost C includes shedding at 10000 dollars per MWh and reserve shortfall at 1000 dollars per MWh"]
  SCORE --> M1["Served share: no shedding and no reserve shortfall"]
  SCORE --> M2["Gap: C minus DB over C, mean, median and max"]
  SCORE --> M3["Speed-up: T_MILP over the method time, mean, median and geometric"]
  SCORE --> M4["Paired instance bootstrap, 95 percent CIs, 2000 resamples"]
```

## Where each arm stands

CPU round, uc12 `test_fresh`, 120 instances, one seed per configuration. Both arms re-ran the shared baselines on
their own cores, so their numbers differ slightly: arm A added an LP-guided adequacy repair to every pipeline, arm B
used the existing repairs. Neither arm meets the success criterion of [`PLAN.md`](PLAN.md), namely at least 95 %
served at a mean gap of 0.5 % or less in well under a second, and no end-to-end pipeline beats the MILP pipeline of
§7–§8.

| pipeline | served | mean gap, 95 % CI | speed-up vs the full MILP, mean |
|---|---|---|---|
| full MILP at 0.5 % | 96.7 % | 0.23 % [0.18, 0.29] | 3.9× |
| *MILP pipeline of §7–§8, quoted* | *97 %* | *0.30 %* | *~10×* |
| LP rounding + repair, 5 LPs, no learning | 98.3–99.2 % | 1.71–1.75 % | 43–62× |
| one-shot temporal-GNN ensemble + repairs, 5 LPs | 95.0–99.2 % | 1.45–2.00 % | 34–55× |
| **A**: TRM schedule + 4 LP roundings + repair | 99.2 % | 1.48 % [1.10, 1.98] | 19× |
| A: one-shot, same net, + REINFORCE + reduced-MILP safety net | 100 % | 1.17 % [0.98, 1.37] | 48× |
| **B**: whole-schedule recursion with LP feedback, 16 steps | 95.8 % | 1.77 % [1.33, 2.30] | 33× |
| B: local search from LP rounding, no learning, 16 LPs | 99.2 % | 1.29 % [1.03, 1.57] | 28× |
| **B**: same search, moves ranked by the recursive network | 99.2 % | **1.14 % [0.93, 1.39]** | 21× |
| B: … + reduced-MILP safety net, 25 % fall back | 100 % | 0.64 % [0.55, 0.74] | 13× |
| B: … with a stricter safety net, 52 % fall back | 99.2 % | 0.46 % [0.38, 0.56] | 8.5× |

The CPU round cannot separate "recursion does not help here" from "recursion was not given enough data or epochs": the
TRM saw 3 epochs of 2,000 instances against the one-shot's 24, at well under 1M parameters against TRM's 5–7M. The
GPU runs are meant to settle three things. First, arm A's go/no-go: TRM against the one-shot and the deep
non-recursive control at equal epochs on 50,000 polished labels, with and without LP-relaxation features, a go only if
the TRM beats the one-shot by more than about 0.2 pp. Second, arm B's open question: a larger recursive network with
LP feedback trained with REINFORCE on 50,000 instances against the same without feedback, since the feedback effect
grew with REINFORCE and this is the run that could change the whole-schedule verdict. Third, cheap either way, the
one-shot plus REINFORCE plus safety-net pipeline on 50,000 labels, the best end-to-end pipeline of the study. Stages 1
and 2 together are estimated at roughly 40–45 GPU-hours and 750–850 CPU core-hours.
