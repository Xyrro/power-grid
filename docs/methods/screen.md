# Screened network and a certified pipeline (12 h, 24 h, congested)

Code: [`otsl/screen.py`](../../otsl/screen.py), [`scripts/screen_eval.py`](../../scripts/screen_eval.py),
[`scripts/screen_run.py`](../../scripts/screen_run.py), [`scripts/screen_report.py`](../../scripts/screen_report.py),
[`scripts/pace/pgu_screen.sbatch`](../../scripts/pace/pgu_screen.sbatch); tests [`tests/test_screen.py`](../../tests/test_screen.py).
Results: [`results/screen/report.md`](../../results/screen/report.md) (+ `.json`), validation choice
`results/screen/select.json`, records `results/screen/{val,test}/*.jsonl`.

## Status

* Exploration (2026-11-06, an Apple M4, validation instances only; nothing of the test set read): profiles of the full
  MILP, exact reduced-cost fixing, agreement rules, the screened network, warm starts and the certificate.
* Validation pass on PACE ICE: **done** (2026-11-06/07, 360 instances, four 2-core jobs, 1 h 50 min; 251 instances
  on Xeon Gold 6226 nodes, 109 on AMD EPYC 7513, every method of an instance in one process; none failed).
  Selection (`screen_report.py --select`): **`cert:s0:0.02`** (`results/screen/select.json`).
* Test pass on PACE ICE: **done** (2026-11-07, 400 instances read once, `PHASE=test`, `CERT=s0:0.02+lp:0.05`, four
  2-core jobs, 2 h 40 min, every instance on Xeon Gold 6226, HiGHS 1.15.1; none failed). Report:
  `python3 scripts/screen_report.py` → `results/screen/report.{md,json}`.

## Summary

**A simpler pipeline that is more accurate, twice as fast and certified.** One temporal GNN, one confidence threshold,
the LP relaxation, and a solver that verifies everything the shortcuts skip. On the 120 12-hour test instances it gives
0.115 % [0.098, 0.135] (max 0.47 %) at 12.9× the full MILP at 0.1 %, against 0.228 % (max 2.89 %) at 6.0× for step 2's
recommended pipeline: −0.113 pp [−0.202, −0.046] and 2.16× [1.76, 2.62] faster, paired on the same core. It is also
more accurate than the full MILP at 0.5 % (−0.078 pp [−0.137, −0.027]) and 6.1× faster, and on the six shifted sets
it stays at 0.08–0.14 % on every set (at or below the full MILP at 0.5 % on all six) at 6.2–11.7×. 292 of the 300 12-hour answers carry a
proof of ≤ 0.5 %; the other 8 hit the 60 s limit in the proof step (the full MILP at 0.5 % hits it on 30).

* **The screened network is a free exact speed-up.** Solving on a copper plate with only the line limits that can bind,
  every flow checked after the solve, the full MILP is 2.1× faster at 0.1 % on 12 h (2.07× on 24 h) and *more*
  accurate on average, because it hits the time limit less often. On the congested benchmark, where ~32 lines start
  monitored and two rounds are common, it is no faster (0.96–1.08×), as expected.
* **The certificate is what makes one threshold enough.** The same fixings without it average 1.01 % on 12 h (max 59 %)
  and 9.1 % under line outages. Step 2 needed a Benders-tuned threshold pair per generator, three guards and a bound-check
  threshold to stay safe; here the solver proves the answer instead, so the fixing rule only decides the speed.
* **Learning buys speed, not accuracy, once answers are certified.** The same pipeline on the LP relaxation's values
  (no learning) is as accurate (−0.011 pp [−0.029, +0.007]) but 2.16× [1.80, 2.59] slower: its fixings are worse, so
  its certificate falls to the MILP on 87 of 120 instances against 40 for the network's. Without learning it is still a
  strong certified solver: 9.0× at 0.171 % on 24 h and 4.3× at 0.152 % on the congested lines (full MILP at 0.5 %:
  2.7× and 2.4×).

## Test (400 instances, PACE ICE, Xeon Gold 6226)

Scoring as step 2: gap to the tightest bound known for the instance (step 1, step 2 and these runs' MILPs), served =
no shed, over-generation or reserve shortfall, speed-up paired with the full MILP at 0.1 % in the same process.

**12 h, test_fresh (120)**

| method | gap mean [95 % CI] | max | served | time s | speed-up geo |
|---|---|---|---|---|---|
| full MILP, 0.1 % (reference) | 0.158 [0.090, 0.248] | 3.64 | 96.7 % | 29.51 | 1.0× |
| full MILP, 0.5 % | 0.193 [0.141, 0.259] | 2.11 | 96.7 % | 16.51 | 2.1× |
| screened full MILP, 0.1 % (no learning) | 0.079 [0.056, 0.109] | 1.27 | 96.7 % | 18.90 | 2.1× |
| screened full MILP, 0.5 % (no learning) | 0.123 [0.097, 0.154] | 1.27 | 99.2 % | 11.41 | 3.7× |
| step 2's recommended pipeline | 0.228 [0.159, 0.310] | 2.89 | 95.0 % | 6.59 | 6.0× |
| step 2's pipeline + bound check | 0.193 [0.149, 0.242] | 1.66 | 95.0 % | 7.59 | 5.9× |
| its fixings on the screened network | 0.240 [0.170, 0.327] | 2.89 | 95.0 % | 2.84 | 17.0× |
| **certified pipeline (proven within 0.5 %)** | **0.115 [0.098, 0.135]** | **0.47** | **97.5 %** | 7.32 | **12.9×** |
| certified pipeline, LP certificate at 1 % | 0.124 [0.104, 0.143] | 0.48 | 97.5 % | 6.87 | 15.4× |
| certified pipeline without the certificate | 1.012 [0.213, 2.240] | 58.59 | 94.2 % | 2.05 | 22.0× |
| certified pipeline on the LP values (no learning) | 0.126 [0.106, 0.147] | 0.53 | 96.7 % | 9.56 | 6.0× |

Paired (A − B in pp, A's speed-up over B; 95 % bootstrap intervals):

| A | B | gap A − B | speed-up |
|---|---|---|---|
| certified pipeline | step 2's pipeline | −0.113 [−0.202, −0.046] | 2.16× [1.76, 2.62] |
| certified pipeline | step 2's pipeline + bound check | −0.077 [−0.130, −0.029] | 2.20× [1.80, 2.66] |
| certified pipeline | full MILP, 0.5 % | −0.078 [−0.137, −0.027] | 6.12× [5.05, 7.34] |
| certified, LP certificate at 1 % | step 2's pipeline | −0.105 [−0.190, −0.036] | 2.57× [2.10, 3.12] |
| certified pipeline | the same on the LP values | −0.011 [−0.029, +0.007] | 2.16× [1.80, 2.59] |
| screened full MILP, 0.5 % | full MILP, 0.5 % | −0.070 [−0.119, −0.030] | 1.74× [1.52, 2.00] |
| screened full MILP, 0.1 % | full MILP, 0.1 % | −0.079 [−0.159, −0.022] | 2.12× [1.87, 2.40] |
| step 2's fixings, screened | step 2's pipeline | +0.012 [−0.001, +0.026] | 2.83× [2.50, 3.26] |

**Shifted 12-hour sets (30 each; nothing re-tuned):** mean gap · served · speed-up

| method | load +15 % | load −15 % | wind + solar ×1.5 | units out | lines out | across midnight |
|---|---|---|---|---|---|---|
| full MILP, 0.5 % | 0.13 · 100 · 1.7× | 0.19 · 90 · 1.8× | 0.28 · 97 · 1.6× | 0.29 · 90 · 1.7× | 0.13 · 90 · 2.0× | 0.21 · 93 · 1.8× |
| screened full MILP, 0.5 % | 0.11 · 100 · 3.0× | 0.13 · 93 · 3.1× | 1.00 · 87 · 2.5× | 0.16 · 97 · 3.2× | 0.10 · 93 · 3.6× | 0.14 · 93 · 3.0× |
| step 2's pipeline + bound check | 0.16 · 100 · 5.0× | 0.23 · 97 · 4.0× | 0.25 · 93 · 3.7× | 0.41 · 93 · 6.0× | 0.16 · 93 · 4.8× | 0.15 · 93 · 5.2× |
| **certified pipeline** | **0.13 · 97 · 11.7×** | **0.10 · 93 · 10.0×** | **0.08 · 90 · 6.2×** | **0.14 · 93 · 10.3×** | **0.12 · 93 · 10.3×** | **0.11 · 90 · 9.6×** |
| … without the certificate | 0.18 · 97 · 17.7× | 0.75 · 93 · 23.5× | 0.25 · 90 · 10.9× | 2.54 · 83 · 21.1× | 9.11 · 67 · 27.5× | 0.23 · 93 · 17.1× |

Over the 180 shifted instances: −0.112 pp [−0.162, −0.070] and 2.03× [1.75, 2.32] against step 2's pipeline + bound
check; −0.093 pp [−0.141, −0.047] and 5.42× [4.61, 6.36] against the full MILP at 0.5 %. (The screened full MILP's
1.00 % with more wind and solar is one instance where it stopped at the 60 s limit holding a schedule 24 % above the
optimum; the certified pipeline cannot do that, as it keeps the better of the reduced schedule and the proof step's.)

**24 hours (40) and congested lines (60)**: no learned model of these benchmarks was kept, so only the learning-free
methods ran.

| method | 24 h: gap · max · speed-up | congested: gap · max · speed-up |
|---|---|---|
| full MILP, 0.1 % (reference) | 0.104 · 0.39 · 1.0× | 0.088 · 0.27 · 1.0× |
| full MILP, 0.5 % | 0.193 · 0.84 · 2.7× | 0.190 · 0.55 · 2.4× |
| screened full MILP, 0.1 % | 0.051 · 0.14 · 2.1× | 0.078 · 0.27 · 1.0× |
| screened full MILP, 0.5 % | 0.121 · 0.38 · 4.4× | 0.152 · 0.46 · 2.6× |
| certified pipeline on the LP values | 0.171 · 0.46 · 9.0× | 0.152 · 0.44 · 4.3× |
| … LP certificate at 1 % | 0.249 · 0.79 · 14.1× | 0.179 · 0.66 · 5.6× |

**Anatomy** (means): the chosen pipeline fixes 93.1 % of the decisions on test_fresh (91.5 % shifted), starts with 4.9
monitored lines (8.5 shifted; line outages move flows) and needs 1.32 screen rounds; the LP bound certifies 80 of 120
answers, the MILP the other 40. Time: LP relaxation + inference 0.35 s, reduced MILP 1.69 s, certificate 5.27 s (the
proof dominates, as in the exploration, where switching the solver's heuristics off did not shorten it).

**Checks.** Every one of the 1,100 screened solves ended with all lines within their ratings; no delivered cost is below
the tightest bound; 315 of 340 full MILPs at 0.1 % and 221 of 300 runs of step 2's pipeline land within 0.1 % of step 2's
costs (another CPU, and both stop at a gap, so incumbents differ); 8 of the chosen pipeline's 300 12-hour answers and
17 of the learning-free variant's 400 are not proven within 0.5 % (the proof step's 60 s limit), against 30 time-limited
runs of the full MILP at 0.5 % on the same 300.

## Caveats

* One trained network (the first of step 2's five, trained once); validation and test ran on different node types (Xeon
  Gold 6226 and AMD EPYC for validation, Xeon Gold 6226 for the test), and not on step 2's Granite Rapids node, so the
  references were re-solved here; step 2's numbers are not mixed in except for the bounds.
* The certificate is only as strong as the proof step's time limit (60 s on 12 h): instances whose full MILP cannot
  prove 0.5 % in that time keep a weaker proven gap, as with the full MILP itself.
* The screen helps where few lines bind (RTS-GMLC as given: 4.9 monitored lines on average); on the congested variant it
  does not, and on a large system with many binding lines its rounds would have to be counted against it.
* The learned pipelines for 24 h and the congested benchmark could not be re-run (their models were not kept).

**Plan, fixed before the validation and test passes.** Methods: the full MILP on the B-theta network at 0.1 % (the
speed-up reference) and 0.5 %; the same MILP on the screened network (no learning); step 2's recommended pipeline and
its bound check; its fixings on the screened network; the certified pipeline `cert:<model>:<delta>` for model ∈ {ens5
(the pipeline's five temporal GNNs, averaged), s0 (its first network alone)} × delta ∈ {0.02, 0.05, 0.1}, and on the LP
relaxation's values (no learning, delta 0.05). Choice: the certified pipeline with the highest geometric-mean speed-up
over the 360 validation instances (every one is proven within 0.5 %, so speed is the only criterion); the single
network if its speed-up is within 5 % of the best. The test set is then read once, with that choice and the
learning-free variant. Two operating points are read off the same test records without re-solving: the reduced
schedule alone (no certificate) and an LP certificate at 1 % (proven within 1 %).

## Method

The certified pipeline has three steps, and each restriction it makes is checked by an exact computation:

1. **Relax.** One LP relaxation of the full network (~0.2 s): its cost is a lower bound z_LP, its line loadings say
   which limits can bind, and its values are the network's input features (as in step 2's pipeline).
2. **Restrict.** A temporal GNN predicts each unit-hour's on-probability p; every decision with min(p, 1 − p) ≤ δ is
   fixed to round(p) (one threshold for all units and hours), and a unit's fixings are released when min up/down times
   make them impossible to complete. The network is restricted to the (hour, line) pairs the relaxation loads to
   ≥ 99 % of their rating; every other line becomes a PTDF row only if a solution overloads it.
3. **Solve and verify.** The reduced MILP on the screened network (0.5 % gap); after every solve all line flows are
   computed and violated limits are added (exact for the network). The schedule's cost C is then certified: if
   (C − z_LP)/C ≤ 0.5 %, it is returned; otherwise the screened full MILP at 0.5 %, warm-started from it, either proves
   it or finds a better one.

Every answer is therefore feasible for the full network and proven within 0.5 % of the optimum, whatever the model
predicts; the model and δ decide only how much of the work the cheap reduced MILP does. Compared with step 2's
pipeline this drops the per-generator thresholds (Learning to Fix's Benders tuning on 360 validation instances), the
adequacy and LP-relaxation guards and the bound-check threshold, and adds the screen and the certificate. The
screen alone (no learning) is also a faster exact solver of the full problem.

## Validation (360 instances, PACE ICE)

Gap to the tightest bound of the instance's own runs (full MILPs, screened MILPs, LP relaxation); speed-up: the full
MILP at 0.1 % on the B-theta network in the same process, geometric mean.

| method | gap mean | max | served | time s | speed-up |
|---|---|---|---|---|---|
| full MILP, 0.1 % | 0.142 % | 5.43 % | 96.7 % | 27.81 | 1.0× |
| full MILP, 0.5 % | 0.193 % | 6.90 % | 96.1 % | 15.43 | 2.1× |
| screened full MILP, 0.1 % (no learning) | 0.106 % | 1.62 % | 96.7 % | 19.77 | 1.9× |
| screened full MILP, 0.5 % (no learning) | 0.144 % | 1.18 % | 95.8 % | 10.86 | 3.4× |
| step 2's pipeline (thresholds tuned on these instances) | 0.208 % | 1.78 % | 94.4 % | 6.90 | 5.9× |
| its fixings on the screened network | 0.203 % | 1.78 % | 94.4 % | 2.33 | 16.8× |
| certified, ens5, δ = 0.05 (best) | 0.151 % | 1.25 % | 94.7 % | 6.35 | 12.3× |
| **certified, s0, δ = 0.02 (chosen)** | **0.148 %** | 1.24 % | 95.3 % | 6.52 | **12.1×** |
| … LP certificate at 1 % | 0.157 % | 1.24 % | 95.0 % | 5.93 | 15.0× |
| … without the certificate | 0.943 % | 76.8 % | 92.5 % | 1.98 | 21.6× |
| certified, LP values (no learning) | 0.161 % | 1.18 % | 96.1 % | 8.55 | 6.0× |

* All six learned certified variants lie within 9.2–12.3×; the single network at δ = 0.02 is within 2 % of the best
  (the ensemble at δ = 0.05) and is chosen by the rule fixed beforehand.
* **The certificate is what makes one threshold enough.** Without it the same fixings give 0.94 % on average and up to
  77 % (a few wrong OFF fixings shed load); with it, 0.148 %. On 213 of 360 instances the LP bound certifies the reduced
  schedule outright; on 147 the screened MILP runs (mean times: LP relaxation + inference 0.30 s, reduced MILP 1.67 s,
  certificate 4.55 s). The pipeline fixes 92.9 % of the decisions and monitors 6.0 lines (1.42 screen rounds).
* The full MILP's means include instances where it hits the 60 s limit (max 5.4–6.9 %); the screened MILP hits it less
  often, so it is both faster and, on average, closer to the bound.

## Why (exploration, validation instances, M4)

* **Where the full MILP's time goes.** At a 0.5 % gap, HiGHS spends 3–10 s finding a schedule within 0.5 % (root
  sub-MIP heuristics) and, on instances whose LP relaxation is weak, as long again raising the bound. Exact reduced-cost
  fixing from the LP relaxation fixes only 1–8 % of the commitments even with the optimal cost as cut-off: the
  relaxation's reduced costs are far below the gap. Fixing only where the LP relaxation and the ensemble agree (per unit
  or per unit-hour) was not better than step 2's thresholds.
* **The network is carried, not used.** At the optimum 0–25 of the 1,440 (hour, line) limits bind; the copper-plate
  MILP solves in 0.4–3.5 s where the network MILP takes 5–27 s, but its schedules can shed load behind congested lines
  (up to 15× the cost). Hence the screen: copper plate + the lines the LP relaxation loads to ≥ 99 %, every line's flow
  checked after the solve, violated ones added (1.6 rounds on average). Exact, and on 60 validation instances 1.8× faster
  for the full MILP and 4.1× [3.3, 4.9] faster for the pipeline's reduced MILP, at the same or lower cost.
* **Certificate.** With the reduced MILP this cheap, every answer can carry a proof: return the reduced schedule when
  the LP relaxation's bound is within 0.5 % of its cost; otherwise solve the screened full MILP at 0.5 % warm-started
  from it. Then the fixing rule only affects speed, never the guarantee, and one global confidence threshold replaces
  the per-generator tuning (Learning to Fix's Benders procedure) and the three guards.
