# Our methods and Learning to Fix on a large public system: PGLib-UC California

*Code: [`otsl/pglib.py`](../../otsl/pglib.py) (PGLib-UC MILP, LP relaxation, dispatch LP, reduced MILP, scenario
generator, repairs), [`otsl/pglib_ml.py`](../../otsl/pglib_ml.py) (adapters: per-unit Model 1, persistent dispatch LP,
LP oracle, Table II kNN, Learning-to-Fix tuner, LP guard, solver workers),
`scripts/uc_pglib_{probe,validate,gen,train,ltf,eval,report}.py`. Results:
[`results/pglib/pglib_results.md`](../../results/pglib/pglib_results.md) (+ `.json`), per-instance records
`results/pglib/pglib_eval_test_{a,b,c}.jsonl`, probe `results/pglib/pglib_probe.jsonl`, validation
`results/pglib/pglib_validate_{lp,milp}.json`, training / tuning statistics `results/pglib/pglib_train_stats.json`,
`results/pglib/pglib_ltf_*.json`, `results/pglib/pglib_stcheck.json`. Data (git-ignored):
`data/generated/pglib_ca/{train,val,test,st_labels,pglib_probs}.npz`; models `results/pglib/pglib_*.pt` (git-ignored).*

**Summary.** The PGLib-UC MILP (Knueven et al.; `MODEL.pdf`) is implemented as a sparse matrix for HiGHS and
validated against the reference Pyomo model (LP relaxations equal to 1e-14 on three instances, MILP optima equal on
two). With HiGHS on this machine the 48-hour California instances take 6 to > 15 min per MILP, so the study uses
their first 24 hours (610 units, 410 free; full MILP 19–565 s, mean 100 s — the paper's regime). On 30 perturbed test
instances, with no full MILP used for training: **faithful Learning to Fix (kNN, ε = 1 %) reproduces the paper's
headline: 0.55 % at 25.5× (96.7 % feasible; paper 0.48 % at 20.8×; gap interval [0.31, 0.91] %)**, but its median
speed-up is 5.8×. **Our guarded
error-cost rule gives 0.195 % at 12.4× (98 % target) and 0.043 % at 8.5× (95 %), 100 % feasible** — more accurate
than LtF-kNN on the same instances (−0.35 pp [−0.70, −0.12]) and faster by median and ratio of mean times, but half
its mean of per-instance speed-ups. Learning to Fix on our probabilities fixes 96 % but fails on 23 % of test
instances. The strongest results come **without learning**: the LP relaxation is within 0.03 % of the optimum, and
rounding it + repair + 5 dispatch LPs gives 0.33 % at 24× (100 % feasible); fixing its integral values gives
0.105 % at 14.3×. Our learned end-to-end mode (0.65 % at 23.5×) does not beat it.

## 1. Why this system

Everything in §3–§6 of [`RESEARCH.md`](../RESEARCH.md) ran on RTS-GMLC (73 thermal units), where the full MILP takes
~20 s (12 h) to ~150 s (24 h) and caps speed-ups. Learning to Fix (Fritz et al. 2026) reports 0.48 % at 20.8× on the
EPRI Irish system (51 units, 72 h, Gurobi, full MILP ~60 s mean). Its data are not public. PGLib-UC is: the
California set (`ca/`, 20 instances) has **610 thermal units** (200 must-run), a copper-plate balance, spinning
reserves of 0–5 % and, in the Scenario400 profiles, an aggregate wind unit for a hypothetical 40 % wind supply.

## 2. Formulation and validation

`otsl/pglib.py` builds the MILP of `MODEL.pdf` / `uc_model.py` (Knueven, Ostrowski & Watson; Morales-España et al.'s
tight 3-binary formulation with Sridhar et al.'s piecewise cost) as one sparse matrix: equations (1)–(24) —
piecewise-linear production cost by convex combination of the curve points, off-time-dependent start-up categories
(15)–(16) and their initial-condition rule (7), start-up / shut-down capability (17)–(18), ramping with reserve counted
against ramp-up (19)–(20) and its first-period versions (8)–(10), min up / down (13)–(14) and the initial requirements
(4)–(5), must-run (11), reserves (3), renewables between their minimum and maximum (24). It is solved with HiGHS
1.15 through highspy (single thread; MIP callback for the incumbent log). Implementation choices, none of which
changes the optimum:

* must-run units carry no binaries (u = 1 substituted; their no-load cost sits on one column fixed at 1), so the
  model's u block holds the **410 free units** — the decisions every method predicts and fixes (9,840 per 24-h
  instance);
* the start-up category variables δ are continuous: for integral (u, v, w) each (g, t) block is an interval matrix,
  so δ is integral at every vertex (checked against binary δ below);
* ramp rows that cannot bind (RU, RD ≥ Pmax − Pmin; all but 36 of the 610 units) are dropped;
* load shedding / over-generation (10,000 $/MWh) and reserve shortfall (5,000 $/MWh) are priced slacks, so every
  commitment that satisfies min up / down has a feasible dispatch — needed by the dispatch LP used as critic and
  for label-free training. In every validation and test MILP the slacks are zero.

Provided: the full MILP (`PGModel.solve_milp`, incumbent log), the LP relaxation and the fixed-commitment dispatch LP
(`solve_dispatch`, with the demand and reserve duals), the reduced MILP (`solve_milp(z_fix=...)`) and the relaxed
reduced LP of the LP guard (`relaxed_reduced`). `PGModel` exposes the attributes of `otsl.uc.UCModel` that the reused
code touches (`A, c, integ, nv, T, off, ix, dims, _rhs_bounds, solve_dispatch`), so `otsl.ltfx`, `otsl.fixpolicy`,
`otsl.combo`, `otsl.selftrain` and `otsl.ucml` run on it unmodified. Size at T = 24: 114,289 columns (29,520 binary),
103,896 rows, 358,695 non-zeros; T = 48 doubles that.

**Validation against the reference implementation** (`scripts/uc_pglib_validate.py`, `results/pglib/pglib_validate_lp.json`,
`pglib_validate_milp.json`). Pyomo was installed in a scratch directory only for this check; the reference model is
built by executing `uc_model.py`'s model-construction code unchanged and solved with HiGHS through Pyomo:

| check | instance | ours | reference (Pyomo) | relative difference |
|---|---|---|---|---|
| LP relaxation | CA 2014-09-01, 3 %, 48 h | 48,392.926 | 48,392.926 | 7e-15 |
| LP relaxation | CA Scenario400, 5 %, 48 h (wind) | 33,834.863 | 33,834.863 | 2e-14 |
| LP relaxation | RTS-GMLC 2020-01-27, 48 h (49 of 73 units off at t0, 1–3 start-up categories, 81 renewables, must-run) | 1,205,494.51 | 1,205,494.51 | 1e-15 |
| fixed commitment priced (our dispatch LP vs reference with u fixed) | CA 2014-09-01 | 48,507.964 | 48,507.964 | 2e-14 |
| same | CA Scenario400 | 34,380.304 | 34,380.304 | 2e-14 |
| same | RTS-GMLC 2020-01-27 | 1,557,614.87 | 1,557,614.87 | 4e-16 |
| MILP optimum (gap 1e-4; ours with δ continuous / δ binary / reference) | CA 2015-03-01, 1 %, first 12 h | 7,121.7018 (25 s) / 7,121.7018 (39 s) | 7,121.7018 (59 s) | 1e-15 |
| same | RTS-GMLC 2020-01-27, first 12 h | 148,851.672 (11 s) / 148,851.672 (14 s) | 148,851.672 (14 s) | 4e-15 |

The LP relaxations agree to machine precision on three instances that exercise the constraint families (piecewise
costs, start-up categories with initial off-times, ramping, start-up / shut-down capability, must-run, renewables
with a must-take minimum; not exercised by any PGLib instance used: an initial minimum up / down time still to serve
(4)–(5) and an initial output above minimum in (8)–(10)), our soft slacks are zero in all of them, and a given schedule costs the
same in both models, and the MILP optima agree (with δ continuous or binary) — i.e. the matrices are the reference
formulation. (A first attempt at the MILP check with a 1e-6 gap on the 24-hour RTS-GMLC instance was stopped after
13 min; the check was rerun on 12-hour instances at 1e-4.)

## 3. Probe: how long does HiGHS take?

Base California instances, 0.1 % gap target, 900 s limit, one thread, at most two solves in parallel on the shared
4-core machine (load average 4.4–5.7 during the probe; `scripts/uc_pglib_probe.py`, `results/pglib/pglib_probe.jsonl`):

| instance | T | status | time s | final gap | B&B nodes | LP relaxation (s) | MILP − LP relaxation | first incumbent within 0.5 % |
|---|---|---|---|---|---|---|---|---|
| 2014-09-01, 3 % reserve | 48 | optimal | 339 | 0.067 % | 1 | 7.1 s | 0.08 % | 339 s |
| Scenario400, 3 % reserve | 48 | **time limit** | 900 | 0.200 % | 2 | 14.8 s | 0.31 % | 385 s |
| 2014-09-01, 3 % reserve | 24 | optimal | 68 | 0.068 % | 1 | 2.2 s | 0.09 % | 68 s |
| Scenario400, 3 % reserve | 24 | optimal | 78 | 0.099 % | 1 | 4.6 s | 0.31 % | 71 s |
| 2015-03-01, 5 % reserve | 24 | optimal | 117 | 0.100 % | 1 | 3.3 s | 0.14 % | 117 s |

* **The 48-hour California instances are impractical for this study with HiGHS on this machine**: 5.7 min for the
  easiest profile, > 15 min (0.20 % at the limit) for the wind profile. A study needing ~60 full MILPs for validation
  and test, plus Learning to Fix's check MILPs, does not fit in the time budget at that size.
* **Fallback: the first 24 hours.** In the four non-wind profiles the 48-hour demand is the same day twice, so the
  first 24 hours keep the instance's structure; the wind profile loses its second day. At T = 24 the full MILP takes
  1–2 minutes (68–117 s here; 15–334 s, mean 84 s over the 30 validation instances), i.e. the regime of the paper
  (Irish system, full MILP 60 s mean, up to 947 s). The 610-unit fleet, all constraints and the 0.1 % target are kept.
* **What HiGHS spends the time on.** Every solve ends at the root node (1–2 nodes). The LP relaxation takes 2–5 s (24 h) and
  is within 0.03–0.3 % of the MILP optimum; the rest is root cut rounds and heuristics. The first incumbents are
  penalty-laden (reserve shortfall priced at 5,000 $/MWh: 40–300 × the optimal cost) and **the first schedule
  within 0.5 % appears only at the end of the root, 90–100 % into the solve at T = 24**. Time-to-quality is therefore close to
  time-to-proof here, unlike on RTS-GMLC 24 h (§X4 of RESEARCH.md, where a 0.5 %-good schedule came after a median
  28 s of a 154 s solve).
* **The LP relaxation is very tight.** On the 30 validation instances the MILP optimum is a median 0.03 % (max 0.41 %)
  above the LP relaxation, and only 0.88 % of the relaxation's commitment values are fractional. Rounding the
  relaxation and repairing it (no learning, no MILP, ~4 s) gives a schedule a median 0.08 % (mean 0.44 %, max 3.3 %)
  above the MILP. This is a far easier fixing problem than RTS-GMLC, and it matters for every comparison below.

## 4. Data, labels and their cost

Perturbed instances of the five California load profiles (`otsl.pglib.make_instance`; 24 h): a profile drawn
uniformly; demand × U(0.92, 1.08) × (1 + AR(1) noise, σ = 1.5 %, ρ = 0.7); spinning reserve U(0, 5 %) of demand;
for Scenario400, wind × U(0.7, 1.3) × (1 + AR(1) noise, σ = 5 %) with the base minimum / maximum ratio (must-take
50 %); demand capped so that peak net load + reserve ≤ 97 % of thermal capacity; initial condition as in the base
files (all 610 units on at minimum output, minimum up time served). Instances whose LP relaxation needs shedding or
reserve shortfall would be redrawn (none was). Train, validation and test use different seeds (1 / 2 / 3); all five
profiles occur in every split.

| split | n | what is solved | cost |
|---|---|---|---|
| train | 160 (144 + 16 label-free hold-out for early stopping) | LP relaxation (2.9 s) + 4 dispatch LPs for the label-free label | 4.1 s / instance, 11 core-min in total |
| self-training labels | first 100 training instances | one reduced MILP each (90 % fixed, 60 s limit) | 18 s / instance, 30 core-min |
| error-cost labels | first 60 of those | 61 wrong decisions / instance, 3,732 warm-started dispatch LPs | 7 core-min |
| REINFORCE | 40 steps × 8 × 4 | 1,280 dispatch LPs | 7 core-min |
| validation | 30 | full MILP (0.1 %, 900 s) + incumbent log | 84 s mean (15–334), 0.70 core-h |
| test | 30 | full MILP (0.1 %, 900 s) + incumbent log | 100 s mean (19–565), 0.84 core-h |

All validation and test MILPs proved the 0.1 % target (mean final gap 0.03 %), none needs shedding or reserve
shortfall, and the dispatch LP of each MILP schedule reproduces its objective to 3e-14.

**Labelled set of the kNN.** Learning to Fix needs labelled schedules; a full-MILP label costs 84–100 s here. The
labelled set is the **100 self-training labels** (one 18-s reduced MILP each), which is also what our error-cost model
uses as reference. Their quality, measured by applying the same labelling step to 20 validation instances with known
MILP (`scripts/uc_pglib_stcheck.py`): **0.015 % above the MILP on average** (median 0.002 %, max 0.17 %; 8 of 20 below
the MILP's 0.1 %-optimal objective), against 0.46 % for the repaired relaxation alone. The kNN therefore sees
near-optimal labels at a fifth of the cost of MILP labels (and with 100 instances, k = 50 averages half the set).

**No full MILP is used for training** any of our models. The validation MILPs serve Learning to Fix (C* in its
constraint (8e)) and the one selection rule; the test MILPs serve the metrics.

## 5. Methods as transferred

Every method is imported from its module and run unchanged on `PGModel`; only the parts that depended on
RTS-GMLC's structure were replaced (`otsl/pglib_ml.py`):

| component | RTS-GMLC version | here |
|---|---|---|
| Model 1 | `otsl.ucml.CommitGNN` (message passing over the transmission graph) | **per-unit network without a graph** (`UnitNet`): unit encoder (pmax, pmin ratio, average and marginal cost, no-load and start-up cost, min up / down, ramp, merit rank + a learned 8-dim unit embedding), hour encoder (demand, net load, reserve, wind, LP-relaxation energy and reserve prices, hour of day), decoder per (hour, unit) with the unit's LP-relaxation commitment at t − 1, t, t + 1 and its price margins; 48 hidden units, shared weights. Wrapped in `otsl.ucml.UCModel1`, so `train_uc_bce`, `train_uc_reinforce` and `otsl.selftrain.train_bce_fixed` train it unchanged |
| label-free labels | relaxation rounded at 0.5, adequacy + min up / down repair | the same repairs (`otsl.uc.adequacy_repair`, `repair_min_updown`, must-run capacity passed as an extra availability column), but the rounding threshold is screened over {0.5, 0.2, 0.05, 0.001} by the exact dispatch LP: at 0.5 the repaired schedule often has a reserve shortfall that capacity-based repairs cannot see (ramp-limited reserve), 30× the optimal cost on one instance |
| dispatch LP / critic | `otsl.ucml.DispatchOracle` (scipy) | the same interface; the LP stays loaded in one highspy object per worker and a new commitment only changes column bounds (warm start: 0.2 s instead of 1.6 s) |
| REINFORCE | 120 steps × 16 instances × 6 samples | 40 steps × 8 × 4 (1,280 LPs, 7 min), reference cost = LP relaxation |
| self-training | 3 rounds, reduced MILPs at 90 % fixed | **one round** on 100 training instances: fix the 90 % most confident decisions that agree with the label-free label, reduced MILP (60 s limit, 0.1 %), keep if cheaper by the dispatch LP; retrain on the improved pool (80 → 40 epochs) |
| error-cost (harm) model | MILP-schedule reference, held-out predictions, compensated harms | reference = **self-training labels** (no MILP), predictions of the self-trained model on its own training instances (in-sample, see §8), single-decision harms (one warm-started dispatch LP per wrong decision); `otsl.fixpolicy.HarmModel` / `HarmEnsemble` (3 members) and `FixFeaturizer` unchanged |
| guards | `otsl.combo.adequacy_guard`, `otsl.fixpolicy.release_conflicting_rows`, `lp_guard` | the first two unchanged; `lp_guard` re-implemented line for line with the relaxed reduced LP solved through highspy (`PGModel.relaxed_reduced`) |
| kNN of Learning to Fix | Table II features of RTS-GMLC | Table II for a copper plate with one wind profile (demand, wind, solar = hydro = 0, net load, deltas, moving average / max, horizon and daily normalisations, sin / cos of the hour, initial status); `k = 50`, eq. (13), z-scored features, canonical labels — as `otsl.ltfx.KNNProb` |
| Learning to Fix tuning | `otsl.ltfx.LtFTuner` | **unchanged** (subclass only builds PGLib scenarios); Algorithm 1 + 2, Appendix A master (Q = 20); check MILPs 120 s, relaxation MILPs 20 s, **K_max = 3** (paper 10), 60-min budget per run (45 min for kNN at ε = 5 %) |

Fixing rules on test (all use the same reduced MILP, 0.1 %, 900 s limit; fixed shares are of the 9,840 free decisions):

* **no learning**: fix every commitment whose LP-relaxation value is integral (99 % of them), row check;
* **paper's kNN**: constant thresholds [0.01, 0.99] and [0.05, 0.95]; Learning to Fix tuned at ε = 1 % and 5 %;
* **Learning to Fix on our probabilities** (ε = 1 %), source chosen on validation (below);
* **RACLearn-style**: constant thresholds [0.01, 0.99] and the 95 % most confident decisions, on each of our models;
* **ours**: error-cost ranking + adequacy guard at 90 %; error-cost + adequacy + min up / down rows + LP guard at 95 %
  and 98 % (the combined pipeline: self-trained probabilities + error cost + both guards);
* **end-to-end** (no MILP): threshold → block adequacy repair → min up / down repair → dispatch LP, at threshold 0.5
  (one LP) or screened over {0.001, 0.05, 0.2, 0.5, 0.8} (five LPs), on each model and on the LP relaxation itself.

**Learning to Fix tuning (validation, 30 instances).** All three runs converged without hitting their budget: kNN
at ε = 1 % in 14 iterations (26 min, 0.42 core-h), kNN at ε = 5 % in 13 (20 min), our self-trained model at ε = 1 % in
15 (15 min); validation fixed shares 86.2 / 86.9 / 95.5 %; the independent check (each instance's witness schedule
priced by the dispatch LP) gives at most 0.91 / 1.92 / 0.94 % above C*, i.e. every validation instance within ε.
Each failing check produced one release set (the second relaxation MILP never found a solution within 20 s). Of the
failing checks that returned a solution, all but one cost 600 to 10⁵ times C* (reserve shortfall or shedding
forced by OFF fixes); the rest had no solution below (1 + ε) C*. Only one cut (self-trained model, 1.2 % above C*)
came from a genuine cost increase, so ε barely changes the kNN thresholds (86.2 vs 86.9 % fixed). The paper's kNN
fixed 78.8 / 82.3 % at ε = 1 / 5 %.

**Selection on validation only.** The probability source for Learning to Fix on our model: the lowest error rate of
the 95 % most confident decisions against the aligned validation MILP schedules — self-trained 0.505 %, label-free
0.525 %, REINFORCE 0.69 %, kNN 1.72 % → self-trained. Learning to Fix's thresholds are tuned on the 30 validation
instances. Everything else (targets 90 / 95 / 98 %, the constant thresholds, the screening grid) was fixed in advance
from the earlier studies; no test instance was read before the final evaluation.

## 6. Results (test)

30 test instances (seed 3, never used for a decision); all reduced MILPs of an instance solved back to back in one
worker process (two such processes in parallel, spawn pools), HiGHS single thread, 0.1 % target, 900 s limit (no
reduced MILP hit it).
`T_MILP` is the generation run's full MILP (100 s mean, 19–565 s); on the first 6 instances it was re-solved back to
back in the evaluation worker: 67.1 s vs 67.5 s (ratio 1.01), and every rule's speed-up against the back-to-back time
is within 4 % of the one reported (`pglib_results.md`). Runtimes include the LP relaxation used as model input
(2.9 s mean), the forward pass (8 ms), kNN search (7 ms), error-cost features and scoring, adequacy guard, min up /
down row check and the LP guard's LPs (0.7–1 s).

**Paper metrics** (Sec. IV-C of the paper): gap (C − DB) / C to the full MILP's dual bound; feasible = the reduced
problem has a solution **without shedding or reserve shortfall** (the PGLib model has hard balance and reserve
constraints, so a penalty-priced solution of our soft model is an infeasible reduced problem there); gap, runtime,
speed-up and fixed share over feasible instances; speed-up = mean of per-instance T_MILP / T_m. Added: median and
ratio of mean times, and the **time-to-quality** speed-up (time at which the full MILP's incumbent first reaches the
rule's cost, over the rule's time). Fixed share of the 9,840 free decisions.

| rule | feasible % | gap mean / max % | runtime mean s | speed-up mean / median / max | ratio of mean times | time-to-quality | fixed % |
|---|---|---|---|---|---|---|---|
| full MILP (HiGHS, 1 thread) | 100 | 0.03 / 0.09 | 100.4 | 1 | 1 | 1 | 0 |
| *no learning*: fix the LP relaxation's integral values | 100 | 0.105 / 0.29 | 7.4 | 14.3 / 11.2 / 76 | 13.5 | 11.0 | 99.2 |
| kNN (k = 50), constant [0.01, 0.99] | 93.3 | 0.03 / 0.10 | 51.4 | 2.9 / 2.3 / 20 | 2.0 | 2.9 | 77.1 |
| kNN, constant [0.05, 0.95] | 93.3 | 0.05 / 0.48 | 45.9 | 4.7 / 2.5 / 28 | 2.3 | 4.1 | 80.6 |
| **Learning to Fix, kNN, ε = 1 %** (paper's setting) | 96.7 | **0.55** / 4.59 | 16.4 | **25.5** / 5.8 / 210 | 6.2 | 21.7 | 86.8 |
| Learning to Fix, kNN, ε = 5 % | 93.3 | 0.72 / 5.62 | 14.8 | 30.8 / 15.6 / 194 | 7.0 | 24.2 | 87.3 |
| Learning to Fix on our self-trained model, ε = 1 % | 76.7 | 0.24 / 0.53 | 15.2 | 12.5 / 6.6 / 80 | 7.4 | 11.7 | 96.0 |
| RACLearn-style constant [0.01, 0.99], self-trained p | 96.7 | 0.05 / 0.30 | 18.6 | 6.0 / 4.8 / 38 | 5.5 | 4.9 | 93.2 |
| RACLearn-style constant [0.01, 0.99], REINFORCE p | 100 | 0.26 / 6.85 | 21.3 | 6.1 / 3.5 / 43 | 4.7 | 6.1 | 90.2 |
| RACLearn, 95 % most confident, self-trained p | 93.3 | 0.05 / 0.26 | 17.0 | 5.3 / 5.1 / 11 | 5.0 | 5.2 | 95.0 |
| RACLearn, 95 % most confident, REINFORCE p | 100 | 0.35 / 7.70 | 12.4 | 12.3 / 5.7 / 89 | 8.1 | 12.1 | 95.0 |
| **ours**: error-cost ranking + adequacy guard, 90 % | 100 | 0.026 / 0.07 | 24.8 | 4.3 / 3.3 / 20 | 4.1 | 4.3 | 90.0 |
| **ours**: error-cost + adequacy + LP guard, 95 % (combined) | 100 | 0.043 / 0.19 | 16.7 | 8.5 / 4.2 / 94 | 6.0 | 8.5 | 95.0 |
| **ours**: error-cost + adequacy + LP guard, 98 % (combined) | 100 | **0.195** / 0.66 | 10.7 | **12.4** / 8.2 / 50 | 9.4 | 11.3 | 97.6 |
| *end-to-end, no learning*: LP relaxation, 5 thresholds + block repair, 5 LPs | 100 | **0.33** / 3.55 | 4.2 | **24.0** / 16.8 / 98 | 24.2 | 19.1 | – |
| *end-to-end, no learning*: LP relaxation, threshold 0.5, 1 LP | 63.3 | 0.10 / 0.26 | 3.3 | 31.8 / 21.0 / 112 | 32.4 | 23.1 | – |
| end-to-end, ours: label-free p, 5 thresholds, 5 LPs | 96.7 | 0.65 / 8.07 | 4.3 | 23.6 / 16.4 / 96 | 23.8 | 18.4 | – |
| end-to-end, ours: self-trained p, 5 thresholds, 5 LPs | 96.7 | 0.66 / 8.34 | 4.3 | 23.4 / 16.4 / 97 | 23.5 | 18.0 | – |
| end-to-end, ours: REINFORCE p, 5 thresholds, 5 LPs | 96.7 | 1.01 / 10.62 | 4.3 | 23.7 / 16.5 / 96 | 23.8 | 18.2 | – |
| end-to-end, ours: self-trained p, threshold 0.5, 1 LP | 80.0 | 0.37 / 1.22 | 3.5 | 28.6 / 20.4 / 112 | 29.2 | 22.7 | – |
| *paper (Irish system, Gurobi): Learning to Fix, kNN, ε = 1 %* | *99.8* | *0.48 / 6.63* | *4.7* | *20.8 / – / 241* | – | – | *78.8* |

Bootstrap 95 % intervals of the means (instances): LtF-kNN ε = 1 % gap [0.31, 0.91] %, speed-up [11.7, 42.6]×; ours
at 98 % gap [0.14, 0.26] %, speed-up [8.5, 16.8]×; LP-relaxation end-to-end gap [0.11, 0.62] %, speed-up [17.7,
31.8]×. Full tables (also our earlier convention: all instances, penalty-priced costs, gap to the MILP objective)
are in `results/pglib/pglib_results.md`.

**Where the infeasible instances come from.** Every infeasible reduced problem on test is a **reserve shortfall (and
for kNN shedding) caused by OFF fixes** — units fixed off that the instance needs; one LtF-on-our-model case is a min
up / down conflict. Test instance 0 (high demand) breaks every unguarded rule except those on REINFORCE probabilities (which commit
more; 6.9–7.7 % gap there): the kNN rules (its neighbours had less demand) shed 400–2,400 MWh, LtF on our model and
the RACLearn rules on self-trained probabilities miss reserve.
Learning to Fix's guarantee is in-sample: its thresholds passed all 30 validation instances within ε, yet on test
kNN at ε = 1 % fails 1 instance and is above 1 % on 2 more, and on our sharper probabilities (96 % fixed) it fails 7
of 30. **Every rule with the adequacy guard is feasible on all 30 instances**, with no instance above 1 %.

**What carries the means.** Learning to Fix with kNN gets its 25.5× from four instances with slow full MILPs
(90–247 s) whose reduced MILPs solve in 1–3 s (74–210×; gaps 0.55–1.04 %); on the other 25 feasible instances its
median is 4.7×.
Our 98 % rule is more even (median 8.2×, ratio of mean times 9.4× vs LtF's 5.8× and 6.2×). Paired on the 29
instances where both are feasible, ours minus LtF-kNN (ε = 1 %): **−0.35 pp [−0.70, −0.12]** (ours more accurate);
ours at 95 %: −0.51 pp [−0.85, −0.28]; the no-learning LP-relaxation fixing: −0.44 pp [−0.78, −0.22].

**Time-to-quality.** HiGHS's incumbents before the end of the root are penalty-laden, so the full MILP reaches a
rule's cost only near its end: time-to-quality speed-ups are 0.7–1.0 of the plain ones (LtF-kNN 21.7× vs 25.5×;
ours 98 % 11.3× vs 12.4×) — unlike RTS-GMLC 24 h, where a 0.5 %-good incumbent came after 18 % of the solve.

**Learning vs the LP relaxation.** Our models reproduce the relaxation more than they improve on it: on test,
rounding the relaxation agrees with the aligned MILP schedule on 99.21 % of the decisions, the label-free and
self-trained models on 99.18 / 99.13 %, REINFORCE on 98.53 %, kNN on 97.18 %. Consequently the no-learning baselines
are strong: fixing the relaxation's integral values (99.2 % fixed) gives 0.105 % at 14.3× with 100 % feasibility,
and rounding + block repair + screening by 5 dispatch LPs gives 0.33 % at 24× — the best end-to-end row; every learned
end-to-end variant is worse (0.65–1.01 %): on one instance every learned schedule except the most over-committed
(threshold 0.001) keeps a reserve shortfall that the capacity-based block repair cannot see (ramp-limited reserve),
so screening falls back to a schedule 8 % above the optimum; on another the cheapest screened schedule keeps a
0.04 MWh reserve shortfall (penalty-priced, still cheaper than the slack-free alternatives), which the paper metric
counts as infeasible (96.7 %). REINFORCE (40 steps) raised the gap of every rule it
fed (with fewer infeasible instances, because it commits more).

## 7. Verdict

* **The setting is the paper's.** On the first 24 hours of the California instances (610 units, 410 free) the full
  MILP takes 100 s on average (19–565 s) with HiGHS — the regime of the paper's Irish system (60 s mean, up to 947 s
  with Gurobi). The 48-hour instances take 6 to more than 15 minutes and were not affordable here (§3).
* **Do our methods reach < 0.5 % at ≥ 20×? No — not with learning.** Our most accurate fast rule (error-cost
  ranking + adequacy + LP guard, 98 %) gives **0.195 % at 12.4×** (100 % feasible, max 0.66 %); the 95 % version
  0.043 % at 8.5×. Our learned end-to-end mode reaches 23–24× but at 0.65 % (one instance at 8 %, four to five
  above 1 %). Two rules reach ≥ 20× at or near 0.5 %: **without any learning, rounding the LP relaxation + block
  repair + 5 dispatch LPs (0.33 % at 24.0×, 100 % feasible)**, and **Learning to Fix with kNN at ε = 1 % (0.55 % at
  25.5×, 96.7 % feasible)** — the paper's 0.48 % at 20.8× within its interval [0.31, 0.91] %, i.e. the paper's
  headline reproduces on this system.
* **Do they beat faithful Learning to Fix at equal speed-up?** At equal *median* or *ratio-of-means* speed-up, yes:
  our 98 % rule is faster by both (8.2× vs 5.8×; 9.4× vs 6.2×), more accurate (paired −0.35 pp [−0.70, −0.12]),
  always feasible and never above 0.66 % (LtF: max 4.6 %, one infeasible). At equal *mean of per-instance speed-ups*,
  the paper's metric, no: LtF-kNN's 25.5× is twice ours, carried by four instances where its reduced MILP solves in
  1–3 s against a 1.5–4-minute full MILP; no rule of ours reaches 20× with a mean gap below 0.5 %. Applying the LtF
  calibration to our sharper probabilities fixes 96 % of decisions but leaves 23 % of test instances infeasible —
  the validated guarantee does not transfer; the adequacy guard does.
* **The larger finding is about the system.** California's LP relaxation is within 0.03 % (median) of the MILP and
  99 % integral, so the LP relaxation itself is the strongest "predictor": fixing its integral values (no learning)
  gives 0.105 % at 14.3×, and it beats every learned end-to-end variant. Learning matters here for *which decisions
  not to fix* (the guards and the error-cost ranking make every rule feasible) more than for predicting commitments.
  The full MILP is slow because of its size (114k columns, root cuts and heuristics), not because the commitment is
  hard to guess — the opposite of RTS-GMLC with a network (§3–§6 of RESEARCH.md), where the relaxation is weak and the
  learned rules beat the relaxation-based ones.

## 8. Limitations

* **24-hour horizon, not the instances' 48 hours.** At 48 h HiGHS needs 5.7 min (easiest profile) to > 15 min
  (wind profile, 0.20 % at the limit) per MILP on this shared machine; the study would not fit the budget. Four of the
  five profiles repeat the same day, so the first 24 h keep the structure; Scenario400 loses its second day. The
  610-unit fleet, every constraint and the 0.1 % target are kept.
* **HiGHS, one thread, on a shared 4-core machine** (load 3–5 throughout). Absolute times are noisy; the paper used
  Gurobi with 8 CPUs. In particular, a HiGHS reduced MILP has a floor of several seconds even with only a few dozen free
  binaries (presolve and root on the 114k-column model), which caps every fixing rule's speed-up; Gurobi would shift
  both the full and the reduced times.
* **Sample sizes**: 30 validation, 30 test instances, from five load profiles shared by all splits (perturbed demand,
  reserve, wind); the paper had ~524 validation and test instances from distinct days. Bootstrap intervals are in the
  JSON; differences of a few tenths of a percent between rules are not resolved.
* **Learning to Fix deviations** (for compute): relaxation MILPs capped at 20 s and K_max = 3 (the paper: 10, solved
  presumably to optimality) — every cut carried a single release set here, so the thresholds are more conservative
  than the exact algorithm's; 30 validation instances (in-sample guarantee); the labelled set is 100 self-training
  labels (0.015 % above the MILP on validation), not ~1,570 optimal schedules, so k = 50 averages half the set.
* **Hard vs soft constraints.** PGLib's model has hard balance and reserve constraints; ours prices their violation.
  For the paper metrics a reduced solution that needs shedding or reserve shortfall counts as infeasible (it would
  be infeasible in the reference model); the soft convention table keeps those instances with their penalty cost.
* **Our models**: one seed each; REINFORCE fine-tuning is short (40 steps) and on this system makes the probabilities
  worse; one self-training round; the error-cost model is trained on in-sample predictions of the self-trained model
  (its training instances), so it sees fewer errors than it meets on test.
* **The system is easy for LP-based heuristics** (tight relaxation, 99 % integral relaxation values, near-zero start-up
  costs, 1-hour minimum up and down times for 370 of the 410 free units). Conclusions about learning on this system do not transfer
  to systems with a weak relaxation (RTS-GMLC with a network, §3–§6 of RESEARCH.md).

## 9. Reproduce

```bash
# probe (48 h and 24 h) and formulation check (Pyomo only for the check, installed outside the repo)
python3 scripts/uc_pglib_probe.py --files 2014-09-01_reserves_3 Scenario400_reserves_3 --T 48 --tl 900 --workers 2
python3 scripts/uc_pglib_probe.py --files 2014-09-01_reserves_3 Scenario400_reserves_3 2015-03-01_reserves_5 --T 24 --tl 900 --workers 1
PYTHONPATH=<pyomo dir> python3 scripts/uc_pglib_validate.py --milp --out results/pglib/pglib_validate_lp.json
PYTHONPATH=<pyomo dir> python3 scripts/uc_pglib_validate.py --lp --milp ca/2015-03-01_reserves_1.json:12 rts_gmlc/2020-01-27.json:12 \
    --gap 1e-4 --tl 300 --out results/pglib/pglib_validate_milp.json
# data (train: no MILP; val / test: full MILP with incumbent log)
python3 scripts/uc_pglib_gen.py --split train --n 160 --seed 1 --mode lp --T 24 --workers 1
python3 scripts/uc_pglib_gen.py --split val --n 30 --seed 2 --mode milp --T 24 --tl 900 --gap 0.001 --workers 1
python3 scripts/uc_pglib_gen.py --split test --n 30 --seed 3 --mode milp --T 24 --tl 900 --gap 0.001 --workers 1
# models (no MILP labels)
python3 scripts/uc_pglib_train.py --stage bce --T 24 --epochs 40
python3 scripts/uc_pglib_train.py --stage st --T 24 --epochs 40 --st_n 100 --st_tl 60 --workers 1
python3 scripts/uc_pglib_train.py --stage rl --T 24 --rl_steps 40 --rl_bs 8 --rl_samples 4 --workers 1
python3 scripts/uc_pglib_train.py --stage harm --T 24 --harm_n 60 --workers 1
python3 scripts/uc_pglib_train.py --stage probs --T 24          # + kNN, + validation choice of the source
python3 scripts/uc_pglib_stcheck.py --n 20                       # quality of the self-training labels
# Learning to Fix (one process per run)
python3 scripts/uc_pglib_ltf.py --jobs knn:0.01 --workers 1 --n_val 30 --T 24 --check_tl 120 --relax_tl 20 --budget_min 60
python3 scripts/uc_pglib_ltf.py --jobs st:0.01  --workers 1 --n_val 30 --T 24 --check_tl 120 --relax_tl 20 --budget_min 60
python3 scripts/uc_pglib_ltf.py --jobs knn:0.05 --workers 1 --n_val 30 --T 24 --check_tl 120 --relax_tl 20 --budget_min 45
# test
python3 scripts/uc_pglib_eval.py --split test --start 0  --n 15 --b2b 3 --T 24 --workers 1 --skip lf_const_0.01,lf_rac_95,ltf_knn_5 --tag _a
python3 scripts/uc_pglib_eval.py --split test --start 15 --n 15 --b2b 3 --T 24 --workers 1 --skip lf_const_0.01,lf_rac_95 --tag _b
python3 scripts/uc_pglib_eval.py --split test --start 0  --n 15 --b2b 0 --T 24 --workers 1 --rules ltf_knn_5 --no_e2e 1 --tag _c
python3 scripts/uc_pglib_report.py
```
Worker pools use the spawn start method; HiGHS and PyTorch run single-threaded.
