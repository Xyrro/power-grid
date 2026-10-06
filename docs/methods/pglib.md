# Our methods and Learning to Fix on a large public system: PGLib-UC California

*Code: [`otsl/pglib.py`](../../otsl/pglib.py) (PGLib-UC MILP, LP relaxation, dispatch LP, reduced MILP, scenario
generator, repairs), [`otsl/pglib_ml.py`](../../otsl/pglib_ml.py) (adapters: per-unit Model 1, persistent dispatch LP,
LP oracle, Table II kNN, Learning-to-Fix tuner, LP guard, solver workers),
`scripts/uc_pglib_{probe,validate,gen,train,ltf,eval,report}.py`. Results:
[`results/pglib/pglib_results.md`](../../results/pglib/pglib_results.md) (+ `.json`), per-instance records
`results/pglib/pglib_eval_test*.jsonl`, probe `results/pglib/pglib_probe.jsonl`, validation
`results/pglib/pglib_validate.json`, training / tuning statistics `results/pglib/pglib_train_stats.json`,
`results/pglib/pglib_ltf_*.json`. Data (git-ignored): `data/generated/pglib_ca/{train,val,test,st_labels}.npz`.*

SUMMARY_PLACEHOLDER

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
* ramp rows that cannot bind (RU, RD ≥ Pmax − Pmin; 590 of 610 units) are dropped;
* load shedding / over-generation (10,000 $/MWh) and reserve shortfall (5,000 $/MWh) are priced slacks, so every
  commitment that satisfies min up / down has a feasible dispatch — needed by the dispatch LP used as critic and
  for label-free training. In every validation and test MILP the slacks are zero.

Provided: the full MILP (`PGModel.solve_milp`, incumbent log), the LP relaxation and the fixed-commitment dispatch LP
(`solve_dispatch`, with the demand and reserve duals), the reduced MILP (`solve_milp(z_fix=...)`) and the relaxed
reduced LP of the LP guard (`relaxed_reduced`). `PGModel` exposes the attributes of `otsl.uc.UCModel` that the reused
code touches (`A, c, integ, nv, T, off, ix, dims, _rhs_bounds, solve_dispatch`), so `otsl.ltfx`, `otsl.fixpolicy`,
`otsl.combo`, `otsl.selftrain` and `otsl.ucml` run on it unmodified. Size at T = 24: 114,289 columns (29,520 binary),
103,896 rows, 358,695 non-zeros; T = 48 doubles that.

VALIDATION_PLACEHOLDER

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
* **What HiGHS spends the time on.** Every solve ends at the root node (1–2 nodes). The LP relaxation takes 2–5 s and
  is within 0.03–0.3 % of the MILP optimum; the rest is root cut rounds and heuristics. The first incumbents are
  penalty-laden (reserve shortfall priced at 5,000 $/MWh: 10²–10³ × the optimal cost) and **the first schedule
  within 0.5 % appears only at the end of the root, 90–100 % into the solve**. Time-to-quality is therefore close to
  time-to-proof here, unlike on RTS-GMLC 24 h (§X4 of RESEARCH.md, where a 0.5 %-good schedule came after a median
  28 s of a 154 s solve).
* **The LP relaxation is very tight.** On the 30 validation instances the MILP optimum is a median 0.03 % (max 0.41 %)
  above the LP relaxation, and only 0.9 % of the relaxation's commitment values are fractional. Rounding the
  relaxation and repairing it (no learning, no MILP, ~4 s) gives a schedule a median 0.08 % (mean 0.44 %, max 3.3 %)
  above the MILP. This is a far easier fixing problem than RTS-GMLC, and it matters for every comparison below.

## 4. Data, labels and their cost

DATA_PLACEHOLDER

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

**Selection on validation only.** The probability source for Learning to Fix on our model: the lowest error rate of
the 95 % most confident decisions against the aligned validation MILP schedules — self-trained 0.505 %, label-free
0.525 %, REINFORCE 0.69 %, kNN 1.72 % → self-trained. Learning to Fix's thresholds are tuned on the 30 validation
instances. Everything else (targets 90 / 95 / 98 %, the constant thresholds, the screening grid) was fixed in advance
from the earlier studies; no test instance was read before the final evaluation.

## 6. Results (test)

RESULTS_PLACEHOLDER

## 7. Verdict

VERDICT_PLACEHOLDER

## 8. Limitations

LIMITS_PLACEHOLDER

## 9. Reproduce

REPRO_PLACEHOLDER
