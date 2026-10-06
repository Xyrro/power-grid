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

PROBE_PLACEHOLDER

## 4. Data, labels and their cost

DATA_PLACEHOLDER

## 5. Methods as transferred

METHODS_PLACEHOLDER

## 6. Results (test)

RESULTS_PLACEHOLDER

## 7. Verdict

VERDICT_PLACEHOLDER

## 8. Limitations

LIMITS_PLACEHOLDER

## 9. Reproduce

REPRO_PLACEHOLDER
