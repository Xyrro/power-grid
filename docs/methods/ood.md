# Robustness under distribution shift (B2, 12-hour network-constrained UC)

Code: [`otsl/ood.py`](../../otsl/ood.py) (shifted scenarios, outage-aware UC model, overrides),
[`scripts/uc_ood_gen.py`](../../scripts/uc_ood_gen.py) (training coverage, line pool, instance lists),
[`scripts/uc_ood_eval.py`](../../scripts/uc_ood_eval.py) (full MILP + every method back to back, one core, resumable),
[`scripts/uc_ood_report.py`](../../scripts/uc_ood_report.py) (tables, bootstrap). Results:
[`results/uc12/ood_results.md`](../../results/uc12/ood_results.md) (+ `.json`), raw records `results/uc12/ood_eval.jsonl`,
run log `results/uc12/ood_eval.log`. Data (git-ignored): `data/generated/uc12_ood/` (`specs.json`, `coverage_train.json`,
one `.npz` per instance with the scenario, the full-MILP schedule, the LP relaxation and every model's probabilities).

## Status

*Kept current for resumption.* Main run started 2026-10-06 15:05 UTC on core 2 (7 instance sets × 40 instances,
round-robin; ~65 s per instance on average, ~5 h); restarted 15:25 after the line-outage design change (the two
line instances already run were discarded, every other finished instance was kept). To resume after a restart (finished instances are skipped):

```bash
OTSL_THREADS=1 setsid nohup taskset -c 2 python3 scripts/uc_ood_eval.py --n 40 >> results/uc12/ood_eval.log 2>&1 &
python3 scripts/uc_ood_report.py            # any time; uses the instances finished so far
```

Planned after the main run (rule fixed before its results were complete): recovery test on the one shift with the
largest sum of mean-gap degradations (shift − in-distribution) of LtF-kNN and the hybrid: 50 labelled shifted instances
on training days, BCE GNN fine-tuned / kNN pool augmented, same 40 test instances (`scripts/uc_ood_finetune.py`).
Not planned: 24-hour shifts (each instance needs a ~154 s full MILP plus the uc24 method stack; does not fit the budget
after the 12-hour run).

## Summary

(pending)

## Shifts

**What the training distribution covers** (`data/generated/uc12_ood/coverage_train.json`). The 500 training instances
come from 196 calendar days of 2020 with day % 5 ∈ {1, 3, 4}, i.e. **all twelve months** (validation: day % 5 = 2, test:
day % 5 = 0, also all months). Every split uses the same generator: a 12-hour window starting at **00:00–12:00** (no
window crosses midnight), area loads × U[0.92, 1.08], bus loads × U[0.97, 1.03], each renewable unit × U[0.8, 1.2],
reserve 3 % of load, priority-list initial status. Training loads span 2,764–8,160 MW (thermal fleet 8,076 MW; annual
peak of the profiles 8,192 MW), net load −2,035–6,425 MW, wind up to its 2,508 MW installed capacity, 0–23 units
running at the start. All 73 units are always available and the network is always intact.

A **season shift is therefore not possible** with this generator: every month is in training. It is replaced by a
time-of-day shift (windows across midnight), which no training instance has.

Every shifted instance uses a **test calendar day** (day % 5 = 0, never a training or validation day), the unchanged
noise model and its own generator seed (`scripts/uc_ood_gen.py`; per-shift seeds 101–106, 40 instances each).
`otsl.ood.make_ood_scenario` draws the same random numbers in the same order as `otsl.uc.make_scenario` and returns
exactly its instance when no shift is applied (checked on 5 seeds).

| shift | exactly what changes | why it is out of distribution |
|---|---|---|
| **load +15 %** (`load_hi`) | area-load multiplier U[0.92, 1.08] × 1.15 = U[1.058, 1.242]; the reserve requirement (3 % of load) and the priority-list initial status are computed from the scaled load | loads up to ~9.4 GW exceed the training maximum and the thermal fleet; shedding / reserve shortfall become possible for every method, the MILP included |
| **load −15 %** (`load_lo`) | multiplier × 0.85 = U[0.782, 0.918] | loads below the training minimum, deeper negative net load (curtailment, few units) |
| **wind + solar × 1.5** (`ren_hi`) | installed capacity *and* availability of every wind, PV and rooftop-PV unit × 1.5 (5.2 → 7.8 GW); hydro unchanged | renewable output and negative net load beyond anything in training; more cycling |
| **2–3 units out** (`gen_out`) | 2 or 3 units (equal odds) drawn uniformly from the units running at the start of the window (u0 = 1) trip at the window start and are unavailable for all 12 hours: u = v = 0 forced in every LP and MILP. u0 is unchanged (the units were running) | no training instance has an unavailable unit; the predictors see u0 = 1 for the tripped units |
| **1–2 lines out** (`line_out`) | 1 or 2 lines (equal odds) out for the horizon: flow fixed to 0 and its angle equation dropped (checked: identical LP relaxation to a model rebuilt without the line). Drawn at random from the 4 lines whose single outage raises the instance's LP relaxation cost most (N-1 screening of its 20 most loaded lines on the intact network; `otsl.ood.with_loaded_line_outage`), never islanding a bus. A first design drew from the 66 lines that matter on some training instance; on the first instances that, and the most loaded lines, changed the relaxation cost by only 0.001–0.7 % (RTS-GMLC is meshed), so it was replaced before the line set was run (2 instances discarded) | no training instance has a topology change |
| **windows across midnight** (`night`) | window start uniform in 13:00–23:00 (training: 00:00–12:00); hours after midnight come from the next calendar day's profiles (the test day's successor, a training day: the window as a whole is new, its morning profile is not) | evening peak, ramp-down and night-to-morning shapes at window positions never seen; the kNN's hour features take unseen combinations |

## Setup

**Methods** (all trained and calibrated earlier on the original data; nothing was re-tuned or retrained for a shift):

| rule | what it is | source |
|---|---|---|
| LtF, kNN, ε = 1 % | faithful Learning to Fix: kNN (k = 50, Table II features) + generator thresholds tuned by Algorithms 1–2 | [`ltfx.md`](ltfx.md), `ltfx_tune_knn_1.json` |
| LtF on our BCE GNN, ε = 1 % | the same calibration on the MILP-label BCE GNN (`uc_model1_4.pt`) | `ltfx_tune_bce_1.json` |
| hybrid | guard-aware LtF on our error-cost scores (BCE GNN, harm seed 0), ε = 1 %, 360 validation instances, adequacy guard + min up/down row release | [`hybrid.md`](hybrid.md), `hybrid_tune_he_bce_s0_e1_n360.json` |
| guarded error-cost rule, 90 % | 90 % of the decisions with the lowest learned error cost fixed + adequacy guard | [`fixpolicy.md`](fixpolicy.md), [`combo.md`](combo.md) |
| guarded error-cost rule, 95 % | 95 % target + adequacy guard + min up/down row release + LP-relaxation guard | [`combo.md`](combo.md) |
| combined pipeline, 98 % | self-trained GNN (`combo_st_s0.pt`) + error cost + adequacy + rows + LP guard | [`combo.md`](combo.md) |
| no learning | fix every unit-hour whose LP-relaxation value is exactly 0 or 1 (98 % of them in distribution) + adequacy guard + row release | new |
| end-to-end REINFORCE | label-free + REINFORCE GNN (`constrained_rl_lf.pt`), threshold 0.5, block adequacy + min up/down repair, 1 dispatch LP; *screening*: the 7 thresholds 0.3–0.9, best by the exact LP | [`constrained.md`](constrained.md), [`combo.md`](combo.md) |
| end-to-end combined | Lagrangian + KL model (`constrained_lag_D.pt`), val threshold 0.6, block repair, 1 LP; screening as above | [`combo.md`](combo.md) |
| line outages only: "GNN sees topology" | hybrid and guarded 90 % with the BCE GNN's messages over the out-of-service lines switched off (edge gate 0; the GNN was trained with all gates 1) | `otsl.ood.predict_gated` |

**What each component sees of a shift.**

| component | load / renewables | unit availability | network topology |
|---|---|---|---|
| kNN (LtF) | system aggregates (Table II: load, wind, solar, hydro, net load, their shapes, hour of day) | no (only the initial status, which shows the tripped units as running) | no |
| GNNs (BCE, self-trained, REINFORCE, Lagrangian) | bus loads and renewables, net load per period | only through the LP-relaxation features (relaxed commitment 0, prices) | message passing on the nominal graph; the outage only through the LP-relaxation features (prices, line loadings 0). Variant: edge gate 0 |
| error-cost model | features of the probabilities, the LP relaxation and system adequacy | through the relaxation features | through relaxed prices |
| adequacy guard, block adequacy repair | net load + reserve | **yes**: they are written against system data and get the available fleet (`otsl.ood.avail_view`) | no (copper plate) |
| min up/down row release / repair | – | – | – |
| LP-relaxation guard, no-learning baseline, dispatch LPs, MILPs | exact | exact | exact |

**Overrides for outages** (`otsl.ood`). Unavailable units are forced off (u = v = 0) in the full MILP, every reduced MILP,
the LP relaxation and every dispatch LP (`OODModel`). The learned predictors do not see availability, so the
on-probabilities of unavailable units are overridden to 0 before any rule or decoder, and their fixings are forced to
OFF after the guards (the guards may release them). The fixed share is reported over the decisions of available units
only. Line outages need no override: every LP / MILP has the true topology, and the predictors simply do not see it
(except through the relaxation features, and the gated-GNN variant).

**Protocol** (`scripts/uc_ood_eval.py`). One process pinned to one core (`taskset -c 2`), HiGHS through highspy with
one thread, PyTorch single-threaded; the machine was shared with three other agents' jobs on the other cores. Per
instance, back to back: full MILP (60 s, 0.1 % gap) → LP relaxation (timed) → probabilities of every model (each
timed) → for each rule: fixings, guards, reduced MILP (60 s, 0.1 %) → end-to-end decoders and their dispatch LPs.
Method time = reduced MILP (or dispatch LPs) + inference + LP relaxation (for every rule that uses it: all GNN-based
rules and the no-learning baseline, not the kNN) + guards (incl. their LPs) + decoding. The in-distribution reference
is the first 40 instances of `test_fresh` (the stored scenarios, i.e. a subset of the instances behind the published
numbers), run in the same process and order as the shifts (round-robin: instance k of every set before k + 1), so
in-distribution and shifted numbers share solver, core and machine conditions. The full MILP is solved with highspy
here; the earlier 12-hour evaluations used SciPy's HiGHS (`milp`), which solves some instances faster, so absolute
in-distribution speed-ups differ from the published ones (e.g. LtF-kNN 4.6× on 60 instances there); the fixings
reproduce the published ones exactly (checked on instance 0: identical fixed shares and objectives for every rule).

**Metrics.** Learning to Fix's (paper metrics): gap = (C − DB) / C against the back-to-back full MILP's dual bound DB,
mean / median / max over feasible instances; feasibility rate (a reduced MILP without solution, i.e. fixings in
conflict with min up/down times, is infeasible); speed-up T_MILP / T_method per instance, mean and median over feasible
instances; fixed share. Added: served share (no shedding, over-generation or reserve shortfall; all instances, an
infeasible one counted with the full MILP's schedule), and our convention (gap to the full MILP's objective over all
instances, infeasible → full-MILP fallback, which pays both times). End-to-end rows never solve a MILP and are always
feasible (all constraints that can fail are soft and priced); their quality shows in the gap and the served share.

**Statistics** (`scripts/uc_ood_report.py`, 2,000 bootstrap resamples of instances): CI of each mean gap; paired
comparisons against LtF-kNN on the instances both solve (Δ gap in pp, Δ log speed-up); degradation = shift −
in-distribution with the two instance sets resampled independently; and the **difference in degradation** against
LtF-kNN: [mean(rule − LtF-kNN) on the shift] − [the same in distribution], each paired within its set — negative means
the rule loses less than LtF-kNN under the shift.

**No tuning on shifted data.** Thresholds, fixing targets, guard margins, decoders and models are those fixed on the
original validation set; the line pool and the instance lists were fixed before any method ran on a shifted instance,
from training instances (line pool) and seeds only.

## Results

(pending)

## Verdict

(pending)

## Caveats

(pending)
