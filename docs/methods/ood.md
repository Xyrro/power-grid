# Robustness under distribution shift (B2, 12-hour network-constrained UC)

Code: [`otsl/ood.py`](../../otsl/ood.py) (shifted scenarios, outage-aware UC model, overrides),
[`scripts/uc_ood_gen.py`](../../scripts/uc_ood_gen.py) (training coverage, line pool, instance lists),
[`scripts/uc_ood_eval.py`](../../scripts/uc_ood_eval.py) (full MILP + every method back to back, one core, resumable),
[`scripts/uc_ood_report.py`](../../scripts/uc_ood_report.py) (tables, bootstrap). Results:
[`results/uc12/ood_results.md`](../../results/uc12/ood_results.md) (+ `.json`), raw records `results/uc12/ood_eval.jsonl`,
recovery test `results/uc12/ood_ft_eval.jsonl`, `ood_ft_bce_line_out.json` ([`scripts/uc_ood_finetune.py`](../../scripts/uc_ood_finetune.py)),
run log `results/uc12/ood_eval.log`. Data (git-ignored): `data/generated/uc12_ood/` (`specs.json`, `coverage_train.json`,
one `.npz` per instance with the scenario, the full-MILP schedule, the LP relaxation and every model's probabilities).

## Status

**Complete** (2026-10-06 23:30 UTC). Main run: 15:05–22:11 UTC on core 2, round-robin over the 7 instance sets;
restarted once at 15:25 after the line-outage design change (the two line instances already run were discarded) and
once at 16:17 to cut the shifted sets from 40 to **30 instances** (shifted instances took ~2 min each — full MILP plus
13 methods, reduced MILPs often slower under shift — so 40 would have needed ~9 h; the 5 finished rounds were kept).
The in-distribution set was then extended to 40. Recovery test (line outages): 50 labels 22:11–22:44, fine-tuning,
evaluation 22:45–23:29. Run order and times: `results/uc12/ood_eval.log`, `ood_chain.log`, `ood_ft_*.log`
(git-ignored logs). To reproduce or resume (finished instances are skipped):

```bash
taskset -c 2 python3 scripts/uc_ood_gen.py                  # specs, coverage (line-pool diagnostic ~11 min)
OTSL_THREADS=1 setsid nohup taskset -c 2 python3 scripts/uc_ood_eval.py --n 30 >> results/uc12/ood_eval.log 2>&1 &
OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_eval.py --n 40 --shifts id >> results/uc12/ood_eval.log 2>&1
OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_finetune.py --shift line_out --stage gen     # then --stage train
OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_eval.py --shifts line_out --n 30 \
    --bce results/uc12/ood_ft_bce_line_out.pt --knn_extra data/generated/uc12_ood/ft_line_out.npz \
    --rules "LtF kNN eps=1%,LtF BCE eps=1%,hybrid eps=1%,guarded error-cost 90%" \
    --out results/uc12/ood_ft_eval.jsonl --inst_dir data/generated/uc12_ood/ft_eval
python3 scripts/uc_ood_report.py                            # results/uc12/ood_results.{md,json}
```

Not done: 24-hour shifts (each instance needs a ~154 s full MILP plus the uc24 method stack; no budget left after the
12-hour run), seeds, shift-severity sweeps.

### Follow-up (running; plan fixed before any run)

Requested after the main study: on the same 7 instance sets, (1) Learning to Fix with the instance's LP-relaxation
values as probabilities (thresholds of the baselines study, `results/uc12/base_tune_lp_1.json`); (2) the hybrid,
LtF-BCE and LtF-kNN followed by the post-hoc LP-relaxation guard; (3) a cheap fix for collapsed thresholds: release OFF
fixings the instance's LP relaxation contradicts. Two variants, fixed a priori: *unit-hour veto* (drop the OFF fixing
of (t, g) whenever u_rel[t, g] > 1e-3, all units) and *rarely-on unit veto* (drop every OFF fixing of a unit that is on
in < 1 % of training unit-hours — 33 units, a training statistic — if its u_rel exceeds 1e-3 in any hour). Selection
on the original validation set only (`val`, 60 instances; hybrid and LtF-kNN): the lower mean validation gap to the
dataset MILP's dual bound, within 0.02 pp the faster. Full-MILP and base-rule times are reused from the main run only
after re-solving ≥ 10 of them on core 2 (2 per set) and confirming agreement within 10 % (median and ratio of means).
`scripts/uc_ood_followup.py --stage retime / val / test`; outputs `results/uc12/ood_fu_*`.

## Summary

Six shifts of the 12-hour RTS-GMLC benchmark — load +15 %, load −15 %, wind and solar × 1.5, 2–3 thermal units out,
1–2 transmission lines out (N-1-screened), and windows across midnight (a season shift is impossible: training covers
all twelve months) — 30 instances each, with full MILPs and every already-trained method run back to back on one core,
against 40 in-distribution instances. Nothing was retuned.

* **The hybrid** (Learning to Fix's calibration on our GNN error-cost scores, with adequacy and min up/down guards) is
  100 % feasible on every shift, never significantly worse than LtF-kNN and significantly better on four of six
  (paired Δ mean gap −3.7, −0.6, −0.6 and −10.2 pp under load +15 %, load −15 %, midnight windows and line outages).
* **Line outages break every LtF-calibrated rule**: LtF-kNN 26 % mean gap to the bound (in distribution 0.42 %), LtF on
  our BCE GNN and the hybrid ~16 %, 37–40 % of instances with load shedding or reserve shortfall. The tuned
  per-generator thresholds fix OFF the local peakers that the outage makes necessary. The rank-based error-cost rules
  stay at 0.47–0.73 % (90 % served with the LP-relaxation guard, MILP 93 %), the no-learning baseline at 1.0 %.
  Giving the GNN the true topology changes nothing; 50 labelled outage instances (kNN pool, GNN fine-tuning) do not remove a single catastrophic instance.
* **LtF-kNN is the more robust method under more wind and solar** (1.15 % vs 2.42 % for our guarded 95 % rule, +1.09 pp
  [0.25, 2.10], and 5.55 % for the 90 % rule) and level under unit outages. Rules that fix 90 % without the LP guard (our
  guarded 90 % rule, LtF on our BCE GNN) degrade worst on low load, more renewables and unit outages (2–7 % mean, single
  instances at 43–96 %, down to 87 % feasible).
* **The physics guard that generalises is the LP-relaxation guard**, not the copper-plate adequacy guard; with it the
  95 % rule stays ≤ 2.4 % mean gap and 100 % feasible on every shift at 6–13× mean speed-up.
* End-to-end (no MILP, 7-threshold screening): 2.7 % → 4–7.5 % mean gap, 80–100 % served, 19–30× mean speed-up.

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
noise model and its own generator seed (`scripts/uc_ood_gen.py`; per-shift seeds 101–106; 40 instances listed per
shift, the first 30 run).
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
numbers), run in the same process as the shifts (round-robin: instance k of every set before k + 1; in-distribution
instances 30–39 right after the main run), so in-distribution and shifted numbers share solver, core and machine
conditions. The full MILP is solved with highspy here; the earlier 12-hour evaluations used SciPy's HiGHS (`milp`),
which solves some instances faster, so absolute in-distribution speed-ups differ from the published ones (LtF-kNN 4.6×
on 60 instances there, 2.9× on 40 here; gaps 0.40 % vs 0.42 %); the fixings
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

Full tables (every rule, medians, maxima, served shares, our convention, paired tests, degradation with CIs, input
diagnostics): [`results/uc12/ood_results.md`](../../results/uc12/ood_results.md). In-distribution: first 40 instances of
`test_fresh`; each shift: 30 instances. Paper metrics (gap to the full MILP's dual bound, feasible instances; speed-up
mean / median of per-instance ratios, every overhead included).

**What the shifts do.** Shifted instances are farther from the training data (median distance of an instance to its
nearest training neighbour in the kNN's standardised features: 5.2 in distribution, 5.8–8.2 for load, renewable and
unit-outage shifts, 13.2 for windows across midnight; 4.6 for line outages, which the kNN cannot see). The BCE GNN's
wrong unit-hours (rounded at 0.5, against the full MILP's schedule aligned inside identical-unit groups) rise from 12.7
to 12.0–20.6 per instance; the kNN's from 31 to 38–46. Unit outages raise the LP relaxation cost by 8.8 % on average
(93 % of instances > 0.1 %), line outages by 4.9 % (90 %). The full MILP itself gets slower (26 s → 30–39 s) and hits
its 60 s limit more often, so its dual bound is weaker (gap of the MILP to its own bound 0.29 % → 0.12–0.49 %).

**Mean gap % [95 % CI], mean / median speed-up, feasible %:**

| rule | in-distribution | load +15 % | load −15 % | wind + solar × 1.5 | 2–3 units out | 1–2 lines out | across midnight |
|---|---|---|---|---|---|---|---|
| LtF, kNN, ε = 1 % (paper's setting) | 0.42 [0.27, 0.63], 2.9 / 1.6×, 100 | 4.04 [0.67, 10.06], 2.5 / 1.6×, 100 | 1.54 [0.74, 2.52], 4.9 / 1.7×, 100 | 1.15 [0.65, 1.79], 2.3 / 1.8×, 97 | 0.85 [0.46, 1.33], 2.6 / 1.3×, 100 | **26.0 [14.2, 38.8]**, 6.0 / 1.9×, 100 | 0.93 [0.60, 1.29], 4.5 / 1.4×, 100 |
| LtF on our BCE GNN, ε = 1 % | 0.25 [0.15, 0.37], 4.6 / 2.5×, 100 | 0.46 [0.20, 0.79], 5.8 / 2.7×, 100 | 4.90 [0.32, 12.0], 4.2 / 2.1×, 100 | 5.30 [1.19, 11.8], 7.3 / 5.0×, **87** | 2.17 [0.38, 5.38], 5.0 / 3.0×, 97 | 15.7 [8.8, 23.5], 5.5 / 3.5×, 100 | 0.25 [0.17, 0.34], 5.7 / 3.4×, **90** |
| **hybrid** (guard-aware LtF on error cost) | 0.31 [0.21, 0.43], 4.7 / 2.3×, 100 | **0.29** [0.15, 0.50], 3.4 / 1.9×, 100 | **0.97** [0.37, 1.81], 3.6 / 1.8×, 100 | 1.66 [0.71, 3.13], 4.3 / 2.7×, 100 | **0.82** [0.35, 1.42], 6.1 / 1.9×, 100 | 15.9 [8.8, 23.6], 7.4 / 4.8×, 100 | 0.30 [0.20, 0.42], 6.2 / 1.8×, 100 |
| guarded error-cost rule, 90 % | 0.50 [0.25, 0.81], 4.3 / 3.0×, 100 | 0.36 [0.15, 0.62], 5.2 / 2.9×, 100 | 6.72 [0.83, 14.7], 9.6 / 6.7×, 97 | 5.55 [1.14, 12.1], 8.3 / 5.7×, 93 | 3.98 [0.49, 9.49], 7.6 / 4.0×, 97 | 0.68 [0.28, 1.18], 6.5 / 3.1×, 100 | 0.89 [0.37, 1.52], 4.3 / 3.7×, 100 |
| guarded error-cost rule, 95 % (+ LP guard) | 0.63 [0.33, 1.00], 9.8 / 4.5×, 100 | 0.52 [0.23, 0.86], 6.4 / 4.2×, 100 | 2.03 [0.74, 3.54], 12.7 / 6.0×, 100 | 2.42 [1.45, 3.55], 11.3 / 7.5×, 100 | 1.00 [0.64, 1.38], 11.7 / 7.3×, 100 | **0.47** [0.26, 0.70], 7.1 / 3.9×, 100 | 0.83 [0.40, 1.42], 7.8 / 4.3×, 100 |
| combined pipeline, 98 % | 0.95 [0.58, 1.35], 11.5 / 7.7×, 100 | 0.97 [0.36, 1.79], 11.8 / 8.8×, 100 | 3.08 [1.19, 5.36], 18.8 / 10.5×, 100 | 3.29 [1.80, 5.07], 12.6 / 8.1×, 100 | 1.59 [0.80, 2.59], 14.7 / 8.4×, 100 | 0.73 [0.36, 1.22], 14.7 / 6.7×, 100 | 1.04 [0.58, 1.60], 13.5 / 9.6×, 100 |
| no learning: fix LP-integral unit-hours + guards | 1.11 [0.76, 1.55], 12.5 / 10.2×, 100 | 1.21 [0.64, 1.92], 9.9 / 6.8×, 100 | 4.23 [1.90, 6.90], 20.9 / 17.5×, 100 | 2.98 [1.97, 4.16], 16.9 / 13.5×, 100 | 1.74 [1.10, 2.47], 14.1 / 12.7×, 100 | 1.01 [0.60, 1.51], 16.4 / 10.0×, 100 | 1.12 [0.68, 1.65], 11.3 / 7.8×, 100 |
| end-to-end REINFORCE, 7-threshold screening (no MILP) | 2.68 [1.72, 3.83], 19 / 16×; served 93 | 2.67 [1.51, 4.00], 19 / 18×; 93 | 7.55 [4.47, 11.0], 28 / 33×; 97 | 6.81 [4.54, 9.42], 26 / 27×; 97 | 4.21 [2.66, 5.91], 28 / 31×; 100 | 4.17 [2.07, 7.00], 24 / 20×; 80 | 2.66 [1.78, 3.69], 23 / 21×; 93 |
| end-to-end combined, screening (no MILP) | 2.83 [1.21, 5.20], 20 / 15×; served 90 | 2.32 [1.26, 3.57], 20 / 16×; 87 | 5.86 [3.54, 8.40], 30 / 33×; 100 | 6.39 [3.69, 9.35], 28 / 32×; 80 | 4.09 [2.05, 6.71], 27 / 29×; 93 | 5.23 [2.32, 8.72], 25 / 21×; 67 | 3.01 [1.25, 5.97], 22 / 21×; 87 |

Fixed shares (available units' decisions) stay near their in-distribution values for the threshold rules (LtF-kNN
64–69 %, LtF-BCE 81–85 %, hybrid 78–86 %); the LP-relaxation guard releases more under shift (95 % target: 93.5 % fixed in
distribution, 83–93 % shifted). Served shares (no shedding / over-generation / reserve shortfall; full MILP 93–100 %)
stay within 10 pp of the MILP's for every MILP-based rule except under line outages: LtF-kNN 60 %, LtF-BCE and hybrid 63 %,
guarded 90 % 80 %, guarded 95 %, combined and no learning 90 % (MILP 93 %).

**Paired against LtF-kNN** (Δ mean gap, pp, rule − LtF-kNN on the same instances; negative = better):

| rule | in-distr. | load +15 % | load −15 % | wind + solar × 1.5 | units out | lines out | midnight |
|---|---|---|---|---|---|---|---|
| hybrid | −0.10 [−0.34, +0.07] | **−3.74 [−9.72, −0.36]** | **−0.57 [−0.98, −0.22]** | +0.44 [−0.49, +1.88] | −0.03 [−0.60, +0.62] | **−10.2 [−16.8, −4.3]** | **−0.63 [−0.98, −0.31]** |
| guarded error-cost 90 % | +0.09 [−0.22, +0.39] | **−3.67 [−9.75, −0.29]** | +5.24 [−0.25, +12.5] | *+4.59 [+0.51, +10.8]* | +3.15 [−0.07, +8.34] | **−25.4 [−38.2, −13.4]** | −0.04 [−0.66, +0.66] |
| guarded error-cost 95 % (+ LP guard) | +0.21 [−0.15, +0.57] | **−3.51 [−9.59, −0.13]** | +0.49 [−1.01, +1.95] | *+1.09 [+0.25, +2.10]* | +0.15 [−0.28, +0.54] | **−25.6 [−38.5, −13.7]** | −0.10 [−0.61, +0.46] |
| LtF on BCE GNN | **−0.17 [−0.40, −0.00]** | **−3.57 [−9.59, −0.18]** | +3.35 [−0.81, +9.74] | *+4.26 [+0.33, +11.1]* | +1.49 [−0.23, +4.66] | **−10.3 [−17.1, −4.3]** | **−0.59 [−0.98, −0.27]** |
| no learning | *+0.69 [+0.34, +1.09]* | −2.83 [−9.05, +0.70] | *+2.68 [+1.00, +4.51]* | *+1.72 [+0.73, +2.81]* | *+0.89 [+0.46, +1.35]* | **−25.0 [−38.0, −13.1]** | +0.20 [−0.36, +0.75] |

**Difference in degradation against LtF-kNN** ([mean(rule − LtF-kNN) on the shift] − [the same in distribution], pp;
negative = the rule loses less than LtF-kNN): hybrid −3.64 [−9.67, −0.23] (load +15 %), −0.47 [−0.91, −0.06]
(load −15 %), +0.54 [−0.41, +2.00] (renewables), +0.08 [−0.54, +0.78] (units out), −10.1 [−16.7, −4.2] (lines out),
−0.53 [−0.93, −0.14] (midnight); guarded 95 %: −3.72 [−9.82, −0.28], +0.28 [−1.31, +1.77], **+0.88 [+0.01, +1.99]**,
−0.06 [−0.61, +0.48], −25.8 [−38.7, −13.9], −0.31 [−0.93, +0.36]; guarded 90 %: −3.76 [−9.83, −0.35], +5.15 [−0.36,
+12.4], **+4.51 [+0.41, +10.7]**, +3.06 [−0.19, +8.22], −25.5 [−38.3, −13.4], −0.13 [−0.84, +0.68].

**Why every LtF-calibrated rule fails under line outages.** In 11 of the 30 line-outage instances LtF-kNN is more than
5 % from the bound (up to 86 %, load shed). Each of the 11 has one of the lines that feed buses 207–208 out (208–209,
208–210 or 210–212; no bus is islanded, but the import into that pocket is limited), and the full MILP then runs the two 55 MW combustion turbines at bus 207 (units 30 / 31), which are on in 0.1 % of training
unit-hours. Learning to Fix's tuning (kNN, BCE and the hybrid alike) collapsed both units' grey zones to [0.5, 0.5] —
on validation they never needed to run — so every hour with a predicted probability below 0.5 is fixed OFF. The kNN's
probability is ~0; the BCE GNN does see the need in some instances (its LP-relaxation inputs put the units on; p up to
0.99) but not in every hour, and its fixings are cut at 0.5 too. The rank-based rules leave those decisions to the
solver because their learned error cost (which reads the LP relaxation and the local prices) is high, and the
LP-relaxation guard releases whatever still forces shedding in the relaxed reduced problem. Switching off the GNN's
messages over the outaged lines ("GNN sees topology") changes nothing (hybrid +0.86 pp [0.00, 2.00], guarded 90 %
+0.02 pp): the graph structure of the GNN is not where the robustness comes from.

**Where our rules degrade more than LtF-kNN.** Lower load, more wind and solar and unit outages hurt the rules that fix
a large share without the LP guard: the guarded 90 % rule (adequacy guard only) has catastrophic instances (max 73–93 %,
reserve shortfall or shedding after a needed unit was fixed OFF in an hour the per-hour capacity check considered
covered) and 93–97 % feasibility (min up/down conflicts; this rule has no row release); LtF on the BCE GNN likewise
(87 % feasible with more renewables). The copper-plate, per-hour adequacy guard checks capacity, not deliverability
(min up/down, ramping, network). With the LP-relaxation guard (95 % rule) the tails disappear (max ≤ 19 %, 100 %
feasible), but with 1.5× wind and solar it is still 1.09 pp worse than LtF-kNN, which fixes only ~66 % and leaves the
solver the most room.

**Recovery test: does a little shifted data fix line outages?** Chosen by the rule fixed before the main run finished
(largest summed degradation of LtF-kNN and the hybrid: line outages, +25.6 and +15.6 pp). 50 line-outage instances on
*training* calendar days (seed 201, N-1-screened outages as in the test set) were solved to optimality and given to the
models: the kNN gets them in its pool; the BCE GNN is fine-tuned on 40 of them plus 200 original training instances
(early stopping on the other 10: log-loss 0.0744 → 0.0661). Thresholds, the error-cost model and the guards stay as
calibrated on the original validation set. Same 30 test instances, new back-to-back run (`ood_ft_eval.jsonl`):

| rule | before: mean gap [CI], max, served | after | Δ gap pp [CI] | instances > 10 % |
|---|---|---|---|---|
| LtF, kNN, ε = 1 % | 26.0 % [14.2, 38.8], 86 %, 60 % | 26.1 % [14.3, 38.8], 86 %, 60 % | +0.05 [−0.01, +0.14] | 11 → 11 |
| LtF on our BCE GNN | 15.7 % [8.8, 23.5], 62 %, 63 % | 11.2 % [5.7, 17.4], 58 %, 60 % (97 % feasible) | −5.06 [−8.42, −2.25] | 11 → 11 |
| hybrid | 15.9 % [8.8, 23.6], 62 %, 63 % | 11.1 % [6.1, 17.0], 58 %, 63 % | −4.80 [−7.83, −2.12] | 11 → 11 |
| guarded error-cost 90 % | 0.68 % [0.28, 1.18], 6.8 %, 80 % | 0.40 % [0.19, 0.65], 2.3 %, 87 % (97 % feasible) | −0.26 [−0.77, +0.07] | 0 → 0 |

More data does not repair the threshold rules: the kNN cannot see the topology at all, and the fine-tuned GNN lowers the
mean gap of the LtF-calibrated rules by ~5 pp but leaves all 11 catastrophic instances in place, because units 30 / 31
keep their collapsed [0.5, 0.5] grey zone and are still fixed OFF in every hour predicted below 0.5. Recovering would
need a recalibration of the thresholds on shifted validation instances (new reference MILPs and a new tuning run), not
just new training data.

## Verdict

* **The claim holds for the hybrid, not for "a GNN with physics guards" in general.** The hybrid (our BCE GNN's
  error-cost scores under Learning to Fix's ε-calibration, with the adequacy guard and the min up/down row release) is
  100 % feasible on every shift, never significantly worse than LtF-kNN, and significantly better on four of six shifts
  (load +15 %, load −15 %, windows across midnight, line outages; also in difference-in-degradation terms). Its mean gap
  stays ≤ 1.7 % except under line outages.
* **Line outages break every rule calibrated by Learning to Fix**, ours included: LtF-kNN 26 %, LtF on the BCE GNN and
  the hybrid ~16 % mean gap, 37–40 % of instances shedding load. The per-generator thresholds encode "this unit never
  needs to run" from the validation set; a topology change creates exactly that need. The **rank-based error-cost rules
  and the LP-relaxation guard are what generalise** there (0.47–0.73 %, 90 % served vs the MILP's 93 %), and so does the
  no-learning baseline (1.01 %). The GNN's message passing is not the reason: switching off the outaged lines in the GNN
  changes nothing.
* **LtF-kNN is the more robust method in some shifts, and we say so plainly.** With 1.5× wind and solar, LtF-kNN
  (1.15 %) beats every guarded rule of ours (90 %: 5.55 %, +4.6 pp [0.5, 10.8]; 95 % with LP guard: 2.42 %, +1.09 pp
  [0.25, 2.10]) and LtF on our BCE GNN (5.30 %, 87 % feasible); only the hybrid is level (1.66 %, n.s.). Under unit
  outages LtF-kNN (0.85 %) is level with the hybrid and the 95 % rule, and lower in mean than the 90 % rule (3.98 %;
  +3.15 pp [−0.07, +8.34], not significant at n = 30) and LtF-BCE (2.17 %). Fixing only ~66 % — LtF-kNN's conservative in-distribution calibration — is itself a
  robustness margin.
* **"Physics guards keep schedules feasible" needs qualifying.** Min up/down feasibility: every rule with the row
  release (hybrid, 95 %, combined, no learning) stayed 100 % feasible; rules without it lost 3–13 % of instances
  (guarded 90 %, LtF-BCE). Serving the load: the copper-plate, per-hour adequacy guard does not prevent shortfall or
  shedding caused by min up/down, ramping or the network (guarded 90 %: catastrophic instances of 73–93 % gap under low
  load, more renewables and unit outages; 80 % served under line outages). The LP-relaxation guard — exact network,
  availability and intertemporal constraints, but relaxed integrality — is the guard that holds under every shift tested.
* **Speed under shift.** Our rules stay faster than LtF-kNN on every shift (paired mean log speed-up +0.1 to +1.4 over
  it); the full MILP slows down under shift, so speed-ups against it do not collapse. The no-learning baseline (fix the
  LP relaxation's integral unit-hours, guards) is the fastest MILP-based rule (10–21× mean) and at most 4.2 % from the
  bound; a learned rule must beat it under shift too. In gap, the guarded 95 % rule does on every set and the hybrid on
  every set except line outages, both at lower speed.
* **Recovery needs recalibration, not data.** Adding 50 labelled line-outage instances (kNN pool, GNN fine-tuning)
  leaves LtF-kNN unchanged and cuts LtF-BCE / hybrid from ~16 % to ~11 %, with all 11 catastrophic instances still there;
  the in-distribution thresholds are the bottleneck.
* **End-to-end (no MILP)** degrades from 2.7 % to 4–7.5 % mean gap (7-threshold screening), keeps 80–100 % of instances
  served, and is the fastest (19–30× mean); the one-LP variants degrade more (up to 15.5 %).

## Caveats

* **Sample sizes**: 30 instances per shift and 40 in distribution (the plan was 40 per shift; at ~2 min per shifted
  instance on one core this was cut to 30 mid-run, see Status). Mean gaps are dominated by a few catastrophic instances;
  their bootstrap intervals are wide (e.g. LtF-kNN under load +15 %: 4.04 % [0.67, 10.06], one instance at 86 %).
* **One draw of each shift.** Shift magnitudes (±15 %, ×1.5, 2–3 units, 1–2 lines) were fixed in advance and not varied.
  The line-outage design was changed once, after two instances, because drawing from all lines that matter on training
  data changed the relaxation cost by ~0.01 %; the N-1-screened draw targets the outages that matter, so it is a harder
  test than random outages would be.
* **Overrides.** The learned predictors cannot see unavailable units; their predictions were overridden (unit forced
  OFF), and the solver-free guards were given the available fleet. A deployed system would do the same; it is not a test
  of a model that learned availability.
* **Reference quality.** The full MILP (60 s, 0.1 %) hits its time limit more often under shift; gaps are measured to its
  dual bound, so a weaker bound inflates every gap equally within a shift (the MILP's own gap rises from 0.29 % to up to
  0.49 %).
* **Timing.** One core, shared machine (three other agents on the other cores); the earlier studies used SciPy's HiGHS,
  this one highspy (one thread), so in-distribution speed-ups here are lower than the published ones for some rules
  (LtF-kNN 2.9× vs 4.6×) while gaps and fixings match. Compare ratios within this study.
* **Single seeds** of every model and threshold set (the hybrid is seed 0 of its family); no retraining for the main
  results, by design.
* **Not done**: 24-hour shifts (budget), seeds, shift-severity sweeps.
