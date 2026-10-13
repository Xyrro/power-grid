# hard: a benchmark with a weaker LP relaxation

Code: [`otsl/hard.py`](../../otsl/hard.py), `scripts/uc_hard_*.py`. Results:
[`results/hard/hard_results.md`](../../results/hard/hard_results.md) (+ `.json`), screening records
`results/hard/hard_screen.jsonl`. Data (git-ignored): `data/generated/hard/`.

## Status

**Complete** (19:55 UTC). Every step below finished; results in `results/hard/hard_results.{md,json}`, screening in
`results/hard/hard_screen.{md,json}`. Nothing runs. To rebuild the tables: `python scripts/uc_hard_screen_report.py`
and `python scripts/uc_hard_report.py`.

Run history (all times UTC, cores 2 and 3, one single-threaded process per core):

- 06:47–08:12 screening (`scripts/uc_hard_screen_q.sh`, `scripts/uc_hard_screen_q2.sh`, 12 h windows added at 08:00;
  records `results/hard/hard_screen.jsonl`). The first choice, 24 h × 0.6, was dropped at 08:05 after its first 5
  validation MILPs took 61–534 s (median 392 s; archived in `data/generated/hard/screen24/`).
- 08:15–09:53 data (`scripts/uc_hard_q.sh data2|data3`, then `run2|run3`): 60 validation + 60 test instances with the
  full MILP, 3,000 LP-only training scenarios (the first 2,000 are used). 09:46 teacher t0; 09:51 polishing pilot.
- 09:55–11:53 (`RATIO=0.9 TL=5 scripts/uc_hard_q.sh run2b|run3b`): LtF-LP tuning (`lp`), 2,000 polished labels,
  guard-aware LtF on the LP values (`lp_g`), final members fin_s0–s2.
- ~12:00 the session stopped (API limit) and the container restarted several times; reruns of run2b / run3b at
  12:50 and 15:20 died with it. 15:38 resumed with `scripts/uc_hard_q.sh run2c|run3c`: fin_s3, fin_s4, ensemble,
  `hg_fin` tuning (core 2); `knn` tuning and the paired original-rating check (`scripts/uc_hard_paired.py`, core 3);
  test evaluation in two shards from 16:28 / 17:25, plus a helper on core 2 from 19:05 that took shard 1's last
  instances (`scripts/uc_hard_eval.py --shard 1 --nshards 2 --reverse --out results/hard/hard_eval_s1b.jsonl`; the
  evaluation skips instances finished in any `hard_eval_s*.jsonl`). 19:52 all 60 test instances done (780 records, no
  duplicates).
- Resume, if ever needed: delete `data/generated/hard/*.lock` and rerun `scripts/uc_hard_q.sh run2c` / `run3c`;
  finished chunks, models, converged tuning runs, paired and test instances are skipped.

## Summary

**Benchmark.** The screened settings were RTS-GMLC with line ratings derated to 0.8 / 0.7 / 0.6, higher reserves, a
48 h horizon and PGLib-UC FERC. The weakest LP relaxation with a full MILP in the 30 s – 5 min range is **RTS-GMLC
12 h with line ratings at 60 %**. On its 60 test instances the LP relaxation is 0.63 % below the optimum (median; mean
0.74 %), with 4.4 % of the commitment values fractional and 8.6 wrong integral values per instance. The full MILP
takes a median 34 s at 0.1 % (data generation). On the same 60 validation scenarios at the original ratings, the
median LP gap to the dual bound is 0.24 % against 0.61 % derated (derated weaker on 44 of 60) and the derated
relaxation has 2.4× the fractional values. The mean gap is unchanged (0.73 % vs 0.74 %), because the original ratings
have rare large outliers. FERC is weak on only 2 of 9 days and its MILPs are slower; 24 h × 0.6 is as weak as 12 h ×
0.6 but its MILPs take 6–9 minutes.

**Result: here learning pays, in accuracy.** No full MILP is used for training: a teacher is trained on label-free
labels, 2,000 labels are polished with one 5-s reduced MILP each, and a 5-member temporal-GNN ensemble is trained on
them. The pipeline then applies guard-aware Learning to Fix thresholds and the LP-relaxation guard, and solves the
reduced MILP at a 0.5 % gap. On 60 test instances it fixes 84 % of the decisions and reaches **0.32 % [0.26, 0.39]
(max 1.13 %) at 11.0× mean speed-up (geometric mean 6.8×), 100 % feasible**. Every learning-free fixing rule is less
accurate on the same instances, by 0.16 pp or more:
- the same pipeline on the LP values ("LP pipeline"): 0.49 % (max 3.05 %) at 14.0× (geo 8.6×); the learned pipeline
  is −0.16 pp [−0.32, −0.03] in gap and 0.79× as fast;
- faithful Learning to Fix on the LP values: 0.49 % (max 2.86 %) at 15.1×, but 4 of 60 reduced problems infeasible;
- fixing the integral LP values with guards: 0.81 % (max 5.57 %) at 11.4×.

LtF-kNN gives 0.52 % at 8.1× (the pipeline is −0.20 pp [−0.35, −0.05] and 1.45× faster). Against the solver
alone, the pipeline at 0.5 % is 1.52× faster than the full MILP at a 1 % gap (Δ log speed-up +0.42 [+0.24, +0.61]) and
statistically as accurate (+0.07 pp [−0.01, +0.14]); it is 0.12 pp less accurate and 2.75× faster than the full MILP
at 0.5 %. To reach the pipeline's cost, the full MILP needs 2.5× the pipeline's time (geometric mean).

**Contrast with the original-rating benchmark.** On uc12 (original ratings) Learning to Fix on the LP values was as
accurate as the learned rules (0.27 % vs the m1x pipeline's 0.30 %; [`base.md`](base.md), [`m1x.md`](m1x.md)). With
derated lines, the LP-based rules lose accuracy (0.49–0.81 % and heavy tails) while the learned pipeline keeps it
(0.32 %). The learned probabilities are not much more often right than the rounded LP (18.8 vs 20.2 wrong decisions
per validation instance). The gain comes from their **confidence ranking**: 94 % of the decisions can be fixed at
99 % precision by the ensemble's confidence, against 0.3 % by the LP's, whose integral values (95.5 % of all) include
the wrong ones. As a result, Learning to Fix's validated thresholds fix 84 % with the ensemble and 63–65 % with the LP
values at the same ε = 1 %. The price: the LP-based rules are faster per instance (geometric mean 8.2–9.0× vs 6.8×) despite
fixing less. Their fixings agree with the LP relaxation, which apparently leaves an easier reduced MILP; we did not
investigate this further. They pay for the speed with the larger gaps.

## 1. Question and screening

The no-learning baselines of [`base.md`](base.md) left little room for learning on RTS-GMLC (98 % of the LP-relaxation
commitment values integral, median LP gap 0.19–0.26 %) and PGLib California (0.03 %): fixing the integral LP values
or running Learning to Fix on the LP values matched the learned rules. The question here: **on a benchmark with a
weaker LP relaxation, does learning help where the LP alone does not?**

Screening (`scripts/uc_hard_screen.py`, one process per setting, peak memory = its maximum resident set; summary
[`results/hard/hard_screen.md`](../../results/hard/hard_screen.md)). Per instance: LP relaxation, full MILP (HiGHS 1.15
through highspy, one thread, mip_rel_gap 0.1 %, bound trace), relaxation statistics against the MILP schedule
(`otsl.hard.relax_stats`; wrong integral values are counted after aligning the MILP schedule over identical units),
served share, and for RTS-GMLC the line loading of the MILP schedule's dispatch LP. Candidates:

* RTS-GMLC 24 h, networked, continuous ratings × 1.0 / 0.8 / 0.7 / 0.6 (`load_rts_gmlc(line_scale=...)`), the same 4
  training-day scenarios in every setting;
* RTS-GMLC 24 h with a 6 % and a 10 % spinning-reserve requirement, and a 48 h (two-day) horizon
  (`otsl.hard.make_scenario_h`);
* PGLib-UC FERC (934 units, 872 free in the winter fleet), first 24 h, copper plate, 9 files (6 winter high-wind,
  2 winter low-wind, 1 summer high-wind);
* after the 24 h choice turned out too slow (below): RTS-GMLC 12 h windows (random start, as the uc12 benchmark) at
  ratings × 1.0 and × 0.6, 6 windows.

(Screening results and the choice: see §2.)

## 2. Screening results and the chosen benchmark

Screening table: [`results/hard/hard_screen.md`](../../results/hard/hard_screen.md). LP gap = (DB − LP) / DB to the
full MILP's dual bound (median / mean over the screened instances); MILP = full MILP at 0.1 % on one thread.

| setting | n | LP gap to DB median / mean % | integral % | MILP s (median / max) | verdict |
|---|---|---|---|---|---|
| RTS-GMLC 24 h, original ratings | 4 | 0.08 / 0.20 | 97.8 | 40 / 300 (limit) | reference |
| 24 h, ratings × 0.8 | 4 | 0.14 / 0.20 | 97.2 | 36 / 70 | barely weaker |
| 24 h, ratings × 0.7 | 4 | 0.24 / 0.37 | 96.7 | 102 / 251 | weaker |
| 24 h, ratings × 0.6 | 4 (+5 val.) | 0.40 / 0.55 (val. 0.37 / 0.46) | 96.2 | 61 / 78 (val. 392 / 534) | weaker, but the MILP takes 6–9 min on most validation days |
| 24 h, reserve 6 % / 10 % | 3 / 3 | 0.25 / 0.23, 0.32 / 0.31 | 97.3 / 97.1 | 176 / 300, 300 / 300 | slower, not weaker; 10 % has reserve shortfall |
| 48 h (two days) | 1 | 0.30 | 98.3 | 600 (limit, 0.75 % gap) | too slow |
| PGLib-UC FERC, first 24 h | 9 | 0.16 / 0.41 | 96.8 | 292 / 600 (limit) | weak on 2 of 9 files (Jan, Oct high wind: 1.75 %, 1.10 %), 0.02–0.25 % on the rest; root-node-dominated MILPs of 2.5–10 min, up to 3 GB |
| RTS-GMLC 12 h, original ratings | 6 | 0.32 / 0.97 | 98.0 | 29 / 198 | reference (one 4.6 % outlier) |
| **12 h, ratings × 0.6** | 6 | 0.58 / 0.91 | 95.9 | 22 / 255 | **chosen** |

**Why 12 h × 0.6.** It has the weakest relaxation by median and by integral share among the settings whose full
MILP stays in the 30 s – 5 min range. The 24 h × 0.6 setting has a similar gap but its MILP takes a median 392 s on
validation days, which would put 120 reference MILPs plus the evaluation's re-timing well over the budget. FERC is
weak only on 2 of 9 days and its MILPs are slower. The stress settings (reserve, 48 h) make the MILP slower without
weakening the relaxation.

**On the generated sets** (`results/hard/hard_results.md`; 60 validation + 60 test instances): LP below the MILP
objective mean 0.83 / 0.74 %, median 0.71 / 0.63 %; below the dual bound median 0.61 / 0.54 %; 95.5 % of the
relaxation values integral; 40 fractional values and 9–10 wrong integral values per instance (876 decisions); full
MILP median 34 s, mean 66–73 s, one instance per set at the 600 s limit. The stored original-rating 12 h sets
(uc12, `data/generated/uc12/{val,val_extra,test_fresh}.npz`) have a median gap to C of 0.23–0.40 % (mean 0.61–0.79 %,
max 4.6–6.9 %) and 98.2–98.3 % integral values: the derated benchmark roughly doubles the median gap and the
fractional share, but not the mean. The paired check (§5.1) confirms this on identical scenarios.

**Legitimacy.** RTS-GMLC is lightly congested. With the original continuous ratings only 0.36 % (24 h) / 0.46 %
(12 h) of line-hours bind (1.7–2.5 lines per instance), and 3.3 % / 4.4 % of line-hours load above 60 % of the rating.
At 60 % of the ratings, 1.5 % / 1.7 % of line-hours bind (6.5–6.7 lines per instance), so congestion becomes a regular
part of every day, while load is still fully served: no shedding in any validation or test MILP, and a reserve
shortfall of 0.3 MWh on 2 of 60 validation instances. Scaling line limits to create congestion is common practice
for test systems that are otherwise uncongested. The benchmark keeps the RTS-GMLC data, the uc12 scenario generator
and the 3 % reserve unchanged; only the ratings change.

## 3. Methods (everything validation-tuned; nothing chosen on test)

All rules end in a reduced MILP (HiGHS 1.15 through highspy, one thread, 600 s limit) at mip_rel_gap 0.1 % or 0.5 %.

| rule | probabilities / score | thresholds | guards | learning |
|---|---|---|---|---|
| full MILP | – | – | – | none |
| LtF-LP | LP-relaxation values u_rel | Learning to Fix, faithful (otsl.uc24ltf.StartLtFTuner), ε = 1 % | none | none |
| LP-int+g | every integral u_rel value fixed (1e-6) | – | adequacy + minimal min up/down release + LP-relaxation guard | none |
| LP pipeline | u_rel | guard-aware Learning to Fix (otsl.hard.HardTuner), ε = 1 % | as LP-int+g | none |
| LtF-kNN | eq. 13, k = 50, Table II features, training scenarios with polished labels | faithful, ε = 1 % | none | kNN |
| pipeline | 5-member temporal-GNN ensemble on polished labels | guard-aware (HardTuner), ε = 1 % | as LP-int+g | GNN |

* **Guards** (`otsl.hard.guard_chain`, copied from otsl.m24): adequacy guard (release OFF fixings in merit order until
  the units not fixed off cover net load + reserve + 5 % every hour), minimal min up/down conflict release (per unit,
  release only the fixings that the nearest min up/down-feasible row contradicts), and the LP-relaxation guard
  (release fixings in hours where the relaxed reduced problem sheds, over-generates or misses reserve).
  Guard-aware tuning applies the first two inside Learning to Fix's check, cuts and verification.
* **LP pipeline** is the learned pipeline with the LP relaxation in place of the network: it isolates what the
  learned probabilities add over the LP.
* **Learned pipeline** (the m1x recipe without any full-MILP training label): (1) teacher t0 = 3 temporal GNNs
  (otsl.m1x.CommitGNNT) trained on 2,000 label-free labels (rounded LP relaxation → block adequacy repair → min up/down
  repair); (2) polished labels: for each of 2,000 training scenarios, fix the teacher's 90 % most confident decisions,
  apply the three guards, solve the reduced MILP for 5 s, keep the schedule if the dispatch LP prices it below the
  label-free label (setting chosen on validation instances 0–19, `pilot.json`); (3) final model = 5 temporal GNNs
  trained on the polished labels (BCE, 40 epochs, early stopping on the validation log-loss), mean probability;
  (4) guard-aware Learning to Fix thresholds on validation; (5) at test: LP relaxation → ensemble → thresholds →
  guards incl. LP-relaxation guard → reduced MILP at 0.5 %.

## 4. Setup

* Data (`scripts/uc_hard_data.py`, `otsl.b3.instance`): scenarios of calendar days by the split of
  `scripts/uc_gen.py` (test: day % 5 = 0, validation: day % 5 = 2, training: the rest), 12-hour windows with a random
  start, load ×U(0.92, 1.08) per area and ×U(0.97, 1.03) per bus, renewables ×U(0.8, 1.2) per unit, 3 % spinning
  reserve, initial status by priority list. Training: 3,000 scenarios, LP relaxation + label-free labels only (no
  MILP); validation 60 and test 60 with the full MILP (0.1 %, 600 s, bound trace every second).
* Evaluation (`scripts/uc_hard_eval.py`): every run of a test instance back to back in one single-threaded process,
  shards on cores 2 and 3; the full MILP at 0.1 % is re-solved there (speed-up denominator T_ref and dual bound DB).
  Method time = reduced MILP + LP relaxation (if used) + inference + thresholding + guards.
* Metrics (`scripts/uc_hard_report.py`): Learning to Fix's paper metrics (feasibility, gap (C − DB) / C over
  feasible instances, mean of per-instance speed-ups, fixed share) plus median / geometric-mean speed-up, ratio of
  mean times, served share, time to the same quality (the reference full MILP's time to first reach the method's
  cost, over the method's time), paired bootstrap 95 % CIs (2,000 resamples, seed 0) for every comparison.

## 5. Results

Full tables: [`results/hard/hard_results.md`](../../results/hard/hard_results.md) (+ `.json`).

### 5.1 The relaxation, paired with the original ratings

60 validation scenarios solved at both ratings (identical load, renewables, reserve and initial status;
`scripts/uc_hard_paired.py`). Differences are derated − original, with instance-bootstrap 95 % CIs.

| quantity | original ratings | ratings × 0.6 | difference |
|---|---|---|---|
| LP below the dual bound: mean / median / max % | 0.73 / 0.24 / 6.98 | 0.74 / 0.61 / 2.85 | mean +0.00 pp [−0.30, +0.25]; median of differences +0.26 pp; derated weaker on 44 / 60 |
| LP below the MILP objective: mean / median % | 0.84 / 0.32 | 0.83 / 0.71 | |
| integral relaxation values % | 98.1 | 95.5 | |
| fractional values / instance (of 876) | 16.4 | 39.6 | +23 [+18, +28] |
| wrong integral values / instance | 7.2 | 9.7 | +2.5 [−0.1, +4.9] |
| not fully served (reserve shortfall 0.3–1.1 MWh, no shedding) | 4 / 60 | 2 / 60 | |

Derating makes the relaxation weaker on most instances: more fractional values, a larger median gap and more wrong
integral values. But the original-rating 12 h windows already have a heavy tail of weak relaxations, so the mean gap
is the same. (MILP times are not comparable between the two columns: the derated runs predate the container
restarts, after which the host re-timed identical MILPs about 1.5× slower.)

### 5.2 Labels, probabilities, tuning (validation)

* Label-free labels (rounded LP → block adequacy repair → min up/down repair) are poor on this benchmark: a median
  67 % above the MILP objective on validation instances 0–19, because the rounded relaxation sheds load behind
  congested lines. Polished labels (teacher's 90 % most confident decisions fixed, guards, 5-s reduced MILP) are a
  median 0.31 % (mean 0.63 %) above the optimum and agree with the MILP schedule on 97.2 % of the decisions; 10 s
  gives the same labels, and fixing 80 % is worse. 2,000 polished labels cost 2.8 core-hours (5.1 s each).
* Probability quality against the validation MILP schedules (aligned over identical units):

| source | log-loss | AUC | wrong decisions / instance | fixable at 99 % precision | at 99.9 % |
|---|---|---|---|---|---|
| LP relaxation (no learning) | 0.176 | 0.976 | 20.2 | 0.3 % | 0.3 % |
| LtF-kNN (k = 50, polished labels) | 0.135 | 0.984 | 33.3 | 86.2 % | 0.0 % |
| teacher t0 (3 GNNs, label-free labels) | 0.089 | 0.987 | 21.1 | 90.5 % | 21.0 % |
| final ensemble (5 GNNs, polished labels) | 0.082 | 0.990 | 18.8 | 94.1 % | 23.3 % |

* Learning to Fix (ε = 1 %, all four runs converged, verification within ε on every validation instance): validated
  fixed share 84.1 % for the learned ensemble (guard-aware), 64.6 % for the LP values (faithful), 62.5 % for the LP
  values (guard-aware), 56.0 % for kNN (faithful).

### 5.3 Test (60 instances, paper metrics)

Gap = (C − DB) / C to the dual bound of the full MILP at 0.1 % re-solved in the same process; speed-up against that
run's time (mean T_ref 96 s, median 52 s), every overhead included.

| rule | gap | feasible % | gap mean % [95 % CI] | max | speed-up mean [95 % CI] | median | geo. mean | time to the same quality (geo.) | fixed % |
|---|---|---|---|---|---|---|---|---|---|
| full MILP | 0.1 % | 100 | 0.10 [0.09, 0.10] | 0.27 | 1.0 | 1.0 | 1.0 | – | 0 |
| full MILP | 0.5 % | 100 | 0.20 [0.17, 0.23] | 0.55 | 3.2 [2.6, 4.0] | 2.3 | 2.5 | 0.6 | 0 |
| full MILP | 1 % | 100 | 0.26 [0.21, 0.31] | 0.82 | 6.1 [4.7, 7.7] | 3.7 | 4.4 | 0.9 | 0 |
| LtF-LP | 0.5 % | 93 | 0.49 [0.37, 0.62] | 2.86 | 15.1 [10.4, 20.7] | 9.9 | 9.0 | 2.4 | 65.0 |
| LP-int+g | 0.5 % | 100 | 0.81 [0.57, 1.09] | 5.57 | 11.4 [9.0, 14.0] | 8.7 | 8.2 | 1.7 | 95.5 |
| LP pipeline | 0.5 % | 100 | 0.49 [0.36, 0.63] | 3.05 | 14.0 [10.4, 18.7] | 7.6 | 8.6 | 2.4 | 62.8 |
| LtF-kNN | 0.5 % | 98 | 0.52 [0.36, 0.68] | 2.65 | 8.1 [5.5, 11.5] | 4.8 | 4.7 | 1.6 | 56.4 |
| **pipeline** | **0.5 %** | **100** | **0.32 [0.26, 0.39]** | **1.13** | **11.0 [8.2, 14.0]** | **6.7** | **6.8** | **2.5** | **84.1** |
| pipeline | 0.1 % | 100 | 0.29 [0.24, 0.36] | 1.13 | 5.0 [3.7, 6.5] | 3.3 | 3.4 | 1.4 | 84.1 |

(Every fixing rule at 0.1 % is in the full table; each is slower than at 0.5 % with about the same gap.)

Paired comparisons of the learned pipeline at 0.5 % (instances both solve):

| reference | n | Δ gap pp [95 % CI] | time ratio (geo.; > 1: pipeline faster) [CI of Δ log speed-up] | pipeline better / worse |
|---|---|---|---|---|
| LP pipeline @ 0.5 % | 60 | −0.163 [−0.319, −0.034] | 0.79× [−0.49, −0.02] | 29 / 23 |
| LtF-LP @ 0.5 % | 56 | −0.157 [−0.303, −0.034] | 0.75× [−0.57, −0.02] | 28 / 20 |
| LP-int+g @ 0.5 % | 60 | −0.485 [−0.757, −0.261] | 0.83× [−0.37, +0.01] | 37 / 17 |
| LtF-kNN @ 0.5 % | 59 | −0.196 [−0.349, −0.054] | 1.45× [+0.18, +0.58] | 28 / 24 |
| full MILP @ 0.5 % | 60 | +0.124 [+0.054, +0.198] | 2.75× [+0.80, +1.24] | 21 / 34 |
| full MILP @ 1 % | 60 | +0.065 [−0.008, +0.138] | 1.52× [+0.24, +0.61] | 27 / 27 |

* **Where the LP-based rules fail.** Their large gaps come from cost errors that no guard sees: LP-int+g fixes wrong
  integral values (5.6 %, 3.8 %, 3.6 % on three instances), and LtF-LP and the LP pipeline reach 2.9–3.1 % on one
  instance. The learned pipeline's worst instance is 1.13 %. Faithful LtF-LP leaves 4 of 60 reduced problems
  infeasible; with guards (LP pipeline) all are feasible. The not-served solutions are small: reserve shortfalls of at
  most 4.2 MWh (also in the full MILP at 0.5 % and 1 % on two instances), and shedding or over-generation of at most
  0.4 MWh in two unguarded rules (LtF-LP once, LtF-kNN once).
* **By relaxation strength** (test split at the median LP gap to DB, 0.54 %), the pipeline's gap advantage over the LP
  pipeline is −0.21 pp [−0.49, +0.01] on the weak half and −0.12 pp [−0.25, +0.01] on the strong half. Both point
  the same way, but neither half alone is significant. On the weak half the pipeline is slower (Δ log speed-up
  −0.60 [−0.92, −0.25]); on the strong half the speeds are equal.
* **Stacking.** Fixing and a loose gap stack again: each fixing rule at 0.5 % has about the same gap as at 0.1 %
  (+0.01 to +0.06 pp) and is 1.5–2.6× faster (geometric mean).

## 6. Verdict

* A legitimately weaker relaxation is hard to find in the cheap range. Among the screened settings, derating
  RTS-GMLC's lines to 60 % is the only one that weakens it markedly while keeping the full MILP under five minutes,
  and then only on 12-hour windows.
  Even there, the weakening shows in the median and in the fractional count, not in the mean, because the
  original-rating windows already have rare very weak relaxations.
* On this benchmark **learning helps**, unlike on the original-rating benchmarks (a cross-benchmark comparison, see
  §7). The learned pipeline is the only fixing rule statistically as accurate as the full MILP at a 1 % gap
  (+0.07 pp [−0.01, +0.14]), and it is 1.5× faster than it. It
  beats every learning-free rule on accuracy by 0.16–0.49 pp, and it beats Learning to Fix's own kNN on both accuracy
  and speed. What learning adds is a confidence ranking that separates the LP's wrong integral values from the right
  ones: the validated fixed share is 84 % against 63–65 % for the LP values.
* If speed matters more than accuracy, the learning-free LP rules are still competitive: Learning to Fix on the LP
  values (guarded, "LP pipeline") reaches 0.49 % at a geometric-mean 8.6× against the pipeline's 0.32 % at 6.8×. The
  Pareto front (mean gap vs. geometric-mean speed-up) is full MILP 0.1 % → 0.5 % → 1 % → learned pipeline 0.5 % →
  LtF-LP 0.5 % (0.49 % at 9.0×, but over only the 56 feasible instances); the LP pipeline at 0.5 %, feasible on all
  60, has the same mean gap and is 4 % slower.
* Practical recommendation for congested systems: the learned pipeline at a 0.5 % gap, trained without any full
  MILP (about 3 core-hours of polishing plus 1 core-hour of training for 2,000 instances).

## 7. Caveats

* **Benchmark choice.** The derated 12 h benchmark is the screening's best compromise, not a uniformly weaker
  relaxation (§5.1). 24 h × 0.6 (similar gap) was dropped for MILP time, not for its relaxation. The 60 % rating is a
  stress setting chosen by us; the original RTS-GMLC data are otherwise unchanged.
* **Cross-benchmark comparisons** with uc12 (original ratings) are indicative only: different instances, a 60 s MILP
  limit there, other hosts. All within-benchmark comparisons are paired on the same 60 test instances and the same
  process.
* **Timing.** The evaluation ran after several container restarts on a host that re-times the data generation's
  MILPs a median 1.52× slower. Every speed-up uses only times from the same evaluation process (all runs of an
  instance back to back on one core), so the comparisons are fair, but absolute seconds differ from the
  data-generation logs. Two other agents' jobs ran on cores 0–1 throughout (load average about 4).
* **One test instance** (21) hits the 600 s limit in the reference full MILP (gap above 0.1 %); its DB is the
  limit-time bound.
* **Labels without full MILPs**: the polished labels are a median 0.31 % above the optimum. Learning-free rules use
  no training data; LtF-kNN uses the same polished labels as the pipeline (no MILP labels are available).
* **No hyperparameter search** for the networks: m1x's temporal GNN and settings (40 epochs, ensemble of 5); only the
  polishing setting and the early-stopping epoch were chosen, on validation. One seed of the whole data and training
  pipeline; the 95 % CIs cover test-instance variation only.
* **Served share** counts any reserve shortfall, shedding or over-generation as not served; all are at most 4.2 MWh.

## 8. Reproduction

```
python scripts/uc_hard_screen.py --case rts --line_scale 0.6 --n 4          # screening (see uc_hard_screen_q*.sh)
python scripts/uc_hard_screen_report.py
scripts/uc_hard_q.sh data2 ; scripts/uc_hard_q.sh data3                     # val / test MILPs, training scenarios
python scripts/uc_hard_train.py --name t0 --seeds 0,1,2 --label lf --n 2000 --epochs 30
python scripts/uc_hard_data.py --pilot --teacher t0
python scripts/uc_hard_data.py --polish --teacher t0 --chunks 0-39 --ratio 0.9 --tl 5
python scripts/uc_hard_train.py --name fin --seeds 0,1,2,3,4 --label pol --epochs 40 ; python scripts/uc_hard_train.py --ensemble fin
python scripts/uc_hard_tune.py --run lp --src lp --tuner faithful
python scripts/uc_hard_tune.py --run lp_g --src lp --tuner guarded
python scripts/uc_hard_tune.py --run knn --src knn --tuner faithful
python scripts/uc_hard_tune.py --run hg_fin --src fin_ens --tuner guarded
python scripts/uc_hard_paired.py --split val
taskset -c 2 python scripts/uc_hard_eval.py --shard 0 --nshards 2 ; taskset -c 3 python scripts/uc_hard_eval.py --shard 1 --nshards 2
python scripts/uc_hard_report.py
```

Every command runs single-threaded (`OTSL_THREADS=1`, HiGHS one thread) under `taskset -c 2` or `-c 3`. Tests:
`tests/test_hard.py` (scenario builder, relaxation statistics, minimal conflict release).
