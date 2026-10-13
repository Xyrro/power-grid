# hard: a benchmark with a weaker LP relaxation

Code: [`otsl/hard.py`](../../otsl/hard.py), `scripts/uc_hard_*.py`. Results:
[`results/hard/hard_results.md`](../../results/hard/hard_results.md) (+ `.json`), screening records
`results/hard/hard_screen.jsonl`. Data (git-ignored): `data/generated/hard/`.

## Status

*Kept current for resume after a container restart. Every step is idempotent (records are appended, finished chunks and
instances skipped): rerun the step's command.*

- 06:47–07:49 screening done (`scripts/uc_hard_screen_q.sh`, `scripts/uc_hard_screen_q2.sh`; records
  `results/hard/hard_screen.jsonl`, summary `results/hard/hard_screen.md` from `scripts/uc_hard_screen_report.py`).
  First choice RTS-GMLC 24 h × 0.6; its first 5 validation MILPs took 61–534 s (median 392 s), so at 08:05 the 24 h
  data were archived (`data/generated/hard/screen24/`) and 12-hour windows screened (`rts12_*` records).
  Chosen: **RTS-GMLC 12 h (random start), line ratings × 0.6** (`otsl/hard.py`: `LINE_SCALE`, `T_H`, `CFG`).
- 08:15 data generation (`scripts/uc_hard_q.sh data2|data3`; logs `results/hard/logs/data_c{2,3}.log`): validation
  (60 instances, full MILP 0.1 %, 600 s) on core 2, 3,000 LP-only training scenarios on core 3. Output
  `data/generated/hard/{val,test,train}_<kk>.npz` (+ `_trace.json`).
- 08:30 queues rebalanced (`scripts/uc_hard_q.sh run2|run3`): core 2 = validation, then teacher t0 (3 temporal GNNs
  on label-free labels; logs `train_c2.log`), polishing pilot (`pilot_c2.log`, `data/generated/hard/pilot.json`);
  core 3 = training scenarios, all 60 test MILPs, then learning-free LtF tuning on the LP relaxation
  (`results/hard/hard_tune_lp.json`, log `tune_c3.log`).
- 09:27 validation done (60; LP gap to C mean 0.83 %, median 0.71 %; 95.5 % integral; MILP median 34 s). 09:46
  teacher t0 done; 09:51 pilot done (`data/generated/hard/pilot.json`): 90 % fixed + 5 s chosen (10 s gives the same
  labels; 80 % is worse).
- 09:55 `RATIO=0.9 TL=5 scripts/uc_hard_q.sh run2b` (core 2: polished labels for training chunks 0–39 = 2,000
  instances, final 5-member ensemble `fin`, guard-aware tuning `hg_fin`, test shard 0) and `... run3b` (core 3: rest of
  the test MILPs, LtF-LP tuning `lp`, joins the polishing, guard-aware LtF on the LP values `lp_g`, LtF-kNN `knn`, test
  shard 1). Logs `results/hard/logs/{polish,train,tune,eval}_c{2,3}.log`.
- 10:17 LtF-LP converged (64.6 % fixed on validation); 11:29 polishing done (2,000 labels, 5.1 s each); 11:41
  `lp_g` converged (62.5 %); 11:37–11:53 final members fin_s0–s2 trained. ~12:00 the session stopped (API limit);
  the container restarted several times (last 15:37). Restarts at 12:50 and 15:20 reran run2b / run3b, which died with
  the container: on disk all 40 polished chunks, all val / test chunks, fin_s0–s2, `lp`, `lp_g`.
- 15:38 resumed with `scripts/uc_hard_q.sh run2c` (core 2: fin_s3, fin_s4, ensemble, `hg_fin` tuning, test shard 0)
  and `run3c` (core 3: `knn` tuning, paired original-rating check on the validation scenarios
  `scripts/uc_hard_paired.py` → `results/hard/hard_paired_orig_val.jsonl`, test shard 1). Logs
  `results/hard/logs/{train,tune,paired,eval}_c{2,3}.log`.
- Resume: delete `data/generated/hard/*.lock`, rerun `scripts/uc_hard_q.sh run2c` and `run3c`; trained members,
  converged tuning runs, finished paired / test instances are skipped (an unfinished tuning run restarts).

## Summary

(pending)

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
(uc12, `data/generated/uc12/{val,val_extra,test_fresh}.npz`) have median 0.23–0.40 % and 98.2 % integral, so the
derated benchmark roughly doubles the median gap and the fractional share. Its mean gap is not much larger, though,
because the original-rating sets have a heavy tail (max 4.6–6.9 %). The paired check of §5 solves the same
validation scenarios at the original ratings.

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


