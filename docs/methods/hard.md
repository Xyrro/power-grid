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
- Resume: rerun the phase (`scripts/uc_hard_q.sh run2|run3`); finished chunks, models and converged tuning runs are
  skipped. Polishing workers share chunks through `polish_<kk>.npz.lock` files: delete stale locks after a restart.

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

(Screening results and the choice: pending the write-up.)

