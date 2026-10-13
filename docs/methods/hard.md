# hard: a benchmark with a weaker LP relaxation

Code: [`otsl/hard.py`](../../otsl/hard.py), `scripts/uc_hard_*.py`. Results:
[`results/hard/hard_results.md`](../../results/hard/hard_results.md) (+ `.json`), screening records
`results/hard/hard_screen.jsonl`. Data (git-ignored): `data/generated/hard/`.

## Status

*Kept current for resume after a container restart. Every step is idempotent (records are appended, finished chunks and
instances skipped): rerun the step's command.*

- 06:47–07:49 screening done (`scripts/uc_hard_screen_q.sh`, `scripts/uc_hard_screen_q2.sh`; records
  `results/hard/hard_screen.jsonl`, summary `results/hard/hard_screen.md` from `scripts/uc_hard_screen_report.py`).
  Chosen: RTS-GMLC 24 h, line ratings × 0.6 (`otsl/hard.py`: `LINE_SCALE`, `CFG`).
- 07:35 data generation (`scripts/uc_hard_q.sh data2|data3`, logs `results/hard/logs/data_c{2,3}.log`): validation
  (60 instances, full MILP 0.1 %, 600 s) on core 2; 1,500 LP-only training scenarios on core 3. Output
  `data/generated/hard/{val,test,train}_<kk>.npz`.

## Summary

(pending)
