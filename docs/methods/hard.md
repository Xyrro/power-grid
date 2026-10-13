# hard: a benchmark with a weaker LP relaxation

Code: [`otsl/hard.py`](../../otsl/hard.py), `scripts/uc_hard_*.py`. Results:
[`results/hard/hard_results.md`](../../results/hard/hard_results.md) (+ `.json`), screening records
`results/hard/hard_screen.jsonl`. Data (git-ignored): `data/generated/hard/`.

## Status

*Kept current for resume after a container restart. Every step is idempotent (records are appended and finished
instances skipped): rerun the step's command.*

- 06:47 screening started on cores 2 and 3 (`scripts/uc_hard_screen_q.sh 2|3`, continued by
  `scripts/uc_hard_screen_q2.sh 2|3`; logs `results/hard/logs/screen_c{2,3}.log`, records
  `results/hard/hard_screen.jsonl`).

## Summary

(pending)
