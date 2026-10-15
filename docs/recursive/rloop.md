# rloop: recursion with the dispatch LP in the loop (arm B)

*Code: [`otsl/rloop.py`](../../otsl/rloop.py), `scripts/rec_rloop_{common,train,val,eval,report}.py`, Slurm jobs
`scripts/pace/rloop_*.sbatch`. Results: [`results/recursive/rloop_results.md`](../../results/recursive/rloop_results.md)
(+ `.json`), raw records `results/recursive/rloop_eval_<bench>_s<shard>.jsonl`, validation
`results/recursive/rloop_val_*.npz` and `rloop_val_summary.json`, training histories `results/recursive/rloop_train_<tag>.json`.
Checkpoints and logs (git-ignored): `data/generated/recursive/rloop/`.*

## Status

*Kept current for resuming after a container restart. Every step resumes: rerun the same command.*

- 23:45 smoke tests passed (training one epoch, RL, worker pool with in-flight batches, test evaluation on one
  instance each of uc12 and line_out).
- 23:51 validation baselines done (`scripts/rec_rloop_val.py --baselines --report`).
- 23:52 `main` (core 2) and 23:54 `nofb` (core 3) supervised training started, 10 epochs each (~5.3 min/epoch):
  `OTSL_THREADS=1 taskset -c 2 python3 scripts/rec_rloop_train.py --tag main --preset cpu`,
  `... -c 3 ... --tag nofb --preset cpu --no_feedback`; logs `data/generated/recursive/rloop/<tag>.{log,out}`.

## Summary

(to be written)

## Method

(to be written)

## Setup

(to be written)

## Results

(to be written)

## Ablations

(to be written)

## Extrapolation

(to be written)

## Verdict

(to be written)

## Caveats

(to be written)

## Running on PACE ICE

(to be written)
