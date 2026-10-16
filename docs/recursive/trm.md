# Arm A: a tiny recursive model (TRM) for end-to-end unit commitment

*Branch `recursive-uc`, plan in [`PLAN.md`](PLAN.md). Code: [`otsl/trm.py`](../../otsl/trm.py) (features, model,
deep-supervision training, REINFORCE stage, decoders), `scripts/rec_trm_{train,val,select,eval,report,data}.py`,
`scripts/rec_trm_queue.sh`, tests [`tests/test_trm.py`](../../tests/test_trm.py), PACE ICE jobs `scripts/pace/trm_*`.
Results: [`results/recursive/trm_results.md`](../../results/recursive/trm_results.md) (+ `.json`), raw records
`results/recursive/trm_eval_test_fresh_s*.jsonl`, validation `results/recursive/trm_val_*.json`, training logs
`results/recursive/trm_train_*.json`. Data and weights (git-ignored): `data/generated/recursive/`.*

## Status

*Kept current for resuming after a container restart. Every job is resumable: rerun the same command.*

- 23:49 main run started on core 0 (`bash scripts/rec_trm_queue.sh 0 "train:main:--preset cpu"`), one-shot controls on
  core 1. Logs: `data/generated/recursive/logs/` (`train_<tag>.log`, queue outputs `queue_core*.out`).
- 00:03 one-shot control done (13 min); validation pass of it running.

## Summary

(pending)

## Method

(pending)

## Setup

(pending)

## Results

(pending)

## Ablations

(pending)

## Verdict

(pending)

## Caveats

(pending)

## Running on PACE ICE

(pending)
