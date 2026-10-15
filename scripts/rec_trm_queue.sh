#!/usr/bin/env bash
# Run queue of the TRM study (arm A). Each job is resumable and skipped when its result file exists:
#   bash scripts/rec_trm_queue.sh <core> <job> [<job> ...]
# A job is "train:<tag>:<flags>" (stage 1 or 2 via the flags), "val:<tag>" (validation pass) or "cmd:<shell command>".
# Logs: data/generated/recursive/logs/. Every process is pinned to <core>, single-threaded.
set -u
cd "$(dirname "$0")/.."
core=$1; shift
mkdir -p data/generated/recursive/logs
export OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for job in "$@"; do
  kind=${job%%:*}; rest=${job#*:}
  case $kind in
    train)
      tag=${rest%%:*}; flags=${rest#*:}
      if [ -f "results/recursive/trm_train_${tag}.json" ]; then echo "skip train $tag"; continue; fi
      echo "$(date +%H:%M:%S) train $tag $flags"
      taskset -c "$core" python3 scripts/rec_trm_train.py --tag "$tag" $flags \
        >> "data/generated/recursive/logs/q_train_${tag}.out" 2>&1 || echo "FAILED train $tag" ;;
    val)
      tag=$rest
      echo "$(date +%H:%M:%S) val $tag"
      taskset -c "$core" python3 scripts/rec_trm_val.py --tag "$tag" \
        >> "data/generated/recursive/logs/q_val_${tag}.out" 2>&1 || echo "FAILED val $tag" ;;
    cmd)
      echo "$(date +%H:%M:%S) cmd $rest"
      taskset -c "$core" bash -c "$rest" || echo "FAILED cmd $rest" ;;
  esac
done
echo "$(date +%H:%M:%S) queue done"
