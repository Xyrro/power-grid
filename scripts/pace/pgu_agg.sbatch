#!/usr/bin/env bash
# Round 8 (docs/methods/agg.md): one scripts/agg_run.py worker per allocated core, pinned to it with
# taskset, each claiming instances from its shared list until none is left (any number of these
# jobs can run at once). Every process is single-threaded.
#   validation (the choices):  python3 scripts/pace/pgu_guard.py submit --kind cpu --name agg-val --cpus 8 --mem 64G \
#       --time 04:00:00 --export PHASE=val scripts/pace/pgu_agg.sbatch
#   test (once, with the validation choice): ... --name agg-test ... --export "PHASE=test,ACERT=s0+x:0.02~lag0.0003+x:0.4" ...
#   (ACERT, METHODS: lists separated by "~", since sbatch --export splits on commas and "+" is part of the names)
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), threads per core $(lscpu | sed -n 's/^Thread(s) per core: *//p')"
: "${PHASE:?PHASE=val or PHASE=test}"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
echo "allocated CPUs: ${CPUS[*]}"
mkdir -p results/agg/"$PHASE"
ARGS=()
if [ -n "${ACERT:-}" ]; then ARGS+=(--acert "${ACERT//\~/,}"); fi
if [ -n "${METHODS:-}" ]; then ARGS+=(--methods "${METHODS//\~/,}"); fi
if [ -n "${SCRIPT:-}" ]; then ARGS+=(--script "$SCRIPT"); fi
pids=()
for w in "${!CPUS[@]}"; do
  tag="j${SLURM_JOB_ID}w$w"
  taskset -c "${CPUS[$w]}" python3 scripts/agg_run.py --phase "$PHASE" --tag "$tag" "${ARGS[@]}" \
    > "results/agg/$PHASE/worker_$tag.out" 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "AGG $PHASE DONE (${#pids[@]} workers): $fail failed"
[ "$fail" = 0 ]
