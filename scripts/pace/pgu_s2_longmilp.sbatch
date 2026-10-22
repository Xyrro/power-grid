#!/usr/bin/env bash
# Step 2 of docs/unified_metrics.md, phase 2: one-hour full-MILP solves that tighten the bound of every instance whose best
# known cost is still more than 0.1 % above it (list: results/unified/retime/longmilp_items.json, written by
# `python3 scripts/unified_longmilp.py --plan`; re-planned after phase 1, the list only grows). Bounds only, no timing:
# any CPU type; one single-threaded solve per allocated core, each worker claiming items until none is left (shifted
# instances wait until their file exists).
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name s2-bounds --cpus 16 --mem 64G --time 08:00:00 \
#       scripts/pace/pgu_s2_longmilp.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
R=results/unified/retime
mkdir -p "$R/workers"
pids=()
for w in "${!CPUS[@]}"; do
  tag="j${SLURM_JOB_ID}w$w"
  taskset -c "${CPUS[$w]}" python3 scripts/unified_longmilp.py --pull --tag "$tag" > "$R/workers/bounds_$tag.out" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "S2 BOUNDS DONE (${#pids[@]} workers): $fail failed"
[ "$fail" = 0 ]
