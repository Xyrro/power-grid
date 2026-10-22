#!/usr/bin/env bash
# Step 2 of docs/unified_metrics.md, phase 2: one-hour full-MILP solves that tighten the bound of every instance whose best
# known cost is still more than 0.1 % above it (list: results/unified/retime/longmilp_items.json, written by
# `python3 scripts/unified_longmilp.py --plan` after phase 1). Bounds only, no timing: any CPU type; one single-threaded
# solve per allocated core (shares TASK0 .. TASK0 + CPUs - 1 of NTASKS).
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name s2-bounds-0 --cpus 16 --mem 64G --time 04:00:00 \
#       --export NTASKS=32,TASK0=0 scripts/pace/pgu_s2_longmilp.sbatch                            (then TASK0=16)
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
R=results/unified/retime
mkdir -p "$R/workers"
pids=()
for w in "${!CPUS[@]}"; do
  t=$((${TASK0:?} + w))
  taskset -c "${CPUS[$w]}" python3 scripts/unified_longmilp.py --task "$t" --ntasks "${NTASKS:?}" \
    > "$R/workers/bounds_t$t.out" 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "S2 BOUNDS DONE tasks ${TASK0}-$((TASK0 + ${#CPUS[@]} - 1)): $fail worker(s) failed"
[ "$fail" = 0 ]
