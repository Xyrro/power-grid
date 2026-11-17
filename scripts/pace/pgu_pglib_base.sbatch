#!/usr/bin/env bash
# Round 8 on the 20 PGLib-UC California instances as distributed (48 h; docs/methods/agg.md): scripts/agg_pglib_base.py,
# one worker per allocated core (pinned), claiming instance files from a shared folder; every process single-threaded.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name pglib-base --cpus 4 --mem 64G --time 10:00:00 \
#       scripts/pace/pgu_pglib_base.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
export PGLIB=$PGU_ROOT/ext/pglib-uc OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
OUT=results/agg/pglib_base
mkdir -p "$OUT/claims"
FILES=$(cd "$PGLIB" && ls ca/*.json)
work() {
  local w=$1 f
  for f in $FILES; do
    ( set -o noclobber; : > "$OUT/claims/$(basename "$f" .json)" ) 2>/dev/null || continue
    taskset -c "${CPUS[$w]}" python3 -W ignore scripts/agg_pglib_base.py --file "$f" --T 48 --tl "${TL:-1800}" \
      --out "$OUT/base_j${SLURM_JOB_ID}w$w.jsonl"
  done
}
pids=()
for w in "${!CPUS[@]}"; do work "$w" > "$OUT/worker_j${SLURM_JOB_ID}w$w.out" 2>&1 & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "PGLIB BASE DONE (${#pids[@]} workers): $fail failed"
