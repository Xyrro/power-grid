#!/usr/bin/env bash
# Round 8 on PGLib-UC California (docs/methods/agg.md): scripts/agg_pglib_eval.py on the 30 test instances, one worker
# per allocated core (pinned with taskset), claiming instances from a shared folder; every process single-threaded.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name pglib-eval --cpus 4 --mem 48G --time 06:00:00 \
#       scripts/pace/pgu_pglib_eval.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
export PGLIB=$PGU_ROOT/ext/pglib-uc OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
OUT=results/agg/pglib
mkdir -p "$OUT/claims"
work() {
  local w=$1 k
  for k in $(seq 0 29); do
    ( set -o noclobber; : > "$OUT/claims/test_$k" ) 2>/dev/null || continue
    taskset -c "${CPUS[$w]}" python3 -W ignore scripts/agg_pglib_eval.py --k "$k" --out "$OUT/test_j${SLURM_JOB_ID}w$w.jsonl" \
      ${PCERT:+--pcert "${PCERT//\~/,}"}
  done
}
pids=()
for w in "${!CPUS[@]}"; do work "$w" > "$OUT/worker_j${SLURM_JOB_ID}w$w.out" 2>&1 & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "PGLIB EVAL DONE (${#pids[@]} workers): $fail failed"
