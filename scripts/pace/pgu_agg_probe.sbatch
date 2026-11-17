#!/usr/bin/env bash
# Round 8 exploration on validation instances (docs/methods/agg.md): one scripts/agg_probe.py worker per allocated core,
# pinned with taskset, sharing one claim folder (any number of these jobs can run at once). Single-threaded processes.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name agg-probe --cpus 4 --mem 32G --time 03:00:00 \
#       --export "KS=0-120,METHODS=ref+afull:0.5+acert:s0:0.02,TAG=e1" scripts/pace/pgu_agg_probe.sbatch
#   (METHODS: '+'-separated, since sbatch --export splits on commas)
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
: "${KS:?KS=a-b[:step]}" "${METHODS:?METHODS=m1+m2}" "${TAG:?TAG=name}"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
OUT=results/agg/explore/$TAG
mkdir -p "$OUT"
pids=()
for w in "${!CPUS[@]}"; do
  taskset -c "${CPUS[$w]}" env OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python3 -W ignore scripts/agg_probe.py --ks "$KS" --methods "${METHODS//+/,}" --claims "$OUT/claims" \
    --out "$OUT/j${SLURM_JOB_ID}w$w.jsonl" > "$OUT/worker_j${SLURM_JOB_ID}w$w.out" 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "AGG PROBE $TAG DONE (${#pids[@]} workers): $fail failed"
[ "$fail" = 0 ]
