#!/usr/bin/env bash
# Step 2 of docs/unified_metrics.md, phase 1: the re-timing. The 400 test instances (12 h test_fresh and six shifted sets,
# 24 h, congested) are split into NTASKS load-balanced shares (scripts/unified_retime.py); this job runs shares
# TASK0 .. TASK0 + (CPUs - 1), one worker per allocated core, each pinned to its core with taskset. A worker runs, per
# instance, every evaluation process that covers it back to back on its core. Every process is single-threaded (HiGHS one
# thread, torch one thread, OMP/MKL 1). Arm A's networks run in bfloat16, so the job runs on the Granite Rapids node (the
# only ICE CPU with bf16 units, no hyper-threading: one worker = one physical core), which keeps every method on one CPU.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name s2-retime-0 --cpus 8 --mem 64G --time 08:00:00 \
#       --constraint cpu-gnr --export NTASKS=24,TASK0=0 scripts/pace/pgu_s2_retime.sbatch        (then TASK0=8, 16)
#   smoke test (ITEMS spread round-robin over the allocated cores, same pinning, outputs in results/unified/retime/smoke):
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name s2-smoke --cpus 2 --mem 32G --time 02:00:00 \
#       --constraint cpu-gnr --export ITEMS=uc12:tf:0,uc12:line_out:0,uc24:tf:0,hard:tf:0 scripts/pace/pgu_s2_retime.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), threads per core $(lscpu | sed -n 's/^Thread(s) per core: *//p')"
grep -qm1 -wE 'amx_bf16|avx512_bf16' /proc/cpuinfo || { echo "refusing: this CPU has no bf16 units (--constraint cpu-gnr)"; exit 1; }
R=results/unified/retime
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
echo "allocated CPUs: ${CPUS[*]}"
mkdir -p "$R/workers" "$R/smoke"
pids=()
for w in "${!CPUS[@]}"; do
  if [ -n "${ITEMS:-}" ]; then        # smoke: items w, w + n, w + 2n, ... of the list
    mine=$(python3 -c "import sys; x = sys.argv[1].split(','); print(','.join(x[int(sys.argv[2])::int(sys.argv[3])]))" \
           "$ITEMS" "$w" "${#CPUS[@]}")
    [ -n "$mine" ] || continue
    taskset -c "${CPUS[$w]}" python3 scripts/unified_retime.py --items "$mine" --tag "s$w" --res "$R/smoke" \
      > "$R/smoke/worker_s$w.out" 2>&1 & pids+=($!)
  else
    t=$((${TASK0:?} + w))
    taskset -c "${CPUS[$w]}" python3 scripts/unified_retime.py --task "$t" --ntasks "${NTASKS:?}" \
      > "$R/workers/t$t.out" 2>&1 & pids+=($!)
  fi
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "S2 RETIME DONE (${ITEMS:+smoke }${#pids[@]} workers): $fail failed"
[ "$fail" = 0 ]
