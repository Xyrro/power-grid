#!/usr/bin/env bash
# Screened network and the certified pipeline (docs/methods/screen.md): one worker per allocated core, pinned to it with
# taskset, each claiming instances from the shared list of scripts/screen_run.py until none is left (any number of these
# jobs can run at once). Every process is single-threaded. Runs on the Granite Rapids node, the CPU of step 2.
#   validation (the choices):  python3 scripts/pace/pgu_guard.py submit --kind cpu --name screen-val --cpus 8 --mem 64G \
#       --time 04:00:00 --constraint cpu-gnr --export PHASE=val scripts/pace/pgu_screen.sbatch
#   test (once, with the validation choice): ... --name screen-test ... --export "PHASE=test,CERT=ens5:0.05+lp:0.05" ...
#   (CERT: '+'-separated, since sbatch --export splits on commas)
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), threads per core $(lscpu | sed -n 's/^Thread(s) per core: *//p')"
: "${PHASE:?PHASE=val or PHASE=test}"
read -r -a CPUS <<< "$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')"
echo "allocated CPUs: ${CPUS[*]}"
mkdir -p results/screen/"$PHASE"
CERT_ARG=()
if [ -n "${CERT:-}" ]; then CERT_ARG=(--cert "${CERT//+/,}"); fi
pids=()
for w in "${!CPUS[@]}"; do
  tag="j${SLURM_JOB_ID}w$w"
  taskset -c "${CPUS[$w]}" python3 scripts/screen_run.py --phase "$PHASE" --tag "$tag" "${CERT_ARG[@]}" \
    > "results/screen/$PHASE/worker_$tag.out" 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "SCREEN $PHASE DONE (${#pids[@]} workers): $fail failed"
[ "$fail" = 0 ]
