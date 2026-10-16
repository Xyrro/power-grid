#!/usr/bin/env bash
# Pre-flight check on one GPU before the real training runs. Verifies that every GPU code path works and measures
# real step times, so the production walltimes are calibrated rather than extrapolated from the CPU study.
#
#   python3 scripts/pace/pgu_guard.py submit --kind gpu --name smoke --gpus 1 --cpus 24 --mem 64G \
#       --time 01:00:00 scripts/pace/pgu_smoke_gpu.sbatch
#
# Every check runs even if an earlier one fails; the job prints a PASS/FAIL summary at the end and exits non-zero if
# anything failed. Nothing here touches the production model tags: every tag is prefixed smoke_ and lives in
# data/generated/recursive/smoke, which is cleared at the start.
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
set +e                       # trm_env.sh sets -e; here we want every check to run
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation at large batch
NCPU=${SLURM_CPUS_PER_TASK:-24}
SM=data/generated/recursive/smoke
rm -rf "$SM"; mkdir -p "$SM"
FAILED=""
run() {  # run <name> <command...>
  local name=$1; shift
  echo "=============== $name"
  local t0=$SECONDS
  "$@"
  local rc=$?
  echo "--- $name rc=$rc elapsed=$((SECONDS - t0))s"
  [ $rc -eq 0 ] || FAILED="$FAILED $name"
  return 0
}

echo "=============== 0. GPU and environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "| total GB", round(p.total_memory / 2**30, 1),
          "| capability", torch.cuda.get_device_capability(0), "| bf16", torch.cuda.is_bf16_supported())
PY

run "A1-train-gpu_small" python3 scripts/rec_trm_train.py --tag smoke_a --preset gpu_small --device cuda \
    --threads 4 --n_train 500 --steps 40 --val_every 20 --log_every 10 --out_dir "$SM"
run "A2-resume" python3 scripts/rec_trm_train.py --tag smoke_a --preset gpu_small --device cuda \
    --threads 4 --n_train 500 --steps 60 --val_every 30 --log_every 10 --out_dir "$SM"
# the paper preset's batch of 768 does not fit on one GPU; micro_batch accumulates over slices (same result)
for MB in 64 128; do
  run "A3-paper-micro$MB" python3 scripts/rec_trm_train.py --tag "smoke_paper_mb$MB" --preset paper --device cuda \
      --threads 4 --n_train 500 --steps 6 --val_every 1000 --log_every 1 --micro_batch $MB --out_dir "$SM"
done
run "A4-rl-stage" python3 scripts/rec_trm_train.py --tag smoke_a_rl --stage rl --base smoke_a --device cuda \
    --threads 2 --n_train 500 --rl_steps 3 --rl_bs 4 --rl_samples 4 --workers 8 --out_dir "$SM"
run "B1-train-gpu+LP" python3 scripts/rec_rloop_train.py --tag smoke_b --preset gpu --device cuda --threads 4 \
    --lp_workers $((NCPU - 4)) --inflight 4 --n_train 256 --epochs 1 --val_n 8
run "B2-feedback-off" python3 scripts/rec_rloop_train.py --tag smoke_b_nofb --preset gpu --device cuda --threads 4 \
    --lp_workers $((NCPU - 4)) --inflight 4 --n_train 256 --epochs 1 --val_n 8 --no_feedback

echo "=============== SUMMARY"
if [ -n "$FAILED" ]; then echo "SMOKE FAILED:$FAILED"; exit 1; fi
echo "SMOKE ALL PASSED"
