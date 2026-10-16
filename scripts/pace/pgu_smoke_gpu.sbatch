#!/usr/bin/env bash
# Pre-flight check on one GPU before the real training runs. Verifies that every GPU code path works and measures
# real step times, so the production walltimes are calibrated rather than extrapolated from the CPU study.
#
#   python3 scripts/pace/pgu_guard.py submit --kind gpu --name smoke --gpus 1 --cpus 24 --mem 64G \
#       --time 01:00:00 scripts/pace/pgu_smoke_gpu.sbatch
#
# Checks, in order:
#   1. GPU visible to torch; device name and memory
#   2. arm A: short run of the gpu_small preset on cuda, then a resume from its checkpoint
#   3. arm A: a few steps of the paper preset, to confirm it fits in GPU memory and to time a step
#   4. arm B: a short supervised run of the gpu preset with the LP workers on this job's CPU cores
# Nothing here writes into the production model tags: every tag is prefixed smoke_.
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
set -x
NCPU=${SLURM_CPUS_PER_TASK:-24}
SM=data/generated/recursive/smoke
mkdir -p "$SM"

echo "=============== 1. GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("total GB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    print("bf16 supported", torch.cuda.is_bf16_supported())
PY

echo "=============== 2. arm A, gpu_small, short run on cuda"
time python3 scripts/rec_trm_train.py --tag smoke_a --preset gpu_small --device cuda --threads 4 \
    --n_train 500 --steps 40 --val_every 20 --log_every 10 --out_dir "$SM"
echo "--- resume (same tag: must continue from the checkpoint, not restart)"
time python3 scripts/rec_trm_train.py --tag smoke_a --preset gpu_small --device cuda --threads 4 \
    --n_train 500 --steps 60 --val_every 30 --log_every 10 --out_dir "$SM"

echo "=============== 3. arm A, paper preset, memory and step time"
time python3 scripts/rec_trm_train.py --tag smoke_paper --preset paper --device cuda --threads 4 \
    --n_train 500 --steps 12 --val_every 1000 --log_every 2 --out_dir "$SM" || echo "PAPER_PRESET_FAILED"
python3 -c "import torch; print('peak GPU GB', round(torch.cuda.max_memory_allocated()/2**30,2) if torch.cuda.is_available() else 0)" || true

echo "=============== 4. arm B, gpu preset with LP workers"
time python3 scripts/rec_rloop_train.py --tag smoke_b --preset gpu --device cuda --threads 4 \
    --lp_workers $((NCPU - 4)) --inflight 4 --n_train 256 --epochs 1 --val_n 8 || echo "RLOOP_FAILED"

echo "=============== SMOKE DONE"
