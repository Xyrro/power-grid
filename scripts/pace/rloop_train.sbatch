#!/bin/bash
# rloop (arm B, docs/recursive/rloop.md): GPU training with the exact dispatch LPs in a CPU worker pool, same job.
#
# The network runs on the GPU; decoding + dispatch LPs (HiGHS, single-threaded) run in LP_WORKERS CPU processes;
# INFLIGHT batches are kept in flight so the GPU trains on one batch while the workers price another.
# Stage 1: deep supervision with LP feedback (preset "gpu"); stage 2: REINFORCE fine-tuning with the LP as critic.
# Both stages checkpoint every epoch / 25 updates under data/generated/recursive/rloop/ and resume when resubmitted.
#
# PLACEHOLDERS: every value marked PLACEHOLDER below is cluster-specific and was NOT verified here -- check the PACE ICE
# documentation (partition / QOS / account names, GPU types, module names, per-job limits) and fill them in, or pass
# them on the command line, e.g.
#   sbatch -p <partition> -q <qos> --gres=gpu:<type>:1 scripts/pace/rloop_train.sbatch
# Lines starting with '##SBATCH' are ignored by Slurm until you remove one '#'.
#
#SBATCH -J rloop_train
##SBATCH -p PLACEHOLDER_GPU_PARTITION
##SBATCH -q PLACEHOLDER_QOS
##SBATCH -A PLACEHOLDER_ACCOUNT
##SBATCH --gres=gpu:PLACEHOLDER_GPU_TYPE:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o logs/rloop_train_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# PLACEHOLDER: load Python / CUDA the PACE ICE way (module names differ per cluster), then activate an environment with
# torch (CUDA build), numpy, scipy, pandas, highspy (see requirements of this repository).
# module load PLACEHOLDER_PYTHON_MODULE PLACEHOLDER_CUDA_MODULE
# source PLACEHOLDER_ENV/bin/activate

# Data: datasets are git-ignored and must be regenerated on the cluster first (shared data job of the recursive study,
# see scripts/pace/README.md). Needed: data/generated/uc12 (train/val/val_extra/val_extra2/test_fresh) and
# data/generated/uc12_m1x (extra_k + polish_k); optional extra polished chunks for --data_dir.

export OTSL_THREADS=1                    # torch intra-op threads in the LP workers (HiGHS is single-threaded anyway)
NCPU=${SLURM_CPUS_PER_TASK:-24}
LP_WORKERS=${LP_WORKERS:-$((NCPU - 4))}  # keep a few cores for the trainer process and data loading
INFLIGHT=${INFLIGHT:-4}
TAG=${TAG:-gpu_main}
N_TRAIN=${N_TRAIN:-2000}
EXTRA=${EXTRA:-}                         # e.g. "--no_feedback" for the ablation, "--data_dir data/generated/uc12_more"

python3 scripts/rec_rloop_train.py --tag "$TAG" --preset gpu --device cuda --threads 4 \
    --lp_workers "$LP_WORKERS" --inflight "$INFLIGHT" --n_train "$N_TRAIN" --val_n 120 $EXTRA
python3 scripts/rec_rloop_train.py --tag "${TAG}_rl" --stage rl --init "$TAG" --preset gpu --device cuda --threads 4 \
    --lp_workers "$LP_WORKERS" --n_train "$N_TRAIN" --val_n 120 $EXTRA
# validation (CPU-side LPs, GPU network) for selection; the timed test runs on CPU (rloop_eval.sbatch)
python3 scripts/rec_rloop_val.py --model "${TAG}_rl" --K 24
