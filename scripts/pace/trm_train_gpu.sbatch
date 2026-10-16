#!/usr/bin/env bash
# Step 4 (GPU): train a TRM. Every knob of otsl.trm.TRMConfig is a flag of scripts/rec_trm_train.py; presets:
#   cpu (this repository's runs), gpu_small (hidden 128, attention mixers), paper (TRM's scale: hidden 512, 2 layers,
#   n 6, T 3, N_sup 16, batch 768, EMA 0.999, AdamW (0.9, 0.95), lr 1e-4, weight decay 1.0, 2k warm-up).
# Usage: TAG=paper PRESET=paper N_TRAIN=100000 EXTRA="--steps 60000" sbatch scripts/pace/trm_train_gpu.sbatch
# Resumable: resubmitting the same TAG continues from data/generated/recursive/models/$TAG.pt.
##SBATCH -J trm-train
##SBATCH -A <ACCOUNT>                 # PLACEHOLDER
##SBATCH -p <GPU_PARTITION>           # PLACEHOLDER (check PACE ICE docs)
##SBATCH -q <QOS>                     # PLACEHOLDER
##SBATCH --gres=gpu:<GPU_TYPE>:1      # PLACEHOLDER: GPU type / syntax per PACE ICE docs (e.g. an H100/A100/V100 name)
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G
##SBATCH -t 16:00:00                  # PLACEHOLDER: respect the partition's wall-time limit
##SBATCH -o logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"   # submit from the repository root (mkdir -p logs first)
TAG=${TAG:-paper}; PRESET=${PRESET:-paper}; N_TRAIN=${N_TRAIN:-2000}; EXTRA=${EXTRA:-}
nvidia-smi || true
python3 scripts/rec_trm_train.py --tag "$TAG" --preset "$PRESET" --device cuda --threads 4 \
  --n_train "$N_TRAIN" --data_dirs data/generated/uc12_m1x data/generated/recursive/pace \
  --log_every 100 --val_every 2000 $EXTRA
# optional stage 2 (REINFORCE with the dispatch LP as critic; the LPs run on the job's CPU cores):
if [ "${RL:-0}" = "1" ]; then
  python3 scripts/rec_trm_train.py --tag "${TAG}_rl" --stage rl --base "$TAG" --device cuda --threads 2 \
    --n_train "$N_TRAIN" --data_dirs data/generated/uc12_m1x data/generated/recursive/pace \
    --rl_steps "${RL_STEPS:-400}" --rl_bs 32 --rl_samples 8 --workers 6 --rl_random_step 1
fi
