#!/usr/bin/env bash
# Arm B, CPU-scale screen of the loop-native attention residual with seeds (otsl.rloop.RLoopNet loopres): three
# variants x seeds 1-3, preset cpu, 2,000 labels; equal seeds give equal initial weights and data order, so each seed
# is a matched triple.
#   plain   carry-last (the existing model)
#   lp      attention residual across LP steps, write scores with the LP verdict (this study's adaptation)
#   paper   the paper's placement: inner axis over latent updates, outer axis over answer updates, state-only scores
# Each task: supervised stage, REINFORCE stage, then the 360-instance validation at K = 24.
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name rlscreen --array 0-8%8 --cpus 4 --mem 16G \
#       --time 08:00:00 scripts/pace/pgu_rloop_screen.sbatch
# Afterwards: python3 scripts/rec_loopres_compare.py --prefix cs
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
export OTSL_THREADS=1
VARIANTS=(plain lp paper)
FLAGS=("" "--loopres" "--loopres --loopres_mode paper")
I=${SLURM_ARRAY_TASK_ID:-0}
V=$(( I % 3 ))
SEED=$(( I / 3 + 1 ))
T="cs_${VARIANTS[$V]}_s${SEED}"
W=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))
echo "screen $T on $(hostname): seed $SEED ${FLAGS[$V]:-carry-last}, $W LP workers"
python3 scripts/rec_rloop_train.py --tag "$T" --preset cpu --threads 1 --lp_workers "$W" --n_train 2000 \
  --val_n 120 --seed "$SEED" ${FLAGS[$V]}
python3 scripts/rec_rloop_train.py --tag "${T}_rl" --stage rl --init "$T" --preset cpu --threads 1 --lp_workers "$W" \
  --n_train 2000 --val_n 120 --seed "$SEED"
python3 scripts/rec_rloop_val.py --model "${T}_rl" --K 24
echo "SCREEN DONE $T"
