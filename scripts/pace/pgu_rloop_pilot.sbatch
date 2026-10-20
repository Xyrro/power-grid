#!/usr/bin/env bash
# Arm B pilot of the loop-native attention residual (InfiLoop, arXiv 2610.11570; otsl.rloop.RLoopNet loopres) at CPU
# scale, before any GPU time: the same model twice, 2,000 labels, preset cpu, seed 0, identical except --loopres.
#   task 0  pilot_base   carry-last (the existing model)
#   task 1  pilot_lr     attention residual across outer steps, write scores with the LP verdict
# Each task: supervised stage, REINFORCE stage, then the 360-instance validation at K = 24 (per-step records).
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name rlpilot --array 0-1%2 --cpus 4 --mem 16G \
#       --time 08:00:00 scripts/pace/pgu_rloop_pilot.sbatch
# Afterwards (pre-registered rule: go to the GPU run only if pilot_lr_rl's best-of-24 validation gap beats
# pilot_base_rl's with a paired 95 % CI excluding zero):
#   python3 scripts/rec_rloop_pair.py --a pilot_lr_rl --b pilot_base_rl
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
export OTSL_THREADS=1
TAGS=(pilot_base pilot_lr)
EXTRAS=("" "--loopres")
S=${SLURM_ARRAY_TASK_ID:-0}
T=${TAGS[$S]}
W=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))
echo "pilot $T on $(hostname): ${EXTRAS[$S]:-carry-last}, $W LP workers"
python3 scripts/rec_rloop_train.py --tag "$T" --preset cpu --threads 1 --lp_workers "$W" --n_train 2000 \
  --val_n 120 ${EXTRAS[$S]}
python3 scripts/rec_rloop_train.py --tag "${T}_rl" --stage rl --init "$T" --preset cpu --threads 1 --lp_workers "$W" \
  --n_train 2000 --val_n 120
python3 scripts/rec_rloop_val.py --model "${T}_rl" --K 24
echo "PILOT DONE $T"
