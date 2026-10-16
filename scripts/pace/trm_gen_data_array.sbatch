#!/usr/bin/env bash
# Step 3 (CPU array): more training data, one chunk of CHUNK instances per array task (scenario + LP relaxation +
# label-free label + teacher-polished label; ~5.5 s per instance on one core -> ~25 min per chunk of 250).
# 100,000 instances = 400 tasks = ~150 core-hours. Submit e.g.:  sbatch --array=0-399%100 scripts/pace/trm_gen_data_array.sbatch
##SBATCH -J trm-data
##SBATCH -A <ACCOUNT>                 # PLACEHOLDER
##SBATCH -p <CPU_PARTITION>           # PLACEHOLDER
##SBATCH -q <QOS>                     # PLACEHOLDER
#SBATCH -N 1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=3G
##SBATCH -t 01:00:00
##SBATCH -o logs/%x-%A_%a.out
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"   # submit from the repository root (mkdir -p logs first)
N_TOTAL=${N_TOTAL:-100000}
python3 scripts/rec_trm_data.py --chunk "${SLURM_ARRAY_TASK_ID}" --n_total "$N_TOTAL" --seed 211 \
  --out data/generated/recursive/pace
