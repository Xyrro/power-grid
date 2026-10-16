#!/usr/bin/env bash
# CPU array for the large training set, sized for this project's share: a FEW tasks, each looping over a strided set
# of chunks (finished chunks are skipped, so a resubmission resumes). 50,000 instances = 200 chunks of 250; with 16
# tasks each task runs ~13 chunks (~25 min each on one core, ~5.5 h). Submit through the guard, e.g.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name gen --array 0-15%8 --time 07:30:00 --cpus 1 --mem 3G \
#       --export N_TOTAL=50000,N_TASKS=16 scripts/pace/pgu_gen_data.sbatch
#SBATCH -N 1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=3G
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
N_TOTAL=${N_TOTAL:-50000}
N_TASKS=${N_TASKS:-16}
N_CHUNKS=$(( (N_TOTAL + 249) / 250 ))
for ((c = SLURM_ARRAY_TASK_ID; c < N_CHUNKS; c += N_TASKS)); do
  python3 scripts/rec_trm_data.py --chunk "$c" --n_total "$N_TOTAL" --seed 211 --out data/generated/recursive/pace
done
