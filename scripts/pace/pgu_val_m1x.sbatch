#!/usr/bin/env bash
# GPU round: re-validate the m1x baseline ensemble that trm_teachers.sbatch trained ON THIS CLUSTER, against this
# cluster's validation MILPs, so arm A's test plan does not borrow the CPU round's thresholds (another ensemble,
# other references). Output: results/recursive/gpu_round/trm_val_m1x.json.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name valm1x --cpus 2 --mem 12G --time 06:00:00 \
#       scripts/pace/pgu_val_m1x.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
[ -f data/generated/recursive/val/lpround.npz ] || { echo "val/lpround.npz missing"; exit 1; }
python3 scripts/rec_trm_val.py --tag m1x --res results/recursive/gpu_round --threads "${SLURM_CPUS_PER_TASK:-2}"
echo "VAL DONE m1x"
