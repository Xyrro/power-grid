#!/usr/bin/env bash
# Step 1 (CPU): regenerate the 12-hour RTS-GMLC splits WITH full MILPs (labels of train.npz, references of val / test).
# ~980 MILPs of ~30 s (60 s limit) = ~8 core-hours; one job, WORKERS processes.
#SBATCH -J uc12-base
#SBATCH -A <ACCOUNT>                 # PLACEHOLDER (check PACE ICE docs: charge account / none on ICE)
#SBATCH -p <CPU_PARTITION>           # PLACEHOLDER (check PACE ICE docs)
#SBATCH -q <QOS>                     # PLACEHOLDER (check PACE ICE docs; delete the line if not used)
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"   # submit from the repository root (mkdir -p logs first)
W=${SLURM_CPUS_PER_TASK:-8}
g() { python3 scripts/uc_gen.py --cfg uc12 --workers "$W" "$@"; }
[ -f data/generated/uc12/train.npz ]      || g --split train      --n 500 --seed 11
[ -f data/generated/uc12/val.npz ]        || g --split val        --n 60  --seed 12
[ -f data/generated/uc12/val_extra.npz ]  || g --split val_extra  --n 120 --seed 31
[ -f data/generated/uc12/val_extra2.npz ] || g --split val_extra2 --n 180 --seed 41
[ -f data/generated/uc12/test_fresh.npz ] || g --split test_fresh --n 120 --seed 23
