#!/bin/bash
# rloop (arm B): regenerate the extrapolation test sets on the cluster (CPU only). The 12-hour training / validation /
# test data, the m1x baseline and extra polished-label chunks come from the shared jobs of the recursive study:
#   scripts/pace/trm_gen_base.sbatch        uc12 train / val / val_extra / val_extra2 / test_fresh (with full MILPs)
#   scripts/pace/trm_teachers.sbatch        uc12_m1x extra_0-5 + polish_0-5 and the m1x ensemble (the m1x baseline)
#   scripts/pace/trm_gen_data_array.sbatch  more polished chunks in data/generated/recursive/pace (same format; pass
#                                           --data_dir data/generated/recursive/pace to rec_rloop_train.py)
# This job only adds the zero-shot test sets of rloop's extrapolation study, with the commands of the original studies
# (docs/methods/m24.md, hard.md, ood.md); it was not run on PACE ICE.
#
# PLACEHOLDERS: check the PACE ICE documentation for partition / QOS / account names and fill them in or pass them.
#SBATCH -J rloop_data_extrap
##SBATCH -p PLACEHOLDER_CPU_PARTITION
##SBATCH -q PLACEHOLDER_QOS
##SBATCH -A PLACEHOLDER_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/rloop_data_extrap_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/pace/trm_env.sh          # shared environment (module / venv placeholders inside)
# uc24: 40 test instances of 24 h with full MILPs (300 s limit), ~2-3 core-hours
[ -f data/generated/uc24/test.npz ] || python3 scripts/uc_b3_gen.py --splits test --n_test 40
# congested benchmark (line ratings x 0.6): 60 test instances in 12 chunks with full MILPs (600 s limit)
python3 scripts/uc_hard_data.py --gen test --chunks 0-11
# line-outage shift: shift specifications, then the per-instance files (data/generated/uc12_ood/line_out/NNN.npz) are
# written by the robustness study's evaluation, which also solves its rules; restrict them with --rules (see --help)
python3 scripts/uc_ood_gen.py
python3 scripts/uc_ood_eval.py --shifts line_out --n 30
