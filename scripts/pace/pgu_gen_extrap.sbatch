#!/usr/bin/env bash
# The zero-shot extrapolation test sets, which a fresh checkout does not have: 24-hour instances, the congested
# benchmark (line ratings x 0.6) and the line-outage shift. Both arms evaluate on these, so they are on the critical
# path for the final comparison. CPU only; every step is skipped if its output already exists, so a resubmission
# resumes.
#
# Submitted as a 3-task array, one benchmark per task, so they run in parallel:
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name extrap --array 0-2%3 --cpus 8 --mem 24G \
#       --time 10:00:00 scripts/pace/pgu_gen_extrap.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
set -x
W=${SLURM_CPUS_PER_TASK:-8}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)  # 24-hour benchmark: 40 test instances with full MILPs (uc_b3_gen supports worker processes)
      [ -f data/generated/uc24/test.npz ] || python3 scripts/uc_b3_gen.py --splits test --n_test 40 --workers "$W"
      ;;
  1)  # congested benchmark (line ratings x 0.6): 60 test instances in 12 chunks
      python3 scripts/uc_hard_data.py --gen test --chunks 0-11
      ;;
  2)  # line-outage shift: shift specifications, then the per-instance files
      python3 scripts/uc_ood_gen.py
      python3 scripts/uc_ood_eval.py --shifts line_out --n 30
      ;;
esac
echo "EXTRAP TASK ${SLURM_ARRAY_TASK_ID:-0} DONE"
