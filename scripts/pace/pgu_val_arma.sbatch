#!/usr/bin/env bash
# Arm A, validation pass: the downstream metrics (cost gap, instances served) for one trained tag per array task,
# so the tags are validated in parallel. This is the pass that decides the go/no-go, because the training logs only
# report BCE and mis-predicted unit-hours, which §3 U7 of the research log shows can mis-rank methods.
#
# Each model is validated at its BEST-validation checkpoint when one exists (<tag>_best.pt, written by the training
# loop), otherwise at its final state. Evaluating a model that overfitted in its degraded final state measures
# nothing useful: the non-recursive controls here lost 10-90 % of their validation BCE after their best step.
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name vala --array 0-6%7 --cpus 2 --mem 16G \
#       --time 16:00:00 scripts/pace/pgu_val_arma.sbatch
# Afterwards:  python3 scripts/rec_trm_select.py --final <trm tag> --controls <control tags>
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
TAGS=(w1_trm w1b_oneshot w1_os_eqep w1b_deep w1_trm_nolp w1_os_nolp lpround)
T=${TAGS[${SLURM_ARRAY_TASK_ID:-0}]}
M=data/generated/recursive/models
if [ -f "$M/${T}_best.pt" ]; then
  T="${T}_best"                      # the best-validation checkpoint of that run
fi
echo "validating $T"
python3 scripts/rec_trm_val.py --tag "$T" --threads "${SLURM_CPUS_PER_TASK:-2}"
echo "VAL DONE $T"
