#!/bin/bash
# rloop (arm B): paired test evaluation on CPU, one Slurm array task per shard, one core per task.
# Each task solves, per instance and back to back on its core: the full MILP (0.1 %, 0.5 %), the LP relaxation, the
# learning-free LP rounding baseline, the m1x ensemble end to end (12 h), the rloop model (K outer steps) and its
# safety net. Speed-ups are only comparable inside a task (same core); do not oversubscribe the node.
#
# PLACEHOLDERS: check the PACE ICE documentation for partition / QOS / account names and fill them in or pass them:
#   BENCH=uc12 sbatch -p <cpu-partition> --array=0-7 scripts/pace/rloop_eval.sbatch
# BENCH: uc12 (test_fresh, 120) | uc24 (40) | hard (60) | line_out (30). Model / K: results/recursive/rloop_select.json
# (written after validation; override with MODEL=<tag> K_SEL=<k>).
#
#SBATCH -J rloop_eval
##SBATCH -p PLACEHOLDER_CPU_PARTITION
##SBATCH -q PLACEHOLDER_QOS
##SBATCH -A PLACEHOLDER_ACCOUNT
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 08:00:00
#SBATCH --array=0-7
#SBATCH -o logs/rloop_eval_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/pace/trm_env.sh          # shared environment (module / venv placeholders inside)
export OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BENCH=${BENCH:-uc12}
NSH=${SLURM_ARRAY_TASK_COUNT:-8}
ARGS=""
[ -n "${MODEL:-}" ] && ARGS="$ARGS --model $MODEL"
[ -n "${K_SEL:-}" ] && ARGS="$ARGS --k_sel $K_SEL"
python3 scripts/rec_rloop_eval.py --bench "$BENCH" --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "$NSH" $ARGS
# after all tasks: python3 scripts/rec_rloop_report.py
