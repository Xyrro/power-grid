#!/usr/bin/env bash
# Step 5 (CPU): validation pass, selection, and the paired test evaluation. The test evaluation solves the full MILP
# (0.1 % and 0.5 %) and every pipeline of an instance back to back on ONE core (paired timing), so it runs as an array
# of single-core shards; TRM inference also runs on that core (CPU deployment timing).
# Usage:  TAGS="paper paper_rl" CONTROLS="oneshot" sbatch scripts/pace/trm_eval_cpu.sbatch             (validation + select)
#         sbatch --array=0-7 --export=ALL,STAGE=test scripts/pace/trm_eval_cpu.sbatch                   (test, 8 shards)
#         python3 scripts/rec_trm_report.py                                                         (tables, afterwards)
#SBATCH -J trm-eval
#SBATCH -A <ACCOUNT>                 # PLACEHOLDER
#SBATCH -p <CPU_PARTITION>           # PLACEHOLDER
#SBATCH -q <QOS>                     # PLACEHOLDER
#SBATCH -N 1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.out
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"   # submit from the repository root (mkdir -p logs first)
STAGE=${STAGE:-val}
if [ "$STAGE" = "val" ]; then
  for t in m1x lpround ${TAGS:-} ${CONTROLS:-}; do python3 scripts/rec_trm_val.py --tag "$t"; done
  python3 scripts/rec_trm_select.py --final ${TAGS} --controls ${CONTROLS:-}
else
  N=${SLURM_ARRAY_TASK_COUNT:-1}
  python3 scripts/rec_trm_eval.py --split test_fresh --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "$N"
fi
