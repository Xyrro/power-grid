#!/usr/bin/env bash
# GPU round, arm A: the single test pass on the 120 test_fresh instances, one array task per shard, one core per task
# (paired timing: the full MILPs and every pipeline of an instance run back to back on that core).
# The plan is results/recursive/gpu_round/trm_select_gpu.json, written from validation only by:
#   python3 scripts/rec_trm_select.py --final w1_trm30 --controls w1_oneshot w1_deep w1b_oneshot_best w1b_deep_best \
#       --val_dirs results/recursive/gpu_round results/recursive --out results/recursive/gpu_round/trm_select_gpu.json
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name testa --array 0-7%8 --cpus 1 --mem 12G \
#       --time 08:00:00 scripts/pace/pgu_test_arma.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
PLAN=results/recursive/gpu_round/trm_select_gpu.json
[ -f "$PLAN" ] || { echo "missing $PLAN (run rec_trm_select.py first)"; exit 1; }
N=${SLURM_ARRAY_TASK_COUNT:-8}
S=${SLURM_ARRAY_TASK_ID:-0}
python3 scripts/rec_trm_eval.py --split test_fresh --shard "$S" --nshards "$N" --plan "$PLAN" --round gpu \
  --out "results/recursive/gpu_round/trm_eval_test_fresh_s${S}.jsonl" --trm_device cpu
echo "TEST A DONE shard $S"
