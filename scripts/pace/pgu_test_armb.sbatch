#!/usr/bin/env bash
# GPU round, arm B: the single test pass on the 120 test_fresh instances, one array task per shard, one core per
# task. Both models (LP feedback on / off) are run in the same task, each with its own paired full-MILP solves.
# Settings come only from results/recursive/gpu_round/rloop_select_gpu.json (validation-chosen; no CPU-round defaults),
# and each run resumes only from its own record file in results/recursive/gpu_round/.
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name testb --array 0-7%8 --cpus 1 --mem 8G \
#       --time 06:00:00 scripts/pace/pgu_test_armb.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
SEL=results/recursive/gpu_round/rloop_select_gpu.json
N=${SLURM_ARRAY_TASK_COUNT:-8}
S=${SLURM_ARRAY_TASK_ID:-0}
for M in w2_rloop_rl w1_rloop_nofb_rl; do
  [ -f "data/generated/recursive/rloop/$M.pt" ] || { echo "missing checkpoint $M"; exit 1; }
  python3 scripts/rec_rloop_eval.py --bench uc12 --select "$SEL" --model "$M" --skip m1x,guided,union --seeded 0 \
    --shard "$S" --nshards "$N" --out "results/recursive/gpu_round/rloop_eval_uc12_${M}_s${S}.jsonl"
done
echo "TEST B DONE shard $S"
