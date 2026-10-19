#!/usr/bin/env bash
# GPU round, arm A: the single test pass on the 120 test_fresh instances, one array task per shard, one core per task
# (paired timing: the full MILPs and every pipeline of an instance run back to back on that core).
# The plan is results/recursive/gpu_round/trm_select_gpu.json, written from validation only by:
#   python3 scripts/rec_trm_select.py --final w1_trm30 --controls w1_oneshot w1_deep w1b_oneshot_best w1b_deep_best \
#       --val_dirs results/recursive/gpu_round results/recursive --out results/recursive/gpu_round/trm_select_gpu.json
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name testa --array 0-7%8 --cpus 1 --mem 12G \
#       --time 08:00:00 --constraint cpu-gnr scripts/pace/pgu_test_arma.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
# The TRM models run in bfloat16 (cfg.amp) also on CPU. A CPU without bf16 units (ICE's AMD EPYC 7513 and Xeon Gold
# 6226 nodes) emulates it many times slower: a validation that takes 31 min on the Granite Rapids node did not finish
# in 6 h on an EPYC node. Every arm A validation ran on Granite Rapids; submit with --constraint cpu-gnr so the test
# runs on the same hardware (same bf16 kernels, and paired timings on the node the validation timings came from).
need_bf16() {
  echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
  grep -qm1 -wE 'amx_bf16|avx512_bf16' /proc/cpuinfo || {
    echo "refusing: this CPU has no bf16 units (submit with --constraint cpu-gnr)"; exit 1; }
}
need_bf16
PLAN=results/recursive/gpu_round/trm_select_gpu.json
[ -f "$PLAN" ] || { echo "missing $PLAN (run rec_trm_select.py first)"; exit 1; }
N=${SLURM_ARRAY_TASK_COUNT:-8}
S=${SLURM_ARRAY_TASK_ID:-0}
python3 scripts/rec_trm_eval.py --split test_fresh --shard "$S" --nshards "$N" --plan "$PLAN" --round gpu \
  --out "results/recursive/gpu_round/trm_eval_test_fresh_s${S}.jsonl" --trm_device cpu
echo "TEST A DONE shard $S"
