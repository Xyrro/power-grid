#!/usr/bin/env bash
# Arm A, validation pass: the downstream metrics (cost gap, instances served) for one trained tag per array task,
# so the tags are validated in parallel. This is the pass that decides the go/no-go, because the training logs only
# report BCE and mis-predicted unit-hours, which §3 U7 of the research log shows can mis-rank methods.
#
# The checkpoint of each task is named EXPLICITLY in TAGS (e.g. `w1b_oneshot` for the final state, `w1b_oneshot_best`
# for the best-validation-loss one); nothing is swapped in silently, and every result file records the checkpoint's
# training step (scripts/rec_trm_val.py). Validation loss is NOT a safe stand-in for the downstream gap here: the
# controls' final checkpoints have higher validation loss but cheaper schedules (docs/recursive/gpu_round.md).
#
# lpround is validated first, in its own task, because every model task reads val/lpround.npz to build the mixed
# pipeline: submit it alone, then the model array with --dependency afterok:<lpround job>  (VAL_TAGS overrides TAGS).
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name vala --array 0-7%8 --cpus 2 --mem 16G \
#       --time 02:00:00 --constraint cpu-gnr scripts/pace/pgu_val_arma.sbatch
# Afterwards:  python3 scripts/rec_trm_select.py --final <trm tag> --controls <control tags>
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
if [ -n "${VAL_TAGS:-}" ]; then read -r -a TAGS <<< "$VAL_TAGS"; else
  TAGS=(w1_trm30 w1_oneshot w1_deep w1b_oneshot_best w1b_deep_best w1_os_eqep w1_trm_nolp w1_os_nolp); fi
T=${TAGS[${SLURM_ARRAY_TASK_ID:-0}]}
M=data/generated/recursive/models
if [ "$T" != lpround ] && [ "$T" != m1x ] && [ ! -f "$M/$T.pt" ]; then echo "missing checkpoint $M/$T.pt"; exit 1; fi
if [ "$T" != lpround ] && [ ! -f data/generated/recursive/val/lpround.npz ]; then
  echo "val/lpround.npz is missing: validate lpround first (the mixed pipeline needs it)"; exit 1; fi
if [ "$T" != lpround ] && [ "$T" != m1x ]; then need_bf16; fi
echo "validating $T"
python3 scripts/rec_trm_val.py --tag "$T" --threads "${SLURM_CPUS_PER_TASK:-2}"
echo "VAL DONE $T"
