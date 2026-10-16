#!/usr/bin/env bash
# Wave 1: the go/no-go comparison for both recursive arms, submitted through the guard.
#
#   bash scripts/pace/pgu_wave1.sh --dry-run     # print the accounting and the sbatch lines, submit nothing
#   bash scripts/pace/pgu_wave1.sh               # submit
#
# Arm A (gpu_small: hidden 128, n 6, T 3, N_sup 16, batch 256), all on the same 50,000 polished-label instances:
#   w1-trm          the recursive model
#   w1-oneshot      same network, one pass per instance, SAME optimizer steps  (sees more distinct instances, because
#                   a control consumes `batch` instances per step while the recursive model consumes batch / n_sup)
#   w1-os-eqep      same network, one pass per instance, SAME number of instance visits as the recursive model
#   w1-deep         deeper non-recursive network, no weight sharing, same optimizer steps
#   w1-trm-nolp / w1-os-nolp   the pair above without the LP-relaxation inputs
# Arm B:
#   w1-rloop-nofb   the recursion with LP feedback switched off (the control for wave 2)
#
# Measured on an H200 by scripts/pace/pgu_smoke_gpu.sbatch: 0.54 s per step at this size, so 30,000 steps is about
# 4.5 h; an 8 h request leaves room for the larger data set and the validation passes, and four such jobs fit exactly
# inside the 32 GPU-hour running cap.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PGU_ROOT:?export PGU_ROOT first}"
DRY=${1:-}
G() { python3 scripts/pace/pgu_guard.py submit "$@" ${DRY:+--dry-run}; }
DATA='--data_dirs data/generated/uc12_m1x data/generated/recursive/pace'
N=${N_TRAIN:-50000}
STEPS=${STEPS:-30000}
EQEP=$((STEPS / 16))          # equal instance visits: the controls consume n_sup = 16 times more per step

A() {  # A <name> <hours> <extra train flags...>
  local name=$1 hrs=$2; shift 2
  G --kind gpu --name "$name" --gpus 1 --cpus 8 --mem 64G --time "$hrs" \
    --export "TAG=${name//-/_},PRESET=gpu_small,N_TRAIN=$N,EXTRA=--steps $STEPS $* $DATA" \
    scripts/pace/trm_train_gpu.sbatch
}

A w1-trm         08:00:00 --mode trm
A w1-oneshot     08:00:00 --mode oneshot
A w1-deep        08:00:00 --mode deep
A w1-trm-nolp    08:00:00 --mode trm --lp_feats 0
A w1-os-nolp     08:00:00 --mode oneshot --lp_feats 0
G --kind gpu --name w1-os-eqep --gpus 1 --cpus 8 --mem 64G --time 02:00:00 \
  --export "TAG=w1_os_eqep,PRESET=gpu_small,N_TRAIN=$N,EXTRA=--steps $EQEP --mode oneshot $DATA" \
  scripts/pace/trm_train_gpu.sbatch

# Arm B control: same recursion, LP feedback off. The LPs run in this job's own CPU cores.
G --kind gpu --name w1-rloop-nofb --gpus 1 --cpus 24 --mem 64G --time 08:00:00 \
  --export "TAG=w1_rloop_nofb,N_TRAIN=$N,EXTRA=--no_feedback --data_dir data/generated/recursive/pace" \
  scripts/pace/rloop_train.sbatch
