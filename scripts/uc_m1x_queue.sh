#!/bin/bash
# m1x study: the run sequence, one job at a time on one core (taskset -c 3), every step idempotent (resume = rerun).
#   setsid nohup scripts/uc_m1x_queue.sh phase1 > data/generated/uc12_m1x/queue_phase1.log 2>&1 &
# Phases 2-4 (ensembles, architecture, validation fixing, selection, tuning, test): scripts/uc_m1x_queue2.sh
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 3 python3"
D=data/generated/uc12_m1x
step() { echo "[$(date +%H:%M:%S)] $*"; "$@" 2>&1 | grep --line-buffered -v -i "^running highs\|^copyright"; }

phase1() {
  # extra scenarios + label-free labels (3500, 14 chunks)
  step $PY scripts/uc_m1x_data.py --gen
  # metrics of the existing probability sources
  step $PY scripts/uc_m1x_train.py --refs
  # polishing setting chosen from the pilot: confidence ranking, 90 %, LP-relaxation guard, 5 s
  grep -q "conf q0.90 tl5 +lp" $D/pilot.json || step $PY scripts/uc_m1x_data.py --pilot --pilot_cfgs "conf q0.90 tl5 +lp"
  # learning curve, label-free labels
  for n in 500 1000 2000 4000; do step $PY scripts/uc_m1x_train.py --label lf --n $n --kind gnn --seed 0; done
  # teacher-polished labels for 1500 extra instances (-> 2000 training instances)
  step $PY scripts/uc_m1x_data.py --polish --rank conf --ratio 0.9 --tl 5 --lp --chunks 0,1,2,3,4,5
  for n in 1000 2000; do step $PY scripts/uc_m1x_train.py --label pol --n $n --kind gnn --seed 0; done
}

"$@"
echo "[$(date +%H:%M:%S)] queue $* done"
