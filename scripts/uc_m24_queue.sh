#!/bin/bash
# m24 study: run sequence on core 1 (taskset -c 1), one job at a time, every step idempotent (resume = rerun the phase).
#   setsid nohup scripts/uc_m24_queue.sh phase1 > data/generated/uc24_m24/queue_phase1.log 2>&1 &
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 1 python3"
D=data/generated/uc24_m24
step() { echo "[$(date +%H:%M:%S)] $*"; "$@" 2>&1 | grep --line-buffered -v -i "^running highs\|^copyright"; }

phase1() {
  # pilot teacher on 100 of the 120 MILP-labelled training instances, polishing pilot on the other 20
  step $PY scripts/uc_m24_train.py --name t0p --seeds 0 --lab 100 --epochs 60
  [ -f $D/pilot.json ] || step $PY scripts/uc_m24_data.py --pilot --teacher t0p --tls 10,20
  # teacher: 3 temporal GNNs on the 120 MILP-labelled instances
  step $PY scripts/uc_m24_train.py --name t0 --seeds 0,1,2 --lab 120 --epochs 60
  step $PY scripts/uc_m24_train.py --ensemble t0
  # 400 new training-day scenarios (LP relaxation + label-free labels, no MILP)
  step $PY scripts/uc_m24_data.py --gen --chunks 0-7
}

phase2() {
  # 200 more scenarios (600 new in total; the pilot gave 7.7 s per polished label, so the budget allows it)
  step $PY scripts/uc_m24_data.py --gen --chunks 0-11
  # polished labels with the teacher t0 (10 s reduced MILPs, chosen by the pilot), orig 120-299 then the new scenarios
  step $PY scripts/uc_m24_data.py --polish --teacher t0 --src orig,extra --tl 10
  # data-size check (one seed): 120 MILP + 180 polished; then the 5-member ensemble on all 900 instances
  step $PY scripts/uc_m24_train.py --name mid --seeds 0 --lab 120 --pol orig --epochs 50
  step $PY scripts/uc_m24_train.py --name fin --seeds 0,1,2,3,4 --lab 120 --pol orig,extra --epochs 40
  step $PY scripts/uc_m24_train.py --ensemble fin
  # guard-aware Learning to Fix on the ensemble probabilities, eps = 1 %, 40 validation instances
  step $PY scripts/uc_m24_tune.py --run hg_fin --src fin_ens --budget_min 120
  # test, 40 instances
  step $PY scripts/uc_m24_eval.py --run hg_fin --src fin
}

"$@"
echo "[$(date +%H:%M:%S)] queue $* done"
