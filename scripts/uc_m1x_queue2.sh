#!/bin/bash
# m1x study, phase 2 (after phase 1 of uc_m1x_queue.sh): ensemble and architecture at 500 MILP labels, guarded-rule
# validation check of every source. One core (taskset -c 3), every step idempotent.
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 3 python3"
step() { echo "[$(date +%H:%M:%S)] $*"; "$@" 2>&1 | grep --line-buffered -v -i "^running highs\|^copyright"; }

phase2() {
  # reproduction of the reference recipe (seed 0 = uc_model1_4.pt) and two more seeds for a 5-member ensemble
  for s in 0 3 4; do step $PY scripts/uc_m1x_train.py --label pol --n 500 --kind gnn --seed $s; done
  step $PY scripts/uc_m1x_train.py --ensemble milp500_gnn_ens5 ref_bce_s0,ref_bce_s1,ref_bce_s2,pol_n500_gnn_s3,pol_n500_gnn_s4
  # architecture at equal data (500 MILP labels)
  for k in gnnt mlp; do step $PY scripts/uc_m1x_train.py --label pol --n 500 --kind $k --seed 0; done
  # downstream guarded rule on 60 validation instances
  step $PY scripts/uc_m1x_valfix.py --tags ref_bce_s0,ref_bce_ens3,milp500_gnn_ens5,lf_n500_gnn_s0,lf_n1000_gnn_s0,lf_n2000_gnn_s0,lf_n4000_gnn_s0,pol_n1000_gnn_s0,pol_n2000_gnn_s0,pol_n500_gnnt_s0,pol_n500_mlp_s0
}

"$@"
echo "[$(date +%H:%M:%S)] queue2 $* done"
