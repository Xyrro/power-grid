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

phase3() {
  # validation-only selection (rule in scripts/uc_m1x_select.py), then the downstream tuning runs
  step $PY scripts/uc_m1x_select.py
  B=$(python3 -c "import json; print(json.load(open('results/uc12/m1x_select.json'))['run_B'])")
  W=$(python3 -c "import json; print(json.load(open('results/uc12/m1x_select.json'))['winner'] or '')")
  step $PY scripts/uc_m1x_tune.py --job he_milp500_ens5:harm:milp500_gnn_ens5:0.01:adeq+rows --budget_min 150
  if [ -n "$W" ]; then
    LAB=$(echo $W | cut -d_ -f1); N=$(echo $W | cut -d_ -f2 | tr -d n); K=$(echo $W | cut -d_ -f3)
    for s in 1 2 3 4; do step $PY scripts/uc_m1x_train.py --label $LAB --n $N --kind $K --seed $s; done
    step $PY scripts/uc_m1x_train.py --ensemble ${W}_ens5 ${W}_s0,${W}_s1,${W}_s2,${W}_s3,${W}_s4
    step $PY scripts/uc_m1x_tune.py --job he_${W}_ens5:$B:0.01:adeq+rows --budget_min 150
  else
    step $PY scripts/uc_m1x_tune.py --job hg_milp500_ens5:$B:0.01:adeq+rows --budget_min 150
  fi
}

phase4() {
  # test: first 60 instances of test_fresh, then the other 60
  W=$(python3 -c "import json; print(json.load(open('results/uc12/m1x_select.json'))['winner'] or '')")
  if [ -n "$W" ]; then RB=he_${W}_ens5; else RB=hg_milp500_ens5; fi
  RUNS=he_milp500_ens5
  [ -f results/uc12/m1x_tune_$RB.json ] && RUNS=$RUNS,$RB
  step $PY scripts/uc_m1x_eval.py --runs $RUNS --start 0 --n 60
  step python3 scripts/uc_m1x_report.py
  step $PY scripts/uc_m1x_eval.py --runs $RUNS --start 60 --n 60
  step python3 scripts/uc_m1x_report.py
}

phase23() { phase2; phase3; phase4; }

"$@"
echo "[$(date +%H:%M:%S)] queue2 $* done"
