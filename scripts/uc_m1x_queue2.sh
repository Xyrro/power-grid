#!/bin/bash
# m1x study, phases 2-4 (after phase 1 of uc_m1x_queue.sh), as run. One core (taskset -c 3), every step idempotent
# (trainings reload finished checkpoints, valfix and eval resume per instance, finished tuning runs are skipped).
#   setsid nohup scripts/uc_m1x_queue2.sh phase2 > data/generated/uc12_m1x/queue_phase2.log 2>&1 &
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
  # validation-only selection (rule in scripts/uc_m1x_select.py, written before the phase-2 results) and run A (the
  # 5-seed ensemble of the default family); then the temporal GNN on 2,000 polished instances, added as a candidate
  # after phase 2 (more data and the temporal head both passed the bar), the selection with it (--extended), 4 more
  # seeds of the winner (pol_n2000_gnnt), its ensemble, run B on error-cost scores ("he") and on the probabilities
  # themselves ("hg", added because the error-cost model was trained on the plain GNN's errors); the paper's criterion
  # (larger guarded validation fixed share) picks between B-he and B-hg
  step $PY scripts/uc_m1x_select.py
  step $PY scripts/uc_m1x_tune.py --job he_milp500_ens5:harm:milp500_gnn_ens5:0.01:adeq+rows --budget_min 150
  step $PY scripts/uc_m1x_train.py --label pol --n 2000 --kind gnnt --seed 0
  step $PY scripts/uc_m1x_select.py --extended
  W=pol_n2000_gnnt
  for s in 1 2 3 4; do step $PY scripts/uc_m1x_train.py --label pol --n 2000 --kind gnnt --seed $s; done
  step $PY scripts/uc_m1x_train.py --ensemble ${W}_ens5 ${W}_s0,${W}_s1,${W}_s2,${W}_s3,${W}_s4
  step $PY scripts/uc_m1x_tune.py --job he_${W}_ens5:harm:${W}_ens5:0.01:adeq+rows --budget_min 150
  step $PY scripts/uc_m1x_tune.py --job hg_${W}_ens5:prob:${W}_ens5:0.01:adeq+rows --budget_min 150
}

phase4() {
  # test, first 60 instances of test_fresh: full MILP + reference hybrid + faithful LtF BCE + runs A, B-he, B-hg
  step $PY scripts/uc_m1x_eval.py --runs he_milp500_ens5,he_pol_n2000_gnnt_ens5,hg_pol_n2000_gnnt_ens5 --start 0 --n 60
  step python3 scripts/uc_m1x_report.py
  # instances 60-119 with the reference hybrid and the selected rule only (budget: ~80 s per instance with every rule)
  step $PY scripts/uc_m1x_eval.py --runs hg_pol_n2000_gnnt_ens5 --skip_faithful --start 60 --n 60
  step python3 scripts/uc_m1x_report.py
}

"$@"
echo "[$(date +%H:%M:%S)] queue2 $* done"
