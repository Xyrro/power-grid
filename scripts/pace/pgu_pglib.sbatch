#!/usr/bin/env bash
# PGLib-UC California (docs/methods/pglib.md, round 8 in docs/methods/agg.md): data, models and Learning to Fix's tuning
# regenerated on PACE with the commands of pglib.md section 9 (same seeds, so the same instances), then the round 8
# evaluation (scripts/agg_pglib_eval.py). The PGLib-UC instances live outside the repo in $PGU_ROOT/ext/pglib-uc.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name pglib-prep --cpus 8 --mem 48G --time 04:00:00 \
#       --export STAGES=gen+models+ltf scripts/pace/pgu_pglib.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
export PGLIB=$PGU_ROOT/ext/pglib-uc OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${STAGES:?STAGES=gen+models+ltf}"
NW=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), $NW CPUs, stages $STAGES"
mkdir -p data/generated/pglib_ca results/pglib
has() { [[ "+$STAGES+" == *"+$1+"* ]]; }
if has gen; then
  [ -f data/generated/pglib_ca/train.npz ] || python3 scripts/uc_pglib_gen.py --split train --n 160 --seed 1 --mode lp --T 24 --workers "$NW"
  [ -f data/generated/pglib_ca/val.npz ] || python3 scripts/uc_pglib_gen.py --split val --n 30 --seed 2 --mode milp --T 24 --tl 900 --gap 0.001 --workers "$NW"
  [ -f data/generated/pglib_ca/test.npz ] || python3 scripts/uc_pglib_gen.py --split test --n 30 --seed 3 --mode lp --T 24 --workers "$NW"
fi
if has models; then
  python3 scripts/uc_pglib_train.py --stage bce --T 24 --epochs 40
  python3 scripts/uc_pglib_train.py --stage st --T 24 --epochs 40 --st_n 100 --st_tl 60 --workers "$NW"
  python3 scripts/uc_pglib_train.py --stage probs --T 24
fi
if has ltf; then
  python3 scripts/uc_pglib_ltf.py --jobs knn:0.01 --workers 1 --n_val 30 --T 24 --check_tl 120 --relax_tl 20 --budget_min 60
fi
echo "PGLIB $STAGES DONE"
