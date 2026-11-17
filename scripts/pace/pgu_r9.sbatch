#!/usr/bin/env bash
# Round 9 (docs/methods/agg.md): the probing-margin feature of the training / validation data (scripts/r9_margins.py),
# then five seeds of the temporal GNN with that feature (scripts/uc_m1x_train.py --margin), one process per core.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name r9 --cpus 6 --mem 32G --time 03:00:00 scripts/pace/pgu_r9.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
export OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p')"
F="data/generated/uc12/train.npz data/generated/uc12/val.npz data/generated/uc12/val_extra.npz data/generated/uc12/val_extra2.npz"
for k in 0 1 2 3 4 5; do F="$F data/generated/uc12_m1x/extra_$k.npz"; done
pids=()
for f in $F; do python3 -W ignore scripts/r9_margins.py --files "$f" & pids+=($!); while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 2; done; done
for p in "${pids[@]}"; do wait "$p"; done
echo "margins done"
pids=()
for s in 0 1 2 3 4; do python3 -W ignore scripts/uc_m1x_train.py --label pol --n 2000 --kind gnnt --seed "$s" --margin \
  --log results/uc12/m1x_train_r9.json > "results/uc12/r9_train_s$s.out" 2>&1 & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
python3 -W ignore scripts/uc_m1x_train.py --ensemble pol_n2000_gnnt_mg_ens5 pol_n2000_gnnt_mg_s0,pol_n2000_gnnt_mg_s1,pol_n2000_gnnt_mg_s2,pol_n2000_gnnt_mg_s3,pol_n2000_gnnt_mg_s4 \
  --margin --log results/uc12/m1x_train_r9.json || fail=$((fail + 1))
echo "R9 DONE: $fail failed"
