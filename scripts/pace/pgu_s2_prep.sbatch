#!/usr/bin/env bash
# Step 2 of docs/unified_metrics.md, phase 0 (CPU, any node; no timing is measured here): what the re-timing needs that
# this cluster does not have yet. Every output goes to results/unified/retime/ or to new files under data/generated/;
# nothing of the original studies' records changes. STAGE=all runs every piece in parallel (one job, 16 CPUs):
#   shifts  the five shifted 12-hour sets not generated here yet (30 instances each, with their reference MILP), then the
#           input checks (test sets, shifted instances, ensemble probabilities against the saved records)
#   ltfx    Learning to Fix's thresholds on THIS cluster's kNN (eps 1 / 5 / 10 %) and BCE GNN (eps 1 %), validation only
#           (scripts/uc_ltfx_prep.py, uc_ltfx_tune.py; the original study's 180 instances and settings)
#   hg      the recommended pipeline's guard-aware thresholds on THIS cluster's 5-member ensemble (uc_m1x_tune.py, eps
#           1 %, adequacy + rows, 360 validation instances, as the original study)
#   early   the early one-shot models (scripts/uc_model1.py, as originally run: 80 epochs, 120 REINFORCE steps)
#   seeds   two more training runs of the pipeline: ensembles of the networks with seeds 5-9 and 10-14 (uc_m1x_train.py,
#           the recipe of seeds 0-4), each with its own thresholds tuned like hg
# A single STAGE re-runs one piece (resubmission after a failure). Status: results/unified/retime/prep/status.txt.
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name s2-prep --cpus 16 --mem 64G --time 06:00:00 \
#       --export STAGE=all scripts/pace/pgu_s2_prep.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
R=results/unified/retime
mkdir -p "$R/prep" "$R/tune" "$R/models"
echo "node $(hostname): $(lscpu | sed -n 's/^Model name: *//p'), CPUs $(nproc)"

shifts() {
  python3 scripts/uc_ood_eval.py --instances_only --shifts load_hi,load_lo,ren_hi,gen_out,night --n 30 \
    --out "$R/prep/ood_instances.jsonl" && python3 scripts/unified_s2_checks.py inputs
}
ltfx() {
  python3 scripts/uc_ltfx_prep.py --models knn,bce --res "$R/tune" && \
  python3 scripts/uc_ltfx_tune.py --jobs knn:0.01,knn:0.05,knn:0.1,bce:0.01 --workers 4 --res "$R/tune"
}
ltfx10() {   # Learning to Fix's kNN at eps 10 % alone, with a longer budget (did not converge within 150 min here)
  python3 scripts/uc_ltfx_tune.py --jobs knn:0.1 --workers 1 --budget_min 720 --res "$R/tune"
}
hg() {
  python3 scripts/uc_m1x_tune.py --job hgP:prob:pol_n2000_gnnt_ens5:0.01:adeq+rows --res "$R/tune"
}
early() {
  python3 scripts/uc_model1.py --cfg uc12 --epochs 80 --rl_steps 120 --workers 3 --n_fix 0 --test_split test_fresh \
    --out_dir "$R/models" && python3 scripts/unified_s2_checks.py early
}
seeds() {
  local s pids=() rc=0
  for s in 5 6 7 8 9 10 11 12 13 14; do
    python3 scripts/uc_m1x_train.py --label pol --n 2000 --kind gnnt --seed "$s" --log "$R/tune/m1x_train.json" \
      > "$R/prep/seed_$s.log" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || return 1
  python3 scripts/uc_m1x_train.py --log "$R/tune/m1x_train.json" --ensemble pol_n2000_gnnt_ens5b \
    pol_n2000_gnnt_s5,pol_n2000_gnnt_s6,pol_n2000_gnnt_s7,pol_n2000_gnnt_s8,pol_n2000_gnnt_s9 || return 1
  python3 scripts/uc_m1x_train.py --log "$R/tune/m1x_train.json" --ensemble pol_n2000_gnnt_ens5c \
    pol_n2000_gnnt_s10,pol_n2000_gnnt_s11,pol_n2000_gnnt_s12,pol_n2000_gnnt_s13,pol_n2000_gnnt_s14 || return 1
  python3 scripts/uc_m1x_tune.py --job hgPb:prob:pol_n2000_gnnt_ens5b:0.01:adeq+rows --res "$R/tune" & local b=$!
  python3 scripts/uc_m1x_tune.py --job hgPc:prob:pol_n2000_gnnt_ens5c:0.01:adeq+rows --res "$R/tune" & local c=$!
  wait "$b" && wait "$c"
}
run() {   # run a stage, log it, record its exit code (trm_env.sh sets -e: a failing command must not end the job unrecorded)
  local t0=$SECONDS rc=0
  if "$1" > "$R/prep/$1.log" 2>&1; then rc=0; else rc=$?; fi
  echo "$1 rc=$rc $((SECONDS - t0))s $(date '+%Y-%m-%dT%H:%M:%S')" >> "$R/prep/status.txt"
  return $rc
}
STAGE=${STAGE:?STAGE not set}
if [ "$STAGE" = all ]; then
  pids=()
  for st in shifts ltfx hg early seeds; do run "$st" & pids+=($!); done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
  if run "$STAGE"; then fail=0; else fail=1; fi
fi
if python3 scripts/unified_s2_checks.py tune > "$R/prep/check_tune.log" 2>&1; then c=0; else c=$?; fi
echo "check_tune rc=$c" >> "$R/prep/status.txt"
cat "$R/prep/status.txt"
[ "$fail" = 0 ] && echo "S2 PREP DONE $STAGE" || { echo "S2 PREP FAILED $STAGE"; exit 1; }
