#!/usr/bin/env bash
# B3 (uc24) end-to-end run, every heavy step sequential (2 worker processes at most).
#   setsid nohup bash scripts/uc_b3_chain.sh > results/uc24/b3_run.log 2>&1 &
#   START=4 bash scripts/uc_b3_chain.sh     # resume from step 4
set -euo pipefail
cd "$(dirname "$0")/.."
START=${START:-1}
STOP=${STOP:-99}
NTEST=${NTEST:-40}
NFIXTEST=${NFIXTEST:-40}
step() { [ "$1" -ge "$START" ] && [ "$1" -le "$STOP" ]; }
ts() { date +"%H:%M:%S"; }

# 1. MILP-free data: train (300) and val (30): scenario, LP relaxation, repaired relaxation labels
step 1 && { echo "[$(ts)] step 1: train/val data (no MILP)"; python3 scripts/uc_b3_gen.py --splits train,val --n_train 300 --n_val 30 --val_mode lp; }
# 2. label-free training: imitation (two repaired-relaxation targets) -> REINFORCE with the LP critic -> val selection
step 2 && { echo "[$(ts)] step 2: training"; python3 scripts/uc_b3_train.py --stage bce,rl,select --epochs 60 --rl_steps 80 --bs 12 --n_samples 6 --val_every 20; }
# 3. fixing ranking selection on val (reduced MILPs only, label-free)
step 3 && { echo "[$(ts)] step 3: fixing selection on val"; python3 scripts/uc_b3_fix.py --split val --n 6 --ratios 0.8,0.9,0.95 --rankings bce_g,bce_asym_g,rl_g --select; }
# 4. test data with the full MILP (300 s, 0.1 %), incumbent logs
step 4 && { echo "[$(ts)] step 4: test MILPs"; python3 scripts/uc_b3_gen.py --splits test --n_test "$NTEST"; }
# 5. end-to-end on test
step 5 && { echo "[$(ts)] step 5: end-to-end test"; python3 scripts/uc_b3_train.py --stage test; }
# 6. solver acceleration on test: RACLearn-style confidence vs the val-selected guarded ranking; back-to-back subset
step 6 && { echo "[$(ts)] step 6: fixing test"; python3 scripts/uc_b3_fix.py --split test --n "$NFIXTEST" --ratios 0.8,0.9,0.95 --rankings raclearn,best --resolve_full 6; }
step 7 && { echo "[$(ts)] step 7: report"; python3 scripts/uc_b3_report.py > /dev/null; }
# 8. extra (not selected on val): REINFORCE-probability rankings, unguarded and guarded, for the speed / quality frontier
step 8 && { echo "[$(ts)] step 8: extra rankings on test"; python3 scripts/uc_b3_fix.py --split test --n "$NFIXTEST" --ratios 0.9,0.95 --rankings rl,rl_g --tag _extra; }
step 9 && { echo "[$(ts)] step 9: report"; python3 scripts/uc_b3_report.py > /dev/null; }
echo "[$(ts)] chain done"
