#!/bin/bash
# uc24ltf: the whole pipeline on ONE core, one step after the other (the machine is shared).
#   bash scripts/uc_uc24ltf_chain.sh [pid to wait for] [first stage: prep | tune | eval | report]
# 1. (scripts/uc_uc24ltf_gen.py, started separately) validation full MILPs + labelled training schedules
# 2. probabilities, 3. Learning-to-Fix tuning (6 jobs), 4. test evaluation, 5. report
cd "$(dirname "$0")/.." || exit 1
if [ -n "$1" ] && [ "$1" != "0" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi
STAGE=${2:-prep}
echo "[chain] start $(date), from stage $STAGE"
if [ "$STAGE" = "prep" ]; then
  python3 scripts/uc_uc24ltf_prep.py > results/uc24/uc24ltf_prep.log 2>&1 || exit 1
  echo "[chain] prep done $(date)"; STAGE=tune
fi
if [ "$STAGE" = "tune" ]; then
  python3 scripts/uc_uc24ltf_tune.py --jobs knn:0.01,gnn:0.05,gnn:0.05:g,gnn:0.01,gnn:0.01:g,knn:0.05 \
      --relax_tl 8 --check_tl 60 --budget_min 22 > results/uc24/uc24ltf_tune_run.log 2>&1 || exit 1
  echo "[chain] tuning done $(date)"; STAGE=eval
fi
if [ "$STAGE" = "eval" ]; then
  python3 scripts/uc_uc24ltf_eval.py --rules knn_1,knn_5,gnn_1,gnn_5,gnn_g_1,gnn_g_5 \
      --old "95%|best=bce_g,95%|rl" --b2b 4 > results/uc24/uc24ltf_eval_run.log 2>&1 || exit 1
  echo "[chain] eval done $(date)"; STAGE=report
fi
python3 scripts/uc_uc24ltf_report.py > results/uc24/uc24ltf_report_run.log 2>&1
echo "[chain] all done $(date)"
