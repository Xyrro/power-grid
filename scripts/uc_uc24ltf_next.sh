#!/bin/bash
# uc24ltf, re-planned queue (compute budget): after the guard-aware eps = 5 % tuning job finishes, stop the running
# tuning queue (GNN eps = 1 % jobs dropped), tune kNN at eps = 5 %, then evaluate on test and report.
#   bash scripts/uc_uc24ltf_next.sh <chain bash pid> <chain inner pid> <tune python pid>
cd "$(dirname "$0")/.." || exit 1
until [ -f results/uc24/uc24ltf_tune_bce_g_5.json ]; do sleep 2; done
kill "$1" "$2" 2>/dev/null; kill "$3" 2>/dev/null
sleep 2
rm -f results/uc24/uc24ltf_tune_bce_1.log
echo "[next] start $(date)"
python3 scripts/uc_uc24ltf_tune.py --jobs knn:0.05 --relax_tl 8 --check_tl 60 --budget_min 22 \
    > results/uc24/uc24ltf_tune_run2.log 2>&1 || exit 1
echo "[next] tuning done $(date)"
python3 scripts/uc_uc24ltf_eval.py --rules knn_1,knn_5,gnn_5,gnn_g_5 --old "95%|best=bce_g,95%|rl" --b2b 3 \
    > results/uc24/uc24ltf_eval_run.log 2>&1 || exit 1
echo "[next] eval done $(date)"
python3 scripts/uc_uc24ltf_report.py > results/uc24/uc24ltf_report_run.log 2>&1
echo "[next] all done $(date)"
