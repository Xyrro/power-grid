#!/bin/bash
# uc24ltf: after the main test evaluation, a second pass with our min up/down conflict release added at test time
# to the two GNN Learning-to-Fix rules (their tuning stopped at the budget; their reduced problems can be infeasible),
# then the merged report.   bash scripts/uc_uc24ltf_next2.sh <pid of the main queue>
cd "$(dirname "$0")/.." || exit 1
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "[next2] start $(date)"
python3 scripts/uc_uc24ltf_eval.py --rules gnn_5+cr,gnn_g_5+cr --old "" --b2b 0 --tag _cr \
    > results/uc24/uc24ltf_eval_run_cr.log 2>&1 || exit 1
echo "[next2] eval done $(date)"
python3 scripts/uc_uc24ltf_report.py --extra uc24ltf_eval_test_cr.jsonl > results/uc24/uc24ltf_report_run.log 2>&1
echo "[next2] all done $(date)"
