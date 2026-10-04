#!/bin/bash
# Re-run Model 2 parts A+B with the slack-aware cost (after the main uc_model2 run has finished).
cd "$(dirname "$0")/.."
while pgrep -f "uc_model2.py --cfg uc1 --epochs 60 --workers 2$" > /dev/null; do sleep 30; done
python3 scripts/uc_model2.py --cfg uc1 --epochs 60 --workers 2 --parts AB 2>&1 | grep --line-buffered -v Highs > results/uc1/model2_AB.log
