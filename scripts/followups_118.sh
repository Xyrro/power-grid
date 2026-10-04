#!/bin/bash
# Follow-up IEEE 118 runs, started once their prerequisites exist (run detached: setsid nohup)
cd "$(dirname "$0")/.."
f() { grep --line-buffered -v Highs; }
(until [ -f results/case118/model1_results.json ]; do sleep 30; done
 python3 scripts/run_label_free.py --cfg case118 --steps 600 --workers 2 2>&1 | f > results/case118/label_free.log) &
(until [ -f results/case118/value_results.json ]; do sleep 30; done
 python3 scripts/run_model2.py --cfg case118 --epochs 60 --n_train 800 --workers 2 2>&1 | f > results/case118/model2.log) &
wait
