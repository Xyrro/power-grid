#!/bin/bash
# case30 topology-shift data + experiment (run detached: setsid nohup)
cd "$(dirname "$0")/.."
until [ -f data/generated/case118_raw/test.npz ]; do sleep 30; done
g() { python3 scripts/gen_data.py --workers 2 "$@" 2>&1 | grep --line-buffered -v Highs; }
g --cfg case30_topo --split train --n 2400 --seed 21 >> results_gen.log
g --cfg case30_topo --split val --n 400 --seed 22 >> results_gen.log
g --cfg case30_topo_test --split test --n 700 --seed 23 >> results_gen.log
python3 scripts/run_topology.py 150 2>&1 | grep --line-buffered -v Highs > results/case30/topology.log
