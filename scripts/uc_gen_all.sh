#!/bin/bash
# Unit-commitment datasets on RTS-GMLC (run detached: setsid nohup scripts/uc_gen_all.sh &)
cd "$(dirname "$0")/.."
g() { python3 scripts/uc_gen.py --workers 2 "$@" 2>&1 | grep --line-buffered -v Highs; }
g --cfg uc1  --split train --n 4000 --seed 1
g --cfg uc1  --split val   --n 500  --seed 2
g --cfg uc1  --split test  --n 1000 --seed 3 --n_alt 2
g --cfg uc12 --split train --n 500  --seed 11
g --cfg uc12 --split val   --n 60   --seed 12
g --cfg uc12 --split test  --n 120  --seed 13
