#!/bin/bash
# Runs the commands in results/uc12/constrained_queue.txt one after another (the queue can be edited
# while it runs). Each command's output goes to results/uc12/constrained_<name>.log.
cd "$(dirname "$0")/.."
Q=results/uc12/constrained_queue.txt
until [ -f results/uc12/constrained_val_lf_bce.json ]; do sleep 20; done
while true; do
  line=$(head -n1 "$Q")
  [ -z "$line" ] && break
  sed -i '1d' "$Q"
  name=$(echo "$line" | sed -E 's/.*--name ([^ ]+).*/\1/')
  echo "$(date +%T) START $line" >> results/uc12/constrained_chain.log
  python3 scripts/uc_constrained.py $line 2>&1 | grep --line-buffered -v Highs > "results/uc12/constrained_${name}.log"
  echo "$(date +%T) END $name" >> results/uc12/constrained_chain.log
done
