#!/bin/bash
# Runs the commands in results/uc12/combo_queue.txt one after another; the queue can be edited while it runs.
# Line format:  <log name>|<command>.  Output goes to results/uc12/combo_<log name>.log, progress to
# results/uc12/combo_chain.log.
cd "$(dirname "$0")/.."
Q=results/uc12/combo_queue.txt
while true; do
  line=$(head -n1 "$Q")
  [ -z "$line" ] && break
  sed -i '1d' "$Q"
  name=${line%%|*}
  cmd=${line#*|}
  echo "$(date +%T) START $name: $cmd" >> results/uc12/combo_chain.log
  bash -c "$cmd" 2>&1 | grep --line-buffered -v Highs > "results/uc12/combo_${name}.log"
  echo "$(date +%T) END $name (exit ${PIPESTATUS[0]})" >> results/uc12/combo_chain.log
done
echo "$(date +%T) queue empty" >> results/uc12/combo_chain.log
