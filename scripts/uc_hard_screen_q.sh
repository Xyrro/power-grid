#!/bin/bash
# Screening queues: core 2 = RTS-GMLC derated lines, core 3 = PGLib FERC, then RTS reserve / 48 h stress.
# usage: scripts/uc_hard_screen_q.sh 2|3
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=results/hard/logs; mkdir -p $L
if [ "$1" = 2 ]; then
  for ls in 1.0 0.8 0.7 0.6; do
    taskset -c 2 python scripts/uc_hard_screen.py --case rts --line_scale $ls --n 4 --time_limit 300 >> $L/screen_c2.log 2>&1
  done
else
  taskset -c 3 python scripts/uc_hard_screen.py --case ferc --T 24 --time_limit 600 \
    --files 2015-01-01_hw 2015-02-01_lw 2015-07-01_hw 2015-12-01_lw >> $L/screen_c3.log 2>&1
  for rf in 0.06 0.10; do
    taskset -c 3 python scripts/uc_hard_screen.py --case rts --reserve $rf --n 3 --time_limit 300 >> $L/screen_c3.log 2>&1
  done
  taskset -c 3 python scripts/uc_hard_screen.py --case rts --T 48 --n 2 --time_limit 600 >> $L/screen_c3.log 2>&1
fi
echo "queue $1 done $(date)" >> $L/screen_c$1.log
