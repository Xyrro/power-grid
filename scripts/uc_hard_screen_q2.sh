#!/bin/bash
# Screening continuation (after scripts/uc_hard_screen_q.sh): the remaining winter-fleet high-wind FERC files split
# over cores 2 and 3, then RTS-GMLC reserve 10 % and the 48 h horizon on core 3.
# usage: scripts/uc_hard_screen_q2.sh 2|3
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=results/hard/logs
if [ "$1" = 2 ]; then
  while pgrep -f "uc_hard_screen_q.sh 2" > /dev/null; do sleep 5; done
  taskset -c 2 python scripts/uc_hard_screen.py --case ferc --T 24 --time_limit 600 \
    --files 2015-12-01_hw 2015-11-02_hw 2015-10-01_hw >> $L/screen_c2.log 2>&1
else
  while pgrep -f "uc_hard_screen.py --case rts --reserve 0.06" > /dev/null; do sleep 5; done
  taskset -c 3 python scripts/uc_hard_screen.py --case rts --reserve 0.06 --n 3 --time_limit 300 >> $L/screen_c3.log 2>&1
  taskset -c 3 python scripts/uc_hard_screen.py --case ferc --T 24 --time_limit 600 \
    --files 2015-02-01_hw 2015-03-01_hw >> $L/screen_c3.log 2>&1
  taskset -c 3 python scripts/uc_hard_screen.py --case rts --reserve 0.10 --n 3 --time_limit 300 >> $L/screen_c3.log 2>&1
  taskset -c 3 python scripts/uc_hard_screen.py --case rts --T 48 --n 2 --time_limit 600 >> $L/screen_c3.log 2>&1
fi
echo "queue2 $1 done $(date)" >> $L/screen_c$1.log
