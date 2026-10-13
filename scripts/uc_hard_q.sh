#!/bin/bash
# hard benchmark run sequence, one single-threaded process per core (cores 2 and 3). Every step skips finished
# outputs, so a phase can be rerun after a restart.
#   scripts/uc_hard_q.sh data2 | data3     validation + test MILPs (core 2), training scenarios + test MILPs (core 3)
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=results/hard/logs; mkdir -p $L
case "$1" in
  data2)
    while pgrep -f "uc_hard_screen" > /dev/null && pgrep -f "line_scale 0.6 --n 8" > /dev/null; do sleep 10; done
    taskset -c 2 python scripts/uc_hard_data.py --gen val --chunks 0-11 >> $L/data_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_data.py --gen test --chunks 0-2 >> $L/data_c2.log 2>&1 ;;
  data3)
    while pgrep -f "uc_hard_screen.py --case rts --T 48" > /dev/null || pgrep -f "uc_hard_screen_q2.sh 3" > /dev/null; do sleep 10; done
    taskset -c 3 python scripts/uc_hard_data.py --gen train --chunks 0-29 >> $L/data_c3.log 2>&1
    taskset -c 3 python scripts/uc_hard_data.py --gen test --chunks 3-7 >> $L/data_c3.log 2>&1 ;;
esac
echo "$1 done $(date)" >> $L/q_$1.log
