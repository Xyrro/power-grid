#!/bin/bash
# hard benchmark run sequence, one single-threaded process per core (cores 2 and 3). Every step skips finished
# outputs, so a phase can be rerun after a restart.
#   scripts/uc_hard_q.sh data2 | data3     validation + test MILPs (core 2), training scenarios + test MILPs (core 3)
cd "$(dirname "$0")/.."
export OTSL_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=results/hard/logs; mkdir -p $L
case "$1" in
  data2)
    taskset -c 2 python scripts/uc_hard_data.py --gen val --chunks 0-11 >> $L/data_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_data.py --gen test --chunks 6-11 >> $L/data_c2.log 2>&1 ;;
  data3)
    taskset -c 3 python scripts/uc_hard_data.py --gen train --chunks 0-59 >> $L/data_c3.log 2>&1
    taskset -c 3 python scripts/uc_hard_data.py --gen test --chunks 0-5 >> $L/data_c3.log 2>&1 ;;
esac
echo "$1 done $(date)" >> $L/q_$1.log
