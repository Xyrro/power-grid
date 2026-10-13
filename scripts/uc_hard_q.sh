#!/bin/bash
# hard benchmark run sequence, one single-threaded process per core (cores 2 and 3). Every step skips finished
# outputs, so a phase can be rerun after a restart.
#   scripts/uc_hard_q.sh data2 | data3     validation + test MILPs (core 2), training scenarios + test MILPs (core 3)
#   scripts/uc_hard_q.sh teach2 | teach3   teacher t0 (3 temporal GNNs on label-free labels), polishing pilot (core 2);
#                                          learning-free LtF tuning on the LP relaxation (core 3)
#   scripts/uc_hard_q.sh run2 | run3       the sequence actually run after 08:30 (data2 / data3 stopped after their first
#                                          step): core 2 teacher + pilot, core 3 all test MILPs + LtF-LP tuning
#   RATIO=0.9 TL=5 scripts/uc_hard_q.sh run2b | run3b   after the pilot (run3 was replaced by run3b at 09:55)
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
  teach2)
    while pgrep -f "uc_hard_q.sh data" > /dev/null; do sleep 10; done
    taskset -c 2 python scripts/uc_hard_train.py --name t0 --seeds 0,1 --label lf --n 2000 --epochs 30 >> $L/train_c2.log 2>&1
    while [ ! -f data/generated/hard/models/t0_s2.pt ]; do sleep 10; done
    taskset -c 2 python scripts/uc_hard_train.py --ensemble t0 >> $L/train_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_data.py --pilot --teacher t0 >> $L/pilot_c2.log 2>&1 ;;
  teach3)
    while pgrep -f "uc_hard_q.sh data" > /dev/null; do sleep 10; done
    taskset -c 3 python scripts/uc_hard_train.py --name t0 --seeds 2 --label lf --n 2000 --epochs 30 >> $L/train_c3.log 2>&1
    taskset -c 3 python scripts/uc_hard_tune.py --run lp --src lp --tuner faithful --budget_min 120 >> $L/tune_c3.log 2>&1 ;;
  run2)   # core 2 after its validation MILPs: teacher (3 seeds), polishing pilot
    while pgrep -f "uc_hard_data.py --gen val" > /dev/null; do sleep 10; done
    taskset -c 2 python scripts/uc_hard_data.py --gen val --chunks 0-11 >> $L/data_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_train.py --name t0 --seeds 0,1,2 --label lf --n 2000 --epochs 30 >> $L/train_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_train.py --ensemble t0 >> $L/train_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_data.py --pilot --teacher t0 >> $L/pilot_c2.log 2>&1 ;;
  run3)   # core 3 after its training scenarios: all test MILPs, then learning-free LtF tuning
    while pgrep -f "uc_hard_data.py --gen train" > /dev/null; do sleep 10; done
    taskset -c 3 python scripts/uc_hard_data.py --gen train --chunks 0-59 >> $L/data_c3.log 2>&1
    taskset -c 3 python scripts/uc_hard_data.py --gen test --chunks 0-11 >> $L/data_c3.log 2>&1
    while pgrep -f "uc_hard_data.py --gen val" > /dev/null; do sleep 10; done
    taskset -c 3 python scripts/uc_hard_tune.py --run lp --src lp --tuner faithful --budget_min 120 >> $L/tune_c3.log 2>&1 ;;
  run2b)  # core 2 after the pilot: polished labels (shared chunk list), final ensemble, guard-aware tuning, test shard 0
    taskset -c 2 python scripts/uc_hard_data.py --polish --teacher t0 --chunks 0-39 --ratio ${RATIO:-0.9} --tl ${TL:-5} >> $L/polish_c2.log 2>&1
    while ls data/generated/hard/polish_*.lock > /dev/null 2>&1; do sleep 10; done
    taskset -c 2 python scripts/uc_hard_train.py --name fin --seeds 0,1,2,3,4 --label pol --epochs 40 >> $L/train_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_train.py --ensemble fin >> $L/train_c2.log 2>&1
    taskset -c 2 python scripts/uc_hard_tune.py --run hg_fin --src fin_ens --tuner guarded --budget_min 120 >> $L/tune_c2.log 2>&1
    while [ ! -f results/hard/hard_tune_knn.json ]; do sleep 10; done
    taskset -c 2 python scripts/uc_hard_eval.py --shard 0 --nshards 2 >> $L/eval_c2.log 2>&1 ;;
  run3b)  # core 3 after its test MILPs: LtF-LP, join the polishing, guard-aware LtF on the LP values, LtF-kNN, test shard 1
    while pgrep -f "uc_hard_data.py --gen test" > /dev/null; do sleep 10; done
    taskset -c 3 python scripts/uc_hard_data.py --gen test --chunks 0-11 >> $L/data_c3.log 2>&1
    taskset -c 3 python scripts/uc_hard_tune.py --run lp --src lp --tuner faithful --budget_min 120 >> $L/tune_c3.log 2>&1
    while [ $(ls data/generated/hard/polish_[0-9][0-9].npz 2>/dev/null | wc -l) -lt 40 ]; do
      taskset -c 3 python scripts/uc_hard_data.py --polish --teacher t0 --chunks 0-39 --ratio ${RATIO:-0.9} --tl ${TL:-5} >> $L/polish_c3.log 2>&1
      sleep 10; done
    taskset -c 3 python scripts/uc_hard_tune.py --run lp_g --src lp --tuner guarded --budget_min 120 >> $L/tune_c3.log 2>&1
    taskset -c 3 python scripts/uc_hard_tune.py --run knn --src knn --tuner faithful --budget_min 120 >> $L/tune_c3.log 2>&1
    while [ ! -f results/hard/hard_tune_hg_fin.json ]; do sleep 10; done
    taskset -c 3 python scripts/uc_hard_eval.py --shard 1 --nshards 2 >> $L/eval_c3.log 2>&1 ;;
esac
echo "$1 done $(date)" >> $L/q_$1.log
