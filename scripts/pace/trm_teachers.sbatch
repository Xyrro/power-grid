#!/usr/bin/env bash
# Step 2 (CPU): the label teacher (3 MILP-label BCE GNNs) and the m1x baseline (1,500 polished labels on seed-111
# scenarios + a 5-member temporal-GNN ensemble), as in docs/methods/m1x.md. ~4 core-hours (polishing dominates).
#SBATCH -J uc12-teachers
#SBATCH -A <ACCOUNT>                 # PLACEHOLDER
#SBATCH -p <CPU_PARTITION>           # PLACEHOLDER
#SBATCH -q <QOS>                     # PLACEHOLDER
#SBATCH -N 1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"   # submit from the repository root (mkdir -p logs first)
python3 scripts/uc_hybrid_train.py --seeds 0,1,2
[ -f results/uc12/uc_model1_4.pt ] || cp results/uc12/hybrid_bce_s0.pt results/uc12/uc_model1_4.pt
# m1x extra scenarios (seed 111) and polished labels, 6 chunks of 250 in parallel
for k in 0 1 2 3 4 5; do
  (python3 scripts/uc_m1x_data.py --gen --chunks $k && \
   python3 scripts/uc_m1x_data.py --polish --rank conf --ratio 0.9 --tl 5 --lp --chunks $k) &
done
wait
for s in 0 1 2 3 4; do
  python3 scripts/uc_m1x_train.py --label pol --n 2000 --kind gnnt --seed $s &
done
wait
for s in 0 1 2 3 4; do
  cp data/generated/uc12_m1x/models/pol_n2000_gnnt_s$s.pt results/uc12/m1x_pol_n2000_gnnt_s$s.pt
done
python3 scripts/uc_m1x_train.py --ensemble pol_n2000_gnnt_ens5 \
  pol_n2000_gnnt_s0,pol_n2000_gnnt_s1,pol_n2000_gnnt_s2,pol_n2000_gnnt_s3,pol_n2000_gnnt_s4
