# RACLearn vs. the two-stage framework

Paper: S. Park, W. Chen, D. Han, M. Tanneau, P. Van Hentenryck, *Confidence-Aware Graph Neural
Networks for Learning Reliability Assessment Commitments*, IEEE Trans. Power Systems 39 (2024),
arXiv 2211.15755v3 (read in full).

## What RACLearn does

| piece | details from the paper |
|---|---|
| problem | MISO Reliability Assessment Commitment (DA-FRAC, 24 periods; LAC, 12 periods), an SCUC-type MILP: binaries x = commitment/start-up/shut-down, continuous y = dispatch, reserves, flows. Power balance, reserves and **transmission limits are soft** (penalised). |
| data | French RTE network: 6,708 buses, 8,965 lines, 1,890 generators, 6,262 loads. ~9,000 DA-FRAC and ~8,600 LAC instances per month, Gurobi 9.5 at 0.1 % gap, 3,600 s limit. Train on month m, test on month m+1 (distribution shift). 80–85 % of commitments are must-run and are masked out of training. |
| GNN | MLP encoders for each generator and load → summed into bus features → GCN-1, GCN-5 or SIGN layers on the (unweighted, fixed) bus graph → **commitment decoder** (per generator, sigmoid per period) and **transmission decoder** (per line, f_i + f_j → active upper/lower limit per period). ~2 M parameters vs 134–859 M for MLP baselines. |
| loss | BCE on commitments (MILP solutions) + λ·BCE on active constraints with negative sampling, λ = 1. |
| confidence | MC-dropout (p = 0.5 on the decoder's penultimate layer), T samples, Conf = 1/σ. Beats softmax/binary entropy at picking an accurate subset (their Fig. 6). |
| Predict-Repair-Optimize | (1) add predicted active constraints (p > 0.1, ≤ 400) at the start of the iterative constraint-generation solve; (2) fix the X % most confident commitments; (3) **repair**: nearest commitment (Hamming) that satisfies the generator-level constraints (min up/down, ramping-related), decomposes per generator, polynomial by dynamic programming; because system constraints are soft, any such x gives a feasible y (Thm. 2); (4) solve the reduced problem. |
| results | accuracy 0.981 (DA-FRAC) / 0.994 (LAC) vs 0.960 / 0.982 for the MLP; DA-FRAC 2.66× faster at 30 % fixed (gap 0.01 %), 3.31× at 90 % (0.16 %), 4.12× at 100 % (0.77 %); LAC 1.67×–2.06× (≤ 0.09 %). Without repair, fixing 100 % leaves only 31 % (DA-FRAC) / 85 % (LAC) of instances feasible. Active-constraint prediction alone: 2.06× / 1.45×. |
| limitations they state | fixed network topology; future work: "address network topology changes, extending its applicability to accelerate optimal transmission planning or generalize transmission contingencies". |

## Side-by-side with the framework diagram

| framework box | RACLearn counterpart | difference |
|---|---|---|
| Power demand PD, QD | load forecasts + full generator offers/ramps/initial status over T periods | the framework only feeds demand; RACLearn needs the generator data and the time axis that makes UC hard |
| Model 1 (GNN) → switching status | commitment decoder of a GNN over the bus graph | same idea. The framework may mean *line* switching (studied here) or *generator* on/off (RACLearn) |
| MILP → Model 1 loss | BCE vs Gurobi solutions | same. Neither handles tied optima (see F1/F8 in RESEARCH.md) |
| LP solver given the switching | fix only confident binaries, repair, solve the reduced MILP (an LP only at 100 % fixed) | RACLearn never trusts all predictions blindly; the repair guarantees feasibility |
| Model 2 (constraints team) → PG, VA | none for dispatch; instead the transmission decoder predicts which line limits will bind | RACLearn spends the second model on *constraint screening*, which speeds up the exact solve, instead of approximating its output |
| dashed arrow into Model 1 loss | none | the framework's only conceptual addition; in our tests it hurts through a learned Model 2 and helps a lot through the exact LP |
| validation: violations, gap | feasibility %, optimality gap, speedup | same spirit |
| testing: MSE to MILP solution | accuracy/AUROC on commitments, gap, time | RACLearn does not use MSE (our F7/F11 explain why it should not) |

## What this means for the framework

1. **If "switching status" is generator on/off**, RACLearn is the paper a reviewer will hold the
   framework against, and the framework as drawn is a subset of it (no confidence, no repair, no
   constraint prediction, single period). The framework would need a distinct contribution.
2. **If it is line switching (OTS)**, RACLearn explicitly lists topology change and transmission
   planning as open — the setting studied in this repository. Its tools carry over directly:
   * confidence-aware fixing ↔ our partial fixing (F9, 30× faster at optimal quality on IEEE 118);
   * repair ↔ un-fixing the "open" decisions (the all-closed topology is always feasible in OTS, so
     fixing lines *closed* never breaks feasibility — a free repair step);
   * active-constraint prediction can be added to the switching GNN in the same way.
3. **Contributions that neither RACLearn nor DA-DNN covers** (supported by results in RESEARCH.md):
   cost-aware fine-tuning with the exact LP as critic (one LP reaches 95–100 % of the gap on IEEE 118),
   tie/equivalence-aware labels, physics-consistent dispatch decoding, and generalisation to unseen
   topologies.

## RACLearn-style confidence in DC-OTS (experiment)

`scripts/run_confidence.py` adds MC-dropout to the switching head (p = 0.5, T = 30), ranks line
decisions by 1/σ (RACLearn) or by |p − 0.5| (entropy-style baseline), fixes the top X %, repairs by
dropping fixed-open decisions if the MILP becomes infeasible, and solves the reduced MILP. Results:
`results/case118/confidence_results.md` (summarised in RESEARCH.md once finished).
