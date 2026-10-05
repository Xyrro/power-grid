# Prior work for the unit-commitment framework

Compiled 2026-10-04 from search-engine abstracts/snippets (full texts were not retrievable here,
except RACLearn, which was read in full: [`raclearn_comparison.md`](raclearn_comparison.md)).
Items marked † were not verified against the paper.

## Commitment prediction followed by LP / MILP (Model 1 → LP)

| work | what is predicted / how it is used | feasibility | systems, results |
|---|---|---|---|
| **Park, Chen, Han, Tanneau & Van Hentenryck 2024, RACLearn** (IEEE TPS, [2211.15755](https://arxiv.org/abs/2211.15755)) | GNN: commitments + active line constraints; MC-dropout confidence; fix confident, reduced MILP | polynomial Hamming repair (system constraints soft) | RTE 6,708 buses: 2–4× faster, ≤ 0.77 % gap |
| **Tang, Bai, Weng & Wang 2023** (Energy Reports 9, [link](https://www.sciencedirect.com/science/article/pii/S2352484723001853)) | GCN predicts unit decisions → SCUC becomes a continuous convex problem | – | 13–17× speed-up† — *the framework without Model 2* |
| Ramesh & Li 2022/2024, feasibility layer (IEEE TPS, [2208.06742](https://arxiv.org/abs/2208.06742)) | LR/NN/RF/kNN classify commitments for model reduction | explicit min on/off feasibility layer + post-processing | IEEE 24/73/118, SC-500, Polish 2383 |
| Ramesh & Li 2023, spatio-temporal reduced SCUC ([2306.01570](https://arxiv.org/abs/2306.01570)) | GNN + LSTM: commitments + critical lines | | IEEE 24/73/118, SC-500 |
| Yang, Li & Jian 2025 ([2505.14408](https://arxiv.org/abs/2505.14408)) | GNN initial commitment + GNN-guided large-neighbourhood search | heuristic restoration | trained on 80 units, beats solvers on 1,080 |
| Yang, Li, Chen & Zheng 2025 (Appl. Sci. 15:4498) | bipartite GNN predicts "stable" variables; dual threshold (GNN confidence + root LP) for hard/soft fixing | | †|
| Za'ter, Van Boven, Hodge & Baker 2026 ([2604.21891](https://arxiv.org/abs/2604.21891)) | transformer, 72-h schedule → warm start + confidence fixing | min up/down and excess-capacity heuristics | single-bus only |
| Wang, Wu, Weng & Zhang 2026 ([2604.02788](https://arxiv.org/abs/2604.02788)) | fix "structurally stable" binaries (incl. LLM selection) | restriction proof, solver-certified | IEEE 57, RTS-73, 118: order-of-magnitude speed-ups |
| Qin & Yu 2023 ([2311.15216](https://arxiv.org/abs/2311.15216)) | physics-informed GCN neural diving + neural branching | inside MIP solver | † |
| **Fritz, Makrides, Fetanat & Pinson 2026, Learning to Fix** ([2609.39396](https://arxiv.org/abs/2609.39396)) | generator-specific, cost-aware fixing thresholds (calibrated after training) | | EPRI 2025 competition winner: < 0.5 % gap, > 20× |
| **He et al. 2026, prediction and repair with dispatch feedback** (IET GTD 20(1), [doi:10.1049/gtd2.70405](https://doi.org/10.1049/gtd2.70405)) | behaviour-cloning commitment prediction; root-relaxation history + confidence split variables into fixed / repairable; a **PPO repair policy trained with LP dispatch cost and feasibility feedback** | RL repair | † (abstract only) — closest published relative of the LP-critic fine-tuning here |
| Shekeew & Venkatesh 2023 (IEEE OAJPE 10) | "trusted" generators fixed | | IEEE 14/118/300, Polish: 48–98 % time reduction |
| Wei, Ai, Fang et al. 2025 (Applied Energy 377) | GNN trained on **many near-optimal solutions per instance** | repair | RTS-GMLC, Jiangsu: 10.7× / 15.9× |
| Pourahmadi & Kazempour 2025 (IEEE TPS 40, [2310.08601](https://arxiv.org/abs/2310.08601)) | distributionally robust SVM with performance guarantee | warm start | IEEE 6/118: 1.7× |
| **Pineda & Morales 2022, "Is learning for UC a low-hanging fruit?"** (EPSR 207, [2106.11687](https://arxiv.org/abs/2106.11687)) | nearest-neighbour candidate schedules | | near-optimal with large speed-ups: the baseline to beat |
| Xavier, Qiu & Ahmed 2021 (INFORMS JoC) | kNN warm starts, transmission-constraint screening, fixing | | large SCUC |

End-to-end / self-supervised / RL: FPG-STGCN (Yang, Qiu, Liu & Liu 2024, [2405.01200](https://arxiv.org/abs/2405.01200):
straight-through estimator + augmented-Lagrangian physics loss), physics-informed GNNs for UC/ED (Xu et al.
2025, SSRN†), learning-to-optimise MINLP with integer correction (Tang, Khalil & Drgoňa 2024,
[2410.11061](https://arxiv.org/abs/2410.11061)), RL for UC (Qin et al. 2023; de Mars & O'Sullivan 2021),
neural second-stage surrogates in stochastic UC (Shao, Qin & Yu 2025; Fusco, Lodi & Marla 2026),
cross-task / cross-topology graph backbones (Memon et al. 2026, [2605.02026](https://arxiv.org/abs/2605.02026)).

## Dispatch given commitments (Model 2)

* Chen, Park, Tanneau & Van Hentenryck 2022, SCED proxies (EPSR, [2112.13469](https://arxiv.org/abs/2112.13469)):
  commitment is an input; classify units at bounds, then regress; < 1 % error on the French system.
* Chen, Tanneau & Van Hentenryck 2024, **E2ELR** (IEEE TPS, [2304.11726](https://arxiv.org/abs/2304.11726)):
  closed-form repair layers → always power-balanced, reserve-feasible dispatch; self-supervised.
* DeepOPF (Pan, Zhao, Chen et al.): predict PG, reconstruct VA from DC power flow.
* Zhang, Karve & Mahadevan 2025 (Applied Energy): GNN surrogates under evolving commitment.
* **A learned dispatch model used as a differentiable critic for the commitment model (the dashed arrow)
  was not found.** Nearest: FPG-STGCN (physics loss, no critic), Learning to Fix (cost-aware calibration
  after training), decision-focused UC for forecasts (Chen, Yang, Liu & Wu 2022, IEEE TPS 37), DiffAPQP
  ([2608.04189](https://arxiv.org/abs/2608.04189)). Expected reviewer question: why a learned critic instead
  of exact LP sensitivities (duals of p ≤ pmax·u, p ≥ pmin·u, no-load cost — envelope theorem)?

## Known issues

* **Identical units / symmetry**: Knueven, Ostrowski & Watson 2018 (IEEE TPS 33(4)) — identical generators
  create symmetric optima; aggregation is exact under conditions. For learning: Chen et al., "When GNNs meet
  symmetry in ILPs" (ICLR 2025, [2501.14211](https://arxiv.org/abs/2501.14211)) — permutation-equivariant GNNs
  cannot distinguish symmetric variables; orbit-based feature augmentation fixes it.
* **LP relaxation tightness**: Morales-España, Latorre & Ramos 2013 (IEEE TPS 28(4)); Knueven, Ostrowski &
  Watson 2020 (INFORMS JoC). Single-period UC has a tight relaxation (each unit's {pmin u ≤ p ≤ pmax u} is its
  own convex hull), so relax-and-round and priority lists are strong; multi-period is where learning can pay.
* **Min up/down feasibility** of predicted schedules is handled by post-hoc repair in almost every work.
