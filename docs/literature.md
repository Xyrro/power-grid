# Prior work and positioning

Compiled from a literature search on 2026-10-04 (search-engine abstracts/snippets; full texts were
not retrievable from this environment, so numbers marked † should be checked against the papers).

## Closest prior work: the two-stage idea is already published

| work | Model 1 (switching) | dispatch stage | training signal | notes |
|---|---|---|---|---|
| **Kim & Kim 2025, DA-DNN** ([arXiv 2507.17194](https://arxiv.org/abs/2507.17194)); Kim, Sun & Kim 2025 ([2512.17516](https://arxiv.org/abs/2512.17516)) | DNN, sigmoid-relaxed line status | **exact differentiable DC-OPF layer** | generation cost (no MILP labels) | essentially *Model 1 + the dashed arrow* with an exact LP instead of a learned Model 2; IEEE 118: $93.02k vs $93.16k DC-OPF, within 0.01% of MILP† |
| **Meng, Haider & Van Hentenryck 2025, OptiGridML** ([2508.01951](https://arxiv.org/abs/2508.01951)) | heterogeneous GNN (breakers) | line-graph GNN flow surrogate (= Model 2) | Kirchhoff-consistency loss | objective = export maximisation, not cost; up to 1,000 breakers |
| Martinez, Donon, Wehenkel & Karangelos 2026 ([2603.23401](https://arxiv.org/abs/2603.23401)) | GNN (substation reconfiguration) | LP solver in the loop | self-supervised "by interaction with an LP solver"† | +10.2% vs +15.2% (MILP) exchange capacity† |
| Johnson, Ahmed, Dey & Watson 2020 ([2003.10565](https://arxiv.org/abs/2003.10565)) | kNN over past instances | DC-OPF for each neighbour topology | none (lookup) | feasibility by evaluation; 118–3,375 buses |
| Pineda, Morales & Jiménez-Cordero 2024, TOP ([2304.07269](https://arxiv.org/abs/2304.07269)) | kNN fixes binaries + tightens big-M | reduced MILP | none | large speed-ups on 118-bus† |
| Yang & Oren 2019, PowerTech | kNN/ANN/trees rank lines | greedy heuristic | supervised | also algorithm selection |
| Bugaje, Cremer & Strbac 2023, IET GTD ([10.1049/gtd2.12698](https://doi.org/10.1049/gtd2.12698)) | NN | – | supervised | real-time OTS on IEEE 118 |
| Han & Hill 2023, IEEE TPS ([9763339](https://ieeexplore.ieee.org/document/9763339/)) | gated GNN as evaluation function in topology search | search | imitation + RL† | OTS + distribution reconfiguration |
| Pham & Li 2024 ([2410.17460](https://arxiv.org/abs/2410.17460)) | GNN predicts topology, confidence filter | optimisation stage | MILP labels | closest to the framework's Model 1 training |
| Abiala, Sang & Gerdes 2026 ([2607.10948](https://arxiv.org/abs/2607.10948)) | RL policy warm-started by behaviour cloning | – | MILP + RL | 88–97% of MILP savings, quality drops at large budgets† |
| Huang, Piansky, Dilkina & Molzahn 2025 ([2510.25147](https://arxiv.org/abs/2510.25147)) | GAT on bipartite graph | guides MILP | supervised | wildfire-risk OTS |
| Di Vito et al. 2026 ([2608.13079](https://arxiv.org/abs/2608.13079)) | graph diffusion + feasibility projection | – | generative | AC-OTS |
| Rajaei, Palensky & Cremer 2025 ([2510.20591](https://arxiv.org/abs/2510.20591)) | edge-aware heterogeneous GNN (busbar splitting) | – | supervised | transfer across systems |

Unit-commitment analogues: Xavier, Qiu & Ahmed 2021 (INFORMS JoC, learn warm starts / fixings /
constraint screening); Fritz et al. 2026 "Learning to Fix" ([2609.39396](https://arxiv.org/abs/2609.39396),
optimisation-aware confidence thresholds for fixing commitments).

**Park, Chen, Han, Tanneau & Van Hentenryck, "Confidence-Aware Graph Neural Networks for Learning
Reliability Assessment Commitments", IEEE Trans. Power Systems 39 (2024), [arXiv 2211.15755](https://arxiv.org/abs/2211.15755)
(RACLearn)** — structurally the closest match to the framework's Model 1 (read in full; see
[`raclearn_comparison.md`](raclearn_comparison.md)). GNN over the bus graph predicts generator
commitments and active line constraints from MILP solutions (BCE); MC-dropout confidence (1/σ) picks
the commitments to fix; a polynomial-time Hamming-distance repair restores feasibility (system
constraints are soft); the reduced MILP is solved with the predicted constraints seeded. RTE France
network (6,708 buses, 1,890 generators): 2.7–4.1× faster for DA-FRAC (gap 0.01–0.77 %), 1.7–2.1× for
LAC. Fixed topology; topology changes / transmission planning named as future work.

## Learning the dispatch (Model 2)

DeepOPF (Pan et al.; predict PG, recover angles by a power-flow solve), Lagrangian-dual training
(Fioretto et al. 2020), DC3 (Donti et al. 2021; equality completion + inequality correction),
Primal-Dual Learning (Park & Van Hentenryck 2023), **E2ELR** (Chen, Tanneau & Van Hentenryck 2024;
closed-form power-balance repair layers, always feasible economic dispatch), homeomorphic projection
(Liang et al. 2023) and gauge mapping (Li et al. 2023) — both need one map per constraint set, i.e.
per topology —, and optimal active-set learning (Misra, Roald & Ng 2022). Datasets PGLearn (2025)
and OPFData (2024) contain **no OTS labels**.

## Learning for MILP

Neural Diving (Nair et al. 2020), confidence-threshold diving (Yoon 2022), Predict-and-Search
(Han et al. ICLR 2023), Voice of Optimization (Bertsimas & Stellato 2021: predict the integer part
→ LP, structurally identical to Model 1 → LP), decision-focused learning (SPO+, perturbed optimizers,
blackbox differentiation; survey Mandi et al. 2024).

## Issues raised in the literature that apply to the framework

* **Non-unique optima** in DC-OTS (e.g. Crozier 2025, [2510.20089](https://arxiv.org/abs/2510.20089)):
  imitation losses and MSE-to-MILP metrics penalise equally optimal answers.
* **Benefit depends on congestion**: on PGLib "typical" cases many instances have ~0 benefit, so
  "all lines closed" looks deceptively good; report savings recovered, not only gap.
* **Islanding / feasibility** of predicted topologies (Han, Song & Hill 2020, [2006.12752](https://arxiv.org/abs/2006.12752)).
* **Switching budget** (Fisher, O'Neill & Ferris 2008) — learned quality degrades with budget size.
* **Big-M strength** dominates MILP runtime (Fattahi et al. 2019 bound strengthening; Pineda et al.
  2024, [2306.02784](https://arxiv.org/abs/2306.02784)): speed-ups should be measured against a tuned MILP.
* **Feasibility is cheap to secure**: LP-scoring several candidate topologies plus an all-closed
  fallback (as kNN does) gives feasible answers never worse than DC-OPF.
