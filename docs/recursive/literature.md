# Prior work: recursive and iterative learned models for power-grid problems

*Searched October 2026 with web search (search snippets and abstracts; most papers were not read in full). The
recursive-reasoning papers themselves are listed in [`PLAN.md`](PLAN.md).*

**No published work was found that applies TRM/HRM-style recursive reasoning (a tiny weight-shared network,
deep supervision, latent recursion) to unit commitment or optimal power flow.** Searches for TRM, HRM, looped or
recursive transformers together with power systems returned only general looped-transformer work. The closest lines:

## 1. Recurrent message passing for power flow (continuous, not combinatorial)
* **Graph Neural Solver** (Donon et al., RTE, PSCC 2020): a GNN trained to reduce Kirchhoff-law violations at every
  bus rather than to imitate Newton-Raphson; IEEE case9–case118. [HAL](https://hal.archives-ouvertes.fr/hal-02372741),
  [PSCC](https://pscc-central.epfl.ch/repo/papers/2020/715.pdf)
* **Deep Statistical Solvers** (Donon et al., NeurIPS 2020): generalizes this to optimization problems on graphs,
  trained on the problem's own objective, no solved labels. [mlanthology](https://mlanthology.org/neurips/2020/donon2020neurips-deep)
* **Recurrent randomized GNN for AC power flow** (arXiv 2204.07000): encoder, recurrent message-passing stack and
  decoder; noisy starts; every iteration yields a candidate and the lowest-loss one is kept, which is the same idea as
  keeping the best LP-priced step in our arm B. [arXiv](https://arxiv.org/pdf/2204.07000)
* Deep Statistical Solver for distribution state estimation. [arXiv](https://arxiv.org/abs/2301.01835v3)

## 2. Iterative correction and unrolled optimization for OPF and constrained problems
* **DC3** (Donti, Rolnick, Kolter, ICLR 2021): completion of equality constraints plus an unrolled gradient-descent
  correction for inequalities; AC-OPF. [arXiv](https://arxiv.org/pdf/2104.12225)
* **Unrolled primal-dual networks** for constrained optimization (arXiv 2601.17274; graph version 2509.17156):
  coupled primal and dual networks with monotone descent/ascent constraints; not applied to grids.
  [arXiv](https://arxiv.org/abs/2601.17274), [arXiv](https://arxiv.org/html/2509.17156)
* **Self-supervised primal-dual learning** (Park & Van Hentenryck, AAAI 2023) for AC-OPF.
  [arXiv](https://arxiv.org/pdf/2208.09046)
* Unsupervised AC-OPF with a physics-aware layer that unrolls fast-decoupled power-flow iterations (arXiv 2604.23548).
  [arXiv](https://arxiv.org/pdf/2604.23548)

## 3. Learning for unit commitment (one-shot predictions or learned search)
* One-shot: spatial-temporal GCN with a straight-through estimator for end-to-end UC (arXiv 2405.01200); GNN variable
  fixation refined by root LP solutions (MDPI Appl. Sci. 2025); RACLearn and Learning to Fix (in `docs/literature_uc.md`).
  [arXiv](https://arxiv.org/html/2405.01200), [MDPI](https://www.mdpi.com/2076-3417/15/8/4498)
* Learned search: GNN initial commitments plus GNN-selected large-neighbourhood search (arXiv 2505.14408); RL and tree
  search, stepping through the hours (de Mars & O'Sullivan, arXiv 2212.06001); RL inside surrogate Lagrangian
  relaxation with unit-level subproblems (IEEE TPWRS 2025; 118-bus and 10k-bus, ≤ 3 % degradation, 25–110× faster than
  Gurobi). [Pith](https://pith.science/paper/2505.14408), [arXiv](https://arxiv.org/pdf/2212.06001),
  [XJTU](https://scholar.xjtu.edu.cn/en/publications/a-reinforcement-learning-embedded-surrogate-lagrangian-relaxation/)
* Neural networks predicting Lagrange multipliers for UC date back to the 1990s (a functional-link network on a
  16-unit system, 1996); no modern deep version was found.

## 4. Recent recursive-reasoning work worth borrowing
* **Energy-guided Recursive Model** (arXiv 2607.10128, 2026): scores many recursive trajectories with an energy
  function and keeps the best (Sudoku 98.97 %, Maze 99.30 %). For unit commitment the exact dispatch LP *is* an
  energy: it prices any candidate schedule exactly. [arXiv](https://arxiv.org/abs/2607.10128)
* Looped-transformer follow-ups on adaptive depth and fixed points (LoopFormer 2602.11451, Fixed-Point Reasoners
  2606.18206).

## Where our two arms sit
Arm A (TRM on UC) and arm B (recursion with the exact LP in the loop) combine line 1's recurrent message passing,
line 2's iterative correction and line 4's candidate selection on a mixed-integer, multi-period problem. To our
knowledge, that combination has not been published.
