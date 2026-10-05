# Constrained fine-tuning of Model 1 for the end-to-end mode (B2)

*Code: [`otsl/constrained.py`](../../otsl/constrained.py), [`scripts/uc_constrained.py`](../../scripts/uc_constrained.py)
(driver), [`scripts/uc_constrained_queue.sh`](../../scripts/uc_constrained_queue.sh) (the exact run sequence),
[`scripts/uc_constrained_report.py`](../../scripts/uc_constrained_report.py),
[`scripts/uc_constrained_diag.py`](../../scripts/uc_constrained_diag.py). Raw numbers:
[`results/uc12/constrained_results.md`](../../results/uc12/constrained_results.md) / `.json`, logs
`results/uc12/constrained_*.log`.*

__RESULTS__

## 3. Why the end-to-end policies over-commit (validation set, 60 instances)

**What the imitation model gets wrong.** The label-free BCE model, decoded at 0.5, serves 7 of 60
validation instances; 26 shed load and 27 more miss reserve, in 2.9 hours per violating instance. In
79 % of the violating hours the committed capacity is below net load + reserve (median −48 MW): a few
missing units in a few hours. When it serves, it is cheap (0.2 % above the MILP).

**What fine-tuning changes** (`scripts/uc_constrained_diag.py`, decisions at threshold 0.5, per instance
of 12 × 73 unit-hours; MILP: 12.5 units on per hour):

| model | units on | unit-hours added vs. init | removed | added, MILP has on | added, MILP has off | MILP-on still missed | uncertain decisions (0.05 < p < 0.95) | KL to init |
|---|---|---|---|---|---|---|---|---|
| LF-BCE (init) | 12.18 | – | – | – | – | ~9.1 | – | – |
| plain REINFORCE (`rl_lf`) | 14.10 | 23.0 | 0.0 | 3.8 | 19.1 | 5.3 | 7.9 % | 52 |
| MILP-label + REINFORCE (`milp_rl`) | 14.29 | 25.8 | 2.1 | 4.8 | 21.0 | 5.3 | 10.2 % | 60 |
| Lagrangian, no repair (`lag_B`) | 14.64 | 29.5 | 0.0 | 4.2 | 25.3 | 4.9 | 9.1 % | 79 |
| Lagrangian + KL 0.05 (`lag_D`) | 13.09 | 11.0 | 0.07 | 2.9 | 8.2 | 6.3 | 5.1 % | 18 |
| Lagrangian through block repair (`lag_R`) | 13.58 | 17.1 | 0.25 | 2.9 | 14.2 | 6.4 | 5.3 % | 30 |

* **Fine-tuning only adds units, and mostly the wrong ones**: of ~23 added unit-hours per instance, ~19
  are units the MILP keeps off, while ~5 MILP-on unit-hours stay missing. A one-shot policy with
  independent per-decision outputs cannot see the capacity of its own schedule, so it secures
  coverage by committing extra units broadly ("substitutes"), mostly decisions the imitation model had
  at p⁰ = 0.01–0.3.
* **Part of it is hedging against the policy's own sampling noise**: the training objective is the cost of
  *sampled* schedules; a needed unit at p = 0.9 is missing in 10 % of samples, so backups pay off in
  training but not at the deterministic threshold. Raising the decision threshold of `lag_B` from 0.5
  to 0.7 keeps 95 % served and cuts its served-instance gap from 15.3 % to 9.4 %; for `milp_rl`
  0.5 → 0.7 gives 81.7 → 75.0 % served, 11.3 → 7.1 %.
* **The Lagrangian alone (`lag_B`) does not fix it**: with the multiplier driven by the deployed
  violation rate (ε = 0.1) it overshoots in the first steps (λ → 11, 12.7 → 19 units), then settles at
  ~15 units — more coverage than plain REINFORCE (95 % vs 90 % on val) at a *higher* cost (15.3 % vs 9.9 %
  served gap). A KL anchor (`lag_D`, α = 0.05, ε = 0.15) halves the drift and the cost (4.4 %) but loses
  coverage (78 %): a different point on the same trade-off, not a Pareto gain.
* **A capacity repair does the job the policy does badly**: the min up/down-aware block repair alone
  takes the imitation model from 11.7 % to 81.7 % served at a 1.55 % median / 2.49 % served-instance
  gap (val), and a reserve margin traces a frontier: +3 % → 85.0 % / 4.5 %, +5 % → 88.3 % / 6.9 %,
  +8 % → 96.7 % / 8.8 %. Plain REINFORCE (90.0 % / 9.9 %) lies *behind* this no-learning frontier.
* **Fine-tuning through the repair**: with the repair in the loop the Lagrangian estimator is markedly
  leaner than plain REINFORCE at equal steps (val, deployed decoder, gap to the LP relaxation on served
  instances; the MILP is 0.8 % above the relaxation on val):

| step | Lagrangian ε = 0.1 (`lag_R`) | Lagrangian ε = 0.2 (`lag_R2`) | plain REINFORCE (`rl_R`) |
|---|---|---|---|
| 30 | 90.0 % / 4.6 % / 12.97 units | **91.7 % / 4.2 % / 12.85 units** | 93.3 % / 8.4 % / 14.08 units |
| 60 | 93.3 % / 5.5 % | (run interrupted) | 98.3 % / 10.8 % |
| 90 | 95.0 % / 6.3 % | | 98.3 % / 7.3 % |
| 120 | 98.3 % / 9.1 % / 13.74 units | | 96.7 % / 8.2 % / 13.56 units |

  Both drift toward more units with more steps (the multiplier keeps pushing coverage on the harder
  training instances; the train violation of `lag_R` reached ε = 0.1 only at λ ≈ 4–5, so the cap
  λ ≤ 5 was active most of the time and the method behaved like a fixed-weight two-stream penalty).
  Checkpoint selection on val is therefore part of the method.


## 1. Problem

End-to-end mode on B2 (12-hour network-constrained UC, RTS-GMLC): Model 1 predicts the 12 × 73 on/off
schedule, min up/down repair, **one** dispatch LP, no MILP at inference. The current best learned model
(label-free BCE on repaired LP-relaxation labels + REINFORCE with the dispatch LP as critic,
`RESEARCH.md` V4) serves 84.2 % of test instances without shedding or reserve shortfall, but costs
9.2 % more than the MILP on the instances it serves. The goal was to keep that coverage and cut the
served-instance gap below ~2 %.

## 2. Method

### 2.1 Lagrangian policy gradient with per-hour credit and an LP-sensitivity control variate

The dispatch LP of every sampled commitment is split per hour into the operating cost
`op_t` (no-load + energy + start-up of hour t) and the penalty `pen_t = VOLL·(shed_t + spill_t) + RES_SHORT·short_t`
(`otsl.constrained.hourly_dispatch`; `Σ_t op_t + pen_t` equals the LP objective, checked against
`UCModel.solve_dispatch` and `UCModel.dispatch_gradient` to machine precision). Model 1 is trained on

    min_θ  E_u~π_θ [ Σ_t op_t(u) / ref ]    s.t.   P( deployed commitment sheds or misses reserve ) ≤ ε

* **Dual ascent on the deployed policy.** At every step the deployed commitment (threshold 0.5 +
  repairs) of each batch instance is solved too (one of the six LPs per instance), and
  `log λ ← log λ + η (violation rate of the deployed commitments − ε)`. The constraint is therefore on
  what is shipped, not on the stochastic samples.
* **Two advantage streams, normalised separately** (as PPO-Lagrangian, Ray et al. 2019):
  `A = (Â_op + λ Â_pen)/(1 + λ)`, with the penalty stream `c_t = log(1 + pen_t / (κ·ref))`, κ = 10⁻³
  (graded, scale-free: a 1 MWh reserve shortfall counts, a 50 MW shed counts more but does not swamp
  the cost signal). Plain REINFORCE uses one stream, `−log(total cost / ref)`, in which the VOLL terms
  dominate the advantage normalisation.
* **Per-hour credit assignment**: the decision for unit g at hour t is credited with the cost of hours
  t−1 … t+1 only (window w = 1; ramping and start-ups couple neighbouring hours).
* **Baseline**: leave-one-out over the five samples of an instance plus the deployed commitment.
* **Control variate from the exact LP sensitivities (MuProp, Gu et al. 2016)**: the dual-based gradient
  `∂op/∂u_tg` at the deployed commitment (`dispatch_gradient`, at hours without penalties) gives a
  first-order Taylor model of the operating cost; it is subtracted from every sampled return and its
  expectation is differentiated analytically (`Σ_tg ∂op/∂u_tg · ∂p_tg/∂θ`). The estimator stays unbiased
  (up to the credit window); the analytic term is a dense per-unit signal that pushes down units whose
  no-load cost exceeds their dispatch value.
* **KL anchor to the imitation policy** (optional, weight α): `α Σ_tg KL(Bern(p_tg) ‖ Bern(p⁰_tg))`, as in
  KL-regularised policy optimisation; the imitation policy is cheap whenever it serves, so the fine-tune
  should move only decisions the LP critic asks for.
* **Label-free**: the reference cost is the LP-relaxation cost (no MILP solve anywhere in training).

### 2.2 Block adequacy repair (solver-free) and fine-tuning through it

`otsl.constrained.adequacy_repair_blocks`: for each hour whose committed capacity is below net load +
reserve, switch on the off unit with the lowest cost per MW of the **block it needs** (start-up +
no-load over its minimum up time; or bridging back to its last on-hour when that is within its minimum
down time, which avoids a start-up), until the deficit is covered; then min up/down repair. The
existing `adequacy_repair` adds single unit-hours, which the Hamming-nearest min up/down repair then
largely reverts for units with long minimum up times. The repair is used (a) as a decoder for every
model and (b) inside the training loop (every sampled and deployed commitment is repaired before the LP),
so the policy no longer has to secure aggregate capacity itself.

### 2.3 Baselines (same LP budget: 120 steps × 16 instances × 6 LPs = 11,520 training LPs)

* **Plain REINFORCE** (current method, `otsl.ucml.train_uc_reinforce`, re-run with the same seed from the
  re-trained label-free BCE model; `rl_lf`), and the same estimator with the block repair in the loop
  (`rl_R`, `otsl.constrained.train_reinforce_plain`).
* **He et al. 2026-style prediction and repair** (`he_bc`): behaviour cloning of MILP commitments (the
  MILP-label BCE model `uc_model1_4.pt`); reliable / repairable split by an uncertainty score
  `(1 − |2p − 1|) + h_tg + 1[root relaxation fractional]`, where `h_tg` is the frequency with which the
  root LP relaxation of the training instances is fractional for that unit-hour ("historical
  root-relaxation records"); the k = 48 most uncertain decisions are repairable; a shared per-decision
  policy (MLP on decision, unit and hour features incl. the BC schedule's capacity margin, initialised
  to reproduce the BC logit) sets them; PPO-clip (4 epochs, clip 0.2) on the reward −log(LP cost /
  MILP cost), where the LP cost contains the operating cost and the priced infeasibility. Simplification:
  He et al. learn a *sequential* repair policy; here the repair is a single step (all repairable
  decisions at once), so that inference needs one LP like the other end-to-end methods.
* **kNN-20**: the 20 nearest training schedules (MILP labels), each checked with the LP (20 LPs).

### 2.4 Protocol

Hyperparameters and variants were chosen on the 60 validation instances only; the test set (120
instances) was evaluated once at the end for all models. Decoders: top-1 (threshold 0.5) + min
up/down repair; + the existing adequacy repair; + the block adequacy repair; candidate screening
(`candidates_from_probs`: 7 thresholds + 8 samples, best by LP); a decision threshold calibrated per
model on val by the label-free expected cost (mean LP cost incl. penalties / LP-relaxation cost).
Single seed (0) throughout.

## 4. Prior work and what is new

Searched 2026-10-05 (search-engine abstracts and snippets; arXiv and publisher pages were not reachable,
so details marked † are from abstracts only).

| work | what it does | relation |
|---|---|---|
| He et al. 2026, *A prediction and repair framework with dispatch feedback for UC via RL* (IET GTD, [doi:10.1049/gtd2.70405](https://doi.org/10.1049/gtd2.70405)) | BC commitment prediction; root-relaxation history + confidence restrict the repair space; PPO repair policy trained on LP operating-cost and feasibility feedback; IEEE 300-bus, French 1888-bus† | closest; baseline `he_bc` (simplified to a one-step repair). Cost and feasibility enter one reward; no constraint / multiplier† |
| Yang et al. 2024, FPG-STGCN ([arXiv 2405.01200](https://arxiv.org/abs/2405.01200)) | few MILP solutions + augmented-Lagrangian physics loss on the constraints, straight-through estimator for the binaries | Lagrangian on the *UC constraints* of a relaxed model; here the multiplier is on the *outcome* of the exact dispatch LP (shedding / shortfall) and the gradient is a score-function estimator |
| Park et al. 2024, RACLearn (IEEE TPS, [2211.15755](https://arxiv.org/abs/2211.15755)); Fritz et al. 2026, Learning to Fix ([2609.39396](https://arxiv.org/abs/2609.39396)) | confidence / cost-aware fixing + reduced MILP | MILP at inference; not end-to-end |
| Pineda & Morales 2022 ([2106.11687](https://arxiv.org/abs/2106.11687)); Xavier, Qiu & Ahmed 2021 | kNN schedules + LP check | baseline kNN-20 |
| Dalal & Mannor 2015 ([1507.05268](https://arxiv.org/abs/1507.05268)); de Mars & O'Sullivan 2021/2022 ([2212.06001](https://arxiv.org/abs/2212.06001), [code](https://github.com/pwdemars/rl4uc)); Qin et al. 2022 ([2206.04249](https://arxiv.org/abs/2206.04249)) | UC as a sequential MDP (hour by hour), PPO / Q-learning, guided tree search; shedding priced in the reward | RL *solves* UC hour by hour; here RL fine-tunes a one-shot predictor with the LP as critic |
| Ray, Achiam & Amodei 2019, PPO-Lagrangian ([Safety Gym](https://github.com/openai/safety-starter-agents)); Tessler et al. 2019, RCPO ([1805.11074](https://arxiv.org/abs/1805.11074)); Stooke et al. 2020, PID Lagrangian ([2007.03964](https://arxiv.org/abs/2007.03964)) | constrained policy optimisation with a dual-ascent multiplier, separately normalised reward / cost advantages; multiplier overshoot and oscillation | the estimator used here; the overshoot they describe is exactly what happened in run `lag_B` |
| Solozabal et al. 2020, constrained combinatorial optimisation with RL ([2006.11984](https://arxiv.org/abs/2006.11984)) | Lagrangian-relaxed penalties for non-maskable constraints in neural combinatorial optimisation, REINFORCE | same idea for routing / scheduling; not UC, no LP critic |
| Fioretto, Mak & Van Hentenryck 2020 ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5403)); Park & Van Hentenryck 2023, PDL ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25520)) | Lagrangian-dual / primal-dual *self-supervised* learning of (continuous) OPF solutions | continuous decisions, differentiable constraints; here decisions are binary and the constraint is the outcome of an LP |
| Gu et al. 2016, MuProp ([1511.05176](https://arxiv.org/abs/1511.05176)); Kool, van Hoof & Welling 2019, RLOO | Taylor control variate for discrete stochastic nodes; leave-one-out baseline | estimator components; the Taylor gradient here comes from LP duals (envelope theorem) rather than backpropagation |
| Za'ter et al. 2026 ([2604.21891](https://arxiv.org/abs/2604.21891)); Ramesh & Li 2024 ([2208.06742](https://arxiv.org/abs/2208.06742)) | min up/down and excess-capacity heuristics / feasibility layers after prediction | the block adequacy repair is of this family; the new part is making it min up/down-aware and training *through* it |

**What is new here** (to the extent the search could establish):
1. A constrained (Lagrangian) policy-gradient fine-tune of a *one-shot* commitment predictor with the
   exact dispatch LP as critic, in which operating cost is the objective and LP shedding / reserve
   shortfall is a chance constraint on the **deployed** (thresholded and repaired) commitment, with the
   multiplier driven by the deployed violation rate. No published UC work found does this; He et al.
   train a separate repair policy on one combined reward, FPG-STGCN dualises the UC constraints of a
   relaxed model.
2. Per-hour credit assignment from a per-hour split of the dispatch LP objective, and a MuProp control
   variate whose Taylor gradient is the LP's exact sensitivities at the deployed commitment.
3. The diagnosis that the end-to-end policies' extra cost comes from units the MILP keeps off (and that
   fine-tuning never removes a unit), partly from hedging against the policy's own sampling noise, and
   that a min up/down-aware capacity repair in the training loop removes most of the pressure.
None of the components is new in isolation (Lagrangian RL, RLOO, MuProp, KL anchors, adequacy
heuristics are all standard); the combination and the application to the UC end-to-end mode are.

