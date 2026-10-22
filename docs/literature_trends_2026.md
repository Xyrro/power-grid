# Hot topics in learning for OPF and grid operations (October 2026)

*Compiled 2026-10-22 from arXiv abstract pages, project pages and reports (web search, then each page fetched). No paper
was read in full; the summaries restate abstracts. Items marked † could not be fetched and rest on search snippets.
"Hot" is a judgement from 2026 arXiv activity, lab releases and the IEEE PES General Meeting 2026 program (Montréal,
19–23 July 2026, theme "Powering the Digital Era"), not a citation count. Unit-commitment prior work is in
[`literature_uc.md`](literature_uc.md).*

## Overview

| topic | why it is hot | key items below |
|---|---|---|
| 1. Grid foundation models | one model across many topologies and sizes; first public releases in 2026 | GridSFM, LUMINA, HydraGNN OPF-GFM, GENCO, Memon et al. |
| 2. LLM agents for grid studies | LLMs that call exact solvers; benchmarks appeared in June 2026; largest PES GM 2026 panel | GridMind, PFAgent, Grid-Mind, governance-aware digital twin, PowerAgentBench, tutorial |
| 3. Generative models for OPF | dispatch as a distribution, warm starts, synthetic data | DiffOPF, fast diffusion with physics correction |
| 4. Feasibility guarantees | from "predict, then repair" to guarantees inside the method | safe gradient flow, hard-constrained DC→AC, L-IPM, inertia-aware PINN |
| 5. GPU solvers | exact solvers get 10–100× faster, raising the bar for learned speed-ups | MadNLP / ExaModels, ExaModelsPower.jl |
| 6. Benchmarks and surveys | standard data was the main complaint of 2024–2025 | PGLearn, OPFData, graph-ML survey |
| 7. Application drivers | AI data-center load, inverter-based resources, wildfire risk | LBNL, Oxford Energy, PNNL, Chen–Zhu–Sioshansi |

## 1. Grid foundation models

**GridSFM: A Foundation Model for Solving AC Optimal Power Flow.** Bhan, Yang, Capetz, Zhang (Microsoft Research,
University of Washington). arXiv [2609.30173](https://arxiv.org/abs/2609.30173), v1 24 Sep 2026, v2 6 Oct 2026.
* Paper: a 15M-parameter physics-inspired GNN pretrained on 54 topologies (500–4,000 buses), then physics-informed
  fine-tuning based on Newton's method for power flow. The disconnected AC-OPF feasible set is handled by lifting the
  problem and relaxing constraints with logarithmically penalised slacks.
* Results: 2.45 % zero-shot generation-cost error on held-out operating conditions of a 10,000-bus case, with no
  degradation as size grows; adapts to unseen grids up to 10,000 buses from 100 solved instances; better warm starts
  (cost and solver iterations) than single-topology models.
* [Project blog](https://www.microsoft.com/en-us/research/blog/gridsfm-a-new-small-foundation-model-for-the-electric-grid/)
  (13 May 2026): 150+ base topologies, about half a million scenarios; GridSFM-Open (up to 4,000 buses, open on GitHub
  and Hugging Face) and GridSFM-Premier (up to 80,000 buses, on request). Median cost gap 2.23 %, mean 3.41 %, 83 % of
  scenarios within 5 %; 1,000× faster than a full AC solve; as a warm start 1.66× faster than a cold start; feasibility
  screening AUC 0.986 on Texas2k. [White paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/05/GridFM_white_paper.pdf).
  The sources give different topology counts (54 in the paper, 150+ on the blog, 200 in some search snippets).

**LUMINA: Foundation Models for Topology Transferable ACOPF.** Y. Li, Memon, Jin, Fenu, Song, Sharma, Gasana, H. Kim,
L. Zhao, K. Kim (Argonne). arXiv [2603.04300](https://arxiv.org/abs/2603.04300), Mar 2026; ICLR 2026 FM4Science
workshop.
* Controlled experiments on architectures, training objectives and system diversity yield three design principles for
  constrained scientific foundation models: physics-invariant representations versus system-specific constraints,
  accuracy versus constraint satisfaction, and reliability in high-impact operating regimes.
* Releases data and training pipelines; [LUMINA-2M](https://huggingface.co/argonne/LUMINA-2M) is a 2.4M-parameter
  heterogeneous graph transformer.

**LUMINA: A Grid Foundation Model for Benchmarking AC Optimal Power Flow Surrogate Learning.** Jin, Song, Memon, Li,
Fenu, H. Kim, Zhao, K. Kim. arXiv [2605.02133](https://arxiv.org/abs/2605.02133), May 2026.
* LUMINA-Bench: homogeneous versus heterogeneous GNNs, single- versus multi-topology training, and MSE versus
  augmented-Lagrangian versus violation-based Lagrangian losses; multi-topology pretraining, transfer and adaptation;
  metrics for accuracy and constraint violation. Open source.

**Scalable Heterogeneous Graph Foundation Models for Data-Driven Optimal Power Flow in Smart Grids.** Lupo Pasini,
Y. Li, K. Kim, Kuruganti. arXiv [2605.23194](https://arxiv.org/abs/2605.23194), v1 May 2026, v2 Sep 2026.
* A HydraGNN workflow that keeps buses, generators, loads, shunts, AC lines and transformers as distinct node and edge
  types; distributed preprocessing, multi-GPU training, hyperparameter search and fine-tuning.
* About 3 million graphs from ten PGLib-OPF cases (14 to 13,659 buses); scaling runs on up to 1,024 Frontier nodes,
  best at 256 nodes (1,024 AMD MI250X GPUs); HeteroSAGE and HeteroHEAT (about 1.6–1.7M parameters) reach the lowest
  validation MSE; downstream tasks on IEEE 118.
* Dataset: [HydraGNN_OPF_GFM_2026](https://www.osti.gov/dataexplorer/biblio/dataset/3239722) (OSTI).

**GENCO: A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis.** Puech,
Mazzonelli, Govindasamy, et al. (22 authors, incl. Dörfler, Hug, Hamann, Brunschwiler). arXiv
[2608.09921](https://arxiv.org/abs/2608.09921), Aug 2026.
* One architecture for power flow, OPF and state estimation on a shared grid representation, with an open "GridFM
  Development Framework" for low-code data generation.
* Millions of PF and OPF scenarios across topologies; evaluated on PFΔ, OPFData and Hydro-Québec SCADA data. Up to 30×
  faster than Newton–Raphson for power flow and 85× faster than IPOPT for OPF, with better feasibility metrics.
* Related community event: [GridFM seminar at NYU](https://engineering.nyu.edu/events/2026/08/06/foundation-model-electric-grid-gridfm), 6 Aug 2026.

**Towards Systematic Generalization for Power Grid Optimization Problems.** Memon, Y. Li, Jin, K. Kim, L. Zhao. arXiv
[2605.02026](https://arxiv.org/abs/2605.02026), May 2026.
* A shared graph backbone with task-specific decoders for **both ACOPF and SCUC**, trained with solver supervision plus
  physics-informed objectives for AC feasibility and inter-temporal constraints; tested on unseen topologies without
  retraining, and on UC-ACOPF with unsupervised objectives and a dispatch consensus mechanism.
* **The only foundation-model-style work found that covers unit commitment.**

**PowerModels-ACOPF-AI.** Dhamala, Tabarez, A. Pandey. arXiv [2609.19360](https://arxiv.org/abs/2609.19360), Sep 2026.
* Two-stage Bayesian neural network predicting generator setpoints, voltages and angles with uncertainty; retrains
  "on the fly" where accuracy degrades; 30-, 200- and 500-bus systems; meant as an advisory tool and solver initialiser.

## 2. LLMs and agents for grid studies

The shared design: **the LLM plans and calls exact engineering tools; it never produces the numbers itself.**

**GridMind: LLMs-Powered Agents for Power System Analysis and Operations.** Jin, K. Kim, Kwon (Argonne). arXiv
[2509.02494](https://arxiv.org/abs/2509.02494), Sep 2025; SC'25 workshop.
* Multi-agent system; agents run AC-OPF and N-1 contingency analysis through function calls. Correct on IEEE cases
  with every LLM tested; smaller LLMs are as accurate with lower latency.

**PFAgent: A Tractable and Self-Evolving Power-Flow Agent for Interactive Grid Analysis.** She, B. Chen, Guo, F. Li.
arXiv [2604.10846](https://arxiv.org/abs/2604.10846), Apr 2026.
* Intent parsing, retrieval, tool execution and reporting; self-evolution from verification-driven refinement and human
  feedback; evaluated on task success, convergence validity, numerical consistency and explanation quality (IEEE
  systems: case edits, voltage violations, N-1).

**Grid-Mind: An LLM-Orchestrated Multi-Fidelity Agent for Automated Connection Impact Assessment.** Shamseldein. arXiv
[2602.20683](https://arxiv.org/abs/2602.20683), Feb 2026.
* Eleven tools (power flow, N-1, transient stability, EMT screening), a violation inspector and a three-layer
  hallucination defence. 50 IEEE 118-bus scenarios with DeepSeek-V3: 84.0 % tool selection, 100 % parsing;
  self-correction passes 49 of 56 cases (87.5 %).

**A Governance-Aware LLM Orchestrated Agentic Digital Twin for TSO Control Room Decision Support.** Mylonas, Foti,
Varvarigos. arXiv [2609.22476](https://arxiv.org/abs/2609.22476), Sep 2026.
* A non-bypassable governance layer: whitelisted tools only, step budgets, approval before side-effect actions, every
  answer from backend results with units and timestamps. Digital twin of the Greek transmission network, 118 tasks.
* 590 runs: 96.5 % tool selection, 93.7 % task success, rules never broken; 1,416 runs over four models, rules held
  every time (95 % lower bound 99.8 %). Without the layer the model executed all 45 approval-requiring runs unasked.
  Overhead 12–16 ms per request.

**PowerAgentBench-SS** (Mylonas, Foti, Pomarico, Duarte, Q. Zhang, Varvarigos; arXiv
[2606.18789](https://arxiv.org/abs/2606.18789), Jun 2026) and **PowerAgentBench-Dyn** (Q. Zhang, Pomarico, Mylonas,
Foti, Berizzi, Le Xie; arXiv [2606.20401](https://arxiv.org/abs/2606.20401), Jun 2026).
* SS: agents get case data, constraints and a tool API; a hidden evaluator checks physics; risk-sensitive metrics
  (false-safe penalties, severity regret, residual violations, tool efficiency). Pilot: IEEE 39-bus DC N-2 contingency
  search with scripted, Ollama and OpenAI agents. Agents differ in validation-budget use and evidence-backed reporting.
* Dyn: dynamic model quality review, and dynamic security screening of short-circuit contingencies with mitigation.

**Power Systems Agent Benchmark: Executable Evaluation of AI Agents in Electric Power Engineering.** Trashchenkov.
arXiv [2606.20950](https://arxiv.org/abs/2606.20950), Jun 2026.
* 41 task families in eight areas (power flow to forecasting); deterministic evaluators recompute quantities and list
  violations; held-out cases generated from private seeds against contamination. Three command-line agents tested;
  the best nears the ceiling; the evaluation also exposed an evaluator bug. Code on GitHub and Zenodo.

**LLMs and Agentic AI Systems for Smart Grids: A Tutorial on Architectures and Applications.** Rojas, Albassam, Leung,
et al., Yize Chen, Yuanyuan Shi. arXiv [2607.18147](https://arxiv.org/abs/2607.18147), Jul 2026.
* "Solver-grounded" design: report a number only if it comes from a trusted tool and passes verification. Four case
  studies: an EV-charging agent matches the CVXPY optimum with 7.5–9.5× less unmet energy than an LLM alone; a
  contingency-debugging agent repairs 17 of 39 cases and cuts violations by 52.3 %.

## 3. Generative models for OPF

**DiffOPF: Diffusion Solver for Optimal Power Flow.** Hoseinpour, Dvorkin. arXiv
[2510.14075](https://arxiv.org/abs/2510.14075), Oct 2025, v2 Mar 2026.
* OPF is a multi-valued map from loads to dispatch when parameters vary; DiffOPF learns the joint distribution of loads
  and setpoints from operating history and samples dispatch given loads. Gives warm starts with good cost and
  constraint trade-offs; analyses sample complexity.

**Fast Diffusion with Physics-Correction for ACOPF.** Shekhar, Karn, Keshav, Bansal, Pareek. arXiv
[2602.03020](https://arxiv.org/abs/2602.03020), Feb 2026.
* DDIM with physics-guided corrections during sampling, for generating large AC-OPF datasets; IEEE 6, 24 and 118 bus;
  up to 20× faster sampling than standard diffusion at comparable accuracy and physical consistency.

## 4. Feasibility guarantees and solvers inside the learning

**Learning to Pursue AC Optimal Power Flow Solutions with Feasibility Guarantees.** Ajeyemi, Y. Chen, Colot, Cortés,
Dall'Anese. arXiv [2505.22399](https://arxiv.org/abs/2505.22399), v2 Feb 2026.
* A safe gradient flow for distribution feeders with DERs; a neural network approximates the QP solution map inside
  it. DER setpoints stay feasible even if stopped early; 93-node feeder.

**A hard-constrained NN learning framework for rapidly restoring AC-OPF from DC-OPF.** K. Chen, Knueven, W. Jones.
arXiv [2602.06255](https://arxiv.org/abs/2602.06255), Feb 2026.
* Unsupervised: a network learns to correct the DC-OPF solution into AC-OPF setpoints through a differentiable
  optimisation layer (implicit function theorem on the KKT conditions), with replay buffers. IEEE 118 and PEGASE-9241:
  40× faster, violations around 1e-4, optimality gap below 1 %; adapts to topology changes.

**Learning Interior Point Method Central Path Projection for OPF (L-IPM).** Amani, Kargarian, Vaidyanathan. arXiv
[2508.19146](https://arxiv.org/abs/2508.19146), v2 Mar 2026.
* An LSTM extrapolates the interior-point central path from its early, well-conditioned iterations, with a
  grid-informed step that enforces generation, voltage and flow limits. 2,869-bus European system: up to 94 % less
  time and 85.5 % fewer iterations.

**Inertia-Aware Optimal Power Flow Using PINN in IBR-Dominated Power Systems.** Tofighi-Milani,
Fattaheian-Dehkordi, Rohrhofer, Lehtonen. arXiv [2607.09297](https://arxiv.org/abs/2607.09297); EEEIC 2026.
* A physics-informed network with a location-aware inertia constraint (inertia strength by electrical distance); 6 GW
  test system, mean absolute error about 0.045 % of capacity.

## 5. GPU solvers

**Accelerating Optimal Power Flow with GPUs.** Shin, Pacaud, Anitescu. arXiv
[2307.16830](https://arxiv.org/abs/2307.16830); EPSR 2024 (PSCC 2024).
* SIMD abstraction for parallel automatic differentiation plus a condensed-space interior-point method whose KKT
  system factorises without pivoting (MadNLP.jl, ExaModels.jl): an order of magnitude faster than CPU tools.

**Scalable Multi-Period AC Optimal Power Flow Utilizing GPUs with High Memory Capacities.** Shin, Rao, Schanen,
Maldonado, Anitescu. arXiv [2405.14032](https://arxiv.org/abs/2405.14032), May 2024.
* NVIDIA GH200 (480 GB): a multi-period OPF with more than 10 million variables solved to 1e-4 in under 10 minutes.

**ExaModelsPower.jl: A GPU-Compatible Modeling Library for Nonlinear Power System Optimization.** Johnson, Lauinger,
Shin, Pacaud. arXiv [2510.12897](https://arxiv.org/abs/2510.12897), v2 Mar 2026.
* GPU-ready AC-OPF models including multi-period and security constraints; up to two orders of magnitude faster than
  CPU tools above 20,000 variables at 1e-4 tolerance; small or tight-tolerance instances may not gain.

**Large-scale Grid Optimization: The Workhorse of Future Grid Computations.** A. Pandey, Almassalkhi, Chevalier. arXiv
[2501.04623](https://arxiv.org/abs/2501.04623) (review).
* Physics-based methods lead on large-scale grid optimisation; physics-constrained data-driven methods are emerging for
  otherwise intractable problems.

## 6. Benchmarks, datasets and surveys

**PGLearn: An Open-Source Learning Toolkit for Optimal Power Flow.** Klamkin, Tanneau, Van Hentenryck. arXiv
[2505.22825](https://arxiv.org/abs/2505.22825), May 2025.
* Standard OPF datasets and evaluation tools: AC, DC and second-order-cone formulations, global and local variability,
  time series for large systems for the first time; over 10 million samples in the first release; on Hugging Face.

**OPFData: Large-scale datasets for AC optimal power flow with topological perturbations.** Lovett, Zgubic, Liguori,
et al. arXiv [2406.07234](https://arxiv.org/abs/2406.07234), Jun 2024.
* The largest open collection of solved AC-OPF instances at release, including topology perturbations.

**Graph Machine Learning: An Opportunity for Power Systems.** Sadric, Pütz, Nauck, Hagenmeyer, Hellmann, Witthaut,
Schäfer. arXiv [2608.16494](https://arxiv.org/abs/2608.16494), Aug 2026.
* Survey of about 800 papers (forecasting, state estimation, optimisation, control, fault diagnosis, cybersecurity).
  Main gaps: scarce standard benchmarks and open data, little real-world deployment, need for interpretability. Names
  unit-commitment uses: predicting schedules, screening decisions, and confidence estimates that guide optimisation.

## 7. Application drivers

**AI data centers as grid load.**
* LBNL, *Integrating AI Data Centers with the Power Grid* (Granderson, Hoffman, Holecek, Crowe, Smith, Mims Frick;
  The Bridge, May 2026; [link](https://datacenters.lbl.gov/publications/integrating-ai-data-centers-power)): US
  data-center energy use projected at two to three times 2023 levels by 2028; four flexibility levers (shifting
  computation, facility systems, storage, onsite generation); utilities adding interruptible riders, mandated
  flexibility and fast interconnection for flexible loads.
* Oxford Energy review of a Khargonekar seminar (19 May 2026;
  [link](https://www.energy.ox.ac.uk/review/ai-data-centre-growth-challenges-and-opportunities-for-the-us-power-system/)):
  US consumption flat at about 4,430 TWh for two decades; a PJM capacity auction 6.6 GW short; about 410 GW of large
  loads seeking interconnection in ERCOT by 2030; close to 100 GW of flexible load servable with little curtailment;
  flexibility raises local absorption by about 90 % without network constraints and 61 % with them.
* † 7–12 % of US electricity by 2028; a December 2025 trial near London where a GPU cluster cut power by 40 % on call.

**Inverter-based resources and grid-forming inverters.**
* Quedan, Goharrizi, Cheng, Huang, Du, Ramasubramanian, *Grid-Forming Inverters for Stability Improvements in Bulk
  Power Systems with High IBR Penetration* (IEEE PES GM 2024;
  [PNNL page](https://www.pnnl.gov/publications/grid-forming-inverters-stability-improvements-bulk-power-systems-high-inverter-based)):
  in a West Texas ERCOT study with 35+ GW of IBRs, replacing grid-following batteries with grid-forming ones improves
  voltage and active-power injection during and after faults.
* † NREL, *Research Roadmap on Grid-Forming Inverters* (NREL/TP-5D00-73476; site unreachable from here).

**Wildfire resilience.**
* Chen, Zhu, Sioshansi, *Large-Scale Resilience Planning for Wildfire-Prone Electricity-System via Adaptive Robust
  Optimization*, arXiv [2604.01232](https://arxiv.org/abs/2604.01232), v4 May 2026: tri-level model of sectionalisation
  and public-safety power shutoffs under uncertain ignition risk, solved by column-and-constraint generation; a large
  investor-owned utility distribution system.

## Unit commitment in this batch

* **Za'ter, Van Boven, Hodge, Baker**, *A Multi-Stage Warm-Start Deep Learning Framework for Unit Commitment*, arXiv
  [2604.21891](https://arxiv.org/abs/2604.21891), Apr 2026: transformer predicting 72-hour commitments, deterministic
  post-processing, then warm start and confidence fixing for the MILP. Single-bus system only; 100 % feasible; on about
  20 % of test instances the schedule is cheaper than the solver alone. (Also in `literature_uc.md`.)
* **Memon et al. 2026** (section 1) is the only cross-topology model that includes SCUC.

## What this means for this project

* **A foundation model for unit commitment is open.** Every 2026 grid foundation model targets power flow, OPF or state
  estimation; only Memon et al. touch SCUC. A cross-topology Model 1 with the exact dispatch LP, the repairs and the
  guards would be new.
* **Accuracy bar.** GridSFM's 2–3.4 % cost gap is acceptable for OPF screening; for UC this project treats anything
  above about 0.5 % to a proven bound as a loss to the MILP. Foundation models trade accuracy for breadth.
* **Generative sampling fits our screening.** DiffOPF-style conditional sampling of schedules, each checked by the
  dispatch LP, is the same "propose many, price exactly, keep the best" pattern as arm B and candidate screening.
* **Solver-grounded agents** state the principle this project landed on: numbers come from exact solvers.
* **Faster baselines.** GPU interior-point and first-order solvers (MadNLP, ExaModels, cuPDLP) speed up the LP
  relaxations and dispatch LPs we use, and the baselines we compare against; our speed-ups are measured against
  single-thread HiGHS.
* **A timely benchmark extension:** UC with large flexible loads (data centers) that can be curtailed or shifted.
