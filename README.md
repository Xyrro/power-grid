# Learning DC Optimal Transmission Switching (OTS)

Research code for the two-stage framework

```
demand (PD) ──► Model 1 (GNN) ──► switching status z ──► LP (DC-OPF | z) ──► Model 2 ──► (PG, VA)
                     ▲  MILP labels (DC-OTS)                                   ▲ LP labels
```

and for the extensions proposed in [`docs/RESEARCH.md`](docs/RESEARCH.md) (findings, literature
positioning, critique of the original framework, new methods and results).

## Layout

| path | content |
|---|---|
| `otsl/case.py` | PGLib-OPF / MATPOWER parser, DC network data |
| `otsl/opt.py` | DC-OPF LP and DC-OTS MILP (big-M, switching budget, disjoint-path big-M tightening, switching cost, no-good cuts) on HiGHS |
| `otsl/data.py` | load sampling, parallel dataset generation (MILP labels, all-closed OPF duals, alternative optima) |
| `otsl/features.py` | graph features; optional *dual features* from one all-closed DC-OPF |
| `otsl/models.py` | edge-aware GNN, switching heads, physics-consistent Model 2 (balance repair + differentiable DC power flow) |
| `otsl/train.py` | BCE imitation, REINFORCE with an LP critic, Model 2 training |
| `otsl/pipeline.py` | decoding, candidate screening with LP verification, dual-greedy and kNN baselines, metrics |
| `scripts/gen_data.py` | dataset generation (`--cfg case118 / case118_raw / case30 / case30_raw`) |
| `scripts/run_model1.py` | Model 1 study (baselines, GNN variants, cost-aware training, partial fixing, MSE-metric analysis) |
| `scripts/run_model2.py` | Model 2 study (direct vs physics decoder, screening, training Model 1 through Model 2) |
| `results/<cfg>/` | result tables (`*.md`) and raw numbers (`*.json`) |
| `tests/test_core.py` | solver / physics-layer sanity checks |

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_core.py
scripts/gen_all.sh                       # all datasets (~3 h on 4 cores; IEEE 118 MILPs take ~10 s each)
python scripts/run_model1.py --cfg case30 --raw case30_raw
python scripts/run_model2.py --cfg case30
python scripts/run_value.py  --cfg case30
python scripts/run_model1.py --cfg case118
python scripts/run_model2.py --cfg case118
python scripts/run_value.py  --cfg case118
```

Test cases are from [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) v23.07 (`data/cases`).
Generated datasets are not committed (`data/generated` is git-ignored).
