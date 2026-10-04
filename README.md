# Learning unit commitment with a GNN → LP → NN framework

Research code for the two-stage framework

```
demand (PD, QD) + units running before ──► Model 1 (GNN) ──► on/off u ──► LP (dispatch | u) ──► Model 2 ──► (PG, VA)
                                               ▲ MILP labels (UC)                               ▲ LP labels
```

on the RTS-GMLC system (73 buses, 73 thermal units, 2020 load / renewable profiles), and for the
extensions proposed in [`docs/RESEARCH.md`](docs/RESEARCH.md) (findings, literature positioning,
critique of the framework, new methods and results). Prior work: [`docs/literature_uc.md`](docs/literature_uc.md),
[`docs/raclearn_comparison.md`](docs/raclearn_comparison.md).

An earlier side study read "switching status" as transmission-line switching (DC-OTS); its code and
report are kept ([`docs/ots/RESEARCH_OTS.md`](docs/ots/RESEARCH_OTS.md)).

## Layout: unit commitment

| path | content |
|---|---|
| `otsl/uc.py` | RTS-GMLC loader; UC MILP (3-binary, min up/down, ramping, start-up and piecewise-linear costs, spinning reserve, DC network, soft shedding / reserve shortfall) and fixed-commitment dispatch LP with exact cost sensitivities; min up/down and adequacy repairs |
| `otsl/ucdata.py` | scenario sampling, parallel dataset generation (MILP labels, LP relaxation, alternative optima) |
| `otsl/ucml.py` | features, commitment GNN / MLP, BCE / REINFORCE / exact-LP-sensitivity training, LP oracle, metrics, Model 2 (direct regression or physics decoder) |
| `scripts/uc_gen.py`, `scripts/uc_gen_all.sh` | datasets: `uc1` (single hour, B1) and `uc12` (12-hour look-ahead, B2) |
| `scripts/uc_model1.py` | Model 1 study: ambiguity, baselines (persistence, merit order, relax-and-round, kNN), GNN variants, repair, screening, REINFORCE, confidence fixing |
| `scripts/uc_model2.py` | Model 2 study: direct vs physics decoder, Model 2 as screener, the dashed arrow vs exact LP sensitivities, MSE of tied optima |
| `scripts/uc_fixing.py` | which decisions to fix before the MILP: symmetric (RACLearn), asymmetric, adequacy-guarded |
| `scripts/uc_followups.sh`, `scripts/uc_model2_rerun.sh`, `scripts/uc_fixing_b2.sh` | the exact experiment runs behind the report |
| `results/uc1/`, `results/uc12/` | result tables (`*.md`), raw numbers (`*.json`), logs |
| `data/rts_gmlc/` | RTS-GMLC tables and day-ahead time series ([GridMod/RTS-GMLC](https://github.com/GridMod/RTS-GMLC)) |

## Reproduce (unit commitment)

```bash
pip install -r requirements.txt
python tests/test_core.py && python tests/test_uc.py
setsid nohup scripts/uc_gen_all.sh > results_uc_gen.log 2>&1 &   # B1 ~1 h, B2 ~4 h on 4 cores
scripts/uc_followups.sh                  # Model 1 (B1, B2) and Model 2 (B1) studies
python scripts/uc_model2.py --cfg uc1 --parts AB
python scripts/uc_fixing.py --cfg uc1 --n_fix 200
python scripts/uc_fixing.py --cfg uc12 --n_fix 60 --ratios 0.8,0.9,0.95
```

HiGHS (through SciPy) solves all MILPs and LPs. Worker pools use the `spawn` start method (HiGHS
stalls in forked children) and PyTorch runs single-threaded. Generated datasets are not committed
(`data/generated` is git-ignored).

## Layout: OTS side study

`otsl/case.py`, `opt.py`, `data.py`, `features.py`, `models.py`, `train.py`, `pipeline.py`, `value.py`;
`scripts/gen_data.py`, `run_model1.py`, `run_model2.py`, `run_value.py`, `run_label_free.py`,
`run_topology.py`, `run_seeds.py`, `ambiguity.py`, `tables.py`, `gen_all.sh`, `run_all_118.sh`;
results in `results/case30/`, `results/case118/`, tables in `docs/ots/`. Test cases are from
[PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) v23.07 (`data/cases`).
