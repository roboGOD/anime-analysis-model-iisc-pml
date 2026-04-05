# Anime Clustering Workbench

This repository is organized for reusable clustering experiments on MyAnimeList anime metadata. Data preparation, feature building, model training, evaluation, and reporting are separated so different clustering backends can share the same upstream pipeline.

## Dataset

The default setup expects the primary raw dataset at:

```text
data/raw/anime-dataset-2023.csv
```

The scaffold is already configured for the observed MyAnimeList schema in `anime-dataset-2023.csv`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run the default full pipeline:

```bash
python scripts/run_all.py
```

Run a different model or pipeline config:

```bash
python scripts/run_all.py --model gmm --pipeline full
```

Common flags:

- `--config-dir configs`
- `--model gmm`
- `--pipeline full`
- `--run-id 20260405_120000`
- `--overwrite`
- `--skip-existing`
- `--from-stage build_matrix --to-stage evaluate`

Useful runner scripts:

- `python scripts/run_data_pipeline.py`
- `python scripts/run_model_selection.py`
- `python scripts/run_train_model.py`
- `python scripts/run_train_gmm.py`
- `python scripts/run_evaluation.py`
- `python scripts/run_report.py`

## Organization

The scaffold is split into three layers:

- `src/data/` contains model-agnostic ingestion, cleaning, transformation, and feature construction.
- `src/clustering/` contains one adapter per clustering backend plus a registry that the orchestrator uses.
- `src/pipelines/` contains composable pipeline groups for data prep, modeling, and reporting.

To add a new clustering model:

1. Add a config file in `configs/models/`.
2. Implement an adapter in `src/clustering/models/`.
3. Register it in `src/clustering/registry.py`.

No data pipeline code or runner script needs to be duplicated for a new model.

## Outputs

Pipeline outputs are written to:

- `data/interim/` for normalized and transformed datasets
- `data/processed/` for feature-ready tables
- `artifacts/checkpoints/` for intermediate feature matrices and metadata
- `artifacts/models/` for trained clustering models
- `artifacts/metrics/` for JSON and CSV metrics
- `artifacts/plots/` for EDA, selection, and report plots
- `artifacts/reports/run_configs/` for per-run config snapshots
- `logs/` for run logs

## Included Backend

The scaffold currently includes a GMM adapter. GMM models the dataset as a weighted mixture of Gaussian components and produces both hard assignments and soft membership probabilities, which is useful when titles sit near cluster boundaries.
