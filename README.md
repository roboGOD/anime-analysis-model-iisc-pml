# Anime Clustering Workbench

This repository is organized for reusable clustering experiments on MyAnimeList anime metadata. Data preparation, feature building, model training, evaluation, and reporting are separated so different clustering backends can share the same upstream pipeline.

## Dataset

The default setup expects the primary raw dataset at:

```text
data/raw/anime-dataset-2023.csv
```

The scaffold is already configured for the observed MyAnimeList schema in `anime-dataset-2023.csv`.

The current raw dataset contains about `24,905` anime rows. The strongest high-level metadata axes in the raw file are format and production context fields such as `Type`, `Source`, and `Rating`, plus broad content tags in `Genres`. In the raw counts, `TV`, `Movie`, `OVA`, and `ONA` dominate the catalog, `Original` and `Manga` are the largest source groups, and common tags include `Comedy`, `Fantasy`, `Action`, `Adventure`, `Sci-Fi`, and `Drama`.

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

## Clustering Approach

The repository currently implements anime-level clustering with a Gaussian Mixture Model (GMM). Each anime is converted into a numeric feature vector and the model assumes those vectors were generated from a mixture of `K` latent Gaussian components. Unlike a hard clustering method, GMM gives both a hard cluster label and a soft probability distribution over clusters for every title.

In this repository, the clustering input is built from the cleaned metadata pipeline:

- numeric popularity and engagement signals such as `score`, `episodes`, `duration_minutes`, `members`, `favorites`, `scored_by`, `popularity`, and `rank`
- categorical production descriptors such as `type`, `source`, `rating`, `status`, and `premiered_season`
- multi-label content tags from `genres`

The data pipeline log-transforms heavily skewed count columns, imputes numeric and categorical missing values, one-hot encodes moderate-cardinality categorical fields, and expands retained genre vocabulary into multi-hot tag features. The final matrix saved at `data/processed/X_gmm.npy` is fully numeric and standardized for model fitting.

### What The Model Is Looking For

The goal is not just to split titles by one obvious field like `Type`. The project is trying to discover latent groups that combine:

- audience scale and mainstream reach
- length and format
- content profile by genre mix
- source-material pattern
- age rating and production positioning
- confidence or ambiguity of membership in a group

In practice, the model is trying to surface clusters that might look like:

- mainstream TV action or shounen-heavy titles with large member counts
- family-friendly films or specials
- niche highly rated psychological or drama-heavy works
- short-form or low-member experimental releases
- genre-dense fantasy or comedy catalogs with similar production profiles

These are hypotheses the model tests, not guaranteed truths. The evaluation and report stages are meant to check whether discovered clusters are actually distinct and interpretable.

### GMM Workflow

The implemented GMM flow is:

1. Build the processed feature matrix and row mapping from the cleaned anime metadata.
2. Search across candidate `K` values and covariance types (`diag`, `full`).
3. Use BIC as the primary selection criterion and AIC as secondary support.
4. Fit the final GMM on the saved matrix.
5. Export soft responsibilities, hard assignments, max assignment confidence, and assignment entropy.
6. Profile each cluster with size, dominant genres, representative anime, ambiguous anime, and distinguishing numeric signals.

This makes the clustering useful for interpretation, not only for optimization.

### What To Look For In Results

When reading outputs, the main questions are:

- Does the selected `K` improve BIC without producing tiny unstable clusters?
- Are max responsibilities reasonably high, or are many titles ambiguous?
- Do clusters differ in meaningful metadata terms such as format, source, rating, and genre composition?
- Are representative anime coherent within each cluster?
- Are there warning signs of overfitting, such as near-empty components or nearly duplicate clusters?

The most important outputs for answering those questions are:

- `artifacts/metrics/gmm_model_selection.csv`
- `artifacts/metrics/final_gmm_metrics.json`
- `artifacts/reports/cluster_profile_summary.md`
- `artifacts/reports/cluster_assignments.parquet`
- `artifacts/reports/gmm_diagnostics.json`

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
