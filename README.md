# Music Genre Classification: Reproducible ML Workflow

A multi-step ML pipeline that classifies songs into genres from audio features and track metadata. Each step is an isolated **MLflow** project with its own conda environment. **Hydra** handles configuration, and **Weights & Biases** tracks artefacts and experiments, so every run is versioned and reproducible.

> Built as the final exercise of the Udacity *ML Workflows* course (ML DevOps Engineer Nanodegree), using Udacity's starter kit.

## Pipeline

```
download ──▶ preprocess ──▶ check_data ──▶ segregate ──▶ random_forest ──▶ evaluate
```

| Step | What it does |
|---|---|
| `download` | Fetches the raw parquet dataset and logs it as a W&B artefact |
| `preprocess` | Cleans the data, removes duplicates, builds a combined text feature from title and song name |
| `check_data` | pytest data tests: schema and types, allowed class names, value ranges, and a **Kolmogorov–Smirnov drift test** against a reference dataset |
| `segregate` | Stratified train/test split |
| `random_forest` | scikit-learn `Pipeline`: imputation, scaling, ordinal encoding, **TF-IDF on text**, then a random forest. Logs ROC AUC, feature importance and a confusion matrix, and exports an MLflow model |
| `evaluate` | Scores the exported model on the held-out test set |

Every input and output is a versioned W&B artefact (e.g. `data_train.csv:latest`), so each model can be traced back to the exact data it was trained on.

## Run it

Requires conda, MLflow and a (free) W&B account (`wandb login`).

```bash
conda env create -f conda.yml && conda activate download_data

mlflow run .                                          # full pipeline
mlflow run . -P hydra_options="main.execute_steps='random_forest,evaluate'"   # selected steps
mlflow run . -P hydra_options="random_forest_pipeline.random_forest.max_depth=10"  # override a hyperparameter
```

All settings (data URL, split sizes, drift-test threshold, model hyperparameters, feature lists) live in [`config.yaml`](config.yaml).

**Stack:** Python · MLflow · Hydra · Weights & Biases · scikit-learn · pandas · pytest · conda
