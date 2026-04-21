# Antibody Expression Titer Prediction

Predicting the expression yield (titer) of therapeutic antibodies from sequence alone, using protein language model (PLM) embeddings combined with CDR-region biophysical features. Five PLMs (ESM-C, ProtT5, ProstT5, Ankh, ProtBERT) are evaluated and combined into a Spearman-weighted ensemble. Models are validated with cluster-aware cross-validation on 246 antibodies (GDPa1) and evaluated on a held-out set of 80 antibodies (GDPa3).

---

## Table of Contents

- [Antibody Expression Titer Prediction](#antibody-expression-titer-prediction)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Environment Setup](#environment-setup)
  - [Option 1: Using `uv` (recommended)](#option-1-using-uv-recommended)
    - [1. Ensure Python 3.11 is installed](#1-ensure-python-311-is-installed)
    - [2. Create virtual environment](#2-create-virtual-environment)
    - [3. Activate your virtual environment](#3-activate-your-virtual-environment)
      - [3a. Confirm Python requirement in pyproject.toml file](#3a-confirm-python-requirement-in-pyprojecttoml-file)
    - [4. Install dependencies using lockfile](#4-install-dependencies-using-lockfile)
  - [Option 2: Using `conda`](#option-2-using-conda)
    - [1. Create the environment](#1-create-the-environment)
    - [2. Activate it](#2-activate-it)
    - [3. Install dependencies](#3-install-dependencies)
  - [Option 3: Using `pip`](#option-3-using-pip)
    - [1. Make sure Python 3.11 is installed](#1-make-sure-python-311-is-installed)
    - [2. Create a virtual environment](#2-create-a-virtual-environment)
    - [3. Activate the environment](#3-activate-the-environment)
    - [4. Install dependencies](#4-install-dependencies)
  - [Running the Pipeline](#running-the-pipeline)
    - [Step 1: Prepare the data (Notebooks)](#step-1-prepare-the-data-notebooks)
    - [Step 2: Compute CDR features](#step-2-compute-cdr-features)
    - [Step 3: Generate protein language model embeddings](#step-3-generate-protein-language-model-embeddings)
    - [Step 4: Build model-ready datasets](#step-4-build-model-ready-datasets)
    - [Step 5: Run ensemble modeling](#step-5-run-ensemble-modeling)
  - [Data Access](#data-access)
  - [Attributions](#attributions)
    - [Data](#data)
    - [Code](#code)
    - [README](#readme)
  - [Outputs](#outputs)

---

## Project Structure

```
siads699_capstone/
├── data/
│   ├── raw/                  # Raw input files (not generated — must be provided)
│   ├── processed/            # Cleaned training CSV (output of eda_notebook.ipynb)
│   ├── embeddings/           # PLM embedding pickles for training set
│   │   └── holdout/          # PLM embedding pickles for holdout set
│   ├── modeling/             # Model-ready datasets and result CSVs
│   │   └── holdout/          # Model-ready datasets for holdout set
│   └── test_data/            # Cleaned holdout CSV (output of holdout_cleanup.ipynb)
├── notebooks/
│   ├── eda_notebook.ipynb    # Training data cleaning and EDA
│   └── holdout_cleanup.ipynb # Holdout data cleaning
└── src/
    ├── CDR_work/             # CDR biophysical feature computation
    ├── models/               # PLM wrapper classes and model registry
    ├── pipelines/            # Embedding generation and dataset assembly scripts
    └── prediction_modeling/  # Ensemble modeling and evaluation
```

---

## Environment Setup
This project was originally developed using **Python 3.11** and the **uv** package manager for fast dependency management.  
Equivalent setup instructions using **pip** or **conda** are also provided for users who prefer those tools.

The project requires:
```toml
requires-python = "~=3.11.0"
```
---

## Option 1: Using `uv` (recommended)

This is the primary environment workflow used for this project.

### 1. Ensure Python 3.11 is installed

```bash
python3 --version
```

Expected output:
```bash
Python 3.11.x
```
### 2. Create virtual environment
It is recommended to create a virtual environment so the project dependencies do not interfere with your global Python installation..

```bash
uv venv --python 3.11.15
```

### 3. Activate your virtual environment

On macOS and Linux:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```

#### 3a. Confirm Python requirement in pyproject.toml file

```toml
requires-python = "~=3.11.0"
```

### 4. Install dependencies using lockfile

```bash
uv sync
```

This will install the exact dependency set recorded in uv.lock.

If the lockfile is not available and you need to recreate the environment manually, the original setup used:

```bash
uv add "pycaret[full]"
uv add esm
uv add ankh
```

## Option 2: Using `conda`

If you prefer conda, create an environment with Python 3.11 and install from requirements.txt.

### 1. Create the environment
```bash
conda create -n name_of_your_choice_here python=3.11
```

### 2. Activate it
```bash
conda activate name_of_your_choice_here
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Note: even inside a conda environment, pip is typically the easiest way to install this project’s Python dependencies from requirements.txt.

## Option 3: Using `pip`
If you prefer pip, you can create a standard virtual environment and install from requirements.txt.

### 1. Make sure Python 3.11 is installed

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Activate the environment

On macOS and Linux:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Before running any scripts, ensure you have the two raw data files in `data/raw/`:
- `GDPa1_246 IgGs_cleaned.csv` — 246 antibody training set
- `GDPa3_80 IgGs_full.xlsx` — 80 antibody holdout set

### Step 1: Prepare the data (Notebooks)

Open and run all cells in each notebook in order:

1. `notebooks/eda_notebook.ipynb` — Filters columns, imputes missing assay values (SMAC, HIC, PR_CHO, etc.), and writes the cleaned training dataset to `data/processed/antibody_developability_cleaned.csv`.
2. `notebooks/holdout_cleanup.ipynb` — Merges the holdout Excel sheets, renames columns to match training conventions, and writes `data/test_data/cleaned_holdout_data.csv`.

These notebooks must be run first because every downstream script depends on the cleaned CSVs they produce.

### Step 2: Compute CDR features

```bash
python -m src.CDR_work.cdr_features_titer
```

Computes 62 biophysical descriptors (hydrophobicity, charge, liability motif counts, etc.) from the CDR regions of each antibody's aligned sequence. Outputs `data/modeling/cdr_features_titer.csv`.

Run before embedding generation because these features are merged alongside PLM embeddings during modeling.

### Step 3: Generate protein language model embeddings

```bash
python -m src.pipelines.generate_embeddings
```

Passes the VH and VL sequences for both the training and holdout sets through five protein language models (ESM-C, ProtT5, ProstT5, Ankh, ProtBERT) and saves one `.pkl` file per model per chain to `data/embeddings/` and `data/embeddings/holdout/`.

This is the slowest step. Already-generated embedding files are skipped automatically, so it is safe to re-run if interrupted.

### Step 4: Build model-ready datasets

```bash
python -m src.pipelines.build_model_datasets
```

Merges each model's VH and VL embeddings with the cleaned data (dropping raw sequence columns) and saves one combined `.pkl` per model to `data/modeling/` and `data/modeling/holdout/`.

Must run after Step 3 because it reads the embedding `.pkl` files produced there.

### Step 5: Run ensemble modeling

```bash
python -m src.prediction_modeling.ensemble_beta_3
```

Loads the model-ready datasets, merges in CDR features, and runs cluster-aware cross-validated Lasso regression for each PLM embedding family. After cross-validation, builds a Spearman-weighted ensemble across all base learners and runs failure overlap diagnostics. Saves timestamped result CSVs to `data/modeling/`.

Must run last because it depends on both the CDR features (Step 2) and the model-ready datasets (Step 4).

---

## Data Access

This project uses two publicly available, gated datasets from [Ginkgo Datapoints](https://datapoints.ginkgo.bio):

| Dataset | Antibodies | Access |
|---------|------------|--------|
| GDPa1 | 246 IgGs (training set) | Available on [Hugging Face](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1) |
| GDPa3 | 80 IgGs (holdout set) | Available on the [Ginkgo Datapoints website](https://datapoints.ginkgo.bio/dataset-access) |

Both datasets are gated — you must request access through the respective platform before downloading. The data files are not included in this repository and must be placed in `data/raw/` before running the pipeline.

---

## Attributions

### Data
GDPa1 and GDPa3 datasets are owned by Ginkgo Datapoints and used here under their respective access terms.

### Code
- `src/CDR_work/` — CDR biophysical feature computation code written by Claude AI (Sonnet 4.6), with prompts and direction by Allen Chezick.
- `src/pipelines/`, `src/models/`, and `src/prediction_modeling/ensemble_beta_3.py` — pipeline architecture and cleanup assisted by Claude Code (Sonnet 4.6). Core modeling logic and research design by Eduardo Pacheco and Allen Chezick.
- EDA notebooks (`notebooks/`) — written by Eduardo Pacheco, Allen Chezick, Miranda Thomas.

### README
This README was drafted with assistance from Claude Code (Sonnet 4.6).

---

## Outputs

All result files are written to `data/modeling/` with a timestamp suffix to prevent runs from overwriting each other.

| File | Description |
|------|-------------|
| `final_results_<timestamp>.csv` | Cross-validated R², RMSE, and Spearman ρ per PLM and model type |
| `ensemble_weights_<timestamp>.csv` | Normalized weight assigned to each base learner in the ensemble |
| `failure_overlap_<timestamp>.csv` | Per-antibody error table showing which models consistently fail on the same samples |
| `error_correlation_<timestamp>.csv` | Pairwise Spearman correlation of absolute error vectors across base learners |
| `holdout_results_<timestamp>.csv` | Aggregate holdout metrics per model (only if holdout evaluation is enabled) |
| `holdout_predictions_<timestamp>.csv` | Per-antibody holdout predictions (only if holdout evaluation is enabled) |

> **Note:** Holdout evaluation is disabled by default. To enable it, set `HOLDOUT_CSV_PATH` in the `driver()` function of `src/prediction_modeling/ensemble_beta_3.py` to the path of the cleaned holdout CSV.