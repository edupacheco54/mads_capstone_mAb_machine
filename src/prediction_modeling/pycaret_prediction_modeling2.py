"""
pycaret_prediction_modeling.py

Run a PyCaret regression benchmark for GDPa1 CDR features vs Titer.

Changes from original pycaret_titer_experiment_repo_aware.py:
  - Random train_test_split REMOVED.  All 239 GDPa1 antibodies (those with
    Titer + fold assignment) are used as the CV training pool.  PyCaret's
    internal CV is driven by a PredefinedSplit built from the
    hierarchical_cluster_IgG_isotype_stratified_fold column, which holds out
    entire sequence clusters so the model is tested on novel antibody families.
  - DOWNSTREAM_ASSAY_COLS explicitly stripped before any feature engineering
    (leakage guard consistent with ensemble_beta_2.py).
  - External holdout set (--holdout-data) replaces the random 20% split as the
    final evaluation dataset.  finalize_model retrains on all 239 CV antibodies,
    then predicts once on the holdout.  No data from the holdout ever enters CV.
  - antibody_name preserved in all saved prediction CSVs for traceability.
  - Failure diagnostics: top-N highest abs-error antibodies printed and saved
    for each model.
  - Model artifacts saved with consistent names so final_holdout_eval_pycaret_model.py
    can locate them directly.

Examples
--------
python pycaret_prediction_modeling.py

python pycaret_prediction_modeling.py \\
  --data data/gdpa1_246_iggs_cleaned.csv \\
  --holdout-data data/test_data/cleaned_holdout_data_with_aho.csv \\
  --target Titer

python pycaret_prediction_modeling.py \\
  --data data/gdpa1_246_iggs_cleaned.csv \\
  --holdout-data data/test_data/cleaned_holdout_data_with_aho.csv \\
  --outdir data/modeling/prediction_modeling_results/pycaret_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from sklearn.model_selection import PredefinedSplit

from datetime import datetime

def find_repo_root(start: Path | None = None) -> Path:
    """
    Detect the Git/repo root by walking upward from this file (preferred)
    and then from the current working directory.
    """
    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path(__file__).resolve())
    candidates.append(Path.cwd().resolve())

    seen = set()
    for candidate in candidates:
        for parent in [candidate] + list(candidate.parents):
            if parent in seen:
                continue
            seen.add(parent)
            if (
                (parent / ".git").exists()
                or (parent / "pyproject.toml").exists()
                or parent.name == "capstone699_repo"
            ):
                return parent

    raise FileNotFoundError(
        "Could not detect repo root. Expected one of: .git, pyproject.toml, "
        "or a directory named 'capstone699_repo'."
    )


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from CDR_work.cdr_feature_utils import build_modeling_table, get_numeric_feature_columns

try:
    from pycaret.regression import (
        compare_models,
        finalize_model,
        predict_model,
        pull,
        save_model,
        setup,
        tune_model,
    )
except ImportError as exc:
    raise SystemExit(
        "PyCaret is not installed in this environment. Install it first:\n"
        "  pip install pycaret\n"
        "  uv add pycaret\n"
    ) from exc


# ── Column constants ──────────────────────────────────────────────────────────

FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"

# Downstream assay readouts — causally downstream of expression titer or
# confounded by the same upstream bioprocess.  Must never enter the feature
# matrix.  Consistent with the guard in ensemble_beta_2.py.
DOWNSTREAM_ASSAY_COLS = [
    "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
    "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
    "Tonset", "Tm1", "Tm2",
]

# All non-feature columns to strip before CDR feature engineering.
# antibody_name is kept here because build_modeling_table doesn't need it,
# but we pull it out separately for traceability before dropping.
_META_COLS_TO_DROP = [
    "Unnamed: 0", "antibody_id",
    *DOWNSTREAM_ASSAY_COLS,
    "highest_clinical_trial_asof_feb2025", "est_status_asof_feb2025",
    "hc_subtype", "lc_subtype",
    "hierarchical_cluster_fold", "random_fold",
    "hc_protein_sequence", "hc_dna_sequence",
    "lc_protein_sequence", "lc_dna_sequence",
]

DEFAULT_DATA_RELATIVE    = "data/raw/GDPa1_246 IgGs_cleaned.csv"
DEFAULT_HOLDOUT_RELATIVE = "data/test_data/cleaned_holdout_data_with_aho.csv"
DEFAULT_OUTDIR_RELATIVE  = "data/modeling/prediction_modeling_results/pycaret_results"


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_repo_path(path_arg: str, must_exist: bool = True) -> Path:
    path = Path(path_arg).expanduser()
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Could not find path: {resolved}\n"
            f"Provided argument: {path_arg}\n"
            f"Repo root: {REPO_ROOT}"
        )
    return resolved


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PyCaret regression benchmark for GDPa1 CDR features vs Titer "
                    "using cluster-aware cross-validation."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_RELATIVE,
        help=(
            f"Path to GDPa1 training CSV. Resolved relative to repo root. "
            f"Default: {DEFAULT_DATA_RELATIVE}"
        ),
    )
    parser.add_argument(
        "--holdout-data",
        default=DEFAULT_HOLDOUT_RELATIVE,
        help=(
            "Path to external holdout CSV (must contain Titer + AHO columns). "
            f"Default: {DEFAULT_HOLDOUT_RELATIVE}"
        ),
    )
    parser.add_argument(
        "--target",
        default="Titer",
        help="Target column to predict. Default: Titer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=5,
        help="Number of top models to keep from compare_models. Default: 5",
    )
    parser.add_argument(
        "--metric",
        default="RMSE",
        help="Metric used to rank models in PyCaret. Default: RMSE",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR_RELATIVE,
        help=(
            f"Directory where outputs will be saved. "
            f"Default: {DEFAULT_OUTDIR_RELATIVE}"
        ),
    )
    parser.add_argument(
        "--candidate-models",
        nargs="*",
        default=["ridge", "lasso", "en", "rf", "et", "gbr", "xgboost", "lightgbm", "svm"],
        help="PyCaret model IDs to compare. Default: ridge lasso en rf et gbr xgboost lightgbm svm",
    )
    parser.add_argument(
        "--failure-top-n",
        type=int,
        default=20,
        help="Number of highest-error antibodies to include in failure diagnostics. Default: 20",
    )
    return parser.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "R2":       float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
    }


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_and_guard(csv_path: Path, target_col: str, label: str) -> pd.DataFrame:
    """
    Load a CSV, apply the leakage guard, validate required columns, and report shape.
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] {label}: raw shape = {df.shape}")

    # Drop downstream assay columns (leakage guard).
    dropped = [c for c in DOWNSTREAM_ASSAY_COLS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"[INFO] {label}: leakage guard dropped {len(dropped)} cols: {dropped}")

    leaked = [c for c in DOWNSTREAM_ASSAY_COLS if c in df.columns]
    assert not leaked, f"LEAKAGE COLUMNS STILL PRESENT in {label}: {leaked}"

    if target_col not in df.columns:
        raise ValueError(f"[{label}] Target column '{target_col}' not found.")
    for aho_col in ("heavy_aligned_aho", "light_aligned_aho"):
        if aho_col not in df.columns:
            raise ValueError(f"[{label}] Required column '{aho_col}' not found.")

    return df


def build_eval_df(
    raw_df: pd.DataFrame,
    target_col: str,
    label: str,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Build a CDR feature matrix from raw_df.  Returns:
        eval_df      : feature columns + target (rows with valid AHO + target)
        antibody_names: Series aligned to eval_df index (or None)

    Rows missing AHO sequences or target are reported and excluded explicitly.
    """
    aho_missing = (
        raw_df["heavy_aligned_aho"].isna() | raw_df["light_aligned_aho"].isna()
    )
    tgt_missing = raw_df[target_col].isna()
    exclude = aho_missing | tgt_missing
    n_excluded = exclude.sum()

    if n_excluded > 0:
        excluded_names = (
            raw_df.loc[exclude, "antibody_name"].tolist()
            if "antibody_name" in raw_df.columns
            else raw_df.index[exclude].tolist()
        )
        print(
            f"[WARN] {label}: {n_excluded} rows excluded "
            f"(AHO missing={aho_missing.sum()}, target missing={tgt_missing.sum()})\n"
            f"       Excluded: {excluded_names}"
        )

    usable = raw_df[~exclude].reset_index(drop=True)
    print(f"[INFO] {label}: {len(usable)} usable rows after exclusion check")

    antibody_names = usable["antibody_name"] if "antibody_name" in usable.columns else None

    # build_modeling_table internally calls build_feature_matrix and attaches target.
    modeling_tbl = build_modeling_table(usable, target_col=target_col)
    feature_cols = get_numeric_feature_columns(modeling_tbl, target_col=target_col)
    eval_df = modeling_tbl[feature_cols + [target_col]].copy()

    return eval_df, antibody_names, usable


# ── Failure diagnostics ───────────────────────────────────────────────────────

def failure_diagnostics(
    antibody_names: pd.Series | None,
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    top_n: int,
    out_path: Path,
    rank: int,
) -> pd.DataFrame:
    """
    Build a per-antibody prediction table, compute abs_error, print and save
    the top_n highest-error antibodies.

    Returns the full per-antibody DataFrame.
    """
    diag_df = pd.DataFrame({
        "antibody_name": (
            antibody_names.values if antibody_names is not None
            else np.arange(len(y_true))
        ),
        "y_true":            y_true.values,
        "y_pred":            y_pred.values,
        "abs_error":         np.abs(y_true.values - y_pred.values),
        "signed_error":      y_pred.values - y_true.values,
    }).sort_values("abs_error", ascending=False).reset_index(drop=True)

    print(f"\n[{model_name}] Top {top_n} highest-error antibodies (holdout):")
    print(diag_df.head(top_n).to_string(index=False))

    diag_df.to_csv(
        out_path / f"holdout_per_antibody_rank_{rank}_{model_name}.csv",
        index=False,
    )
    return diag_df


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(
    data_path: str,
    holdout_path: str,
    target_col: str,
    outdir: str,
    seed: int = 42,
    n_select: int = 5,
    metric: str = "RMSE",
    candidate_models: Iterable[str] | None = None,
    failure_top_n: int = 20,
) -> None:
    data_file    = resolve_repo_path(data_path,    must_exist=True)
    holdout_file = resolve_repo_path(holdout_path, must_exist=True)
    out_path     = resolve_repo_path(outdir,        must_exist=False)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root   : {REPO_ROOT}")
    print(f"[INFO] Training CSV: {data_file}")
    print(f"[INFO] Holdout CSV : {holdout_file}")
    print(f"[INFO] Output dir  : {out_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # ── Load and guard training data ──────────────────────────────────────────
    raw_train = load_and_guard(data_file, target_col, label="GDPa1 training")

    # Pull the fold column out BEFORE feature engineering strips non-feature cols.
    if FOLD_COL not in raw_train.columns:
        raise ValueError(
            f"Fold column '{FOLD_COL}' not found in training data. "
            "Cannot build cluster-aware CV splits."
        )

    # ── Build training feature matrix ─────────────────────────────────────────
    train_eval_df, train_names, train_usable = build_eval_df(
        raw_train, target_col, label="GDPa1 training"
    )

    # Re-attach the fold column aligned to the usable rows.
    # train_usable was reset_index(drop=True) inside build_eval_df, so
    # we need to extract folds from the same usable subset.
    fold_values = (
        raw_train[~(
            raw_train["heavy_aligned_aho"].isna() |
            raw_train["light_aligned_aho"].isna() |
            raw_train[target_col].isna()
        )][FOLD_COL].reset_index(drop=True).astype(int).values
    )

    assert len(fold_values) == len(train_eval_df), (
        f"Fold array length ({len(fold_values)}) != training df length "
        f"({len(train_eval_df)}).  Check AHO/target exclusion logic."
    )

    n_folds = len(np.unique(fold_values))
    print(f"\n[INFO] Training set: {len(train_eval_df)} antibodies | "
          f"{n_folds} cluster folds | "
          f"{train_eval_df.shape[1] - 1} CDR features")
    print(f"[INFO] Fold distribution: "
          + ", ".join(f"fold{f}={n}" for f, n in
                      zip(*np.unique(fold_values, return_counts=True))))

    # ── Build PredefinedSplit for cluster-aware CV ─────────────────────────────
    # PredefinedSplit(test_fold=fold_values) creates exactly n_folds splits:
    # in split i, antibodies with fold_values==i are the test set and the rest
    # are training.  This mirrors the do_hugging_face fold logic in
    # ensemble_beta_3.py so PyCaret CV is evaluated on the same partition scheme.
    

    # ── Load and guard holdout data ───────────────────────────────────────────
    raw_holdout = load_and_guard(holdout_file, target_col, label="Holdout")
    holdout_eval_df, holdout_names, _ = build_eval_df(
        raw_holdout, target_col, label="Holdout"
    )

    # Align holdout feature columns to training feature columns.
    train_feature_cols = [c for c in train_eval_df.columns if c != target_col]
    missing_in_holdout = [c for c in train_feature_cols if c not in holdout_eval_df.columns]
    if missing_in_holdout:
        print(f"[WARN] Holdout missing {len(missing_in_holdout)} feature cols — "
              f"will be NaN-imputed by PyCaret: {missing_in_holdout}")
    holdout_eval_df = holdout_eval_df.reindex(
        columns=train_feature_cols + [target_col]
    )

    print(f"\n[INFO] Holdout set: {len(holdout_eval_df)} antibodies | "
          f"{len(train_feature_cols)} features")

    # ── PyCaret setup ─────────────────────────────────────────────────────────
    # fold_strategy=ps passes our PredefinedSplit directly to PyCaret.
    # fold=n_folds tells PyCaret how many splits to expect.
    # No test_data is passed — the external holdout is evaluated separately after
    # finalize_model so no holdout rows ever touch the CV loop.
    print("\n[INFO] Running PyCaret setup with cluster-aware CV folds...")
    setup(
        data=train_eval_df,
        target=target_col,
        session_id=seed,
        fold_strategy="groupkfold",
        fold=n_folds,
        fold_groups=pd.Series(fold_values, index=train_eval_df.index),
        preprocess=True,
        imputation_type="simple",
        numeric_imputation="median",
        normalize=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        verbose=False,
        html=False,
    )

    # ── Compare models ────────────────────────────────────────────────────────
    print("\n[INFO] Running compare_models...")
    best_models = compare_models(
        include=list(candidate_models) if candidate_models else None,
        sort=metric,
        n_select=n_select,
    )
    leaderboard = pull().copy()
    leaderboard.to_csv(out_path / "leaderboard_compare_models.csv", index=False)
    print("\n[INFO] Leaderboard (CV metrics):")
    print(leaderboard.to_string(index=False))

    if not isinstance(best_models, list):
        best_models = [best_models]

    # ── Per-model tune → finalize → holdout evaluation ────────────────────────
    summary_rows = []

    for idx, model in enumerate(best_models, start=1):
        tuned_model  = tune_model(model, optimize=metric)
        tuned_results = pull().copy()
        model_name   = tuned_model.__class__.__name__

        print(f"\n[INFO] Rank {idx}: {model_name} — finalizing on all {len(train_eval_df)} CV antibodies...")

        # finalize_model retrains on the FULL training set (all 239 antibodies)
        # using the best hyperparameters found during CV.  No antibodies from the
        # holdout are used here.
        final_model = finalize_model(tuned_model)

        # Evaluate once on the external holdout (no further splitting).
        print(f"[INFO] Rank {idx}: {model_name} — predicting on holdout ({len(holdout_eval_df)} antibodies)...")
        holdout_preds_df = predict_model(final_model, data=holdout_eval_df.copy())

        if "prediction_label" not in holdout_preds_df.columns:
            raise ValueError(
                f"PyCaret prediction output missing 'prediction_label' for {model_name}."
            )

        y_true = holdout_preds_df[target_col]
        y_pred = holdout_preds_df["prediction_label"]
        metrics = compute_metrics(y_true, y_pred)

        print(f"[INFO] {model_name} HOLDOUT | N={len(y_true)} | "
              f"RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f} | "
              f"Spearman={metrics['Spearman']:.4f}")

        # ── Build tidy output CSV with antibody_name ──────────────────────────
        output_preds = holdout_preds_df.copy()
        output_preds.insert(0, "antibody_name",
                            holdout_names.values if holdout_names is not None
                            else np.arange(len(y_true)))
        output_preds["abs_error"]    = np.abs(y_true.values - y_pred.values)
        output_preds["signed_error"] = y_pred.values - y_true.values

        # Column order: identifiers first, then features.
        id_cols = ["antibody_name", target_col, "prediction_label",
                   "abs_error", "signed_error"]
        feat_cols_out = [c for c in output_preds.columns if c not in id_cols]
        output_preds = output_preds[id_cols + feat_cols_out]

        # ── Save per-model artifacts ───────────────────────────────────────────
        tuned_results.to_csv(
            out_path / f"cv_results_rank_{idx}_{model_name}_{timestamp}.csv", index=False
        )
        output_preds.to_csv(
            out_path / f"holdout_predictions_rank_{idx}_{model_name}_{timestamp}.csv", index=False
        )
        save_model(
            final_model,
            str(out_path / f"final_model_rank_{idx}_{model_name}_{timestamp}")
        )
        print(f"[INFO] Saved: final_model_rank_{idx}_{model_name}_{timestamp}.pkl")

        # ── Failure diagnostics ───────────────────────────────────────────────
        diag_df = failure_diagnostics(
            antibody_names=holdout_names,
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            top_n=failure_top_n,
            out_path=out_path,
            rank=idx,
        )

        summary_rows.append({
            "rank":        idx,
            "model_class": model_name,
            "n_holdout":   len(y_true),
            **metrics,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError(
            "[ERROR] No models completed successfully — compare_models returned an empty leaderboard. "
            "Check PyCaret setup and fold_strategy configuration."
        )
    summary_df = summary_df.sort_values("RMSE").reset_index(drop=True)
    summary_df.to_csv(out_path / f"holdout_summary_pycaret_{timestamp}.csv", index=False)

    print("\n" + "=" * 60)
    print("HOLDOUT SUMMARY (all models, sorted by RMSE)")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # ── Run manifest ──────────────────────────────────────────────────────────
    manifest = {
        "repo_root":        str(REPO_ROOT),
        "data_path":        str(data_file),
        "holdout_path":     str(holdout_file),
        "target":           target_col,
        "fold_column":      FOLD_COL,
        "n_folds":          int(n_folds),
        "n_rows_training":  int(len(train_eval_df)),
        "n_rows_holdout":   int(len(holdout_eval_df)),
        "n_features":       int(len(train_feature_cols)),
        "seed":             seed,
        "metric":           metric,
        "candidate_models": list(candidate_models) if candidate_models else None,
        "outdir":           str(out_path),
    }
    with open(out_path / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[INFO] All outputs saved to: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_path=args.data,
        holdout_path=args.holdout_data,
        target_col=args.target,
        outdir=args.outdir,
        seed=args.seed,
        n_select=args.n_select,
        metric=args.metric,
        candidate_models=args.candidate_models,
        failure_top_n=args.failure_top_n,
    )