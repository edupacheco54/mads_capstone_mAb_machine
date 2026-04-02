"""Run a PyCaret regression benchmark for GDPa1 CDR features vs Titer.

This version is repo-aware:
- detects the repo root automatically
- resolves paths relative to the repo root
- avoids hardcoded /home/<user>/... imports
- uses a default output directory inside the repo

Examples
--------
python pycaret_titer_experiment.py

python pycaret_titer_experiment.py \
  --data data/gdpa1_246_iggs_cleaned.csv \
  --target Titer

python pycaret_titer_experiment.py \
  --data data/gdpa1_246_iggs_cleaned.csv \
  --outdir data/modeling/prediction_modeling_results/initial_sandbox
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def find_repo_root(start: Path | None = None) -> Path:
    """
    Detect the Git/repo root by walking upward from this file (preferred)
    and then from the current working directory.

    Signals used:
    - .git directory
    - pyproject.toml
    - repo directory name 'capstone699_repo'
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
        "PyCaret is not installed in this environment. Install it first, for example:\n"
        "  pip install pycaret\n"
        "or in conda:\n"
        "  conda install -c conda-forge pycaret\n"
        "or in uv:\n"
        "  uv add pycaret\n"
    ) from exc


DEFAULT_DATA_RELATIVE = "data/gdpa1_246_iggs_cleaned.csv"
DEFAULT_OUTDIR_RELATIVE = "data/modeling/prediction_modeling_results/initial_sandbox"


def resolve_repo_path(path_arg: str, must_exist: bool = True) -> Path:
    """
    Resolve a path in a teammate-friendly way.

    Rules:
    1. If absolute, use it directly.
    2. Otherwise treat it as relative to the repo root.
    """
    path = Path(path_arg).expanduser()

    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (REPO_ROOT / path).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Could not find path: {resolved}\n"
            f"Provided argument: {path_arg}\n"
            f"Repo root: {REPO_ROOT}"
        )

    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PyCaret regression benchmark for GDPa1 CDR features vs Titer."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_RELATIVE,
        help=(
            "Path to GDPa1 CSV. Absolute paths are allowed; otherwise the path is "
            f"resolved relative to the repo root. Default: {DEFAULT_DATA_RELATIVE}"
        ),
    )
    parser.add_argument(
        "--target",
        default="Titer",
        help="Target column to predict. Default: Titer",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of rows used for the random holdout set. Default: 0.20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=5,
        help="Number of CV folds used by PyCaret on the training split. Default: 5",
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
            "Directory where outputs will be saved. Absolute paths are allowed; "
            "otherwise resolved relative to the repo root. "
            f"Default: {DEFAULT_OUTDIR_RELATIVE}"
        ),
    )
    parser.add_argument(
        "--candidate-models",
        nargs="*",
        default=["ridge", "lasso", "en", "rf", "et", "gbr", "xgboost", "lightgbm", "svm"],
        help="Optional subset to compare. Names must be valid PyCaret model IDs.",
    )
    return parser.parse_args()


def compute_holdout_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def run_experiment(
    data_path: str,
    target_col: str,
    outdir: str,
    test_size: float = 0.20,
    seed: int = 42,
    fold: int = 5,
    n_select: int = 5,
    metric: str = "RMSE",
    candidate_models: Iterable[str] | None = None,
) -> None:
    data_file = resolve_repo_path(data_path, must_exist=True)
    out_path = resolve_repo_path(outdir, must_exist=False)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Data file: {data_file}")
    print(f"[INFO] Output dir: {out_path}")

    df = pd.read_csv(data_file)
    modeling_df = build_modeling_table(df, target_col=target_col)
    feature_cols = get_numeric_feature_columns(modeling_df, target_col=target_col)

    keep_cols = feature_cols + [target_col]
    model_df = modeling_df[keep_cols].copy()
    model_df = model_df[model_df[target_col].notna()].copy()

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=seed,
    )

    setup(
        data=train_df,
        test_data=test_df,
        target=target_col,
        session_id=seed,
        fold=fold,
        preprocess=True,
        imputation_type="simple",
        numeric_imputation="median",
        normalize=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        verbose=False,
        html=False,
    )

    best_models = compare_models(
        include=list(candidate_models) if candidate_models else None,
        sort=metric,
        n_select=n_select,
    )
    leaderboard = pull().copy()
    leaderboard.to_csv(out_path / "leaderboard_compare_models.csv", index=False)

    if not isinstance(best_models, list):
        best_models = [best_models]

    summary_rows = []

    for idx, model in enumerate(best_models, start=1):
        tuned_model = tune_model(model, optimize=metric)
        tuned_results = pull().copy()
        model_name = tuned_model.__class__.__name__

        final_model = finalize_model(tuned_model)
        holdout_preds = predict_model(final_model, data=test_df)
        metrics = compute_holdout_metrics(
            holdout_preds[target_col], holdout_preds["prediction_label"]
        )

        tuned_results.to_csv(out_path / f"cv_results_rank_{idx}_{model_name}.csv", index=False)
        holdout_preds.to_csv(out_path / f"holdout_predictions_rank_{idx}_{model_name}.csv", index=False)
        save_model(final_model, str(out_path / f"final_model_rank_{idx}_{model_name}"))

        summary_rows.append(
            {
                "rank": idx,
                "model_class": model_name,
                **metrics,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE")
    summary_df.to_csv(out_path / "holdout_summary.csv", index=False)

    manifest = {
        "repo_root": str(REPO_ROOT),
        "data_path": str(data_file),
        "target": target_col,
        "n_rows_total": int(len(df)),
        "n_rows_modeling": int(len(model_df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "test_size": test_size,
        "seed": seed,
        "fold": fold,
        "metric": metric,
        "candidate_models": list(candidate_models) if candidate_models else None,
        "outdir": str(out_path),
    }
    with open(out_path / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved outputs to:", out_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_path=args.data,
        target_col=args.target,
        outdir=args.outdir,
        test_size=args.test_size,
        seed=args.seed,
        fold=args.fold,
        n_select=args.n_select,
        metric=args.metric,
        candidate_models=args.candidate_models,
    )