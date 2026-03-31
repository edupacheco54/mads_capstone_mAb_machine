"""Run a PyCaret regression benchmark for GDPa1 CDR features vs Titer.

Example
-------
python pycaret_titer_experiment.py \
  --data /path/to/gdpa1_246_iggs_cleaned.csv \
  --target Titer \
  --outdir outputs/cdr_pycaret
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cdr_feature_utils import build_modeling_table, get_numeric_feature_columns

try:
    from pycaret.regression import (
        compare_models,
        create_model,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to GDPa1 CSV with aligned AHO columns.")
    parser.add_argument("--target", default="Titer")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=5)
    parser.add_argument("--metric", default="RMSE")
    parser.add_argument("--outdir", required=True)
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
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
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

    exp = setup(
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
    leaderboard = pull()
    leaderboard.to_csv(out_path / "leaderboard_compare_models.csv", index=False)

    if not isinstance(best_models, list):
        best_models = [best_models]

    summary_rows = []
    tuned_models = []

    for idx, model in enumerate(best_models, start=1):
        base_results = pull()
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
        tuned_models.append(final_model)

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE")
    summary_df.to_csv(out_path / "holdout_summary.csv", index=False)

    manifest = {
        "data_path": data_path,
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
