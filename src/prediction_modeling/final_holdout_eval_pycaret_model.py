import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# repo-aware import
current = Path(__file__).resolve()
for parent in [current] + list(current.parents):
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from CDR_work.cdr_feature_utils import build_feature_matrix

try:
    from pycaret.regression import load_model, predict_model
except ImportError as exc:
    raise SystemExit(
        "PyCaret is not installed in this environment. Install it first, then rerun."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PyCaret model on the external holdout dataset."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to saved PyCaret model, without the .pkl suffix.",
    )
    parser.add_argument(
        "--holdout-data",
        required=True,
        help="Path to holdout CSV containing Titer plus aligned AHO columns.",
    )
    parser.add_argument(
        "--target",
        default="Titer",
        help="Target column name. Default: Titer",
    )
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def main(model_path, holdout_path, target_col="Titer"):
    print("[INFO] Loading holdout data...")
    holdout_df = pd.read_csv(holdout_path)

    print("[INFO] Building holdout CDR feature matrix...")
    holdout_feat = build_feature_matrix(holdout_df)

    eval_df = pd.concat(
        [holdout_feat.reset_index(drop=True), holdout_df[[target_col]].reset_index(drop=True)],
        axis=1,
    ).dropna()

    print(f"[INFO] Holdout rows used for evaluation: {len(eval_df)}")

    print("[INFO] Loading saved PyCaret model...")
    model = load_model(model_path)

    print("[INFO] Predicting on holdout...")
    preds_df = predict_model(model, data=eval_df.copy())

    if "prediction_label" not in preds_df.columns:
        raise ValueError("PyCaret predictions missing 'prediction_label' column.")

    y_true = preds_df[target_col]
    y_pred = preds_df["prediction_label"]

    metrics = compute_metrics(y_true, y_pred)

    print("\n=== HOLDOUT RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_path = Path(str(model_path) + "_holdout_predictions.csv")
    preds_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved holdout predictions to: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        holdout_path=args.holdout_data,
        target_col=args.target,
    )
