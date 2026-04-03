import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

# repo-aware import
current = Path(__file__).resolve()
for parent in [current] + list(current.parents):
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from CDR_work.cdr_feature_utils import build_feature_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--holdout-data", required=True)
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def main(train_path, holdout_path):
    print("[INFO] Loading data...")

    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)

    # Build feature matrices explicitly
    train_feat = build_feature_matrix(train_df)
    holdout_feat = build_feature_matrix(holdout_df)

    # Force holdout to have the exact same feature columns/order as train
    feature_cols = train_feat.columns.tolist()
    holdout_feat = holdout_feat.reindex(columns=feature_cols)

    train_model_df = pd.concat(
        [train_feat.reset_index(drop=True), train_df[["Titer"]].reset_index(drop=True)],
        axis=1,
    ).dropna()

    holdout_model_df = pd.concat(
        [holdout_feat.reset_index(drop=True), holdout_df[["Titer"]].reset_index(drop=True)],
        axis=1,
    ).dropna()

    X_train = train_model_df[feature_cols]
    y_train = train_model_df["Titer"]

    X_holdout = holdout_model_df[feature_cols]
    y_holdout = holdout_model_df["Titer"]

    print("[INFO] Training model on full dataset...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating on holdout...")
    preds = model.predict(X_holdout)

    metrics = compute_metrics(y_holdout, preds)

    print("\\n=== HOLDOUT RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args.train_data, args.holdout_data)
