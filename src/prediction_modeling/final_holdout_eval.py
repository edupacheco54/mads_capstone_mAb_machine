"""
final_holdout_eval.py

Trains a selected sklearn regression model on the full GDPa1 training set
using engineered CDR feature matrices, then evaluates that fitted model on
an external holdout dataset and reports RMSE, MAE, R2, and Spearman correlation.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
import json
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR

# repo-aware import
# search upward for the repo root containing /src so this script can be run
# from different working directories without breaking imports.
current = Path(__file__).resolve()
for parent in [current] + list(current.parents):
    if (parent / "src").exists():
        sys.path.insert(0, str(parent / "src"))
        break

from CDR_work.cdr_feature_utils import build_feature_matrix

# Simple registry that maps model names from the CLI to constructors.
# Lambdas let us inject fixed defaults like random_state where supported.
MODEL_REGISTRY = {
    "Ridge":      Ridge,
    "Lasso":      lambda **kw: Lasso(random_state=42, **kw),
    "ElasticNet": lambda **kw: ElasticNet(random_state=42, **kw),
    "SVR_rbf":    lambda **kw: SVR(kernel="rbf", **kw),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--holdout-data", required=True)
    parser.add_argument("--model-name", default="Ridge",
        choices=["Ridge", "Lasso", "ElasticNet", "SVR_rbf"])
    parser.add_argument("--model-params", default="{}", type=str,
        help='JSON string of params e.g. \'{"alpha": 10}\'')
    return parser.parse_args()

def compute_metrics(y_true, y_pred):
    # Centralized metric function keeps evaluation reporting consistent
    # across train/holdout scripts.
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def main(train_path, holdout_path, model_name="Ridge", model_params="{}"):
    print("[INFO] Loading data...")

    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)

    # Build feature matrices explicitly using the same feature engineering logic
    # for both train and holdout. This is crucial so the holdout sees the same
    # feature definition as the training dataset.
    train_feat = build_feature_matrix(train_df)
    holdout_feat = build_feature_matrix(holdout_df)

    # Force holdout to have the exact same feature columns and same column order
    # as the training matrix. reindex() also creates any missing columns if needed.
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
    
    # model_params comes in as a JSON string from the CLI, e.g. '{"alpha": 10}'    
    params = json.loads(model_params)
    
    # Instantiate chosen model from registry.    
    model = MODEL_REGISTRY[model_name](**params)
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

'''
Call it using the best row from final_results CSV:

python final_holdout_eval.py \
    --train-data ../../data/raw/GDPa1_246\ IgGs_cleaned.csv \
    --holdout-data ../../data/test_data/cleaned_holdout_data_with_aho.csv \
    --model-name Ridge \
    --model-params '{"alpha": 10}'

'''
