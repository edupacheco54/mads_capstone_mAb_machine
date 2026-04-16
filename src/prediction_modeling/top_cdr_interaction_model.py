"""
top_cdr_interaction_model.py

Minimal starter script to test whether a small set of top CDR features,
plus a few biologically motivated interaction features, can explain Titer
and/or separate easy vs hard-to-predict antibodies.

Inputs
------
1. GDPa1 raw CSV
2. failure_overlap CSV
3. cdr_features_titer CSV

What it does
------------
- Normalizes antibody_name keys
- Rebuilds failure_group from n_models_high_err using the max count found
  in the failure file as the "all models failed" threshold
- Merges GDPa1 + failure_overlap + CDR features
- Selects a small top-feature set based on your failure analysis
- Adds a few interaction / nonlinear helper features
- Fits Ridge regression with CV on:
    * all antibodies
    * consistently_right only
    * consistently_wrong only
- Saves merged data and metrics CSVs

Example
-------
python top_cdr_interaction_model.py \
  --gdpa1-csv ... \
  --failure-csv ... \
  --cdr-csv ... \
  --out-dir ../../data/modeling \
  --include-subtype \
  --tag with_subtype
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOP_CDR_FEATURES = [
    "cdrL3_len_CDR_feature",
    "cdrL1_charge_CDR_feature",
    "cdrH1_hyd_frac_CDR_feature",
    "cdrL2_oxidn_CDR_feature",
    "cdrL3_isom_CDR_feature",
    "cdrH3_gravy_CDR_feature",
    "cdrL3_deamid_CDR_feature",
    "cdrL1_len_CDR_feature",
    "cdrL3_charge_CDR_feature",
    "VL_unpaired_cys_CDR_feature",
    "VH_gravy_CDR_feature",
    "cdrL1_gravy_CDR_feature",
    "VL_isom_total_CDR_feature",
    "cdrH1_gravy_CDR_feature",
    "VL_deamid_total_CDR_feature",
    "cdrH2_charge_CDR_feature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdpa1-csv", required=True)
    parser.add_argument("--tag", default=None, help="Optional experiment tag")
    parser.add_argument("--failure-csv", required=True)
    parser.add_argument("--cdr-csv", required=True)
    parser.add_argument("--target", default="Titer")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge alpha. Keep simple at first.",
    )
    parser.add_argument(
        "--include-subtype",
        action="store_true",
        help="If set, include hc_subtype and lc_subtype as categorical features.",
    )
    return parser.parse_args()


def normalize_key(df: pd.DataFrame, col: str = "antibody_name") -> pd.DataFrame:
    df = df.copy()
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return df


def rebuild_failure_group(failure_df: pd.DataFrame) -> pd.DataFrame:
    failure_df = failure_df.copy()
    max_failures = int(failure_df["n_models_high_err"].max())

    failure_df["failure_group"] = np.select(
        [
            failure_df["n_models_high_err"] == max_failures,
            failure_df["n_models_high_err"] == 0,
        ],
        [
            "consistently_wrong",
            "consistently_right",
        ],
        default="sometimes_wrong",
    )
    return failure_df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Only create features when source columns exist
    def has(*cols: str) -> bool:
        return all(c in df.columns for c in cols)

    if "cdrL1_charge_CDR_feature" in df.columns:
        df["cdrL1_charge_abs"] = df["cdrL1_charge_CDR_feature"].abs()

    if "cdrL3_charge_CDR_feature" in df.columns:
        df["cdrL3_charge_abs"] = df["cdrL3_charge_CDR_feature"].abs()

    if has("cdrL3_len_CDR_feature", "cdrL3_charge_CDR_feature"):
        df["cdrL3_len_x_charge"] = (
            df["cdrL3_len_CDR_feature"] * df["cdrL3_charge_CDR_feature"]
        )

    if has("cdrL3_len_CDR_feature", "cdrH1_hyd_frac_CDR_feature"):
        df["cdrL3_len_x_cdrH1_hyd"] = (
            df["cdrL3_len_CDR_feature"] * df["cdrH1_hyd_frac_CDR_feature"]
        )

    if has("cdrL1_charge_CDR_feature", "cdrH1_hyd_frac_CDR_feature"):
        df["cdrL1_charge_x_cdrH1_hyd"] = (
            df["cdrL1_charge_CDR_feature"] * df["cdrH1_hyd_frac_CDR_feature"]
        )

    if has("cdrH3_gravy_CDR_feature", "VH_gravy_CDR_feature"):
        df["cdrH3_minus_VH_gravy"] = (
            df["cdrH3_gravy_CDR_feature"] - df["VH_gravy_CDR_feature"]
        )

    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def run_cv(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    n_splits: int,
    alpha: float,
    include_subtype: bool,
) -> tuple[dict, pd.DataFrame]:
    df = df.copy()

    numeric_features = [c for c in TOP_CDR_FEATURES if c in df.columns]
    # fallback if empty
    if len(numeric_features) == 0:
        print("[WARN] No TOP_CDR_FEATURES found, falling back to all CDR features")
        numeric_features = [c for c in df.columns if c.endswith("_CDR_feature")]
    engineered_features = [
        c for c in [
            "cdrL1_charge_abs",
            "cdrL3_charge_abs",
            "cdrL3_len_x_charge",
            "cdrL3_len_x_cdrH1_hyd",
            "cdrL1_charge_x_cdrH1_hyd",
            "cdrH3_minus_VH_gravy",
        ]
        if c in df.columns
    ]

    categorical_features = []
    if include_subtype:
        for c in ["hc_subtype", "lc_subtype"]:
            if c in df.columns:
                categorical_features.append(c)

    feature_cols = numeric_features + engineered_features + categorical_features

    if len(feature_cols) == 0:
        raise ValueError(
            "No usable features found. Check whether CDR columns were renamed with _CDR_feature."
        )

    keep_cols = feature_cols + [target_col]
    df = df[keep_cols].copy()
    df = df[df[target_col].notna()].copy()

    if len(df) < n_splits:
        raise ValueError(f"Not enough rows ({len(df)}) for n_splits={n_splits}")

    X = df[feature_cols]
    y = df[target_col].to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features + engineered_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("ridge", Ridge(alpha=alpha)),
    ])

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    metrics = compute_metrics(y, y_pred)

    pred_df = df.copy()
    pred_df["y_true"] = y
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])

    return metrics, pred_df


def main() -> None:
    args = parse_args()
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"top_cdr_interaction{tag}_{timestamp}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gdpa1_df = normalize_key(pd.read_csv(args.gdpa1_csv))
    failure_df = normalize_key(pd.read_csv(args.failure_csv))
    cdr_df = normalize_key(pd.read_csv(args.cdr_csv))

    cdr_feature_cols = [c for c in cdr_df.columns if c != "antibody_name"]
    cdr_df = cdr_df.rename(columns={c: f"{c}_CDR_feature" for c in cdr_feature_cols})


    failure_df = rebuild_failure_group(failure_df)

    # Keep only the failure columns we need
    failure_keep = [
        c for c in [
            "antibody_name",
            "n_models_high_err",
            "failure_group",
        ]
        if c in failure_df.columns
    ]
    failure_slim = failure_df[failure_keep].copy()

    merged = gdpa1_df.merge(failure_slim, on="antibody_name", how="left")
    merged = merged.merge(cdr_df, on="antibody_name", how="left")

    merged = add_interaction_features(merged)

    merged.to_csv(out_dir / "merged_top_cdr_modeling_table.csv", index=False)

    results = []
    pred_tables = []

    subsets = {
        "all_rows": merged,
        "consistently_right": merged[merged["failure_group"] == "consistently_right"].copy(),
        "consistently_wrong": merged[merged["failure_group"] == "consistently_wrong"].copy(),
    }

    for subset_name, subset_df in subsets.items():
        if subset_df.empty:
            print(f"[WARN] {subset_name}: no rows, skipping")
            continue

        try:
            metrics, pred_df = run_cv(
                df=subset_df,
                target_col=args.target,
                seed=args.seed,
                n_splits=args.n_splits,
                alpha=args.alpha,
                include_subtype=args.include_subtype,
            )
        except ValueError as exc:
            print(f"[WARN] {subset_name}: {exc}")
            continue

        row = {
            "subset": subset_name,
            "n_rows": int(len(pred_df)),
            "include_subtype": bool(args.include_subtype),
            "alpha": float(args.alpha),
            **metrics,
        }
        results.append(row)

        pred_df = pred_df.copy()
        pred_df["subset"] = subset_name
        pred_tables.append(pred_df)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("[ERROR] No results were produced.")
        return

    results_df = results_df.sort_values("RMSE")
    results_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    if pred_tables:
        preds_all = pd.concat(pred_tables, ignore_index=True)
        preds_all.to_csv(out_dir / "cv_predictions_by_subset.csv", index=False)

    print("\n=== METRICS SUMMARY ===")
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No results were produced.")


if __name__ == "__main__":
    main()