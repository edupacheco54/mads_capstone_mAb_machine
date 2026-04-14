"""
final_holdout_eval_pycaret_model.py

Evaluate a saved PyCaret regression model on the external holdout dataset.

Changes vs original:
  - DOWNSTREAM_ASSAY_COLS explicitly stripped from holdout_df before any
    processing (leakage guard, consistent with ensemble_beta_3.py).
  - antibody_name preserved through to the output CSV so every prediction row
    is identifiable.
  - dropna() replaced with explicit per-row diagnostics: rows missing AHO
    sequences are reported and excluded with a clear count; the remaining
    antibodies all receive a prediction (no silent shrinkage).
  - Output CSV column order: antibody_name, y_true, prediction_label,
    abs_error, then all feature columns.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

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
        "PyCaret is not installed in this environment. "
        "Install it first, then rerun."
    ) from exc


# ── Downstream assay readouts — must never enter the feature matrix ───────────
# Consistent with the DOWNSTREAM_ASSAY_COLS guard in ensemble_beta_2.py.
DOWNSTREAM_ASSAY_COLS = [
    "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
    "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
    "Tonset", "Tm1", "Tm2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PyCaret model on the external holdout dataset."
    )
    parser.add_argument(
        "--model-path",
        default="../../data/modeling/prediction_modeling_results/pycaret_results/final_model_rank_3_RandomForestRegressor_20260414_1815" ,
        help="Path to saved PyCaret model, without the .pkl suffix.",
    )
    parser.add_argument(
        "--holdout-data",
        default="../../data/test_data/cleaned_holdout_data_with_aho.csv",
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
    print(f"[INFO] Raw holdout shape: {holdout_df.shape}")

    # ── Leakage guard ─────────────────────────────────────────────────────────
    # Strip downstream assay readouts from holdout_df before any feature
    # engineering or model inference.  build_feature_matrix only reads AHO /
    # sequence columns so leakage wouldn't occur through that path, but being
    # explicit here keeps the pipeline consistent with ensemble_beta_2.py and
    # prevents any future code from accidentally forwarding these columns to
    # predict_model as covariates.
    cols_to_drop = [c for c in DOWNSTREAM_ASSAY_COLS if c in holdout_df.columns]
    if cols_to_drop:
        holdout_df = holdout_df.drop(columns=cols_to_drop)
        print(f"[INFO] Leakage guard: dropped {len(cols_to_drop)} downstream assay "
              f"columns: {cols_to_drop}")

    # Hard assertion — fail loudly if any assay column survived.
    leaked = [c for c in DOWNSTREAM_ASSAY_COLS if c in holdout_df.columns]
    assert not leaked, f"LEAKAGE COLUMNS STILL PRESENT after drop: {leaked}"

    # ── Preserve antibody_name for output traceability ────────────────────────
    has_name_col = "antibody_name" in holdout_df.columns
    if not has_name_col:
        print("[WARN] 'antibody_name' column not found in holdout CSV. "
              "Output will use integer row index as identifier.")

    # ── Identify rows that can be evaluated ───────────────────────────────────
    # build_feature_matrix returns an empty-dict row (all NaN) when
    # heavy_aligned_aho or light_aligned_aho is missing.  We detect these
    # upfront, report them, and exclude them from evaluation explicitly rather
    # than relying on a silent dropna().
    aho_missing_mask = (
        holdout_df["heavy_aligned_aho"].isna() | holdout_df["light_aligned_aho"].isna()
    )
    target_missing_mask = holdout_df[target_col].isna()
    exclude_mask = aho_missing_mask | target_missing_mask

    n_total = len(holdout_df)
    n_excluded = exclude_mask.sum()
    n_usable = n_total - n_excluded

    if n_excluded > 0:
        excluded_names = (
            holdout_df.loc[exclude_mask, "antibody_name"].tolist()
            if has_name_col else holdout_df.index[exclude_mask].tolist()
        )
        print(
            f"[WARN] {n_excluded}/{n_total} holdout antibodies excluded from evaluation:\n"
            f"       AHO sequence missing: {aho_missing_mask.sum()} | "
            f"Target missing: {target_missing_mask.sum()}\n"
            f"       Excluded: {excluded_names}"
        )

    print(f"[INFO] Holdout antibodies used for evaluation: {n_usable}/{n_total}")

    eval_source = holdout_df[~exclude_mask].reset_index(drop=True)

    # ── Build CDR feature matrix ───────────────────────────────────────────────
    print("[INFO] Building holdout CDR feature matrix...")
    holdout_feat = build_feature_matrix(eval_source)

    # Attach target and (optionally) antibody_name so predict_model receives
    # the full feature + label context PyCaret expects.
    eval_df = pd.concat(
        [holdout_feat.reset_index(drop=True),
         eval_source[[target_col]].reset_index(drop=True)],
        axis=1,
    )

    # Preserve antibody_name as a pass-through column (not a feature).
    # PyCaret ignores non-numeric / non-feature columns it wasn't trained on.
    if has_name_col:
        eval_df.insert(0, "antibody_name", eval_source["antibody_name"].values)

    print(f"[INFO] eval_df shape entering predict_model: {eval_df.shape}")

    # ── Load model and predict ─────────────────────────────────────────────────
    print("[INFO] Loading saved PyCaret model...")
    model = load_model(model_path)

    print("[INFO] Predicting on holdout (all eligible antibodies, no further splitting)...")
    preds_df = predict_model(model, data=eval_df.copy())

    if "prediction_label" not in preds_df.columns:
        raise ValueError(
            "PyCaret prediction output is missing 'prediction_label' column. "
            "Check that the loaded model is a regression pipeline."
        )

    y_true = preds_df[target_col]
    y_pred = preds_df["prediction_label"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)

    print("\n=== HOLDOUT RESULTS ===")
    print(f"N evaluated : {len(y_true)}")
    for metric_name, val in metrics.items():
        print(f"{metric_name}: {val:.4f}")

    # ── Build tidy output DataFrame ───────────────────────────────────────────
    # Column order: identifier | y_true | y_pred | abs_error | features …
    output_df = preds_df.copy()
    output_df["abs_error"] = (y_true.values - y_pred.values).__abs__()

    # Reorder so identifier columns come first.
    id_cols = []
    if has_name_col and "antibody_name" in output_df.columns:
        id_cols.append("antibody_name")
    id_cols += [target_col, "prediction_label", "abs_error"]
    remaining = [c for c in output_df.columns if c not in id_cols]
    output_df = output_df[id_cols + remaining]

    # ── Save ──────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = Path(str(model_path) + f"_holdout_predictions_pycaret_{timestamp}.csv")
    output_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved holdout predictions to: {out_path}")
    print(f"[INFO] Output shape: {output_df.shape}  "
          f"(one row per evaluated antibody)")


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        holdout_path=args.holdout_data,
        target_col=args.target,
    )