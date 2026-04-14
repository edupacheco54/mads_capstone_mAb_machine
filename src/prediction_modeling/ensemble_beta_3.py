"""
ensemble_beta.py

Runs cluster-aware cross-validated Titer modeling on GDPa1 by combining
engineered CDR features with protein language model embedding features,
then computes out-of-fold diagnostics, failure-overlap analysis, and a
Spearman-weighted ensemble across base learners.

Changes from ensemble_beta_2.py (original):
  - DOWNSTREAM_ASSAY_COLS dropped explicitly in do_modeling (early guard + hard assertion)
    before the merged df ever reaches do_hugging_face.
  - Debug ************ CURRENT DF .COLUMNS print replaced with a post-guard diagnostic.
  - do_hugging_face gains holdout_merged_df parameter: after CV, a final model is
    retrained on all CV antibodies and evaluated once on the withheld set.
  - do_modeling gains holdout_merged_df parameter and returns a 3-tuple
    (results_df, oof_predictions, holdout_results_df) to thread holdout results upstream.
  - prepare_holdout_merged_df() helper merges holdout CSV with CDR features and
    the per-embedding-family embedding rows.
  - driver() gains HOLDOUT_CSV_PATH variable; collects and prints holdout summary.
  - pd.concat FutureWarning suppressed by accumulating result dfs in a list.
"""

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)

from datetime import datetime
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, KFold

# Inner CV used inside GridSearchCV for tuning hyperparameters within each
# outer held-out fold.  This keeps tuning separate from final fold evaluation.
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# ── Downstream assay readouts that must NEVER enter the feature matrix ────────
# Purity / SEC %Monomer / SMAC are causally downstream of expression titer.
# HIC / HAC / AC-SINS / thermal assays are confounded by the same upstream
# manufacturing process.  Including any of them would constitute target leakage.
DOWNSTREAM_ASSAY_COLS = [
    "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
    "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
    "Tonset", "Tm1", "Tm2",
]

# Full column exclusion list: assay leakage + metadata / fold / raw sequences.
# Applied both in do_modeling (early drop) and do_hugging_face (secondary defence).
# antibody_name is intentionally absent here — it is captured for diagnostics
# and dropped separately inside do_hugging_face.
_FULL_DROP_COLS = [
    "Unnamed: 0_CDR_feature", "Unnamed: 0",
    *DOWNSTREAM_ASSAY_COLS,
    "highest_clinical_trial_asof_feb2025", "hierarchical_cluster_fold",
    "random_fold", "n_missing_CDR_feature", "est_status_asof_feb2025",
    "lc_subtype", "hc_subtype", "n_missing",
    "vh_protein_sequence", "hc_protein_sequence", "hc_dna_sequence",
    "vl_protein_sequence", "lc_protein_sequence", "lc_dna_sequence",
    "heavy_aligned_aho", "light_aligned_aho",
]


# ─────────────────────────────────────────────────────────────────────────────
def do_hugging_face(df, k, feature_set_description=None, models=None,
                    holdout_merged_df=None):
    """
    Cross-validated modeling over hierarchical cluster folds.
    Uses CDR + LLM embeddings combined (no subsetting).

    The fold column — hierarchical_cluster_IgG_isotype_stratified_fold — holds
    out entire sequence clusters so the model is tested on novel antibody
    families, preventing near-duplicate label leakage.

    If holdout_merged_df is supplied, a FINAL model is trained on ALL CV
    antibodies (after CV completes) and evaluated exactly once on the withheld
    holdout set.  The same feature columns, imputer, and scaler derived from the
    full CV dataset are applied to the holdout to prevent any preprocessing
    leakage.

    Args:
        df               : merged CV dataframe (GDPa1 + CDR features + embeddings).
                           Downstream assay columns should already be absent
                           (dropped upstream in do_modeling), but this function
                           applies _FULL_DROP_COLS as a secondary defence.
        k                : embedding family name (used for logging).
        feature_set_description: label stored in the results table.
        models           : dict { model_name -> sklearn estimator or GridSearchCV }.
        holdout_merged_df: optional merged holdout DataFrame in the same column
                           format as df (same target + same embedding / CDR cols).

    Returns:
        results_df        : per-model CV summary metrics (pd.DataFrame).
        oof_predictions   : dict { base_learner_key ->
                                   {y_pred_oof, y_true_oof, abs_error,
                                    antibody_names, fold_labels, spearman_rho} }.
        holdout_results_df: per-model holdout metrics (pd.DataFrame).
                            Empty if holdout_merged_df is None.
    """
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    target = "Titer"

    # ── Secondary leakage defence ─────────────────────────────────────────────
    # do_modeling already dropped DOWNSTREAM_ASSAY_COLS and asserted.
    # This block catches any edge-case path that bypasses do_modeling.
    df = df.drop(columns=[c for c in _FULL_DROP_COLS if c in df.columns]).copy()
    still_leaked = [c for c in DOWNSTREAM_ASSAY_COLS if c in df.columns]
    if still_leaked:
        raise ValueError(
            f"[{k}] LEAKAGE DETECTED in do_hugging_face after drop: {still_leaked}"
        )

    # Keep only rows where target exists.
    df = df[df[target].notna()].copy()

    # Keep only rows with an assigned fold.
    df = df[df[fold_col].notna()].copy()

    # Capture identifiers BEFORE dropping them for downstream diagnostics.
    antibody_names = (
        df["antibody_name"].values.copy() if "antibody_name" in df.columns else None
    )
    fold_labels = df[fold_col].values.copy()

    # Remove identifier before modeling.
    df = df.drop(columns=["antibody_name"], errors="ignore")

    # Restrict to numeric columns only; exclude target + fold indicator.
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in [target, fold_col]
    ]

    print(f"[{k}] n_features: {len(feature_cols)}")
    print(f"[{k}] CDR features: {sum('_CDR_feature' in c for c in feature_cols)}")
    print(f"[{k}] Embedding features: {sum('_CDR_feature' not in c for c in feature_cols)}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    fold_values = df[fold_col].to_numpy()

    nan_mask = np.isnan(X)
    print(
        f"[{k}] X shape: {X.shape} | NaNs: {nan_mask.sum()} "
        f"| NaN rows: {nan_mask.any(axis=1).sum()}"
    )

    assert len(X) == len(y), "X/y length mismatch after filtering"

    # ── Prepare holdout arrays using the SAME feature_cols ────────────────────
    # Alignment to feature_cols is enforced via reindex; any column present in CV
    # data but absent in holdout will become NaN and be handled by the imputer.
    X_holdout = y_holdout = holdout_names_arr = None
    if holdout_merged_df is not None:
        hld = holdout_merged_df.drop(
            columns=[c for c in _FULL_DROP_COLS if c in holdout_merged_df.columns]
        ).copy()
        hld = hld[hld[target].notna()].copy()

        # Capture antibody names BEFORE dropping the identifier column so they
        # are available for the per-antibody predictions DataFrame later.
        holdout_names_arr = (
            hld["antibody_name"].values.copy() if "antibody_name" in hld.columns else None
        )
        hld = hld.drop(columns=["antibody_name"], errors="ignore")

        missing_in_holdout = [c for c in feature_cols if c not in hld.columns]
        if missing_in_holdout:
            print(
                f"[{k}] Holdout warning: {len(missing_in_holdout)} feature cols absent "
                f"from holdout df — will be NaN-imputed."
            )

        X_holdout = hld.reindex(columns=feature_cols).to_numpy(dtype=float)
        y_holdout = hld[target].to_numpy(dtype=float)
        print(
            f"[{k}] Holdout set: N={len(y_holdout)} | "
            f"X_holdout shape={X_holdout.shape}"
        )

    unique_folds = sorted(f for f in np.unique(fold_values) if f == f)

    results_rows = []
    holdout_results_rows = []
    holdout_preds_rows = []   # accumulates one dict per antibody per model
    oof_predictions = {}

    # Default to a simple Ridge model if none were supplied.
    if models is None:
        models = {"Ridge": Ridge()}

    for model_name, model in models.items():
        # Full-length arrays to hold out-of-fold predictions for every antibody.
        y_pred_all = np.full(len(df), np.nan)
        y_true_all = np.full(len(df), np.nan)
        per_fold_stats = []

        for f in unique_folds:
            test_idx = np.where(fold_values == f)[0]
            train_idx = np.where(fold_values != f)[0]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Fit preprocessing INSIDE the fold to prevent leakage.
            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            fitted_model = model.fit(X_train, y_train)
            best_params = (
                fitted_model.best_params_ if hasattr(fitted_model, "best_params_") else {}
            )

            y_pred = np.asarray(fitted_model.predict(X_test)).reshape(-1)

            y_pred_all[test_idx] = y_pred
            y_true_all[test_idx] = y_test

            fold_rho = spearmanr(y_test, y_pred).statistic
            fold_r2 = r2_score(y_test, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            per_fold_stats.append(
                (int(f), fold_r2, fold_rmse, fold_rho, len(y_test), best_params)
            )

        # Compute overall OOF metrics using only rows that received predictions.
        mask = ~np.isnan(y_true_all)
        overall_rho = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic
        overall_r2 = r2_score(y_true_all[mask], y_pred_all[mask])
        overall_rmse = np.sqrt(mean_squared_error(y_true_all[mask], y_pred_all[mask]))
        all_best_params = [p for *_, p in per_fold_stats]

        print(f"\n[{k}] {model_name}")
        print("Fold\tN\tR2\tRMSE\tSpearman_rho")
        for f, r2, rmse, rho, n, bp in per_fold_stats:
            print(f"{f}\t{n}\t{r2:.4f}\t{rmse:.4f}\t{rho:.4f}")
        print(f"Overall\t{mask.sum()}\t{overall_r2:.4f}\t{overall_rmse:.4f}\t{overall_rho:.4f}")

        results_rows.append({
            "feature_set_description": feature_set_description or k,
            "model_type": model_name,
            "LLM_name": k,
            "r_Squared": overall_r2,
            "RMSE": overall_rmse,
            "Spearman_rho": overall_rho,
            "best_params_per_fold": str(all_best_params),
        })

        base_learner_key = f"{feature_set_description or k}__{model_name}"
        abs_error = np.abs(y_true_all - y_pred_all)

        oof_predictions[base_learner_key] = {
            "y_pred_oof": y_pred_all.copy(),
            "y_true_oof": y_true_all.copy(),
            "abs_error": abs_error,
            "antibody_names": antibody_names,
            "fold_labels": fold_labels,
            "spearman_rho": overall_rho,
        }

        # ── Holdout evaluation ────────────────────────────────────────────────
        # Retrain on ALL CV antibodies (no fold withheld), apply to holdout once.
        # Preprocessing is fitted on the full CV dataset so the imputer and scaler
        # statistics are as stable as possible.
        if X_holdout is not None and y_holdout is not None:
            imputer_final = SimpleImputer(strategy="mean")
            X_all_imp = imputer_final.fit_transform(X)        # fit on all CV data
            X_hld_imp = imputer_final.transform(X_holdout)    # transform holdout

            scaler_final = StandardScaler()
            X_all_sc = scaler_final.fit_transform(X_all_imp)
            X_hld_sc = scaler_final.transform(X_hld_imp)

            final_model = model.fit(X_all_sc, y)              # train on all CV data
            y_hld_pred = np.asarray(final_model.predict(X_hld_sc)).reshape(-1)

            # Build per-antibody predictions DataFrame for this model.
            # holdout_names_arr was captured before antibody_name was dropped.
            names_col = (
                holdout_names_arr
                if holdout_names_arr is not None
                else np.arange(len(y_holdout))
            )
            for ab_name, y_true_val, y_pred_val in zip(names_col, y_holdout, y_hld_pred):
                holdout_preds_rows.append({
                    "antibody_name": ab_name,
                    "y_true": y_true_val,
                    "y_pred": y_pred_val,
                    "abs_error": abs(y_true_val - y_pred_val),
                    "LLM_name": k,
                    "model_type": model_name,
                    "feature_set_description": feature_set_description or k,
                })

            hld_rho = spearmanr(y_holdout, y_hld_pred).statistic
            hld_r2 = r2_score(y_holdout, y_hld_pred)
            hld_rmse = np.sqrt(mean_squared_error(y_holdout, y_hld_pred))

            print(
                f"[{k}] {model_name} HOLDOUT | N={len(y_holdout)} "
                f"| R2={hld_r2:.4f} | RMSE={hld_rmse:.4f} | Spearman_rho={hld_rho:.4f}"
            )

            holdout_results_rows.append({
                "feature_set_description": feature_set_description or k,
                "model_type": model_name,
                "LLM_name": k,
                "split": "holdout",
                "n_holdout": len(y_holdout),
                "r_Squared": hld_r2,
                "RMSE": hld_rmse,
                "Spearman_rho": hld_rho,
            })

    return (
        pd.DataFrame(results_rows),
        oof_predictions,
        pd.DataFrame(holdout_results_rows),
        pd.DataFrame(holdout_preds_rows),
    )


# ─────────────────────────────────────────────────────────────────────────────
def do_modeling(v, k, feature_set_description=None, models=None,
                holdout_merged_df=None):
    """
    Load GDPa1, merge CDR features by antibody_name (safe join), merge LLM
    embeddings, drop all leakage columns, then run cross-validated modeling.
    CDR + LLM features are always used together.

    Args:
        v                : embedding DataFrame (from pkl).
        k                : embedding family name.
        feature_set_description: label for results table.
        models           : dict of sklearn estimators / GridSearchCV.
        holdout_merged_df: optional merged holdout DataFrame (same structure
                           as the CV merged df).  Passed through to
                           do_hugging_face for final holdout evaluation.

    Returns:
        results_df        : pd.DataFrame
        oof_predictions   : dict
        holdout_results_df: pd.DataFrame (empty if no holdout)
    """
    print("---------------------------------------------------------")
    og_df = pd.read_csv("../../data/raw/GDPa1_246 IgGs_cleaned.csv")

    # Load precomputed CDR features.
    cdr_features = pd.read_csv("../../data/modeling/cdr_features_titer.csv")

    # Rename so CDR columns are distinguishable from embedding columns and
    # can be reliably identified / dropped by suffix.
    cdr_feature_cols = [c for c in cdr_features.columns if c != "antibody_name"]
    cdr_features = cdr_features.rename(
        columns={c: f"{c}_CDR_feature" for c in cdr_feature_cols}
    )

    before_n = len(og_df)
    og_df = og_df.merge(
        cdr_features, on="antibody_name", how="left", validate="one_to_one"
    )
    if len(og_df) != before_n:
        raise ValueError(
            f"Row count changed after CDR merge ({before_n} -> {len(og_df)})"
        )

    added_cdr_cols = [c for c in og_df.columns if c.endswith("_CDR_feature")]
    missing_cdr_rows = og_df[added_cdr_cols].isna().all(axis=1).sum()
    print(
        f"[{k}] CDR merge: {len(added_cdr_cols)} cols added | "
        f"rows with all CDR missing = {missing_cdr_rows}"
    )

    # Drop embedding columns that would duplicate existing columns in GDPa1 + CDR df.
    join_key = "antibody_name"
    cols_drop = [c for c in v.columns if c in og_df.columns and c != join_key]
    v_clean = v.drop(columns=cols_drop)

    # Inner merge: only antibodies present in both GDPa1 and the embedding df.
    merged = og_df.merge(v_clean, on=join_key, how="inner")
    print(
        f"[{k}] og_df: {og_df.shape} | v_clean: {v_clean.shape} | "
        f"merged (pre-leakage-guard): {merged.shape}"
    )

    # ── EARLY LEAKAGE GUARD ───────────────────────────────────────────────────
    # Drop all downstream assay readouts from the merged df HERE, before any
    # data reaches do_hugging_face or any modeling code.  This makes the leakage
    # prevention explicit and auditable at the pipeline boundary.
    #
    # These columns are causally downstream of expression titer or confounded by
    # the same upstream bioprocess:
    #   - Purity / SEC %Monomer / SMAC : downstream of expression yield
    #   - HIC / HAC / AC-SINS          : hydrophobic / charge surface assays,
    #                                    correlated with expression but not causal
    #   - Tonset / Tm1 / Tm2           : thermal stability, process-confounded
    merged = merged.drop(
        columns=[c for c in DOWNSTREAM_ASSAY_COLS if c in merged.columns]
    ).copy()

    # Hard assertion: if any assay column survived, fail loudly.
    leaked = [c for c in DOWNSTREAM_ASSAY_COLS if c in merged.columns]
    assert not leaked, (
        f"[{k}] LEAKAGE COLUMNS STILL PRESENT after early drop: {leaked}"
    )

    # Diagnostic: report numeric column count after the guard so it is clear
    # what enters modeling (Titer + fold indicator + CDR features + embeddings only).
    numeric_cols_after_guard = merged.select_dtypes(include=[np.number]).columns.tolist()
    print(
        f"[{k}] After leakage guard: {len(numeric_cols_after_guard)} numeric cols remain "
        f"(target + fold + CDR features + embeddings only)"
    )

    return do_hugging_face(
        merged,
        k,
        feature_set_description=feature_set_description,
        models=models,
        holdout_merged_df=holdout_merged_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
def prepare_holdout_merged_df(holdout_csv_path: str,
                               embedding_df: pd.DataFrame,
                               cdr_csv_path: str) -> pd.DataFrame:
    """
    Load a holdout CSV, attach precomputed CDR features and the corresponding
    embedding rows, and return a merged DataFrame in the same format expected
    by do_hugging_face.

    The holdout CSV must contain at minimum:
        antibody_name (str)   — must match rows present in embedding_df.
        Titer (float)         — the evaluation target for the holdout set.

    CDR features are sourced from cdr_csv_path (the same file used for CV).
    If a holdout antibody is absent from cdr_csv_path, its CDR feature columns
    will be NaN and handled by the imputer inside do_hugging_face.

    Antibodies in the holdout CSV that are absent from embedding_df are silently
    dropped (inner join).  The caller should verify the resulting N.

    Args:
        holdout_csv_path : path to the holdout CSV file.
        embedding_df     : PLM embedding DataFrame for this embedding family.
        cdr_csv_path     : path to the precomputed CDR features CSV.

    Returns:
        pd.DataFrame with columns matching the CV merged DataFrame format.
    """
    holdout_raw = pd.read_csv(holdout_csv_path)
    print(f"[holdout] Loaded '{holdout_csv_path}': {holdout_raw.shape}")

    # Attach CDR features (same suffix convention as in do_modeling).
    cdr_features = pd.read_csv(cdr_csv_path)
    cdr_feature_cols = [c for c in cdr_features.columns if c != "antibody_name"]
    cdr_features = cdr_features.rename(
        columns={c: f"{c}_CDR_feature" for c in cdr_feature_cols}
    )
    holdout_raw = holdout_raw.merge(cdr_features, on="antibody_name", how="left")

    # Attach embeddings — inner join keeps only holdout antibodies that have
    # embeddings in this embedding family.
    join_key = "antibody_name"
    cols_drop = [
        c for c in embedding_df.columns if c in holdout_raw.columns and c != join_key
    ]
    emb_clean = embedding_df.drop(columns=cols_drop)
    holdout_merged = holdout_raw.merge(emb_clean, on=join_key, how="inner")

    n_dropped = len(holdout_raw) - len(holdout_merged)
    print(
        f"[holdout] After embedding merge: {holdout_merged.shape} "
        f"({n_dropped} antibodies dropped — no embedding found)"
    )
    if len(holdout_merged) == 0:
        raise ValueError(
            "[holdout] No holdout antibodies matched embedding_df. "
            "Check that antibody_name values align between holdout CSV and embedding pickle."
        )

    return holdout_merged


# ─────────────────────────────────────────────────────────────────────────────
def load_pickles_to_df_dict(folder_path: str) -> dict:
    """
    Load all .pkl embedding DataFrames from folder.
    CDR features are NOT appended here — do_modeling handles the safe merge.
    """
    folder = Path(folder_path)
    out = {}

    for p in folder.glob("*.pkl"):
        try:
            df = pd.read_pickle(p)
            if isinstance(df, pd.DataFrame):
                key = p.stem.replace("_df", "")
                out[key] = df
                print(f"Loaded {p.name} -> key='{key}' shape={df.shape}")
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
def compute_weighted_ensemble(all_oof: dict) -> dict:
    """
    Weighted average ensemble using each base learner's overall Spearman ρ as
    weight (clipped to 0 so negatively-correlated learners contribute nothing).

    Args:
        all_oof : {base_learner_key -> {y_pred_oof, y_true_oof, spearman_rho}}

    Returns:
        dict with ensemble metrics, weight table, and prediction vector.
    """
    first_key = next(iter(all_oof))
    y_true_ref = all_oof[first_key]["y_true_oof"]

    for key, val in all_oof.items():
        if not np.allclose(val["y_true_oof"], y_true_ref, equal_nan=True):
            raise ValueError(
                f"y_true mismatch for '{key}' vs '{first_key}'. "
                "All base learners must use identical fold assignments."
            )

    raw_weights = {k: max(v["spearman_rho"], 0.0) for k, v in all_oof.items()}
    total_w = sum(raw_weights.values())

    if total_w == 0:
        raise ValueError(
            "All base learner Spearman ρ values are ≤ 0. "
            "Cannot form a meaningful weighted ensemble."
        )

    norm_weights = {k: w / total_w for k, w in raw_weights.items()}

    ensemble_pred = np.zeros(len(y_true_ref))
    for key, w in norm_weights.items():
        ensemble_pred += w * all_oof[key]["y_pred_oof"]

    mask = ~np.isnan(y_true_ref)
    ens_rho = spearmanr(y_true_ref[mask], ensemble_pred[mask]).statistic
    ens_r2 = r2_score(y_true_ref[mask], ensemble_pred[mask])
    ens_rmse = np.sqrt(mean_squared_error(y_true_ref[mask], ensemble_pred[mask]))

    weight_table = pd.DataFrame([
        {
            "base_learner": k,
            "spearman_rho": all_oof[k]["spearman_rho"],
            "weight": norm_weights[k],
        }
        for k in all_oof
    ]).sort_values("weight", ascending=False).reset_index(drop=True)

    return {
        "ensemble_spearman_rho": ens_rho,
        "ensemble_r2": ens_r2,
        "ensemble_rmse": ens_rmse,
        "weight_table": weight_table,
        "y_pred_ensemble": ensemble_pred,
        "y_true": y_true_ref,
    }


# ─────────────────────────────────────────────────────────────────────────────
def compute_failure_overlap_diagnostics(all_oof: dict,
                                        high_error_pct: float = 0.25) -> dict:
    """
    Examine whether base learners fail on the same antibodies.

    For each antibody:
      - Records absolute error from every base learner
      - Flags as high error if error exceeds the top high_error_pct quantile
        for that base learner (learner-specific threshold, so comparisons are fair)
      - Counts how many base learners flag it as high error

    Also computes a pairwise Spearman correlation matrix of absolute error vectors.
    High off-diagonal values (>0.6) confirm correlated failure modes.

    Args:
        all_oof       : dict from do_hugging_face
        high_error_pct: top fraction to define high error per learner (default 25%)

    Returns:
        dict with:
          per_antibody_df    : antibody-level error table (sorted by n_models_high_err)
          error_corr_matrix  : pairwise Spearman rho of abs_error vectors (pd.DataFrame)
          consistently_wrong : antibodies flagged high-error by ALL base learners
          consistently_right : antibodies flagged high-error by NO base learner
    """
    first_key = next(iter(all_oof))
    antibody_names = all_oof[first_key]["antibody_names"]
    fold_labels = all_oof[first_key]["fold_labels"]
    y_true = all_oof[first_key]["y_true_oof"]

    records = pd.DataFrame({
        "antibody_name": antibody_names,
        "y_true": y_true,
        "fold": fold_labels,
    })

    error_vectors = {}
    high_err_flags = {}

    def short_label(k):
        return k.replace("CDR_and_", "").replace("__Lasso", "")

    for key, val in all_oof.items():
        lbl = short_label(key)
        abs_err = val["abs_error"]
        threshold = np.nanquantile(abs_err, 1 - high_error_pct)

        records[f"{lbl}_abserr"] = abs_err
        records[f"{lbl}_pred"] = val["y_pred_oof"]
        error_vectors[lbl] = abs_err
        high_err_flags[lbl] = (abs_err >= threshold).astype(float)
        records[f"{lbl}_high_err"] = high_err_flags[lbl]

    flag_cols = [c for c in records.columns if c.endswith("_high_err")]
    records["n_models_high_err"] = records[flag_cols].sum(axis=1)
    records = records.sort_values(
        "n_models_high_err", ascending=False
    ).reset_index(drop=True)

    learner_labels = list(error_vectors.keys())
    n_learners = len(learner_labels)
    corr_matrix = np.full((n_learners, n_learners), np.nan)

    for i, l1 in enumerate(learner_labels):
        for j, l2 in enumerate(learner_labels):
            e1, e2 = error_vectors[l1], error_vectors[l2]
            mask = ~(np.isnan(e1) | np.isnan(e2))
            if mask.sum() > 5:
                corr_matrix[i, j] = spearmanr(e1[mask], e2[mask]).statistic

    error_corr_df = pd.DataFrame(
        corr_matrix, index=learner_labels, columns=learner_labels
    )

    n_learners_total = len(flag_cols)
    consistently_wrong = records[
        records["n_models_high_err"] == n_learners_total
    ]["antibody_name"].tolist()
    consistently_right = records[
        records["n_models_high_err"] == 0
    ]["antibody_name"].tolist()

    print("\n" + "=" * 55)
    print("FAILURE OVERLAP DIAGNOSTICS")
    print("=" * 55)
    print(f"\nHigh-error threshold: top {int(high_error_pct * 100)}% abs error per learner")
    print(
        f"Antibodies flagged high-error by ALL {n_learners_total} learners : "
        f"{len(consistently_wrong)}"
    )
    print(
        f"Antibodies flagged high-error by NO learner              : "
        f"{len(consistently_right)}"
    )

    print("\nPairwise Spearman rho of absolute error vectors:")
    print(error_corr_df.round(3).to_string())

    display_cols = (
        ["antibody_name", "y_true", "fold", "n_models_high_err"]
        + [c for c in records.columns if c.endswith("_abserr")]
    )

    print("\nTop 20 most-failed antibodies:")
    print(records[display_cols].head(20).to_string(index=False))

    if consistently_wrong:
        print(f"\nConsistently wrong ({len(consistently_wrong)} antibodies):")
        print(
            records[records["antibody_name"].isin(consistently_wrong)][
                display_cols
            ].to_string(index=False)
        )

    return {
        "per_antibody_df": records,
        "error_corr_matrix": error_corr_df,
        "consistently_wrong": consistently_wrong,
        "consistently_right": consistently_right,
    }


# ─────────────────────────────────────────────────────────────────────────────
def driver():
    """
    Orchestrate pipeline:
      1. Load PLM embedding pickles
      2. Run CDR + LLM combined CV modeling for each LLM -> collect OOF vectors
      3. Optionally evaluate a withheld holdout set (retrain on all CV data, evaluate once)
      4. Compute Spearman-weighted ensemble over all base learners
      5. Print and save results
    """
    # ── Data paths ─────────────────────────────────────────────────────────────
    PICKLE_DIR = "../../data/modeling"
    CDR_CSV_PATH = "../../data/modeling/cdr_features_titer.csv"

    # Set HOLDOUT_CSV_PATH to the path of your holdout CSV to enable holdout
    # evaluation.  The CSV must contain at minimum:
    #   antibody_name (str) — must match antibody_name values in the embedding pickles
    #   Titer (float)       — held-out evaluation target
    # Set to None to skip holdout evaluation entirely.
    HOLDOUT_CSV_PATH = None  # e.g. "../../data/raw/holdout_set.csv"

    df_dict = load_pickles_to_df_dict(PICKLE_DIR)

    models = {
        "Lasso": GridSearchCV(
            Lasso(random_state=42),
            {"alpha": [10, 12, 15]},
            cv=inner_cv,
            scoring="r2",
        ),
    }

    # Accumulate results in lists to avoid the pd.concat FutureWarning that
    # arises from concatenating into a pre-allocated empty DataFrame.
    cv_results_list = []
    holdout_results_list = []
    holdout_preds_list = []   # per-antibody predictions across all embedding families
    all_oof = {}

    for k, v in df_dict.items():
        # Build holdout merged df for this embedding family (if path is set).
        holdout_merged = None
        if HOLDOUT_CSV_PATH is not None:
            holdout_merged = prepare_holdout_merged_df(
                HOLDOUT_CSV_PATH, v, CDR_CSV_PATH
            )

        run_results, oof_preds, holdout_results, holdout_preds = do_modeling(
            v,
            k,
            feature_set_description=f"CDR_and_{k}",
            models=models,
            holdout_merged_df=holdout_merged,
        )

        cv_results_list.append(run_results)
        all_oof.update(oof_preds)
        if not holdout_results.empty:
            holdout_results_list.append(holdout_results)
        if not holdout_preds.empty:
            holdout_preds_list.append(holdout_preds)

    # Run diagnostics before ensembling.
    diag_result = compute_failure_overlap_diagnostics(all_oof, high_error_pct=0.25)

    print("\n" + "=" * 55)
    print("WEIGHTED ENSEMBLE (Spearman rho-weighted average)")
    print("=" * 55)

    ensemble_result = compute_weighted_ensemble(all_oof)

    print("\nBase learner weights:")
    print(ensemble_result["weight_table"].to_string(index=False))
    print(f"\nEnsemble Spearman rho : {ensemble_result['ensemble_spearman_rho']:.4f}")
    print(f"Ensemble R2           : {ensemble_result['ensemble_r2']:.4f}")
    print(f"Ensemble RMSE         : {ensemble_result['ensemble_rmse']:.4f}")

    # ── Holdout summary ────────────────────────────────────────────────────────
    if holdout_results_list:
        holdout_summary = pd.concat(holdout_results_list, ignore_index=True).sort_values(
            "Spearman_rho", ascending=False
        )
        print("\n" + "=" * 55)
        print("HOLDOUT EVALUATION SUMMARY")
        print("=" * 55)
        print(holdout_summary.to_string(index=False))
    else:
        holdout_summary = pd.DataFrame()
        print(
            "\n[INFO] Holdout evaluation skipped "
            "(set HOLDOUT_CSV_PATH in driver() to enable)."
        )

    # Sort and compile CV results.
    final_results = pd.concat(cv_results_list, ignore_index=True).sort_values(
        by=["feature_set_description", "RMSE", "Spearman_rho"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Save result artifacts — timestamps prevent runs from overwriting each other.
    final_results.to_csv(f"../../data/modeling/final_results_{timestamp}.csv", index=False)
    ensemble_result["weight_table"].to_csv(f"../../data/modeling/ensemble_weights_{timestamp}.csv", index=False)
    diag_result["per_antibody_df"].to_csv(f"../../data/modeling/failure_overlap_{timestamp}.csv", index=False)
    diag_result["error_corr_matrix"].to_csv(f"../../data/modeling/error_correlation_{timestamp}.csv")

    if not holdout_summary.empty:
        holdout_summary.to_csv(f"../../data/modeling/holdout_results_{timestamp}.csv", index=False)
        print(f"Saved: holdout_results_{timestamp}.csv  (aggregate metrics per model)")

    if holdout_preds_list:
        holdout_preds_all = pd.concat(holdout_preds_list, ignore_index=True).sort_values(
            ["feature_set_description", "antibody_name"]
        ).reset_index(drop=True)
        holdout_preds_all.to_csv(
            f"../../data/modeling/holdout_predictions_{timestamp}.csv", index=False
        )
        print(f"Saved: holdout_predictions_{timestamp}.csv  (one row per antibody per model)")

    print(f"Saved: final_results_{timestamp}.csv")
    print(f"Saved: ensemble_weights_{timestamp}.csv")
    print(f"Saved: failure_overlap_{timestamp}.csv")
    print(f"Saved: error_correlation_{timestamp}.csv")

    return "Complete"


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("hello world")
    print(driver())