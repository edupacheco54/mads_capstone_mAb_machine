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

inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)


def do_hugging_face(df, k, feature_set_description=None, models=None):
    """
    Cross-validated modeling over hierarchical cluster folds.
    Uses CDR + LLM embeddings combined (no subsetting).

    The fold column — hierarchical_cluster_IgG_isotype_stratified_fold — holds out
    entire sequence clusters so the model is tested on novel antibody families,
    preventing near-duplicate label leakage across train/test splits.

    Returns:
        results_df      : per-model summary metrics (pd.DataFrame)
        oof_predictions : dict { base_learner_key -> {y_pred_oof, y_true_oof, spearman_rho} }
    """
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    target   = "Titer"

    drop_cols = [
        "Unnamed: 0_CDR_feature", "Unnamed: 0",
        # downstream assay readouts — causal descendants of Titer, must be excluded
        "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
        "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
        "Tonset", "Tm1", "Tm2",
        # fold / metadata columns
        "highest_clinical_trial_asof_feb2025", "hierarchical_cluster_fold",
        "random_fold", "n_missing_CDR_feature", "est_status_asof_feb2025",
        "lc_subtype", "hc_subtype", "n_missing",
        # raw sequence / identifier columns — captured below before dropping
        "vh_protein_sequence", "hc_protein_sequence", "hc_dna_sequence",
        "vl_protein_sequence", "lc_protein_sequence", "lc_dna_sequence",
        "heavy_aligned_aho", "light_aligned_aho",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    df = df[df[target].notna()].copy()
    df = df[df[fold_col].notna()].copy()

    # Capture antibody_name and fold BEFORE dropping — needed for error diagnostics
    antibody_names = df["antibody_name"].values.copy() if "antibody_name" in df.columns else None
    fold_labels    = df[fold_col].values.copy()

    df = df.drop(columns=["antibody_name"], errors="ignore")

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in [target, fold_col]
    ]

    print(f"[{k}] n_features: {len(feature_cols)}")
    print(f"[{k}] CDR features: {sum('_CDR_feature' in c for c in feature_cols)}")
    print(f"[{k}] Embedding features: {sum('_CDR_feature' not in c for c in feature_cols)}")

    X           = df[feature_cols].to_numpy(dtype=float)
    y           = df[target].to_numpy(dtype=float)
    fold_values = df[fold_col].to_numpy()
    nan_mask    = np.isnan(X)

    print(f"[{k}] X shape: {X.shape} | NaNs: {nan_mask.sum()} "
          f"| NaN rows: {nan_mask.any(axis=1).sum()}")

    assert len(X) == len(y), "X/y length mismatch after filtering"

    unique_folds = sorted(f for f in np.unique(fold_values) if f == f)  # exclude NaN
    results_rows = []
    oof_predictions = {}

    if models is None:
        models = {"Ridge": Ridge()}

    for model_name, model in models.items():
        y_pred_all = np.full(len(df), np.nan)
        y_true_all = np.full(len(df), np.nan)
        per_fold_stats = []

        for f in unique_folds:
            test_idx  = np.where(fold_values == f)[0]
            train_idx = np.where(fold_values != f)[0]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test,  y_test  = X[test_idx],  y[test_idx]

            # Fit preprocessing inside the fold — no leakage
            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test  = imputer.transform(X_test)

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            fitted_model = model.fit(X_train, y_train)
            best_params  = fitted_model.best_params_ if hasattr(fitted_model, "best_params_") else {}

            y_pred = np.asarray(fitted_model.predict(X_test)).reshape(-1)

            y_pred_all[test_idx] = y_pred
            y_true_all[test_idx] = y_test

            fold_rho  = spearmanr(y_test, y_pred).statistic
            fold_r2   = r2_score(y_test, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            per_fold_stats.append((int(f), fold_r2, fold_rmse, fold_rho, len(y_test), best_params))

        mask         = ~np.isnan(y_true_all)
        overall_rho  = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic
        overall_r2   = r2_score(y_true_all[mask], y_pred_all[mask])
        overall_rmse = np.sqrt(mean_squared_error(y_true_all[mask], y_pred_all[mask]))
        all_best_params = [p for *_, p in per_fold_stats]

        print(f"\n[{k}] {model_name}")
        print("Fold\tN\tR2\tRMSE\tSpearman_rho")
        for f, r2, rmse, rho, n, bp in per_fold_stats:
            print(f"{f}\t{n}\t{r2:.4f}\t{rmse:.4f}\t{rho:.4f}")
        print(f"Overall\t{mask.sum()}\t{overall_r2:.4f}\t{overall_rmse:.4f}\t{overall_rho:.4f}")

        results_rows.append({
            "feature_set_description": feature_set_description or k,
            "model_type":              model_name,
            "LLM_name":                k,
            "r_Squared":               overall_r2,
            "RMSE":                    overall_rmse,
            "Spearman_rho":            overall_rho,
            "best_params_per_fold":    str(all_best_params),
        })

        # ── Save OOF vector for ensemble + diagnostics ───────────────────────
        base_learner_key = f"{feature_set_description or k}__{model_name}"
        abs_error = np.abs(y_true_all - y_pred_all)   # NaN where not predicted

        oof_predictions[base_learner_key] = {
            "y_pred_oof":     y_pred_all.copy(),
            "y_true_oof":     y_true_all.copy(),
            "abs_error":      abs_error,
            "antibody_names": antibody_names,           # maps row index -> name
            "fold_labels":    fold_labels,
            "spearman_rho":   overall_rho,
        }

    return pd.DataFrame(results_rows), oof_predictions


def do_modeling(v, k, feature_set_description=None, models=None):
    """
    Load GDPa1, merge CDR features by antibody_name (safe join), merge LLM
    embeddings, then run cross-validated modeling. CDR + LLM features are
    always used together.

    Returns:
        results_df      : pd.DataFrame
        oof_predictions : dict
    """
    print("---------------------------------------------------------")
    og_df = pd.read_csv("../../data/raw/GDPa1_246 IgGs_cleaned.csv")

    # ── CDR features: safe merge on antibody_name ────────────────────────────
    cdr_features     = pd.read_csv("../../data/modeling/cdr_features_titer.csv")
    cdr_feature_cols = [c for c in cdr_features.columns if c != "antibody_name"]
    cdr_features     = cdr_features.rename(
        columns={c: f"{c}_CDR_feature" for c in cdr_feature_cols}
    )

    before_n = len(og_df)
    og_df    = og_df.merge(cdr_features, on="antibody_name", how="left", validate="one_to_one")
    if len(og_df) != before_n:
        raise ValueError(f"Row count changed after CDR merge ({before_n} -> {len(og_df)})")

    added_cdr_cols   = [c for c in og_df.columns if c.endswith("_CDR_feature")]
    missing_cdr_rows = og_df[added_cdr_cols].isna().all(axis=1).sum()
    print(f"[{k}] CDR merge: {len(added_cdr_cols)} cols added | "
          f"rows with all CDR missing = {missing_cdr_rows}")

    # ── LLM embeddings: drop columns already in og_df before merging ────────
    join_key  = "antibody_name"
    cols_drop = [c for c in v.columns if c in og_df.columns and c != join_key]
    v_clean   = v.drop(columns=cols_drop)

    merged = og_df.merge(v_clean, on=join_key, how="inner")
    print(f"[{k}] og_df: {og_df.shape} | v_clean: {v_clean.shape} | merged: {merged.shape}")

    return do_hugging_face(
        merged, k,
        feature_set_description=feature_set_description,
        models=models,
    )


def load_pickles_to_df_dict(folder_path: str) -> dict:
    """
    Load all .pkl embedding DataFrames from folder.
    CDR features are NOT appended here — do_modeling handles the safe merge.
    """
    folder = Path(folder_path)
    out    = {}
    for p in folder.glob("*.pkl"):
        try:
            df = pd.read_pickle(p)
            if isinstance(df, pd.DataFrame):
                key      = p.stem.replace("_df", "")
                out[key] = df
                print(f"Loaded {p.name} -> key='{key}' shape={df.shape}")
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
    return out


def compute_weighted_ensemble(all_oof: dict) -> dict:
    """
    Weighted average ensemble using each base learner's overall Spearman ρ as
    weight (clipped to 0 so negatively-correlated learners contribute nothing).

    The fold structure guarantees that every antibody appears in exactly one
    test fold, so y_true_oof vectors are identical across all base learners
    and can be directly compared.

    Args:
        all_oof : {base_learner_key -> {y_pred_oof, y_true_oof, spearman_rho}}

    Returns:
        dict with ensemble metrics, weight table, and prediction vector
    """
    # ── Verify fold alignment: all base learners must share the same y_true ─
    first_key  = next(iter(all_oof))
    y_true_ref = all_oof[first_key]["y_true_oof"]

    for key, val in all_oof.items():
        if not np.allclose(val["y_true_oof"], y_true_ref, equal_nan=True):
            raise ValueError(
                f"y_true mismatch for '{key}' vs '{first_key}'. "
                "All base learners must use identical fold assignments."
            )

    # ── Weights: clip negatives, normalize to sum = 1 ───────────────────────
    raw_weights = {k: max(v["spearman_rho"], 0.0) for k, v in all_oof.items()}
    total_w     = sum(raw_weights.values())

    if total_w == 0:
        raise ValueError(
            "All base learner Spearman ρ values are ≤ 0. "
            "Cannot form a meaningful weighted ensemble."
        )

    norm_weights = {k: w / total_w for k, w in raw_weights.items()}

    # ── Weighted sum of OOF prediction vectors ───────────────────────────────
    ensemble_pred = np.zeros(len(y_true_ref))
    for key, w in norm_weights.items():
        ensemble_pred += w * all_oof[key]["y_pred_oof"]

    mask         = ~np.isnan(y_true_ref)
    ens_rho      = spearmanr(y_true_ref[mask], ensemble_pred[mask]).statistic
    ens_r2       = r2_score(y_true_ref[mask], ensemble_pred[mask])
    ens_rmse     = np.sqrt(mean_squared_error(y_true_ref[mask], ensemble_pred[mask]))

    weight_table = pd.DataFrame([
        {
            "base_learner":  k,
            "spearman_rho":  all_oof[k]["spearman_rho"],
            "weight":        norm_weights[k],
        }
        for k in all_oof
    ]).sort_values("weight", ascending=False).reset_index(drop=True)

    return {
        "ensemble_spearman_rho": ens_rho,
        "ensemble_r2":           ens_r2,
        "ensemble_rmse":         ens_rmse,
        "weight_table":          weight_table,
        "y_pred_ensemble":       ensemble_pred,
        "y_true":                y_true_ref,
    }




def compute_failure_overlap_diagnostics(all_oof: dict, high_error_pct: float = 0.25) -> dict:
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
    first_key      = next(iter(all_oof))
    antibody_names = all_oof[first_key]["antibody_names"]
    fold_labels    = all_oof[first_key]["fold_labels"]
    y_true         = all_oof[first_key]["y_true_oof"]

    records = pd.DataFrame({
        "antibody_name": antibody_names,
        "y_true":        y_true,
        "fold":          fold_labels,
    })

    error_vectors  = {}
    high_err_flags = {}

    def short_label(k):
        return k.replace("CDR_and_", "").replace("__Lasso", "")

    for key, val in all_oof.items():
        lbl       = short_label(key)
        abs_err   = val["abs_error"]
        threshold = np.nanquantile(abs_err, 1 - high_error_pct)

        records[f"{lbl}_abserr"]   = abs_err
        records[f"{lbl}_pred"]     = val["y_pred_oof"]
        error_vectors[lbl]         = abs_err
        high_err_flags[lbl]        = (abs_err >= threshold).astype(float)
        records[f"{lbl}_high_err"] = high_err_flags[lbl]

    flag_cols = [c for c in records.columns if c.endswith("_high_err")]
    records["n_models_high_err"] = records[flag_cols].sum(axis=1)
    records = records.sort_values("n_models_high_err", ascending=False).reset_index(drop=True)

    # Pairwise Spearman correlation of absolute error vectors
    learner_labels = list(error_vectors.keys())
    n_learners     = len(learner_labels)
    corr_matrix    = np.full((n_learners, n_learners), np.nan)

    for i, l1 in enumerate(learner_labels):
        for j, l2 in enumerate(learner_labels):
            e1, e2 = error_vectors[l1], error_vectors[l2]
            mask   = ~(np.isnan(e1) | np.isnan(e2))
            if mask.sum() > 5:
                corr_matrix[i, j] = spearmanr(e1[mask], e2[mask]).statistic

    error_corr_df = pd.DataFrame(corr_matrix, index=learner_labels, columns=learner_labels)

    n_learners_total   = len(flag_cols)
    consistently_wrong = records[records["n_models_high_err"] == n_learners_total]["antibody_name"].tolist()
    consistently_right = records[records["n_models_high_err"] == 0]["antibody_name"].tolist()

    print("\n" + "=" * 55)
    print("FAILURE OVERLAP DIAGNOSTICS")
    print("=" * 55)
    print(f"\nHigh-error threshold: top {int(high_error_pct*100)}% abs error per learner")
    print(f"Antibodies flagged high-error by ALL {n_learners_total} learners : {len(consistently_wrong)}")
    print(f"Antibodies flagged high-error by NO learner              : {len(consistently_right)}")

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
        print(records[records["antibody_name"].isin(consistently_wrong)][display_cols].to_string(index=False))

    return {
        "per_antibody_df":    records,
        "error_corr_matrix":  error_corr_df,
        "consistently_wrong": consistently_wrong,
        "consistently_right": consistently_right,
    }

def driver():
    """
    Orchestrate pipeline:
      1. Load PLM embedding pickles
      2. Run CDR + LLM combined CV modeling for each LLM -> collect OOF vectors
      3. Compute Spearman-weighted ensemble over all base learners
      4. Print and save results
    """
    df_dict = load_pickles_to_df_dict("../../data/modeling")

    final_results = pd.DataFrame(columns=[
        "feature_set_description", "model_type", "LLM_name",
        "r_Squared", "RMSE", "Spearman_rho", "best_params_per_fold",
    ])

    models = {
        "Lasso": GridSearchCV(
            Lasso(random_state=42),
            {"alpha": [10, 12, 15]},
            cv=inner_cv,
            scoring="r2",
        ),
    }

    all_oof = {}  # accumulates OOF dicts from every base learner across all LLMs

    for k, v in df_dict.items():
        run_results, oof_preds = do_modeling(
            v, k,
            feature_set_description=f"CDR_and_{k}",
            models=models,
        )
        final_results = pd.concat([final_results, run_results], ignore_index=True)
        all_oof.update(oof_preds)

    # ── Failure overlap diagnostics (run BEFORE ensemble) ───────────────────
    diag_result = compute_failure_overlap_diagnostics(all_oof, high_error_pct=0.25)

    # ── Weighted ensemble ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("WEIGHTED ENSEMBLE (Spearman rho-weighted average)")
    print("=" * 55)

    ensemble_result = compute_weighted_ensemble(all_oof)

    print("\nBase learner weights:")
    print(ensemble_result["weight_table"].to_string(index=False))
    print(f"\nEnsemble Spearman rho : {ensemble_result['ensemble_spearman_rho']:.4f}")
    print(f"Ensemble R2           : {ensemble_result['ensemble_r2']:.4f}")
    print(f"Ensemble RMSE         : {ensemble_result['ensemble_rmse']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    final_results = final_results.sort_values(
        by=["feature_set_description", "RMSE", "Spearman_rho"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    final_results.to_csv(
        f"../../data/modeling/final_results_{timestamp}.csv", index=False
    )
    ensemble_result["weight_table"].to_csv(
        f"../../data/modeling/ensemble_weights_{timestamp}.csv", index=False
    )
    diag_result["per_antibody_df"].to_csv(
        f"../../data/modeling/failure_overlap_{timestamp}.csv", index=False
    )
    diag_result["error_corr_matrix"].to_csv(
        f"../../data/modeling/error_correlation_{timestamp}.csv"
    )

    print(f"\nSaved: final_results_{timestamp}.csv")
    print(f"Saved: ensemble_weights_{timestamp}.csv")
    print(f"Saved: failure_overlap_{timestamp}.csv")
    print(f"Saved: error_correlation_{timestamp}.csv")

    return "Complete"


if __name__ == "__main__":
    print("hello world")
    print(driver())