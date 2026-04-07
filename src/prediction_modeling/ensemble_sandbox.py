
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
from datetime import datetime

import argparse
from pathlib import Path
import sys

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

models = {
    "ElasticNet": ElasticNet(),
    
 
}
 
# models = {
#     "Ridge": Ridge(),
# } #"SVR_rbf": SVR(kernel="rbf"),
    # "Ridge": Ridge(),
    # "Lasso": Lasso(),
    # "ElasticNet": ElasticNet(),


def do_hugging_face(df, k, v, remove_cdr_features=False, feature_set_description=None):
    """
    Cross-validated modeling over hierarchical cluster folds for multiple model types.

    Args:
        df (pd.DataFrame): merged dataframe with target/fold/features
        k (str): LLM name
        v (pd.DataFrame): embedding df (kept for compatibility with your existing call pattern)
        remove_cdr_features (bool): if True, drop columns containing '_CDR_feature'
        feature_set_description (str): label for results table
            expected examples: 'CDR', 'prost_t5_model', 'CDR and prost_t5_model'

    Returns:
        pd.DataFrame: per-model summary results
    """
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    target = "Titer"

    drop_cols = [
        "Unnamed: 0", "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
        "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "highest_clinical_trial_asof_feb2025",
        "AC-SINS_pH7.4", "Tonset", "Tm1", "Tm2", "hierarchical_cluster_fold",
        "random_fold", "n_missing_CDR_feature", "est_status_asof_feb2025",
        "lc_subtype", "hc_subtype",
        # raw seq / ids that should not enter numeric modeling
        "antibody_name", "vh_protein_sequence", "hc_protein_sequence", "hc_dna_sequence",
        "vl_protein_sequence", "lc_protein_sequence", "lc_dna_sequence",
        "heavy_aligned_aho", "light_aligned_aho",
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Optional: remove all CDR feature columns for easy subsetting
    if remove_cdr_features:
        cdr_cols_to_remove = [c for c in df.columns if "_CDR_feature" in c]
        df = df.drop(columns=cdr_cols_to_remove)
        print(f"[{k}] Removed {len(cdr_cols_to_remove)} _CDR_feature columns")

    # keep rows with non-missing target and fold
    df = df[df[target].notna()].copy()
    df = df[df[fold_col].notna()].copy()

    # numeric features only, excluding target/fold
    embedding_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    embedding_cols = [c for c in embedding_cols if c not in [target, fold_col]]

    X = df[embedding_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    fold_values = df[fold_col].to_numpy()

    nan_mask = np.isnan(X)
    print(f"[{k}] X shape: {X.shape}")
    print(f"[{k}] Total NaNs in X: {nan_mask.sum()}")
    print(f"[{k}] Rows with any NaN: {nan_mask.any(axis=1).sum()} / {X.shape[0]}")
    print(f"[{k}] Cols with any NaN: {nan_mask.any(axis=0).sum()} / {X.shape[1]}")

    assert len(X) == len(df) == len(y)

    unique_folds = [f for f in np.unique(fold_values) if f == f]

    results_rows = []

    for model_name, model in models.items():
        y_pred_all = np.full(len(df), np.nan)
        y_true_all = np.full(len(df), np.nan)
        per_fold_stats = []

        for f in unique_folds:
            test_idx = np.where(fold_values == f)[0]
            train_idx = np.where(fold_values != f)[0]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            fitted_model = model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)

            # PLSRegression can return shape (n,1)
            y_pred = np.asarray(y_pred).reshape(-1)

            y_pred_all[test_idx] = y_pred
            y_true_all[test_idx] = y_test

            fold_r2 = r2_score(y_test, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            fold_rho = spearmanr(y_test, y_pred).statistic
            per_fold_stats.append((int(f), fold_r2, fold_rmse, fold_rho, len(y_test)))

        mask = ~np.isnan(y_true_all)
        overall_r2 = r2_score(y_true_all[mask], y_pred_all[mask])
        overall_rmse = np.sqrt(mean_squared_error(y_true_all[mask], y_pred_all[mask]))
        overall_rho = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic

        print(f"\n[{k}] {model_name}")
        print("Fold\tN\tR2\tRMSE\tSpearman_rho")
        for f, fold_r2, fold_rmse, fold_rho, n in per_fold_stats:
            print(f"{f}\t{n}\t{fold_r2:.4f}\t{fold_rmse:.4f}\t{fold_rho:.4f}")
        print(f"Overall\t{mask.sum()}\t{overall_r2:.4f}\t{overall_rmse:.4f}\t{overall_rho:.4f}")

        results_rows.append({
            "feature_set_description": feature_set_description if feature_set_description is not None else k,
            "model_type": model_name,
            "LLM_name": k,
            "r_Squared": overall_r2,
            "RMSE": overall_rmse,
            "Spearman_rho":overall_rho,
        })

    results_df = pd.DataFrame(results_rows)
    return results_df

def do_modeling(v, k, remove_cdr_features=False, feature_set_description=None):
    """
    Load raw GDPa1 data, merge with embedding/features df, and run CV modeling.

    Args:
        v (pd.DataFrame): features/embeddings df
        k (str): LLM name
        remove_cdr_features (bool): whether to remove *_CDR_feature cols before modeling
        feature_set_description (str): e.g. 'CDR', 'prot_t5_model', 'CDR and prot_t5_model'

    Returns:
        pd.DataFrame: results for this feature/model run
    """
    print("---------------------------------------------------------")
    og_df_ = pd.read_csv("../../data/raw/GDPa1_246 IgGs_cleaned.csv")
    og_df = og_df_

    join_key = "antibody_name"
    cols_to_drop = [c for c in v.columns if c in og_df.columns and c != join_key]
    v_clean = v.drop(columns=cols_to_drop)

    merged = og_df.merge(v_clean, on=join_key, how="inner")
    print(f"[{k}] og_df: {og_df.shape}, v_clean: {v_clean.shape}, merged: {merged.shape}")

    run_results = do_hugging_face(
        merged,
        k,
        v_clean,
        remove_cdr_features=remove_cdr_features,
        feature_set_description=feature_set_description,
    )

    return run_results

def load_pickles_to_df_dict(folder_path: str) -> dict:
    """
    Load all .pkl files in a folder into pandas DataFrames.

    Returns:
        dict: {file_key: DataFrame}
              where file_key strips '_df' and extension
              e.g. esmc_model_df.pkl -> 'esmc_model'
    """

    cdr_features = pd.read_csv(r"../../data/modeling/cdr_features_titer.csv")
    # ── Rename all CDR columns with suffix ─────────────────────────────────
    cdr_features = cdr_features.add_suffix("_CDR_feature")

    folder = Path(folder_path)
    out = {}

    for p in folder.glob("*.pkl"):
        try:
            #print('doing try')
            df = pd.read_pickle(p)

            if isinstance(df, pd.DataFrame):
                key = p.stem.replace("_df", "")
                # ── Append CDR features column-wise ────────────────────
                # Reset index on both to guarantee row alignment
                df = pd.concat(
                    [df.reset_index(drop=True),
                     cdr_features.reset_index(drop=True)],
                    axis=1
                )
                out[key] = df

        except Exception as e:
            print(f"Skipping {p.name}: {e}")
    
    return out





def driver():
    """
    Orchestrate the full modeling pipeline: data loading, exploration, modeling,
    and result compilation.

    Loads all pickled DataFrames from the modeling data directory, prints a
    summary of each dataset, and passes each DataFrame through the modeling
    pipeline (do_modeling). Intended as the main entry point for the script.

    Returns:
        str: The string "Complete" upon successful execution of the pipeline.
    """
 
    #load file 
    df_dict = load_pickles_to_df_dict("../../data/modeling")

    final_results = pd.DataFrame(columns=[
    "feature_set_description",
    "model_type",
    "LLM_name",
    "r_Squared",
    "RMSE",
    "Spearman_rho",])

    for k,v in df_dict.items(): # for each LLM embedding
        #do_cdr
        run_cdr = do_modeling(v,k, remove_cdr_features=False, feature_set_description=f"CDR and {k}",)
        final_results = pd.concat([final_results, run_cdr], ignore_index=True)

        #do_non_cdr
        run_non_cdr = do_modeling(v,k, remove_cdr_features=True, feature_set_description=k,)
        final_results = pd.concat([final_results, run_non_cdr], ignore_index=True)


        #compile results & save
    final_results = final_results.sort_values(by=["feature_set_description", "RMSE","Spearman_rho"],ascending=[True, True, True]).reset_index(drop=True)
    print(final_results.head(3))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    final_results.to_csv(f"../../data/modeling/final_results_{timestamp}.csv", index=False)
    return "Complete"




if __name__ == "__main__":
    print('hello world')
    get_ans = driver()
    print(get_ans)