
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt

import argparse
from pathlib import Path
import sys

from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# def do_hugging_face(df, k,v):
#     """
#     Perform cross-validated Ridge regression on antibody embedding features to predict titer values.

#     Adopted from:
#     https://huggingface.co/blog/ginkgo-datapoints/making-antibody-embeddings-and-predictions

#     Iterates over predefined hierarchical cluster folds, trains a Ridge regression model
#     on each training split, evaluates on the held-out fold using Spearman correlation,
#     and aggregates predictions across all folds. Prints per-fold and overall Spearman rho,
#     and displays a scatter plot of true vs. predicted titer values.

#     Args:
#         df (pd.DataFrame): Input dataframe containing feature columns, a fold assignment
#                            column ('hierarchical_cluster_IgG_isotype_stratified_fold'),
#                            and a 'Titer' target column.

#     Returns:
#         None
#     """
#     fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
#     target = 'Titer' # HARDCODED
#     #k = name of LLM


#     #DROP COLUMNS
#     drop_cols = ['Unnamed: 0', 'Purity', 'SEC %Monomer', 'SMAC', 'HIC', 'HAC', 'PR_CHO', 'PR_Ova', 'AC-SINS_pH6.0', 'highest_clinical_trial_asof_feb2025'
#                  'AC-SINS_pH7.4', 'Tonset', 'Tm1', 'Tm2', 'hierarchical_cluster_fold', 'random_fold','n_missing_CDR_feature','est_status_asof_feb2025','lc_subtype','hc_subtype']

#     df = df.drop(columns=[c for c in drop_cols if c in df.columns])

#     embedding_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     X = df[embedding_cols].to_numpy(dtype=float)



#     # Identify embedding columns — everything that's not metadata
#     meta_cols = [target, fold_col, "antibody_id",
#                  "vh_protein_sequence", "vl_protein_sequence"]
    
#     # Diagnose NaNs before doing anything else
#     nan_mask = np.isnan(X)
#     print(f"[{k}] X shape: {X.shape}")
#     print(f"[{k}] Total NaNs in X: {nan_mask.sum()}")
#     print(f"[{k}] Rows with any NaN: {nan_mask.any(axis=1).sum()} / {X.shape[0]}")
#     print(f"[{k}] Cols with any NaN: {nan_mask.any(axis=0).sum()} / {X.shape[1]}")

#     y = df[target].to_numpy(dtype=float)
#     fold_values = df[fold_col].to_numpy()
#     # sanity check
#     assert len(X) == len(df) == len(y)

#     unique_folds = [f for f in np.unique(fold_values) if f == f]  # drop NaN
#     per_fold_stats = []
#     y_pred_all = np.full(len(df), np.nan)   # align with df rows
#     y_true_all = np.full(len(df), np.nan)   # optional, for plotting/metrics

#     for f in unique_folds:
#         test_idx = np.where(fold_values == f)[0]
#         train_idx = np.where(fold_values != f)[0]

#         X_train, y_train = X[train_idx], y[train_idx]
#         X_test,  y_test  = X[test_idx],  y[test_idx]

#         # Impute NaNs — fit on train only, apply to both
#         imputer = SimpleImputer(strategy="mean")
#         X_train = imputer.fit_transform(X_train)
#         X_test  = imputer.transform(X_test)

#         # Scale — fit on train only, apply to both
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test  = scaler.transform(X_test)

#         lm = Ridge()
#         lm.fit(X_train, y_train)
#         y_pred = lm.predict(X_test)

#         # write back into the positions of df
#         y_pred_all[test_idx] = y_pred
#         y_true_all[test_idx] = y_test

#         rho = spearmanr(y_test, y_pred).statistic
#         per_fold_stats.append((int(f), rho, len(y_test)))

#     # Overall metric across all rows that participated in CV
#     mask = ~np.isnan(y_true_all)
#     overall_rho = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic

#     print("Fold\tN\tSpearman_rho")
#     for f, rho, n in per_fold_stats:
#         print(f"{f}\t{n}\t{rho:.4f}")
#     print(f"Overall (all folds)\t{mask.sum()}\t{overall_rho:.4f}")

#     plt.figure()
#     plt.scatter(y_true_all[mask], y_pred_all[mask], alpha=0.7)
#     plt.title(f"Ridge CV over {fold_col}\nOverall Spearman: {overall_rho:.3f}")
#     plt.xlabel(f"True {target}")
#     plt.ylabel(f"Predicted {target}")
#     plt.show()

#     return None 

def do_hugging_face(df, k, v):
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    target = 'Titer'

    drop_cols = ['Unnamed: 0', 'Purity', 'SEC %Monomer', 'SMAC', 'HIC', 'HAC',
                 'PR_CHO', 'PR_Ova', 'AC-SINS_pH6.0', 'highest_clinical_trial_asof_feb2025',
                 'AC-SINS_pH7.4', 'Tonset', 'Tm1', 'Tm2', 'hierarchical_cluster_fold',
                 'random_fold', 'n_missing_CDR_feature', 'est_status_asof_feb2025',
                 'lc_subtype', 'hc_subtype']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    embedding_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[embedding_cols].to_numpy(dtype=float)

    nan_mask = np.isnan(X)
    print(f"[{k}] X shape: {X.shape}")
    print(f"[{k}] Total NaNs in X: {nan_mask.sum()}")
    print(f"[{k}] Rows with any NaN: {nan_mask.any(axis=1).sum()} / {X.shape[0]}")
    print(f"[{k}] Cols with any NaN: {nan_mask.any(axis=0).sum()} / {X.shape[1]}")

    y = df[target].to_numpy(dtype=float)
    fold_values = df[fold_col].to_numpy()
    assert len(X) == len(df) == len(y)

    unique_folds = [f for f in np.unique(fold_values) if f == f]
    per_fold_stats = []
    y_pred_all = np.full(len(df), np.nan)
    y_true_all = np.full(len(df), np.nan)
    coef_accumulator = np.zeros(X.shape[1])                          # ← ADD

    for f in unique_folds:
        test_idx  = np.where(fold_values == f)[0]
        train_idx = np.where(fold_values != f)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        lm = Ridge()
        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)

        coef_accumulator += np.abs(lm.coef_)                         # ← ADD

        y_pred_all[test_idx] = y_pred
        y_true_all[test_idx] = y_test
        rho = spearmanr(y_test, y_pred).statistic
        per_fold_stats.append((int(f), rho, len(y_test)))

    # ── Average coefficients across folds ──────────────────────────────── ← ADD
    mean_coef = coef_accumulator / len(unique_folds)                  # ← ADD
    feature_importance = pd.Series(mean_coef, index=embedding_cols)  # ← ADD

    # ── CDR-specific importance (interpretable subset) ─────────────────── ← ADD
    cdr_importance = (                                                 # ← ADD
        feature_importance                                            # ← ADD
        .filter(like="_CDR_feature")                                  # ← ADD
        .sort_values(ascending=False)                                 # ← ADD
    )                                                                 # ← ADD
    print(f"\n[{k}] Top 10 CDR features by mean |coef|:")            # ← ADD
    print(cdr_importance.head(10))                                    # ← ADD

    # ── Overall CV results ─────────────────────────────────────────────────
    mask = ~np.isnan(y_true_all)
    overall_rho = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic
    print("\nFold\tN\tSpearman_rho")
    for f, rho, n in per_fold_stats:
        print(f"{f}\t{n}\t{rho:.4f}")
    print(f"Overall (all folds)\t{mask.sum()}\t{overall_rho:.4f}")

    # ── Scatter plot ───────────────────────────────────────────────────────
    plt.figure()
    plt.scatter(y_true_all[mask], y_pred_all[mask], alpha=0.7)
    plt.title(f"Ridge CV over {fold_col}\nOverall Spearman: {overall_rho:.3f}")
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.show()

    # ── CDR feature importance bar chart ──────────────────────────────── ← ADD
    plt.figure(figsize=(10, 5))                                        # ← ADD
    cdr_importance.plot(kind="bar")                                    # ← ADD
    plt.title(f"[{k}] CDR Feature Importance (mean |Ridge coef| across folds)")  # ← ADD
    plt.ylabel("Mean |coefficient|")                                   # ← ADD
    plt.xticks(rotation=45, ha="right")                               # ← ADD
    plt.tight_layout()                                                 # ← ADD
    plt.show()                                                         # ← ADD

    return feature_importance                                          # ← ADD (was None)

def do_modeling(v, k):
    """
    Load the raw IgG dataset, filter rows with valid titer values, and trigger
    cross-validated modeling via do_hugging_face().

    Reads the primary antibody dataset from a fixed CSV path, drops rows where
    'Titer' is missing, and passes the cleaned dataframe to do_hugging_face()
    for Ridge regression cross-validation.

    Args:
        v: LLM embedding vectors 
        k: name of LLM
    Returns:
        None
    """
    print('---------------------------------------------------------')
    # # ['AC_SINS_pH6.0', 'AC-SINS_pH7.4']
    og_df_ =  pd.read_csv("../../data/raw/GDPa1_246 IgGs_cleaned.csv")
    og_df = og_df_ #.dropna(subset=['Titer'])

    # Drop any columns in v that already exist in og_df (except the join key)
    join_key = "antibody_name"
    cols_to_drop = [c for c in v.columns if c in og_df.columns and c != join_key]
    
    #print(f"[{k}] Dropping duplicate cols from v before merge: {cols_to_drop}")
    v_clean = v.drop(columns=cols_to_drop)
    merged = og_df.merge(v_clean, on=join_key, how="inner")
    print(f"[{k}] og_df: {og_df.shape}, v_clean: {v_clean.shape}, merged: {merged.shape}")
    
    get_hugging = do_hugging_face(merged, k, v_clean)

    return None 




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

    for k,v in df_dict.items(): # for each LLM embedding
        print(k, len(v.columns) )

        #explore / transform data

        #call model
        
        get_model = do_modeling(v, k)
        print(k)
        print()
        #compile results & save

    return "Complete"




if __name__ == "__main__":
    print('hello world')
    get_ans = driver()
    print(get_ans)