
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

def do_hugging_face():
    """
    Perform cross-validated Ridge regression on antibody embedding features to predict titer values.

    Adopted from:
    https://huggingface.co/blog/ginkgo-datapoints/making-antibody-embeddings-and-predictions

    Iterates over predefined hierarchical cluster folds, trains a Ridge regression model
    on each training split, evaluates on the held-out fold using Spearman correlation,
    and aggregates predictions across all folds. Prints per-fold and overall Spearman rho,
    and displays a scatter plot of true vs. predicted titer values.

    Args:
        df (pd.DataFrame): Input dataframe containing feature columns, a fold assignment
                           column ('hierarchical_cluster_IgG_isotype_stratified_fold'),
                           and a 'Titer' target column.

    Returns:
        None
    """

    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    target = 'Titer'

    X = "feature_colmns "
    y = df[target].to_numpy(dtype=float)

    # sanity check
    assert len(X) == len(df) == len(y)

    fold_values = df[fold_col].to_numpy()
    unique_folds = [f for f in np.unique(fold_values) if f == f]  # drop NaN

    per_fold_stats = []
    y_pred_all = np.full(len(df), np.nan)   # align with df rows
    y_true_all = np.full(len(df), np.nan)   # optional, for plotting/metrics

    for f in unique_folds:
        test_idx = np.where(fold_values == f)[0]
        train_idx = np.where(fold_values != f)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        lm = Ridge()
        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)

        # write back into the positions of df
        y_pred_all[test_idx] = y_pred
        y_true_all[test_idx] = y_test

        rho = spearmanr(y_test, y_pred).statistic
        per_fold_stats.append((int(f), rho, len(y_test)))

    # Overall metric across all rows that participated in CV
    mask = ~np.isnan(y_true_all)
    overall_rho = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic

    print("Fold\tN\tSpearman_rho")
    for f, rho, n in per_fold_stats:
        print(f"{f}\t{n}\t{rho:.4f}")
    print(f"Overall (all folds)\t{mask.sum()}\t{overall_rho:.4f}")

    plt.figure()
    plt.scatter(y_true_all[mask], y_pred_all[mask], alpha=0.7)
    plt.title(f"Ridge CV over {fold_col}\nOverall Spearman: {overall_rho:.3f}")
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.show()

    return None 

def do_modeling(v):
    """
    Load the raw IgG dataset, filter rows with valid titer values, and trigger
    cross-validated modeling via do_hugging_face().

    Reads the primary antibody dataset from a fixed CSV path, drops rows where
    'Titer' is missing, and passes the cleaned dataframe to do_hugging_face()
    for Ridge regression cross-validation.

    Args:
        v: Unused parameter. Reserved for future use or passed-in feature data.

    Returns:
        None
    """
    print('---------------------------------------------------------')
    # ['AC_SINS_pH6.0', 'AC-SINS_pH7.4']
    og_df_ =  pd.read_csv("../../data/raw/GDPa1_246 IgGs_cleaned.csv")
    og_df = og_df_.dropna(subset=['Titer'])

    get_hugging = do_hugging_face(og_df)

    return None 




def load_pickles_to_df_dict(folder_path: str) -> dict:
    """
    Load all .pkl files in a folder into pandas DataFrames.

    Returns:
        dict: {file_key: DataFrame}
              where file_key strips '_df' and extension
              e.g. esmc_model_df.pkl -> 'esmc_model'
    """
    folder = Path(folder_path)
    out = {}

    for p in folder.glob("*.pkl"):
        try:
            print('doing try')
            df = pd.read_pickle(p)

            if isinstance(df, pd.DataFrame):
                key = p.stem.replace("_df", "")
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

    for k,v in df_dict.items():
        print(k, v.info() )

        #explore / transform data

        #call model
        get_model = do_modeling(v)
        #compile results & save

    return "Complete"




if __name__ == "__main__":
    print('hello world')
    get_ans = driver()
    print(get_ans)