import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
from pathlib import Path
import sys


model_name = "ollieturnbull/p-IgGen"
df = pd.read_csv( "../../data/raw/GDPa1_246 IgGs_cleaned.csv") 
# Show number of NaNs per assay
print(df[["Titer", "HIC", "PR_CHO", "Tm2", 'AC-SINS_pH7.4']].isna().sum())
target = "HIC"

# Example: Just predict HIC, so we'll drop NaN rows for that
df = df.dropna(subset=[target])

# Tokenize the sequences
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Paired sequence handling: Concatenate heavy and light chains and add beginning ("1") and end ("2") tokens
# (e.g. ["EVQLV...", "DIQMT..."] -> "1E V Q L V ... D I Q M T ... 2")
sequences = [
    "1" + " ".join(heavy) + " ".join(light) + "2"
    for heavy, light in zip(
        df["vh_protein_sequence"],
        df["vl_protein_sequence"],
    )
]

print(sequences[0])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Takes about 60 seconds for 242 sequences on my CPU, and 1.1s on GPU
batch_size = 16
mean_pooled_embeddings = []
for i in tqdm(range(0, len(sequences), batch_size)):
    batch = tokenizer(sequences[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
    outputs = model(batch["input_ids"].to(device), return_rep_layers=[-1], output_hidden_states=True)
    embeddings = outputs["hidden_states"][-1].detach().cpu().numpy()
    mean_pooled_embeddings.append(embeddings.mean(axis=1))
mean_pooled_embeddings = np.concatenate(mean_pooled_embeddings)

# Train a linear regression on these
X = mean_pooled_embeddings
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = Ridge()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)


# Calculate score
print(spearmanr(y_pred, y_test))

sns.scatterplot(x=y_test, y=y_pred)
plt.title(f"Scatter plot of predicted vs. true {target}\nSpearman's rho: {spearmanr(y_pred, y_test)[0]:.2f}")
plt.xlabel(f"True {target}")
plt.ylabel(f"Predicted {target}")
plt.show()


fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
X = mean_pooled_embeddings
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
