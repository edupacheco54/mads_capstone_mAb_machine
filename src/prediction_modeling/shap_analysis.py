"""
shap_analysis.py — SHAP feature importance for CDR + PLM embedding features.

Trains linear models on the full GDPa1 set WITHOUT dimensionality reduction so
that CDR feature names are preserved and SHAP values are directly interpretable.
For PLM embeddings, aggregate |SHAP| per chain block is reported rather than
per-dimension, since individual embedding dimensions have no labels.

Models: Ridge, Lasso, ElasticNet (LinearExplainer — exact and fast).
SVR is excluded; KernelExplainer would be prohibitively slow here.

Output (data/figures/shap/):
  shap_cdr_bar_{plm}_{model}.png        — top-20 CDR features by mean |SHAP|
  shap_cdr_beeswarm_{plm}_{model}.png   — SHAP distribution per CDR feature
  shap_feature_group_{plm}_{model}.png  — CDR block vs embedding block importance
  shap_cross_plm_{model}.png            — CDR/embedding split across all PLMs
  shap_mean_abs_{plm}.csv               — full mean |SHAP| table per model

Run:
    uv run src/prediction_modeling/shap_analysis.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from CDR_work.cdr_feature_utils import build_feature_matrix  # noqa: E402

try:
    import shap
except ImportError as exc:
    raise SystemExit("shap not installed. Run: uv add shap") from exc

# ── Config ───────────────────────────────────────────────────────────────────

TARGET      = "Titer"
FIGURES_DIR = ROOT / "data/figures/shap"

DROP_COLS = [
    "Unnamed: 0", "Unnamed: 0_CDR_feature",
    "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
    "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
    "Tonset", "Tm1", "Tm2", "highest_clinical_trial_asof_feb2025",
    "hierarchical_cluster_fold", "hierarchical_cluster_IgG_isotype_stratified_fold",
    "random_fold", "n_missing_CDR_feature", "est_status_asof_feb2025",
    "lc_subtype", "hc_subtype", "n_missing", "missing_cols",
    "vh_protein_sequence", "hc_protein_sequence", "hc_dna_sequence",
    "vl_protein_sequence", "lc_protein_sequence", "lc_dna_sequence",
    "heavy_aligned_aho", "light_aligned_aho", "antibody_name",
]

MODELS = {
    "Ridge":      Ridge(alpha=10),
    "Lasso":      Lasso(alpha=10, random_state=42, max_iter=100000),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=100000),
}

# Colour palette — CDR = blue, embeddings = orange
BLUE   = "#4C72B0"
ORANGE = "#DD8452"

# ── Data loading ─────────────────────────────────────────────────────────────

def load_train_cdr() -> pd.DataFrame:
    cdr      = pd.read_csv(ROOT / "data/modeling/cdr_features_titer.csv")
    cdr_cols = [c for c in cdr.columns if c != "antibody_name"]
    return cdr.rename(columns={c: f"{c}_CDR_feature" for c in cdr_cols})


def load_plm_df(plm_key: str, cdr_df: pd.DataFrame) -> pd.DataFrame:
    emb_df  = pd.read_pickle(ROOT / "data/modeling" / f"{plm_key}_model_df.pkl")
    overlap = [c for c in cdr_df.columns if c in emb_df.columns and c != "antibody_name"]
    return emb_df.drop(columns=overlap).merge(cdr_df, on="antibody_name", how="left")


def get_X_y(df: pd.DataFrame):
    df        = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    df        = df[df[TARGET].notna()].copy()
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    return df[feat_cols].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float), feat_cols


def preprocess_no_pca(X: np.ndarray):
    """Impute + scale without PCA — keeps features named for SHAP."""
    imp    = SimpleImputer(strategy="mean")
    X      = imp.fit_transform(X)
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)
    return X


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save(fig, path: Path, tight=True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path.name}")


def plot_cdr_bar(shap_vals, feat_cols, plm_key, model_name, top_n=20) -> None:
    cdr_mask  = np.array(["_CDR_feature" in c for c in feat_cols])
    cdr_shap  = np.abs(shap_vals[:, cdr_mask])
    cdr_names = [c.replace("_CDR_feature", "") for c in np.array(feat_cols)[cdr_mask]]

    mean_abs = cdr_shap.mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [cdr_names[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color=BLUE,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} CDR features — {plm_key} | {model_name}")
    _save(fig, FIGURES_DIR / f"shap_cdr_bar_{plm_key}_{model_name}.png")


def plot_cdr_beeswarm(shap_vals, X, feat_cols, plm_key, model_name, top_n=20) -> None:
    cdr_mask  = np.array(["_CDR_feature" in c for c in feat_cols])
    cdr_names = [c.replace("_CDR_feature", "") for c in np.array(feat_cols)[cdr_mask]]

    shap.summary_plot(
        shap_vals[:, cdr_mask],
        X[:, cdr_mask],
        feature_names=cdr_names,
        max_display=top_n,
        show=False,
        plot_size=(9, 8),
    )
    plt.title(f"CDR SHAP beeswarm — {plm_key} | {model_name}", pad=10)
    path = FIGURES_DIR / f"shap_cdr_beeswarm_{plm_key}_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"    Saved {path.name}")


def plot_feature_group(shap_vals, feat_cols, plm_key, model_name) -> dict:
    """Bar chart: CDR block vs embedding block aggregate |SHAP|. Returns values."""
    cdr_mask = np.array(["_CDR_feature" in c for c in feat_cols])
    emb_mask = ~cdr_mask

    cdr_agg = float(np.abs(shap_vals[:, cdr_mask]).sum(axis=1).mean())
    emb_agg = float(np.abs(shap_vals[:, emb_mask]).sum(axis=1).mean()) if emb_mask.any() else 0.0

    labels = ["CDR features", f"{plm_key}\nembeddings"]
    values = [cdr_agg, emb_agg]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=[BLUE, ORANGE], width=0.45, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Mean aggregate |SHAP value| per antibody")
    ax.set_title(f"Feature group importance — {plm_key} | {model_name}")
    ax.set_ylim(0, max(values) * 1.18)
    _save(fig, FIGURES_DIR / f"shap_feature_group_{plm_key}_{model_name}.png")

    return {"cdr": cdr_agg, "emb": emb_agg}


def plot_cross_plm(group_results: dict, model_name: str) -> None:
    """Grouped bar: CDR vs embedding importance across all PLMs for one model."""
    plm_keys = list(group_results.keys())
    cdr_vals = [group_results[k]["cdr"] for k in plm_keys]
    emb_vals = [group_results[k]["emb"] for k in plm_keys]

    x     = np.arange(len(plm_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, cdr_vals, width, label="CDR features",   color=BLUE,   edgecolor="white")
    ax.bar(x + width / 2, emb_vals, width, label="PLM embeddings", color=ORANGE, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(plm_keys, rotation=15, ha="right")
    ax.set_ylabel("Mean aggregate |SHAP value| per antibody")
    ax.set_title(f"CDR vs PLM embedding importance across all models — {model_name}")
    ax.legend()
    _save(fig, FIGURES_DIR / f"shap_cross_plm_{model_name}.png")


# ── Per-PLM orchestration ─────────────────────────────────────────────────────

def run_shap_for_plm(plm_key: str, cdr_df: pd.DataFrame) -> dict:
    """
    Run SHAP analysis for all models on one PLM.
    Returns group_agg dict {model_name: {cdr, emb}} for cross-PLM plot.
    """
    df           = load_plm_df(plm_key, cdr_df)
    X_raw, y, feat_cols = get_X_y(df)
    X            = preprocess_no_pca(X_raw)

    # Background sample for LinearExplainer (subset of training data)
    n_bg = min(100, len(X))
    rng  = np.random.default_rng(42)
    bg   = X[rng.choice(len(X), n_bg, replace=False)]

    group_agg    = {}
    mean_abs_rows = {"feature": feat_cols}

    for model_name, model in MODELS.items():
        print(f"  [{model_name}] fitting + computing SHAP values...")
        try:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X, y)

            oof_pred = model_clone.predict(X)
            rho      = spearmanr(y, oof_pred).statistic
            print(f"    Train Spearman ρ = {rho:.4f}  "
                  f"(note: no CV here — SHAP uses full-data fit for interpretability)")

            explainer  = shap.LinearExplainer(model_clone, bg)
            shap_vals  = explainer.shap_values(X)

            mean_abs_rows[model_name] = np.abs(shap_vals).mean(axis=0)

            plot_cdr_bar(shap_vals, feat_cols, plm_key, model_name)
            plot_cdr_beeswarm(shap_vals, X, feat_cols, plm_key, model_name)
            group_agg[model_name] = plot_feature_group(shap_vals, feat_cols, plm_key, model_name)

        except Exception as exc:
            warnings.warn(f"SHAP failed for {plm_key}/{model_name}: {exc}")
            plt.close("all")
            continue

    # Save mean |SHAP| CSV for all models in this PLM
    pd.DataFrame(mean_abs_rows).to_csv(
        FIGURES_DIR / f"shap_mean_abs_{plm_key}.csv", index=False
    )
    print(f"    Saved shap_mean_abs_{plm_key}.csv")

    return group_agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cdr_df    = load_train_cdr()
    train_dir = ROOT / "data/modeling"
    plm_keys  = sorted(
        p.stem.replace("_model_df", "") for p in train_dir.glob("*_model_df.pkl")
    )
    print(f"PLM keys: {plm_keys}\n")

    # Accumulate per-model cross-PLM group results
    # cross_plm_results[model_name][plm_key] = {cdr, emb}
    cross_plm_results: dict[str, dict] = {m: {} for m in MODELS}

    for plm_key in plm_keys:
        print(f"\n{'=' * 55}")
        print(f"SHAP: {plm_key}")
        try:
            group_agg = run_shap_for_plm(plm_key, cdr_df)
            for model_name, vals in group_agg.items():
                cross_plm_results[model_name][plm_key] = vals
        except Exception as exc:
            warnings.warn(f"Skipping {plm_key} entirely: {exc}")
            plt.close("all")
            continue

    # Cross-PLM comparison for each model (requires ≥2 PLMs with results)
    print(f"\n{'=' * 55}")
    print("Cross-PLM comparison plots")
    for model_name, plm_dict in cross_plm_results.items():
        if len(plm_dict) < 2:
            continue
        try:
            plot_cross_plm(plm_dict, model_name)
        except Exception as exc:
            warnings.warn(f"Cross-PLM plot failed for {model_name}: {exc}")
            plt.close("all")

    print(f"\nAll SHAP figures saved to {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
