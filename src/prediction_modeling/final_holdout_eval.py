"""
Final holdout evaluation: train on full GDPa1 set, evaluate on GDPa3 holdout.
Uses CDR + PLM embedding features, consistent with ensemble_beta.py CV pipeline.
PCA is applied to embedding dimensions only (same as training pipeline).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

# sys.path must be modified before the local CDR_work import below
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from CDR_work.cdr_feature_utils import build_feature_matrix  # noqa: E402

TARGET = "Titer"

# Same drop list as ensemble_beta.py
DROP_COLS = [
    "Unnamed: 0",
    "Unnamed: 0_CDR_feature",
    "Purity",
    "SEC %Monomer",
    "SMAC",
    "HIC",
    "HAC",
    "PR_CHO",
    "PR_Ova",
    "AC-SINS_pH6.0",
    "AC-SINS_pH7.4",
    "Tonset",
    "Tm1",
    "Tm2",
    "highest_clinical_trial_asof_feb2025",
    "hierarchical_cluster_fold",
    "hierarchical_cluster_IgG_isotype_stratified_fold",
    "random_fold",
    "n_missing_CDR_feature",
    "est_status_asof_feb2025",
    "lc_subtype",
    "hc_subtype",
    "n_missing",
    "missing_cols",
    "vh_protein_sequence",
    "hc_protein_sequence",
    "hc_dna_sequence",
    "vl_protein_sequence",
    "lc_protein_sequence",
    "lc_dna_sequence",
    "heavy_aligned_aho",
    "light_aligned_aho",
    "antibody_name",
]

# Fixed hyperparams — set these to the best values found from CV
MODELS = {
    "Ridge": Ridge(alpha=10),
    "Lasso": Lasso(alpha=10, random_state=42, max_iter=100000),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=100000),
    "SVR_rbf": SVR(kernel="rbf", C=1.0),
}


def load_train_cdr() -> pd.DataFrame:
    """Load pre-computed CDR features for training antibodies."""
    cdr = pd.read_csv(ROOT / "data/modeling/cdr_features_titer.csv")
    cdr_cols = [c for c in cdr.columns if c != "antibody_name"]
    return cdr.rename(columns={c: f"{c}_CDR_feature" for c in cdr_cols})


def load_holdout_cdr() -> pd.DataFrame:
    """Compute CDR features for holdout antibodies from AHO-aligned sequences."""
    holdout_csv = ROOT / "data/test_data/cleaned_holdout_data_with_aho.csv"
    holdout_df = pd.read_csv(holdout_csv)
    cdr_feats = build_feature_matrix(holdout_df)
    cdr_feats = cdr_feats.add_suffix("_CDR_feature")
    # Carry antibody_name and Titer for merging and evaluation
    carry = [c for c in ["antibody_name", TARGET] if c in holdout_df.columns]
    return pd.concat(
        [holdout_df[carry].reset_index(drop=True), cdr_feats.reset_index(drop=True)],
        axis=1,
    )


def load_plm_df(folder: Path, plm_key: str, cdr_df: pd.DataFrame) -> pd.DataFrame:
    """Load a PLM embedding pickle and left-join CDR features on antibody_name."""
    emb_df = pd.read_pickle(folder / f"{plm_key}_model_df.pkl")
    # Avoid duplicate columns (keep version from cdr_df for CDR cols)
    overlap = [
        c for c in cdr_df.columns if c in emb_df.columns and c != "antibody_name"
    ]
    return emb_df.drop(columns=overlap).merge(cdr_df, on="antibody_name", how="left")


def get_X_y(df: pd.DataFrame):
    """Drop non-feature columns, filter to labelled rows, return X, y, feature_cols."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    df = df[df[TARGET].notna()].copy()
    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET
    ]
    return (
        df[feat_cols].to_numpy(dtype=float),
        df[TARGET].to_numpy(dtype=float),
        feat_cols,
    )


def preprocess(X_train, X_test, feature_cols):
    """Impute → scale → PCA (embedding dims only). Fit on train, apply to both."""
    cdr_mask = np.array(["_CDR_feature" in c for c in feature_cols])
    emb_mask = ~cdr_mask

    def _pipe(Xtr, Xte, apply_pca):
        imp = SimpleImputer(strategy="mean")
        Xtr = imp.fit_transform(Xtr)
        Xte = imp.transform(Xte)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        if apply_pca and Xtr.shape[1] > 1:
            pca = PCA(n_components=0.95, random_state=42)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
            print(
                f"    PCA: {emb_mask.sum()} embedding dims → {Xtr.shape[1]} components"
            )
        return Xtr, Xte

    Xtr_cdr, Xte_cdr = _pipe(X_train[:, cdr_mask], X_test[:, cdr_mask], apply_pca=False)
    Xtr_emb, Xte_emb = _pipe(X_train[:, emb_mask], X_test[:, emb_mask], apply_pca=True)
    return np.hstack([Xtr_cdr, Xtr_emb]), np.hstack([Xte_cdr, Xte_emb])


def bootstrap_spearman_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple:
    """
    Percentile bootstrap CI for Spearman ρ.
    With only 80 holdout antibodies a point estimate alone is unreliable —
    this CI quantifies sampling variability.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rhos.append(spearmanr(y_true[idx], y_pred[idx]).statistic)
    lo = float(np.percentile(boot_rhos, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_rhos, (1 + ci) / 2 * 100))
    return lo, hi


def compute_metrics(y_true, y_pred):
    rho = float(spearmanr(y_true, y_pred).statistic)
    rho_lo, rho_hi = bootstrap_spearman_ci(y_true, y_pred)
    return {
        "RMSE":           float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":            float(mean_absolute_error(y_true, y_pred)),
        "R2":             float(r2_score(y_true, y_pred)),
        "Spearman":       rho,
        "Spearman_CI_lo": rho_lo,
        "Spearman_CI_hi": rho_hi,
    }


def main():
    train_cdr = load_train_cdr()
    holdout_cdr = load_holdout_cdr()
    train_dir = ROOT / "data/modeling"
    holdout_dir = ROOT / "data/modeling/holdout"

    plm_keys = sorted(
        p.stem.replace("_model_df", "") for p in train_dir.glob("*_model_df.pkl")
    )
    print(f"PLM keys found: {plm_keys}\n")

    all_results = []

    for plm_key in plm_keys:
        if not (holdout_dir / f"{plm_key}_model_df.pkl").exists():
            print(f"[{plm_key}] No holdout pickle — skipping.")
            continue

        print(f"\n{'=' * 55}")
        print(f"PLM: {plm_key}")

        train_df = load_plm_df(train_dir, plm_key, train_cdr)
        holdout_df = load_plm_df(holdout_dir, plm_key, holdout_cdr)

        X_train, y_train, feat_cols = get_X_y(train_df)
        X_holdout_raw, y_holdout, h_fc = get_X_y(holdout_df)

        # Align holdout feature set to training (fills missing cols with NaN)
        X_holdout = (
            pd.DataFrame(X_holdout_raw, columns=h_fc)
            .reindex(columns=feat_cols)
            .to_numpy(dtype=float)
        )

        X_train_p, X_holdout_p = preprocess(X_train, X_holdout, feat_cols)

        for model_name, model in MODELS.items():
            model.fit(X_train_p, y_train)
            preds = model.predict(X_holdout_p)
            metrics = compute_metrics(y_holdout, preds)
            all_results.append({"plm_key": plm_key, "model": model_name, **metrics})
            print(
                f"  [{model_name}] Spearman={metrics['Spearman']:.4f} "
                f"[{metrics['Spearman_CI_lo']:.4f}, {metrics['Spearman_CI_hi']:.4f}]  "
                f"RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}"
            )

    print(f"\n{'=' * 55}")
    print("FINAL HOLDOUT RESULTS (sorted by Spearman ρ)")
    print("=" * 55)
    results_df = (
        pd.DataFrame(all_results)
        .sort_values("Spearman", ascending=False)
        .reset_index(drop=True)
    )
    print(results_df.to_string(index=False))

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    out_path = ROOT / f"data/modeling/holdout_eval_{timestamp}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    main()
