"""
failure_eda.py By Alaude Sonnet 4.6 Extended | Prompts by Allen Chezick

Merges ensemble failure-overlap diagnostics back onto the full GDPa1 246-antibody
dataframe, then runs exploratory data analysis comparing antibodies that were
consistently hard to predict vs. those that were consistently easy.

Usage:
    python failure_eda.py \
        --failure-csv  ../../data/modeling/failure_overlap_20260410_2127.csv \
        --gdpa1-csv    "../../data/raw/GDPa1_246 IgGs_cleaned.csv" \
        --out-dir      ../../data/modeling/failure_eda

Usage:
python failure_eda.py \
--failure-csv  ../../data/modeling/failure_overlap_20260410_2127.csv \
--gdpa1-csv    "../../data/raw/GDPa1_246 IgGs_cleaned.csv" \
--cdr-csv      ../../data/modeling/cdr_features_titer.csv \
--out-dir      ../../data/modeling/failure_eda

Outputs (in --out-dir):
    gdpa1_with_failure_flags.csv    full 246-row dataframe with all error/flag cols appended
    01_titer_by_group.png           Titer distribution split by failure group
    02_assay_by_group.png           Key assay readouts (HIC, HAC, SEC, AC-SINS) by group
    03_thermal_by_group.png         Tonset, Tm1, Tm2 by group
    04_n_high_err_vs_titer.png      Scatter: true Titer vs n_models_high_err
    05_mean_abserr_vs_titer.png     Scatter: mean abs error across learners vs true Titer
    06_subtype_breakdown.png        hc_subtype / lc_subtype composition by group
    07_clinical_phase_breakdown.png Clinical trial phase composition by group
    08_fold_composition.png         Which folds contain the hard antibodies
    09_error_top_distribution.png   Mean abs error distribution + top antibodies (hard + borderline)
    10_cdr_features_by_group.png    CDR biophysical features: wrong vs right (requires --cdr-csv)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import mannwhitneyu


# ── Plotting defaults ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"consistently_wrong": "#d62728",
           "sometimes_wrong":    "#ff7f0e",
           "consistently_right": "#2ca02c"}
GROUP_ORDER = ["consistently_wrong", "sometimes_wrong", "consistently_right"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def assign_group(n: int, n_learners: int) -> str:
    if n == n_learners:
        return "consistently_wrong"
    elif n == 0:
        return "consistently_right"
    else:
        return "sometimes_wrong"


def mwu_annotation(a, b) -> str:
    """Mann-Whitney U p-value string for two groups (ignores NaNs)."""
    a = a.dropna()
    b = b.dropna()
    if len(a) < 3 or len(b) < 3:
        return ""
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    if p < 0.001:
        return "p<0.001"
    elif p < 0.01:
        return f"p={p:.3f}"
    elif p < 0.05:
        return f"p={p:.3f}"
    else:
        return f"p={p:.2f} (ns)"


def save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def strip_violin(ax, data, x_col, y_col, palette, order, title, ylabel):
    """Combined violin + strip plot on a given Axes."""
    sns.violinplot(data=data, x=x_col, y=y_col, hue=x_col, palette=palette,
                   order=order, ax=ax, inner=None, alpha=0.35, linewidth=0.8,
                   legend=False)
    sns.stripplot(data=data, x=x_col, y=y_col, hue=x_col, palette=palette,
                  order=order, ax=ax, size=4, jitter=True,
                  alpha=0.75, linewidth=0.3, edgecolor="gray", legend=False)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)


# ── Main ─────────────────────────────────────────────────────────────────────


def resolve_failure_csv(failure_csv: Path) -> Path:
    """
    If the given path exists, use it directly.
    Otherwise, search the same directory for the most recent failure_overlap_*.csv.
    """
    if failure_csv.exists():
        return failure_csv

    search_dir = failure_csv.parent
    candidates = sorted(search_dir.glob("failure_overlap_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No failure_overlap_*.csv found in '{search_dir}'.\n"
            "Run ensemble_sandbox.py first to generate it."
        )
    chosen = candidates[-1]   # most recent by filename (timestamps sort lexicographically)
    print(f"[INFO] --failure-csv not found; auto-selecting most recent: {chosen.name}")
    return chosen


def main(failure_csv: Path, gdpa1_csv: Path, out_dir: Path, cdr_csv: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[INFO] Loading data...")
    failure_csv = resolve_failure_csv(failure_csv)
    failure_df  = pd.read_csv(failure_csv)
    gdpa1_df    = pd.read_csv(gdpa1_csv)

    print(f"  failure_overlap rows : {len(failure_df)}")
    print(f"  GDPa1 rows          : {len(gdpa1_df)}")

    # Infer error/pred columns for summaries only
    flag_cols   = [c for c in failure_df.columns if c.endswith("_high_err")]
    abserr_cols = [c for c in failure_df.columns if c.endswith("_abserr")]
    pred_cols   = [c for c in failure_df.columns if c.endswith("_pred")]

    print(f"  Flag columns detected: {len(flag_cols)}")
    print("  Unique n_models_high_err values:", sorted(failure_df["n_models_high_err"].dropna().unique()))

    # Compute mean abs error across learners per antibody
    failure_df["mean_abserr"] = failure_df[abserr_cols].mean(axis=1)
    failure_df["std_abserr"]  = failure_df[abserr_cols].std(axis=1)

    # Use the actual max count present in the file as the all-models-failed threshold
    max_failures = int(failure_df["n_models_high_err"].max())
    print(f"  Max failures observed in file: {max_failures}")

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

    # ── Merge onto GDPa1 ─────────────────────────────────────────────────────
    # Drop y_true from failure_df (already in gdpa1_df as Titer) to avoid collision
    failure_cols_to_keep = (
        ["antibody_name", "n_models_high_err", "failure_group", "mean_abserr", "std_abserr"]
        + abserr_cols + pred_cols + flag_cols
    )
    failure_slim = failure_df[failure_cols_to_keep]

    merged = gdpa1_df.merge(failure_slim, on="antibody_name", how="left")
    print(f"  Merged shape: {merged.shape}")

    # Deduplicate columns that may arise from overlapping merge keys
    merged = merged.loc[:, ~merged.columns.duplicated()]
    print(f"  Columns after dedup: {merged.shape[1]}")

    # Re-detect flag cols after dedup (source of "6 learners" if a stale col crept in)
    flag_cols   = [c for c in merged.columns if c.endswith("_high_err")]
    abserr_cols = [c for c in merged.columns if c.endswith("_abserr")]
    pred_cols   = [c for c in merged.columns if c.endswith("_pred")]
    n_learners  = len(flag_cols)
    print(f"  Base learners (post-dedup): {n_learners}")

    # ── Optionally merge CDR features ────────────────────────────────────────
    cdr_cols = []
    if cdr_csv is not None and Path(cdr_csv).exists():
        cdr_df = pd.read_csv(cdr_csv)
        cdr_feature_cols = [c for c in cdr_df.columns if c != "antibody_name"]
        cdr_df = cdr_df.rename(columns={c: f"{c}_CDR_feature" for c in cdr_feature_cols})
        before_n = len(merged)
        merged = merged.merge(cdr_df, on="antibody_name", how="left")
        merged = merged.loc[:, ~merged.columns.duplicated()]
        cdr_cols = [c for c in merged.columns if c.endswith("_CDR_feature")]
        print(f"  CDR features merged: {len(cdr_cols)} cols | rows: {before_n} -> {len(merged)}")
    elif cdr_csv is not None:
        print(f"  [WARN] --cdr-csv path not found: {cdr_csv}. Skipping CDR analysis.")

    # ── Save enriched dataframe ───────────────────────────────────────────────
    save_path = out_dir / "gdpa1_with_failure_flags.csv"
    merged.to_csv(save_path, index=False)
    print(f"  Saved enriched dataframe -> {save_path.name}")

    # Only plot rows that have predictions (n=239; 7 have no Titer)
    plot_df = merged[merged["n_models_high_err"].notna()].copy()
    plot_df["n_models_high_err"] = plot_df["n_models_high_err"].astype(int)

    group_counts = plot_df["failure_group"].value_counts()
    print("\n  Failure group counts:")
    for g in GROUP_ORDER:
        print(f"    {g:<25} : {group_counts.get(g, 0)}")

    # ── Plot 01: Titer by group ───────────────────────────────────────────────
    print("\n[PLOT 01] Titer by failure group...")
    fig, ax = plt.subplots(figsize=(7, 5))
    strip_violin(ax, plot_df, "failure_group", "Titer", PALETTE, GROUP_ORDER,
                 "Titer Distribution by Failure Group", "Titer (mg/L)")

    # Annotate MWU p-values: wrong vs right
    wrong = plot_df[plot_df["failure_group"] == "consistently_wrong"]["Titer"]
    right = plot_df[plot_df["failure_group"] == "consistently_right"]["Titer"]
    ann   = mwu_annotation(wrong, right)
    ax.text(0.5, 0.97, f"Wrong vs Right: {ann}", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="dimgray")

    save(fig, out_dir, "01_titer_by_group.png")

    # ── Plot 02: Key assay readouts by group ──────────────────────────────────
    print("[PLOT 02] Assay readouts by group...")
    assay_pairs = [
        ("HIC",           "HIC Score"),
        ("HAC",           "HAC Score"),
        ("SEC %Monomer",  "SEC %Monomer"),
        ("AC-SINS_pH6.0", "AC-SINS pH 6.0"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Key Assay Readouts by Failure Group", fontweight="bold", fontsize=13)

    for ax, (col, label) in zip(axes.flat, assay_pairs):
        if col not in plot_df.columns:
            ax.set_visible(False)
            continue
        strip_violin(ax, plot_df, "failure_group", col, PALETTE, GROUP_ORDER, label, label)
        w = plot_df[plot_df["failure_group"] == "consistently_wrong"][col]
        r = plot_df[plot_df["failure_group"] == "consistently_right"][col]
        ann = mwu_annotation(w, r)
        ax.set_title(f"{label}\n(Wrong vs Right: {ann})", fontweight="bold", fontsize=10)

    fig.tight_layout()
    save(fig, out_dir, "02_assay_by_group.png")

    # ── Plot 03: Thermal stability by group ───────────────────────────────────
    print("[PLOT 03] Thermal stability by group...")
    thermal_cols = [c for c in ["Tonset", "Tm1", "Tm2"] if c in plot_df.columns]
    fig, axes = plt.subplots(1, len(thermal_cols), figsize=(5 * len(thermal_cols), 5))
    if len(thermal_cols) == 1:
        axes = [axes]
    fig.suptitle("Thermal Stability by Failure Group", fontweight="bold")

    for ax, col in zip(axes, thermal_cols):
        strip_violin(ax, plot_df, "failure_group", col, PALETTE, GROUP_ORDER, col, f"{col} (°C)")
        w = plot_df[plot_df["failure_group"] == "consistently_wrong"][col]
        r = plot_df[plot_df["failure_group"] == "consistently_right"][col]
        ann = mwu_annotation(w, r)
        ax.set_title(f"{col}\nWrong vs Right: {ann}", fontweight="bold", fontsize=10)

    fig.tight_layout()
    save(fig, out_dir, "03_thermal_by_group.png")

    # ── Plot 04: n_models_high_err vs true Titer ──────────────────────────────
    print("[PLOT 04] n_models_high_err vs Titer...")
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        plot_df["Titer"], plot_df["n_models_high_err"],
        c=plot_df["n_models_high_err"],
        cmap="RdYlGn_r", vmin=0, vmax=n_learners,
        alpha=0.7, edgecolors="gray", linewidths=0.4, s=55
    )
    plt.colorbar(sc, ax=ax, label="# Models Flagging High Error")
    ax.set_xlabel("True Titer (mg/L)")
    ax.set_ylabel("N Models High Error")
    ax.set_title("Number of Models with High Error vs True Titer", fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Spearman rho annotation
    from scipy.stats import spearmanr
    mask = plot_df[["Titer", "n_models_high_err"]].notna().all(axis=1)
    rho, pval = spearmanr(plot_df.loc[mask, "Titer"], plot_df.loc[mask, "n_models_high_err"])
    ax.text(0.02, 0.97, f"Spearman ρ = {rho:.3f}  (p={pval:.3f})",
            transform=ax.transAxes, va="top", fontsize=9, color="dimgray")

    save(fig, out_dir, "04_n_high_err_vs_titer.png")

    # ── Plot 05: Mean abs error vs true Titer ────────────────────────────────
    print("[PLOT 05] Mean abs error vs Titer...")
    fig, ax = plt.subplots(figsize=(7, 5))
    for grp in GROUP_ORDER:
        sub = plot_df[plot_df["failure_group"] == grp]
        ax.scatter(sub["Titer"], sub["mean_abserr"],
                   label=grp, color=PALETTE[grp],
                   alpha=0.7, edgecolors="gray", linewidths=0.3, s=50)

    ax.set_xlabel("True Titer (mg/L)")
    ax.set_ylabel("Mean Abs Error Across Learners (mg/L)")
    ax.set_title("Mean Prediction Error vs True Titer", fontweight="bold")
    ax.legend(title="Group", fontsize=9)

    rho2, pval2 = spearmanr(
        plot_df["Titer"].dropna(),
        plot_df.loc[plot_df["Titer"].notna(), "mean_abserr"]
    )
    ax.text(0.02, 0.97, f"Spearman ρ = {rho2:.3f}  (p={pval2:.3f})",
            transform=ax.transAxes, va="top", fontsize=9, color="dimgray")

    save(fig, out_dir, "05_mean_abserr_vs_titer.png")

    # ── Plot 06: Subtype composition by group ─────────────────────────────────
    print("[PLOT 06] Subtype breakdown by group...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Antibody Subtype Composition by Failure Group", fontweight="bold")

    for ax, col, label in zip(axes, ["hc_subtype", "lc_subtype"], ["HC Subtype", "LC Subtype"]):
        ct = (plot_df.groupby(["failure_group", col])
                     .size()
                     .reset_index(name="count"))
        ct["pct"] = ct.groupby("failure_group")["count"].transform(lambda x: x / x.sum() * 100)
        pivot = ct.pivot(index="failure_group", columns=col, values="pct").reindex(GROUP_ORDER).fillna(0)
        pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% within group")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title=col, fontsize=8)

    fig.tight_layout()
    save(fig, out_dir, "06_subtype_breakdown.png")

    # ── Plot 07: Clinical trial phase by group ────────────────────────────────
    print("[PLOT 07] Clinical phase by group...")
    phase_col = "highest_clinical_trial_asof_feb2025"
    if phase_col in plot_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        ct = (plot_df.groupby(["failure_group", phase_col])
                     .size()
                     .reset_index(name="count"))
        ct["pct"] = ct.groupby("failure_group")["count"].transform(lambda x: x / x.sum() * 100)
        pivot = ct.pivot(index="failure_group", columns=phase_col, values="pct").reindex(GROUP_ORDER).fillna(0)
        pivot.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", linewidth=0.5)
        ax.set_title("Clinical Trial Phase by Failure Group", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% within group")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Phase", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        save(fig, out_dir, "07_clinical_phase_breakdown.png")

    # ── Plot 08: Fold composition of hard antibodies ──────────────────────────
    print("[PLOT 08] Fold composition...")
    # The fold column from failure_overlap is named "fold"; after merge with gdpa1
    # it may be present as "fold" or shadowed — find it defensively
    fold_col_name = "fold" if "fold" in plot_df.columns else None
    if fold_col_name is None:
        print("  [SKIP] No 'fold' column found in merged dataframe — skipping plot 08")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ct = (plot_df.groupby([fold_col_name, "failure_group"])
                     .size()
                     .reset_index(name="count"))
        pivot = ct.pivot(index=fold_col_name, columns="failure_group", values="count").fillna(0)
        pivot = pivot.reindex(columns=[c for c in GROUP_ORDER if c in pivot.columns])
        pivot.plot(kind="bar", ax=ax,
                   color=[PALETTE[c] for c in pivot.columns],
                   edgecolor="white", linewidth=0.5)
        ax.set_title("Failure Group Composition per Fold", fontweight="bold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("# Antibodies")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Group", fontsize=9)
        fig.tight_layout()
        save(fig, out_dir, "08_fold_composition.png")

    # ── Plot 09: Mean abs error distribution + top antibodies (hard + borderline) ─
    print("[PLOT 09] Error distribution + top antibodies...")
    problem_df = (plot_df[plot_df["failure_group"].isin(["consistently_wrong", "sometimes_wrong"])]
                  .sort_values("n_models_high_err", ascending=False))

    if len(problem_df) > 0 and len(abserr_cols) > 0:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(9, 8.5), gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.32}
        )

        err_series = problem_df["mean_abserr"].dropna()
        if len(err_series) > 0:
            use_kde = len(err_series) >= 8
            sns.histplot(
                err_series,
                kde=use_kde,
                ax=ax0,
                color="#c0392b",
                edgecolor="white",
                alpha=0.85,
                stat="count",
            )
            med = float(err_series.median())
            ax0.axvline(med, color="#2c3e50", ls="--", lw=1.2, label=f"median = {med:.0f} mg/L")
            ax0.set_xlabel("Mean absolute error across learners (mg/L)")
            ax0.set_ylabel("Antibodies")
            ax0.set_title(
                "Distribution of mean absolute error\n(consistently wrong + sometimes wrong)",
                fontweight="bold",
            )
            ax0.legend(frameon=False, loc="upper right")

        top_n = min(25, len(problem_df))
        top = (
            problem_df.nlargest(top_n, "mean_abserr")
            .sort_values("mean_abserr", ascending=True)
        )
        ax1.barh(
            top["antibody_name"],
            top["mean_abserr"],
            color="#e67e22",
            edgecolor="white",
            linewidth=0.45,
        )
        ax1.set_xlabel("Mean absolute error across learners (mg/L)")
        ax1.set_ylabel("")
        ax1.set_title(f"Top {top_n} antibodies by mean absolute error", fontweight="bold")

        fig.suptitle(
            "Problem antibodies: error spread and worst cases",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        save(fig, out_dir, "09_error_top_distribution.png")


    # ── Plot 10: CDR feature distributions by failure group ──────────────────
    # Only runs if --cdr-csv was provided and CDR columns are present in plot_df
    cdr_cols_in_plot = [c for c in plot_df.columns if c.endswith("_CDR_feature")]
    print(f"Total CDR feature columns detected: {len(cdr_cols_in_plot)}")
    print("Sample CDR columns:", cdr_cols_in_plot[:5])
    if cdr_cols_in_plot:
        print("[PLOT 10] CDR features by failure group...")

        from scipy.stats import mannwhitneyu as _mwu

        # Compute MWU p-value and median difference (wrong - right) for each feature
        grp_wrong = plot_df[plot_df["failure_group"] == "consistently_wrong"]
        grp_right = plot_df[plot_df["failure_group"] == "consistently_right"]

        print("grp_wrong rows:", len(grp_wrong))
        print("grp_right rows:", len(grp_right))
        print(plot_df["failure_group"].value_counts(dropna=False))

        for col in cdr_cols_in_plot[:10]:
            print(f"\nColumn: {col}")
            print("  wrong non-null:", grp_wrong[col].notna().sum())
            print("  right non-null:", grp_right[col].notna().sum())

        print("grp_wrong rows:", len(grp_wrong))
        print("grp_right rows:", len(grp_right))
        print("failure_group value counts:")
        print(plot_df["failure_group"].value_counts(dropna=False))
        for col in cdr_cols_in_plot[:5]:
            print(f"\nColumn: {col}")
            print("  wrong non-null:", grp_wrong[col].notna().sum())
            print("  right non-null:", grp_right[col].notna().sum())
            print("  wrong sample:", grp_wrong[col].dropna().head().tolist())
            print("  right sample:", grp_right[col].dropna().head().tolist())
            
        feature_stats = []
        for col in cdr_cols_in_plot:
            w = grp_wrong[col] #.dropna()
            r = grp_right[col] #.dropna()
            if len(w) < 2 or len(r) < 2:
                continue
            _, p = _mwu(w, r, alternative="two-sided")
            median_diff = w.median() - r.median()
            feature_stats.append({
                "feature": col.replace("_CDR_feature", ""),
                "col":     col,
                "p_value": p,
                "median_diff_wrong_minus_right": median_diff,
                "median_wrong": w.median(),
                "median_right": r.median(),
            })
        print("Feature stats length:", len(feature_stats))


        feat_df = pd.DataFrame(feature_stats)
        if feat_df.empty:
            print("[ERROR] No valid CDR features found for MWU comparison.")
            print("Check grouped row counts and CDR non-null counts.")
            return

        feat_df = feat_df.sort_values("p_value").reset_index(drop=True)

        if feat_df.empty:
            print("[ERROR] No valid CDR features found for MWU comparison.")
            print("Check CDR merge, column names, and missing values.")
            return

        feat_df = feat_df.sort_values("p_value").reset_index(drop=True)
 
        # Print top findings
        print("\n  Top CDR features differentiating consistently-wrong vs right (MWU):")
        print(feat_df.head(15)[["feature", "p_value", "median_diff_wrong_minus_right",
                                  "median_wrong", "median_right"]].to_string(index=False))

        # Save feature stats table
        feat_df.to_csv(out_dir / "cdr_feature_group_stats.csv", index=False)
        print("  Saved: cdr_feature_group_stats.csv")

        # Plot top N most significant features as strip+violin small multiples
        top_n   = min(16, len(feat_df))
        top_features = feat_df.head(top_n)["col"].tolist()

        ncols = 4
        nrows = int(np.ceil(top_n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 3.8, nrows * 3.5),
                                  constrained_layout=True)
        fig.suptitle(
            "Top CDR Features: Consistently Wrong vs Right\n"
            "(sorted by Mann-Whitney U p-value, wrong vs right only)",
            fontweight="bold", fontsize=12
        )

        two_group_order   = ["consistently_wrong", "consistently_right"]
        two_group_palette = {k: PALETTE[k] for k in two_group_order}

        for ax, col in zip(axes.flat, top_features):
            row = feat_df[feat_df["col"] == col].iloc[0]
            sub = plot_df[plot_df["failure_group"].isin(two_group_order)][
                ["failure_group", col]
            ].dropna()

            sns.violinplot(data=sub, x="failure_group", y=col,
                           hue="failure_group", palette=two_group_palette,
                           order=two_group_order, ax=ax,
                           inner=None, alpha=0.35, linewidth=0.8, legend=False)
            sns.stripplot(data=sub, x="failure_group", y=col,
                          hue="failure_group", palette=two_group_palette,
                          order=two_group_order, ax=ax,
                          size=4, jitter=True, alpha=0.75,
                          linewidth=0.3, edgecolor="gray", legend=False)

            p_str = f"p={row['p_value']:.3f}" if row["p_value"] >= 0.001 else "p<0.001"
            ax.set_title(f"{row['feature']}\n{p_str}", fontsize=9, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(["Wrong", "Right"], fontsize=8)

        # Hide unused axes
        for ax in axes.flat[top_n:]:
            ax.set_visible(False)

        save(fig, out_dir, "10_cdr_features_by_group.png")
    else:
        print("[SKIP] Plot 10: no CDR feature columns found. Pass --cdr-csv to enable.")

    # ── Summary printout ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY: Consistently Wrong Antibodies")
    print("=" * 55)
    # fold column may be named differently after merge — detect it
    fold_col_name = "fold" if "fold" in plot_df.columns else None
    id_cols = ["antibody_name", "Titer", "n_models_high_err", "mean_abserr"]
    if fold_col_name:
        id_cols.insert(2, fold_col_name)
    assay_extras = [c for c in ["HIC", "HAC", "SEC %Monomer", "AC-SINS_pH6.0", "hc_subtype", "lc_subtype"]
                    if c in plot_df.columns]
    wrong_df = plot_df[plot_df["failure_group"] == "consistently_wrong"][
        id_cols + assay_extras
    ].sort_values("mean_abserr", ascending=False)

    print(wrong_df.to_string(index=False))

    print("\n" + "=" * 55)
    print("GROUP MEDIANS (numeric assay readouts)")
    print("=" * 55)
    numeric_assays = [c for c in
        ["Titer", "HIC", "HAC", "SEC %Monomer", "AC-SINS_pH6.0", "Tonset", "Tm1", "Tm2", "mean_abserr"]
        if c in plot_df.columns]
    medians = plot_df.groupby("failure_group")[numeric_assays].median().reindex(GROUP_ORDER)
    print(medians.round(2).to_string())

    print(f"\n[DONE] All plots saved to: {out_dir.resolve()}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--failure-csv", required=True,
                        help="Path to failure_overlap_*.csv from ensemble_sandbox.py")
    parser.add_argument("--gdpa1-csv",   required=True,
                        help="Path to GDPa1_246 IgGs_cleaned.csv")
    parser.add_argument("--out-dir",     default="../../data/modeling/failure_eda",
                        help="Output directory for plots and enriched CSV")
    parser.add_argument("--cdr-csv",     default=None,
                        help="Path to cdr_features_titer.csv (enables plot 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        failure_csv=Path(args.failure_csv),
        gdpa1_csv=Path(args.gdpa1_csv),
        out_dir=Path(args.out_dir),
        cdr_csv=Path(args.cdr_csv) if args.cdr_csv else None,
    )