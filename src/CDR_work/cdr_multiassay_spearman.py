"""
CDR-region decomposed biophysical features — multi-assay Spearman battery
GDPa1 dataset (246 IgGs)

Code by Claude AI Sonnet 4.6 Extended | Prompts by Allen Chezick

Purpose
-------
Extends cdr_features_titer.py to correlate the full CDR feature matrix
against all 13 developability assays in the GDPa1 panel.  The goal is to
identify which sequence descriptors carry signal for each assay and to
understand the three biological regimes observed in prior analysis:

  1. Charge-dominated signals (HAC, AC-SINS pH 6.0 / 7.4, polyreactivity)
     → driven by CDR-L3 charge and VH/VL net charge.
  2. CDR-H3 hydrophobicity / length signals (HIC, SMAC)
     → CDR-H3 GRAVY, hyd_frac, aromatic_frac, and length.
  3. CDR-L3 length + charge hybrid (Titer, SEC %Monomer)
     → mixed composition of charge and size features.

AHO alignment boundaries (0-indexed Python slices)
---------------------------------------------------
  Heavy:  CDR-H1 [25:42]   CDR-H2 [56:76]   CDR-H3 [106:138]
  Light:  CDR-L1 [25:42]   CDR-L2 [56:72]   CDR-L3 [106:138]

Assays tested
-------------
  Titer, Purity, SEC %Monomer, SMAC, HIC, HAC,
  PR_CHO, PR_Ova, AC-SINS_pH6.0, AC-SINS_pH7.4,
  Tonset, Tm1, Tm2

Outputs
-------
  - Per-assay ranked Spearman table (stdout, top 10 per assay)
  - cdr_multiassay_spearman_results.csv  (all feature × assay correlations)
  - cdr_multiassay_top20_heatmap.png     (|ρ| heatmap for top features)


  =================================================================
  Titer  (n=239)
=================================================================
  Feature                                 rho       p_value  sig
  ----------------------------------------------------------
  cdrL3_len                            0.2438    1.4074e-04  ***
  VH_charge                            0.1747    6.7729e-03  ** 
  VL_hyd_frac                         -0.1579    1.4523e-02  *  
  cdrH3_charge                         0.1494    2.0850e-02  *  
  cdrL3_hyd_frac                       0.1427    2.7431e-02  *  
  cdrH2_charge                         0.1327    4.0378e-02  *  
  cdrL3_gravy                          0.1318    4.1748e-02  *  
  cdrH2_hyd_frac                       0.1274    4.9109e-02  *  
  cdrL2_gravy                         -0.1257    5.2204e-02     
  VL_glycan_total                     -0.1245    5.4590e-02     


"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Filename of the GDPa1 cleaned dataset as it exists in the repo.
# Note the space in the filename — quote or escape it if referencing from shell.
# Repo layout:  capstone699_repo/data/raw/GDPa1_246 IgGs_cleaned.csv
DATA_FILENAME = "GDPa1_246 IgGs_cleaned.csv"  # note: space in filename is intentional


def find_data_file(filename: str = DATA_FILENAME) -> Path:
    """Search a prioritised list of candidate locations for the GDPa1 CSV.

    The canonical location in the shared repo is:
        <repo_root>/data/raw/GDPa1_246 IgGs_cleaned.csv

    Locations are checked in this order:
      1. <repo_root>/data/raw/   — resolved by walking up from the script's
                                   own directory until a 'data/raw' sibling is
                                   found; works for any contributor who has
                                   cloned the repo with the standard layout.
      2. The current working directory  — works when running from data/raw/
         directly.
      3. The directory containing this script  — fallback if the CSV was
         placed next to the script during development.
      4. Absolute fallback for allen on tensorbook2  — in case cwd is
         unexpected and the upward walk fails for any reason.
      5. Claude.ai project mount  — for when the script is run in the
         Claude sandbox environment.

    To add a new contributor's machine, append a Path to the candidates list.

    Parameters
    ----------
    filename : str
        Bare filename to search for (default DATA_FILENAME).

    Returns
    -------
    pathlib.Path
        Resolved, existing path to the CSV file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found in any candidate location, with a
        diagnostic message listing every path that was tried.
    """
    script_dir = Path(__file__).resolve().parent

    # --- Strategy 1: walk up from the script toward the repo root and look
    # for a data/raw/ subdirectory at each level.
    # src/CDR_work/ → src/ → capstone699_repo/ ← repo root has data/raw/
    repo_raw_candidates = [
        ancestor / "data" / "raw"
        for ancestor in [script_dir, *script_dir.parents]
    ]

    candidates = [
        *repo_raw_candidates,                                       # 1. repo data/raw/ (any depth)
        Path.cwd(),                                                 # 2. cwd
        script_dir,                                                 # 3. next to script
        # --- per-contributor absolute fallbacks ---
        Path.home() / "mads_UMICH" / "capstone699_repo" / "data" / "raw",  # allen / tensorbook2
        Path("/mnt/project"),                                       # Claude.ai project mount
    ]

    tried = []
    for directory in candidates:
        candidate = directory / filename
        tried.append(candidate)
        if candidate.exists():
            print(f"[find_data_file] Found data at: {candidate}")
            return candidate

    tried_str = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"Could not find '{filename}' in any of the following locations:\n"
        f"  {tried_str}\n\n"
        f"Expected location: <repo_root>/data/raw/{filename}\n"
        f"To fix: run the script from the repo root, or add your machine's\n"
        f"data path to the candidates list in find_data_file()."
    )

# Full list of GDPa1 developability assay columns to test.
# Notes on leakage risk for titer prediction:
#   - Purity (ELISA bioassay) and SEC %Monomer are causally downstream of
#     expression and process steps — including them as predictors of Titer
#     risks data leakage.
#   - SMAC (hydrophobic interaction resin) reflects process-confounded
#     information and should be excluded from titer modeling features.
#   - All 13 are included here for the correlation study only (no models).
ASSAYS = [
    "Titer", "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
    "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
    "Tonset", "Tm1", "Tm2",
]

# Minimum number of non-null (feature, assay) pairs required to report a
# Spearman correlation.  Below 30 the rank correlation becomes unreliable
# and p-values are poorly calibrated.
MIN_N = 30

# ---------------------------------------------------------------------------
# Amino acid physicochemical look-up tables
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity scale (1982).  Used for GRAVY computation.
KD = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Binary hydrophobic set — aliphatics + aromatics; C excluded to avoid
# confounding with disulfide cysteines.
HYDROPHOBIC = set("AILVMFYW")

# Aromatics only; elevated CDR-H3 aromatic content → HIC retention risk.
AROMATIC = set("FYW")

# Simplified pH-7.4 charge model: +1 per K/R, -1 per D/E.
POSITIVE = set("KR")   # +1 at pH 7.4
NEGATIVE = set("DE")   # -1 at pH 7.4

# ---------------------------------------------------------------------------
# CDR region boundaries  (AHO numbering, 0-indexed Python half-open slices)
# ---------------------------------------------------------------------------

# Validated against AHO consensus alignment gap-density profiles.
# CDR-H3 spans 32 columns — the widest window — due to its extreme
# length variability (typically 3–28 residues in clinical antibodies).
CDR_SLICES = {
    "H1": (25, 42), "H2": (56, 76),  "H3": (106, 138),
    "L1": (25, 42), "L2": (56, 72),  "L3": (106, 138),
}

# Maps CDR label → aligned-sequence column in the source DataFrame.
REGION_CHAIN = {
    "H1": "heavy", "H2": "heavy", "H3": "heavy",
    "L1": "light", "L2": "light", "L3": "light",
}


# ---------------------------------------------------------------------------
# Sequence utility functions
# ---------------------------------------------------------------------------

def extract_cdr(aligned_seq, cdr_name):
    """Slice and ungap a CDR from an AHO-aligned variable-domain sequence.

    Parameters
    ----------
    aligned_seq : str
        Fixed-width AHO-aligned sequence with gaps as '-'.
    cdr_name : str
        One of 'H1', 'H2', 'H3', 'L1', 'L2', 'L3'.

    Returns
    -------
    str
        Ungapped CDR residues; variable length per antibody.
    """
    s, e = CDR_SLICES[cdr_name]
    return aligned_seq[s:e].replace("-", "")


def gravy(seq):
    """Mean Kyte-Doolittle hydrophobicity (GRAVY score).
    Returns np.nan for empty sequences.
    """
    if not seq:
        return np.nan
    return sum(KD.get(aa, 0) for aa in seq) / len(seq)


def hyd_frac(seq):
    """Fraction of residues in HYDROPHOBIC set (A,I,L,V,M,F,Y,W).
    Returns np.nan for empty sequences.
    """
    if not seq:
        return np.nan
    return sum(aa in HYDROPHOBIC for aa in seq) / len(seq)


def aromatic_frac(seq):
    """Fraction of aromatic residues (F, Y, W).
    Computed only for CDR-H3 where aromatic content is most developability-relevant.
    Returns np.nan for empty sequences.
    """
    if not seq:
        return np.nan
    return sum(aa in AROMATIC for aa in seq) / len(seq)


def net_charge(seq):
    """Approximate net charge at pH 7.4 (K,R=+1; D,E=-1).
    Returns np.nan for empty sequences.
    """
    if not seq:
        return np.nan
    return sum(1 if aa in POSITIVE else -1 if aa in NEGATIVE else 0 for aa in seq)


def cdr_length(seq):
    """Number of residues in an ungapped CDR sequence.

    CDR-H3 length is a primary predictor of aggregation propensity
    and expression difficulty in the GDPa1 dataset.
    """
    return len(seq)


# ---------------------------------------------------------------------------
# Liability motif counters
# ---------------------------------------------------------------------------
# Chemical degradation hotspots in CDRs are especially risky because
# PTMs in the antigen-binding site can directly reduce potency.

def count_deamidation(seq):
    """NG, NS, NA dipeptides — asparagine deamidation risk.

    Deamidation converts Asn → Asp/iso-Asp, introducing a negative charge.
    NG is fastest; NS and NA are elevated but slower risk.
    """
    return len(re.findall(r"N[GSA]", seq))


def count_isomerization(seq):
    """DG, DS dipeptides — aspartate isomerization risk.

    Proceeds via a succinimide intermediate; creates iso-Asp and may distort
    CDR conformation, reducing binding affinity.
    """
    return len(re.findall(r"D[GS]", seq))


def count_oxidation(seq):
    """Trp (W) + Met (M) count — oxidation-prone residues.

    Both are susceptible to ROS under forced-oxidation stress and storage.
    Solvent-exposed CDR tryptophans are particularly at risk.
    """
    return seq.count("W") + seq.count("M")


def count_glycosylation(seq):
    """N-X-S/T sequon count (X ≠ P) — N-linked glycosylation risk.

    Glycans in CDRs introduce heterogeneity and can reduce antigen-binding
    affinity; Pro at X prevents glycosylation (rigid backbone).
    """
    return len(re.findall(r"N[^P][ST]", seq))


def count_unpaired_cys(seq):
    """Return 1 if Cys count is odd (potential free thiol), else 0.

    A free thiol can cause disulfide scrambling, aggregation, or conjugation
    artefacts.  This is a binary flag rather than a raw count.
    """
    return seq.count("C") % 2


# ---------------------------------------------------------------------------
# Per-antibody feature extraction
# ---------------------------------------------------------------------------

def build_features(row):
    """Compute the full CDR biophysical descriptor vector for one antibody.

    Feature groups
    --------------
    1. Per-CDR (6 loops × 7 = 42 features):
       length, GRAVY, hydrophobic fraction, net charge,
       deamidation, isomerization, oxidation counts.
    2. CDR-H3 extras (2): aromatic fraction, glycosylation sequon count.
    3. Full VH/VL domain summaries (2 × 7 = 14 features):
       deamidation, isomerization, oxidation, glycosylation totals;
       hydrophobic fraction, GRAVY, net charge.
    4. Cross-chain composites (4):
       total_deamid, total_oxidn, VH_unpaired_cys, VL_unpaired_cys.

    Parameters
    ----------
    row : pd.Series
        Requires 'heavy_aligned_aho', 'light_aligned_aho',
        'vh_protein_sequence', 'vl_protein_sequence'.

    Returns
    -------
    dict
        {feature_name: scalar} flat dict; NaN for empty/missing sequences.
    """
    heavy_aln = row["heavy_aligned_aho"]
    light_aln = row["light_aligned_aho"]
    feats = {}

    # --- Group 1: per-CDR descriptors ---
    for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        chain_aln = heavy_aln if REGION_CHAIN[cdr] == "heavy" else light_aln
        seq = extract_cdr(chain_aln, cdr)  # ungapped CDR residues

        feats[f"cdr{cdr}_len"]      = cdr_length(seq)
        feats[f"cdr{cdr}_gravy"]    = gravy(seq)
        feats[f"cdr{cdr}_hyd_frac"] = hyd_frac(seq)
        feats[f"cdr{cdr}_charge"]   = net_charge(seq)
        feats[f"cdr{cdr}_deamid"]   = count_deamidation(seq)
        feats[f"cdr{cdr}_isom"]     = count_isomerization(seq)
        feats[f"cdr{cdr}_oxidn"]    = count_oxidation(seq)

    # --- Group 2: CDR-H3 extras ---
    # Aromatic fraction and glycan sequons are specific to H3 because
    # this loop drives the most biologically distinct developability signals.
    h3_seq = extract_cdr(heavy_aln, "H3")
    feats["cdrH3_aromatic_frac"] = aromatic_frac(h3_seq)
    feats["cdrH3_glycan_sequon"] = count_glycosylation(h3_seq)

    # --- Group 3: full VH / VL domain summaries ---
    # Uses whole variable-domain protein sequences (not AHO-aligned) to
    # capture framework-region liabilities that CDR slices miss.
    vh = row["vh_protein_sequence"] if pd.notna(row["vh_protein_sequence"]) else ""
    vl = row["vl_protein_sequence"] if pd.notna(row["vl_protein_sequence"]) else ""

    for label, seq in [("VH", vh), ("VL", vl)]:
        feats[f"{label}_deamid_total"] = count_deamidation(seq)
        feats[f"{label}_isom_total"]   = count_isomerization(seq)
        feats[f"{label}_oxidn_total"]  = count_oxidation(seq)
        feats[f"{label}_glycan_total"] = count_glycosylation(seq)
        feats[f"{label}_hyd_frac"]     = hyd_frac(seq)
        feats[f"{label}_gravy"]        = gravy(seq)
        feats[f"{label}_charge"]       = net_charge(seq)

    # --- Group 4: cross-chain composites ---
    # Additive sums give models a pre-computed total-burden signal without
    # having to discover VH + VL addition themselves.
    feats["total_deamid"]    = feats["VH_deamid_total"] + feats["VL_deamid_total"]
    feats["total_oxidn"]     = feats["VH_oxidn_total"]  + feats["VL_oxidn_total"]
    feats["VH_unpaired_cys"] = count_unpaired_cys(vh)
    feats["VL_unpaired_cys"] = count_unpaired_cys(vl)

    return feats


# ---------------------------------------------------------------------------
# Spearman battery
# ---------------------------------------------------------------------------

def run_spearman_battery(feat_df, df, assays, min_n=MIN_N):
    """Compute Spearman ρ for every (feature, assay) pair that has ≥ min_n pairs.

    This is the core analysis function.  Each feature is tested against each
    assay independently.  No multiple-testing correction is applied here;
    p-values are included for reference but should be interpreted with caution
    given the ~800 simultaneous tests (62 features × 13 assays).

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix (antibodies × features).
    df : pd.DataFrame
        Source DataFrame containing raw assay columns.
    assays : list of str
        Assay column names to test (skips any absent from df.columns).
    min_n : int
        Minimum non-null paired observations required to compute ρ.

    Returns
    -------
    pd.DataFrame
        Long-format results with columns: assay, feature, rho, p_value, n.
    """
    all_rows = []
    for assay in assays:
        if assay not in df.columns:
            continue   # silently skip absent assay columns
        target = df[assay]
        for col in feat_df.columns:
            x = feat_df[col]
            # Build a mask for rows where both feature and assay are non-null
            mask = x.notna() & target.notna()
            if mask.sum() < min_n:
                continue  # skip if insufficient paired observations
            rho, p = spearmanr(x[mask], target[mask])
            all_rows.append({
                "assay":   assay,
                "feature": col,
                "rho":     rho,
                "p_value": p,
                "n":       mask.sum(),
            })
    return pd.DataFrame(all_rows)


def sig_stars(p):
    """Return significance star string for a p-value (conventional thresholds)."""
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def print_results(results_df):
    """Print the top-10 features by |ρ| for each assay to stdout.

    Only the top 10 per assay are displayed to keep the console output
    manageable; the full results are saved to CSV separately.
    """
    for assay in results_df["assay"].unique():
        sub = (results_df[results_df["assay"] == assay]
               .sort_values("rho", key=abs, ascending=False))
        n_val = int(sub["n"].iloc[0])   # sample size is same for all features/assay
        print(f"\n{'='*65}")
        print(f"  {assay}  (n={n_val})")
        print(f"{'='*65}")
        print(f"  {'Feature':<35} {'rho':>7}  {'p_value':>12}  sig")
        print(f"  {'-'*58}")
        for _, r in sub.head(10).iterrows():
            print(f"  {r['feature']:<35} {r['rho']:>7.4f}  {r['p_value']:>12.4e}  "
                  f"{sig_stars(r['p_value'])}")


# ---------------------------------------------------------------------------
# Heatmap visualization
# ---------------------------------------------------------------------------

def plot_heatmap(results_df, top_n=20, out_path="cdr_multiassay_top20_heatmap.png"):
    """Generate a Spearman ρ heatmap (features × assays) and save to disk.

    Feature selection: keep only features that reach |ρ| ≥ 0.15 in at least
    one assay, then take the top_n by maximum |ρ|.  This threshold (~weak
    correlation) is chosen to show features with any meaningful signal while
    filtering out pure noise.

    Columns are ordered by biological theme:
      Expression / purity → Aggregation → Hydrophobicity → Charge → Thermal

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-format output of run_spearman_battery().
    top_n : int
        Maximum number of features to display (default 20).
    out_path : str
        File path for the saved PNG.
    """
    # Pivot to (feature × assay) matrix of ρ values
    pivot = results_df.pivot_table(index="feature", columns="assay", values="rho")

    # Select features with any strong-enough signal across assays
    max_abs = pivot.abs().max(axis=1)
    selected = max_abs[max_abs >= 0.15].nlargest(top_n).index
    pivot_sub = pivot.loc[selected]

    # Reorder columns to group by biological theme for interpretability
    col_order = [c for c in
                 ["Titer", "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
                  "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
                  "Tonset", "Tm1", "Tm2"]
                 if c in pivot_sub.columns]
    pivot_sub = pivot_sub[col_order]

    # Symmetric diverging colour scale centred on ρ=0;
    # vmin/vmax set to ±0.75 to avoid colour saturation on the strongest signals
    fig, ax = plt.subplots(figsize=(13, max(6, len(selected) * 0.42)))
    sns.heatmap(
        pivot_sub,
        cmap="RdBu_r",         # red = positive ρ, blue = negative ρ
        center=0,
        vmin=-0.75, vmax=0.75,
        annot=True, fmt=".2f", # annotate each cell with the ρ value
        linewidths=0.4,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
        ax=ax,
    )
    ax.set_title(
        "CDR biophysical features × assay outcomes\n"
        "Spearman ρ  (features with max |ρ| ≥ 0.15 shown)",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nHeatmap saved → {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Locate and load the GDPa1 cleaned dataset (246 clinical-stage IgGs).
    # find_data_file() searches cwd, script directory, parent directories, and
    # known absolute contributor paths — so no hardcoded path is needed.
    data_path = find_data_file()
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} antibodies from {data_path}")

    # --- Build feature matrix ---
    # Rows with missing alignment data are empty dicts → all-NaN feature row,
    # preserving index alignment with df for downstream target joining.
    feat_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("heavy_aligned_aho")) or pd.isna(row.get("light_aligned_aho")):
            feat_rows.append({})
            continue
        feat_rows.append(build_features(row))

    feat_df = pd.DataFrame(feat_rows, index=df.index)
    print(f"Feature matrix: {feat_df.shape[0]} antibodies × {feat_df.shape[1]} features")

    # --- Run multi-assay Spearman battery ---
    results = run_spearman_battery(feat_df, df, ASSAYS)

    # --- Console output (top 10 per assay) ---
    print_results(results)

    # --- Save full results to CSV ---
    csv_path = "cdr_multiassay_spearman_results.csv"
    results.sort_values(
        ["assay", "rho"],
        key=lambda x: x if x.name != "rho" else x.abs(),
        ascending=[True, False],
    ).to_csv(csv_path, index=False)
    print(f"\nFull results saved → {csv_path}")

    # --- Generate |ρ| heatmap ---
    plot_heatmap(results)