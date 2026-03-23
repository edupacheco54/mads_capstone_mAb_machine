"""
CDR-region decomposed biophysical features — multi-assay Spearman battery
GDPa1 dataset (246 IgGs)

AHO alignment boundaries (0-indexed Python slices):
  Heavy:  CDR-H1 [25:42]  CDR-H2 [56:76]  CDR-H3 [106:138]
  Light:  CDR-L1 [25:42]  CDR-L2 [56:72]  CDR-L3 [106:138]

Assays tested:
  Titer, Purity, SEC %Monomer, SMAC, HIC, HAC,
  PR_CHO, PR_Ova, AC-SINS_pH6.0, AC-SINS_pH7.4,
  Tonset, Tm1, Tm2

Outputs:
  - Per-assay ranked Spearman table (console)
  - cdr_multiassay_spearman_results.csv (all results)
  - cdr_multiassay_top10_heatmap.png (|rho| heatmap, top features)
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "gdpa1_246_iggs_cleaned.csv"   # update path if needed

ASSAYS = [
    "Titer", "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
    "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
    "Tonset", "Tm1", "Tm2",
]

MIN_N = 30   # minimum non-null pairs to include a correlation

# ── Amino acid lookup tables ─────────────────────────────────────────────────

KD = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

HYDROPHOBIC = set("AILVMFYW")
AROMATIC    = set("FYW")
POSITIVE    = set("KR")
NEGATIVE    = set("DE")

# ── CDR region boundaries ────────────────────────────────────────────────────

CDR_SLICES = {
    "H1": (25, 42), "H2": (56, 76),  "H3": (106, 138),
    "L1": (25, 42), "L2": (56, 72),  "L3": (106, 138),
}

REGION_CHAIN = {
    "H1": "heavy", "H2": "heavy", "H3": "heavy",
    "L1": "light", "L2": "light", "L3": "light",
}


# ── Sequence utilities ───────────────────────────────────────────────────────

def extract_cdr(aligned_seq, cdr_name):
    s, e = CDR_SLICES[cdr_name]
    return aligned_seq[s:e].replace("-", "")

def gravy(seq):
    if not seq:
        return np.nan
    return sum(KD.get(aa, 0) for aa in seq) / len(seq)

def hyd_frac(seq):
    if not seq:
        return np.nan
    return sum(aa in HYDROPHOBIC for aa in seq) / len(seq)

def aromatic_frac(seq):
    if not seq:
        return np.nan
    return sum(aa in AROMATIC for aa in seq) / len(seq)

def net_charge(seq):
    if not seq:
        return np.nan
    return sum(1 if aa in POSITIVE else -1 if aa in NEGATIVE else 0 for aa in seq)

def cdr_length(seq):
    return len(seq)


# ── Liability motif counters ─────────────────────────────────────────────────

def count_deamidation(seq):
    """NG, NS, NA dipeptides — prone to deamidation"""
    return len(re.findall(r"N[GSA]", seq))

def count_isomerization(seq):
    """DG, DS dipeptides — prone to Asp isomerization"""
    return len(re.findall(r"D[GS]", seq))

def count_oxidation(seq):
    """W and M residues — prone to oxidation"""
    return seq.count("W") + seq.count("M")

def count_glycosylation(seq):
    """N-X-S/T sequons where X != P — N-linked glycosylation risk"""
    return len(re.findall(r"N[^P][ST]", seq))

def count_unpaired_cys(seq):
    """Odd C count suggests unpaired disulfide"""
    return seq.count("C") % 2


# ── Feature extraction ───────────────────────────────────────────────────────

def build_features(row):
    heavy_aln = row["heavy_aligned_aho"]
    light_aln = row["light_aligned_aho"]
    feats = {}

    # Per-CDR descriptors
    for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        chain_aln = heavy_aln if REGION_CHAIN[cdr] == "heavy" else light_aln
        seq = extract_cdr(chain_aln, cdr)

        feats[f"cdr{cdr}_len"]      = cdr_length(seq)
        feats[f"cdr{cdr}_gravy"]    = gravy(seq)
        feats[f"cdr{cdr}_hyd_frac"] = hyd_frac(seq)
        feats[f"cdr{cdr}_charge"]   = net_charge(seq)
        feats[f"cdr{cdr}_deamid"]   = count_deamidation(seq)
        feats[f"cdr{cdr}_isom"]     = count_isomerization(seq)
        feats[f"cdr{cdr}_oxidn"]    = count_oxidation(seq)

    # CDR-H3 extras
    h3_seq = extract_cdr(heavy_aln, "H3")
    feats["cdrH3_aromatic_frac"]  = aromatic_frac(h3_seq)
    feats["cdrH3_glycan_sequon"]  = count_glycosylation(h3_seq)

    # Full VH / VL domain summaries
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

    # Composite cross-chain features
    feats["total_deamid"]    = feats["VH_deamid_total"] + feats["VL_deamid_total"]
    feats["total_oxidn"]     = feats["VH_oxidn_total"]  + feats["VL_oxidn_total"]
    feats["VH_unpaired_cys"] = count_unpaired_cys(vh)
    feats["VL_unpaired_cys"] = count_unpaired_cys(vl)

    return feats


# ── Spearman battery ─────────────────────────────────────────────────────────

def run_spearman_battery(feat_df, df, assays, min_n=MIN_N):
    all_rows = []
    for assay in assays:
        if assay not in df.columns:
            continue
        target = df[assay]
        for col in feat_df.columns:
            x = feat_df[col]
            mask = x.notna() & target.notna()
            if mask.sum() < min_n:
                continue
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
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def print_results(results_df):
    for assay in results_df["assay"].unique():
        sub = (results_df[results_df["assay"] == assay]
               .sort_values("rho", key=abs, ascending=False))
        n_val = int(sub["n"].iloc[0])
        print(f"\n{'='*65}")
        print(f"  {assay}  (n={n_val})")
        print(f"{'='*65}")
        print(f"  {'Feature':<35} {'rho':>7}  {'p_value':>12}  sig")
        print(f"  {'-'*58}")
        for _, r in sub.head(10).iterrows():
            print(f"  {r['feature']:<35} {r['rho']:>7.4f}  {r['p_value']:>12.4e}  "
                  f"{sig_stars(r['p_value'])}")


# ── Heatmap ──────────────────────────────────────────────────────────────────

def plot_heatmap(results_df, top_n=20, out_path="cdr_multiassay_top20_heatmap.png"):
    # Select features with at least one |rho| >= 0.15 across any assay
    pivot = results_df.pivot_table(index="feature", columns="assay", values="rho")
    max_abs = pivot.abs().max(axis=1)
    selected = max_abs[max_abs >= 0.15].nlargest(top_n).index
    pivot_sub = pivot.loc[selected]

    # Column order: group by biological theme
    col_order = [c for c in
                 ["Titer", "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
                  "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
                  "Tonset", "Tm1", "Tm2"]
                 if c in pivot_sub.columns]
    pivot_sub = pivot_sub[col_order]

    fig, ax = plt.subplots(figsize=(13, max(6, len(selected) * 0.42)))
    sns.heatmap(
        pivot_sub,
        cmap="RdBu_r",
        center=0,
        vmin=-0.75, vmax=0.75,
        annot=True, fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
        ax=ax,
    )
    ax.set_title(
        "CDR biophysical features × assay outcomes\nSpearman ρ  (features with max |ρ| ≥ 0.15 shown)",
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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} antibodies from {DATA_PATH}")

    # Build feature matrix
    feat_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("heavy_aligned_aho")) or pd.isna(row.get("light_aligned_aho")):
            feat_rows.append({})
            continue
        feat_rows.append(build_features(row))

    feat_df = pd.DataFrame(feat_rows, index=df.index)
    print(f"Feature matrix: {feat_df.shape[0]} antibodies × {feat_df.shape[1]} features")

    # Run battery
    results = run_spearman_battery(feat_df, df, ASSAYS)

    # Console output
    print_results(results)

    # Save full results CSV
    csv_path = "cdr_multiassay_spearman_results.csv"
    results.sort_values(["assay", "rho"], key=lambda x: x if x.name != "rho" else x.abs(),
                        ascending=[True, False]).to_csv(csv_path, index=False)
    print(f"\nFull results saved → {csv_path}")

    # Heatmap
    plot_heatmap(results)
