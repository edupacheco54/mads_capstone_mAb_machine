"""
CDR-region decomposed biophysical features + liability motif counts
Spearman correlation vs Titer (GDPa1 dataset)

Code by Claude AI Sonnet 4.6 Extended | Prompts by Allen Chezick

Purpose
-------
This script computes a 62-feature CDR biophysical descriptor matrix from
AHO-aligned antibody sequences and then reports Spearman rank correlations
between each feature and the expression titer (HEK293 transient production).

The feature set covers:
  - Per-CDR length, GRAVY, hydrophobic fraction, net charge,
    deamidation / isomerization / oxidation motif counts  (×6 CDRs)
  - CDR-H3-specific aromatic fraction and N-glycosylation sequon count
  - Whole-domain VH / VL liability summaries (deamidation, isomerization,
    oxidation, glycosylation, hydrophobic fraction, GRAVY, net charge)
  - Cross-chain composite counts (total_deamid, total_oxidn) and
    unpaired-cysteine flags

AHO alignment boundaries (0-indexed Python slices)
---------------------------------------------------
  Heavy:  CDR-H1 [25:42]   CDR-H2 [56:76]   CDR-H3 [106:138]
  Light:  CDR-L1 [25:42]   CDR-L2 [56:72]   CDR-L3 [106:138]

Output (stdout)
---------------
  - Feature matrix dimensions
  - Full ranked Spearman table (|ρ| descending)
  - TOP 10 by |ρ|
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import re

# ---------------------------------------------------------------------------
# Amino acid physicochemical look-up tables
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity scale (J. Mol. Biol. 1982, 157:105-132).
# Used to compute the GRAVY index: sum(KD[aa]) / len(seq).
# Higher = more hydrophobic.
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

# Binary hydrophobic set — aliphatics + aromatics; C excluded to avoid
# confounding with disulfide cysteines.
HYDROPHOBIC = set('AILVMFYW')

# Aromatics only.  CDR-H3 aromatic content is linked to hydrophobic
# surface patches and elevated non-specific binding risk.
AROMATIC = set('FYW')

# Simple pH-7.4 charge assignments: +1 for K/R, -1 for D/E.
# H is excluded (pKa ~6.0 → mostly uncharged at pH 7.4).
POSITIVE = set('KR')   # +1 at pH 7.4
NEGATIVE = set('DE')   # -1 at pH 7.4


# ---------------------------------------------------------------------------
# CDR boundary definitions  (AHO numbering, 0-indexed Python half-open slices)
# ---------------------------------------------------------------------------

# Boundaries validated against AHO consensus gap-density profiles.
# CDR-H3 spans the widest window (32 columns) due to its extreme length
# variability — it is the primary determinant of antigen specificity and
# is the most developability-critical loop.
CDR_SLICES = {
    'H1': (25, 42), 'H2': (56, 76), 'H3': (106, 138),
    'L1': (25, 42), 'L2': (56, 72), 'L3': (106, 138),
}

# Maps each CDR label to the aligned-sequence column it is extracted from.
REGION_CHAIN = {
    'H1': 'heavy', 'H2': 'heavy', 'H3': 'heavy',
    'L1': 'light', 'L2': 'light', 'L3': 'light',
}


def extract_cdr(aligned_seq, cdr_name):
    """Slice a CDR from an AHO-aligned sequence and remove gap characters.

    Parameters
    ----------
    aligned_seq : str
        Full fixed-width AHO-aligned variable domain (gaps as '-').
    cdr_name : str
        One of 'H1', 'H2', 'H3', 'L1', 'L2', 'L3'.

    Returns
    -------
    str
        Ungapped CDR residue string; length varies per antibody.
    """
    s, e = CDR_SLICES[cdr_name]
    return aligned_seq[s:e].replace('-', '')


# ---------------------------------------------------------------------------
# Per-region biochemical descriptors
# ---------------------------------------------------------------------------

def gravy(seq):
    """Mean Kyte-Doolittle hydrophobicity (GRAVY score).

    Returns np.nan for empty sequences rather than raising ZeroDivisionError,
    so NaN-aware downstream routines (Spearman filter, PyCaret imputer) can
    handle missing data cleanly.
    """
    if not seq:
        return np.nan
    return sum(KD.get(aa, 0) for aa in seq) / len(seq)


def hyd_frac(seq):
    """Fraction of residues belonging to the HYDROPHOBIC set (A,I,L,V,M,F,Y,W).

    Complements GRAVY with a binary signal that is less sensitive to
    outlier residue scores (e.g., extreme I/V vs. extreme R/K values).
    Returns np.nan for empty input.
    """
    if not seq:
        return np.nan
    return sum(aa in HYDROPHOBIC for aa in seq) / len(seq)


def aromatic_frac(seq):
    """Fraction of aromatic residues (F, Y, W) in the sequence.

    Elevated aromatic content in CDR-H3 is a known predictor of HIC
    retention (hydrophobic interaction chromatography) and polyreactivity.
    Only computed for CDR-H3 in this pipeline.
    Returns np.nan for empty input.
    """
    if not seq:
        return np.nan
    return sum(aa in AROMATIC for aa in seq) / len(seq)


def net_charge(seq):
    """Approximate net charge at pH 7.4 (K,R = +1; D,E = -1).

    Simplified electrostatic model capturing the dominant charge signal
    relevant to AC-SINS (acidic charge-induced aggregation) and HAC
    (hydrophobic / charge affinity chromatography) assays.
    Returns np.nan for empty input.
    """
    if not seq:
        return np.nan
    return sum(1 if aa in POSITIVE else -1 if aa in NEGATIVE else 0 for aa in seq)


def cdr_length(seq):
    """Number of residues in the (ungapped) CDR sequence.

    CDR-H3 length is one of the strongest sequence-level predictors of
    developability risk: longer loops tend to be more aggregation-prone and
    harder to express at high titers.
    """
    return len(seq)


# ---------------------------------------------------------------------------
# Liability motif counters (operate on raw ungapped sequences)
# ---------------------------------------------------------------------------
# Chemical degradation motifs in CDRs are a primary concern during
# biologics development because PTMs in the antigen-binding region can
# directly reduce potency and complicate manufacturing batch release.

def count_deamidation(seq):
    """NG, NS, NA dipeptides — asparagine deamidation risk.

    Deamidation converts Asn → Asp/iso-Asp, adding a negative charge and
    a backbone modification.  NG is the fastest-reacting motif (t1/2 ~1 day
    at pH 7.4, 37°C); NS and NA are elevated but slower.
    """
    return len(re.findall(r'N[GSA]', seq))


def count_isomerization(seq):
    """DG, DS dipeptides — aspartate isomerization risk.

    Asp isomerization proceeds via a succinimide intermediate, creating
    iso-Asp and altering CDR backbone geometry.  DG is the highest-risk
    motif; DS is also flagged as elevated risk.
    """
    return len(re.findall(r'D[GS]', seq))


def count_oxidation(seq):
    """Tryptophan (W) + methionine (M) residue count — oxidation risk.

    W and M are the two residues most susceptible to reactive-oxygen-species
    (ROS) oxidation under forced-oxidation stress and normal storage conditions.
    CDR-exposed tryptophans are particularly prone because of solvent access.
    """
    return seq.count('W') + seq.count('M')


def count_glycosylation(seq):
    """N-X-S/T sequon count (X ≠ P) — N-linked glycosylation risk.

    N-linked glycosylation at CDR Asn residues introduces glycan heterogeneity
    in the antigen-binding site, potentially reducing binding affinity and
    complicating batch-to-batch comparability.
    Proline at position X prevents glycosylation (rigid conformation).
    """
    return len(re.findall(r'N[^P][ST]', seq))


def count_unpaired_cys(seq):
    """Return 1 if Cys count is odd (possible free thiol), else 0.

    An odd number of cysteines implies at least one unpaired thiol, which can
    participate in unwanted disulfide scrambling, covalent aggregation, or
    conjugation artefacts during manufacture.
    """
    c_count = seq.count('C')
    return c_count % 2   # 1 = potential free thiol; 0 = all Cys paired


# ---------------------------------------------------------------------------
# Per-antibody feature extraction
# ---------------------------------------------------------------------------

def build_features(row):
    """Compute the full CDR biophysical feature vector for one antibody row.

    Feature groups
    --------------
    1. Per-CDR descriptors (6 loops × 7 = 42 features):
       length, GRAVY, hydrophobic fraction, net charge,
       deamidation, isomerization, oxidation motif counts.
    2. CDR-H3 extras (2): aromatic fraction, glycosylation sequon count.
    3. Whole-domain VH/VL summaries (2 chains × 7 = 14 features):
       deamidation, isomerization, oxidation, glycosylation totals,
       plus hydrophobic fraction, GRAVY, and net charge.
    4. Cross-chain composites (4): total_deamid, total_oxidn,
       VH_unpaired_cys, VL_unpaired_cys.

    Parameters
    ----------
    row : pd.Series
        One row of the GDPa1 DataFrame; requires
        'heavy_aligned_aho', 'light_aligned_aho',
        'vh_protein_sequence', 'vl_protein_sequence'.

    Returns
    -------
    dict
        Flat {feature_name: value} dict.  NaN for any empty/missing sequence.
    """
    heavy_aln = row['heavy_aligned_aho']
    light_aln = row['light_aligned_aho']

    feats = {}

    # --- Group 1: per-CDR descriptors ---
    for cdr in ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']:
        # Select aligned sequence for the correct chain
        chain_aln = heavy_aln if REGION_CHAIN[cdr] == 'heavy' else light_aln
        seq = extract_cdr(chain_aln, cdr)   # ungapped CDR residues

        feats[f'cdr{cdr}_len']      = cdr_length(seq)
        feats[f'cdr{cdr}_gravy']    = gravy(seq)
        feats[f'cdr{cdr}_hyd_frac'] = hyd_frac(seq)
        feats[f'cdr{cdr}_charge']   = net_charge(seq)
        feats[f'cdr{cdr}_deamid']   = count_deamidation(seq)
        feats[f'cdr{cdr}_isom']     = count_isomerization(seq)
        feats[f'cdr{cdr}_oxidn']    = count_oxidation(seq)

    # --- Group 2: CDR-H3-specific extras ---
    # H3 gets two additional features because its composition and length
    # dominate the developability landscape in the GDPa1 dataset.
    h3_seq = extract_cdr(heavy_aln, 'H3')
    feats['cdrH3_aromatic_frac'] = aromatic_frac(h3_seq)
    feats['cdrH3_glycan_sequon'] = count_glycosylation(h3_seq)

    # --- Group 3: whole variable-domain VH / VL summaries ---
    # Full VH/VL sequences (ungapped) capture framework-region liabilities
    # that CDR slices alone miss.
    vh = row['vh_protein_sequence'] if pd.notna(row['vh_protein_sequence']) else ''
    vl = row['vl_protein_sequence'] if pd.notna(row['vl_protein_sequence']) else ''

    for label, seq in [('VH', vh), ('VL', vl)]:
        feats[f'{label}_deamid_total'] = count_deamidation(seq)
        feats[f'{label}_isom_total']   = count_isomerization(seq)
        feats[f'{label}_oxidn_total']  = count_oxidation(seq)
        feats[f'{label}_glycan_total'] = count_glycosylation(seq)
        feats[f'{label}_hyd_frac']     = hyd_frac(seq)
        feats[f'{label}_gravy']        = gravy(seq)
        feats[f'{label}_charge']       = net_charge(seq)

    # --- Group 4: cross-chain composites ---
    # Additive sums give models a pre-computed total liability burden signal.
    feats['total_deamid']    = feats['VH_deamid_total'] + feats['VL_deamid_total']
    feats['total_oxidn']     = feats['VH_oxidn_total']  + feats['VL_oxidn_total']
    feats['VH_unpaired_cys'] = count_unpaired_cys(vh)
    feats['VL_unpaired_cys'] = count_unpaired_cys(vl)

    return feats


# ---------------------------------------------------------------------------
# Spearman correlation analysis
# ---------------------------------------------------------------------------

def spearman_vs_titer(df_feat, titer, label=''):
    """Compute pairwise Spearman ρ between each feature column and titer.

    Only features with at least 30 non-null paired observations are included
    (MIN_N=30) to avoid reporting meaningless correlations from near-empty
    columns.

    Parameters
    ----------
    df_feat : pd.DataFrame
        Feature matrix (rows = antibodies, columns = features).
    titer : pd.Series
        Expression titer values, aligned to df_feat index.
    label : str
        Optional label for display purposes (not used in output DataFrame).

    Returns
    -------
    pd.DataFrame
        Columns: feature, rho, p_value, n.
        Sorted by |ρ| descending so the most predictive features appear first.
    """
    results = []
    for col in df_feat.columns:
        x = df_feat[col]
        # Only correlate where both feature and titer are non-null
        mask = x.notna() & titer.notna()
        if mask.sum() < 30:  # skip features with too few valid pairs
            continue
        rho, p = spearmanr(x[mask], titer[mask])
        results.append({'feature': col, 'rho': rho, 'p_value': p, 'n': mask.sum()})

    return pd.DataFrame(results).sort_values('rho', key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Load the GDPa1 cleaned dataset (246 clinical-stage IgGs)
    df = pd.read_csv('/mnt/project/gdpa1_246_iggs_cleaned.csv')

    print(f"Dataset: {len(df)} antibodies, {df['Titer'].notna().sum()} with Titer")
    print()

    # --- Build the feature matrix ---
    # Rows with missing alignment data are represented as empty dicts
    # so that the resulting DataFrame keeps the same index as df.
    feat_rows = []
    for _, row in df.iterrows():
        if pd.isna(row['heavy_aligned_aho']) or pd.isna(row['light_aligned_aho']):
            feat_rows.append({})   # → all-NaN row; preserves index alignment
            continue
        feat_rows.append(build_features(row))

    feat_df = pd.DataFrame(feat_rows, index=df.index)
    print(f"Feature matrix: {feat_df.shape[1]} features\n")
    print("Features:", list(feat_df.columns))
    print()

    # --- Spearman correlation vs. Titer ---
    results = spearman_vs_titer(feat_df, df['Titer'])

    # Print ranked table with significance stars
    print("=" * 70)
    print("Spearman ρ vs Titer — sorted by |ρ|")
    print("=" * 70)
    print(f"{'Feature':<35} {'rho':>7}  {'p_value':>12}  {'n':>4}  sig")
    print("-" * 70)
    for _, r in results.iterrows():
        # Significance stars follow conventional thresholds
        sig = ('***' if r['p_value'] < 0.001 else
               '** ' if r['p_value'] < 0.01  else
               '*  ' if r['p_value'] < 0.05  else
               '   ')
        print(f"{r['feature']:<35} {r['rho']:>7.4f}  {r['p_value']:>12.4e}  "
              f"{int(r['n']):>4}  {sig}")

    print()
    print("=== TOP 10 ABSOLUTE ===")
    print(results.head(10).to_string(index=False))