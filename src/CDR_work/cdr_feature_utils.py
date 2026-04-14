"""Utility functions for GDPa1 CDR feature engineering.

Code by Claude AI Sonnet 4.6 Extended | Prompts by Allen Chezick

This module refactors the CDR feature scripts into reusable, importable
functions so that downstream modeling pipelines (PyCaret, sklearn, etc.)
can call `build_feature_matrix` or `build_modeling_table` directly without
copy-pasting sequence-processing logic.

Design notes
------------
- All sequence operations work on AHO-aligned strings (fixed-width, gap-padded).
  Gap characters ('-') are stripped before computing per-CDR descriptors so
  that, e.g., GRAVY reflects only the real residues present in each CDR.
- CDR boundary slices are 0-indexed Python half-open intervals validated
  against gap-density patterns in AHO consensus alignments.
- Feature naming convention:  cdr{LOOP}_{descriptor}  (e.g. cdrH3_gravy)
  for per-CDR features, and  {CHAIN}_{descriptor}_total  for whole-domain
  summaries (e.g. VH_deamid_total).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Amino acid physicochemical look-up tables
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity scale (J. Mol. Biol. 1982, 157:105-132).
# Higher values = more hydrophobic.  Used to compute the GRAVY index
# (Grand Average of hYdropathicity), which summarises the overall
# hydrophobic character of a sequence as a single scalar.
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Classic hydrophobic residue set — used for binary hydrophobic fraction.
# Includes aliphatics (A, I, L, V, M) and aromatics (F, Y, W).
# Note: C is excluded to avoid confounding with disulfide bridge cysteines.
HYDROPHOBIC = set("AILVMFYW")

# Aromatic residues only.  CDR-H3 aromatic content is associated with
# hydrophobic surface patches that elevate HIC retention and non-specific
# binding risk (relevant to AC-SINS, polyreactivity).
AROMATIC = set("FYW")

# Charged residue sets used for net charge calculation at pH 7.4.
# Simplified model: K and R each contribute +1, D and E each contribute -1.
# H is excluded because its pKa (~6.0) puts it mostly uncharged at pH 7.4.
POSITIVE = set("KR")  # +1 each at pH 7.4
NEGATIVE = set("DE")  # -1 each at pH 7.4

# ---------------------------------------------------------------------------
# CDR boundary definitions (AHO numbering scheme, 0-indexed Python slices)
# ---------------------------------------------------------------------------

# These half-open slice intervals were derived from AHO consensus alignment
# column ranges.  Boundaries were validated by comparing gap-density profiles
# against published AHO CDR positions.
#
# Heavy chain: CDR-H1 cols 25-41, CDR-H2 cols 56-75, CDR-H3 cols 106-137
# Light chain: CDR-L1 cols 25-41, CDR-L2 cols 56-71, CDR-L3 cols 106-137
#
# CDR-H3 spans the widest and most variable range (up to 32 alignment columns)
# because this loop has the highest sequence diversity and is the primary
# determinant of antigen specificity and many developability liabilities.
CDR_SLICES = {
    "H1": (25, 42), "H2": (56, 76), "H3": (106, 138),
    "L1": (25, 42), "L2": (56, 72), "L3": (106, 138),
}

# Maps each CDR loop name to the sequence column it originates from.
# Used internally by build_features() to pick the correct aligned sequence.
REGION_CHAIN = {
    "H1": "heavy", "H2": "heavy", "H3": "heavy",
    "L1": "light", "L2": "light", "L3": "light",
}


# ---------------------------------------------------------------------------
# Low-level sequence utilities
# ---------------------------------------------------------------------------

def extract_cdr(aligned_seq: str, cdr_name: str) -> str:
    """Slice a CDR region from an AHO-aligned sequence and strip gap characters.

    Parameters
    ----------
    aligned_seq : str
        Full AHO-aligned variable-domain sequence (fixed width, gaps as '-').
    cdr_name : str
        One of 'H1', 'H2', 'H3', 'L1', 'L2', 'L3'.

    Returns
    -------
    str
        Ungapped CDR amino acid sequence.  Length varies per antibody.
    """
    s, e = CDR_SLICES[cdr_name]
    return aligned_seq[s:e].replace("-", "")


def gravy(seq: str) -> float:
    """Mean Kyte-Doolittle hydrophobicity (GRAVY score) of a sequence.

    Returns np.nan for empty sequences so that NaN-aware Spearman / PyCaret
    routines can filter or impute rather than silently dividing by zero.

    Interpretation: positive GRAVY → net hydrophobic character (higher HIC
    retention risk); negative → net hydrophilic.
    """
    if not seq:
        return np.nan
    return sum(KD.get(aa, 0.0) for aa in seq) / len(seq)


def hyd_frac(seq: str) -> float:
    """Fraction of residues in HYDROPHOBIC set (A, I, L, V, M, F, Y, W).

    Complements GRAVY by using a binary classification instead of a
    continuous scale, which can be more robust to outlier residue scores.
    Returns np.nan for empty input.
    """
    if not seq:
        return np.nan
    return sum(aa in HYDROPHOBIC for aa in seq) / len(seq)


def aromatic_frac(seq: str) -> float:
    """Fraction of aromatic residues (F, Y, W) in a sequence.

    Aromatic content in CDR-H3 is specifically associated with
    hydrophobic surface patches linked to non-specific binding and
    elevated HIC retention.  Computed only for CDR-H3 in this pipeline.
    Returns np.nan for empty input.
    """
    if not seq:
        return np.nan
    return sum(aa in AROMATIC for aa in seq) / len(seq)


def net_charge(seq: str) -> float:
    """Approximate net charge at pH 7.4 assuming K,R=+1 and D,E=-1.

    This simplified model ignores histidine (pKa ~6.0, mostly uncharged
    at pH 7.4) and terminus contributions.  It captures the dominant
    electrostatic signal relevant to AC-SINS and HAC assays.
    Returns np.nan for empty input.
    """
    if not seq:
        return np.nan
    return sum(1 if aa in POSITIVE else -1 if aa in NEGATIVE else 0 for aa in seq)


def cdr_length(seq: str) -> int:
    """Number of residues in an ungapped CDR sequence.

    CDR length (especially CDR-H3) is among the strongest sequence-level
    predictors of both antigen-binding diversity and developability risk
    (longer H3 loops are more prone to aggregation and expression issues).
    """
    return len(seq)


# ---------------------------------------------------------------------------
# Chemical liability motif counters
# ---------------------------------------------------------------------------
# These regex-based counters flag sequence motifs known to undergo
# post-translational modifications (PTMs) or chemical degradation under
# typical biologics process and storage conditions.  Their presence in CDRs
# is of particular concern because modifications can directly alter
# antigen-binding affinity.

def count_deamidation(seq: str) -> int:
    """Count asparagine (N) deamidation risk motifs: NG, NS, NA.

    Deamidation converts Asn → Asp/iso-Asp, introducing a negative charge
    and a backbone rearrangement.  The rate is strongly context-dependent:
    NG and NS are the highest-risk dipeptides.  NA is included as a
    moderate-risk variant.  Pattern: N followed by G, S, or A.
    """
    return len(re.findall(r"N[GSA]", seq))


def count_isomerization(seq: str) -> int:
    """Count aspartate (D) isomerization risk motifs: DG, DS.

    Asp isomerization forms iso-Asp via a succinimide intermediate, altering
    backbone geometry in a manner that can disrupt CDR conformation and reduce
    potency.  DG is the fastest-isomerizing motif; DS is also elevated risk.
    """
    return len(re.findall(r"D[GS]", seq))


def count_oxidation(seq: str) -> int:
    """Count oxidation-prone residues: tryptophan (W) and methionine (M).

    Trp and Met are the two amino acids most susceptible to oxidation under
    typical forced-oxidation stress conditions.  W oxidation is particularly
    common in CDRs exposed to the solvent, while M oxidation is relevant
    in framework regions but can appear in CDRs too.
    """
    return seq.count("W") + seq.count("M")


def count_glycosylation(seq: str) -> int:
    """Count N-linked glycosylation sequons: N-X-S/T where X is not P.

    N-linked glycosylation at Asn in the N-X-S/T motif is a well-known
    CDR liability; glycans in antigen-binding regions can reduce potency,
    increase molecular weight heterogeneity, and complicate manufacturing.
    Proline at the X position is excluded because it prevents glycosylation.
    """
    return len(re.findall(r"N[^P][ST]", seq))


def count_unpaired_cys(seq: str) -> int:
    """Return 1 if Cys count is odd (suggesting an unpaired disulfide), else 0.

    An odd number of cysteines implies at least one free thiol, which can
    participate in disulfide scrambling, aggregation, or conjugation
    artefacts.  This is a binary flag rather than a count.
    """
    return seq.count("C") % 2


# ---------------------------------------------------------------------------
# Per-antibody feature construction
# ---------------------------------------------------------------------------

def build_features(row: pd.Series) -> Dict[str, float]:
    """Compute the full 62-feature CDR biophysical descriptor vector for one antibody.

    Feature groups produced
    -----------------------
    1. Per-CDR (×6 loops × 7 descriptors = 42 features):
       length, GRAVY, hydrophobic fraction, net charge,
       deamidation count, isomerization count, oxidation count.

    2. CDR-H3 extras (2 features):
       aromatic fraction, glycosylation sequon count.
       (H3-specific because its length and composition drive the most
       developability-relevant signals in the GDPa1 dataset.)

    3. Whole-domain VH / VL summaries (×2 chains × 7 descriptors = 14 features):
       deamidation total, isomerization total, oxidation total,
       glycosylation total, hydrophobic fraction, GRAVY, net charge.
       (Captures framework-region liabilities that CDR slices miss.)

    4. Cross-chain composites (4 features):
       total_deamid, total_oxidn (VH + VL sums),
       VH_unpaired_cys, VL_unpaired_cys.

    Parameters
    ----------
    row : pd.Series
        One row of the GDPa1 dataframe; must contain
        'heavy_aligned_aho', 'light_aligned_aho',
        'vh_protein_sequence', 'vl_protein_sequence'.

    Returns
    -------
    Dict[str, float]
        Flat dict mapping feature names to scalar values.
        NaN is returned for any descriptor that cannot be computed
        (e.g., empty sequence after gap stripping).
    """
    heavy_aln = row["heavy_aligned_aho"]
    light_aln = row["light_aligned_aho"]

    feats: Dict[str, float] = {}

    # --- Group 1: per-CDR descriptors ---
    for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        # Select the correct aligned chain for this CDR
        chain_aln = heavy_aln if REGION_CHAIN[cdr] == "heavy" else light_aln
        seq = extract_cdr(chain_aln, cdr)  # ungapped CDR residues only

        feats[f"cdr{cdr}_len"]    = cdr_length(seq)
        feats[f"cdr{cdr}_gravy"]  = gravy(seq)
        feats[f"cdr{cdr}_hyd_frac"] = hyd_frac(seq)
        feats[f"cdr{cdr}_charge"] = net_charge(seq)
        feats[f"cdr{cdr}_deamid"] = count_deamidation(seq)
        feats[f"cdr{cdr}_isom"]   = count_isomerization(seq)
        feats[f"cdr{cdr}_oxidn"]  = count_oxidation(seq)

    # --- Group 2: CDR-H3-specific extras ---
    # CDR-H3 is computed separately because its aromatic content and
    # glycosylation sequons are developability signals important enough
    # to warrant dedicated features beyond the standard 7.
    h3_seq = extract_cdr(heavy_aln, "H3")
    feats["cdrH3_aromatic_frac"] = aromatic_frac(h3_seq)
    feats["cdrH3_glycan_sequon"] = count_glycosylation(h3_seq)

    # --- Group 3: whole-domain VH / VL liability summaries ---
    # Use the full variable-domain protein sequence (not AHO-aligned)
    # so that framework-region motifs are included.
    # Fall back to empty string if the field is NaN (some rows may be missing).
    vh = row["vh_protein_sequence"] if pd.notna(row.get("vh_protein_sequence")) else ""
    vl = row["vl_protein_sequence"] if pd.notna(row.get("vl_protein_sequence")) else ""

    for label, seq in [("VH", vh), ("VL", vl)]:
        feats[f"{label}_deamid_total"] = count_deamidation(seq)
        feats[f"{label}_isom_total"]   = count_isomerization(seq)
        feats[f"{label}_oxidn_total"]  = count_oxidation(seq)
        feats[f"{label}_glycan_total"] = count_glycosylation(seq)
        feats[f"{label}_hyd_frac"]     = hyd_frac(seq)
        feats[f"{label}_gravy"]        = gravy(seq)
        feats[f"{label}_charge"]       = net_charge(seq)

    # --- Group 4: cross-chain composite features ---
    # Simple additive sums allow a model to capture total liability burden
    # without needing to learn the VH + VL combination itself.
    feats["total_deamid"]    = feats["VH_deamid_total"] + feats["VL_deamid_total"]
    feats["total_oxidn"]     = feats["VH_oxidn_total"]  + feats["VL_oxidn_total"]
    feats["VH_unpaired_cys"] = count_unpaired_cys(vh)
    feats["VL_unpaired_cys"] = count_unpaired_cys(vl)

    return feats


# ---------------------------------------------------------------------------
# DataFrame-level wrappers
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply build_features() to every row of a GDPa1-format DataFrame.

    Rows missing either aligned sequence column are represented as all-NaN
    feature vectors so that the index alignment with the source DataFrame
    is preserved (important for downstream target joining).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'heavy_aligned_aho', 'light_aligned_aho',
        'vh_protein_sequence', 'vl_protein_sequence' columns.

    Returns
    -------
    pd.DataFrame
        Shape (len(df), n_features) with the same index as df.

    Raises
    ------
    ValueError
        If any required column is absent from df.
    """
    required = {"heavy_aligned_aho", "light_aligned_aho",
                "vh_protein_sequence", "vl_protein_sequence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input dataframe is missing required columns for CDR features: "
            + ", ".join(missing)
        )

    feat_rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        # Rows with missing alignment data yield an empty dict → all-NaN row
        if pd.isna(row["heavy_aligned_aho"]) or pd.isna(row["light_aligned_aho"]):
            feat_rows.append({})
            continue
        feat_rows.append(build_features(row))

    return pd.DataFrame(feat_rows, index=df.index)


def build_modeling_table(
    df: pd.DataFrame,
    target_col: str = "Titer",
    id_cols: Sequence[str] = ("antibody_id", "antibody_name"),
) -> pd.DataFrame:
    """Build a ready-to-model DataFrame: CDR features + optional IDs + target.

    Intended as the single entry point for generating the input to PyCaret
    or sklearn pipelines.  ID columns are carried through for traceability
    (e.g., for labelling holdout predictions) but are not numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        GDPa1-format source DataFrame.
    target_col : str
        Name of the assay column to use as prediction target (default 'Titer').
    id_cols : sequence of str
        Columns to copy into output for traceability; ignored if not present.

    Returns
    -------
    pd.DataFrame
        Feature columns + id columns (if present) + target column.
    """
    feature_df = build_feature_matrix(df)
    out = feature_df.copy()

    # Carry through any ID columns that exist in the source DataFrame
    for col in id_cols:
        if col in df.columns:
            out[col] = df[col]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in dataframe.")
    out[target_col] = df[target_col]

    return out


def get_numeric_feature_columns(
    modeling_df: pd.DataFrame,
    target_col: str = "Titer",
    exclude_cols: Iterable[str] = ("antibody_id", "antibody_name"),
) -> List[str]:
    """Return numeric column names suitable for use as model inputs.

    Filters out the target variable and any non-numeric identifier columns
    so the caller can pass the result directly to X = df[feature_cols].

    Parameters
    ----------
    modeling_df : pd.DataFrame
        Output of build_modeling_table().
    target_col : str
        Target column to exclude from features.
    exclude_cols : iterable of str
        Additional columns to exclude (e.g., string ID fields).

    Returns
    -------
    List[str]
        Ordered list of numeric feature column names.
    """
    exclude = set(exclude_cols) | {target_col}
    numeric_cols = modeling_df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]
