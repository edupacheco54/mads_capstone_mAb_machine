"""Utility functions for GDPa1 CDR feature engineering.

This module refactors the user's earlier CDR feature scripts into reusable
functions so they can be imported by modeling pipelines.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

HYDROPHOBIC = set("AILVMFYW")
AROMATIC = set("FYW")
POSITIVE = set("KR")
NEGATIVE = set("DE")

CDR_SLICES = {
    "H1": (25, 42), "H2": (56, 76), "H3": (106, 138),
    "L1": (25, 42), "L2": (56, 72), "L3": (106, 138),
}

REGION_CHAIN = {
    "H1": "heavy", "H2": "heavy", "H3": "heavy",
    "L1": "light", "L2": "light", "L3": "light",
}


def extract_cdr(aligned_seq: str, cdr_name: str) -> str:
    s, e = CDR_SLICES[cdr_name]
    return aligned_seq[s:e].replace("-", "")


def gravy(seq: str) -> float:
    if not seq:
        return np.nan
    return sum(KD.get(aa, 0.0) for aa in seq) / len(seq)


def hyd_frac(seq: str) -> float:
    if not seq:
        return np.nan
    return sum(aa in HYDROPHOBIC for aa in seq) / len(seq)


def aromatic_frac(seq: str) -> float:
    if not seq:
        return np.nan
    return sum(aa in AROMATIC for aa in seq) / len(seq)


def net_charge(seq: str) -> float:
    if not seq:
        return np.nan
    return sum(1 if aa in POSITIVE else -1 if aa in NEGATIVE else 0 for aa in seq)


def cdr_length(seq: str) -> int:
    return len(seq)


def count_deamidation(seq: str) -> int:
    return len(re.findall(r"N[GSA]", seq))


def count_isomerization(seq: str) -> int:
    return len(re.findall(r"D[GS]", seq))


def count_oxidation(seq: str) -> int:
    return seq.count("W") + seq.count("M")


def count_glycosylation(seq: str) -> int:
    return len(re.findall(r"N[^P][ST]", seq))


def count_unpaired_cys(seq: str) -> int:
    return seq.count("C") % 2


def build_features(row: pd.Series) -> Dict[str, float]:
    heavy_aln = row["heavy_aligned_aho"]
    light_aln = row["light_aligned_aho"]

    feats: Dict[str, float] = {}

    for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        chain_aln = heavy_aln if REGION_CHAIN[cdr] == "heavy" else light_aln
        seq = extract_cdr(chain_aln, cdr)
        feats[f"cdr{cdr}_len"] = cdr_length(seq)
        feats[f"cdr{cdr}_gravy"] = gravy(seq)
        feats[f"cdr{cdr}_hyd_frac"] = hyd_frac(seq)
        feats[f"cdr{cdr}_charge"] = net_charge(seq)
        feats[f"cdr{cdr}_deamid"] = count_deamidation(seq)
        feats[f"cdr{cdr}_isom"] = count_isomerization(seq)
        feats[f"cdr{cdr}_oxidn"] = count_oxidation(seq)

    h3_seq = extract_cdr(heavy_aln, "H3")
    feats["cdrH3_aromatic_frac"] = aromatic_frac(h3_seq)
    feats["cdrH3_glycan_sequon"] = count_glycosylation(h3_seq)

    vh = row["vh_protein_sequence"] if pd.notna(row.get("vh_protein_sequence")) else ""
    vl = row["vl_protein_sequence"] if pd.notna(row.get("vl_protein_sequence")) else ""

    for label, seq in [("VH", vh), ("VL", vl)]:
        feats[f"{label}_deamid_total"] = count_deamidation(seq)
        feats[f"{label}_isom_total"] = count_isomerization(seq)
        feats[f"{label}_oxidn_total"] = count_oxidation(seq)
        feats[f"{label}_glycan_total"] = count_glycosylation(seq)
        feats[f"{label}_hyd_frac"] = hyd_frac(seq)
        feats[f"{label}_gravy"] = gravy(seq)
        feats[f"{label}_charge"] = net_charge(seq)

    feats["total_deamid"] = feats["VH_deamid_total"] + feats["VL_deamid_total"]
    feats["total_oxidn"] = feats["VH_oxidn_total"] + feats["VL_oxidn_total"]
    feats["VH_unpaired_cys"] = count_unpaired_cys(vh)
    feats["VL_unpaired_cys"] = count_unpaired_cys(vl)
    return feats


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    required = {"heavy_aligned_aho", "light_aligned_aho", "vh_protein_sequence", "vl_protein_sequence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input dataframe is missing required columns for CDR features: "
            + ", ".join(missing)
        )

    feat_rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
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
    feature_df = build_feature_matrix(df)
    out = feature_df.copy()
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
    exclude = set(exclude_cols) | {target_col}
    numeric_cols = modeling_df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]
