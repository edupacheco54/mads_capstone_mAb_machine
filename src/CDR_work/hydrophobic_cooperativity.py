from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict
import math
import numpy as np
import pandas as pd 
from scipy.stats import spearmanr 

# ----------------------------
# Core definitions / utilities
# ----------------------------

DEFAULT_HYDROPHOBIC = frozenset(list("AVILMFWY"))  # common hydrophobic set (C excluded)
DEFAULT_GAP = "-"

df = pd.read_csv("/home/allen/mads_UMICH/capstone699/GDPa1_246_IgG_cleaned.csv")
HYDROPHOBIC = set("AVILMFWY")
GAP = "-"

def _non_gap_indices(aligned_seq: str, gap: str = DEFAULT_GAP) -> np.ndarray:
    """Return indices (0-based) of positions in aligned_seq that are not gaps."""
    arr = np.frombuffer(aligned_seq.encode("ascii"), dtype="S1")
    return np.where(arr != gap.encode("ascii"))[0]


def _hydrophobic_indicator(
    aligned_seq: str,
    hydrophobic_set: Iterable[str] = DEFAULT_HYDROPHOBIC,
    gap: str = DEFAULT_GAP,
) -> np.ndarray:
    """
    Return an array h of length len(aligned_seq) with:
      h[i]=1 if aligned_seq[i] is hydrophobic amino acid, else 0.
    Gaps are always 0.
    """
    hyd = set(hydrophobic_set)
    out = np.zeros(len(aligned_seq), dtype=np.int8)
    for i, ch in enumerate(aligned_seq):
        if ch != gap and ch in hyd:
            out[i] = 1
    return out


def _pair_weight_sums(
    non_gap_idx: np.ndarray,
    radius: int,
    weight: str = "inv",  # "inv" => 1/d, "inv2" => 1/d^2, "one" => 1
) -> float:
    """
    Compute denominator: sum_{i in R} sum_{j in R, 1<=|i-j|<=radius} w(|i-j|).
    Used to normalize HCI_raw into [0,1]-ish scale comparable across sequences.
    """
    if radius <= 0 or non_gap_idx.size <= 1:
        return 0.0

    # For efficiency: use distance counts across index differences.
    # We'll count distances between all pairs within radius among non-gap indices.
    # Since aligned indices are on the same coordinate system, we can use a boolean mask.
    # O(n*radius) approach:
    idx_set = set(non_gap_idx.tolist())
    denom = 0.0
    for i in non_gap_idx:
        for d in range(1, radius + 1):
            j1 = i - d
            j2 = i + d
            if j1 in idx_set:
                denom += (1.0 / d) if weight == "inv" else (1.0 / (d * d) if weight == "inv2" else 1.0)
            if j2 in idx_set:
                denom += (1.0 / d) if weight == "inv" else (1.0 / (d * d) if weight == "inv2" else 1.0)
    return denom


def _w(d: int, weight: str) -> float:
    if weight == "inv":
        return 1.0 / d
    if weight == "inv2":
        return 1.0 / (d * d)
    if weight == "one":
        return 1.0
    raise ValueError(f"Unknown weight='{weight}'. Use 'inv', 'inv2', or 'one'.")


# ----------------------------
# HCI computations
# ----------------------------

@dataclass(frozen=True)
class HCIResult:
    hci_raw: float
    hci_norm: float
    hci_frac: float
    hci_z: Optional[float] = None
    null_mean: Optional[float] = None
    null_sd: Optional[float] = None


def hci_raw(
    aligned_seq: str,
    radius: int = 5,
    hydrophobic_set: Iterable[str] = DEFAULT_HYDROPHOBIC,
    gap: str = DEFAULT_GAP,
    weight: str = "inv",
) -> float:
    """
    Distance-weighted hydrophobic clustering score on an aligned sequence.

    HCI_raw = sum_{i in R} sum_{j in R, 1<=|i-j|<=radius} h_i h_j w(|i-j|)

    - R: non-gap aligned positions
    - h_i: 1 if hydrophobic residue at position i else 0
    - w(d): distance kernel (default 1/d)

    Notes:
    - Counts both (i,j) and (j,i). That's fine because normalization uses the same convention.
    - Gaps are ignored (treated as non-residues).
    """
    if radius <= 0:
        return 0.0

    non_gap_idx = _non_gap_indices(aligned_seq, gap=gap)
    if non_gap_idx.size <= 1:
        return 0.0

    idx_set = set(non_gap_idx.tolist())
    h = _hydrophobic_indicator(aligned_seq, hydrophobic_set=hydrophobic_set, gap=gap)

    score = 0.0
    for i in non_gap_idx:
        if h[i] == 0:
            continue
        for d in range(1, radius + 1):
            wgt = _w(d, weight)
            j1 = i - d
            j2 = i + d
            if j1 in idx_set and h[j1] == 1:
                score += wgt
            if j2 in idx_set and h[j2] == 1:
                score += wgt
    return score


def hci_normalized(
    aligned_seq: str,
    radius: int = 5,
    hydrophobic_set: Iterable[str] = DEFAULT_HYDROPHOBIC,
    gap: str = DEFAULT_GAP,
    weight: str = "inv",
    eps: float = 1e-9,
) -> Tuple[float, float, float]:
    """
    Compute:
      - HCI_raw (see hci_raw)
      - HCI_norm: normalized by the maximum possible weighted neighbor-pair sum (depends on gaps/length)
      - HCI_frac: normalized by (hydrophobic_count^2) to reduce dependence on overall hydrophobic content

    Returns (hci_raw, hci_norm, hci_frac).
    """
    raw = hci_raw(
        aligned_seq,
        radius=radius,
        hydrophobic_set=hydrophobic_set,
        gap=gap,
        weight=weight,
    )

    non_gap_idx = _non_gap_indices(aligned_seq, gap=gap)
    denom_pairs = _pair_weight_sums(non_gap_idx, radius=radius, weight=weight)

    h = _hydrophobic_indicator(aligned_seq, hydrophobic_set=hydrophobic_set, gap=gap)
    hyd_count = float(h.sum())

    hci_norm = raw / (denom_pairs + eps)
    hci_frac = raw / ((hyd_count * hyd_count) + eps)

    return raw, hci_norm, hci_frac


def hci_zscore(
    aligned_seq: str,
    radius: int = 5,
    hydrophobic_set: Iterable[str] = DEFAULT_HYDROPHOBIC,
    gap: str = DEFAULT_GAP,
    weight: str = "inv",
    n_perm: int = 200,
    seed: Optional[int] = 0,
    eps: float = 1e-9,
) -> HCIResult:
    """
    Compute hydrophobic cooperativity index as a Z-score vs a composition-preserving null.

    Procedure:
      1) Compute observed HCI_raw on aligned_seq
      2) Permute residues across non-gap positions ONLY (gaps fixed), n_perm times
      3) Compute HCI_raw for each permuted sequence => null distribution
      4) HCI_Z = (obs - mean(null)) / (sd(null) + eps)

    Returns HCIResult with raw/norm/frac and z + null stats.

    Notes:
    - This null preserves:
        * number of hydrophobic residues
        * overall residue composition across non-gap positions (exactly, because it’s a shuffle)
        * gap structure
    - This makes HCI_Z specifically measure *excess clustering beyond chance*.
    """
    raw, norm, frac = hci_normalized(
        aligned_seq,
        radius=radius,
        hydrophobic_set=hydrophobic_set,
        gap=gap,
        weight=weight,
        eps=eps,
    )

    if n_perm <= 0:
        return HCIResult(hci_raw=raw, hci_norm=norm, hci_frac=frac, hci_z=None)

    rng = np.random.default_rng(seed)
    non_gap_idx = _non_gap_indices(aligned_seq, gap=gap)
    if non_gap_idx.size <= 1:
        return HCIResult(hci_raw=raw, hci_norm=norm, hci_frac=frac, hci_z=0.0, null_mean=raw, null_sd=0.0)

    # Extract the residue characters at non-gap positions and shuffle them.
    seq_list = list(aligned_seq)
    residues = np.array([seq_list[i] for i in non_gap_idx], dtype="<U1")

    null_scores = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        perm = residues.copy()
        rng.shuffle(perm)
        # Build permuted aligned string with gaps preserved
        for k, pos in enumerate(non_gap_idx):
            seq_list[pos] = perm[k]
        permuted = "".join(seq_list)

        null_scores[b] = hci_raw(
            permuted,
            radius=radius,
            hydrophobic_set=hydrophobic_set,
            gap=gap,
            weight=weight,
        )

    mu = float(null_scores.mean())
    sd = float(null_scores.std(ddof=1)) if n_perm > 1 else 0.0
    z = (raw - mu) / (sd + eps)

    return HCIResult(
        hci_raw=raw,
        hci_norm=norm,
        hci_frac=frac,
        hci_z=z,
        null_mean=mu,
        null_sd=sd,
    )


 

def compute_hci_for_aligned_column(
    aligned_seqs: Sequence[str],
    radius: int = 5,
    hydrophobic_set: Iterable[str] = DEFAULT_HYDROPHOBIC,
    gap: str = DEFAULT_GAP,
    weight: str = "inv",
    n_perm: int = 200,
    seed: Optional[int] = 0,
) -> Dict[str, np.ndarray]:
    """
    Compute HCI metrics for a sequence of aligned strings (e.g., a pandas Series).

    Returns a dict of arrays:
      - hci_raw, hci_norm, hci_frac, hci_z, null_mean, null_sd
    """
    n = len(aligned_seqs)
    out = {
        "hci_raw": np.zeros(n, dtype=np.float64),
        "hci_norm": np.zeros(n, dtype=np.float64),
        "hci_frac": np.zeros(n, dtype=np.float64),
        "hci_z": np.zeros(n, dtype=np.float64),
        "null_mean": np.zeros(n, dtype=np.float64),
        "null_sd": np.zeros(n, dtype=np.float64),
    }

    # Make seeds per row reproducible but distinct
    base = 0 if seed is None else int(seed)

    for i, s in enumerate(aligned_seqs):
        res = hci_zscore(
            s,
            radius=radius,
            hydrophobic_set=hydrophobic_set,
            gap=gap,
            weight=weight,
            n_perm=n_perm,
            seed=base + i,  # per-row deterministic
        )
        out["hci_raw"][i] = res.hci_raw
        out["hci_norm"][i] = res.hci_norm
        out["hci_frac"][i] = res.hci_frac
        out["hci_z"][i] = float(res.hci_z) if res.hci_z is not None else np.nan
        out["null_mean"][i] = float(res.null_mean) if res.null_mean is not None else np.nan
        out["null_sd"][i] = float(res.null_sd) if res.null_sd is not None else np.nan

    return out

def spearman_hci_vs_hic(df, hci_column, hic_column="HIC"):
    """
    Compute Spearman rank correlation between an HCI-Z column and HIC.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing HCI-Z and HIC columns.
    hci_column : str
        Column name for HCI-Z values (e.g., 'HCIz_heavy').
    hic_column : str, default='HIC'
        Column name for HIC assay values.

    Returns
    -------
    dict
        Dictionary containing:
            - rho : Spearman correlation coefficient
            - p_value : two-sided p-value
            - n : number of valid paired observations
    """

    # Drop rows with missing values
    subset = df[[hci_column, hic_column]].dropna()

    if subset.shape[0] < 3:
        raise ValueError("Not enough non-missing paired observations.")

    rho, p_value = spearmanr(subset[hci_column], subset[hic_column])

    return {
        "rho": rho,
        "p_value": p_value,
        "n": subset.shape[0]
    }

def hydrophobic_fraction(aligned_seq):
    """Compute hydrophobic fraction excluding gaps."""
    residues = [aa for aa in aligned_seq if aa != GAP]
    if len(residues) == 0:
        return np.nan
    hyd_count = sum(aa in HYDROPHOBIC for aa in residues)
    return hyd_count / len(residues)

def spearman_test(df, xcol, ycol="HIC"):
    subset = df[[xcol, ycol]].dropna()
    rho, p = spearmanr(subset[xcol], subset[ycol])
    return {"rho": rho, "p_value": p, "n": len(subset)}



if __name__ == "__main__":

    ## Heavy-chain cooperativity (AHO aligned)
    heavy_hci = compute_hci_for_aligned_column(
        df["heavy_aligned_aho"].values,
        radius=5,
        n_perm=200,
        seed=123,
    )

    ## Light-chain cooperativity (AHO aligned)
    light_hci = compute_hci_for_aligned_column(
        df["light_aligned_aho"].values,
        radius=5,
        n_perm=200,
        seed=456,
    )

    ## Add to df (example: use HCI_Z)
    df["HCIz_heavy"] = heavy_hci["hci_z"]
    df["HCIz_light"] = light_hci["hci_z"]
    df["HCIz_mean"] = (df["HCIz_heavy"] + df["HCIz_light"]) / 2
    df["HCIz_max"] = df[["HCIz_heavy", "HCIz_light"]].max(axis=1)


    # print(df.head(3))
    # print()
    # print(df.columns)

    result_heavy = spearman_hci_vs_hic(df, "HCIz_heavy")
    result_light = spearman_hci_vs_hic(df, "HCIz_light")
    result_mean  = spearman_hci_vs_hic(df, "HCIz_mean")

    # print(result_heavy)
    # print(result_light)
    # print(result_mean)

    # Compute heavy and light hydrophobic fractions
    df["hyd_frac_heavy"] = df["heavy_aligned_aho"].apply(hydrophobic_fraction)
    df["hyd_frac_light"] = df["light_aligned_aho"].apply(hydrophobic_fraction)
    df["hyd_frac_mean"] = df[["hyd_frac_heavy", "hyd_frac_light"]].mean(axis=1)

    # Run correlations
    #result_heavy = spearman_test(df, "hyd_frac_heavy")  
    #result_light = spearman_test(df, "hyd_frac_light")
    #result_mean  = spearman_test(df, "hyd_frac_mean")
    # {'rho': np.float64(0.21229933168620513), 'p_value': np.float64(0.000889054114963695), 'n': 242} {'rho': np.float64(0.18242378148432056), 'p_value': np.float64(0.0044121747024409245), 'n': 242} {'rho': np.float64(0.2958103403947861), 'p_value': np.float64(2.8252431563320334e-06), 'n': 242}

    # result_heavy = spearman_test(df, "hyd_frac_heavy", ycol="AC-SINS_pH6.0")  
    # result_light = spearman_test(df, "hyd_frac_light", ycol="AC-SINS_pH6.0")
    # result_mean  = spearman_test(df, "hyd_frac_mean", ycol="AC-SINS_pH6.0")
    # print(result_heavy, result_light, result_mean) 
    # {'rho': np.float64(0.03183133932400535), 'p_value': np.float64(0.6221959733975716), 'n': 242} 
    # {'rho': np.float64(0.012825796463814767), 'p_value': np.float64(0.8426556966907905), 'n': 242} 
    # {'rho': np.float64(0.03279563043891831), 'p_value': np.float64(0.6116810309432361), 'n': 242}

# =========================================================================================

    result_heavy_AC_SINSpH74 = spearman_test(df, "hyd_frac_heavy", ycol="AC-SINS_pH7.4")  
    result_light_AC_SINSpH74  = spearman_test(df, "hyd_frac_light", ycol="AC-SINS_pH7.4")
    result_mean_AC_SINSpH74   = spearman_test(df, "hyd_frac_mean", ycol="AC-SINS_pH7.4")
    print(result_heavy_AC_SINSpH74 , result_light_AC_SINSpH74 , result_mean_AC_SINSpH74 ) 
 
