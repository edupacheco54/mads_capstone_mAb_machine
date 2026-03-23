"""
CDR-region decomposed biophysical features + liability motif counts
Spearman correlation vs Titer (GDPa1 dataset)

AHO alignment boundaries (0-indexed Python slices):
  Heavy:  CDR-H1 [25:42]  CDR-H2 [56:76]  CDR-H3 [106:138]
  Light:  CDR-L1 [25:42]  CDR-L2 [56:72]  CDR-L3 [106:138]
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import re

# ── Amino acid lookup tables ────────────────────────────────────────────────
# Kyte-Doolittle hydrophobicity
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

HYDROPHOBIC = set('AILVMFYW')     # classic hydrophobic set
AROMATIC    = set('FYW')
POSITIVE    = set('KR')           # +1 at pH 7.4
NEGATIVE    = set('DE')           # -1 at pH 7.4


# ── Region slicing ──────────────────────────────────────────────────────────
CDR_SLICES = {
    'H1': (25, 42), 'H2': (56, 76), 'H3': (106, 138),
    'L1': (25, 42), 'L2': (56, 72), 'L3': (106, 138),
}

REGION_CHAIN = {
    'H1': 'heavy', 'H2': 'heavy', 'H3': 'heavy',
    'L1': 'light', 'L2': 'light', 'L3': 'light',
}

def extract_cdr(aligned_seq, cdr_name):
    s, e = CDR_SLICES[cdr_name]
    return aligned_seq[s:e].replace('-', '')


# ── Per-region biochemical descriptors ─────────────────────────────────────
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


# ── Liability motif counts (on raw ungapped sequence) ──────────────────────
def count_deamidation(seq):
    """NG, NS, NA dipeptides — prone to deamidation"""
    return len(re.findall(r'N[GSA]', seq))

def count_isomerization(seq):
    """DG, DS dipeptides — prone to Asp isomerization"""
    return len(re.findall(r'D[GS]', seq))

def count_oxidation(seq):
    """W and M residue counts — prone to oxidation"""
    return seq.count('W') + seq.count('M')

def count_glycosylation(seq):
    """N-X-S/T sequons where X != P — N-linked glycosylation risk"""
    return len(re.findall(r'N[^P][ST]', seq))

def count_unpaired_cys(seq):
    """Odd number of C residues suggests unpaired disulfide"""
    c_count = seq.count('C')
    return c_count % 2  # 1 if unpaired, 0 if paired


# ── Feature extraction per antibody ────────────────────────────────────────
def build_features(row):
    heavy_aln = row['heavy_aligned_aho']
    light_aln = row['light_aligned_aho']

    feats = {}

    # Per-CDR descriptors
    for cdr in ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']:
        chain_aln = heavy_aln if REGION_CHAIN[cdr] == 'heavy' else light_aln
        seq = extract_cdr(chain_aln, cdr)

        feats[f'cdr{cdr}_len']          = cdr_length(seq)
        feats[f'cdr{cdr}_gravy']        = gravy(seq)
        feats[f'cdr{cdr}_hyd_frac']     = hyd_frac(seq)
        feats[f'cdr{cdr}_charge']       = net_charge(seq)
        feats[f'cdr{cdr}_deamid']       = count_deamidation(seq)
        feats[f'cdr{cdr}_isom']         = count_isomerization(seq)
        feats[f'cdr{cdr}_oxidn']        = count_oxidation(seq)

    # CDR-H3 specific (most important CDR for developability)
    h3_seq = extract_cdr(heavy_aln, 'H3')
    feats['cdrH3_aromatic_frac']        = aromatic_frac(h3_seq)
    feats['cdrH3_glycan_sequon']        = count_glycosylation(h3_seq)

    # Full VH/VL liability counts (whole variable domain)
    vh = row['vh_protein_sequence'] if pd.notna(row['vh_protein_sequence']) else ''
    vl = row['vl_protein_sequence'] if pd.notna(row['vl_protein_sequence']) else ''

    for label, seq in [('VH', vh), ('VL', vl)]:
        feats[f'{label}_deamid_total']  = count_deamidation(seq)
        feats[f'{label}_isom_total']    = count_isomerization(seq)
        feats[f'{label}_oxidn_total']   = count_oxidation(seq)
        feats[f'{label}_glycan_total']  = count_glycosylation(seq)
        feats[f'{label}_hyd_frac']      = hyd_frac(seq)
        feats[f'{label}_gravy']         = gravy(seq)
        feats[f'{label}_charge']        = net_charge(seq)

    # Composite summaries across chains
    feats['total_deamid']               = feats['VH_deamid_total'] + feats['VL_deamid_total']
    feats['total_oxidn']                = feats['VH_oxidn_total']  + feats['VL_oxidn_total']
    feats['VH_unpaired_cys']            = count_unpaired_cys(vh)
    feats['VL_unpaired_cys']            = count_unpaired_cys(vl)

    return feats


# ── Run Spearman vs Titer ───────────────────────────────────────────────────
def spearman_vs_titer(df_feat, titer, label=''):
    results = []
    for col in df_feat.columns:
        x = df_feat[col]
        mask = x.notna() & titer.notna()
        if mask.sum() < 30:
            continue
        rho, p = spearmanr(x[mask], titer[mask])
        results.append({'feature': col, 'rho': rho, 'p_value': p, 'n': mask.sum()})
    return pd.DataFrame(results).sort_values('rho', key=abs, ascending=False)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('/mnt/project/gdpa1_246_iggs_cleaned.csv')

    print(f"Dataset: {len(df)} antibodies, {df['Titer'].notna().sum()} with Titer")
    print()

    # Build feature matrix
    feat_rows = []
    for _, row in df.iterrows():
        if pd.isna(row['heavy_aligned_aho']) or pd.isna(row['light_aligned_aho']):
            feat_rows.append({})
            continue
        feat_rows.append(build_features(row))

    feat_df = pd.DataFrame(feat_rows, index=df.index)
    print(f"Feature matrix: {feat_df.shape[1]} features\n")
    print("Features:", list(feat_df.columns))
    print()

    # Spearman correlations
    results = spearman_vs_titer(feat_df, df['Titer'])

    print("=" * 70)
    print("Spearman ρ vs Titer — sorted by |ρ|")
    print("=" * 70)
    print(f"{'Feature':<35} {'rho':>7}  {'p_value':>12}  {'n':>4}  sig")
    print("-" * 70)
    for _, r in results.iterrows():
        sig = '***' if r['p_value'] < 0.001 else '** ' if r['p_value'] < 0.01 else '*  ' if r['p_value'] < 0.05 else '   '
        print(f"{r['feature']:<35} {r['rho']:>7.4f}  {r['p_value']:>12.4e}  {int(r['n']):>4}  {sig}")

    print()
    print("=== TOP 10 ABSOLUTE ===")
    print(results.head(10).to_string(index=False))
