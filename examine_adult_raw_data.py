#!/usr/bin/env python3
"""
Examine adult H2B and VSV raw data at replicate level.

Loads the same summary CSVs used by the correlation pipeline, filters to adult
(p60 -> adult), reports identifier overlap, builds per-region per-replicate
tables with area-normalized intensity, flags outliers (z-score and IQR), and
optionally produces boxplots and a scatter of region means.

IMPORTANT: Adult H2B and VSV currently have NO shared identifiers. H2B adult
uses p60 animals (M762, M763, M773, M776); VSV adult uses different animals
(M608, M609, M610). Therefore:
- averaged_replicates adult correlation is between different animals in the two
  modalities, not a within-animal correlation.
- complete_pairs adult has no pairs and is undefined.
Treat low adult r as largely structural unless paired adult data becomes available.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Match correlation_analysis.py paths and constants
BASE = Path(__file__).resolve().parent / 'data' / 'dataframes_by_section'
H2B_RAW_SUMMARY = BASE / 'h2b' / 'h2b_raw_summary.csv'
VSV_RAW_SUMMARY = BASE / 'vsv' / 'vsv_raw_summary.csv'
VALUE_COL = 'raw_intensity_sum'
EXCLUDED_REGIONS = ['VISp', 'RSPv', 'RSPd', 'RSPagl', 'VISpl']


def load_adult_long(h2b_path, vsv_path, value_col, area_normalize=True):
    """Load summary CSVs, normalize p60->adult, filter to adult, return long-format DataFrames."""
    h2b = pd.read_csv(h2b_path)
    vsv = pd.read_csv(vsv_path)
    for df in (h2b, vsv):
        df['age'] = df['age'].astype(str).str.strip().replace('p60', 'adult')
    h2b = h2b[h2b['age'] == 'adult'].copy()
    vsv = vsv[vsv['age'] == 'adult'].copy()

    if value_col not in h2b.columns or value_col not in vsv.columns:
        raise ValueError(f"Value column '{value_col}' not in both CSVs. H2B: {list(h2b.columns)}, VSV: {list(vsv.columns)}")

    if area_normalize:
        if 'tissue_area_pixels' not in h2b.columns or 'tissue_area_pixels' not in vsv.columns:
            raise ValueError("tissue_area_pixels required for area normalization")
        area_h = h2b['tissue_area_pixels'].replace(0, np.nan)
        area_v = vsv['tissue_area_pixels'].replace(0, np.nan)
        h2b = h2b.assign(intensity=pd.to_numeric(h2b[value_col], errors='coerce') / area_h)
        vsv = vsv.assign(intensity=pd.to_numeric(vsv[value_col], errors='coerce') / area_v)
    else:
        h2b = h2b.assign(intensity=pd.to_numeric(h2b[value_col], errors='coerce'))
        vsv = vsv.assign(intensity=pd.to_numeric(vsv[value_col], errors='coerce'))

    return h2b, vsv


def get_common_regions(h2b_long, vsv_long):
    """Regions present in both modalities, excluding EXCLUDED_REGIONS."""
    r_h = set(h2b_long['region'].dropna().unique())
    r_v = set(vsv_long['region'].dropna().unique())
    common = sorted(r_h & r_v)
    return [r for r in common if r not in EXCLUDED_REGIONS]


def build_replicate_table(long_df, modality, regions):
    """One row per (region, identifier) with intensity and tissue_area_pixels (if present)."""
    df = long_df[long_df['region'].isin(regions)].copy()
    # One value per (identifier, region); if multiple rows (e.g. sections), aggregate by sum
    agg_cols = {'intensity': 'sum'}
    if 'tissue_area_pixels' in df.columns:
        agg_cols['tissue_area_pixels'] = 'sum'
    agg = df.groupby(['identifier', 'region'], as_index=False).agg(agg_cols)
    agg = agg.rename(columns={'intensity': f'{modality}_intensity'})
    return agg


def flag_outliers_zscore(df, value_col, region_col='region', identifier_col='identifier', threshold=2.0):
    """Within each region, flag rows where value is beyond threshold std from mean. Returns list of dicts."""
    flagged = []
    for region, g in df.groupby(region_col):
        vals = g[value_col].dropna()
        if len(vals) < 2:
            continue
        mean, std = vals.mean(), vals.std()
        if std == 0:
            continue
        z = (g[value_col] - mean) / std
        for _, row in g.iterrows():
            z_val = z.loc[row.name] if row.name in z.index else np.nan
            if np.abs(z_val) > threshold:
                flagged.append({
                    'region': region,
                    'identifier': row[identifier_col],
                    'value': row[value_col],
                    'z_score': float(z_val),
                    'method': 'z_score',
                })
    return flagged


def flag_outliers_iqr(df, value_col, region_col='region', identifier_col='identifier', k=1.5):
    """Within each region, flag rows below Q1 - k*IQR or above Q3 + k*IQR. Returns list of dicts."""
    flagged = []
    for region, g in df.groupby(region_col):
        vals = g[value_col].dropna()
        if len(vals) < 2:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        for _, row in g.iterrows():
            v = row[value_col]
            if pd.isna(v):
                continue
            if v < lo or v > hi:
                flagged.append({
                    'region': region,
                    'identifier': row[identifier_col],
                    'value': row[value_col],
                    'Q1': float(q1),
                    'Q3': float(q3),
                    'IQR': float(iqr),
                    'method': 'IQR',
                })
    return flagged


def main():
    parser = argparse.ArgumentParser(description='Examine adult raw data and replicate-level outliers.')
    parser.add_argument('--h2b', type=Path, default=H2B_RAW_SUMMARY, help='H2B summary CSV')
    parser.add_argument('--vsv', type=Path, default=VSV_RAW_SUMMARY, help='VSV summary CSV')
    parser.add_argument('--out-dir', type=Path, default=Path('adult_raw_diagnostic'), help='Output directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip boxplots and scatter')
    parser.add_argument('--no-area-normalize', action='store_true', help='Use raw intensity instead of area-normalized')
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    area_norm = not args.no_area_normalize
    h2b_long, vsv_long = load_adult_long(
        args.h2b, args.vsv, VALUE_COL, area_normalize=area_norm
    )

    id_h2b = sorted(h2b_long['identifier'].dropna().unique().tolist())
    id_vsv = sorted(vsv_long['identifier'].dropna().unique().tolist())
    overlap = sorted(set(id_h2b) & set(id_vsv))

    # Identifier report (and write to file)
    report_lines = [
        '=== Adult identifier report ===',
        f'H2B adult identifiers ({len(id_h2b)}): {id_h2b}',
        f'VSV adult identifiers ({len(id_vsv)}): {id_vsv}',
        f'Overlap (shared): {overlap if overlap else "NONE"}',
        '',
        'Adult H2B and VSV have no shared identifiers. Averaged-replicate adult',
        'correlation is between different animals in the two modalities. Complete-pairs',
        'adult has no pairs and is undefined.',
    ]
    report_text = '\n'.join(report_lines)
    print(report_text)
    (out_dir / 'identifier_report.txt').write_text(report_text, encoding='utf-8')

    regions = get_common_regions(h2b_long, vsv_long)
    print(f'\nCommon regions (excl. {EXCLUDED_REGIONS}): {len(regions)} — {regions}')

    # Replicate-level tables
    h2b_rep = build_replicate_table(h2b_long, 'H2B', regions)
    vsv_rep = build_replicate_table(vsv_long, 'VSV', regions)

    h2b_rep.to_csv(out_dir / 'adult_h2b_replicate_level.csv', index=False)
    vsv_rep.to_csv(out_dir / 'adult_vsv_replicate_level.csv', index=False)
    print(f'\nWrote replicate-level tables to {out_dir}')

    # Outlier flags
    h2b_flag_z = flag_outliers_zscore(h2b_rep, 'H2B_intensity', 'region', 'identifier')
    vsv_flag_z = flag_outliers_zscore(vsv_rep, 'VSV_intensity', 'region', 'identifier')
    h2b_flag_iqr = flag_outliers_iqr(h2b_rep, 'H2B_intensity', 'region', 'identifier')
    vsv_flag_iqr = flag_outliers_iqr(vsv_rep, 'VSV_intensity', 'region', 'identifier')

    outliers = []
    for d in h2b_flag_z:
        d = d.copy()
        d['modality'] = 'H2B'
        outliers.append(d)
    for d in vsv_flag_z:
        d = d.copy()
        d['modality'] = 'VSV'
        outliers.append(d)
    for d in h2b_flag_iqr:
        d = d.copy()
        d['modality'] = 'H2B'
        outliers.append(d)
    for d in vsv_flag_iqr:
        d = d.copy()
        d['modality'] = 'VSV'
        outliers.append(d)

    if outliers:
        out_df = pd.DataFrame(outliers)
        out_df.to_csv(out_dir / 'adult_replicate_outliers.csv', index=False)
        print(f'Wrote {len(outliers)} outlier flags to {out_dir / "adult_replicate_outliers.csv"}')
    else:
        print('No replicate-level outliers flagged (z-score |z|>2 or IQR outside 1.5*IQR).')

    # Optional visualizations
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not available; skipping plots.')
        else:
            # Boxplots: one figure per region (H2B and VSV side by side) or one figure with all regions
            fig, axes = plt.subplots(len(regions), 2, figsize=(8, 2 * max(len(regions), 1)))
            if len(regions) == 1:
                axes = axes.reshape(1, -1)
            for i, reg in enumerate(regions):
                h = h2b_rep[h2b_rep['region'] == reg]['H2B_intensity'].dropna()
                v = vsv_rep[vsv_rep['region'] == reg]['VSV_intensity'].dropna()
                axes[i, 0].boxplot(h, tick_labels=['H2B'])
                axes[i, 0].set_title(f'{reg} H2B (n={len(h)})')
                axes[i, 0].set_ylabel('intensity')
                axes[i, 1].boxplot(v, tick_labels=['VSV'])
                axes[i, 1].set_title(f'{reg} VSV (n={len(v)})')
                axes[i, 1].set_ylabel('intensity')
            plt.tight_layout()
            plt.savefig(out_dir / 'adult_boxplots_by_region.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved {out_dir / "adult_boxplots_by_region.png"}')

            # Scatter of region means (matching pipeline: mean H2B vs mean VSV per region)
            h2b_means = h2b_rep.groupby('region')['H2B_intensity'].mean()
            vsv_means = vsv_rep.groupby('region')['VSV_intensity'].mean()
            regs = sorted(h2b_means.index.intersection(vsv_means.index))
            x = [h2b_means[r] for r in regs]
            y = [vsv_means[r] for r in regs]
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, s=80, alpha=0.8)
            for r in regs:
                plt.annotate(r, (h2b_means[r], vsv_means[r]), xytext=(5, 5), textcoords='offset points', fontsize=8)
            plt.xlabel('Mean H2B intensity (area-norm)' if area_norm else 'Mean H2B intensity')
            plt.ylabel('Mean VSV intensity (area-norm)' if area_norm else 'Mean VSV intensity')
            plt.title('Adult: region means (H2B vs VSV)\nDifferent animals in each modality')
            r_val = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
            plt.text(0.05, 0.95, f'r = {r_val:.4f}', transform=plt.gca().transAxes, fontsize=11)
            plt.tight_layout()
            plt.savefig(out_dir / 'adult_region_means_scatter.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved {out_dir / "adult_region_means_scatter.png"}')

    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    main()
