#!/usr/bin/env python3
"""
Pearson Correlation Analysis between H2B and VSV signals
across age cohorts (p3, p12, p20, adult) for each brain region.
"""

import argparse
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os
import sys
from pathlib import Path


class Tee:
    """Write to both a file and the original stream (e.g. stdout/stderr)."""
    def __init__(self, stream, path):
        self._stream = stream
        self._file = open(path, 'w', encoding='utf-8')
    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def close(self):
        self._file.close()

# Plot defaults: editable SVG text, Helvetica/Arial
plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['svg.fonttype'] = 'none'

# Short labels for region names on multi-region plots
REGION_LABEL_CODEX = {
    'VISa': 'A', 'VISal': 'AL', 'VISpor': 'POR', 'VISli': 'LI',
    'VISam': 'AM', 'VISpm': 'PM', 'VISrl': 'RL', 'VISl': 'LM',
}

# Constants
AGE_COHORTS = ['p3', 'p12', 'p20', 'adult']
OUTPUT_DIRS = {
    'complete_pairs': 'results/complete_pairs',
    'averaged_replicates': 'results/averaged_replicates',
    'imputed_data': 'results/imputed_data'
}

# Data file paths (defaults for dataframes_by_section)
BASE = 'data/dataframes_by_section'
VSV_FILE = 'data/sums/vsv_intensity_sums_20260127.csv'
H2B_FILE = f'{BASE}/h2b/h2b_raw_summary.csv'

# Value column for long-format summary CSVs (infer from path if missing)
SUMMARY_VALUE_COLUMNS = {
    'raw_summary': 'raw_intensity_sum',
    'dist_weighted_summary': 'dist_weighted_intensity_sum',
    'masked_mean_summary': 'masked_mean_intensity_sum',
    'masked_median_summary': 'masked_median_intensity_sum',
    'masked_gauss_summary': 'masked_gauss_intensity_sum',
    'percent_total_summary': 'percent_total',
}

VSV_PATH_WIDE = 'data/sums/vsv_intensity_sums_20260127.csv'
# (name, h2b_path, vsv_path, h2b_value_col, vsv_value_col). Use dataframes_by_section VSV so both modalities are area-normalized by default.
DATASET_RUNS = [
    ('by_section_raw', f'{BASE}/h2b/h2b_raw_summary.csv', f'{BASE}/vsv/vsv_raw_summary.csv', 'raw_intensity_sum', 'raw_intensity_sum'),
    ('by_section_masked_mean', f'{BASE}/h2b/h2b_masked_mean_summary.csv', f'{BASE}/vsv/vsv_masked_mean_summary.csv', 'masked_mean_intensity_sum', 'masked_mean_intensity_sum'),
    ('by_section_masked_median', f'{BASE}/h2b/h2b_masked_median_summary.csv', f'{BASE}/vsv/vsv_masked_median_summary.csv', 'masked_median_intensity_sum', 'masked_median_intensity_sum'),
    ('by_section_masked_gauss', f'{BASE}/h2b/h2b_masked_gauss_summary.csv', f'{BASE}/vsv/vsv_masked_gauss_summary.csv', 'masked_gauss_intensity_sum', 'masked_gauss_intensity_sum'),
    ('by_section_dist_weighted', f'{BASE}/h2b/h2b_dist_weighted_summary.csv', f'{BASE}/vsv/vsv_dist_weighted_summary.csv', 'dist_weighted_intensity_sum', 'dist_weighted_intensity_sum'),
    ('by_percent_total', f'{BASE}/h2b/h2b_percent_total_summary.csv', f'{BASE}/vsv/vsv_percent_total_summary.csv', 'percent_total', 'percent_total'),
]


def _infer_summary_value_column(path):
    """Infer value column for long-format summary CSV from path. Returns None if not a known summary type."""
    path = str(path)
    for key, col in SUMMARY_VALUE_COLUMNS.items():
        if key in path:
            return col
    return None


def _pivot_summary_to_wide(df, value_col, suffix):
    """
    Pivot long-format summary (age, identifier, region, value_col) to wide (age, identifier, Region-Suffix).
    """
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not in dataframe: {list(df.columns)}")
    wide = df.pivot_table(index=['age', 'identifier'], columns='region', values=value_col, aggfunc='first').reset_index()
    wide.columns = [f"{c}-{suffix}" if c not in ('age', 'identifier') else c for c in wide.columns]
    return wide


def _load_vsv_prior_area(path):
    """
    Load a wide-format CSV (identifier/Animal + one column per region) and return long form.
    Returns DataFrame with columns identifier, region, prior_area. Zeros are replaced with NaN.
    """
    df = pd.read_csv(path)
    if 'Animal' in df.columns and 'identifier' not in df.columns:
        df = df.rename(columns={'Animal': 'identifier'})
    id_col = 'identifier'
    if id_col not in df.columns:
        raise ValueError(f"Prior area CSV must have 'identifier' or 'Animal' column; got {list(df.columns)}")
    region_cols = [c for c in df.columns if c != id_col]
    long_df = df.melt(id_vars=[id_col], value_vars=region_cols, var_name='region', value_name='prior_area')
    long_df['prior_area'] = pd.to_numeric(long_df['prior_area'], errors='coerce').replace(0, np.nan)
    return long_df


def setup_directories(output_base_dir='results'):
    """Create output directories for each analysis type under output_base_dir."""
    dirs = {
        'complete_pairs': os.path.join(output_base_dir, 'complete_pairs'),
        'averaged_replicates': os.path.join(output_base_dir, 'averaged_replicates'),
        'imputed_data': os.path.join(output_base_dir, 'imputed_data'),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'normalized'), exist_ok=True)
    print(f"Output directories created under {output_base_dir}.")
    return dirs


def load_and_preprocess_data(h2b_path=None, vsv_path=None, h2b_value_col=None, vsv_value_col=None, area_normalize=True, vsv_prior_area_path=None):
    """
    Load CSV files and preprocess data.
    If h2b_value_col / vsv_value_col are provided, the path is treated as long-format summary
    (age, identifier, region, value_col): pivot to wide (age, identifier, Region-H2b / Region-VSV).
    Area-normalized intensity is computed per animal (per identifier) per region; the resulting
    wide table has one row per (age, identifier) and is passed to comparison calculations
    (replicate averages, imputation, complete pairs) without lumping across animals.
    If vsv_prior_area_path is set, VSV area for normalization comes from that CSV (identifier x region)
    instead of tissue_area_pixels; fallback to tissue_area_pixels when prior has no row.
    Returns: h2b_df, vsv_df, region_mapping
    """
    h2b_path = h2b_path or H2B_FILE
    vsv_path = vsv_path or VSV_FILE
    
    # Load data
    h2b_df = pd.read_csv(h2b_path)
    vsv_df = pd.read_csv(vsv_path)
    
    # Pivot long-format summary to wide; optionally area-normalize (per animal per region)
    if h2b_value_col is not None and 'region' in h2b_df.columns:
        if area_normalize and 'tissue_area_pixels' in h2b_df.columns:
            area = h2b_df['tissue_area_pixels'].replace(0, np.nan)
            h2b_df = h2b_df.copy()
            h2b_df['_value_for_pivot'] = h2b_df[h2b_value_col] / area
            h2b_df = _pivot_summary_to_wide(h2b_df, '_value_for_pivot', 'H2b')
        else:
            h2b_df = _pivot_summary_to_wide(h2b_df, h2b_value_col, 'H2b')
    if vsv_value_col is not None and 'region' in vsv_df.columns:
        if area_normalize and 'tissue_area_pixels' in vsv_df.columns:
            vsv_df = vsv_df.copy()
            if vsv_prior_area_path and os.path.isfile(vsv_prior_area_path):
                prior_long = _load_vsv_prior_area(vsv_prior_area_path)
                vsv_merged = vsv_df.merge(prior_long, on=['identifier', 'region'], how='left')
                vsv_df['_area'] = vsv_merged['prior_area'].fillna(vsv_merged['tissue_area_pixels'])
            else:
                vsv_df['_area'] = vsv_df['tissue_area_pixels']
            area = vsv_df['_area'].replace(0, np.nan)
            vsv_df['_value_for_pivot'] = vsv_df[vsv_value_col] / area
            vsv_df = _pivot_summary_to_wide(vsv_df, '_value_for_pivot', 'VSV')
        else:
            vsv_df = _pivot_summary_to_wide(vsv_df, vsv_value_col, 'VSV')
    
    # Normalize age labels: p60 and adult are the same age group.
    # NOTE: H2B adult rows are from p60 (e.g. M762, M763, M773, M776). The default VSV
    # summary (vsv_raw_summary.csv) has adult rows with M608, M609, M610 only — no overlap.
    # If both modalities are from the same image set, use a VSV path that contains p60 (same
    # identifiers as H2B) so adult correlation is on the same animals.
    h2b_df['age'] = h2b_df['age'].replace('p60', 'adult')
    vsv_df['age'] = vsv_df['age'].replace('p60', 'adult')
    
    # Filter for target age cohorts
    h2b_df = h2b_df[h2b_df['age'].isin(AGE_COHORTS)].copy()
    vsv_df = vsv_df[vsv_df['age'].isin(AGE_COHORTS)].copy()
    
    # Convert empty strings to NaN
    h2b_df = h2b_df.replace('', np.nan)
    vsv_df = vsv_df.replace('', np.nan)
    
    # Convert numeric columns to float (excluding age and identifier)
    numeric_cols_h2b = [col for col in h2b_df.columns if col not in ['age', 'identifier']]
    numeric_cols_vsv = [col for col in vsv_df.columns if col not in ['age', 'identifier']]
    
    for col in numeric_cols_h2b:
        h2b_df[col] = pd.to_numeric(h2b_df[col], errors='coerce')
    for col in numeric_cols_vsv:
        vsv_df[col] = pd.to_numeric(vsv_df[col], errors='coerce')
    
    # Extract brain region names and create mapping
    # H2B: region = substring before "-H2b" (handles "RSPv-H2b" and "RSPv-H2b-masked_mean" etc.)
    h2b_regions = {}
    for col in numeric_cols_h2b:
        if '-H2b' in col:
            region = col.split('-H2b')[0]
            h2b_regions[region] = col
    
    # VSV: "RSPv-VSV" -> "RSPv"
    vsv_regions = {}
    for col in numeric_cols_vsv:
        if '-VSV' in col:
            region = col.replace('-VSV', '')
            vsv_regions[region] = col
    
    # Find common regions
    common_regions = sorted(set(h2b_regions.keys()) & set(vsv_regions.keys()))
    
    # Exclude irrelevant areas
    excluded_regions = ['VISp', 'RSPv', 'RSPd', 'RSPagl', 'VISpl']
    common_regions = [r for r in common_regions if r not in excluded_regions]
    
    region_mapping = {}
    for region in common_regions:
        region_mapping[region] = {
            'h2b_col': h2b_regions[region],
            'vsv_col': vsv_regions[region]
        }
    
    print(f"Found {len(common_regions)} common brain regions.")
    # Per-animal verification: wide tables must have one row per (age, identifier); no aggregation across identifier before analysis functions.
    if 'age' in h2b_df.columns and 'identifier' in h2b_df.columns:
        assert h2b_df.drop_duplicates(['age', 'identifier']).shape[0] == len(h2b_df), "H2B wide table should have one row per (age, identifier)"
    if 'age' in vsv_df.columns and 'identifier' in vsv_df.columns:
        assert vsv_df.drop_duplicates(['age', 'identifier']).shape[0] == len(vsv_df), "VSV wide table should have one row per (age, identifier)"
    return h2b_df, vsv_df, region_mapping


def validate_csv_format(h2b_df, vsv_df, region_mapping):
    """
    Validate loaded data before running analysis.
    Returns True if valid, False otherwise (caller may skip run).
    """
    if 'age' not in h2b_df.columns or 'identifier' not in h2b_df.columns:
        print("Validation failed: H2B CSV must have columns 'age' and 'identifier'.")
        return False
    if 'age' not in vsv_df.columns or 'identifier' not in vsv_df.columns:
        print("Validation failed: VSV CSV must have columns 'age' and 'identifier'.")
        return False
    if len(region_mapping) == 0:
        print("Validation failed: No common region columns (H2B columns containing '-H2b', VSV containing '-VSV').")
        return False
    if len(h2b_df) == 0 or len(vsv_df) == 0:
        print("Validation failed: At least one dataframe is empty after filtering age cohorts.")
        return False
    print("Validation passed.")
    return True


def normalize_intensities(values, method='none'):
    """
    Normalize intensity values using various methods.
    
    Parameters:
    - values: array-like of intensity values
    - method: normalization method ('none', 'log', 'zscore', 'robust', 'combined')
    
    Returns:
    - Normalized values
    """
    values = np.array(values)
    
    if method == 'none':
        return values
    elif method == 'log':
        # Log10 transformation, handle zeros/negatives
        return np.log10(values + 1)  # Add 1 to avoid log(0)
    elif method == 'zscore':
        # Z-score normalization
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0 or np.isnan(std):
            return values
        return (values - mean) / std
    elif method == 'robust':
        # For robust, we use median and MAD (median absolute deviation)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if mad == 0 or np.isnan(mad):
            return values
        return (values - median) / (mad * 1.4826)  # 1.4826 makes MAD consistent with std for normal dist
    elif method == 'combined':
        # Log transformation followed by z-score
        log_values = np.log10(values + 1)
        mean = np.nanmean(log_values)
        std = np.nanstd(log_values)
        if std == 0 or np.isnan(std):
            return log_values
        return (log_values - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def plot_correlation(x, y, age, region, analysis_type, r, r_squared, output_dir):
    """
    Create scatter plot with regression line and correlation statistics.
    
    Parameters:
    - x: H2B values
    - y: VSV values
    - age: age cohort
    - region: brain region
    - analysis_type: type of analysis
    - r: Pearson correlation coefficient
    - r_squared: coefficient of determination
    - output_dir: directory to save plot
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot (solid dark blue)
    plt.scatter(x, y, c='#00008B', s=50, edgecolors='none', alpha=1)
    
    # Regression line (solid black)
    if len(x) > 1 and not np.isnan(r):
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_line, p(x_line), color='black', linestyle='-', linewidth=2)
    
    # Labels and title
    plt.xlabel('H2B Intensity', fontsize=12)
    plt.ylabel('VSV Intensity', fontsize=12)
    plt.title(f'{age} - {region}\n{analysis_type.replace("_", " ").title()}\nr = {r:.4f}, r² = {r_squared:.4f}', 
              fontsize=11)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save plot (PNG and SVG)
    base = f"{age}_{region}_correlation"
    filepath = os.path.join(output_dir, base + '.png')
    filepath_svg = os.path.join(output_dir, base + '.svg')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_svg, format='svg', bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_multi_region_correlation(regions, x, y, age, r, r_squared, output_dir, analysis_type='Averaged Replicates', normalization='none'):
    """
    Create scatter plot with all regions as points, regression line, and correlation statistics.
    
    Parameters:
    - regions: list of region names (for labeling points)
    - x: H2B values (one per region)
    - y: VSV values (one per region)
    - age: age cohort
    - r: Pearson correlation coefficient
    - r_squared: coefficient of determination
    - output_dir: directory to save plot
    - analysis_type: type of analysis for title (default: 'Averaged Replicates')
    - normalization: normalization method used
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot (solid dark blue)
    plt.scatter(x, y, c='#00008B', s=100, edgecolors='none', alpha=1)
    
    # Label points with region names (codex for short labels)
    for i, region in enumerate(regions):
        display_label = REGION_LABEL_CODEX.get(region, region)
        plt.annotate(display_label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=1)
    
    # Regression line (solid black)
    if len(x) > 1 and not np.isnan(r):
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_line, p(x_line), color='black', linestyle='-', linewidth=2)
    
    # Labels and title
    norm_label = f" ({normalization})" if normalization != 'none' else ""
    xlabel = 'H2B Intensity (Averaged)'
    ylabel = 'VSV Intensity (Averaged)'
    
    if normalization == 'log':
        xlabel = 'H2B Intensity (log10)'
        ylabel = 'VSV Intensity (log10)'
    elif normalization == 'zscore':
        xlabel = 'H2B Intensity (Z-score)'
        ylabel = 'VSV Intensity (Z-score)'
    elif normalization == 'robust':
        xlabel = 'H2B Intensity (Robust Normalized)'
        ylabel = 'VSV Intensity (Robust Normalized)'
    elif normalization == 'combined':
        xlabel = 'H2B Intensity (log10 + Z-score)'
        ylabel = 'VSV Intensity (log10 + Z-score)'
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{age} - All Regions\n{analysis_type}\nr = {r:.4f}, r² = {r_squared:.4f}', 
              fontsize=12)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save plot (PNG and SVG)
    base = f"{age}_correlation"
    filepath = os.path.join(output_dir, base + '.png')
    filepath_svg = os.path.join(output_dir, base + '.svg')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_svg, format='svg', bbox_inches='tight')
    plt.close()
    
    return filepath


def complete_pairs_analysis(h2b_df, vsv_df, region_mapping, normalization='none', output_base_dir=None):
    """
    Analysis using only identifiers that have both H2B and VSV signals.
    For each age cohort, calculate mean H2B and mean VSV per region using only complete pairs,
    then calculate one correlation across all regions.
    
    Parameters:
    - normalization: normalization method ('none', 'log', 'zscore', 'robust', 'combined')
    - output_base_dir: base directory for output (default: OUTPUT_DIRS['complete_pairs'])
    """
    if output_base_dir is None:
        output_base_dir = OUTPUT_DIRS['complete_pairs']
    
    norm_label = f" ({normalization})" if normalization != 'none' else ""
    print(f"\n=== Complete Pairs Analysis{norm_label} ===")
    results = []
    
    # Determine correlation method
    use_spearman = (normalization == 'robust')
    corr_func = spearmanr if use_spearman else pearsonr
    
    for age in AGE_COHORTS:
        h2b_age = h2b_df[h2b_df['age'] == age].copy()
        vsv_age = vsv_df[vsv_df['age'] == age].copy()
        
        # Collect region averages (using only complete pairs)
        region_data = []
        
        for region, cols in region_mapping.items():
            h2b_col = cols['h2b_col']
            vsv_col = cols['vsv_col']
            
            # Merge on identifier to get pairs
            merged = pd.merge(
                h2b_age[['identifier', h2b_col]],
                vsv_age[['identifier', vsv_col]],
                on='identifier',
                how='inner'
            )
            
            # Remove rows with NaN in either column (only complete pairs)
            merged_clean = merged.dropna(subset=[h2b_col, vsv_col])
            
            # Only include region if we have at least one complete pair
            if len(merged_clean) > 0:
                h2b_values = merged_clean[h2b_col].values
                vsv_values = merged_clean[vsv_col].values
                
                # Apply normalization before calculating mean
                h2b_normalized = normalize_intensities(h2b_values, normalization)
                vsv_normalized = normalize_intensities(vsv_values, normalization)
                
                # Use median for robust, mean otherwise
                if normalization == 'robust':
                    h2b_avg = np.nanmedian(h2b_normalized)
                    vsv_avg = np.nanmedian(vsv_normalized)
                else:
                    h2b_avg = np.nanmean(h2b_normalized)
                    vsv_avg = np.nanmean(vsv_normalized)
                
                region_data.append({
                    'region': region,
                    'avg_h2b': h2b_avg,
                    'avg_vsv': vsv_avg
                })
            else:
                print(f"  {age} - {region}: Skipping (no complete pairs)")
        
        if len(region_data) < 2:
            print(f"  {age}: Insufficient regions for correlation (n={len(region_data)})")
            continue
        
        # Write tab-separated output file
        output_file = os.path.join(output_base_dir, f'{age}_averages.txt')
        with open(output_file, 'w') as f:
            for rd in region_data:
                # Format numbers appropriately (avoid scientific notation for readability)
                f.write(f"{rd['region']}\t{rd['avg_h2b']:.4f}\t{rd['avg_vsv']:.4f}\n")
        
        print(f"  {age}: Wrote averages for {len(region_data)} regions to {output_file}")
        
        # Extract vectors for correlation
        regions = [rd['region'] for rd in region_data]
        x = np.array([rd['avg_h2b'] for rd in region_data])
        y = np.array([rd['avg_vsv'] for rd in region_data])
        
        # Calculate correlation across all regions
        r, p_value = corr_func(x, y)
        r_squared = r ** 2
        
        # Plot all regions on single plot
        analysis_title = f'Complete Pairs Analysis{norm_label}'
        plot_multi_region_correlation(regions, x, y, age, r, r_squared,
                                     output_base_dir, analysis_title, normalization)
        
        results.append({
            'age': age,
            'normalization': normalization,
            'n_regions': len(region_data),
            'r': r,
            'r_squared': r_squared,
            'p_value': p_value,
            'correlation_method': 'Spearman' if use_spearman else 'Pearson'
        })
        
        print(f"  {age}: r={r:.4f}, r²={r_squared:.4f}, n_regions={len(region_data)}")
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_base_dir, 'summary.csv'), index=False)
    return results


def averaged_replicates_analysis(h2b_df, vsv_df, region_mapping, normalization='none', output_base_dir=None):
    """
    Analysis using averaged values across all replicates per region.
    For each age cohort, calculate mean H2B and mean VSV for each region,
    then calculate one correlation across all regions.
    
    Parameters:
    - normalization: normalization method ('none', 'log', 'zscore', 'robust', 'combined')
    - output_base_dir: base directory for output (default: OUTPUT_DIRS['averaged_replicates'])
    """
    if output_base_dir is None:
        output_base_dir = OUTPUT_DIRS['averaged_replicates']
    
    norm_label = f" ({normalization})" if normalization != 'none' else ""
    print(f"\n=== Averaged Replicates Analysis{norm_label} ===")
    results = []
    
    # Determine correlation method
    use_spearman = (normalization == 'robust')
    corr_func = spearmanr if use_spearman else pearsonr
    
    for age in AGE_COHORTS:
        h2b_age = h2b_df[h2b_df['age'] == age].copy()
        vsv_age = vsv_df[vsv_df['age'] == age].copy()
        
        # Collect region averages
        region_data = []
        
        for region, cols in region_mapping.items():
            h2b_col = cols['h2b_col']
            vsv_col = cols['vsv_col']
            
            # Calculate mean across all identifiers (replicates) for this region
            # Drop NaN values before calculating mean
            h2b_values = h2b_age[h2b_col].dropna().values
            vsv_values = vsv_age[vsv_col].dropna().values
            
            # Only include region if we have at least one value for both signals
            if len(h2b_values) > 0 and len(vsv_values) > 0:
                # Apply normalization before calculating mean
                h2b_normalized = normalize_intensities(h2b_values, normalization)
                vsv_normalized = normalize_intensities(vsv_values, normalization)
                
                # Use median for robust, mean otherwise
                if normalization == 'robust':
                    h2b_avg = np.nanmedian(h2b_normalized)
                    vsv_avg = np.nanmedian(vsv_normalized)
                else:
                    h2b_avg = np.nanmean(h2b_normalized)
                    vsv_avg = np.nanmean(vsv_normalized)
                
                region_data.append({
                    'region': region,
                    'avg_h2b': h2b_avg,
                    'avg_vsv': vsv_avg
                })
            else:
                print(f"  {age} - {region}: Skipping (insufficient data)")
        
        if len(region_data) < 2:
            print(f"  {age}: Insufficient regions for correlation (n={len(region_data)})")
            continue
        
        # Write tab-separated output file
        output_file = os.path.join(output_base_dir, f'{age}_averages.txt')
        with open(output_file, 'w') as f:
            for rd in region_data:
                # Format numbers appropriately (avoid scientific notation for readability)
                f.write(f"{rd['region']}\t{rd['avg_h2b']:.4f}\t{rd['avg_vsv']:.4f}\n")
        
        print(f"  {age}: Wrote averages for {len(region_data)} regions to {output_file}")
        
        # Extract vectors for correlation
        regions = [rd['region'] for rd in region_data]
        x = np.array([rd['avg_h2b'] for rd in region_data])
        y = np.array([rd['avg_vsv'] for rd in region_data])
        
        # Calculate correlation across all regions
        r, p_value = corr_func(x, y)
        r_squared = r ** 2
        
        # Plot all regions on single plot
        analysis_title = f'Averaged Replicates Analysis{norm_label}'
        plot_multi_region_correlation(regions, x, y, age, r, r_squared,
                                     output_base_dir, analysis_title, normalization)
        
        results.append({
            'age': age,
            'normalization': normalization,
            'n_regions': len(region_data),
            'r': r,
            'r_squared': r_squared,
            'p_value': p_value,
            'correlation_method': 'Spearman' if use_spearman else 'Pearson'
        })
        
        print(f"  {age}: r={r:.4f}, r²={r_squared:.4f}, n_regions={len(region_data)}")
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_base_dir, 'summary.csv'), index=False)
    return results


def imputed_data_analysis(h2b_df, vsv_df, region_mapping, normalization='none', output_base_dir=None):
    """
    Analysis with missing values imputed using mean for that region and age group.
    For each age cohort, calculate mean H2B and mean VSV per region (with imputation),
    then calculate one correlation across all regions.
    
    Parameters:
    - normalization: normalization method ('none', 'log', 'zscore', 'robust', 'combined')
    - output_base_dir: base directory for output (default: OUTPUT_DIRS['imputed_data'])
    """
    if output_base_dir is None:
        output_base_dir = OUTPUT_DIRS['imputed_data']
    
    norm_label = f" ({normalization})" if normalization != 'none' else ""
    print(f"\n=== Imputed Data Analysis{norm_label} ===")
    results = []
    
    # Determine correlation method
    use_spearman = (normalization == 'robust')
    corr_func = spearmanr if use_spearman else pearsonr
    
    for age in AGE_COHORTS:
        h2b_age = h2b_df[h2b_df['age'] == age].copy()
        vsv_age = vsv_df[vsv_df['age'] == age].copy()
        
        # Collect region averages (with imputation)
        region_data = []
        
        for region, cols in region_mapping.items():
            h2b_col = cols['h2b_col']
            vsv_col = cols['vsv_col']
            
            # Merge on identifier (outer join to include all identifiers)
            merged = pd.merge(
                h2b_age[['identifier', h2b_col]],
                vsv_age[['identifier', vsv_col]],
                on='identifier',
                how='outer'
            )
            
            # Impute missing values with mean for that region in that age group
            h2b_mean = merged[h2b_col].mean()
            vsv_mean = merged[vsv_col].mean()
            merged[h2b_col] = merged[h2b_col].fillna(h2b_mean)
            merged[vsv_col] = merged[vsv_col].fillna(vsv_mean)
            
            # Only include region if we have at least one value (after imputation)
            if len(merged) > 0:
                h2b_values = merged[h2b_col].values
                vsv_values = merged[vsv_col].values
                
                # Apply normalization before calculating mean
                h2b_normalized = normalize_intensities(h2b_values, normalization)
                vsv_normalized = normalize_intensities(vsv_values, normalization)
                
                # Use median for robust, mean otherwise
                if normalization == 'robust':
                    h2b_avg = np.nanmedian(h2b_normalized)
                    vsv_avg = np.nanmedian(vsv_normalized)
                else:
                    h2b_avg = np.nanmean(h2b_normalized)
                    vsv_avg = np.nanmean(vsv_normalized)
                
                region_data.append({
                    'region': region,
                    'avg_h2b': h2b_avg,
                    'avg_vsv': vsv_avg
                })
            else:
                print(f"  {age} - {region}: Skipping (insufficient data)")
        
        # Drop regions with non-finite averages (NaN/inf) so correlation is valid
        region_data = [rd for rd in region_data if np.isfinite(rd['avg_h2b']) and np.isfinite(rd['avg_vsv'])]
        
        if len(region_data) < 2:
            print(f"  {age}: Insufficient regions for correlation (n={len(region_data)})")
            continue
        
        # Write tab-separated output file
        output_file = os.path.join(output_base_dir, f'{age}_averages.txt')
        with open(output_file, 'w') as f:
            for rd in region_data:
                # Format numbers appropriately (avoid scientific notation for readability)
                f.write(f"{rd['region']}\t{rd['avg_h2b']:.4f}\t{rd['avg_vsv']:.4f}\n")
        
        print(f"  {age}: Wrote averages for {len(region_data)} regions to {output_file}")
        
        # Extract vectors for correlation
        regions = [rd['region'] for rd in region_data]
        x = np.array([rd['avg_h2b'] for rd in region_data])
        y = np.array([rd['avg_vsv'] for rd in region_data])
        
        # Calculate correlation across all regions
        r, p_value = corr_func(x, y)
        r_squared = r ** 2
        
        # Plot all regions on single plot
        analysis_title = f'Imputed Data Analysis{norm_label}'
        plot_multi_region_correlation(regions, x, y, age, r, r_squared,
                                     output_base_dir, analysis_title, normalization)
        
        results.append({
            'age': age,
            'normalization': normalization,
            'n_regions': len(region_data),
            'r': r,
            'r_squared': r_squared,
            'p_value': p_value,
            'correlation_method': 'Spearman' if use_spearman else 'Pearson'
        })
        
        print(f"  {age}: r={r:.4f}, r²={r_squared:.4f}, n_regions={len(region_data)}")
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_base_dir, 'summary.csv'), index=False)
    return results


def comparison_analysis(h2b_df, vsv_df, region_mapping, output_base_dir='results'):
    """
    Run all normalization methods and compare results.
    Generates summary table comparing r, r², p-values across methods.
    Writes under output_base_dir/normalized/.
    """
    print("\n" + "=" * 50)
    print("COMPARISON ANALYSIS - All Normalization Methods")
    print("=" * 50)
    
    normalization_methods = ['none', 'log', 'zscore', 'robust', 'combined']
    all_results = []
    
    for norm_method in normalization_methods:
        print(f"\n{'='*50}")
        print(f"Running with normalization: {norm_method}")
        print(f"{'='*50}")
        
        # Create output directories for this normalization under output_base_dir
        norm_dirs = {}
        for analysis_type in ['complete_pairs', 'averaged_replicates', 'imputed_data']:
            if norm_method == 'none':
                norm_dir = os.path.join(output_base_dir, analysis_type)
            else:
                norm_dir = os.path.join(output_base_dir, 'normalized', norm_method, analysis_type)
            os.makedirs(norm_dir, exist_ok=True)
            norm_dirs[analysis_type] = norm_dir
        
        # Run all three analysis types with this normalization
        complete_results = complete_pairs_analysis(h2b_df, vsv_df, region_mapping, 
                                                   normalization=norm_method,
                                                   output_base_dir=norm_dirs['complete_pairs'])
        averaged_results = averaged_replicates_analysis(h2b_df, vsv_df, region_mapping,
                                                       normalization=norm_method,
                                                       output_base_dir=norm_dirs['averaged_replicates'])
        imputed_results = imputed_data_analysis(h2b_df, vsv_df, region_mapping,
                                                normalization=norm_method,
                                                output_base_dir=norm_dirs['imputed_data'])
        
        # Collect results
        for result_list, analysis_type in [(complete_results, 'complete_pairs'),
                                          (averaged_results, 'averaged_replicates'),
                                          (imputed_results, 'imputed_data')]:
            for result in result_list:
                result['analysis_type'] = analysis_type
                all_results.append(result)
    
    # Create comparison summary under output_base_dir/normalized/
    norm_dir = os.path.join(output_base_dir, 'normalized')
    os.makedirs(norm_dir, exist_ok=True)
    comparison_file = os.path.join(norm_dir, 'comparison_summary.csv')
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(comparison_file, index=False)
    
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"\nResults saved to: {comparison_file}")
    if len(comparison_df) > 0:
        print("\nSummary by normalization method and age:")
        print(comparison_df.groupby(['normalization', 'age'])[['r', 'r_squared', 'p_value']].mean())
    
    return comparison_df


# Default paths for human-verified analysis
HUMAN_COUNTS_PATH = 'data/human_data/human_counts_dataframe.csv'
H2B_PER_SECTION_DIR = f'{BASE}/h2b'


def run_human_verified_analysis(
    human_counts_path=None,
    h2b_per_section_path=None,
    h2b_value_column='raw_intensity_sum',
    vsv_path=None,
    vsv_value_col=None,
    output_base_dir='results/by_section_raw_human_verified',
    area_normalize=True,
    vsv_prior_area_path=None,
):
    """
    Run the same analyses (complete_pairs, averaged_replicates, imputed_data, comparison)
    using H2B sections that appear in the human counts dataframe. Sections with has_human_count
    False are included as zero intensity (area still counted) so they are zeros, not omitted.
    VSV is never filtered or subset by human H2B counting constraints; only H2B is restricted
    to (identifier, section, region) present in both human counts and H2B. VSV is still
    area-normalized when loaded from a source that has tissue_area_pixels.
    Area-normalized intensity is computed per animal (per identifier) per region; the wide
    table is passed to comparison calculations without lumping across animals.
    """
    human_counts_path = human_counts_path or HUMAN_COUNTS_PATH
    h2b_per_section_path = h2b_per_section_path or f'{BASE}/h2b/h2b_raw.csv'
    vsv_path = vsv_path or f'{BASE}/vsv/vsv_raw_summary.csv'
    vsv_value_col = vsv_value_col if vsv_value_col is not None else 'raw_intensity_sum'
    if not os.path.isfile(human_counts_path):
        print(f"Human counts file not found: {human_counts_path}. Skipping human-verified analysis.")
        return
    human_df = pd.read_csv(human_counts_path)
    human_df['human_cell_count'] = pd.to_numeric(human_df['human_cell_count'], errors='coerce')
    # Parse has_human_count robustly (CSV may have string "True"/"False")
    human_df['has_human_count'] = human_df['has_human_count'].replace(
        {'True': True, 'true': True, 'False': False, 'false': False}
    ).fillna(False).astype(bool)
    # All (identifier, section, region) in human_df with normalized keys
    human_keys_raw = set(
        (str(row['identifier']), str(row['section']).strip(), str(row['region']))
        for _, row in human_df[['identifier', 'section', 'region']].drop_duplicates().iterrows()
    )
    # Sections where human count is False (treated as zeros)
    human_false_df = human_df[human_df['has_human_count'] == False]
    human_false_set = set(
        (str(row['identifier']), str(row['section']).strip(), str(row['region']))
        for _, row in human_false_df[['identifier', 'section', 'region']].drop_duplicates().iterrows()
    )
    if not h2b_per_section_path or not os.path.isfile(h2b_per_section_path):
        print(f"H2B per-section file not found: {h2b_per_section_path}. Skipping human-verified analysis.")
        return
    h2b_section = pd.read_csv(h2b_per_section_path)
    if 'image_filename' not in h2b_section.columns:
        print("H2B per-section CSV has no 'image_filename' column. Skipping human-verified analysis.")
        return
    h2b_section['section'] = h2b_section['image_filename'].str.extract(r'_(\d{3})_', expand=False)
    if h2b_section['section'].isna().all():
        print("Could not parse section from image_filename. Skipping human-verified analysis.")
        return
    h2b_section = h2b_section.dropna(subset=['section']).copy()
    h2b_section['section'] = h2b_section['section'].astype(str).str.strip()
    # Restrict to (identifier, section, region) present in both human_df and H2B (so we always have area)
    h2b_keys = set(
        (str(r['identifier']), str(r['section']), str(r['region']))
        for _, r in h2b_section[['identifier', 'section', 'region']].drop_duplicates().iterrows()
    )
    human_set_all = human_keys_raw & h2b_keys
    human_false_set = human_false_set & human_set_all
    if len(human_set_all) == 0:
        print("No (identifier, section, region) in both human counts and H2B. Skipping human-verified analysis.")
        return
    if h2b_value_column not in h2b_section.columns:
        print(f"Value column '{h2b_value_column}' not in H2B per-section CSV. Skipping.")
        return
    # Filter H2B to human_set_all; for sections in human_false_set use intensity 0 (area unchanged)
    key_in_human = h2b_section.apply(
        lambda r: (str(r['identifier']), str(r['section']), str(r['region'])) in human_set_all, axis=1
    )
    h2b_section = h2b_section[key_in_human].copy()
    row_key = h2b_section.apply(
        lambda r: (str(r['identifier']), str(r['section']), str(r['region'])), axis=1
    )
    h2b_section['_effective_intensity'] = np.where(
        row_key.isin(human_false_set), 0, h2b_section[h2b_value_column]
    )
    if area_normalize and 'tissue_area_pixels' in h2b_section.columns:
        agg = h2b_section.groupby(['age', 'identifier', 'region'], as_index=False).agg(
            intensity=('_effective_intensity', 'sum'), area=('tissue_area_pixels', 'sum')
        )
        agg['_value_for_pivot'] = agg['intensity'] / agg['area'].replace(0, np.nan)
        h2b_wide = _pivot_summary_to_wide(agg, '_value_for_pivot', 'H2b')
    else:
        agg = h2b_section.groupby(['age', 'identifier', 'region'], as_index=False)['_effective_intensity'].sum()
        agg = agg.rename(columns={'_effective_intensity': h2b_value_column})
        h2b_wide = _pivot_summary_to_wide(agg, h2b_value_column, 'H2b')
    h2b_wide['age'] = h2b_wide['age'].replace('p60', 'adult')
    h2b_wide = h2b_wide[h2b_wide['age'].isin(AGE_COHORTS)].copy()
    h2b_wide = h2b_wide.replace('', np.nan)
    for col in h2b_wide.columns:
        if col not in ('age', 'identifier'):
            h2b_wide[col] = pd.to_numeric(h2b_wide[col], errors='coerce')
    vsv_df = pd.read_csv(vsv_path)
    if vsv_value_col is not None and 'region' in vsv_df.columns:
        if area_normalize and 'tissue_area_pixels' in vsv_df.columns:
            vsv_df = vsv_df.copy()
            if vsv_prior_area_path and os.path.isfile(vsv_prior_area_path):
                prior_long = _load_vsv_prior_area(vsv_prior_area_path)
                vsv_merged = vsv_df.merge(prior_long, on=['identifier', 'region'], how='left')
                vsv_df['_area'] = vsv_merged['prior_area'].fillna(vsv_merged['tissue_area_pixels'])
            else:
                vsv_df['_area'] = vsv_df['tissue_area_pixels']
            area = vsv_df['_area'].replace(0, np.nan)
            vsv_df['_value_for_pivot'] = vsv_df[vsv_value_col] / area
            vsv_df = _pivot_summary_to_wide(vsv_df, '_value_for_pivot', 'VSV')
        else:
            vsv_df = _pivot_summary_to_wide(vsv_df, vsv_value_col, 'VSV')
    vsv_df['age'] = vsv_df['age'].replace('p60', 'adult')
    vsv_df = vsv_df[vsv_df['age'].isin(AGE_COHORTS)].copy()
    vsv_df = vsv_df.replace('', np.nan)
    for col in vsv_df.columns:
        if col not in ('age', 'identifier'):
            vsv_df[col] = pd.to_numeric(vsv_df[col], errors='coerce')
    h2b_regions = {c.split('-H2b')[0]: c for c in h2b_wide.columns if '-H2b' in c}
    vsv_regions = {c.replace('-VSV', ''): c for c in vsv_df.columns if '-VSV' in c}
    common_regions = sorted(set(h2b_regions.keys()) & set(vsv_regions.keys()))
    excluded_regions = ['VISp', 'RSPv', 'RSPd', 'RSPagl', 'VISpl']
    common_regions = [r for r in common_regions if r not in excluded_regions]
    region_mapping = {r: {'h2b_col': h2b_regions[r], 'vsv_col': vsv_regions[r]} for r in common_regions}
    if len(region_mapping) == 0:
        print("No common regions after exclusion. Skipping human-verified analysis.")
        return
    # Per-animal verification: wide tables have one row per (age, identifier); no aggregation across identifier before analysis functions.
    assert h2b_wide.drop_duplicates(['age', 'identifier']).shape[0] == len(h2b_wide), "H2B wide table should have one row per (age, identifier)"
    assert vsv_df.drop_duplicates(['age', 'identifier']).shape[0] == len(vsv_df), "VSV wide table should have one row per (age, identifier)"
    n_false_as_zero = len(human_false_set)
    print(f"\nHuman-verified: {len(human_set_all)} (identifier, section, region) in analysis "
          f"({n_false_as_zero} as zeros); {len(h2b_wide)} (age, identifier) in H2B wide; {len(common_regions)} regions.")
    run_dirs = setup_directories(output_base_dir)
    if not validate_csv_format(h2b_wide, vsv_df, region_mapping):
        print("Human-verified run validation failed. Skipping.")
        return
    complete_pairs_analysis(h2b_wide, vsv_df, region_mapping, output_base_dir=run_dirs['complete_pairs'])
    averaged_replicates_analysis(h2b_wide, vsv_df, region_mapping, output_base_dir=run_dirs['averaged_replicates'])
    imputed_data_analysis(h2b_wide, vsv_df, region_mapping, output_base_dir=run_dirs['imputed_data'])
    comparison_analysis(h2b_wide, vsv_df, region_mapping, output_base_dir=output_base_dir)
    print(f"Human-verified run complete. Results in {output_base_dir}")


def main(args=None):
    """Main execution function. Runs analysis for each dataset in DATASET_RUNS."""
    if args is None:
        args = argparse.Namespace(log=None, human_verified=False, no_area_normalize=False)
    area_normalize = not getattr(args, 'no_area_normalize', False)
    print("Starting Pearson Correlation Analysis (multi-dataset)")
    print("=" * 50)
    
    for run in DATASET_RUNS:
        name = run[0]
        h2b_path = run[1]
        vsv_path = run[2]
        h2b_value_col = run[3] if len(run) > 3 else None
        vsv_value_col = run[4] if len(run) > 4 else None
        output_base = os.path.join('results', name)
        print("\n" + "=" * 50)
        print(f"DATASET: {name}")
        print(f"  H2B: {h2b_path}")
        print(f"  VSV: {vsv_path}")
        print(f"  Output: {output_base}")
        print("=" * 50)
        
        # Create output directories for this run
        run_dirs = setup_directories(output_base)
        
        # Load and preprocess (area-normalized per animal per region when area_normalize=True)
        try:
            h2b_df, vsv_df, region_mapping = load_and_preprocess_data(
                h2b_path=h2b_path, vsv_path=vsv_path,
                h2b_value_col=h2b_value_col, vsv_value_col=vsv_value_col,
                area_normalize=area_normalize,
                vsv_prior_area_path=getattr(args, 'vsv_prior_area', None),
            )
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
        
        if not validate_csv_format(h2b_df, vsv_df, region_mapping):
            print(f"Skipping run {name} due to validation failure.")
            continue
        
        print(f"\nAge cohorts: {AGE_COHORTS}")
        print(f"Brain regions: {list(region_mapping.keys())}")
        
        # Run standard analyses (no normalization)
        print("\n" + "=" * 50)
        print("STANDARD ANALYSES (No Normalization)")
        print("=" * 50)
        complete_pairs_analysis(h2b_df, vsv_df, region_mapping, output_base_dir=run_dirs['complete_pairs'])
        averaged_replicates_analysis(h2b_df, vsv_df, region_mapping, output_base_dir=run_dirs['averaged_replicates'])
        imputed_data_analysis(h2b_df, vsv_df, region_mapping, output_base_dir=run_dirs['imputed_data'])
        
        # Run comparison analysis with all normalization methods
        comparison_analysis(h2b_df, vsv_df, region_mapping, output_base_dir=output_base)
        
        print(f"\nRun '{name}' complete. Results in {output_base}")
    
    print("\n" + "=" * 50)
    print("All dataset runs complete!")
    print("Results per run: results/<dataset_name>/")

    if getattr(args, 'human_verified', False):
        print("\n" + "=" * 50)
        print("HUMAN-VERIFIED ANALYSIS (H2B sections with positive human cell count only)")
        print("=" * 50)
        run_human_verified_analysis(
            human_counts_path=HUMAN_COUNTS_PATH,
            h2b_per_section_path=f'{BASE}/h2b/h2b_raw.csv',
            h2b_value_column='raw_intensity_sum',
            vsv_path=f'{BASE}/vsv/vsv_raw_summary.csv',
            vsv_value_col='raw_intensity_sum',
            output_base_dir='results/by_section_raw_human_verified',
            area_normalize=area_normalize,
            vsv_prior_area_path=getattr(args, 'vsv_prior_area', None),
        )

    # Regenerate correlation_summary.html from result CSVs (prior match + all datasets by r_RMSE)
    try:
        import build_correlation_summary
        build_correlation_summary.main(results_dir='results', output_path='correlation_summary.html')
    except Exception as e:
        print(f"Note: could not build correlation summary: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pearson correlation analysis (H2B vs VSV) across datasets.')
    parser.add_argument('--log', '-l', metavar='FILE', default=None,
                        help='Also write all output to this log file')
    parser.add_argument('--human-verified', action='store_true',
                        help='Also run analysis restricted to H2B sections with positive human cell count')
    parser.add_argument('--no-area-normalize', action='store_true',
                        help='Disable area normalization (intensity/tissue_area_pixels) for H2B and VSV')
    parser.add_argument('--vsv-prior-area', metavar='FILE', default=None,
                        help='Use this CSV for VSV area normalization (identifier/Animal x region columns) instead of tissue_area_pixels')
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
        tee_stdout = Tee(sys.stdout, log_path)
        tee_stderr = Tee(sys.stderr, log_path)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        try:
            main(args)
        finally:
            sys.stdout = tee_stdout._stream
            sys.stderr = tee_stderr._stream
            tee_stdout.close()
            tee_stderr.close()
            print(f"Log written to {log_path}", file=sys.__stdout__)
    else:
        main(args)
