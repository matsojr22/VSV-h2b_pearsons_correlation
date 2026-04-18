#!/usr/bin/env python3
"""
Compare human cell counts to H2B intensity per timepoint (p3, p12, p20, p60).

Uses the same (identifier, section, region) set from human counts. Outputs
side-by-side bar plots (raw + softmax) with SEM and Pearson r per timepoint.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
HUMAN_COUNTS_PATH = DATA / "human_data" / "human_counts_dataframe.csv"
H2B_RAW_PATH = DATA / "dataframes_by_section" / "h2b" / "h2b_raw.csv"
VSV_HUMAN_VERIFIED_PATH = DATA / "human_data" / "vsv_human_verified.csv"
VSV_RAW_SUMMARY_PATH = DATA / "dataframes_by_section" / "vsv" / "vsv_raw_summary.csv"
VSV_PRIOR_AREA_PATH = DATA / "vsv_prior_total_area.csv"
OUT_DIR = BASE / "results" / "human_vs_intensity"
TIMEPOINTS = ["p3", "p12", "p20", "p60"]
EXCLUDED_REGIONS = ["VISp", "RSPv", "RSPd", "RSPagl", "VISpl"]


def load_human(human_path):
    """Load human counts, normalize section to 3-digit, parse has_human_count."""
    df = pd.read_csv(human_path)
    df["age"] = df["age"].astype(str).str.strip().str.lower()
    df["identifier"] = df["identifier"].astype(str)
    df["region"] = df["region"].astype(str)
    df["section"] = (
        pd.to_numeric(df["section"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )
    df["human_cell_count"] = pd.to_numeric(df["human_cell_count"], errors="coerce")
    df["has_human_count"] = (
        df["has_human_count"]
        .replace({"True": True, "true": True, "False": False, "false": False})
        .fillna(False)
        .astype(bool)
    )
    return df


def load_h2b_with_section(h2b_path):
    """Load H2B raw and parse section from image_filename."""
    df = pd.read_csv(h2b_path)
    df["age"] = df["age"].astype(str).str.strip().str.lower()
    df["identifier"] = df["identifier"].astype(str)
    df["region"] = df["region"].astype(str)
    sec = df["image_filename"].str.extract(r"_s?(\d{2,3})_", expand=False)
    if sec.isna().all():
        sec = df["image_filename"].str.extract(r"_(\d{3})_", expand=False)
    df["section"] = sec.dropna().astype(int).astype(str).str.zfill(3)
    df = df.dropna(subset=["section"]).copy()
    df["raw_intensity_sum"] = pd.to_numeric(df["raw_intensity_sum"], errors="coerce")
    df["tissue_area_pixels"] = pd.to_numeric(df["tissue_area_pixels"], errors="coerce").replace(0, np.nan)
    return df


def build_aggregates_per_timepoint(human_df, h2b_df):
    """
    For each (age, identifier, section, region) in human:
    - human: sum human_cell_count (0 where has_human_count False or NaN)
    - intensity: same keys; effective intensity 0 where has_human_count False, else raw_intensity_sum
    Aggregate to (age, identifier, region): human = sum(cells), intensity = sum(raw)/sum(area).
    Returns dict age -> (human_agg_df, intensity_agg_df) where each has identifier, region, value.
    """
    human_df = human_df.copy()
    human_df["human_val"] = np.where(
        human_df["has_human_count"],
        human_df["human_cell_count"].fillna(0),
        0,
    )
    human_agg = (
        human_df.groupby(["age", "identifier", "region"], as_index=False)["human_val"]
        .sum()
        .rename(columns={"human_val": "value"})
    )

    # Keys (identifier, section, region) per age for joining H2B
    human_keys = {}
    human_false = {}
    for age in TIMEPOINTS:
        g = human_df[human_df["age"] == age]
        human_keys[age] = set(
            (r["identifier"], r["section"], r["region"])
            for _, r in g[["identifier", "section", "region"]].drop_duplicates().iterrows()
        )
        gfalse = human_df[(human_df["age"] == age) & (~human_df["has_human_count"])]
        human_false[age] = set(
            (r["identifier"], r["section"], r["region"])
            for _, r in gfalse[["identifier", "section", "region"]].drop_duplicates().iterrows()
        )
    for age in TIMEPOINTS:
        if age not in human_false:
            human_false[age] = set()

    out = {}
    for age in TIMEPOINTS:
        h2b_age = h2b_df[h2b_df["age"] == age].copy()
        keys_age = human_keys.get(age, set())
        if not keys_age:
            out[age] = (pd.DataFrame(), pd.DataFrame())
            continue
        h2b_age["_key"] = h2b_age.apply(
            lambda r: (r["identifier"], r["section"], r["region"]), axis=1
        )
        h2b_age = h2b_age[h2b_age["_key"].isin(keys_age)].copy()
        false_age = human_false.get(age, set())
        h2b_age["_eff_intensity"] = np.where(
            h2b_age["_key"].isin(false_age),
            0,
            h2b_age["raw_intensity_sum"],
        )
        agg = (
            h2b_age.groupby(["age", "identifier", "region"], as_index=False)
            .agg(
                intensity_sum=("_eff_intensity", "sum"),
                area_sum=("tissue_area_pixels", "sum"),
            )
        )
        agg["value"] = agg["intensity_sum"] / agg["area_sum"].replace(0, np.nan)
        intensity_agg = agg[["age", "identifier", "region", "value"]].copy()

        h_agg_age = human_agg[human_agg["age"] == age][["identifier", "region", "value"]].copy()
        i_agg_age = intensity_agg[["identifier", "region", "value"]].copy()
        out[age] = (h_agg_age, i_agg_age)
    return out


def load_vsv_prior_area(path):
    """Load wide-format prior area CSV to long form (identifier, region, prior_area)."""
    df = pd.read_csv(path)
    if "Animal" in df.columns and "identifier" not in df.columns:
        df = df.rename(columns={"Animal": "identifier"})
    region_cols = [c for c in df.columns if c != "identifier"]
    long_df = df.melt(
        id_vars=["identifier"],
        value_vars=region_cols,
        var_name="region",
        value_name="prior_area",
    )
    long_df["prior_area"] = (
        pd.to_numeric(long_df["prior_area"], errors="coerce").replace(0, np.nan)
    )
    return long_df


def load_vsv_human_verified(path):
    """Load human-verified VSV; normalize age (adult -> p60); return age, identifier, region, value."""
    df = pd.read_csv(path)
    df["age"] = df["age"].astype(str).str.strip().str.lower().replace("adult", "p60")
    df["identifier"] = df["identifier"].astype(str)
    df["region"] = df["region"].astype(str)
    df["value"] = pd.to_numeric(df["human_verified_vsv"], errors="coerce")
    return df[["age", "identifier", "region", "value"]].copy()


def load_vsv_intensity(vsv_summary_path, prior_area_path):
    """Load VSV summary; normalize age (adult -> p60); area-normalize using prior where available."""
    vsv = pd.read_csv(vsv_summary_path)
    vsv["age"] = vsv["age"].astype(str).str.strip().str.lower().replace("adult", "p60")
    vsv["identifier"] = vsv["identifier"].astype(str)
    vsv["region"] = vsv["region"].astype(str)
    vsv["raw_intensity_sum"] = pd.to_numeric(vsv["raw_intensity_sum"], errors="coerce")
    vsv["tissue_area_pixels"] = pd.to_numeric(vsv["tissue_area_pixels"], errors="coerce").replace(0, np.nan)
    if prior_area_path and Path(prior_area_path).is_file():
        prior = load_vsv_prior_area(prior_area_path)
        prior["identifier"] = prior["identifier"].astype(str)
        prior["region"] = prior["region"].astype(str)
        vsv = vsv.merge(prior, on=["identifier", "region"], how="left")
        vsv["area"] = vsv["prior_area"].fillna(vsv["tissue_area_pixels"]).replace(0, np.nan)
    else:
        vsv["area"] = vsv["tissue_area_pixels"]
    vsv["value"] = vsv["raw_intensity_sum"] / vsv["area"]
    return vsv[["age", "identifier", "region", "value"]].copy()


def build_vsv_aggregates_per_timepoint(vsv_human_df, vsv_intensity_df):
    """For each age in TIMEPOINTS, return (human_vsv_df, intensity_vsv_df) with identifier, region, value."""
    out = {}
    for age in TIMEPOINTS:
        h = vsv_human_df[vsv_human_df["age"] == age][["identifier", "region", "value"]].copy()
        i = vsv_intensity_df[vsv_intensity_df["age"] == age][["identifier", "region", "value"]].copy()
        out[age] = (h, i)
    return out


def softmax_per_replicate(df, value_col="value"):
    """Per identifier: softmax across regions. Adds column 'softmax'."""
    out = df.copy()
    out["softmax"] = np.nan
    for ident, g in df.groupby("identifier"):
        vals = g[value_col].values.astype(float)
        mask = ~np.isnan(vals)
        if not np.any(mask):
            continue
        v = np.where(mask, vals, -np.inf)
        v_max = np.max(v)
        exp = np.exp(v - v_max)
        exp[~mask] = 0
        s = exp / np.sum(exp)
        s[~mask] = np.nan
        out.loc[g.index, "softmax"] = s
    return out


def plot_side_by_side(
    human_df,
    intensity_df,
    region_order,
    title_prefix,
    value_col,
    ylabel_left,
    ylabel_right,
    out_path,
    jitter=0.15,
    left_title="Human cell count",
    right_title="H2B intensity (area-norm)",
):
    """Two panels: left = human, right = intensity; same region order, bars + SEM + dots."""
    n_regions = len(region_order)
    x = np.arange(n_regions)
    width = 0.6

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(max(12, n_regions * 1.2), 5), sharey=False)

    for ax, df, ylabel, title in [
        (ax_l, human_df, ylabel_left, left_title),
        (ax_r, intensity_df, ylabel_right, right_title),
    ]:
        wide = df.pivot(index="region", columns="identifier", values=value_col)
        wide = wide.reindex(region_order)
        means = wide.mean(axis=1)
        sems = wide.sem(axis=1)
        ax.bar(x, means, width, yerr=sems, capsize=4, error_kw={"linewidth": 1.2})
        for i, reg in enumerate(region_order):
            if reg not in wide.index:
                continue
            row = wide.loc[reg].dropna()
            for j, (_, val) in enumerate(row.items()):
                jitter_x = x[i] + (j - (len(row) - 1) / 2) * jitter
                ax.scatter(jitter_x, val, color="black", s=24, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(region_order, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def pearson_for_timepoint(human_df, intensity_df):
    """Merge on (identifier, region), return n, r, r_squared, p_value."""
    merged = human_df.merge(
        intensity_df,
        on=["identifier", "region"],
        how="inner",
        suffixes=("_human", "_intensity"),
    )
    if merged.empty or len(merged) < 2:
        return 0, np.nan, np.nan, np.nan
    h = merged["value_human"].astype(float).values
    i = merged["value_intensity"].astype(float).values
    valid = ~(np.isnan(h) | np.isnan(i))
    if np.sum(valid) < 2:
        return int(np.sum(valid)), np.nan, np.nan, np.nan
    h, i = h[valid], i[valid]
    if np.std(h) == 0 or np.std(i) == 0:
        return len(h), np.nan, np.nan, np.nan
    r, p = stats.pearsonr(h, i)
    return len(h), r, r * r, p


def main():
    parser = argparse.ArgumentParser(
        description="Compare human cell counts to H2B intensity per timepoint."
    )
    parser.add_argument("--human", type=Path, default=HUMAN_COUNTS_PATH)
    parser.add_argument("--h2b", type=Path, default=H2B_RAW_PATH)
    parser.add_argument("--vsv-human", type=Path, default=VSV_HUMAN_VERIFIED_PATH)
    parser.add_argument("--vsv-summary", type=Path, default=VSV_RAW_SUMMARY_PATH)
    parser.add_argument("--vsv-prior", type=Path, default=VSV_PRIOR_AREA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    human_df = load_human(args.human)
    h2b_df = load_h2b_with_section(args.h2b)
    aggregates = build_aggregates_per_timepoint(human_df, h2b_df)

    correlation_rows = []

    for age in TIMEPOINTS:
        h_agg, i_agg = aggregates[age]
        if h_agg.empty or i_agg.empty:
            correlation_rows.append(
                {"timepoint": age, "n": 0, "r": np.nan, "r_squared": np.nan, "p_value": np.nan}
            )
            continue

        # Common regions (intersection, excluding comparative-area exclusions)
        reg_h = set(h_agg["region"].dropna().unique())
        reg_i = set(i_agg["region"].dropna().unique())
        region_order = [r for r in sorted(reg_h & reg_i) if r not in EXCLUDED_REGIONS]
        if not region_order:
            correlation_rows.append(
                {"timepoint": age, "n": 0, "r": np.nan, "r_squared": np.nan, "p_value": np.nan}
            )
            continue

        h_agg = h_agg[h_agg["region"].isin(region_order)].copy()
        i_agg = i_agg[i_agg["region"].isin(region_order)].copy()

        n, r, r_sq, p = pearson_for_timepoint(h_agg, i_agg)
        correlation_rows.append(
            {"timepoint": age, "n": n, "r": r, "r_squared": r_sq, "p_value": p}
        )

        # Raw side-by-side
        plot_side_by_side(
            h_agg,
            i_agg,
            region_order,
            f"{age}: Human cell count vs H2B intensity",
            "value",
            "Human cell count",
            "Area-normalized intensity",
            out_dir / f"{age}_human_vs_intensity_bars.png",
        )

        # Softmax side-by-side
        h_soft = softmax_per_replicate(h_agg)
        i_soft = softmax_per_replicate(i_agg)
        plot_side_by_side(
            h_soft,
            i_soft,
            region_order,
            f"{age}: Human vs H2B (softmax 0–1)",
            "softmax",
            "Softmax (0–1)",
            "Softmax (0–1)",
            out_dir / f"{age}_human_vs_intensity_bars_softmax.png",
        )

    corr_df = pd.DataFrame(correlation_rows)
    corr_path = out_dir / "human_vs_intensity_correlation_by_timepoint.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved H2B plots and {corr_path}")
    print("H2B (human cell count vs H2B intensity):")
    print(corr_df.to_string(index=False))

    # VSV: human-verified VSV vs VSV intensity
    vsv_human_df = load_vsv_human_verified(args.vsv_human)
    vsv_intensity_df = load_vsv_intensity(args.vsv_summary, args.vsv_prior)
    vsv_aggregates = build_vsv_aggregates_per_timepoint(vsv_human_df, vsv_intensity_df)
    correlation_rows_vsv = []

    for age in TIMEPOINTS:
        h_agg, i_agg = vsv_aggregates[age]
        if h_agg.empty or i_agg.empty:
            correlation_rows_vsv.append(
                {"timepoint": age, "n": 0, "r": np.nan, "r_squared": np.nan, "p_value": np.nan}
            )
            continue

        reg_h = set(h_agg["region"].dropna().unique())
        reg_i = set(i_agg["region"].dropna().unique())
        region_order = [r for r in sorted(reg_h & reg_i) if r not in EXCLUDED_REGIONS]
        if not region_order:
            correlation_rows_vsv.append(
                {"timepoint": age, "n": 0, "r": np.nan, "r_squared": np.nan, "p_value": np.nan}
            )
            continue

        h_agg = h_agg[h_agg["region"].isin(region_order)].copy()
        i_agg = i_agg[i_agg["region"].isin(region_order)].copy()

        n, r, r_sq, p = pearson_for_timepoint(h_agg, i_agg)
        correlation_rows_vsv.append(
            {"timepoint": age, "n": n, "r": r, "r_squared": r_sq, "p_value": p}
        )

        plot_side_by_side(
            h_agg,
            i_agg,
            region_order,
            f"{age}: Human verified VSV vs VSV intensity",
            "value",
            "Human verified VSV",
            "Area-normalized intensity",
            out_dir / f"{age}_vsv_human_vs_intensity_bars.png",
            left_title="Human verified VSV",
            right_title="VSV intensity (area-norm)",
        )

        h_soft = softmax_per_replicate(h_agg)
        i_soft = softmax_per_replicate(i_agg)
        plot_side_by_side(
            h_soft,
            i_soft,
            region_order,
            f"{age}: Human verified VSV vs VSV (softmax 0–1)",
            "softmax",
            "Softmax (0–1)",
            "Softmax (0–1)",
            out_dir / f"{age}_vsv_human_vs_intensity_bars_softmax.png",
            left_title="Human verified VSV",
            right_title="VSV intensity",
        )

    vsv_corr_df = pd.DataFrame(correlation_rows_vsv)
    vsv_corr_path = out_dir / "vsv_human_vs_intensity_correlation_by_timepoint.csv"
    vsv_corr_df.to_csv(vsv_corr_path, index=False)
    print(f"\nSaved VSV plots and {vsv_corr_path}")
    print("VSV (human verified VSV vs VSV intensity):")
    print(vsv_corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
