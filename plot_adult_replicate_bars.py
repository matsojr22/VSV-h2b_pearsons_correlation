#!/usr/bin/env python3
"""
Adult H2B and VSV replicate bar plots (raw and softmax-scaled).

Uses human-defined dataset flags and prior VSV area. Builds H2B from per-section
data + human counts (effective intensity 0 where has_human_count False); builds VSV
from raw summary + prior area for normalization. Outputs four PNGs to adult_raw_diagnostic.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths (repo root relative)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
HUMAN_COUNTS_PATH = DATA / "human_data" / "human_counts_dataframe.csv"
H2B_RAW_PATH = DATA / "dataframes_by_section" / "h2b" / "h2b_raw.csv"
VSV_RAW_SUMMARY_PATH = DATA / "dataframes_by_section" / "vsv" / "vsv_raw_summary.csv"
VSV_PRIOR_AREA_PATH = DATA / "vsv_prior_total_area.csv"
OUT_DIR = BASE / "adult_raw_diagnostic"
VALUE_COL = "raw_intensity_sum"
EXCLUDED_REGIONS = ["VISp", "RSPv", "RSPd", "RSPagl", "VISpl"]


def load_vsv_prior_area(path):
    """Load wide-format prior area CSV to long form (identifier, region, prior_area)."""
    df = pd.read_csv(path)
    if "Animal" in df.columns and "identifier" not in df.columns:
        df = df.rename(columns={"Animal": "identifier"})
    if "identifier" not in df.columns:
        raise ValueError(f"Prior area CSV must have 'identifier' or 'Animal'; got {list(df.columns)}")
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


def build_human_defined_set(human_path):
    """(identifier, region) with at least one has_human_count True for adult."""
    human_df = pd.read_csv(human_path)
    human_df["age"] = human_df["age"].astype(str).str.strip().replace("p60", "adult")
    human_df = human_df[human_df["age"] == "adult"].copy()
    human_df["has_human_count"] = human_df["has_human_count"].replace(
        {"True": True, "true": True, "False": False, "false": False}
    ).fillna(False).astype(bool)
    included = human_df[human_df["has_human_count"]].drop_duplicates(
        subset=["identifier", "region"]
    )
    return set(
        (str(r["identifier"]), str(r["region"]))
        for _, r in included[["identifier", "region"]].iterrows()
    )


def build_h2b_replicate_table(h2b_path, human_path, human_set, value_col):
    """H2B replicate-level: per-section + human flags, aggregate by (identifier, region)."""
    h2b = pd.read_csv(h2b_path)
    h2b["age"] = h2b["age"].astype(str).str.strip().replace("p60", "adult")
    h2b = h2b[h2b["age"] == "adult"].copy()

    # Parse section: support _s039_ or _039_
    sec = h2b["image_filename"].str.extract(r"_s?(\d{2,3})_", expand=False)
    if sec.isna().all():
        sec = h2b["image_filename"].str.extract(r"_(\d{3})_", expand=False)
    h2b["section"] = sec.dropna().astype(int).astype(str).str.zfill(3)
    h2b = h2b.dropna(subset=["section"]).copy()

    human_df = pd.read_csv(human_path)
    human_df["age"] = human_df["age"].astype(str).str.strip().replace("p60", "adult")
    human_df = human_df[human_df["age"] == "adult"].copy()
    human_df["has_human_count"] = human_df["has_human_count"].replace(
        {"True": True, "true": True, "False": False, "false": False}
    ).fillna(False).astype(bool)
    human_df["identifier"] = human_df["identifier"].astype(str)
    # Normalize section to 3-digit string to match H2B (e.g. 28 -> 028)
    human_df["section"] = (
        pd.to_numeric(human_df["section"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )
    human_df["region"] = human_df["region"].astype(str)

    # Keys in human with has_human_count
    human_true = set(
        (r["identifier"], r["section"], r["region"])
        for _, r in human_df[human_df["has_human_count"]][
            ["identifier", "section", "region"]
        ].drop_duplicates().iterrows()
    )
    human_false = set(
        (r["identifier"], r["section"], r["region"])
        for _, r in human_df[~human_df["has_human_count"]][
            ["identifier", "section", "region"]
        ].drop_duplicates().iterrows()
    )
    human_all = set(
        (r["identifier"], r["section"], r["region"])
        for _, r in human_df[["identifier", "section", "region"]].drop_duplicates().iterrows()
    )

    h2b["identifier"] = h2b["identifier"].astype(str)
    h2b["region"] = h2b["region"].astype(str)
    h2b_keys = h2b.apply(
        lambda r: (r["identifier"], r["section"], r["region"]), axis=1
    )
    keep = h2b_keys.isin(human_all)
    h2b = h2b.loc[keep].copy()
    row_key = h2b.apply(
        lambda r: (r["identifier"], r["section"], r["region"]), axis=1
    )
    intensity = np.where(
        row_key.isin(human_false), 0, pd.to_numeric(h2b[value_col], errors="coerce")
    )
    h2b["_eff_intensity"] = intensity
    area = pd.to_numeric(h2b["tissue_area_pixels"], errors="coerce").replace(0, np.nan)
    agg = (
        h2b.groupby(["identifier", "region"], as_index=False)
        .agg(intensity=("_eff_intensity", "sum"), area=("tissue_area_pixels", "sum"))
    )
    agg["value"] = agg["intensity"] / agg["area"]
    mask = agg.apply(
        lambda r: (str(r["identifier"]), str(r["region"])) in human_set, axis=1
    )
    agg = agg.loc[mask, ["identifier", "region", "value"]].copy()
    return agg


def build_vsv_replicate_table(vsv_summary_path, prior_area_path, human_set, value_col):
    """VSV replicate-level: raw summary + prior area for normalization."""
    vsv = pd.read_csv(vsv_summary_path)
    vsv["age"] = vsv["age"].astype(str).str.strip().replace("p60", "adult")
    vsv = vsv[vsv["age"] == "adult"].copy()

    prior = load_vsv_prior_area(prior_area_path)
    prior["identifier"] = prior["identifier"].astype(str)
    prior["region"] = prior["region"].astype(str)
    vsv["identifier"] = vsv["identifier"].astype(str)
    vsv["region"] = vsv["region"].astype(str)
    merged = vsv.merge(
        prior,
        on=["identifier", "region"],
        how="left",
        suffixes=("", "_prior"),
    )
    merged["area"] = merged["prior_area"].fillna(
        pd.to_numeric(merged["tissue_area_pixels"], errors="coerce")
    ).replace(0, np.nan)
    merged["value"] = (
        pd.to_numeric(merged[value_col], errors="coerce") / merged["area"]
    )
    merged = merged[
        merged.apply(
            lambda r: (str(r["identifier"]), str(r["region"])) in human_set, axis=1
        )
    ]
    return merged[["identifier", "region", "value"]].copy()


def softmax_per_replicate(df, value_col="value"):
    """Per identifier: softmax across regions (only non-NaN). Returns same shape, values in (0,1)."""
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


def plot_bars(
    df,
    value_col,
    title,
    ylabel,
    region_order,
    out_path,
    jitter=0.15,
):
    """Bar plot: mean per region, SEM, dots per replicate (jittered)."""
    # Pivot to (region x identifier) for mean/SEM and dots
    wide = df.pivot(index="region", columns="identifier", values=value_col)
    wide = wide.reindex(region_order)
    means = wide.mean(axis=1)
    sems = wide.sem(axis=1)
    n_regions = len(region_order)
    x = np.arange(n_regions)
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(8, n_regions * 0.8), 5))
    bars = ax.bar(x, means, width, yerr=sems, capsize=4, error_kw={"linewidth": 1.2})
    # Dots per replicate
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
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot adult H2B/VSV replicate bars (raw + softmax) to adult_raw_diagnostic."
    )
    parser.add_argument("--human", type=Path, default=HUMAN_COUNTS_PATH)
    parser.add_argument("--h2b", type=Path, default=H2B_RAW_PATH)
    parser.add_argument("--vsv", type=Path, default=VSV_RAW_SUMMARY_PATH)
    parser.add_argument("--vsv-prior", type=Path, default=VSV_PRIOR_AREA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    human_set = build_human_defined_set(args.human)
    if not human_set:
        raise SystemExit("No (identifier, region) with has_human_count True for adult.")

    # Regions that are in the human-defined set (for fallback when VSV uses different identifiers)
    regions_in_human_set = set(r for (_, r) in human_set)

    h2b_rep = build_h2b_replicate_table(
        args.h2b, args.human, human_set, VALUE_COL
    )
    vsv_rep = build_vsv_replicate_table(
        args.vsv, args.vsv_prior, human_set, VALUE_COL
    )
    # If VSV has no (identifier, region) in human_set (e.g. different animals), keep VSV by region only (comparative areas)
    comparative_regions = regions_in_human_set - set(EXCLUDED_REGIONS)
    if vsv_rep.empty and comparative_regions:
        vsv_raw = pd.read_csv(args.vsv)
        vsv_raw["age"] = vsv_raw["age"].astype(str).str.strip().replace("p60", "adult")
        vsv_raw = vsv_raw[vsv_raw["age"] == "adult"].copy()
        prior = load_vsv_prior_area(args.vsv_prior)
        vsv_raw["identifier"] = vsv_raw["identifier"].astype(str)
        vsv_raw["region"] = vsv_raw["region"].astype(str)
        prior["identifier"] = prior["identifier"].astype(str)
        prior["region"] = prior["region"].astype(str)
        merged = vsv_raw.merge(prior, on=["identifier", "region"], how="left")
        merged["area"] = merged["prior_area"].fillna(
            pd.to_numeric(merged["tissue_area_pixels"], errors="coerce")
        ).replace(0, np.nan)
        merged["value"] = pd.to_numeric(merged[VALUE_COL], errors="coerce") / merged["area"]
        vsv_rep = merged[merged["region"].isin(comparative_regions)][
            ["identifier", "region", "value"]
        ].copy()
    if h2b_rep.empty or vsv_rep.empty:
        raise SystemExit(
            "H2B or VSV replicate table is empty after human filter. Check data paths and human set."
        )

    region_order = [
        r
        for r in sorted(
            set(h2b_rep["region"].dropna().unique())
            & set(vsv_rep["region"].dropna().unique())
        )
        if r not in EXCLUDED_REGIONS
    ]
    if not region_order:
        raise SystemExit(
            "No common regions between H2B and VSV after human filter."
        )

    # Raw plots
    plot_bars(
        h2b_rep,
        "value",
        "Adult H2B (area-normalized)",
        "Area-normalized intensity",
        region_order,
        out_dir / "adult_h2b_replicate_bars.png",
    )
    plot_bars(
        vsv_rep,
        "value",
        "Adult VSV (area-normalized, prior area)",
        "Area-normalized intensity",
        region_order,
        out_dir / "adult_vsv_replicate_bars.png",
    )

    # Softmax
    h2b_soft = softmax_per_replicate(h2b_rep)
    vsv_soft = softmax_per_replicate(vsv_rep)
    plot_bars(
        h2b_soft,
        "softmax",
        "Adult H2B (softmax, 0–1)",
        "Softmax (0–1)",
        region_order,
        out_dir / "adult_h2b_replicate_bars_softmax.png",
    )
    plot_bars(
        vsv_soft,
        "softmax",
        "Adult VSV (softmax, 0–1)",
        "Softmax (0–1)",
        region_order,
        out_dir / "adult_vsv_replicate_bars_softmax.png",
    )

    print(f"Saved 4 plots to {out_dir}")


if __name__ == "__main__":
    main()
