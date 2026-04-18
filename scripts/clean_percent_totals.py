#!/usr/bin/env python3
"""
Clean wide-format percent-total CSVs into long-format summary CSVs for the
correlation analysis script (age, identifier, region, percent_total).
Reads from data/percent_total_data/; writes to data/dataframes_by_section/{h2b,vsv}/.
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PERCENT_DIR = BASE / "data" / "percent_total_data"
OUTPUT_H2B = BASE / "data" / "dataframes_by_section" / "h2b" / "h2b_percent_total_summary.csv"
OUTPUT_VSV = BASE / "data" / "dataframes_by_section" / "vsv" / "vsv_percent_total_summary.csv"

H2B_SOURCE = PERCENT_DIR / "h2b_percent_totals - Sheet1.csv"
VSV_SOURCE = PERCENT_DIR / "vsv_percent_totals - Sheet1 (1).csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize age (p60 -> adult) and identifier strings."""
    df = df.copy()
    df["age"] = df["age"].astype(str).str.strip().str.lower()
    df["age"] = df["age"].replace("p60", "adult")
    df["identifier"] = df["identifier"].astype(str).str.strip()
    df["percent_total"] = pd.to_numeric(df["percent_total"], errors="coerce")
    return df


def clean_h2b_percent_totals() -> pd.DataFrame:
    """Read H2B wide CSV, fix column names, melt to long format."""
    df = pd.read_csv(H2B_SOURCE, header=0)
    # First two columns are age and identifier (header may be %totals, empty)
    id_cols = list(df.columns[:2])
    region_cols = [c for c in df.columns[2:] if str(c).strip()]
    df = df.rename(columns={id_cols[0]: "age", id_cols[1]: "identifier"})
    df = df[["age", "identifier"] + region_cols]
    long = df.melt(
        id_vars=["age", "identifier"],
        value_vars=region_cols,
        var_name="region",
        value_name="percent_total",
    )
    long["region"] = long["region"].astype(str).str.strip()
    return _normalize(long)


def clean_vsv_percent_totals() -> pd.DataFrame:
    """Read VSV wide CSV (header row 1), fix column names, melt to long format."""
    df = pd.read_csv(VSV_SOURCE, header=1)
    # First column = age, second = identifier (Animal), rest = regions
    id_cols = list(df.columns[:2])
    region_cols = [c for c in df.columns[2:] if str(c).strip()]
    df = df.rename(columns={id_cols[0]: "age", id_cols[1]: "identifier"})
    df = df[["age", "identifier"] + region_cols]
    long = df.melt(
        id_vars=["age", "identifier"],
        value_vars=region_cols,
        var_name="region",
        value_name="percent_total",
    )
    long["region"] = long["region"].astype(str).str.strip()
    return _normalize(long)


def main():
    h2b_long = clean_h2b_percent_totals()
    vsv_long = clean_vsv_percent_totals()
    OUTPUT_H2B.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_VSV.parent.mkdir(parents=True, exist_ok=True)
    h2b_long.to_csv(OUTPUT_H2B, index=False)
    vsv_long.to_csv(OUTPUT_VSV, index=False)
    print(f"Wrote {OUTPUT_H2B} ({len(h2b_long)} rows)")
    print(f"Wrote {OUTPUT_VSV} ({len(vsv_long)} rows)")


if __name__ == "__main__":
    main()
