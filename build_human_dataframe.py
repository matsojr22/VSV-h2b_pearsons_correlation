#!/usr/bin/env python3
"""
Build a consolidated human counts dataframe from data/human_data/ CSVs.

Each CSV: "H2B Quant - {Age} - {Identifier}.csv"
- Row 0: header "Section", then columns 1, 2, ... 95 (section indices).
- Rows 1-12: first column = region (RSPagl, RSPd, ...), remaining = human cell count
  for that (region, section). Empty = not counted; 0 = counted, zero cells.

Output: data/human_data/human_counts_dataframe.csv
  Columns: age, identifier, section (3-digit), region, human_cell_count, has_human_count
  One row per (age, identifier, region, section). Section normalized to "001"-"095"
  for joining with H2B image_filename (e.g. M677_062_RSPagl.tif).
"""

import re
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_HUMAN_DATA_DIR = Path(__file__).resolve().parent / "data" / "human_data"
OUTPUT_CSV = "human_counts_dataframe.csv"
CSV_PATTERN = "H2B Quant - *.csv"
DATA_ROW_START = 1   # 0-based index of first region row
DATA_ROW_END = 13    # exclusive: rows 1-12 (indices 1..12) contain region x section counts


def parse_filename(path: Path) -> tuple[str, str] | None:
    """Parse 'H2B Quant - P3 - M729.csv' -> (age='p3', identifier='M729'). Returns None if no match."""
    m = re.match(r"H2B Quant - (P\d+|P60) - (M\d+)\.csv", path.name, re.IGNORECASE)
    if not m:
        return None
    age = m.group(1).lower()  # P3 -> p3, P60 -> p60
    identifier = m.group(2)    # M729
    return (age, identifier)


def load_one_csv(path: Path, age: str, identifier: str) -> pd.DataFrame:
    """Read first 13 rows of a human quant CSV; return long-form rows (region, section, human_cell_count)."""
    df = pd.read_csv(path, nrows=13)
    if df.empty or df.shape[0] < 2:
        return pd.DataFrame()

    # First column = region; rest = section numbers (as column headers)
    region_col = df.columns[0]
    section_cols = [c for c in df.columns[1:] if str(c).isdigit()]
    if not section_cols:
        return pd.DataFrame()

    # Take only data rows (1-12) and valid region names
    data = df.iloc[DATA_ROW_START:DATA_ROW_END].copy()
    data = data[data[region_col].notna() & (data[region_col].astype(str).str.len() > 0)]

    long = data.melt(
        id_vars=[region_col],
        value_vars=section_cols,
        var_name="section_num",
        value_name="human_cell_count",
    )
    long = long.rename(columns={region_col: "region"})
    long["section_num"] = pd.to_numeric(long["section_num"], errors="coerce")
    long = long.dropna(subset=["section_num"])
    long["human_cell_count"] = pd.to_numeric(long["human_cell_count"], errors="coerce")
    long["section"] = long["section_num"].astype(int).astype(str).str.zfill(3)
    long["has_human_count"] = long["human_cell_count"].notna()
    long["age"] = age
    long["identifier"] = identifier
    long = long[["age", "identifier", "section", "region", "human_cell_count", "has_human_count"]]
    return long.drop(columns=["section_num"], errors="ignore")


def build_human_dataframe(human_data_dir: Path) -> pd.DataFrame:
    """Load all H2B Quant CSVs and return one consolidated dataframe."""
    if not human_data_dir.is_dir():
        raise FileNotFoundError(f"Human data directory not found: {human_data_dir}")

    frames = []
    for path in sorted(human_data_dir.glob(CSV_PATTERN)):
        parsed = parse_filename(path)
        if not parsed:
            continue
        age, identifier = parsed
        one = load_one_csv(path, age, identifier)
        if not one.empty:
            frames.append(one)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Drop rows with empty region (safety)
    out = out[out["region"].astype(str).str.strip().str.len() > 0]
    out = out.sort_values(["identifier", "section", "region"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build human counts dataframe from data/human_data/ CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_HUMAN_DATA_DIR,
        help="Directory containing H2B Quant - *.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input-dir>/human_counts_dataframe.csv)",
    )
    parser.add_argument(
        "--report-h2b-overlap",
        action="store_true",
        help="Report overlap with existing H2B identifiers/sections (h2b_raw)",
    )
    args = parser.parse_args()

    human_data_dir = args.input_dir
    output_path = args.output or (human_data_dir / OUTPUT_CSV)

    df = build_human_dataframe(human_data_dir)
    if df.empty:
        print("No data loaded. Check --input-dir and CSV filenames (H2B Quant - P* - M*.csv).")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")

    with_count = df[df["has_human_count"]]
    n_with = len(with_count)
    n_pairs = with_count.drop_duplicates(["identifier", "section"]).shape[0]
    print(f"Rows with has_human_count=True: {n_with}; unique (identifier, section): {n_pairs}")

    if args.report_h2b_overlap:
        root = Path(__file__).resolve().parent
        for candidate in [
            root / "data" / "dataframes_by_section" / "h2b" / "h2b_raw.csv",
            root / "data" / "fishing" / "h2b" / "h2b_raw.csv",
        ]:
            if candidate.exists():
                h2b_raw_path = candidate
                break
        else:
            h2b_raw_path = None
        if h2b_raw_path is not None:
            h2b = pd.read_csv(h2b_raw_path, nrows=0)
            if "image_filename" in h2b.columns:
                h2b_full = pd.read_csv(h2b_raw_path)
                # Parse section from image_filename: M677_062_RSPagl.tif -> 062
                h2b_full["section"] = h2b_full["image_filename"].str.extract(r"_(\d{3})_", expand=False)
                h2b_ids = set(h2b_full["identifier"].dropna().unique())
                h2b_sections = set(h2b_full[["identifier", "section"]].drop_duplicates().itertuples(index=False))
                human_ids = set(df["identifier"].unique())
                human_pairs = set(df[df["has_human_count"]][["identifier", "section"]].drop_duplicates().itertuples(index=False))
                overlap_ids = human_ids & h2b_ids
                overlap_pairs = human_pairs & h2b_sections
                print(f"Identifiers in both human and H2B: {len(overlap_ids)} ({sorted(overlap_ids)})")
                print(f"(identifier, section) pairs with human count that exist in H2B: {len(overlap_pairs)}")
        else:
            print("H2B raw file not found for overlap report (tried data/dataframes_by_section/h2b/h2b_raw.csv, data/fishing/h2b/h2b_raw.csv).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
