#!/usr/bin/env python3
"""
Augment a merged dataset with Lat, Lon, and Height_m columns using ONE metadata Excel file.

- Metadata: /Users/lisa-marieweidl/Desktop/Corr Height/all_metadata.xlsx
  (one or more sheets; auto-detects the sheet that has ID, Lat, Lon, Height_m)
- Merged dataset: /Users/lisa-marieweidl/Desktop/MergingAll/Water_Weather_Final_Prep.xlsx
- Inserts Lat, Lon, Height_m right after column B (index 1)
- Forces ID to string; handles European decimal commas
- Writes output next to the merged dataset with suffix "_geo"

Dependencies: pandas, openpyxl
"""

from pathlib import Path
import sys
import pandas as pd

# ============================ CONFIG (edit if needed) =========================
META_PATH = Path(r"/Users/lisa-marieweidl/Desktop/Corr Height/all_metadata.xlsx")
MERGED_INPUT = Path(r"/Users/lisa-marieweidl/Desktop/MergingAll/Water_Weather_Final_Prep.xlsx")
OUTPUT_PATH = MERGED_INPUT.with_name(MERGED_INPUT.stem + "_geo" + MERGED_INPUT.suffix)

ID_COL = "ID"
LAT_COL = "Lat"
LON_COL = "Lon"
HGT_COL = "Height_m"

INSERT_AFTER_COL_INDEX = 1
OVERWRITE_EXISTING = True

# Set these if you want to force a specific sheet; otherwise leave as None
META_SHEET_NAME = None
MERGED_SHEET_NAME = None
# ============================================================================


def read_any_table(path: Path, sheet_name=None) -> pd.DataFrame:
    """Read CSV/Excel. For Excel with sheet_name=None, load the FIRST sheet."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        if sheet_name is None:
            # load first sheet name explicitly
            xls = pd.ExcelFile(path)
            if not xls.sheet_names:
                raise ValueError(f"No sheets found in workbook: {path}")
            return pd.read_excel(path, sheet_name=xls.sheet_names[0])
        df = pd.read_excel(path, sheet_name=sheet_name)
        # If someone passed sheet_name=None accidentally and got a dict, pick the first
        if isinstance(df, dict):
            first_name = next(iter(df.keys()))
            df = df[first_name]
        return df
    # CSV
    return pd.read_csv(path)


def write_any_table(df: pd.DataFrame, path: Path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def _normalize_numeric_comma(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Replace comma decimals and cast to numeric for given columns."""
    for c in cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(",", ".", regex=False).str.strip()
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_metadata_excel(meta_path: Path, expected_cols=None, sheet_name=None) -> pd.DataFrame:
    """Load metadata from Excel; auto-detect the sheet containing required columns."""
    if expected_cols is None:
        expected_cols = {ID_COL, LAT_COL, LON_COL, HGT_COL}

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata Excel not found: {meta_path}")

    if sheet_name is not None:
        df = pd.read_excel(meta_path, sheet_name=sheet_name)
        df.columns = [c.strip() for c in df.columns]
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{sheet_name}' missing columns: {missing} "
                             f"(found: {df.columns.tolist()})")
    else:
        xls = pd.ExcelFile(meta_path)
        found = None
        for sh in xls.sheet_names:
            tmp = pd.read_excel(meta_path, sheet_name=sh)
            tmp.columns = [c.strip() for c in tmp.columns]
            if expected_cols.issubset(tmp.columns):
                found = tmp
                break
        if found is None:
            raise ValueError(f"No sheet in {meta_path} contains all columns {expected_cols}. "
                             f"Sheets found: {xls.sheet_names}")
        df = found

    df = df[[ID_COL, LAT_COL, LON_COL, HGT_COL]].copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = _normalize_numeric_comma(df, [LAT_COL, LON_COL, HGT_COL])

    # Deduplicate by ID (keep first)
    df = (df.groupby(ID_COL, as_index=False)
            .agg({LAT_COL: "first", LON_COL: "first", HGT_COL: "first"}))
    return df


def insert_after_column_b(df: pd.DataFrame, cols_to_insert) -> pd.DataFrame:
    """Insert specified columns right after column B (index 1)."""
    cols = list(df.columns)
    for c in cols_to_insert:
        if c in cols:
            cols.remove(c)
    insert_at = min(INSERT_AFTER_COL_INDEX + 1, len(cols))
    for c in reversed(cols_to_insert):
        cols.insert(insert_at, c)
    return df[cols]


def augment_with_geo(merged_path: Path, meta_df: pd.DataFrame) -> pd.DataFrame:
    df = read_any_table(merged_path, sheet_name=MERGED_SHEET_NAME)
    if isinstance(df, dict):
        # safety: if a dict slipped through, take first sheet
        first_name = next(iter(df.keys()))
        df = df[first_name]
    if ID_COL not in df.columns:
        raise ValueError(f"'{ID_COL}' not found in merged dataset columns: {df.columns.tolist()}")
    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    if not OVERWRITE_EXISTING:
        tmp_meta = meta_df.rename(columns={
            ID_COL: "__KEY__",
            LAT_COL: "__Lat_meta__",
            LON_COL: "__Lon_meta__",
            HGT_COL: "__Height_meta__"
        })
        merged = df.merge(tmp_meta, how="left", left_on=ID_COL, right_on="__KEY__")
        merged = merged.drop(columns=["__KEY__"])
        for c in (LAT_COL, LON_COL, HGT_COL):
            if c not in merged.columns:
                merged[c] = pd.NA
        merged[LAT_COL] = merged[LAT_COL].combine_first(merged["__Lat_meta__"])
        merged[LON_COL] = merged[LON_COL].combine_first(merged["__Lon_meta__"])
        merged[HGT_COL] = merged[HGT_COL].combine_first(merged["__Height_meta__"])
        merged = merged.drop(columns=["__Lat_meta__", "__Lon_meta__", "__Height_meta__"])
    else:
        tmp_meta = meta_df.rename(columns={ID_COL: "__KEY__"})
        merged = df.merge(tmp_meta, how="left", left_on=ID_COL, right_on="__KEY__")
        merged = merged.drop(columns=["__KEY__"])
        for c in (LAT_COL, LON_COL, HGT_COL):
            if c not in merged.columns:
                merged[c] = pd.NA

    merged = insert_after_column_b(merged, [LAT_COL, LON_COL, HGT_COL])
    return merged


def main():
    try:
        meta_df = load_metadata_excel(META_PATH, sheet_name=META_SHEET_NAME)
    except Exception as e:
        print(f"[ERROR] Loading metadata failed: {e}")
        sys.exit(1)

    try:
        out_df = augment_with_geo(MERGED_INPUT, meta_df)
    except Exception as e:
        print(f"[ERROR] Augmenting merged dataset failed: {e}")
        sys.exit(2)

    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_any_table(out_df, OUTPUT_PATH)
    except Exception as e:
        print(f"[ERROR] Writing output failed: {e}")
        sys.exit(3)

    missing_ids = out_df[out_df[[LAT_COL, LON_COL, HGT_COL]].isna().all(axis=1)][ID_COL].tolist()
    print(f"[OK] Saved augmented dataset -> {OUTPUT_PATH}")
    if missing_ids:
        print(f"[NOTE] {len(missing_ids)} rows had no metadata match by ID. "
              f"Examples: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")


if __name__ == "__main__":
    main()
