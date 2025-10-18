import pandas as pd

MODE = "weather"   # or "weather"

# Drop columns (C→end) with more than this fraction missing
MISSING_DROP_THRESHOLD = 0.20

# Define input/output files
FILEMAP = {
    "weather": {
        "input":  "Weather_2109.xlsx",
        "output": "Cleaned_Weather_2109.xlsx",
    },
    "water": {
        "input":  "Water_2109.xlsx",
        "output": "Cleaned_Water_2109.xlsx",
    },
}


# =========================
# Helper functions
# =========================
def print_structure(df: pd.DataFrame, title: str):
    """Print a compact structure table: dtype, n_rows, n_missing per column."""
    n = len(df)
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dt) for dt in df.dtypes],
        "n_rows": n,
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
    })
    info = info.set_index("column")[["dtype", "n_rows", "n_missing"]]
    print(f"\n=== {title} ===")
    print(info.to_string())


def _normalize_dates_to_day(series: pd.Series) -> pd.Series:
    """Make datetimes tz-naive and floor to day."""
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


def load_and_clean_first_sheet(path: str, label: str, missing_drop_threshold: float = 0.20):
    """
    Load first sheet; ensure A/B = ID/Date; normalize Date (tz-naive, day);
    add year. Then:
      1) drop rows completely empty from column C onward
      2) drop columns (C onward) with > `missing_drop_threshold` missing (row-wise fraction)
    Return (clean_df, remaining_param_cols).
    """
    df = pd.read_excel(path)   # reads first sheet
    df.columns = df.columns.str.strip()

    # Ensure column A = ID, column B = Date
    if df.columns[0] != "ID":
        df = df.rename(columns={df.columns[0]: "ID"})
    if df.columns[1] != "Date":
        df = df.rename(columns={df.columns[1]: "Date"})

    # Normalize dates (tz-naive, day-level) and add year
    df["Date"] = _normalize_dates_to_day(df["Date"])
    df["year"] = df["Date"].dt.year

    print_structure(df, f"{label}: BEFORE cleaning")

    # Column C onward = parameter columns
    param_cols = [c for c in df.columns if c not in ["ID", "Date", "year"]]

    # drop rows completely empty across param cols
    rows_before = len(df)
    if param_cols:
        df = df.dropna(how="all", subset=param_cols)
    print(f"{label}: dropped rows completely empty (C→end): {rows_before - len(df)}")

    # drop param columns with > threshold missing
    to_drop = []
    if param_cols:
        missing_frac = df[param_cols].isna().mean()
        to_drop = missing_frac[missing_frac > missing_drop_threshold].index.tolist()

    if to_drop:
        df = df.drop(columns=to_drop)
    pct = int(missing_drop_threshold * 100)
    print(f"{label}: dropped columns with >{pct}% missing: {len(to_drop)}")
    if to_drop:
        print(sorted(to_drop))

    # Recompute remaining params & print after-structure
    param_cols = [c for c in df.columns if c not in ["ID", "Date", "year"]]
    print_structure(df, f"{label}: AFTER cleaning")

    return df, param_cols


# =========================
# MAIN EXECUTION
# =========================
if MODE not in FILEMAP:
    raise ValueError("MODE must be 'water' or 'weather'")

INPUT_FILE = FILEMAP[MODE]["input"]
OUTPUT_FILE = FILEMAP[MODE]["output"]

df, params = load_and_clean_first_sheet(INPUT_FILE, MODE.upper(), MISSING_DROP_THRESHOLD)

# Save cleaned file
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ Cleaned {MODE} data saved to: {OUTPUT_FILE}")
