import pandas as pd

MODE = "weather"   # or "weather"

MISSING_DROP_THRESHOLD = 0.20

FILEMAP = {
    "weather": {
        "input":  "Weather_Raw.xlsx",
        "output": "Weather_Cleaned.xlsx",
    },
    "water": {
        "input":  "Water_Raw.xlsx",
        "output": "Water_Cleaned.xlsx",
    },
}


def print_structure(df: pd.DataFrame, title: str):
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
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


def load_and_clean_first_sheet(path: str, label: str, missing_drop_threshold: float = 0.20):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    # A = ID, B = Date
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("Expected at least two columns (ID, Date) in the input file.")
    cols[0] = "ID"
    cols[1] = "Date"
    df.columns = cols

    # Normalize dates and add year
    df["Date"] = _normalize_dates_to_day(df.iloc[:, 1])
    df["year"] = df["Date"].dt.year

    # Keep ID as-is (no recoding)
    print_structure(df, f"{label}: BEFORE cleaning")

    # Define parameter columns = everything except ID/Date/year
    param_cols = [c for c in df.columns if c not in ["ID", "Date", "year"]]

    # Drop rows completely empty across parameter columns
    rows_before = len(df)
    if param_cols:
        df = df.dropna(how="all", subset=param_cols)
    print(f"{label}: dropped rows completely empty (param cols): {rows_before - len(df)}")

    # Drop parameter columns with > threshold missing
    to_drop = []
    if param_cols:
        missing_frac = df[param_cols].isna().mean()
        to_drop = missing_frac[missing_frac > missing_drop_threshold].index.tolist()

    if to_drop:
        df = df.drop(columns=to_drop)

    pct = int(missing_drop_threshold * 100)
    print(f"{label}: dropped columns with >{pct}% missing: {len(to_drop)}")
    if to_drop:
        print("Dropped columns:", sorted(to_drop))

    # Recompute remaining parameter columns after dropping
    param_cols = [c for c in df.columns if c not in ["ID", "Date", "year"]]
    print_structure(df, f"{label}: AFTER cleaning")

    return df, param_cols


INPUT_FILE = FILEMAP[MODE]["input"]
OUTPUT_FILE = FILEMAP[MODE]["output"]

df, params = load_and_clean_first_sheet(INPUT_FILE, MODE.upper(), MISSING_DROP_THRESHOLD)
df.to_excel(OUTPUT_FILE, index=False)
print(f"Remaining parameter columns: {len(params)}")
print(f"Cleaned {MODE} data saved to: {OUTPUT_FILE}")