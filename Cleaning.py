import pandas as pd
from sklearn.impute import KNNImputer
from typing import List


MODE = ("water")   # or weather

MISSING_DROP_THRESHOLD = 0.25

# KNN imputation
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "uniform"   # or "distance"

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

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    s = pd.DataFrame({
        "mean": num.mean(),
        "median": num.median(),
        "min": num.min(),
        "max": num.max(),
        "std": num.std(),
    })
    s["dtype"] = df.dtypes.reindex(s.index).astype(str)
    s["n_missing"] = df[s.index].isna().sum()
    return s



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

def knn_impute_remaining_missing(df: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame:
    """
    KNN-impute missing values for (coercible) numeric parameter columns only.
    Non-numeric parameter columns are left unchanged.
    """
    if not param_cols:
        return df

    # Try to coerce parameter cols to numeric (helps with numeric columns read as object)
    X = df[param_cols].apply(pd.to_numeric, errors="coerce")

    # Only impute columns that have at least one non-missing value (and at least one missing)
    valid_cols = [c for c in param_cols if X[c].notna().any()]
    if not valid_cols:
        print("KNN: No numeric parameter columns with usable data for imputation. Skipping.")
        return df

    n_missing_before = int(X[valid_cols].isna().sum().sum())
    if n_missing_before == 0:
        print("KNN: No missing values remaining in numeric parameter columns. Skipping.")
        return df

    cols_imputed = [c for c in valid_cols if X[c].isna().any()]
    print("KNN imputed columns:", cols_imputed)


    imputer = KNNImputer(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS)

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X[valid_cols]),
        columns=valid_cols,
        index=df.index
    )

    # Write back only the imputed numeric columns
    df.loc[:, valid_cols] = X_imputed

    n_missing_after = int(pd.DataFrame(df[valid_cols]).isna().sum().sum())
    print(f"KNN: missing before = {n_missing_before}, missing after = {n_missing_after} (numeric param cols)")

    return df

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

    # Keep ID as-is (no recoding)
    print_structure(df, f"{label}: BEFORE cleaning")

    print("\nSUMMARY TABLE BEFORE")
    print(summary_table(df).to_string())


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

    # KNN impute remaining missing values
    df = knn_impute_remaining_missing(df, param_cols)

    print_structure(df, f"{label}: AFTER cleaning")

    print("\nSUMMARY TABLE AFTER")
    print(summary_table(df).to_string())


    return df, param_cols


INPUT_FILE = FILEMAP[MODE]["input"]
OUTPUT_FILE = FILEMAP[MODE]["output"]

df, params = load_and_clean_first_sheet(INPUT_FILE, MODE.upper(), MISSING_DROP_THRESHOLD)
df.to_excel(OUTPUT_FILE, index=False)
print(f"Remaining parameter columns: {len(params)}")
print(f"Cleaned {MODE} data saved to: {OUTPUT_FILE}")