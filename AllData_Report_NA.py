import pandas as pd
from pathlib import Path

# --- SETTINGS ---
file_path    = Path("/Users/lisa-marieweidl/Desktop/ELB/Cleaned_Water_ELB.xlsx")
sheets       = None  # None = all sheets, or list like ["Sheet1","Sheet2"]
cleaned_xlsx = "columns_cleaned_water_NA.xlsx"

# Strings to consider missing (add more if needed)
NA_STRINGS = ["", " ", "NA", "N/A", "na", "n/a", "NaN", "null", "NULL", "-", "–", "—"]

# --- Read workbook (coerce NA-like strings) ---
dfs = (pd.read_excel(file_path, sheet_name=None, na_values=NA_STRINGS, keep_default_na=True)
       if sheets is None else
       {s: pd.read_excel(file_path, sheet_name=s, na_values=NA_STRINGS, keep_default_na=True) for s in sheets})

report = []

for sheet, df in dfs.items():
    df = df.copy()

    # Normalize text fields: strip whitespace and turn NA-like strings into NaN
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace(NA_STRINGS, pd.NA)

    cols = list(df.columns)
    data_cols = cols[2:] if len(cols) > 2 else []  # columns after first two

    if data_cols:
        # keep rows that have ANY non-missing value in data_cols
        keep_mask = df[data_cols].notna().any(axis=1)
        dropped = int((~keep_mask).sum())
        df_clean = df.loc[keep_mask].copy()
        # sanity check: no row left with all data_cols missing
        assert not df_clean[data_cols].isna().all(axis=1).any()
    else:
        df_clean = df.copy()
        dropped = 0

    # Make tz-aware datetimes Excel-safe (drop tz)
    for col in df_clean.select_dtypes(include=["datetimetz"]).columns:
        df_clean[col] = df_clean[col].dt.tz_convert("UTC").dt.tz_localize(None)

    dfs[sheet] = df_clean
    report.append({
        "sheet": sheet,
        "rows_before": len(df),
        "rows_after": len(df_clean),
        "dropped_rows_no_data_after_col2": dropped
    })

# Summary
summary = pd.DataFrame(report)
print("\n=== Rows dropped where all columns after the first two were empty ===")
print(summary.to_string(index=False))

# Save cleaned workbook
with pd.ExcelWriter(cleaned_xlsx) as writer:
    for sheet, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"\nCleaned workbook saved to {cleaned_xlsx}")
