import pandas as pd
from pathlib import Path

file_path = "/Users/lisa-marieweidl/Desktop/ELB/Cleaned_Water_ELB.xlsx"
sheets     = None
output_xlsx = "column_summary_cleaned_water.xlsx"


def profile_dataframe(df: pd.DataFrame, sheet: str):
    """Return a detailed column summary for one DataFrame."""
    summary = pd.DataFrame({
        "dtype":        df.dtypes,
        "n_rows":       len(df),
        "n_missing":    df.isna().sum(),
        "n_unique":     df.nunique(dropna=True),
        "sample_values": [df[col].dropna().unique()[:5] for col in df.columns]
    })
    summary.insert(0, "sheet", sheet)
    return summary


def workbook_overview(all_frames: dict) -> pd.DataFrame:
    """Overall stats across all sheets combined."""
    big = pd.concat(all_frames.values(), ignore_index=True, sort=False)
    total_rows = len(big)
    total_cols = len(big.columns)
    total_cells = total_rows * max(total_cols, 1)
    total_missing = int(big.isna().sum().sum())
    pct_missing = (total_missing / total_cells * 100) if total_cells else 0
    total_unique = int(big.nunique(dropna=True).sum())
    return pd.DataFrame([{
        "n_sheets": len(all_frames),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "pct_missing": round(pct_missing, 2),
        "total_unique_values": total_unique
    }])


# ---- Read all sheets ----
if sheets is None:
    dfs = pd.read_excel(file_path, sheet_name=None)
else:
    dfs = {s: pd.read_excel(file_path, sheet_name=s) for s in sheets}

# ---- Per-column summaries ----
all_summaries = [profile_dataframe(df, name) for name, df in dfs.items()]
final_summary = pd.concat(all_summaries)

# ---- Overall workbook summary ----
overall_summary = workbook_overview(dfs)

# ---- Save both to one Excel file ----
with pd.ExcelWriter(output_xlsx) as writer:
    overall_summary.to_excel(writer, sheet_name="Overview", index=False)   # TOP summary
    final_summary.to_excel(writer, sheet_name="Column Details", index=True)

print(overall_summary.to_string(index=False))
print("\nDetailed column summary saved to", output_xlsx)
