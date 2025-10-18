import pandas as pd
import numpy as np
import re
from pathlib import Path
from openpyxl.utils import get_column_letter

# ---------- INPUT ----------
IN_PATH = Path("/Users/lisa-marieweidl/Desktop/Table1/RawData_Weather_2109.xlsx")
SHEET_NAME = "Blatt 1 - RawData_Weather_2109"

# ---------- OUTPUT ----------
OUT_SUBDIR = "outlier_output_weather"
OUT_PREFIX = "outlier"

# ---------- LOAD ----------
df = pd.read_excel(IN_PATH, sheet_name=SHEET_NAME)

# ---------- PREP ----------
# Drop empty / near-empty columns (tunable threshold)
MIN_VALID = 10
keep_mask = df.count(axis=0) >= MIN_VALID
dropped_cols = df.columns[~keep_mask].tolist()
df = df.loc[:, keep_mask]

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

exclude_cols = [c for c in ["A", "B"] if c in df.columns]
numeric_cols = df.select_dtypes(include="number").columns.difference(exclude_cols)

if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found after preprocessing.")

# ---------- OUTLIER FUNCTION ----------
def identify_outliers(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    if pd.isna(IQR) or IQR == 0:
        return pd.Series(False, index=s.index)
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (s < lower) | (s > upper)

# ---------- DETECT OUTLIERS ----------
outlier_mask = df[numeric_cols].apply(identify_outliers)
cols_with_outliers = outlier_mask.columns[outlier_mask.any(axis=0)]
outlier_mask_reduced = outlier_mask[cols_with_outliers]

# ---------- SUMMARY ----------
summary_records = []
for col in numeric_cols:
    s = df[col].astype(float)
    mask = outlier_mask[col]
    n_total = int(s.notna().sum())
    n_out = int(mask.sum())

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    out_values = s[mask]
    out_min = out_values.min() if n_out > 0 else np.nan
    out_max = out_values.max() if n_out > 0 else np.nan

    summary_records.append({
        "Parameter": col,
        "N_Total": n_total,
        "N_Outliers": n_out,
        "%_Outliers": round(100 * n_out / n_total, 2) if n_total > 0 else np.nan,
        "Q1": round(Q1, 6) if pd.notna(Q1) else np.nan,
        "Q3": round(Q3, 6) if pd.notna(Q3) else np.nan,
        "IQR": round(IQR, 6) if pd.notna(IQR) else np.nan,
        "Lower_Bound": round(lower, 6) if pd.notna(lower) else np.nan,
        "Upper_Bound": round(upper, 6) if pd.notna(upper) else np.nan,
        "Min_Outlier_Value": round(out_min, 6) if pd.notna(out_min) else np.nan,
        "Max_Outlier_Value": round(out_max, 6) if pd.notna(out_max) else np.nan,
    })

summary_df = pd.DataFrame(summary_records).sort_values("%_Outliers", ascending=False)

# ---------- DETAILED OUTLIER ROWS ----------
outlier_details = {}
for col in cols_with_outliers:
    rows = df[outlier_mask[col]]
    if not rows.empty:
        outlier_details[col] = rows

# ---------- FLAG OUTLIERS ----------
flags_df = outlier_mask_reduced.add_suffix("_outlier").astype("boolean")
df_outliers = pd.concat([df, flags_df], axis=1)

# ---------- HELPER: Safe Excel Sheet Names ----------
INVALID_SHEET_CHARS = r'[:\\/*?\[\]]'

def make_safe_sheet_name(name: str, used: set) -> str:
    safe = re.sub(INVALID_SHEET_CHARS, "_", str(name)).strip()
    if not safe:
        safe = "Sheet"
    base = safe[:31]
    safe = base
    i = 1
    while safe in used:
        suffix = f"_{i}"
        safe = (base[:(31 - len(suffix))] + suffix) if len(base) + len(suffix) > 31 else base + suffix
        i += 1
    used.add(safe)
    return safe

# ---------- SAVE ----------
out_dir = IN_PATH.parent / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

flags_path = out_dir / f"{OUT_PREFIX}_flags.xlsx"
summary_path = out_dir / f"{OUT_PREFIX}_summary.xlsx"
details_path = out_dir / f"{OUT_PREFIX}_details.xlsx"
dropped_path = out_dir / f"{OUT_PREFIX}_dropped_columns.txt"

# 0️⃣ Log dropped columns
with open(dropped_path, "w") as f:
    if dropped_cols:
        f.write("Dropped (insufficient data):\n")
        for c in dropped_cols:
            f.write(f"- {c}\n")
    else:
        f.write("No columns dropped for insufficient data.\n")

# 1️⃣ Save dataset with flags
df_outliers.to_excel(flags_path, index=False, engine="xlsxwriter")

# 2️⃣ Save formatted summary
with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False, sheet_name="Outlier_Summary")
    ws = writer.sheets["Outlier_Summary"]
    for i, col in enumerate(summary_df.columns, start=1):
        max_len = max(summary_df[col].astype(str).map(len, na_action='ignore').fillna(0).astype(int).max(), len(col)) + 2
        ws.column_dimensions[get_column_letter(i)].width = max_len

# 3️⃣ Save detailed outlier rows
with pd.ExcelWriter(details_path, engine="openpyxl") as writer:
    used_names = set()
    if not outlier_details:
        pd.DataFrame({"info": ["No outliers detected in any parameter."]}).to_excel(writer, sheet_name="Info", index=False)
    else:
        mapping_records = []
        for col, data in outlier_details.items():
            safe_name = make_safe_sheet_name(col, used_names)
            data = data.replace([np.inf, -np.inf], np.nan)
            data.to_excel(writer, sheet_name=safe_name, index=False)
            mapping_records.append({"Parameter": col, "Sheet_Name": safe_name})
        # Add index sheet mapping parameter → sheet name
        pd.DataFrame(mapping_records).sort_values("Parameter").to_excel(
            writer, sheet_name="Index", index=False
        )

# ---------- REPORT ----------
print("\n✅ OUTLIER ANALYSIS COMPLETE")
print(f"Analyzed numeric columns (excluding A, B): {list(numeric_cols)}")
print(f"Columns with ≥1 outlier: {list(cols_with_outliers)}")
print("\nOutlier Summary (Top 10):")
print(summary_df.head(10))
print(f"\nSaved flagged data:   {flags_path}")
print(f"Saved summary Excel:  {summary_path}")
print(f"Saved detailed cases: {details_path}")
print(f"Saved dropped columns log: {dropped_path}")
