import pandas as pd

INPUT_XLSX  = "/Users/lisa-marieweidl/Desktop/MergingAll/AllData_Merge_Test.xlsx"
INPUT_SHEET = "AllGroups_Union_PGonly"
OUTPUT_XLSX = "Weather_Water_missing_report_by_ID_year.xlsx"
OUTPUT_SHEET = "missing_report"

DESIRED_PARAMS = [
    "Water_Level","Water_Temperature","Electrical_Conductivity","Ph_Value",
    "Oxygen_Content","Total_Hardness","Carbonate_Hardness","Calcium","Magnesium",
    "Sodium","Potassium","Iron","Manganese","Cadmium","Mercury","Zinc","Copper",
    "Aluminium","Lead","Chrome","Nickel", "Arsenic","Boron","Ammonium","Nitrite",
    "Nitrate","Chloride","Sulfate","Hydrogen_Carbonate", "Orthophosphate","Doc","Aox",
    "Dimethachlor_Metabolit", "Precipitation","cglo_j","ffx","rf_i","rf_ii","rf_iii",
    "rf_mittel","rr","rr_i","rr_iii","so_h","tl_i","tl_ii","tl_iii","tlmax",
    "tlmin","tl_mittel","tsmin","vvbft_i","vvbft_ii","vvbft_iii","vv_mittel"
]


df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)
df.columns = df.columns.str.strip()

if "Group" in df.columns:
    df = df.rename(columns={"Group": "ID"})
if "ID" not in df.columns:
    raise ValueError("Need a 'Group' (or 'ID') column.")
if "Date" not in df.columns:
    raise ValueError("Need a 'Date' column.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["year"] = df["Date"].dt.year

param_cols = [c for c in DESIRED_PARAMS if c in df.columns]
if not param_cols:
    raise ValueError("No expected parameter columns found.")

# --- GROUP & AGGREGATE ---
n_rows = df.groupby(["ID", "year"]).size().rename("n_rows")
missing_counts = df.groupby(["ID", "year"])[param_cols].apply(lambda g: g.isna().sum())
report = missing_counts.merge(n_rows, left_index=True, right_index=True).reset_index()

# --- RENAME PARAMETER COLUMNS WITH 'Na_' PREFIX ---
report = report.rename(columns={col: f"Na_{col}" for col in param_cols})

# Put columns in order: ID, year, n_rows, then Na_*
ordered_cols = ["ID", "year", "n_rows"] + [f"Na_{col}" for col in param_cols]
report = report[ordered_cols]

# --- SAVE ---
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    report.to_excel(writer, sheet_name=OUTPUT_SHEET, index=False)

print(f"Done. Excel report with 'Na_' prefixes saved to: {OUTPUT_XLSX}")
