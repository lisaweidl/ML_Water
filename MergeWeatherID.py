import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype

file_path = "/Users/lisa-marieweidl/Desktop/Table2.1/CleanedData_Weather_2109_NA.xlsx"
sheet = 0
ID_COL = "ID"
DATE_COL = "Date"
GROUPS = {
    "191_701": [191, 701],
    "905_81": [905, 81],
    "1903_123_1901_1900": [1903, 123, 1901, 1900],
    "199_2117_2114": [199, 2117, 2114],
}
output_xlsx = "merged_id_timeseries_allcols.xlsx"


def normalize_id(x):
    try:
        return int(x)
    except Exception:
        return str(x)

def mode_or_first(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return pd.NA
    if s.nunique(dropna=True) == 1:
        return s.iloc[0]
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else s.iloc[0]


# --- Read & prep ---
df = pd.read_excel(file_path, sheet_name=sheet).copy()

# Parse Date; keep absolute time in UTC (naive after tz drop)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True).dt.tz_localize(None)

df["_ID_norm"] = df[ID_COL].apply(normalize_id)

# Map IDs to group labels
id_to_group = {}
for g_label, ids in GROUPS.items():
    for _id in ids:
        id_to_group[normalize_id(_id)] = g_label

df["_group"] = df["_ID_norm"].map(id_to_group)

# Work rows: must be in one of the groups and have a valid date
work = df[df["_group"].notna() & df[DATE_COL].notna()].copy()

# --- Build aggregation map for ALL non-key columns ---
agg_map = {}
for col in work.columns:
    if col in ("_group", DATE_COL, ID_COL, "_ID_norm"):
        continue
    if is_numeric_dtype(work[col]):
        agg_map[col] = "mean"
    elif is_datetime64_any_dtype(work[col]):
        # keep first non-null timestamp
        agg_map[col] = "first"
    elif is_bool_dtype(work[col]):
        agg_map[col] = mode_or_first
    else:
        agg_map[col] = mode_or_first

# --- Aggregate per (group, date) ---
if agg_map:
    agg_main = (
        work.groupby(["_group", DATE_COL], dropna=False)
            .agg(agg_map)
            .reset_index()
    )
else:
    # If only key columns exist, keep unique (group, date) pairs
    agg_main = work[["_group", DATE_COL]].drop_duplicates().reset_index(drop=True)

# Add metadata: N_Sources and Source_IDs
n_sources = (
    work.groupby(["_group", DATE_COL], dropna=False)
        .size().rename("N_Sources").reset_index()
)
src_ids = (
    work.groupby(["_group", DATE_COL], dropna=False)["_ID_norm"]
        .apply(lambda s: ",".join(sorted({str(x) for x in s.dropna()})))
        .rename("Source_IDs").reset_index()
)

merged = (
    agg_main
    .merge(n_sources, on=["_group", DATE_COL], how="left")
    .merge(src_ids,  on=["_group", DATE_COL], how="left")
    .rename(columns={"_group": "Merged_Group", DATE_COL: "Date"})
    .sort_values(["Merged_Group", "Date"])
    .reset_index(drop=True)
)

# --- Make ALL datetime columns timezone-naive for Excel ---
# (If any non-Date datetime columns are tz-aware, drop tz info)
for col in merged.select_dtypes(include=["datetimetz"]).columns:
    # keep absolute time by converting to UTC then dropping tz
    merged[col] = merged[col].dt.tz_convert("UTC").dt.tz_localize(None)

# --- Write to Excel ---
with pd.ExcelWriter(output_xlsx) as writer:
    merged.to_excel(writer, sheet_name="MergedSeriesAllCols", index=False)

print(f"Done. Merged series (all columns) saved to {output_xlsx}")
print(merged.head(10).to_string(index=False))
