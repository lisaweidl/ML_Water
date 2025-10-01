import pandas as pd

# --- Paths ---
input_file  = "/Users/lisa-marieweidl/Desktop/Weather_Final.xlsx"
output_file = "/Users/lisa-marieweidl/Desktop/AllData_Merge_Final.xlsx"  # single sheet, one row per group+date

# --- Groups (group_no, [IDs in desired order]) ---
groups = [
    (1,  ["PG31000292", "905"]),
    (2,  ["PG31100282", "191"]),
    (3,  ["PG31100312", "191"]),
    (4,  ["PG31100322", "905", "109280"]),
    (5,  ["PG32200012", "1903", "109751"]),
    (6,  ["PG32200042", "726", "109306"]),
    (7,  ["PG32200102", "726", "109256"]),
    (8,  ["PG32500392", "1903"]),
    (9,  ["PG32200092", "109314"]),
]

# --- Load once ---
df = pd.read_excel(input_file).copy()
for col in ("Date", "ID"):
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}. Found: {list(df.columns)}")

# Normalize Date & ID
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
df["ID"] = (
    df["ID"].astype(str)
             .str.replace(r"\.0$", "", regex=True)
             .str.strip()
             .where(lambda s: ~s.str.lower().isin(["nan", "none", "nat", "<na>", ""]), pd.NA)
)

# All value columns (everything except keys)
value_cols_order = [c for c in df.columns if c not in ("ID", "Date")]

def first_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else pd.NA

frames = []
for grp_no, id_list in groups:
    ids_norm = [str(x).strip().replace(".0", "") for x in id_list]

    sub = df[df["ID"].isin(set(ids_norm))].copy()
    if sub.empty:
        continue

    # --- Keep only dates present for ALL IDs in this group ---
    dates_per_id = {
        i: set(sub.loc[sub["ID"] == i, "Date"].dropna().unique())
        for i in ids_norm
    }
    # If any ID has zero dates, intersection is empty
    if any(len(s) == 0 for s in dates_per_id.values()):
        continue
    common_dates = set.intersection(*dates_per_id.values())
    if not common_dates:
        continue

    sub = sub[sub["Date"].isin(common_dates)].copy()

    # Keep your ID order within each date
    sub["__id_order"] = pd.Categorical(sub["ID"], categories=ids_norm, ordered=True)

    # Coalesce to ONE row per Date (no averaging; just first non-null per column)
    coalesced = (
        sub.sort_values(["Date", "__id_order"], kind="mergesort")
           .groupby(["Date"], dropna=False)[value_cols_order]
           .agg(first_non_null)
           .reset_index()
    )

    # IDs present (should be all ids_norm now), in your specified order
    ids_present = (
        sub.sort_values(["Date", "__id_order"], kind="mergesort")
           .groupby(["Date"])["ID"]
           .apply(lambda s: "+".join(pd.unique(s.astype(str))))
           .reset_index()
           .rename(columns={"ID": "IDs_present"})
    )

    out = coalesced.merge(ids_present, on="Date", how="left")
    out.insert(0, "Group", "+".join(id_list))
    out.insert(0, "GroupNo", grp_no)

    frames.append(out.drop(columns="__id_order", errors="ignore"))

# Combine all groups into ONE sheet
combined = (
    pd.concat(frames, ignore_index=True, sort=False)
    if frames else
    pd.DataFrame(columns=["GroupNo","Group","Date","IDs_present"]+value_cols_order)
)

# Final column order: GroupNo, Group, Date, IDs_present, then original value columns
front = ["GroupNo", "Group", "Date", "IDs_present"]
other = [c for c in value_cols_order if c in combined.columns]
combined = combined[front + other].sort_values(["GroupNo", "Date"]).reset_index(drop=True)

# Write a single sheet
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    combined.to_excel(writer, sheet_name="AllGroupsCommonDates", index=False)

print(f"✅ Done. Kept only common dates per group and wrote one row per (Group, Date) → {output_file}")
