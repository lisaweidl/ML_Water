import pandas as pd
import numpy as np

# -------------------- Paths --------------------
input_file  = "/Users/lisa-marieweidl/Desktop/MergingAll/Water_Weather_Final_Prep.xlsx"
output_file = "/Users/lisa-marieweidl/Desktop/MergingAll/AllData_Merge_Test.xlsx"

# -------------------- Groups -------------------
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

# -------------------- Load ---------------------
df = pd.read_excel(input_file).copy()
for col in ("Date", "ID"):
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}. Found: {list(df.columns)}")

# Normalize ID
df["ID"] = (
    df["ID"].astype(str)
             .str.replace(r"\.0$", "", regex=True)
             .str.strip()
)
df.loc[df["ID"].str.lower().isin(["nan","none","nat","<na>",""]), "ID"] = pd.NA

# Robust Date parsing
_date_raw = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
mask_bad = _date_raw.isna() & df["Date"].notna()
if mask_bad.any():
    _date_raw2 = pd.to_datetime(
        df.loc[mask_bad, "Date"].astype(str).str.replace(r"[./]", "-", regex=True),
        errors="coerce", infer_datetime_format=True
    )
    _date_raw.loc[mask_bad] = _date_raw2

df["Date"] = _date_raw
df = df[~df["Date"].isna()].copy()

# Value columns
value_cols_order = [c for c in df.columns if c not in ("ID", "Date")]

# -------------------- Helpers --------------------
def first_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else pd.NA

def coverage_dates(s: pd.Series):
    u = pd.to_datetime(s.dropna().unique())
    if len(u) == 0:
        return pd.NaT, pd.NaT, 0
    return u.min(), u.max(), len(u)

# -------------------- Diagnostics: per-ID coverage --------------------
per_id_rows = []
for the_id, sub_df in df.groupby("ID", dropna=True):
    mn, mx, n = coverage_dates(sub_df["Date"])
    per_id_rows.append({
        "ID": the_id,
        "min_date": None if pd.isna(mn) else mn.date(),
        "max_date": None if pd.isna(mx) else mx.date(),
        "n_unique_dates": int(n),
    })
per_id_cov = pd.DataFrame(per_id_rows).sort_values("ID")

# -------------------- Build per-group output (UNION + PG gate) --------------------
group_frames = []
per_group_rows = []

for grp_no, id_list in groups:
    ids_norm = [str(x).strip().replace(".0", "") for x in id_list]
    sub = df[df["ID"].isin(set(ids_norm))].copy()
    if sub.empty:
        per_group_rows.append({
            "GroupNo": grp_no, "Group": "+".join(id_list),
            "min_date": None, "max_date": None, "n_unique_dates": 0,
            "note": "no rows for any ID in this group"
        })
        continue

    # Dates per ID (as normalized day)
    dates_per_id = {i: set(sub.loc[sub["ID"] == i, "Date"].dropna().dt.normalize().unique())
                    for i in ids_norm}

    # --- UNION of dates across the group's IDs ---
    sel_dates = set().union(*dates_per_id.values())
    if not sel_dates:
        per_group_rows.append({
            "GroupNo": grp_no, "Group": "+".join(id_list),
            "min_date": None, "max_date": None, "n_unique_dates": 0,
            "note": "no dates in union"
        })
        continue

    # Normalize date for filtering and gating
    sub["DateNorm"] = sub["Date"].dt.normalize()

    # --- PG gate: keep only dates where at least one PG* ID is present ---
    dates_with_pg = set(
        sub.loc[sub["ID"].astype(str).str.startswith("PG"), "DateNorm"].unique()
    )
    sel_dates = set(pd.to_datetime(list(sel_dates))).intersection(dates_with_pg)

    if not sel_dates:
        per_group_rows.append({
            "GroupNo": grp_no, "Group": "+".join(id_list),
            "min_date": None, "max_date": None, "n_unique_dates": 0,
            "note": "union has no dates with a PG station present"
        })
        continue

    # Filter to chosen dates
    sub = sub[sub["DateNorm"].isin(sel_dates)].copy()

    # Preserve your ID order (PG first if you listed it first)
    sub["__id_order"] = pd.Categorical(sub["ID"], categories=ids_norm, ordered=True)

    # Coalesce to one row per Date using first non-null by ID priority
    coalesced = (
        sub.sort_values(["DateNorm", "__id_order"], kind="mergesort")
           .groupby("DateNorm", dropna=False)[value_cols_order]
           .agg(first_non_null)
           .reset_index()
           .rename(columns={"DateNorm": "Date"})
    )

    # For transparency: which IDs contributed on each date
    ids_present = (
        sub.sort_values(["DateNorm", "__id_order"], kind="mergesort")
           .groupby("DateNorm")["ID"]
           .apply(lambda s: "+".join(pd.unique(s.astype(str))))
           .reset_index()
           .rename(columns={"DateNorm": "Date", "ID": "IDs_present"})
    )

    out = coalesced.merge(ids_present, on="Date", how="left")
    out.insert(0, "Group", "+".join(id_list))
    out.insert(0, "GroupNo", grp_no)

    # Coverage row for diagnostics
    ds = pd.to_datetime(out["Date"])
    per_group_rows.append({
        "GroupNo": grp_no, "Group": "+".join(id_list),
        "min_date": ds.min().date() if not ds.empty else None,
        "max_date": ds.max().date() if not ds.empty else None,
        "n_unique_dates": int(ds.nunique()),
        "note": ""
    })

    # Final date format for output
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

    group_frames.append(out.drop(columns="__id_order", errors="ignore"))

# Combine all groups
if group_frames:
    combined = pd.concat(group_frames, ignore_index=True, sort=False)
else:
    combined = pd.DataFrame(columns=["GroupNo","Group","Date","IDs_present"]+value_cols_order)

# Order columns
front = ["GroupNo", "Group", "Date", "IDs_present"]
other = [c for c in value_cols_order if c in combined.columns]
combined = combined[front + other].sort_values(["GroupNo", "Date"]).reset_index(drop=True)

# Per-group coverage table
per_group_cov = pd.DataFrame(per_group_rows).sort_values(["GroupNo"])

# -------------------- Write Excel --------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    combined.to_excel(writer, sheet_name="AllGroups_Union_PGonly", index=False)
    per_id_cov.to_excel(writer, sheet_name="PerID_Coverage", index=False)
    per_group_cov.to_excel(writer, sheet_name="PerGroup_Coverage", index=False)

print(f"✅ Done. Wrote:\n  - AllGroups_Union_PGonly\n  - PerID_Coverage\n  - PerGroup_Coverage\n→ {output_file}")
