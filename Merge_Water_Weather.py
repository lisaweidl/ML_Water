import pandas as pd

# -------------------- Paths --------------------
WATER_FILE   = "Cleaned_Water_2109.xlsx"
WEATHER_FILE = "Merged_Weather.xlsx"
OUTPUT_FILE  = "Merged_Water_Weather.xlsx"

# -------------------- Groups (weather IDs -> PG water ID) -------------------
groups = [
    (1,  ["PG31000292", "905"]),
    (2,  ["PG31100282", "191"]),
    (3,  ["PG31100312", "191"]),
    (4,  ["PG31100322", "905"]),
    (5,  ["PG32200012", "1903"]),
    (6,  ["PG32200042", "726"]),
    (7,  ["PG32200102", "726"]),
    (8,  ["PG32500392", "1903"]),
]

# Build a tidy mapping table: each NON-PG weather ID -> the group's PG water ID
map_rows = []
for _, id_list in groups:
    water_id = next((x for x in id_list if str(x).startswith("PG")), None)
    if not water_id:
        continue
    for x in id_list:
        sx = str(x).strip().replace(".0", "")
        if not sx.startswith("PG"):
            map_rows.append({"weather_id": sx, "PG_ID": water_id})
id_map = pd.DataFrame(map_rows)

# -------------------- Helpers --------------------
def normalize_id(s: pd.Series) -> pd.Series:
    out = (
        s.astype(str)
         .str.replace(r"\.0$", "", regex=True)
         .str.strip()
    )
    out = out.mask(out.str.lower().isin(["nan", "none", "nat", "<na>", ""]))
    return out

def normalize_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dtype, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()  # strict midnight

def first_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else pd.NA

# -------------------- Load & normalize --------------------
wtr = pd.read_excel(WATER_FILE).copy()
wx  = pd.read_excel(WEATHER_FILE).copy()

for name, df in [("water", wtr), ("weather", wx)]:
    for col in ("ID", "Date"):
        if col not in df.columns:
            raise ValueError(f"{name} file missing required column '{col}'. Found: {list(df.columns)}")

wtr["ID"]   = normalize_id(wtr["ID"])
wtr["Date"] = normalize_date(wtr["Date"])

wx["ID"]    = normalize_id(wx["ID"])
wx["Date"]  = normalize_date(wx["Date"])

# -------------------- Prepare weather: map to PG_ID, aggregate per (PG_ID, Date) --------------------
# Keep only weather rows that have a mapping to some PG water ID
wx = wx.merge(id_map, left_on="ID", right_on="weather_id", how="inner")
wx.drop(columns=["weather_id"], inplace=True)

# Drop metadata/admin columns if present
wx = wx.drop(columns=[c for c in ["year", "N_Sources", "Source_IDs"] if c in wx.columns], errors="ignore")

# Weather parameter columns (everything except keys)
weather_params = [c for c in wx.columns if c not in ("ID", "Date", "PG_ID")]

# Aggregate multiple weather stations in a group by Date using first-non-null per column
wx_g = (
    wx.groupby(["PG_ID", "Date"], dropna=False)[weather_params]
      .agg(first_non_null)
      .reset_index()
)

# -------------------- Water columns & ordering --------------------
# Drop water metadata cols if present
wtr = wtr.drop(columns=[c for c in ["year"] if c in wtr.columns], errors="ignore")
water_params = [c for c in wtr.columns if c not in ("ID", "Date")]

# Handle name collisions: suffix weather columns that clash with water names
collisions = set(water_params).intersection(set(weather_params))
if collisions:
    wx_g = wx_g.rename(columns={c: f"{c}_WEATHER" for c in collisions})
    weather_params = [c if c not in collisions else f"{c}_WEATHER" for c in weather_params]

# -------------------- Merge: left join water with aggregated weather --------------------
merged = wtr.merge(
    wx_g.rename(columns={"PG_ID": "ID"}),  # so keys are (ID, Date) on both sides
    on=["ID", "Date"],
    how="left",
)

# Column order: ID, Date, water (C→...), then weather
ordered_cols = ["ID", "Date"] + water_params + weather_params
merged = merged.reindex(columns=ordered_cols)

# -------------------- Diagnostics --------------------
water_keys = set(zip(wtr["ID"], wtr["Date"]))
weather_keys = set(zip(wx_g["PG_ID"], wx_g["Date"]))
n_keys_overlap = len(water_keys & {(i, d) for i, d in weather_keys})

print(f"Water rows: {len(wtr):,}")
print(f"Weather grouped rows: {len(wx_g):,}")
print(f"Overlapping (ID,Date) keys: {n_keys_overlap:,}")

# -------------------- Save --------------------
merged.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Merged with water-first columns → {OUTPUT_FILE}")
