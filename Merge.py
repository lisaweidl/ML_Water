import pandas as pd

MERGE_IDS = [
    "PG31000292+905",
    "PG31100282+191",
    "PG31100312+191",
    "PG31100322+905",
    "PG32200042+726",
    "PG32200102+726",
]

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

pairs = [s.split("+") for s in MERGE_IDS]
id_map = pd.DataFrame(pairs, columns=["ID", "ID_weather"])
id_map["ID_weather"] = id_map["ID_weather"].astype(int)

water = pd.read_excel("Water_Cleaned.xlsx")
weather = pd.read_excel("Weather_Cleaned.xlsx")

water.columns = water.columns.str.strip()
weather.columns = weather.columns.str.strip()

print_structure(water, "WATER: loaded (before filtering)")
print_structure(weather, "WEATHER: loaded (before filtering)")

# keep only those water IDs
water = water[water["ID"].isin(id_map["ID"])].copy()

# attach weather ID to water rows
water = water.merge(id_map, on="ID", how="left")

# restrict weather to those stations
weather = weather[weather["ID"].isin(id_map["ID_weather"])].copy()

print_structure(water, "WATER: after ID filter + ID_weather merge")
print_structure(weather, "WEATHER: after station filter")

# dates
water["Date"] = pd.to_datetime(water["Date"], errors="coerce").dt.normalize()

weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")
if getattr(weather["Date"].dtype, "tz", None) is not None:
    weather["Date"] = weather["Date"].dt.tz_localize(None)
weather["Date"] = weather["Date"].dt.normalize()

print_structure(water, "WATER: after Date normalize")
print_structure(weather, "WEATHER: after Date normalize")

# merge on (weather ID, Date)
merged = pd.merge(
    water,
    weather,
    left_on=["ID_weather", "Date"],
    right_on=["ID", "Date"],
    how="left",
)

weather_feature_cols = [c for c in weather.columns if c not in ["ID", "Date"]]
matched = merged[weather_feature_cols].notna().any(axis=1).sum() if weather_feature_cols else 0
print(f"\nWeather match rows: {matched} / {len(merged)}")

# rename ID columns
merged = merged.rename(columns={
    "ID_x": "ID",
    "ID_y": "weather_station_id",
})

# order cols: water first, then weather
water_cols = ["ID", "Date"] + [c for c in water.columns if c not in ["ID", "Date", "ID_weather"]]
weather_cols = [c for c in weather.columns if c not in ["ID", "Date"]]
final_cols = [c for c in water_cols if c in merged.columns] + [c for c in weather_cols if c in merged.columns]

merged = merged[final_cols]

rows_per_id = merged.groupby("ID").size().reset_index(name="n_rows")
print("\nRows per ID:")
print(rows_per_id.to_string(index=False))

print_structure(merged, "MERGED: final (after merge + reorder)")


df = merged
df = df[~df["ID"].isin(["PG32200042", "PG32200102"])]
df = df[(df["Date"].dt.year >= 1994) & (df["Date"].dt.year <= 2007)]

print_structure(df, "DF: final")
print(df["ID"].value_counts())

df.to_excel("Merged.xlsx", index=False)
print("\nSaved: Merged.xlsx")
