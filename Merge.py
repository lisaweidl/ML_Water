import pandas as pd

MERGE_IDS = [
    "PG31000292+905",
    "PG31100282+191",
    "PG31100312+191",
    "PG31100322+905",
    "PG32200012+1903",
    "PG32200042+726",
    "PG32200102+726",
    "PG32500392+1903",
]

def print_final_structure(df: pd.DataFrame, title: str):
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "value_count": df.notna().sum(),
        "n_missing": df.isna().sum(),
    })

    print(f"\n=== {title} ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}\n")
    print(summary.to_string())


pairs = [s.split("+") for s in MERGE_IDS]
id_map = pd.DataFrame(pairs, columns=["ID", "ID_weather"])
id_map["ID_weather"] = id_map["ID_weather"].astype(int)

water = pd.read_excel("Cleaned_Water_2109.xlsx")
weather = pd.read_excel("Cleaned_Weather_2109.xlsx")

water.columns = water.columns.str.strip()
weather.columns = weather.columns.str.strip()

# keep only those water IDs
water = water[water["ID"].isin(id_map["ID"])]

# attach weather ID to water rows
water = water.merge(id_map, on="ID", how="left")

# restrict weather to those stations
weather = weather[weather["ID"].isin(id_map["ID_weather"])]

# dates
water["Date"] = pd.to_datetime(water["Date"], errors="coerce").dt.normalize()
weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")
if getattr(weather["Date"].dtype, "tz", None) is not None:
    weather["Date"] = weather["Date"].dt.tz_localize(None)
weather["Date"] = weather["Date"].dt.normalize()

# merge on (weather ID, Date)
merged = pd.merge(
    water,
    weather,
    left_on=["ID_weather", "Date"],
    right_on=["ID", "Date"],
    how="left",
)

# rename ID columns
merged = merged.rename(columns={
    "ID_x": "ID",
    "ID_y": "weather_station_id",
    "year_x": "year_water",
    "year_y": "year_weather",
})

# order cols: water first, then weather
water_cols = ["ID", "Date"] + [c for c in water.columns if c not in ["ID", "Date", "ID_weather"]]
weather_cols = [c for c in weather.columns if c not in ["ID", "Date"]]
final_cols = [c for c in water_cols if c in merged.columns] + [c for c in weather_cols if c in merged.columns]

merged = merged[final_cols]
merged = merged.dropna()

rows_per_id = merged.groupby("ID").size().reset_index(name="n_rows")
print(rows_per_id)

# only keep merged IDs that have continuous data from 1994 to 2007 (since n_values is highest)
df = merged[~merged["ID"].isin(["PG32200042", "PG32200102"])]
df = df[(df["Date"].dt.year >= 1994) & (df["Date"].dt.year <= 2007)]

df.to_excel("df_Merged.xlsx", index=False)

