import pandas as pd

df = pd.read_excel("Merged.xlsx")
df["ID"] = df["ID"].astype("category")

STATIC_COLS = [
    "ELEVATION",
    "LAND_USE",
    "SOIL_TYPE",
    "WATER_AVAILABILITY",
    "SOURCE_MATERIAL",
    "WATER_PERMEABILITY",
]

weather_cols = {
    "rr", "rr_i", "rr_iii",
    "tlmax", "tlmin", "tl_mittel",
    "vvbft_i", "vvbft_ii", "vvbft_iii",
    "rf_i", "rf_ii", "rf_iii", "rf_mittel"
}

water_cols = [
    "ABSTICH m",
    "WASSERTEMPERATUR °C",
    "ELEKTR. LEITF. (bei 25°C) µS/cm",
    "PH-WERT",
    "SAUERSTOFFGEHALT mg/l",
    "GESAMTHAERTE °dH",
    "KARBONATHAERTE °dH",
    "CALCIUM mg/l",
    "MAGNESIUM mg/l",
    "NATRIUM mg/l",
    "KALIUM mg/l",
    "EISEN mg/l",
    "MANGAN mg/l",
    "BOR mg/l",
    "AMMONIUM mg/l",
    "NITRIT mg/l",
    "NITRAT mg/l",
    "CHLORID mg/l",
    "SULFAT mg/l",
    "HYDROGENK. mg/l",
    "ORTHOPHOSPHAT mg/l",
    "DOC mg/l",
]

ROLL_WINDOWS = [30, 60]


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    drop_static: bool = False,
    id_col: str = "ID",
    date_col: str = "Date",
    year_col: str = "year",
    windows=ROLL_WINDOWS,
) -> pd.DataFrame:
    df = df.sort_values([id_col, date_col]).copy()

    if drop_static:
        cols_to_drop = [c for c in STATIC_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    # numeric feature candidates
    num_cols = df.select_dtypes(include="number").columns

    # use ONLY weather features for rolling
    feature_cols = [c for c in num_cols if c in weather_cols]

    # --------------------------
    # LAG 1–3 FEATURES (ONLY WATER COLS)
    # --------------------------
    for col in water_cols:
        if col in df.columns:  # optional safety
            df[f"{col}_lag1"] = df.groupby(id_col)[col].shift(1)
            df[f"{col}_lag2"] = df.groupby(id_col)[col].shift(2)
            df[f"{col}_lag3"] = df.groupby(id_col)[col].shift(3)

    # --------------------------
    # ROLLING WINDOW FEATURES (ONLY IF WEATHER COLS EXIST)
    # --------------------------
    roll_features_list = []

    if feature_cols:
        for w in windows:
            roll = (
                df.groupby(id_col)[feature_cols]
                  .rolling(w, min_periods=1)
                  .agg(["mean", "min", "max"])
            )
            roll.columns = [f"{col}_roll{w}_{stat}" for col, stat in roll.columns]
            roll = roll.reset_index(level=0, drop=True)
            roll_features_list.append(roll)
    else:
        print("Skip rolling weather features.")

    if roll_features_list:
        df = pd.concat([df] + roll_features_list, axis=1)

    numeric_cols_all = df.select_dtypes(include="number").columns
    df[numeric_cols_all] = df[numeric_cols_all].apply(
        lambda col: col.fillna(col.mean())
    )
    return df

# Experiment/ with or without static columns
df_clean = add_lag_and_rolling_features(df, drop_static=True)
df_clean.to_excel("Merged_FE.xlsx", index=False)

print(df_clean.head())

