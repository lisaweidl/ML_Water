import pandas as pd

# water or merged
df = pd.read_excel("Merged.xlsx")
df["ID"] = df["ID"].astype("category")

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

def print_structure(df: pd.DataFrame, title: str):
    n = len(df)
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dt) for dt in df.dtypes],
        "n_rows": n,
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
    }).set_index("column")[["dtype", "n_rows", "n_missing"]]
    print(f"\n=== {title} ===")
    print(info.to_string())

#water or merged
print_structure(df, "MERGED: loaded (before feature engineering)")

def add_lag_and_rolling_features(
    df: pd.DataFrame,
    id_col: str = "ID",
    date_col: str = "Date",
    windows=ROLL_WINDOWS,
) -> pd.DataFrame:
    df = df.sort_values([id_col, date_col]).copy()

    # numeric feature candidates
    num_cols = df.select_dtypes(include="number").columns

    # only weather features for rolling
    feature_cols = [c for c in num_cols if c in weather_cols]

    # Lag only water features
    for col in water_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby(id_col)[col].shift(1)
            df[f"{col}_lag2"] = df.groupby(id_col)[col].shift(2)
            df[f"{col}_lag3"] = df.groupby(id_col)[col].shift(3)

    # rolling window if weather features exist
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


# Add lags and rolls
df_clean = add_lag_and_rolling_features(df)
df_clean.to_excel("Merged_FE.xlsx", index=False)

#water or merged
print_structure(df_clean, "MERGED: loaded (after feature engineering)")


