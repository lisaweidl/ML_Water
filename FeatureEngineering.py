import pandas as pd

# water or merged
df = pd.read_excel("Merged.xlsx")
df["ID"] = df["ID"].astype("category")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

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

print_structure(df, "Before feature engineering)")



ROLL_WINDOWS_DAYS = [30, 60]

def add_lag_and_rolling_features(
    df: pd.DataFrame,
    id_col: str = "ID",
    date_col: str = "Date",
    windows_days=ROLL_WINDOWS_DAYS,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Sort within ID by time and enforce a clean unique row index
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    # --- LAGS (per ID) for water features ---
    for col in water_cols:
        if col in df.columns:
            g = df.groupby(id_col, sort=False)[col]
            df[f"{col}_lag1"] = g.shift(1)
            df[f"{col}_lag2"] = g.shift(2)
            df[f"{col}_lag3"] = g.shift(3)

    # --- ROLLING (per ID) for weather features (TIME-based on Date) ---
    num_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in num_cols if c in weather_cols and c in df.columns]

    roll_features_list = []

    if feature_cols:
        def _roll_per_group(g: pd.DataFrame, w: int) -> pd.DataFrame:
            # g is already in Date order because df is sorted
            r = (
                g.set_index(date_col)[feature_cols]
                 .rolling(f"{w}D", min_periods=1)
                 .agg(["mean", "min", "max"])
            )
            r.columns = [f"{col}_roll{w}D_{stat}" for col, stat in r.columns]
            # CRITICAL: align back to the original row index of this group
            r.index = g.index
            return r

        for w in windows_days:
            rolled = (
                df.groupby(id_col, group_keys=False, sort=False)
                  .apply(_roll_per_group, w=w)
            )
            roll_features_list.append(rolled)
    else:
        print("Skip rolling weather features (no matching weather numeric columns found).")

    if roll_features_list:
        df = pd.concat([df] + roll_features_list, axis=1)

        # fill missing numeric values with mean instead of deleting because of small dataset
    numeric_cols_all = df.select_dtypes(include="number").columns
    df[numeric_cols_all] = df.groupby(id_col, sort=False)[numeric_cols_all].transform(
        lambda s: s.fillna(s.mean())
    )

    return df

# Add lags and rolls
df_clean = add_lag_and_rolling_features(df)
df_clean.to_excel("Merged_FE.xlsx", index=False)



print_structure(df_clean, "After feature engineering)")

print(
    df.groupby("ID", observed=True)["Date"]
      .agg(n="count", min_date="min", max_date="max")
      .to_string()
)


