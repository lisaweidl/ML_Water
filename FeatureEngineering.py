import pandas as pd

df = pd.read_excel("df_Merged.xlsx")

STATIC_COLS = [
    "ELEVATION",
    "LAND_USE",
    "SOIL_TYPE",
    "WATER_AVAILABILITY",
    "SOURCE_MATERIAL",
    "WATER_PERMEABILITY",
]

ROLL_WINDOWS = [7, 15, 30, 60]


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    drop_static: bool = False,
    id_col: str = "ID",
    date_col: str = "Date",
    year_col: str = "year",
    windows=ROLL_WINDOWS,
) -> pd.DataFrame:
    df = df.sort_values([id_col, date_col]).copy()

    # delete static columns from df (optional)
    if drop_static:
        cols_to_drop = [c for c in STATIC_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    # numeric feature candidates
    num_cols = df.select_dtypes(include="number").columns

    # exclude ID, year
    exclude = {id_col, year_col}
    feature_cols = [c for c in num_cols if c not in exclude]

    # --------------------------
    # LAG 1 FEATURES
    # --------------------------
    for col in feature_cols:
        df[f"{col}_lag1"] = df.groupby(id_col)[col].shift(1)

    # --------------------------
    # ROLLING WINDOW FEATURES
    # --------------------------
    roll_features_list = []

    for w in windows:
        roll = (
            df.groupby(id_col)[feature_cols]
              .rolling(w, min_periods=1)
              .agg(["mean", "min", "max"])
        )

        # flatten MultiIndex columns
        roll.columns = [f"{col}_roll{w}_{stat}" for col, stat in roll.columns]

        # drop group level and align index with df
        roll = roll.reset_index(level=0, drop=True)

        roll_features_list.append(roll)

    # concat rolling features back to df
    if roll_features_list:
        df = pd.concat([df] + roll_features_list, axis=1)

    numeric_cols_all = df.select_dtypes(include="number").columns
    df[numeric_cols_all] = df[numeric_cols_all].apply(
        lambda col: col.fillna(col.mean())
    )
    return df


# Experiment with or without static columns
df_clean = add_lag_and_rolling_features(df, drop_static=True)
df_clean.to_excel("df_Merged_clean.xlsx", index=False)
