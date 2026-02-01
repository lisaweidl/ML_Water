import pandas as pd

df = pd.read_csv("Water_Cleaned.csv", sep=";", encoding="utf-8")

for c in df.columns:
    if c not in ["ID", "DATE"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")


df = df.sort_values(["ID", "DATE"]).reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ["ID", "DATE"] and pd.api.types.is_numeric_dtype(df[c])]

g = df.groupby("ID", sort=False)

lag1 = g[feature_cols].shift(1).add_suffix("_lag1")
lag2 = g[feature_cols].shift(2).add_suffix("_lag2")
lag3 = g[feature_cols].shift(3).add_suffix("_lag3")

df = pd.concat([df, lag1, lag2, lag3], axis=1)

df.to_csv("Water_FE.csv", sep=";", index=False)
