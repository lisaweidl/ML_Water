# %%
import pandas as pd

df = pd.read_csv("Water_Cleaned.csv", sep=";", encoding="utf-8")

# convert columns to numeric (except ID, DATE)
for c in df.columns:
    if c not in ["ID", "DATE"]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

# sort within sites
df = df.sort_values(["ID", "DATE"]).reset_index(drop=True)

# numeric feature columns (exclude ID, DATE)
feature_cols = [
    c for c in df.columns
    if c not in ["ID", "DATE"] and pd.api.types.is_numeric_dtype(df[c])
]

g = df.groupby("ID", sort=False)

# --- Lag 1, 2, 3 for ALL numeric features ---
lag1_all = g[feature_cols].shift(1).add_suffix("_lag1")
lag2_all = g[feature_cols].shift(2).add_suffix("_lag2")
lag3_all = g[feature_cols].shift(3).add_suffix("_lag3")

# combine
df = pd.concat([df, lag1_all, lag2_all, lag3_all], axis=1)

# save
df.to_csv("Water_FE.csv", sep=";", index=False)
# %%
df.head()
# %%
df.info()