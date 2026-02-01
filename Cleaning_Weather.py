import pandas as pd

df = pd.read_csv("Weather_Raw.csv", sep=";", encoding="utf-8")

df = df.rename(columns={"Date": "DATE"})
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)

value_cols = [c for c in df.columns if c not in ["ID", "DATE"]]

for c in value_cols:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

MISSING_DROP_THRESHOLD = 0.20
missing_frac = df[value_cols].isna().mean()
to_drop = missing_frac[missing_frac > MISSING_DROP_THRESHOLD].index.tolist()

if to_drop:
    print("\nDropped columns (>20% missing):", to_drop)
    df = df.drop(columns=to_drop)

df = df.sort_values(["ID", "DATE"]).reset_index(drop=True)

#rename columns not needed

df.to_csv("Weather_Cleaned.csv", sep=";", index=False)