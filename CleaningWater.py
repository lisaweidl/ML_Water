import pandas as pd

df = pd.read_csv("Water_Raw.csv", sep=";", encoding="utf-8")

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

RENAME_COLS = {
    "ABSTICH m": "Water.Level",
    "WASSERTEMPERATUR °C": "Water.Temperature",
    "ELEKTR. LEITF. (bei 25°C) µS/cm": "Electrical.Conductivity",
    "PH-WERT": "PH",
    "SAUERSTOFFGEHALT mg/l": "Dissolved.Oxygen",
    "GESAMTHAERTE °dH": "Total.Hardness",
    "KARBONATHAERTE °dH": "Carbonate.Hardness",
    "CALCIUM mg/l": "Calcium",
    "MAGNESIUM mg/l": "Magnesium",
    "NATRIUM mg/l": "Sodium",
    "KALIUM mg/l": "Potassium",
    "EISEN mg/l": "Iron",
    "MANGAN mg/l": "Manganese",
    "CADMIUM mg/l": "Cadmium",
    "QUECKSILBER mg/l": "Mercury",
    "ZINK mg/l": "Zinc",
    "KUPFER mg/l": "Copper",
    "ALUMINIUM mg/l": "Aluminum ",
    "BLEI mg/l": "Lead",
    "CHROM-GESAMT (filtriert) mg/l": "Chromium",
    "NICKEL mg/l": "Nickel",
    "ARSEN mg/l": "Arsenic",
    "BOR mg/l": "Boron",
    "AMMONIUM mg/l": "Ammonium",
    "NITRIT mg/l": "Nitrit",
    "NITRAT mg/l": "Nitrate",
    "CHLORID mg/l": "Chloride",
    "SULFAT mg/l": "Sulfate",
    "HYDROGENK. mg/l": "Bicarbonate",
    "ORTHOPHOSPHAT mg/l": "Orthophosphate",
    "DOC mg/l": "DOC",
    "AOX µg/l": "AOX",
    "Dimethachlor Metabolit µg/l": "Dimethachlor.Metabolite",
}

df = df.rename(columns=RENAME_COLS)

#save file
df.to_csv("Water_Cleaned.csv", sep=";", index=False)