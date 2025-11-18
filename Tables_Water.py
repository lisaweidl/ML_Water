import pandas as pd

RAW_FILE = "Water_2109.xlsx"
CLEAN_FILE = "Cleaned_Water_2109.xlsx"
DATE_COL = "Date"
ID_COL = "ID"

def classify_resolution(hours):
    if 20 <= hours <= 28:
        return "Daily"
    elif 24*28 <= hours <= 24*34:
        return "Monthly"
    elif 24*80 <= hours <= 24*100:
        return "Quarterly"
    elif 24*350 <= hours <= 24*380:
        return "Yearly"
    else:
        return "Irregular"

def summarize_per_id(file_path: str, label: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    output_rows = []

    for station_id, group in df.groupby(ID_COL):
        group = group.sort_values(DATE_COL)

        # time frame
        t_min = group[DATE_COL].min()
        t_max = group[DATE_COL].max()
        availability = f"{t_min.date()} – {t_max.date()}"

        # resolution
        diffs = group[DATE_COL].diff().dt.total_seconds().dropna() / 3600
        if len(diffs) > 0:
            median_res = diffs.median()
            resolution_label = classify_resolution(median_res)
        else:
            resolution_label = "Unknown"

        # values
        n_values = group.shape[0] * group.shape[1]
        n_missing = group.isna().sum().sum()

        output_rows.append({
            "Dataset": label,
            "ID": station_id,
            "Availability of Data": availability,
            "Data Time Resolution": resolution_label,
            "Total Values": n_values,
            "Missing Values": n_missing
        })

    return pd.DataFrame(output_rows)


raw_table = summarize_per_id(RAW_FILE, "Raw (uncleaned)")
print("RAW DATA SUMMARY")
print(raw_table)
raw_table.to_excel("table_raw_per_ID.xlsx", index=False)

clean_table = summarize_per_id(CLEAN_FILE, "Cleaned dataset")
print("\nCLEANED DATA SUMMARY")
print(clean_table)
clean_table.to_excel("table_cleaned_per_ID.xlsx", index=False)
