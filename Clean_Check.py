import pandas as pd

# -------------------- Settings --------------------
INPUT_FILE   = "Merged_Water_Weather.xlsx"
OUTPUT_FILE  = "Merged_Filtered_<=20pctNA.xlsx"
SUMMARY_FILE = "NA_Summary_Filtered.xlsx"
THRESHOLD    = 0.20  # 20%

# -------------------- Load --------------------
df = pd.read_excel(INPUT_FILE)

# Safety checks
for col in ("ID", "Date"):
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {INPUT_FILE}")

# Parameter columns = everything except ID/Date
param_cols = [c for c in df.columns if c not in ("ID", "Date")]
if not param_cols:
    raise ValueError("No parameter columns found (only 'ID' and 'Date' present).")

# -------------------- NA summary per ID (over parameter columns) --------------------
summary = (
    df.groupby("ID", dropna=True)[param_cols]
      .apply(lambda g: pd.Series({
          "n_rows": len(g),
          "n_params": len(param_cols),
          "na_params_total": int(g[param_cols].isna().sum().sum()),
      }))
      .reset_index()
)
summary["n_cells"] = summary["n_rows"] * summary["n_params"]
summary["na_fraction"] = summary["na_params_total"] / summary["n_cells"]

# -------------------- Filter IDs --------------------
keep_ids = summary.loc[summary["na_fraction"] <= THRESHOLD, "ID"].astype(str)
drop_ids = summary.loc[summary["na_fraction"] > THRESHOLD, "ID"].astype(str)

filtered_df = df[df["ID"].astype(str).isin(set(keep_ids))].copy()

# -------------------- Save outputs --------------------
filtered_df.to_excel(OUTPUT_FILE, index=False)

with pd.ExcelWriter(SUMMARY_FILE) as writer:
    summary.sort_values(["na_fraction","ID"]).to_excel(writer, sheet_name="Summary_All", index=False)
    pd.DataFrame({"ID": keep_ids.sort_values().tolist()}).to_excel(writer, sheet_name="Keep_IDs", index=False)
    pd.DataFrame({"ID": drop_ids.sort_values().tolist()}).to_excel(writer, sheet_name="Drop_IDs", index=False)

print(f"✅ Kept {filtered_df['ID'].nunique()} IDs (≤ {int(THRESHOLD*100)}% NA) → {OUTPUT_FILE}")
print(f"ℹ️ Summary + ID lists saved → {SUMMARY_FILE}")
