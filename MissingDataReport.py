# missing_reports_mode.py
import pandas as pd

# =========================
# CONFIG — switch here
# =========================
MODE = "join"   # "weather", "water", or "join"

# (Optional) Drop parameter columns (C→end) with > this fraction missing.
# Set to None to disable (default = no column dropping here).
MISSING_DROP_THRESHOLD = None  # e.g., 0.20

# Define input/output files
FILEMAP = {
    "weather": {
        "input":  "Cleaned_Weather_2109.xlsx",
        "output": "Missing_Weather_2109.xlsx",
    },
    "water": {
        "input":  "Cleaned_Water_2109.xlsx",
        "output": "Missing_Water_2109.xlsx",
    },
    "join": {
        "water_input":   "Cleaned_Water_2109.xlsx",
        "weather_input": "Cleaned_Weather_2109.xlsx",
        "output":        "MissingReport_Water_Weather_Joined.xlsx",
    }
}

# Water → Weather station mapping (strings)
WATER_TO_WEATHER = {
    "PG31000292": "905",
    "PG31100282": "191",
    "PG31100312": "191",
    "PG31100322": "905",
    "PG32200012": "1903",
    "PG32200042": "726",
    "PG32200102": "726",
    "PG32500392": "1903",
}

# Presence rule for water (per day, per parameter): >= N samples = present
MIN_WATER_SAMPLES_PER_DAY = 1  # 1 = "any sample that day" counts

# Use all calendar days (365/366) per year when counting missing
USE_CIVIL_CALENDAR = True

# Constant year span for reports (inclusive)
YEAR_RANGE = range(1993, 2025)  # 1993–2024


# =========================
# Helpers (no cleaning here)
# =========================
def _normalize_dates_to_day(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


def _day_universe_from_group(g: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(g["Date"].dropna().unique()))


def civil_year_index(year: int) -> pd.DatetimeIndex:
    start = f"{int(year)}-01-01"
    end   = f"{int(year)}-12-31"
    return pd.date_range(start, end, freq="D")


def _missing_days_from_universe(group: pd.DataFrame, cols: list, day_universe: pd.DatetimeIndex) -> pd.Series:
    """
    Count missing *days* for each column relative to `day_universe`.
    A day is 'present' if ANY non-NaN exists for that column on that date.
    """
    if not cols or len(day_universe) == 0:
        return pd.Series({c: 0 for c in cols}, dtype="int64")
    pres = group.groupby("Date", dropna=False)[cols].agg(lambda s: s.notna().any())
    pres = pres.reindex(day_universe, fill_value=False)
    return (~pres).sum(axis=0).astype("int64")


# =========================
# Single-dataset reports
# =========================
def build_missing_report_single(df: pd.DataFrame, id_col: str, outfile: str, label: str):
    """
    Per-(ID, year) missing-DAYS report against CIVIL calendar (all days in the year),
    expanded to include EVERY year in YEAR_RANGE for each ID (rows with no data -> zeros).
    Also includes 'n_rows' = number of raw data records for that ID/year.
    """
    if "Date" not in df.columns:
        raise ValueError(f"{label}: expected a 'Date' column.")
    if id_col not in df.columns:
        raise ValueError(f"{label}: expected an '{id_col}' column.")

    df = df.copy()
    df["Date"] = _normalize_dates_to_day(df["Date"])
    df["year"] = df["Date"].dt.year

    # Parameter columns = all except ID/Date/year
    param_cols = [c for c in df.columns if c not in [id_col, "Date", "year"]]

    # Optional (local) column drop—does NOT affect missing counting universe
    if MISSING_DROP_THRESHOLD is not None and param_cols:
        miss_frac = df[param_cols].isna().mean()
        to_drop = miss_frac[miss_frac > MISSING_DROP_THRESHOLD].index.tolist()
        if to_drop:
            df = df.drop(columns=to_drop)
            param_cols = [c for c in param_cols if c not in to_drop]
            print(f"{label}: dropped {len(to_drop)} columns with >{int(MISSING_DROP_THRESHOLD*100)}% missing.")

    # n_rows per (ID, year)
    nrows = df.groupby([id_col, "year"], dropna=False).size().rename("n_rows").reset_index()

    # Compute per-(ID,year) missing days vs CIVIL calendar
    rows = []
    ids = sorted(df[id_col].astype(str).unique())

    for the_id in ids:
        g_id = df[df[id_col].astype(str) == the_id]
        for yr in YEAR_RANGE:
            U = civil_year_index(yr) if USE_CIVIL_CALENDAR else _day_universe_from_group(g_id[g_id["year"] == yr])

            # Daily presence: any non-NaN that day for each param
            if param_cols and not g_id.empty:
                g_y = g_id[g_id["year"] == yr]
                if not g_y.empty:
                    pres = g_y.groupby("Date", dropna=False)[param_cols].agg(lambda s: s.notna().any())
                else:
                    pres = pd.DataFrame(index=pd.DatetimeIndex([]), columns=param_cols, dtype="bool")
            else:
                pres = pd.DataFrame(index=pd.DatetimeIndex([]), columns=param_cols, dtype="bool")

            pres = pres.reindex(U, fill_value=False)
            na_counts = (~pres).sum(axis=0).astype("int64").to_dict() if len(param_cols) else {}

            # n_rows for this (ID, yr)
            mask = (nrows[id_col].astype(str) == the_id) & (nrows["year"] == yr)
            n_rows_val = int(nrows.loc[mask, "n_rows"].sum()) if mask.any() else 0

            row = {"ID": the_id, "year": int(yr), "n_rows": n_rows_val}
            for c in param_cols:
                row[f"Na_{c}"] = int(na_counts.get(c, 0))
            rows.append(row)

    report = pd.DataFrame(rows).sort_values(["ID", "year"])
    report.to_excel(outfile, index=False)
    print(f"✅ {label} missing-days report (civil calendar, {YEAR_RANGE.start}-{YEAR_RANGE.stop-1}) saved to: {outfile}")


# =========================
# Joined report (civil calendar)
# =========================
def build_missing_report_join(water_df: pd.DataFrame, weather_df: pd.DataFrame, outfile: str):
    """
    Joined missing-days report aligned to a CIVIL calendar (all days in the year),
    expanded to include EVERY year in YEAR_RANGE for each Water_ID.
    Water presence = >= MIN_WATER_SAMPLES_PER_DAY samples that day.
    Includes n_rows_water and n_rows_weather from input files.
    """
    # Ensure essential columns
    if "Date" not in water_df.columns or "ID" not in water_df.columns:
        raise ValueError("WATER: expected 'ID' and 'Date' columns.")
    if "Date" not in weather_df.columns:
        raise ValueError("WEATHER: expected 'Date' column.")
    if "weather_ID" not in weather_df.columns:
        if "ID" in weather_df.columns:
            weather_df = weather_df.rename(columns={"ID": "weather_ID"})
        else:
            raise ValueError("WEATHER: expected 'weather_ID' or 'ID' column for station ID.")

    water_df = water_df.copy()
    weather_df = weather_df.copy()

    # Normalize to day + year
    water_df["Date"] = _normalize_dates_to_day(water_df["Date"])
    weather_df["Date"] = _normalize_dates_to_day(weather_df["Date"])
    water_df["year"] = water_df["Date"].dt.year
    weather_df["year"] = weather_df["Date"].dt.year

    # IDs as strings
    water_df["ID"] = water_df["ID"].astype(str)
    weather_df["weather_ID"] = weather_df["weather_ID"].astype(str)

    # Parameter columns
    water_params = [c for c in water_df.columns if c not in ["ID", "Date", "year"]]
    weather_params = [c for c in weather_df.columns if c not in ["weather_ID", "Date", "year"]]

    # Map water → weather
    water_df["WeatherJoinID"] = water_df["ID"].map(WATER_TO_WEATHER).astype("string")

    # Weather per-day presence (any non-NaN). We'll compute it from data,
    # but reindex to the CIVIL calendar so missing weather days are counted.
    def _bool_any(x): return x.notna().any()
    weather_by_day = {}
    if weather_params:
        for (wid, yr), g in weather_df.groupby(["weather_ID", "year"], dropna=False):
            if g.empty:
                continue
            by_day = g.groupby("Date", dropna=False)[weather_params].agg(_bool_any)
            weather_by_day[(str(wid), int(yr))] = by_day  # Date × weather_params (bool)

    # --- Count n_rows per ID/year before aggregation ---
    nrows_water = water_df.groupby(["ID", "year"], dropna=False).size().rename("n_rows_water")
    nrows_weather = weather_df.groupby(["weather_ID", "year"], dropna=False).size().rename("n_rows_weather")

    out_rows = []
    water_ids = sorted(water_df["ID"].unique())  # only IDs present in WATER file

    for water_id in water_ids:
        wid = WATER_TO_WEATHER.get(water_id)
        g_w_all = water_df[water_df["ID"] == water_id]

        for yr in YEAR_RANGE:
            U = civil_year_index(yr) if USE_CIVIL_CALENDAR else pd.DatetimeIndex([])

            # Subset WATER rows for this (ID, year)
            g_w = g_w_all[g_w_all["year"] == yr]

            # Water daily presence rule (>= MIN_WATER_SAMPLES_PER_DAY)
            if water_params and not g_w.empty:
                def _enough(s: pd.Series) -> bool:
                    return s.notna().sum() >= MIN_WATER_SAMPLES_PER_DAY
                w_by_day = g_w.groupby("Date", dropna=False)[water_params].agg(_enough)
            else:
                w_by_day = pd.DataFrame(index=pd.DatetimeIndex([]), columns=water_params, dtype="bool")

            w_by_day = w_by_day.reindex(U, fill_value=False)

            # Missing-days per param (water)
            na_water_series = (~w_by_day).sum(axis=0).astype("int64") if len(water_params) else pd.Series(dtype="int64")

            # Weather per-day presence on CIVIL calendar
            if weather_params and wid is not None and (str(wid), int(yr)) in weather_by_day:
                wx_by_day = weather_by_day[(str(wid), int(yr))].reindex(U, fill_value=False)
                na_weather_series = (~wx_by_day).sum(axis=0).astype("int64")
            else:
                # If no weather data for that station/year, every civil day is "missing" for weather params
                na_weather_series = pd.Series({c: len(U) for c in weather_params}, dtype="int64") if len(weather_params) else pd.Series(dtype="int64")

            # KPIs
            n_weather_days = int(len(U))
            water_days_present = int(w_by_day.any(axis=1).sum()) if len(water_params) else 0
            water_days_missing = int(n_weather_days - water_days_present)
            pct_coverage_water = (water_days_present / n_weather_days) if n_weather_days else 0.0

            # Row counts (from precomputed Series)
            n_w = int(nrows_water.get((water_id, yr), 0))
            n_wx = int(nrows_weather.get((str(wid), yr), 0)) if wid is not None else 0

            row = {
                "Water_ID": water_id,
                "Weather_ID": wid,
                "year": int(yr),
                "n_rows_water": n_w,
                "n_rows_weather": n_wx,
                "n_weather_days": n_weather_days,
                "water_days_present": water_days_present,
                "water_days_missing": water_days_missing,
                "pct_coverage_water": round(pct_coverage_water, 4),
            }
            for c in water_params:
                row[f"Na_{c}"] = int(na_water_series.get(c, 0))
            for c in weather_params:
                row[f"Na_{c}"] = int(na_weather_series.get(c, 0))

            out_rows.append(row)

    report = pd.DataFrame(out_rows)

    # Order columns
    base = [
        "Water_ID", "Weather_ID", "year",
        "n_rows_water", "n_rows_weather",
        "n_weather_days", "water_days_present", "water_days_missing",
        "pct_coverage_water"
    ]
    water_na_cols = sorted([c for c in report.columns if c.startswith("Na_")])
    # keep weather Na_ after water Na_
    weather_na_cols = [c for c in water_na_cols if c.replace("Na_", "") in weather_params]
    water_na_cols = [c for c in water_na_cols if c.replace("Na_", "") in water_params]

    final_cols = [c for c in base if c in report.columns] + water_na_cols + weather_na_cols
    report = report[final_cols].sort_values(["Water_ID", "year"])
    report.to_excel(outfile, index=False)
    print(f"✅ JOIN (civil calendar, {YEAR_RANGE.start}-{YEAR_RANGE.stop-1}) saved to: {outfile}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if MODE not in FILEMAP:
        raise ValueError("MODE must be 'weather', 'water', or 'join'.")

    if MODE in ("weather", "water"):
        in_path  = FILEMAP[MODE]["input"]
        out_path = FILEMAP[MODE]["output"]
        df = pd.read_excel(in_path)

        # Determine ID col
        if MODE == "weather":
            if "weather_ID" in df.columns:
                id_col = "weather_ID"
            elif "ID" in df.columns:
                id_col = "ID"
            else:
                raise ValueError("WEATHER: expected 'weather_ID' or 'ID' column.")
        else:  # water
            if "ID" not in df.columns:
                raise ValueError("WATER: expected an 'ID' column.")
            id_col = "ID"

        build_missing_report_single(df, id_col=id_col, outfile=out_path, label=MODE.upper())

    elif MODE == "join":
        paths = FILEMAP["join"]
        water_df   = pd.read_excel(paths["water_input"])
        weather_df = pd.read_excel(paths["weather_input"])
        build_missing_report_join(water_df, weather_df, paths["output"])
