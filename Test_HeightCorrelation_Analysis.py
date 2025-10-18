#!/usr/bin/env python3
"""
Spatial Height Analysis (Option B) with distance thresholds
- Aggregates to 1 row per ID (median)
- Computes pairwise distances & |ΔHeight_m|
- Repeats correlation analysis for:
    * ALL pairs
    * pairs within 10 km
    * pairs within 5 km
- Saves per-scenario plots and Excel sheets, plus a comparison table

Dependencies:
    pip install pandas numpy scipy matplotlib openpyxl tqdm
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import random
import warnings
from tqdm import tqdm

# ===================== CONFIG (edit if needed) =====================
INPUT_PATH = Path("/Users/lisa-marieweidl/Desktop/Corr Height/Water_Weather_Final_Prep_geo.xlsx")
OUTPUT_DIR = INPUT_PATH.parent / "spatial_height_analysis"
SHEET_NAME = None          # None -> first sheet
ID_COL = "ID"
LAT_COL = "Lat"
LON_COL = "Lon"
HGT_COL = "Height_m"

MAX_PAIRS = 2_000_000      # cap pair count (sampling safeguard)
TOP_N_PLOTS = 6            # number of plots per scenario
RANDOM_SEED = 42

# Distance thresholds to evaluate (km). "all" is always included automatically.
DISTANCE_THRESHOLDS = [10, 5]   # order matters for comparison sheet
# ==================================================================


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="ignore")
            except Exception:
                pass
    return df


def load_and_prepare(path: Path, sheet_name=None) -> pd.DataFrame:
    # Excel: pick first sheet if not specified
    if path.suffix.lower() in {".xlsx", ".xls"}:
        if sheet_name is None:
            xls = pd.ExcelFile(path)
            first = xls.sheet_names[0]
            print(f"[INFO] Using first sheet: {first}")
            df = pd.read_excel(path, sheet_name=first)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    for c in [ID_COL, LAT_COL, LON_COL, HGT_COL]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found. Got: {df.columns.tolist()}")
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = _safe_numeric(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = list(dict.fromkeys([ID_COL, LAT_COL, LON_COL, HGT_COL] + numeric_cols))
    df = df[keep_cols]

    # Aggregate to one row per station ID (median across numeric)
    agg_spec = {c: "median" for c in numeric_cols}
    agg_spec.update({LAT_COL: "first", LON_COL: "first", HGT_COL: "first"})
    g = df.groupby(ID_COL, as_index=False).agg(agg_spec)
    g = g.dropna(subset=[LAT_COL, LON_COL])
    return g


def all_unique_pairs(n: int) -> np.ndarray:
    m = n * (n - 1) // 2
    if m <= MAX_PAIRS:
        i_idx, j_idx = [], []
        for i in tqdm(range(n - 1), desc="Building pairs"):
            cnt = n - i - 1
            i_idx.append(np.full(cnt, i, dtype=np.int32))
            j_idx.append(np.arange(i + 1, n, dtype=np.int32))
        return np.column_stack([np.concatenate(i_idx), np.concatenate(j_idx)])
    else:
        tqdm.write(f"[INFO] Sampling {MAX_PAIRS:,} pairs (too many for full matrix).")
        random.seed(RANDOM_SEED)
        pairs = set()
        while len(pairs) < MAX_PAIRS:
            i = random.randrange(n)
            j = random.randrange(n)
            if i < j:
                pairs.add((i, j))
            elif j < i:
                pairs.add((j, i))
        arr = np.fromiter((k for ij in pairs for k in ij), dtype=np.int32).reshape(-1, 2)
        return arr


def compute_pairwise_core(df_station: pd.DataFrame) -> pd.DataFrame:
    ids = df_station[ID_COL].to_numpy()
    coords_deg = df_station[[LAT_COL, LON_COL]].to_numpy()
    height = df_station[HGT_COL].to_numpy(dtype=float)
    n = len(ids)
    pairs = all_unique_pairs(n)

    # Vectorized haversine
    lat1 = np.radians(coords_deg[pairs[:, 0], 0])
    lon1 = np.radians(coords_deg[pairs[:, 0], 1])
    lat2 = np.radians(coords_deg[pairs[:, 1], 0])
    lon2 = np.radians(coords_deg[pairs[:, 1], 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    distance_km = 6371.0 * c

    abs_hdiff = np.abs(height[pairs[:, 0]] - height[pairs[:, 1]])

    return pd.DataFrame({
        "ID1": ids[pairs[:, 0]],
        "ID2": ids[pairs[:, 1]],
        "Distance_km": distance_km,
        "AbsHeightDiff": abs_hdiff
    })


def corr_safe(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan, 0), (np.nan, np.nan, 0)
    r_p, p_p = pearsonr(x[mask], y[mask])
    r_s, p_s = spearmanr(x[mask], y[mask])
    return (r_p, p_p, mask.sum()), (r_s, p_s, mask.sum())


def analyze_for_pairs(df_station: pd.DataFrame,
                      pairs_df: pd.DataFrame,
                      out_dir: Path,
                      scenario_tag: str,
                      top_n_plots: int = TOP_N_PLOTS) -> pd.DataFrame:
    """
    Compute correlations for each numeric variable vs distance and vs |Δheight|,
    restricted to 'pairs_df' (already filtered by distance if needed).
    Save plots under out_dir/scenario_tag.
    """
    id_to_idx = {idv: i for i, idv in enumerate(df_station[ID_COL].tolist())}
    drop_cols = {ID_COL, HGT_COL, LAT_COL, LON_COL}
    num_cols = [c for c in df_station.select_dtypes(include=[np.number]).columns if c not in drop_cols]

    idx1 = pairs_df["ID1"].map(id_to_idx).to_numpy()
    idx2 = pairs_df["ID2"].map(id_to_idx).to_numpy()
    dist = pairs_df["Distance_km"].to_numpy(dtype=float)
    abs_hdiff = pairs_df["AbsHeightDiff"].to_numpy(dtype=float)

    scenario_plot_dir = out_dir / "plots" / f"within_{scenario_tag}"
    scenario_plot_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for col in tqdm(num_cols, desc=f"Analyzing ({scenario_tag})"):
        vals = df_station[col].to_numpy(dtype=float)
        abs_diff = np.abs(vals[idx1] - vals[idx2])

        (pr, pp, n1), (sr, sp, n2) = corr_safe(dist, abs_diff)
        (pr_h, pp_h, n1h), (sr_h, sp_h, n2h) = corr_safe(abs_hdiff, abs_diff)

        records.append({
            "variable": col,
            "n_pairs_used": int(n1),
            "pearson_r_dist": pr,
            "pearson_p_dist": pp,
            "spearman_r_dist": sr,
            "spearman_p_dist": sp,
            "pearson_r_absH": pr_h,
            "pearson_p_absH": pp_h,
            "spearman_r_absH": sr_h,
            "spearman_p_absH": sp_h
        })

    res = pd.DataFrame(records).sort_values(by="pearson_r_dist", key=lambda s: s.abs(), ascending=False)

    # Top plots for this scenario
    top_vars = res.reindex(res["pearson_r_dist"].abs().sort_values(ascending=False).index)["variable"].head(top_n_plots)
    for col in top_vars:
        vals = df_station[col].to_numpy(dtype=float)
        abs_diff = np.abs(vals[idx1] - vals[idx2])

        plt.figure()
        plt.scatter(dist, abs_diff, alpha=0.25)
        plt.xlabel("Distance between stations (km)")
        plt.ylabel(f"|Δ {col}|")
        plt.title(f"{col} — vs distance (within {scenario_tag})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(scenario_plot_dir / f"dist_vs_absdiff__{col}.png", dpi=160)
        plt.close()

        plt.figure()
        plt.scatter(abs_hdiff, abs_diff, alpha=0.25)
        plt.xlabel("|Δ Height_m|")
        plt.ylabel(f"|Δ {col}|")
        plt.title(f"{col} — vs elevation diff (within {scenario_tag})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(scenario_plot_dir / f"absH_vs_absdiff__{col}.png", dpi=160)
        plt.close()

    return res


def add_sig_flags(results: pd.DataFrame) -> pd.DataFrame:
    def sig_flag(p): return "✓" if pd.notna(p) and p < 0.05 else ""
    out = results.copy()
    out["sig_dist"] = out["pearson_p_dist"].apply(sig_flag)
    out["sig_absH"] = out["pearson_p_absH"].apply(sig_flag)
    return out


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tqdm.write("[1/5] Loading & aggregating…")
    stations = load_and_prepare(INPUT_PATH, sheet_name=SHEET_NAME)
    stations = stations.dropna(subset=[HGT_COL])
    tqdm.write(f"Stations after aggregation: {len(stations)}")

    tqdm.write("[2/5] Building pairwise set…")
    pairs_all = compute_pairwise_core(stations)
    tqdm.write(f"All pairs: {len(pairs_all):,}")

    # Build scenarios: "all", "<=10km", "<=5km"
    scenarios = [("all", pairs_all)]  # unfiltered
    for thr in DISTANCE_THRESHOLDS:
        scenarios.append((f"{thr}km", pairs_all[pairs_all["Distance_km"] <= float(thr)].reset_index(drop=True)))
        tqdm.write(f"Pairs within {thr} km: {len(scenarios[-1][1]):,}")

    # Analyze each scenario
    tqdm.write("[3/5] Analyzing scenarios…")
    results_by_tag = {}
    for tag, pairs_df in scenarios:
        if len(pairs_df) < 3:
            tqdm.write(f"[WARN] Not enough pairs for scenario '{tag}'. Skipping.")
            continue
        res = analyze_for_pairs(stations, pairs_df, OUTPUT_DIR, scenario_tag=tag, top_n_plots=TOP_N_PLOTS)
        results_by_tag[tag] = res

    # Add significance flags and toplists for each scenario
    tqdm.write("[4/5] Preparing outputs…")
    cleaned_by_tag = {}
    tops_by_tag = {}
    for tag, res in results_by_tag.items():
        clean = add_sig_flags(res)
        cleaned_by_tag[tag] = clean[[
            "variable", "n_pairs_used",
            "pearson_r_dist", "pearson_p_dist", "sig_dist",
            "spearman_r_dist", "spearman_p_dist",
            "pearson_r_absH", "pearson_p_absH", "sig_absH",
            "spearman_r_absH", "spearman_p_absH",
        ]].copy()

        tops_by_tag[tag] = {
            "top_by_distance": clean.reindex(clean["pearson_r_dist"].abs().sort_values(ascending=False).index).head(15),
            "top_by_height":   clean.reindex(clean["pearson_r_absH"].abs().sort_values(ascending=False).index).head(15),
        }

    # Build a compact comparison table for key metrics across scenarios
    # (Pearson r with distance for each variable; columns: all, 10km, 5km, and deltas)
    tqdm.write("[5/5] Writing Excel report…")
    out_xlsx = OUTPUT_DIR / "spatial_height_report_thresholds.xlsx"
    with pd.ExcelWriter(out_xlsx, mode="w", engine="openpyxl") as xl:
        # Per-scenario sheets
        for tag, res in results_by_tag.items():
            res.to_excel(xl, sheet_name=f"summary_{tag}", index=False)
        for tag, clean in cleaned_by_tag.items():
            clean.to_excel(xl, sheet_name=f"summary_clean_{tag}", index=False)
        for tag, tops in tops_by_tag.items():
            tops["top_by_distance"].to_excel(xl, sheet_name=f"top_by_distance_{tag}", index=False)
            tops["top_by_height"].to_excel(xl, sheet_name=f"top_by_height_{tag}", index=False)

        # Height vs distance sanity (all pairs)
        if "all" in results_by_tag:
            pr, pp = pearsonr(pairs_all["Distance_km"], pairs_all["AbsHeightDiff"])
            sr, sp = spearmanr(pairs_all["Distance_km"], pairs_all["AbsHeightDiff"])
            pd.DataFrame({
                "pearson_r": [pr], "pearson_p": [pp],
                "spearman_r": [sr], "spearman_p": [sp],
                "n_pairs": [len(pairs_all)]
            }).to_excel(xl, sheet_name="height_vs_distance_all", index=False)

        # Comparison table (Pearson r with distance by scenario)
        # start from ALL as base, then join <=10km and <=5km if available
        base = results_by_tag.get("all", pd.DataFrame()).set_index("variable")[["pearson_r_dist"]].rename(columns={"pearson_r_dist": "r_dist_all"})
        comp = base.copy()
        if "10km" in results_by_tag:
            comp = comp.join(results_by_tag["10km"].set_index("variable")[["pearson_r_dist"]]
                             .rename(columns={"pearson_r_dist": "r_dist_10km"}), how="outer")
        if "5km" in results_by_tag:
            comp = comp.join(results_by_tag["5km"].set_index("variable")[["pearson_r_dist"]]
                             .rename(columns={"pearson_r_dist": "r_dist_5km"}), how="outer")

        # Deltas relative to ALL (positive delta = stronger correlation when restricting distance)
        if "r_dist_all" in comp.columns and "r_dist_10km" in comp.columns:
            comp["delta_10km_minus_all"] = comp["r_dist_10km"] - comp["r_dist_all"]
        if "r_dist_all" in comp.columns and "r_dist_5km" in comp.columns:
            comp["delta_5km_minus_all"] = comp["r_dist_5km"] - comp["r_dist_all"]

        comp.reset_index().to_excel(xl, sheet_name="comparison_r_by_distance", index=False)

    tqdm.write(f"[✅ Done] Excel: {out_xlsx}\nPlots (per-scenario): {OUTPUT_DIR / 'plots'}")


if __name__ == "__main__":
    main()
