import pandas as pd, re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILE = "Water_2109.xlsx"
OUT_SUBDIR = "outlier_output_water_test"
OUT_PREFIX = "outlier"
MIN_VALID = 10
GROUP_COL = "ID"
FS = 11
TS = 13
PAD = 0.2

THRESHOLDS = {
    "ARSEN mg/l": {"limit": 9e-3, "trend": 7.5e-3},
    "BOR mg/l": {"limit": 0.9, "trend": 0.75},
    "CADMIUM mg/l": {"limit": 4.5e-3, "trend": 3.75e-3},
    "CHROM-GESAMT mg/l": {"limit": 45e-3, "trend": 37.5e-3},
    "KUPFER mg/l": {"limit": 1.8, "trend": 1.5},
    "NICKEL mg/l": {"limit": 18e-3, "trend": 15e-3},
    "NITRAT mg/l": {"limit": 45.0, "trend": 37.5},
    "NITRIT mg/l": {"limit": 0.09, "trend": 0.075},
    "QUECKSILBER µg/l": {"limit": 0.9, "trend": 0.75},
    "AMMONIUM mg/l": {"limit": 0.45, "trend": 0.375},
    "CHLORID mg/l": {"limit": 180.0, "trend": 150.0},
    "SULFAT mg/l": {"limit": 225.0, "trend": 187.5},
    "ELEKTR. LEITF. (bei 25°C) µS/cm": {"limit": 2250.0, "trend": 1875.0},
    "ORTHOPHOSPHAT mg/l": {"limit": 0.30, "trend": 0.225},
}

def identify_outliers(s: pd.Series) -> pd.Series:
    # Z-score (Iglewicz & Hoaglin, 1993)
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=s.index)
    mz = 0.6745 * (s - med) / mad
    return mz.abs() > 3.5

def safe_name(s: str) -> str:
    s = str(s).strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z._-äöüÄÖÜß ]", "", s).replace(" ", "_") or "Var"

def plot_ts(df, param, mask, out_dir: Path, pdf=None):
    s = df[param].astype(float)
    valid = s.notna()
    if not valid.any():
        return
    dates, vals = df.loc[valid, "Date"], s[valid]
    out_dates, out_vals = dates[mask[valid]], vals[mask[valid]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(dates, vals, s=10, alpha=0.5, color="0.5", label="Data")
    if not out_dates.empty:
        ax.scatter(out_dates, out_vals, s=25, color="red", alpha=0.9, label="Outliers")

    thr = THRESHOLDS.get(param)
    if thr:
        ax.axhline(thr["limit"], color="orange", linestyle="--", linewidth=1.3, label="Threshold")
        ax.axhline(thr["trend"], color="green", linestyle=":", linewidth=1.3, label="Trend reversal")

    ax.set_xlabel("Date", fontsize=FS)
    ax.set_ylabel("Value", fontsize=FS)
    ax.set_title(param, fontsize=TS, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=FS)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=FS - 1)

    fname = out_dir / f"{OUT_PREFIX}_water_ts_outliers_{safe_name(param)}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=PAD)
    if pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=PAD)
    plt.close(fig)

def plot_ortho_per_id(df, mask, out_dir: Path):
    param = "ORTHOPHOSPHAT mg/l"
    if param not in df.columns or GROUP_COL not in df.columns:
        return
    thr = THRESHOLDS[param]
    limit, trend = thr["limit"], thr["trend"]
    mask = mask.astype(bool)

    for gid, df_id in df.groupby(GROUP_COL):
        s = df_id[param].astype(float)
        dates = df_id["Date"]
        valid = s.notna() & dates.notna()
        if not valid.any():
            continue
        s, dates = s[valid], dates[valid]
        id_mask = mask.reindex(df_id.index, fill_value=False)[valid]
        out_dates, out_vals = dates[id_mask], s[id_mask]

        vals_for_range = [s.min(), s.max()]
        for v in (limit, trend):
            if v is not None:
                vals_for_range.append(v)
        y_min = min(vals_for_range)
        y_max = max(vals_for_range)

        step = 0.5
        y_min_tick = int(y_min / step) * step
        y_max_scaled = y_max / step
        if y_max_scaled == int(y_max_scaled):
            y_max_tick = y_max
        else:
            y_max_tick = (int(y_max_scaled) + 1) * step

        ticks = []
        v = y_min_tick
        while v <= y_max_tick + 1e-9:
            ticks.append(round(v, 3))
            v += step

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(dates, s, s=10, alpha=0.5, color="0.5", label="Data")
        if not out_dates.empty:
            ax.scatter(out_dates, out_vals, s=25, color="red", alpha=0.9, label="Outliers")

        ax.set_ylim(y_min_tick, y_max_tick)
        ax.set_yticks(ticks)

        ax.axhline(limit, color="orange", linestyle="--", linewidth=1.3, label="Threshold")
        ax.axhline(trend, color="green", linestyle=":", linewidth=1.3, label="Trend reversal")

        ax.set_xlabel("Date", fontsize=FS)
        ax.set_ylabel("Value", fontsize=FS)
        ax.set_title(f"Orthophosphate for {gid}", fontsize=TS, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=FS)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=FS - 1)

        fname = out_dir / f"{OUT_PREFIX}_water_orthophosphate_ID_{safe_name(gid)}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=PAD)
        plt.close(fig)

def main():
    in_path = Path(INPUT_FILE).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_dir = in_path.parent / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path)
    df = df.loc[:, df.count() >= MIN_VALID]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    exclude = [c for c in ("A", "B") if c in df.columns]
    numeric = df.select_dtypes(include="number").columns.difference(exclude)
    outlier_mask = df[numeric].apply(identify_outliers)

    ts_pdf = out_dir / f"{OUT_PREFIX}_water_timeseries_outliers.pdf"
    with PdfPages(ts_pdf) as pdf:
        for col in numeric:
            plot_ts(df, col, outlier_mask[col], out_dir, pdf)

    if "ORTHOPHOSPHAT mg/l" in numeric:
        plot_ortho_per_id(df, outlier_mask["ORTHOPHOSPHAT mg/l"], out_dir)

    print("Saved:", ts_pdf)

if __name__ == "__main__":
    main()
