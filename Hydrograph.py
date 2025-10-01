# hydrography_quarters_fixed_axis_grid.py

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from pathlib import Path

# ---------- INPUT ----------
INPUT_FILE = r"/Users/lisa-marieweidl/Desktop/SEEP/MSc Thesis/Data/Water/Hydrographs/WaterLevel_DetailedDate.xlsx"
SHEET_NAME = 0  # or a sheet name string

# ---------- COLUMNS ----------
DATE_COL = "DATE"
ID_COL = "ID"
LEVEL_COL = "WATER LEVEL"

# ---------- OUTPUT ----------
OUT_SUBDIR = "hydrography_output_quarters"
OUT_PREFIX = "hydro_quarters_grid"

# ---------- PLOT SETTINGS ----------
POINT_SIZE = 8           # small points
LINE_WIDTH = 1.0         # connecting line
WIDTH_INCH = 9
HEIGHT_PER_TICK = 0.26   # vertical size per quarter
GREY_LABEL = "0.6"       # missing quarter label color
OBS_COLOR = "k"          # observed data color
XTICK_STEP = 0.5
X_MAX = 13.0

# ---------- HELPERS ----------
def _safe_filename(s: str) -> str:
    s = str(s).strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z._-äöüÄÖÜß ]", "", s).replace(" ", "_") or "ID"

def _parse_level(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("\u2212", "-").replace("\xa0", " ")
    m = re.search(r'[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?', s)
    if not m: return np.nan
    token = m.group(0).replace(",", ".")
    try: return float(token)
    except ValueError: return np.nan

def _quarter_str(period_q: pd.Period) -> str:
    return f"Q{period_q.quarter} {period_q.year}"

def _quarter_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.PeriodIndex:
    return pd.period_range(start=start_dt.to_period("Q"), end=end_dt.to_period("Q"), freq="Q")

def _quarter_to_numeric_y(q: pd.Period) -> float:
    return q.year + (q.quarter - 1) / 4.0

# ---------- MAIN ----------
def main():
    in_path = Path(INPUT_FILE).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = in_path.parent / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path, sheet_name=SHEET_NAME)
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[LEVEL_COL] = df[LEVEL_COL].apply(_parse_level)
    df = df.dropna(subset=[DATE_COL])
    df[ID_COL] = df[ID_COL].astype(str)

    if df.empty:
        raise ValueError("No valid rows after parsing dates.")

    pdf_file = out_dir / f"{_safe_filename(OUT_PREFIX)}.pdf"
    with PdfPages(pdf_file) as pdf:
        for id_val in df[ID_COL].dropna().unique():
            sub = df[df[ID_COL] == id_val].sort_values(DATE_COL)
            if sub.empty: continue

            # Build quarters
            q_index_full = _quarter_range(sub[DATE_COL].min(), sub[DATE_COL].max())
            sub_q = sub.copy()
            sub_q["quarter"] = sub_q[DATE_COL].dt.to_period("Q")
            quarterly = sub_q.groupby("quarter", observed=True)[LEVEL_COL].mean()
            quarterly = quarterly.reindex(q_index_full)

            y_all = np.array([_quarter_to_numeric_y(q) for q in q_index_full])
            x_all = quarterly.values.astype(float)

            n_q = len(q_index_full)
            height = max(HEIGHT_PER_TICK * n_q, 3.0)
            fig, ax = plt.subplots(figsize=(WIDTH_INCH, height))

            # Open diagram style: hide top/right
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Grid in background
            ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.7)

            # Line + dots
            if np.isfinite(x_all).any():
                ax.plot(x_all, y_all, "-", linewidth=LINE_WIDTH, color=OBS_COLOR)
            obs_mask = ~np.isnan(x_all)
            ax.scatter(x_all[obs_mask], y_all[obs_mask], s=POINT_SIZE, c=OBS_COLOR)

            # Y-axis ticks
            ax.set_yticks(y_all)
            ax.set_yticklabels([_quarter_str(q) for q in q_index_full])
            for label, is_missing in zip(ax.get_yticklabels(), np.isnan(x_all)):
                if is_missing:
                    label.set_color(GREY_LABEL)

            ax.invert_yaxis()  # latest on top

            # Fixed X-axis
            xticks = np.arange(0.0, X_MAX + 1e-9, XTICK_STEP)
            ax.set_xticks(xticks)
            ax.set_xlim(0.0, X_MAX)
            ax.set_xlabel("Water level")
            ax.set_ylabel("Time")

            ax.set_title(f"Water Level of {id_val}")

            handles = [
                Line2D([0], [0], color=OBS_COLOR, linewidth=LINE_WIDTH, label="connected quarters"),
                Line2D([0], [0], marker="o", linestyle="", color=OBS_COLOR, markersize=6, label="quarterly mean"),
            ]
            ax.legend(handles=handles, frameon=False, loc="best")

            plt.tight_layout()

            png_path = out_dir / f"{_safe_filename(OUT_PREFIX)}_{_safe_filename(id_val)}.png"
            fig.savefig(png_path, dpi=300)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Saved: {png_path}")

    print(f"Saved PDF: {pdf_file}")

if __name__ == "__main__":
    main()
