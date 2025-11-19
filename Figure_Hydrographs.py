import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from pathlib import Path

INPUT_FILE = r"Water_2109.xlsx"
DATE_COL = "Date"
ID_COL = "ID"
LEVEL_COL = "ABSTICH m"

OUT_SUBDIR = "hydrographs"
OUT_PREFIX = "hydrographs"

POINT_SIZE = 8
LINE_WIDTH = 1.0
WIDTH_INCH = 13
HEIGHT_INCH = 6
GREY_LABEL = "0.6"
OBS_COLOR = "k"
XTICK_STEP = 0.5
X_MAX = 13.0
LABEL_FONTSIZE = 6


def _safe_filename(s: str) -> str:
    s = str(s).strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z._-äöüÄÖÜß ]", "", s).replace(" ", "_") or "ID"


def _parse_level(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\u2212", "-").replace("\xa0", " ")
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return np.nan


def main():
    in_path = Path(INPUT_FILE).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = in_path.parent / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path)
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
            if sub.empty:
                continue

            q_full = pd.period_range(
                start=sub[DATE_COL].min().to_period("Q"),
                end=sub[DATE_COL].max().to_period("Q"),
                freq="Q",
            )
            sub_q = sub.copy()
            sub_q["quarter"] = sub_q[DATE_COL].dt.to_period("Q")
            quarterly = sub_q.groupby("quarter", observed=True)[LEVEL_COL].mean().reindex(q_full)

            x = np.array([q.year + (q.quarter - 1) / 4.0 for q in q_full])
            y = quarterly.values.astype(float)

            fig, ax = plt.subplots(figsize=(WIDTH_INCH, HEIGHT_INCH))

            ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
            ax.margins(x=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.7)

            if np.isfinite(y).any():
                ax.plot(x, y, "-", linewidth=LINE_WIDTH, color=OBS_COLOR)
            obs_mask = np.isfinite(y)
            ax.scatter(x[obs_mask], y[obs_mask], s=POINT_SIZE, c=OBS_COLOR)

            ax.set_xticks(x)
            labels = [f"Q{q.quarter} {q.year}" for q in q_full]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            for label, is_missing in zip(ax.get_xticklabels(), ~obs_mask):
                if is_missing:
                    label.set_color(GREY_LABEL)

            ax.tick_params(axis="both", which="major", labelsize=LABEL_FONTSIZE)

            yticks = np.arange(0.0, X_MAX + 1e-9, XTICK_STEP)
            ax.set_yticks(yticks)
            ax.set_ylim(0.0, X_MAX)
            ax.set_ylabel("Water Level (m)", fontsize=LABEL_FONTSIZE)
            ax.set_xlabel("Time (quarters)", fontsize=LABEL_FONTSIZE)
            ax.set_title(f"Water Level of {id_val}", fontsize=10)

            handles = [
                Line2D([0], [0], color=OBS_COLOR, linewidth=LINE_WIDTH, label="continuous measurements"),
                Line2D([0], [0], marker="o", linestyle="", color=OBS_COLOR, markersize=6, label="single measurements"),
            ]
            ax.legend(handles=handles, frameon=False, loc="best", fontsize=LABEL_FONTSIZE)

            plt.tight_layout()
            png_path = out_dir / f"{_safe_filename(OUT_PREFIX)}_{_safe_filename(id_val)}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {png_path}")

    print(f"Saved PDF: {pdf_file}")


if __name__ == "__main__":
    main()
