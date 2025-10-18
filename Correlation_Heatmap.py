import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import re

# ---------- INPUT ----------
INPUT_FILE = r"/Users/lisa-marieweidl/Desktop/MergingAll/AllData_Merge_Corr.xlsx"
SHEET_NAME = "AllGroups_Union_PGonly"

# ---------- OUTPUT ----------
OUT_SUBDIR = "correlation_output_all"
OUT_PREFIX = "correlation"

# ---------- STYLE SETTINGS ----------
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 13
CELL_SIZE = 0.28
CMAP = "coolwarm"

# Axes layout (fractions of figure)
LEFT   = 0.11
RIGHT  = 0.985
BOTTOM = 0.13
TOP    = 0.92
CBAR_W = 0.012

# Padding around saved figure (in inches ≈ 0.5 cm)
PADDING_INCH = 0.2

# ---------- HELPERS ----------
def _safe_filename(s: str) -> str:
    s = str(s).strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z._-äöüÄÖÜß ]", "", s).replace(" ", "_") or "Group"

def _plot_heatmap(corr: pd.DataFrame, title: str, out_dir: Path, filename_prefix: str, pdf: PdfPages = None):
    """Heatmap with readable labels, tight grid, thin colorbar, and 0.5 cm padding."""
    n = corr.shape[0]
    fig_w = max(8, n * CELL_SIZE)
    fig_h = max(6, n * CELL_SIZE)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Main axes occupy nearly full canvas
    ax_width  = RIGHT - LEFT - CBAR_W - 0.006
    ax_height = TOP - BOTTOM
    ax = fig.add_axes([LEFT, BOTTOM, ax_width, ax_height])
    cbar_ax = fig.add_axes([LEFT + ax_width + 0.006, BOTTOM, CBAR_W, ax_height])

    sns.heatmap(
        corr,
        annot=False,
        cmap=CMAP,
        square=True,
        linewidths=0.6,
        linecolor="0.75",
        cbar_ax=cbar_ax,
        cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1.0], "label": ""},
        ax=ax,
    )

    # Full labels
    ax.set_xticks(np.arange(len(corr.columns)) + 0.5)
    ax.set_yticks(np.arange(len(corr.index)) + 0.5)
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=LABEL_FONTSIZE)
    ax.set_yticklabels(corr.index, rotation=0, fontsize=LABEL_FONTSIZE)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)

    # Clean frame
    ax.tick_params(axis="both", which="major", length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ---- Save with safe padding ----
    png_path = out_dir / f"{filename_prefix}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=PADDING_INCH)
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=PADDING_INCH)
    plt.close(fig)
    print(f"Saved: {png_path}")

# ---------- MAIN ----------
def main():
    in_path = Path(INPUT_FILE).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_dir = in_path.parent / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read Excel
    df = pd.read_excel(in_path, sheet_name=SHEET_NAME)
    if df.shape[1] < 3:
        raise ValueError("Expected at least 3 columns (Date, Group, parameters...).")

    # Numeric columns from column C onwards
    param_cols = df.columns[2:]
    df[param_cols] = df[param_cols].apply(pd.to_numeric, errors="coerce")

    # ---------- OVERVIEW ----------
    corr_all = df[param_cols].corr(method="pearson")
    corr_all.to_excel(out_dir / f"{OUT_PREFIX}_overview.xlsx")

    pdf_file = out_dir / f"{OUT_PREFIX}_all_and_Groups.pdf"
    with PdfPages(pdf_file) as pdf:
        _plot_heatmap(
            corr=corr_all,
            title="Correlation Heatmap – All Parameters (Overview)",
            out_dir=out_dir,
            filename_prefix=f"{OUT_PREFIX}_overview",
            pdf=pdf,
        )

        # ---------- PER GROUP ----------
        for group_val in df["Group"].dropna().unique():
            sub = df[df["Group"] == group_val]
            if sub.shape[0] < 2:
                print(f"Skipping {group_val}: not enough data.")
                continue

            corr = sub[param_cols].corr(method="pearson")
            if corr.isna().all().all():
                print(f"Skipping {group_val}: all correlations NaN.")
                continue

            corr_path = out_dir / f"{OUT_PREFIX}_{_safe_filename(group_val)}.xlsx"
            corr.to_excel(corr_path, index=True)

            _plot_heatmap(
                corr=corr,
                title=f"Correlation Heatmap – Group {group_val}",
                out_dir=out_dir,
                filename_prefix=f"{OUT_PREFIX}_{_safe_filename(group_val)}",
                pdf=pdf,
            )

    print(f"Combined PDF saved to: {pdf_file}")

if __name__ == "__main__":
    main()

