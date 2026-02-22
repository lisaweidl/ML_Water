# %%
import pandas as pd
# %%

df = pd.read_csv("Water_Cleaned.csv", sep=";", encoding="utf-8")
#sep=";", encoding="utf-8"
# %%
ID_COL = "ID"
DATE_COL = "DATE" if "DATE" in df.columns else "Date"

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

def time_resolution(s):
    diffs = s.dropna().sort_values().diff().dropna()
    return diffs.mode().iloc[0] if not diffs.empty else pd.NaT

value_cols = [c for c in df.columns if c not in [ID_COL, DATE_COL]]

out = (
    df.groupby(ID_COL, dropna=False)
      .apply(lambda g: pd.Series({
          "Timeframe_start": g[DATE_COL].min(),
          "Timeframe_end": g[DATE_COL].max(),
          "Time_Resolution": time_resolution(g[DATE_COL]),
          "Total_Values": g.shape[0] * len(value_cols),
          "Missing_Values": g[value_cols].isna().sum().sum(),
      }))
      .reset_index()
)

for col in ["Timeframe_start", "Timeframe_end"]:
    out[col] = out[col].dt.strftime("%Y-%m-%d")

print(out.to_string(index=False))
# %%

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})

data = df.copy()
DPI = 300

num_all = data.select_dtypes(include="number")

if num_all.shape[1] >= 2 and len(num_all) >= 2:
    plt.figure(figsize=(10, 8), dpi=DPI)
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(num_all.corr(), cmap=colormap, square=True, annot=False)
    plt.title("Correlation (All IDs)")
    plt.tight_layout()
    plt.show()

for _id, g in data.groupby("ID", sort=True):
    num = g.select_dtypes(include="number")

    plt.figure(figsize=(8, 6), dpi=DPI)
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(num.corr(), cmap=colormap, square=True, annot=False)
    plt.title(f"Correlation ({_id})")
    plt.tight_layout()
    plt.show()


# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
})


HIST_DIR = "histograms_1"
os.makedirs(HIST_DIR, exist_ok=True)

# QZV-relevant chemical parameter
QZV_PARAMS = [
    "Arsenic",
    "Boron",
    "Cadmium",
    "Chromium",
    "Copper",
    "Nickel",
    "Nitrate",
    "Nitrite",
    "Mercury",
    "Ammonium",
    "Chloride",
    "Sulfate",
    "Electrical.Conductivity",
    "Orthophosphate",
]

params_in_df = [p for p in QZV_PARAMS if p in df.columns]

for param in params_in_df:
    s = pd.to_numeric(df[param], errors="coerce").dropna()
    if s.empty:
        continue

    n = int(s.shape[0])
    bins = int(np.round(np.sqrt(n)))  # square-root rule (n=822 -> 29 bins)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(s, bins=bins, color="dodgerblue", edgecolor="black")

    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{param}")

    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black")

    plt.tight_layout()
    safe_param = param.replace("/", "_")
    fig.savefig(os.path.join(HIST_DIR, f"hist_{safe_param}.png"), dpi=300, facecolor="white")
    plt.show()
# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

Z_THR = 3.0
DPI = 300
THRESHOLDS_DIR = "thresholds"
os.makedirs(THRESHOLDS_DIR, exist_ok=True)

THRESHOLDS = {
    "Arsenic": {"limit": 9e-3, "trend": 7.5e-3},
    "Boron": {"limit": 0.9, "trend": 0.75},
    "Cadmium": {"limit": 4.5e-3, "trend": 3.75e-3},
    "Chromium": {"limit": 45e-3, "trend": 37.5e-3},
    "Copper": {"limit": 1.8, "trend": 1.5},
    "Nickel": {"limit": 18e-3, "trend": 15e-3},
    "Nitrate": {"limit": 45.0, "trend": 37.5},
    "Nitrite": {"limit": 0.09, "trend": 0.075},
    "Mercury": {"limit": 0.9, "trend": 0.75},
    "Ammonium": {"limit": 0.45, "trend": 0.375},
    "Chloride": {"limit": 180.0, "trend": 150.0},
    "Sulfate": {"limit": 225.0, "trend": 187.5},
    "Electrical.Conductivity": {"limit": 2250.0, "trend": 1875.0},
    "Orthophosphate": {"limit": 0.30, "trend": 0.225},
}

df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
for param, t in THRESHOLDS.items():
    if param not in df.columns:
        continue

    limit, trend = t["limit"], t["trend"]

    for gid, g in df.groupby("ID"):
        d = g["DATE"]
        v = pd.to_numeric(g[param], errors="coerce")
        m = d.notna() & v.notna()
        d, v = d[m], v[m]
        if v.empty:
            continue

        sd = v.std(ddof=0)
        out = (np.abs((v - v.mean()) / sd) > 3) if (sd and not np.isnan(sd)) else pd.Series(False, index=v.index)

        y0, y1 = min(v.min(), limit, trend), max(v.max(), limit, trend)
        pad = 0.05 * (y1 - y0 if y1 > y0 else 1)

        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.scatter(d, v, s=10, alpha=0.5, color="black", label="Data")
        if out.any():
            ax.scatter(d[out], v[out], s=25, color="orangered", label="Outliers (|z|>3)")

        ax.axhline(limit, color="darkorange", ls="--", lw=1.8, label="Threshold")
        ax.axhline(trend, color="dodgerblue", ls=":", lw=1.8, label="Trend reversal")

        ax.set_ylim(y0 - pad, y1 + pad)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.set_title(f"{param} ({gid})")
        ax.legend(facecolor="white", edgecolor="black", labelcolor="black")
        plt.tight_layout()
        safe_param = str(param).replace("/", "_")
        safe_gid = str(gid).replace("/", "_")
        out_path = os.path.join(THRESHOLDS_DIR, f"{safe_param}_{safe_gid}.png")
        plt.savefig(out_path, dpi=DPI, facecolor="white")
        plt.show()
