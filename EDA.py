import pandas as pd

df = pd.read_csv("Water_Cleaned.csv", sep=";", encoding="utf-8")

import seaborn as sns
import matplotlib.pyplot as plt

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



import seaborn as sns
import matplotlib.pyplot as plt

num = df.select_dtypes("number")
cols = num.columns

fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))

for ax, c in zip(axes, cols):
    sns.boxplot(y=num[c], ax=ax)
    ax.set_title(c)
    ax.set_ylabel("value")

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

num = df.select_dtypes("number")

for col in num.columns:
    s = pd.to_numeric(num[col], errors="coerce").dropna()
    if s.empty:
        continue

    plt.figure()
    plt.hist(s, bins=29)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

Z_THR = 3.0
DPI = 300

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

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(d, v, s=10, alpha=0.5, color="0.5", label="Data")
        if out.any():
            ax.scatter(d[out], v[out], s=25, color="red", label="Outliers (|z|>3)")

        ax.axhline(limit, color="orange", ls="--", lw=1.8, label="Threshold")
        ax.axhline(trend, color="green", ls=":", lw=1.8, label="Trend reversal")

        ax.set_ylim(y0 - pad, y1 + pad)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.set_title(f"{param} ({gid})")
        ax.legend()
        plt.tight_layout()
        plt.show()
