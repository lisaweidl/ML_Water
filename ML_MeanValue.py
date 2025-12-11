
import pandas as pd
import numpy as np
from numpy.random import default_rng
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET = "ORTHOPHOSPHAT mg/l"
ID_COL = "ID"

train = pd.read_excel("df_Water_train.xlsx")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train[ID_COL] = train[ID_COL].astype(str)

mean_per_id = train.groupby(ID_COL, dropna=False)[TARGET].mean()
train["pred_mean_per_ID"] = train[ID_COL].map(mean_per_id)

# Remove rows where target or prediction is missing
mask = train[TARGET].notna() & train["pred_mean_per_ID"].notna()
y_true = train.loc[mask, TARGET].values
y_pred = train.loc[mask, "pred_mean_per_ID"].values


r2  = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))

print("\n=== Mean Value Baseline Results ===")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# Permutation test
rng = default_rng(42)
n_perm = 200

perm_r2 = np.empty(n_perm)
for i in range(n_perm):
    y_pred_perm = rng.permutation(y_pred)
    perm_r2[i] = r2_score(y_true, y_pred_perm)

# one-sided p-value: P(R²_perm >= R²_observed)
p_value = (1.0 + np.sum(perm_r2 >= r2)) / (n_perm + 1.0)

print(f"\n=== Permutation Test (R² > 0) ===")
print(f"P-value: {p_value:.5f}")
