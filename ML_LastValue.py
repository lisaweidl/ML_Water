import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from numpy.random import default_rng

df = pd.read_excel("df_Water_train.xlsx")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ID", "Date"])

col = "ORTHOPHOSPHAT mg/l"


group = df.groupby("ID")[col]
df["y_true"] = group.shift(1)       # previous value
df["y_pred"] = df[col]             # current value

# drop rows where there is no previous value
mask = df["y_true"].notna()
y_true = df.loc[mask, "y_true"].values
y_pred = df.loc[mask, "y_pred"].values

r2  = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = sqrt(mse)

print("R²:", r2)
print("MSE:", mse)
print("RMSE:", rmse)

# permutation test
rng = default_rng(42)
n_perm = 200

perm_r2 = np.empty(n_perm)
for i in range(n_perm):
    y_pred_perm = rng.permutation(y_pred)
    perm_r2[i] = r2_score(y_true, y_pred_perm)

# one-sided p-value: P(R²_perm >= R²_observed)
p_value = (1.0 + np.sum(perm_r2 >= r2)) / (n_perm + 1.0)

print(f"P-value (permutation test for R² > 0): {p_value:.5f}")
