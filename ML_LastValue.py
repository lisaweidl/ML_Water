import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from numpy.random import default_rng

df = pd.read_excel("df_train.xlsx")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ID", "Date"])

penult = df.groupby("ID")["ORTHOPHOSPHAT mg/l"].nth(-2)
last   = df.groupby("ID")["ORTHOPHOSPHAT mg/l"].nth(-1)

r2  = r2_score(last, penult)
mse = mean_squared_error(last, penult)
rmse = sqrt(mse)

print("Last Value Results:")
print("R²:", r2)
print("MSE:", mse)
print("RMSE:", rmse)


y = last.values
x = penult.values

rng = default_rng(42)
n_perm = 200

perm_r2 = np.empty(n_perm)
for i in range(n_perm):
    x_perm = rng.permutation(x)
    perm_r2[i] = r2_score(y, x_perm)

# one-sided p-value: P(R²_perm >= R²_observed)
p_value = (1.0 + np.sum(perm_r2 >= r2)) / (n_perm + 1.0)

print(f"P-value (permutation test for R² > 0): {p_value:.5f}")
