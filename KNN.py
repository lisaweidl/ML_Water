import pandas as pd
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#water or merged
df = pd.read_excel("Water_FE.xlsx")

TARGET = "ORTHOPHOSPHAT mg/l"
X = df.drop(TARGET, axis=1)
y = df[TARGET]

if "ID" in X.columns:
    X["ID"] = X["ID"].astype("category")
    X = pd.get_dummies(X, columns=["ID"], dummy_na=True)

DATE_COL = "Date"
X[DATE_COL] = pd.to_datetime(X[DATE_COL], errors="coerce")

doy = X[DATE_COL].dt.dayofyear
X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
X = X.drop(columns=[DATE_COL])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


k_values = list(range(1, 31))
scores = []

for k in k_values:
    knn_k = KNeighborsRegressor(n_neighbors=k)
    score = cross_val_score(knn_k, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    scores.append(score.mean())

best_k = k_values[int(np.argmax(scores))]
print("\nBest k:", best_k, "Best CV R2:", max(scores))

knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

pred_best = knn_best.predict(X_test)

r2  = r2_score(y_test, pred_best)
mae = mean_absolute_error(y_test, pred_best)
mse = mean_squared_error(y_test, pred_best)
rmse = sqrt(mse)

print("\n=== Tuned KNN ===")
print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


#pvalue
import numpy as np

def p_improves(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)  # >0 => model better
    obs = d.mean()
    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)
    return (np.sum(pm >= obs) + 1) / (n_perm + 1)

yte = y_test.to_numpy()
mean_pred = np.full(len(y_test), float(y_train.mean()))
last_pred = np.full(len(y_test), float(y_train.iloc[-1]))

print("\nP-values (Tuned KNN improves):")
print("KNN+Tuned vs MEAN:", round(p_improves(yte, pred_best, mean_pred), 5))
print("KNN+Tuned vs LAST:", round(p_improves(yte, pred_best, last_pred), 5))

print("\ny_train mean:", float(y_train.mean()))
print("y_train last:", float(y_train.iloc[-1]))


