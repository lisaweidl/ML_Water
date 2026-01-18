import pandas as pd
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_excel("Merged_FE.xlsx")

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


from sklearn.model_selection import permutation_test_score

# Permutation test on train
score_cv, perm_scores, p_value = permutation_test_score(
    estimator=knn_best, X=X_train, y=y_train,
    scoring="r2",
    n_permutations=200,
    random_state=42,
    n_jobs=-1,
    cv=5,
)

print(f"Permutation-test p-value (train/CV): {p_value:.5f}")
