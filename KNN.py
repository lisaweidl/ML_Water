# %%
import pandas as pd
from math import sqrt
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
# %%
#water or merged
df = pd.read_csv("Water_FE.csv", sep=";")
#sep=";"
# %%
TARGET = "Orthophosphate" #Orthophosphate #ORTHOPHOSPHATE
X = df.drop(TARGET, axis=1)
y = df[TARGET]
# %%
y = pd.to_numeric(y, errors="coerce")
# %%
#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1,
)
# %%
y_train_mean = y_train.mean()
y_train = y_train.fillna(y_train_mean)

mask = y_test.notna()
X_test = X_test.loc[mask].copy()
y_test = y_test.loc[mask].copy()
# %%
#TRAIN
X_train.info()
# %%
X_train.head()
# %%
X_train.isnull().sum()
# %%
X_train = X_train.dropna(axis=1, how="all")
X_train[X_train.select_dtypes(include="number").columns] = X_train.select_dtypes(include="number").fillna(
    X_train.select_dtypes(include="number").mean())
# %%
X_train.isnull().sum()
# %%
X_train.ID.value_counts()
# %%
X_train = pd.get_dummies(X_train, columns=["ID"], prefix="ID", dtype=int)
X_train.head()
# %%
X_train["season"] = pd.to_datetime(X_train["DATE"], errors="coerce").dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"})
X_train = pd.get_dummies(X_train.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_train.head()
# %%
#TEST
X_test.head()
# %%
X_test.info()
# %%
X_test.isnull().sum()
# %%
X_test.ID.value_counts()
# %%
X_test = pd.get_dummies(X_test, columns=["ID"], prefix="ID", dtype=int)
X_test.head()
# %%
X_test["season"] = pd.to_datetime(X_test["DATE"], errors="coerce").dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"})
X_test = pd.get_dummies(X_test.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_test.head()
# %%
X_test.info()
# %%
X_test.head(20)
# %%
# align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
feature_names = X_train.columns
# %%
k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)
# %%
knn_base = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=3))
])

knn_base.fit(X_train, y_train)
y_pred_base = knn_base.predict(X_test)

print("\nKNN BASELINE (k=3) TEST")
print("R2:", r2_score(y_test, y_pred_base))
print("MAE:", mean_absolute_error(y_test, y_pred_base))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred_base)))
# %%
param_grid = {
    "knn__n_neighbors": list(range(1, 31)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],  # Manhattan vs Euclidean
}

grid = GridSearchCV(
    estimator=knn_base,
    param_grid=param_grid,
    cv=kf,
    scoring="r2",
    n_jobs=-1,
    refit=True
)

grid.fit(X_train, y_train)

print("\nKNN TUNED")
print("Best params:", grid.best_params_)
print("Best CV R2:", round(grid.best_score_, 4))

best_knn = grid.best_estimator_
# %%
y_pred = best_knn.predict(X_test)

print("\nKNN TUNED TEST")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))
# %%
cv_base = cross_validate(
    knn_base,
    X_train, y_train,
    cv=kf,
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"
    },
    return_train_score=False,
    n_jobs=-1
)

base_r2_cv = cv_base["test_r2"].mean()
base_mae_cv = (-cv_base["test_mae"]).mean()
base_rmse_cv = (-cv_base["test_rmse"]).mean()
# %%
# --- TUNED CV ---
cv_tuned = cross_validate(
    best_knn,
    X_train, y_train,
    cv=kf,
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"
    },
    return_train_score=False,
    n_jobs=-1
)

tuned_r2_cv = cv_tuned["test_r2"].mean()
tuned_mae_cv = (-cv_tuned["test_mae"]).mean()
tuned_rmse_cv = (-cv_tuned["test_rmse"]).mean()
# %%

# --- Summary table ---
df_cv = pd.DataFrame(
    [
        [1, "KNN baseline (CV mean)", base_r2_cv, base_mae_cv, base_rmse_cv],
        [2, "KNN tuned (CV mean)", tuned_r2_cv, tuned_mae_cv, tuned_rmse_cv],
    ],
    columns=["experiment nr.:", "experiment name", "R2", "MAE", "RMSE"]
)

print("\nKNN CV summary:")
print(df_cv)
# %%
result = permutation_importance(
    best_knn,
    X_test,
    y_test,
    n_repeats=50,
    random_state=42,
    scoring="r2"
)

importances = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean
}).sort_values("importance_mean", ascending=False)

print("\nTop 10 permutation importances:")
print(importances.head(10))
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")

top_n = 10
plot_df = importances.head(top_n).sort_values("importance_mean")

plt.figure(figsize=(8, 6))
plt.barh(plot_df["feature"], plot_df["importance_mean"], color="dodgerblue")
plt.title("KNN Permutation Importance_Water")
plt.xlabel("Mean decrease in R2")
plt.tight_layout()
plt.show()
# %%
knn_base.fit(X_train, y_train)

models = {
    "KNN Baseline": knn_base,
    "KNN Tuned (best k)": best_knn,
}

model_performance = []
exp = 1
for name, model in models.items():
    yhat = model.predict(X_test)
    model_performance.append([
        exp, name,
        r2_score(y_test, yhat),
        mean_absolute_error(y_test, yhat),
        sqrt(mean_squared_error(y_test, yhat))
    ])
    exp += 1

df_compare = pd.DataFrame(
    model_performance,
    columns=["experiment nr.:", "experiment name", "R2", "MAE", "RMSE"]
)
print("\nKNN comparison (test set):")
print(df_compare)
# %%
y_true = y_test.to_numpy()

pred_mean = np.full_like(y_true, fill_value=y_train.mean(), dtype=float)

pred_persist = np.empty_like(y_true, dtype=float)
pred_persist[0] = y_train.iloc[-1]
pred_persist[1:] = y_true[:-1]


def p_improves_mae_signflip(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)
    return (np.sum(pm >= obs) + 1) / (n_perm + 1), obs


p_mean, delta_mean = p_improves_mae_signflip(y_true, y_pred, pred_mean)
p_pers, delta_pers = p_improves_mae_signflip(y_true, y_pred, pred_persist)

print(f"\nΔMAE (Mean - KNN)    = {delta_mean:.4f}, p = {p_mean:.5f}")
print(f"ΔMAE (Persist - KNN) = {delta_pers:.4f}, p = {p_pers:.5f}")