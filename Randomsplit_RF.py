
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

#water or joined
df = pd.read_csv("Water_FE.csv", sep=";")
#sep=";"

TARGET = "Orthophosphate"  #Orthophosphate #ORTHOPHOSPHATE
X = df.drop(TARGET, axis=1)
y = df[TARGET]

y = y.fillna(y.mean())
y

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1,
)

#TRAIN
X_train = X_train.dropna(axis=1, how="all")
X_train[X_train.select_dtypes(include="number").columns] = X_train.select_dtypes(include="number").fillna(
    X_train.select_dtypes(include="number").mean())

X_train = pd.get_dummies(X_train, columns=["ID"], prefix="ID", dtype=int)

X_train["season"] = pd.to_datetime(X_train["DATE"], errors="coerce").dt.month.map(
    {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer",
     8: "summer", 9: "fall", 10: "fall", 11: "fall"})

X_train = pd.get_dummies(X_train.drop(columns=["DATE"]), columns=["season"], dtype=int)


#TEST
X_test = X_test.dropna(axis=1, how="all")
X_test[X_test.select_dtypes(include="number").columns] = X_test.select_dtypes(include="number").fillna(
    X_test.select_dtypes(include="number").mean())

X_test = pd.get_dummies(X_test, columns=["ID"], prefix="ID", dtype=int)

X_test["season"] = pd.to_datetime(X_test["DATE"], errors="coerce").dt.month.map(
    {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer",
     8: "summer", 9: "fall", 10: "fall", 11: "fall"})

X_test = pd.get_dummies(X_test.drop(columns=["DATE"]), columns=["season"], dtype=int)

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

num_cols = X_train.select_dtypes(include="number").columns
train_means = X_train[num_cols].mean()
X_train[num_cols] = X_train[num_cols].fillna(train_means)
X_test[num_cols] = X_test[num_cols].fillna(train_means)

# remaining categorical columns encoded
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
if len(cat_cols) > 0:
    X_train = pd.get_dummies(X_train, columns=cat_cols, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cat_cols, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

rf_baseline = RandomForestRegressor(random_state=42, n_estimators=1000, max_depth=5, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
y_pred = rf_baseline.predict(X_test)

print("\nBASELINE TEST")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))


from sklearn.model_selection import KFold, cross_validate

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)

cv_base = cross_validate(
    rf_baseline,
    X_train, y_train,
    cv=kf,
    scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
    return_train_score=False
)

base_r2_cv = cv_base["test_r2"].mean()
base_mae_cv = (-cv_base["test_mae"]).mean()
base_rmse_cv = (-cv_base["test_rmse"]).mean()

print(f"\nBASELINE CV on TRAIN ({k}-fold mean)")
print("R2:", round(base_r2_cv, 4))
print("MAE:", round(base_mae_cv, 4))
print("RMSE:", round(base_rmse_cv, 4))


#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "bootstrap": [True, False],
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [8, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2', 0.5, None],
}

rf_grid = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search_all = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid,
    cv=kf,
    n_jobs=-1,
    verbose=2,
    scoring="r2",
    refit=True
)

grid_search_all.fit(X_train, y_train)

print("\nTUNED")
print("Best hyperparameters:", grid_search_all.best_params_)
print("Best CV R2 (GridSearch):", grid_search_all.best_score_)
best_rf = grid_search_all.best_estimator_

# tuned TEST
y_pred = best_rf.predict(X_test)
print("\nTUNED TEST")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))


# tuned CV (R2/MAE/RMSE)
cv_tuned = cross_validate(
    best_rf,
    X_train, y_train,
    cv=kf,
    scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
    return_train_score=False
)

tuned_r2_cv = cv_tuned["test_r2"].mean()
tuned_mae_cv = (-cv_tuned["test_mae"]).mean()
tuned_rmse_cv = (-cv_tuned["test_rmse"]).mean()

print(f"\nTUNED CV on TRAIN ({k}-fold mean)")
print("R2:", round(tuned_r2_cv, 4))
print("MAE:", round(tuned_mae_cv, 4))
print("RMSE:", round(tuned_rmse_cv, 4))


#boruta
from boruta import BorutaPy

best_params = grid_search_all.best_params_

forest = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    **best_params,
)

boruta = BorutaPy(
    estimator=forest,
    n_estimators="auto",
    verbose=2,
    random_state=42,
)

boruta.fit(X_train.to_numpy(), y_train.to_numpy())

print("Selected Features:", boruta.support_)
print("Ranking:", boruta.ranking_)
print("Nr of significant features:", boruta.n_features_)

selected_rf_features = (
    pd.DataFrame({"Feature": X_train.columns, "Ranking": boruta.ranking_})
    .sort_values(by="Ranking")
    .reset_index(drop=True)
)
print(selected_rf_features)


X_important_train = boruta.transform(X_train.to_numpy())
X_important_test = boruta.transform(X_test.to_numpy())
rf_important = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    **best_params
)

rf_important.fit(X_important_train, y_train)

print("\nBORUTA TEST")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))


cv_boruta = cross_validate(
    rf_important,
    X_important_train, y_train,
    cv=kf,
    scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
    return_train_score=False
)

boruta_r2_cv = cv_boruta["test_r2"].mean()
boruta_mae_cv = (-cv_boruta["test_mae"]).mean()
boruta_rmse_cv = (-cv_boruta["test_rmse"]).mean()

print(f"\nBORUTA CV on TRAIN ({k}-fold mean)")
print("R2:", round(boruta_r2_cv, 4))
print("MAE:", round(boruta_mae_cv, 4))
print("RMSE:", round(boruta_rmse_cv, 4))


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")

accepted = X_train.columns[boruta.support_]

imp = pd.Series(
    rf_important.feature_importances_,
    index=accepted
).sort_values()

plt.figure(figsize=(8, 6))
imp.plot(kind="barh", color="dodgerblue")

plt.title("RandomSplit_Water_BorutaFeatures")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()


# comparison
models = [
    ("RF baseline", rf_baseline, X_test),
    ("RF tuned (all features)", best_rf, X_test),
    ("RF tuned + Boruta features", rf_important, X_important_test),
]

model_performance = []
for exp, (name, model, X_te) in enumerate(models, start=1):
    yhat = model.predict(X_te)
    model_performance.append([
        exp, name,
        r2_score(y_test, yhat),
        mean_absolute_error(y_test, yhat),
        sqrt(mean_squared_error(y_test, yhat))
    ])

df_perf = pd.DataFrame(model_performance, columns=["experiment nr.:", "experiment name", "R2", "MAE", "RMSE"])
print(df_perf)


df_cv = pd.DataFrame(
    [
        [1, "RF baseline (CV mean)", base_r2_cv, base_mae_cv, base_rmse_cv],
        [2, "RF tuned (CV mean)", tuned_r2_cv, tuned_mae_cv, tuned_rmse_cv],
        [3, "RF boruta (CV mean)", boruta_r2_cv, boruta_mae_cv, boruta_rmse_cv],
    ],
    columns=["experiment nr.:", "experiment name", "R2", "MAE", "RMSE"]
)

print(df_cv)


import numpy as np

y_true = y_test.to_numpy()

pred_mean = np.full_like(y_true, fill_value=y_train.mean(), dtype=float)

pred_persist = np.empty_like(y_true, dtype=float)
pred_persist[0] = y_train.iloc[-1]
pred_persist[1:] = y_true[:-1]

def p_improves_mae_signflip(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)  # >0 RF better
    obs = d.mean()

    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)

    return (np.sum(pm >= obs) + 1) / (n_perm + 1), obs

y_pred_rf = rf_important.predict(X_important_test)

p_mean, delta_mean = p_improves_mae_signflip(y_true, y_pred_rf, pred_mean)
p_pers, delta_pers = p_improves_mae_signflip(y_true, y_pred_rf, pred_persist)

print(f"ΔMAE (Mean - RF)  = {delta_mean:.4f}, p = {p_mean:.5f}")
print(f"ΔMAE (Persist - RF)= {delta_pers:.4f}, p = {p_pers:.5f}")
