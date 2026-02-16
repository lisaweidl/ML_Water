import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from boruta import BorutaPy
from math import sqrt
import numpy as np

# water or merged
df = pd.read_csv("Water_FE.csv", sep=";")
# sep=";"

df.describe()

TARGET = "Orthophosphate"  # Orthophosphate #ORTHOPHOSPHATE
X = df.drop(TARGET, axis=1)
y = df[TARGET]

y = y.fillna(y.mean())
y

ID_COL = "ID"
DATE_COL = "DATE"
TARGET = "Orthophosphate" # Orthophosphate #ORTHOPHOSPHATE
TEST_FRAC_DATES = 0.20

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[ID_COL, DATE_COL, TARGET]).copy()
df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

unique_dates = pd.Series(df[DATE_COL].unique()).dropna().sort_values().to_numpy()

# at least 1 date in test
n_dates = len(unique_dates)
n_test_dates = max(1, int(round(n_dates * TEST_FRAC_DATES)))
cutoff_date = unique_dates[-n_test_dates]  # first date in test period

train_df = df[df[DATE_COL] < cutoff_date].copy()
test_df = df[df[DATE_COL] >= cutoff_date].copy()

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("Cutoff date:", cutoff_date)
print("Train date range:", train_df[DATE_COL].min(), "→", train_df[DATE_COL].max())
print("Test date range:", test_df[DATE_COL].min(), "→", test_df[DATE_COL].max())

X_train.tail(4)
X_test.head(4)

# TRAIN
X_train.head()
X_train.info()
X_train.isnull().sum()

# drop all-NaN columns from train/test together
all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
if all_nan_cols:
    X_train = X_train.drop(columns=all_nan_cols)
    X_test = X_test.drop(columns=all_nan_cols, errors="ignore")

# fill numeric NaNs later using train means (after alignment)
X_train.isnull().sum()
X_train.head()
X_train.ID.value_counts()

X_train = pd.get_dummies(X_train, columns=["ID"], prefix="ID", dtype=int)
X_train.head()

X_train["season"] = pd.to_datetime(X_train["DATE"], errors="coerce").dt.month.map(
    {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer",
     8: "summer", 9: "fall", 10: "fall", 11: "fall"})
X_train = pd.get_dummies(X_train.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_train.head()

X_train = X_train.copy()

# TEST
X_test.head()
X_test.info()
X_test.isnull().sum()

X_test = pd.get_dummies(X_test, columns=["ID"], prefix="ID", dtype=int)
X_test.head()

X_test["season"] = pd.to_datetime(X_test["DATE"], errors="coerce").dt.month.map(
    {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer",
     8: "summer", 9: "fall", 10: "fall", 11: "fall"})
X_test = pd.get_dummies(X_test.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_test.head()

X_test = X_test.copy()

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

num_cols = X_train.select_dtypes(include="number").columns
train_means = X_train[num_cols].mean()
X_train[num_cols] = X_train[num_cols].fillna(train_means)
X_test[num_cols] = X_test[num_cols].fillna(train_means)

# ensure any remaining categorical columns are encoded
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
if len(cat_cols) > 0:
    X_train = pd.get_dummies(X_train, columns=cat_cols, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cat_cols, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

#rf
rf = RandomForestRegressor(random_state=1, n_estimators=1000, max_depth=5)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

# hyperparameter tuning
param_grid = {
    "bootstrap": [True, False],
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [5, 8, 12, 15, 20, 25, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, None],
}

rf_grid = RandomForestRegressor(random_state=42, n_jobs=-1)

# chronological order for CV
train_order = train_df.sort_values(DATE_COL).index
X_train = X_train.loc[train_order]
y_train = y_train.loc[train_order]

tscv = TimeSeriesSplit(n_splits=3)

grid_search_all = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid,
    cv=tscv,
    n_jobs=-1,
    verbose=2,
    scoring="r2",
    refit=True,
)
grid_search_all.fit(X_train, y_train)

print("Best hyperparameters:", grid_search_all.best_params_)
print("Best CV R2:", grid_search_all.best_score_)

y_pred = grid_search_all.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

# boruta
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
    **best_params,
)

rf_important.fit(X_important_train, y_train)

y_pred = rf_important.predict(X_important_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

import matplotlib.pyplot as plt

accepted = X_train.columns[boruta.support_]

imp = pd.Series(rf_important.feature_importances_, index=accepted).sort_values()
imp.plot(kind="barh")
plt.title("Boruta RF importance")
plt.tight_layout()
plt.show()

# comparison
models = {
    "RF baseline": rf,
    "RF tuned (all features)": grid_search_all,
    "RF tuned + Boruta features": rf_important,
}
test_data = [X_test, X_test, X_important_test]

model_performance = []
exp = 1

for key, value in models.items():
    yhat = value.predict(test_data[exp - 1])
    model_performance.append([exp, key,
                              r2_score(y_test, yhat),
                              mean_absolute_error(y_test, yhat),
                              sqrt(mean_squared_error(y_test, yhat))])
    exp += 1

df = pd.DataFrame(model_performance, columns=["experiment nr.:", "experiment name", "R2", "MAE", "RMSE"])
print(df)

def p_improves(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)  # >0 => model better
    obs = d.mean()
    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)
    return (np.sum(pm >= obs) + 1) / (n_perm + 1)

# predictions from final model in the Boruta feature space
y_pred_final = rf_important.predict(X_important_test)

# baselines (constants from TRAIN)
mean_base = np.full(len(y_test), y_train.mean())
last_base = np.full(len(y_test), y_train.iloc[-1])

print("p (FINAL Boruta+tuned vs MEAN):",
      round(p_improves(y_test.to_numpy(), y_pred_final, mean_base), 5))
print("p (FINAL Boruta+tuned vs LAST):",
      round(p_improves(y_test.to_numpy(), y_pred_final, last_base), 5))
