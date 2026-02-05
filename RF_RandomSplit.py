
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

#water or merged
df = pd.read_csv("Water_FE.csv", sep=";")

df.describe()

TARGET = "Orthophosphate" #Orthophosphate #ORTHOPHOSPHATE
X = df.drop(TARGET, axis=1)
y = df[TARGET]

y = y.fillna(y.mean())
y

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1,
)

#TRAIN
X_train.info()
X_train.head()
X_train.isnull().sum()

X_train = X_train.dropna(axis=1, how="all")
X_train[X_train.select_dtypes(include="number").columns] = X_train.select_dtypes(include="number").fillna(X_train.select_dtypes(include="number").mean())

X_train.isnull().sum()

X_train = pd.get_dummies(X_train, columns=["ID"], prefix="ID", dtype=int)
X_train.head()

X_train["season"] = pd.to_datetime(X_train["DATE"], errors="coerce").dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"})
X_train = pd.get_dummies(X_train.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_train.head()

X_train = X_train.copy()

#TEST
X_test.head()
X_test.info()
X_test.isnull().sum()

X_test = X_test.dropna(axis=1, how="all")
X_test[X_test.select_dtypes(include="number").columns] = X_test.select_dtypes(include="number").fillna(
    X_test.select_dtypes(include="number").mean())

X_test.isnull().sum()

X_test = pd.get_dummies(X_test, columns=["ID"], prefix="ID", dtype=int)
X_test.head()

X_test["season"] = pd.to_datetime(X_test["DATE"], errors="coerce").dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"})
X_test = pd.get_dummies(X_test.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_test.head()

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test=X_test.copy()

#RF
rf = RandomForestRegressor(random_state=1, n_estimators=1000, max_depth=5)
rf.fit(X_train.copy(), y_train)

y_pred = rf.predict(X_test.copy())

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

#joined
#R2: 0.7127994288951212
#MAE: 0.12439204616932721
#RMSE: 0.34167348133578834

#water
#R2: 0.7498347446242883
#MAE: 0.14167802319231074
#RMSE: 0.2698735168345072


#hyperparameter tuning
from sklearn.model_selection import GridSearchCV

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
    cv=3,              
    n_jobs=-1,
    verbose=2,
    scoring="r2",
    refit=True
)

grid_search_all.fit(X_train, y_train)

print("Best hyperparameters:", grid_search_all.best_params_)
print("Best CV R2:", grid_search_all.best_score_)

y_pred = grid_search_all.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

#joined
#Best hyperparameters: {"bootstrap": True, "max_depth": 8, "max_features": None,"min_samples_leaf": 4, "min_samples_split": 2,"n_estimators": 300}
#R2: 0.7124028273451168
#MAE: 0.12202044066835117
#RMSE: 0.34190931213945736

#water
#Best hyperparameters: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
#R2: 0.7887404241628766
#MAE: 0.1287677476469937
#RMSE: 0.24800188076967786

#boruta
from boruta import BorutaPy

best_params = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}

forest = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    **best_params,
)

boruta = BorutaPy(
    estimator=forest,
    n_estimators='auto',
    verbose=2,
    random_state=42
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
X_important_test  = boruta.transform(X_test.to_numpy())

rf_important = RandomForestRegressor(
    n_jobs=-1,
    random_state=42,
    **best_params
)

rf_important.fit(X_important_train, y_train)

y_pred = rf_important.predict(X_important_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))

#joined
#R2: 0.7313820004336136
#MAE: 0.12340010055261691
#RMSE: 0.33043510381536256

#water
#R2: 0.796640855924915
#MAE: 0.12283737808289959
#RMSE: 0.2433204580218488


import matplotlib.pyplot as plt

accepted = X_train.columns[boruta.support_]

imp = pd.Series(rf_important.feature_importances_, index=accepted).sort_values()
imp.plot(kind="barh")
plt.title("Boruta RF importance")
plt.tight_layout()
plt.show()

#comparison
models = {
    "RF baseline": rf,
    "RF tuned (all features)": grid_search_all,
    "RF tuned + Boruta features": rf_important
}

test_data = [X_test, X_test, X_important_test]

model_performance = []
exp = 1

for key, value in models.items():
    yhat = value.predict(test_data[exp-1])
    model_performance.append([exp, key,
                              r2_score(y_test, yhat),
                              mean_absolute_error(y_test, yhat),
                              sqrt(mean_squared_error(y_test, yhat))])
    exp += 1


df = pd.DataFrame(model_performance, columns=["experiment nr.:", "experiment name", "R2", "MAE", "RMSE"])
print(df)

import numpy as np

def p_improves(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)  # >0 => model better
    obs = d.mean()
    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)
    return (np.sum(pm >= obs) + 1) / (n_perm + 1)

# predictions from your FINAL model in the Boruta feature space
y_pred_final = rf_important.predict(X_important_test)

# baselines (constants from TRAIN)
mean_base = np.full(len(y_test), y_train.mean())
last_base = np.full(len(y_test), y_train.iloc[-1])

print("p (FINAL Boruta+tuned vs MEAN):",
      round(p_improves(y_test.to_numpy(), y_pred_final, mean_base), 5))
print("p (FINAL Boruta+tuned vs LAST):",
      round(p_improves(y_test.to_numpy(), y_pred_final, last_base), 5))
