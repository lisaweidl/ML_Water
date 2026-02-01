import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np

#water or merged
df = pd.read_csv("df_joined.csv")
df.describe()

TARGET = "ORTHOPHOSPHATE" #ORTHOPHOSPHAT mg/l
X = df.drop(TARGET, axis=1)
y = df[TARGET]

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1,
)

#TRAIN
X_train = X_train.dropna(axis=1, how="all")
X_train[X_train.select_dtypes(include="number").columns] = X_train.select_dtypes(include="number").fillna(X_train.select_dtypes(include="number").mean())

X_train = pd.get_dummies(X_train, columns=["ID"], prefix="ID", dtype=int)
X_train.head()

X_train["season"] = pd.to_datetime(X_train["DATE"], errors="coerce").dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"})
X_train = pd.get_dummies(X_train.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_train.head()

X_train = X_train.copy()

#TEST
X_test = X_test.dropna(axis=1, how="all")
X_test[X_test.select_dtypes(include="number").columns] = X_test.select_dtypes(include="number").fillna(
    X_test.select_dtypes(include="number").mean())

X_test = pd.get_dummies(X_test, columns=["ID"], prefix="ID", dtype=int)
X_test.head()

X_test["season"] = pd.to_datetime(X_test["DATE"], errors="coerce").dt.month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",11:"fall"})
X_test = pd.get_dummies(X_test.drop(columns=["DATE"]), columns=["season"], dtype=int)
X_test.head()

X_test=X_test.copy()

#RF
rf = RandomForestRegressor(random_state=1, n_estimators=1000, max_depth=5)
rf.fit(X_train.copy(), y_train)

y_pred = rf.predict(X_test.copy())

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))


# feature selection
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_jobs=-1, n_estimators=1000, max_depth=5)
boruta = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
boruta.fit(np.array(X_train.copy()), np.array(y_train))

print("Selected Features:", boruta.support_)
print("Ranking:", boruta.ranking_)
print("Nr of significant features:", boruta.n_features_)

X_important_train = boruta.transform(np.array(X_train.copy()))
X_important_test = boruta.transform(np.array(X_test.copy()))

rf_important = RandomForestRegressor(n_jobs=-1, n_estimators=500, max_depth=12)
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

#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

param_grid = {
    "bootstrap": [True, False],
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 5, 8, 12, 15],
}

rf_grid = RandomForestRegressor(random_state=1)
grid_search = GridSearchCV(estimator=rf_grid, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_important_train, y_train)

grid_search.best_params_
print("Best hyperparameters:", grid_search.best_params_)

y_pred = grid_search.predict(X_important_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))


#comparison
models = {"RF": rf, "RF+Boruta": rf_important, "RF+Boruta+Hyperparameter Tuning": grid_search}
test_data = [X_test.copy(), X_important_test, X_important_test]

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

#pvalue
import numpy as np

y_pred_boruta = grid_search.predict(X_important_test)

def p_improves(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)  # >0 => model better
    obs = d.mean()
    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)
    return (np.sum(pm >= obs) + 1) / (n_perm + 1)

mean_pred = np.full(len(y_test), y_train.mean())
last_pred = np.full(len(y_test), y_train.iloc[-1])

y_pred_final = grid_search.predict(X_important_test)
print("p (FINAL tuned vs MEAN):",
      round(p_improves(y_test.to_numpy(), y_pred_final, np.full(len(y_test), y_train.mean())), 5))
print("p (FINAL tuned vs LAST):",
      round(p_improves(y_test.to_numpy(), y_pred_final, np.full(len(y_test), y_train.iloc[-1])), 5))