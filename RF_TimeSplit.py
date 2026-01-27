import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

#water or merged
df = pd.read_csv("df_joined.csv")

TARGET = "ORTHOPHOSPHATE" #ORTHOPHOSPHAT mg/l
DATE_COL = "DATE" #Date
ID_COL = "ID"
TEST_SIZE = 0.2


df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[ID_COL, DATE_COL, TARGET]).copy()

df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
df["t_idx"] = df.groupby(ID_COL).cumcount()

def chronological_split_by_id(df_in, id_col, test_size=0.2, min_test=1):
    train_parts, test_parts = [], []
    for _, g in df_in.groupby(id_col, sort=False):
        n = len(g)
        if n < 2:
            train_parts.append(g)
            continue
        n_test = max(min_test, int(round(n * test_size)))
        n_test = min(n_test, n - 1)  # mind. 1 Train
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])
    train_df = pd.concat(train_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0) if test_parts else df_in.iloc[0:0]
    return train_df, test_df

train_df, test_df = chronological_split_by_id(df, ID_COL, TEST_SIZE)

drop_cols = [TARGET, DATE_COL]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[TARGET]

X_train[ID_COL] = X_train[ID_COL].astype("category")
X_test[ID_COL] = X_test[ID_COL].astype("category")

X_train = pd.get_dummies(X_train, columns=[ID_COL], dummy_na=True)
X_test = pd.get_dummies(X_test, columns=[ID_COL], dummy_na=True)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
X_test = X_test[X_train.columns]


print("Train rows:", len(X_train), "Test rows:", len(X_test))
print("Train per ID:\n", train_df.groupby(ID_COL).size())
print("Test per ID:\n", test_df.groupby(ID_COL).size())


# when I load the df joined, water temperature rolling windows are all NaN
import numpy as np

threshold = 0.20
keep_cols = X_train.columns[X_train.isna().mean() <= threshold]
X_train = X_train[keep_cols].copy()
X_test  = X_test.reindex(columns=keep_cols)

num_cols = X_train.select_dtypes(include=[np.number]).columns
train_means = X_train[num_cols].mean()

X_train[num_cols] = X_train[num_cols].fillna(train_means)
X_test[num_cols]  = X_test[num_cols].fillna(train_means)


#RF
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))


from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True, False],
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True
)

grid.fit(X_train, y_train)
best_params = grid.best_params_
best_rf = grid.best_estimator_

print("\nBest CV R2:", grid.best_score_)
print("Best hyperparameters:", grid.best_params_)

y_pred_best = best_rf.predict(X_test)
print("\nTUNED R2:", r2_score(y_test, y_pred_best))
print("TUNED MAE:", mean_absolute_error(y_test, y_pred_best))
print("TUNED RMSE:", sqrt(mean_squared_error(y_test, y_pred_best)))



from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

rf_for_boruta = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    **best_params
)

boruta_selector = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators="auto",
    verbose=2,
    random_state=42
)

boruta_selector.fit(X_train.values, y_train.values)

selected_features = X_train.columns[boruta_selector.support_].tolist()
print(f"#Features used: {len(selected_features)}")
print(f"Selected features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

best_rf.fit(X_train_sel, y_train)

y_pred_boruta = best_rf.predict(X_test_sel)
print("Boruta+Tuned R2:", r2_score(y_test, y_pred_boruta))
print("Boruta+Tuned MAE:", mean_absolute_error(y_test, y_pred_boruta))
print("Boruta+Tuned RMSE:", sqrt(mean_squared_error(y_test, y_pred_boruta)))


print("Original features:", X_train.shape[1])
print("Selected features:", len(selected_features))


#permutation importance
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

y0 = best_rf.predict(X_test_sel)
r2_base  = r2_score(y_test, y0)
mse_base = mean_squared_error(y_test, y0)
rmse_base = np.sqrt(mse_base)


rng = np.random.default_rng(42)
n_repeats = 30

rows = []
Xtmp = X_test_sel.copy()

for col in X_test_sel.columns:
    dR2, dMSE, dRMSE = [], [], []
    orig = Xtmp[col].to_numpy(copy=True)

    for _ in range(n_repeats):
        perm = orig.copy()
        rng.shuffle(perm)
        Xtmp[col] = perm

        yp = best_rf.predict(Xtmp)
        r2_p  = r2_score(y_test, yp)
        mse_p = mean_squared_error(y_test, yp)
        rmse_p = np.sqrt(mse_p)

        dR2.append(r2_base - r2_p)
        dMSE.append(mse_p - mse_base)
        dRMSE.append(rmse_p - rmse_base)

    Xtmp[col] = orig

    dR2 = np.array(dR2)
    rows.append({
        "feature": col,
        "mean_ΔR2": dR2.mean(),
        "std_ΔR2": dR2.std(ddof=1),
        "mean_ΔRMSE": np.mean(dRMSE),
        "robustness_P(perm>=base)": np.mean(dR2 <= 0)
    })

imp = pd.DataFrame(rows).sort_values("mean_ΔR2", ascending=False)
top10 = imp.head(10)
top10

import matplotlib.pyplot as plt

plot_df = top10.set_index("feature")
ax = plot_df["mean_ΔR2"].plot(kind="bar", yerr=plot_df["std_ΔR2"], capsize=3)
ax.set_ylabel("Mean ΔR²")
plt.tight_layout()
plt.show()



#pvalue
import numpy as np

def p_improves(y_true, pred_model, pred_base, n_perm=5000, seed=42):
    d = np.abs(y_true - pred_base) - np.abs(y_true - pred_model)   # >0 => model better
    obs = d.mean()
    rng = np.random.default_rng(seed)
    pm = (rng.choice([-1, 1], size=(n_perm, d.size)) * d).mean(axis=1)
    return (np.sum(pm >= obs) + 1) / (n_perm + 1)

# baselines (constants from TRAIN)
mean_pred = np.full(len(y_test), y_train.mean())
last_pred = np.full(len(y_test), y_train.iloc[-1])

print("p (Boruta+Tuned vs MEAN):", round(p_improves(y_test.to_numpy(), y_pred_boruta, mean_pred), 5))
print("p (Boruta+Tuned vs LAST):", round(p_improves(y_test.to_numpy(), y_pred_boruta, last_pred), 5))

print("y_train mean:", y_train.mean())
print("y_train last:", y_train.iloc[-1])
