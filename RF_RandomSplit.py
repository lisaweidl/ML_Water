import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np

#water or joined
df = pd.read_csv("df_joined.csv")

TARGET = "ORTHOPHOSPHATE"  #ORTHOPHOSPHAT mg/l
X = df.drop(TARGET, axis=1)
y = df[TARGET]

if "ID" in X.columns:
    X["ID"] = X["ID"].astype("category")
    X = pd.get_dummies(X, columns=["ID"], dummy_na=True)

DATE_COL = "DATE" #Date
X[DATE_COL] = pd.to_datetime(X[DATE_COL], errors="coerce")

doy = X[DATE_COL].dt.dayofyear
X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
X = X.drop(columns=[DATE_COL])

# drop columns with >20% missing
# when I load the df joined, water temperature rolling windows are all NaN
threshold = 0.20
keep_cols = X.columns[X.isna().mean() <= threshold]
X = X[keep_cols].copy()

# impute remaining with mean
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

#split random
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_test, y_pred)))



from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint
from math import sqrt

# tuning
param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 15),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
}

rand_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42
)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
best_params = rand_search.best_params_
print("Best hyperparameters:", best_params)


y_pred_best = rand_search.best_estimator_.predict(X_test)
print("Tuned R2:", r2_score(y_test, y_pred_best))
print("Tuned MAE:", mean_absolute_error(y_test, y_pred_best))
print("Tuned RMSE:", sqrt(mean_squared_error(y_test, y_pred_best)))



from boruta import BorutaPy

# Boruta to decrease noise
rf_for_boruta = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    **best_params
)

boruta_selector = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators=best_params["n_estimators"],
    verbose=2,
    random_state=42
)

boruta_selector.fit(X_train.values, y_train.values)

selected_features = X_train.columns[boruta_selector.support_].tolist()
print(f"#Features used: {len(selected_features)}")
print(f"Selected features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

# fit final tuned model on selected features
best_rf.fit(X_train_sel, y_train)

# evaluate boruta model
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
    # one-sided paired permutation (sign-flip) test on absolute errors
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
