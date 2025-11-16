import pandas as pd
import numpy as np
from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import permutation_test_score, RandomizedSearchCV
from boruta import BorutaPy
from sklearn.model_selection import permutation_test_score, RandomizedSearchCV, KFold, cross_val_score


TARGET = "ORTHOPHOSPHAT mg/l"

train = pd.read_excel("df_train.xlsx")
test  = pd.read_excel("df_test.xlsx")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
test[TARGET]  = pd.to_numeric(test[TARGET],  errors="coerce")

train = train.dropna(subset=[TARGET])
test  = test.dropna(subset=[TARGET])

y_train = train[TARGET].values
y_test  = test[TARGET].values

# numeric predictors
X_train = train.drop(columns=[TARGET]).select_dtypes(include="number")
X_test  = test.drop(columns=[TARGET]).select_dtypes(include="number")


#   RANDOMIZED HYPERPARAMETER TUNING
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [5, 10, 15, None],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=20,
    scoring="r2",
    n_jobs=-1,
    cv=5,
    random_state=42
)

search.fit(X_train, y_train)

best_params = search.best_params_
print("Best RF params from RandomizedSearchCV:")
print(best_params)

#   BORUTA WITH TUNED RF
rf_for_boruta = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    **best_params
)

boruta_selector = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators='auto',
    verbose=2,
    random_state=42
)
boruta_selector.fit(X_train.values, y_train)

selected_features = X_train.columns[boruta_selector.support_].tolist()
print(f"#Features used: {len(selected_features)}")
print(f"Selected features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

# Cross-validation on training data
cv = KFold(n_splits=10, shuffle=True, random_state=42)

rf_cv = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    **best_params
)

cv_scores = cross_val_score(
    rf_cv,
    X_train_sel,
    y_train,
    scoring="r2",
    cv=cv,
    n_jobs=-1
)

print(f"Mean CV R² ({cv.get_n_splits()}-fold): {cv_scores.mean():.4f}")
print(f"Std  CV R²: {cv_scores.std():.4f}")

# Final RF fit on full train + test prediction
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    **best_params
)
rf.fit(X_train_sel, y_train)
pred = rf.predict(X_test_sel)

#   EVALUATION
r2  = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = sqrt(mse)

# p-value (test model significance) using tuned RF
rf_for_perm = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    **best_params
)

score_cv, perm_scores, p_value = permutation_test_score(
    estimator=rf_for_perm,
    X=X_train_sel,
    y=y_train,
    scoring="r2",
    n_permutations=200,
    random_state=42,
    n_jobs=-1,
    cv=5
)

print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"P-value (Permutation test on CV R²): {p_value:.5f}")

#   MDA TABLE ON BORUTA-SELECTED FEATURES
def mda_table(model, X_test_df, y_true, baseline_r2, baseline_mse, baseline_rmse,
              n_repeats=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for feat in X_test_df.columns:
        r2_perm = []
        mse_perm = []
        rmse_perm = []
        for _ in range(n_repeats):
            Xp = X_test_df.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            y_hat = model.predict(Xp)
            r2p = r2_score(y_true, y_hat)
            msep = mean_squared_error(y_true, y_hat)
            rmsep = sqrt(msep)
            r2_perm.append(r2p)
            mse_perm.append(msep)
            rmse_perm.append(rmsep)

        r2_perm = np.array(r2_perm)
        mse_perm = np.array(mse_perm)
        rmse_perm = np.array(rmse_perm)

        # Drops (Δ) = baseline − permuted (for R²); permuted − baseline for errors
        delta_r2   = baseline_r2 - r2_perm
        delta_mse  = mse_perm - baseline_mse
        delta_rmse = rmse_perm - baseline_rmse

        # One-sided p-value (probability that permuting this feature does NOT reduce R²)
        p_feat = (1.0 + np.sum(r2_perm >= baseline_r2)) / (len(r2_perm) + 1.0)

        rows.append({
            "feature": feat,
            "delta_R2_mean": float(delta_r2.mean()),
            "delta_R2_std":  float(delta_r2.std(ddof=1)),
            "delta_MSE_mean": float(delta_mse.mean()),
            "delta_RMSE_mean": float(delta_rmse.mean()),
            "p_value": float(p_feat)
        })
    df_imp = pd.DataFrame(rows).sort_values("delta_R2_mean", ascending=False).reset_index(drop=True)
    return df_imp

mda_df = mda_table(
    model=rf,
    X_test_df=X_test_sel,
    y_true=y_test,
    baseline_r2=r2,
    baseline_mse=mse,
    baseline_rmse=rmse,
    n_repeats=30,
    seed=42
)

print("\n=== MDA ranked by ΔR² ===")
with pd.option_context('display.max_rows', None, 'display.width', 500):
    print(mda_df[["feature", "delta_R2_mean", "delta_MSE_mean", "delta_RMSE_mean", "p_value"]])
