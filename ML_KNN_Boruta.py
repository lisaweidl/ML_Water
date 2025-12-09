import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    permutation_test_score,
    KFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

TARGET = "ORTHOPHOSPHAT mg/l"
SEED = 42

train = pd.read_excel("df_train.xlsx")
test  = pd.read_excel("df_test.xlsx")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
test[TARGET]  = pd.to_numeric(test[TARGET],  errors="coerce")

train = train.dropna(subset=[TARGET])
test  = test.dropna(subset=[TARGET])

y_train = train[TARGET].values
y_test  = test[TARGET].values

X_train = train.drop(columns=[TARGET]).select_dtypes(include="number")
X_test  = test.drop(columns=[TARGET]).select_dtypes(include="number")

# ================================
# BORUTA FEATURE SELECTION (RF)
# ================================
rf_for_boruta = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=300,
    random_state=SEED,
    max_depth=5,
)

boruta = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators="auto",
    verbose=2,
    random_state=SEED,
)
boruta.fit(X_train.values, y_train)

selected_features  = X_train.columns[boruta.support_].tolist()
tentative_features = X_train.columns[boruta.support_weak_].tolist()

# Safety: if Boruta selects nothing, fall back
if len(selected_features) == 0:
    selected_features = tentative_features
if len(selected_features) == 0:
    selected_features = X_train.columns.tolist()

print("\nBoruta results:")
print(f"Selected ({len(selected_features)}): {selected_features}")
print(f"Tentative ({len(tentative_features)}): {tentative_features}")

X_train_sel = X_train[selected_features].copy()
X_test_sel  = X_test[selected_features].copy()

# ================================
# MANUAL HYPERPARAMETER TUNING FOR KNN (FOR-LOOPS)
# ================================

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 13],
    "weights": ["uniform", "distance"],
    "p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
}

cv_inner = KFold(n_splits=5, shuffle=True, random_state=SEED)

best_score = -np.inf
best_params = None

for n_neighbors in param_grid["n_neighbors"]:
    for weights in param_grid["weights"]:
        for p in param_grid["p"]:
            pipe_tmp = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p
                ))
            ])

            scores = cross_val_score(
                estimator=pipe_tmp,
                X=X_train_sel,
                y=y_train,
                scoring="r2",
                cv=cv_inner,
                n_jobs=-1,
            )
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    "n_neighbors": n_neighbors,
                    "weights": weights,
                    "p": p,
                }

print("\n=== Best KNN params from manual for-loop search ===")
print(best_params)
print(f"Best mean CV R² (5-fold) during tuning: {best_score:.4f}")

# Build final tuned KNN pipeline
knn_best = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        p=best_params["p"],
    ))
])

# ================================
# 10-FOLD CV ON TUNED KNN
# ================================
cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

cv_scores = cross_val_score(
    estimator=knn_best,
    X=X_train_sel,
    y=y_train,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
)

print(f"\nMean CV R² ({cv.get_n_splits()}-fold, tuned KNN): {cv_scores.mean():.4f}")
print(f"Std  CV R²: {cv_scores.std():.4f}")

# ================================
# Fit tuned KNN on full train + predict test
# ================================
knn_best.fit(X_train_sel, y_train)
pred = knn_best.predict(X_test_sel)

r2  = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = sqrt(mse)

print("\n=== KNN on Boruta-selected features (held-out test) ===")
print(f"#Features used: {X_train_sel.shape[1]}")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ================================
# Permutation-test p-value (KNN)
# ================================
score_cv, perm_scores, p_value = permutation_test_score(
    estimator=knn_best,
    X=X_train_sel,
    y=y_train,
    scoring="r2",
    n_permutations=200,
    random_state=SEED,
    n_jobs=-1,
    cv=cv,
)

print("\n=== Permutation test (KNN) ===")
print(f"Mean CV R² (true): {score_cv:.4f}")
print(f"P-value:           {p_value:.5f}")

# ================================
# MDA (Permutation importance) on TEST set
# ================================
def mda_table(model, X_test_df, y_true, baseline_r2, baseline_mse, baseline_rmse,
              n_repeats=30, seed=SEED):
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

        r2_perm  = np.array(r2_perm)
        mse_perm = np.array(mse_perm)
        rmse_perm = np.array(rmse_perm)

        delta_r2   = baseline_r2 - r2_perm
        delta_mse  = mse_perm - baseline_mse
        delta_rmse = rmse_perm - baseline_rmse

        p_feat = (1.0 + np.sum(r2_perm >= baseline_r2)) / (len(r2_perm) + 1.0)

        rows.append({
            "feature": feat,
            "delta_R2_mean": float(delta_r2.mean()),
            "delta_R2_std":  float(delta_r2.std(ddof=1)),
            "delta_MSE_mean": float(delta_mse.mean()),
            "delta_RMSE_mean": float(delta_rmse.mean()),
            "p_value": float(p_feat),
        })

    df_imp = pd.DataFrame(rows).sort_values("delta_R2_mean", ascending=False).reset_index(drop=True)
    return df_imp

mda_df = mda_table(
    model=knn_best,
    X_test_df=X_test_sel,  # unscaled; pipeline handles scaling internally
    y_true=y_test,
    baseline_r2=r2,
    baseline_mse=mse,
    baseline_rmse=rmse,
    n_repeats=30,
    seed=SEED,
)

print("\n=== KNN MDA ranked by ΔR² ===")
with pd.option_context("display.max_rows", None, "display.width", 500):
    print(mda_df[["feature", "delta_R2_mean", "delta_MSE_mean", "delta_RMSE_mean", "p_value"]])
