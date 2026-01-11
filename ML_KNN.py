import pandas as pd
import numpy as np
from math import sqrt

from sklearn.model_selection import KFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from boruta import BorutaPy

TARGET = "ORTHOPHOSPHAT mg/l"
SEED = 42

train = pd.read_excel("Merged_train.xlsx")
test  = pd.read_excel("Merged_test.xlsx")

# Coerce target to numeric + drop missing target rows
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
test[TARGET]  = pd.to_numeric(test[TARGET],  errors="coerce")
train = train.dropna(subset=[TARGET]).copy()
test  = test.dropna(subset=[TARGET]).copy()

y_train = train[TARGET].to_numpy()
y_test  = test[TARGET].to_numpy()

# Use numeric predictors only
X_train = train.drop(columns=[TARGET]).select_dtypes(include="number").copy()
X_test  = test.drop(columns=[TARGET]).select_dtypes(include="number").copy()

# Align columns (in case train/test have slightly different numeric cols)
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test  = X_test[common_cols]

# ================================
# BORUTA FEATURE SELECTION (on train only)
# ================================
rf_for_boruta = RandomForestRegressor(
    n_estimators=800,
    random_state=SEED,
    n_jobs=-1,
)

boruta_selector = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators="auto",
    verbose=2,
    random_state=SEED
)

# BorutaPy can't handle NaNs -> impute for Boruta input only
X_train_boruta = X_train.copy()
X_train_boruta = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_train_boruta),
    columns=X_train.columns,
    index=X_train.index
)

boruta_selector.fit(X_train_boruta.to_numpy(), y_train)

selected_features = X_train.columns[boruta_selector.support_].tolist()
if len(selected_features) == 0:
    raise RuntimeError("Boruta selected 0 features. Try increasing n_estimators or check your data.")

print(f"\n#Features selected by Boruta: {len(selected_features)}")
print("Selected features:", selected_features)

# Subset + align (safety)
selected_features = [c for c in selected_features if c in X_test.columns]
X_train_sel = X_train[selected_features].copy()
X_test_sel  = X_test[selected_features].copy()

# ================================
# MANUAL HYPERPARAMETER TUNING (CV on selected features)
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
                ("imputer", SimpleImputer(strategy="median")),
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

            mean_score = float(scores.mean())
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"n_neighbors": n_neighbors, "weights": weights, "p": p}

print("\n=== Best KNN params from manual for-loop search ===")
print(best_params)
print(f"Best mean CV R² (5-fold) during tuning: {best_score:.4f}")

# Final tuned KNN pipeline
knn_best = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        p=best_params["p"],
    ))
])

# ================================
# Fit + Evaluate on held-out test
# ================================
knn_best.fit(X_train_sel, y_train)
pred = knn_best.predict(X_test_sel)

r2  = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = sqrt(mse)

# Permutation test (on train, with CV)
score_cv, perm_scores, p_value = permutation_test_score(
    estimator=knn_best,
    X=X_train_sel,
    y=y_train,
    scoring="r2",
    n_permutations=200,
    random_state=SEED,
    n_jobs=-1,
    cv=5,
)

print("\n=== Test set performance ===")
print(f"#Features used: {X_train_sel.shape[1]}")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Permutation-test p-value (train/CV): {p_value:.5f}")

# ================================
# MDA TABLE ON BORUTA-SELECTED FEATURES
# ================================
def mda_table(model, X_test_df, y_true, baseline_r2, baseline_mse, baseline_rmse,
              n_repeats=30, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []

    # IMPORTANT: model is a Pipeline -> it will impute/scale internally
    for feat in X_test_df.columns:
        r2_perm, mse_perm, rmse_perm = [], [], []

        for _ in range(n_repeats):
            Xp = X_test_df.copy()
            Xp[feat] = rng.permutation(Xp[feat].to_numpy())
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

        delta_r2   = baseline_r2 - r2_perm
        delta_mse  = mse_perm - baseline_mse
        delta_rmse = rmse_perm - baseline_rmse

        # one-sided: how often permuted >= baseline (bigger is "as good or better")
        p_feat = (1.0 + np.sum(r2_perm >= baseline_r2)) / (len(r2_perm) + 1.0)

        rows.append({
            "feature": feat,
            "delta_R2_mean": float(delta_r2.mean()),
            "delta_R2_std":  float(delta_r2.std(ddof=1)),
            "delta_MSE_mean": float(delta_mse.mean()),
            "delta_RMSE_mean": float(delta_rmse.mean()),
            "p_value": float(p_feat),
        })

    return (pd.DataFrame(rows)
              .sort_values("delta_R2_mean", ascending=False)
              .reset_index(drop=True))

mda_df = mda_table(
    model=knn_best,
    X_test_df=X_test_sel,
    y_true=y_test,
    baseline_r2=r2,
    baseline_mse=mse,
    baseline_rmse=rmse,
    n_repeats=30,
    seed=SEED,
)

print("\n=== MDA ranked by ΔR² ===")
with pd.option_context("display.max_rows", None, "display.width", 500):
    print(mda_df[["feature", "delta_R2_mean", "p_value"]])
