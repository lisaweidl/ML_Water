import pandas as pd
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import permutation_test_score
from boruta import BorutaPy

TARGET = "ORTHOPHOSPHAT mg/l"

train = pd.read_excel("df_train.xlsx")
test  = pd.read_excel("df_test.xlsx")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
test[TARGET]  = pd.to_numeric(test[TARGET],  errors="coerce")

train = train.dropna(subset=[TARGET])
test  = test.dropna(subset=[TARGET])


# test new added value: ID_code + seasonal features
if "ID" in train.columns and "ID" in test.columns:
    all_ids = pd.Index(train["ID"].astype(str)).append(pd.Index(test["ID"].astype(str))).unique()
    id_map = {k: i for i, k in enumerate(all_ids)}
    train["ID_code"] = train["ID"].astype(str).map(id_map)
    test["ID_code"]  = test["ID"].astype(str).map(id_map)


y_train = train[TARGET].values
y_test  = test[TARGET].values

X_train = train.drop(columns=[TARGET]).select_dtypes(include="number").fillna(0)
X_test  = test.drop(columns=[TARGET]).select_dtypes(include="number").fillna(0)

# Boruta
rf_for_boruta = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=300,
    random_state=42,
    max_depth=5
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

# RF + selected features
X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

rf = RandomForestRegressor(
    random_state=42,
    n_estimators=300,
    n_jobs=-1
)
rf.fit(X_train_sel, y_train)
pred = rf.predict(X_test_sel)

# Evaluation
r2  = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = sqrt(mse)

# p-value (test model significance)
score_cv, perm_scores, p_value = permutation_test_score(
    estimator=RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1),
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

# MDA table on Boruta selected features
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
            r2_perm.append(r2p); mse_perm.append(msep); rmse_perm.append(rmsep)

        r2_perm = np.array(r2_perm); mse_perm = np.array(mse_perm); rmse_perm = np.array(rmse_perm)

        # Drops (Δ) = baseline − permuted (for R²); permuted − baseline for errors
        delta_r2   = baseline_r2 - r2_perm
        delta_mse  = mse_perm - baseline_mse
        delta_rmse = rmse_perm - baseline_rmse

        # One-sided p-value(probability that permuting this feature does NOT reduce R²)
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