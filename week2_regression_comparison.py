"""
Week 2: Your First Models - Regression, Overfitting, and Ensembles
==================================================================
Trains and evaluates four regression model families on the California Housing
dataset:
  Day 6  - Linear Regression (baseline)
  Day 7  - Decision Tree (overfitting demo + regularisation sweep)
  Day 8  - Random Forest (ensemble, scale-invariance demo)
  Day 9  - XGBoost (gradient boosting)
  Day 10 - Comparative performance table

Run:
    python week2_regression_comparison.py

Dependencies:
    scikit-learn, xgboost, pandas, numpy
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# -- shared helpers ----------------------------------------------------------

def print_metrics(label, y_true_train, y_pred_train, y_true_test, y_pred_test):
    """Print a compact train / test metric block for one model."""
    metrics = {
        "MAE":  (mean_absolute_error, y_true_train, y_pred_train,
                 y_true_test,  y_pred_test),
        "RMSE": (root_mean_squared_error, y_true_train, y_pred_train,
                 y_true_test,  y_pred_test),
        "R2":   (r2_score, y_true_train, y_pred_train,
                 y_true_test,  y_pred_test),
    }
    print(f"\n{'--'*25}")
    print(f"  {label}")
    print(f"{'--'*25}")
    print(f"  {'Metric':<8}  {'Train':>10}  {'Test':>10}")
    print(f"  {'------':<8}  {'-----':>10}  {'----':>10}")
    for name, (fn, ytr, ptr, yte, pte) in metrics.items():
        print(f"  {name:<8}  {fn(ytr, ptr):>10.4f}  {fn(yte, pte):>10.4f}")
    print(f"{'--'*25}\n")


# -- 0. Data preparation (mirrors Week 1 preprocessing) ----------------------

print("=" * 60)
print("  Week 2 - Regression, Overfitting & Ensembles")
print("=" * 60)

data = fetch_california_housing(as_frame=True)
df   = data.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaled versions for Linear Regression (and scale-invariance comparison)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nDataset ready - train: {len(X_train):,}  test: {len(X_test):,} rows")

# -- Day 6: Linear Regression -------------------------------------------------

print("\n" + "=" * 60)
print("  DAY 6 - Linear Regression")
print("=" * 60)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_train_lr = lr.predict(X_train_scaled)
y_pred_test_lr  = lr.predict(X_test_scaled)

print_metrics("Linear Regression", y_train, y_pred_train_lr, y_test, y_pred_test_lr)

# Feature coefficients - show which features push prices up / down
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lr.coef_})
print("Feature Coefficients (sorted, largest -> smallest):")
print(coef_df.sort_values("Coefficient", ascending=False).to_string(index=False))

# Collect summary row for Day 10 table
summary_rows = []
summary_rows.append({
    "Model":            "Linear Regression",
    "Scaling Required": "Yes",
    "Train R2":         round(r2_score(y_train, y_pred_train_lr), 4),
    "Test R2":          round(r2_score(y_test,  y_pred_test_lr),  4),
    "Test RMSE":        round(root_mean_squared_error(y_test, y_pred_test_lr), 4),
})

# -- Day 7: Decision Trees & Overfitting --------------------------------------

print("\n" + "=" * 60)
print("  DAY 7 - Decision Trees & Overfitting")
print("=" * 60)

from sklearn.tree import DecisionTreeRegressor

# Unlimited depth - classic overfitting example
dt_unlimited = DecisionTreeRegressor(random_state=42)
dt_unlimited.fit(X_train, y_train)

train_r2_dt_ul = r2_score(y_train, dt_unlimited.predict(X_train))
test_r2_dt_ul  = r2_score(y_test,  dt_unlimited.predict(X_test))
print(f"Unlimited tree - Train R2: {train_r2_dt_ul:.3f}  |  Test R2: {test_r2_dt_ul:.3f}")
print("  -> Near-perfect train R2 with much lower test R2 = classic overfitting.\n")

# Add unlimited DT to summary
summary_rows.append({
    "Model":            "Decision Tree (default)",
    "Scaling Required": "No",
    "Train R2":         round(train_r2_dt_ul, 4),
    "Test R2":          round(test_r2_dt_ul,  4),
    "Test RMSE":        round(root_mean_squared_error(y_test, dt_unlimited.predict(X_test)), 4),
})

# Regularisation sweep - bias-variance trade-off
print("max_depth regularisation sweep:")
print(f"  {'max_depth':>10}  {'Train R2':>10}  {'Test R2':>10}")
print(f"  {'----------'}  {'----------'}  {'----------'}")

dt_depth5 = None
for depth in [2, 3, 5, 8, 10, None]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    tr2 = r2_score(y_train, dt.predict(X_train))
    te2 = r2_score(y_test,  dt.predict(X_test))
    label = str(depth) if depth is not None else "None (unlimited)"
    print(f"  {label:>10}  {tr2:>10.3f}  {te2:>10.3f}")
    if depth == 5:
        dt_depth5 = dt  # keep depth-5 model for summary

print("\n  -> Sweet spot: Test R2 peaks around max_depth=5-8 before overfitting returns.")

# Add depth-5 DT to summary
if dt_depth5:
    summary_rows.append({
        "Model":            "Decision Tree (max_depth=5)",
        "Scaling Required": "No",
        "Train R2":         round(r2_score(y_train, dt_depth5.predict(X_train)), 4),
        "Test R2":          round(r2_score(y_test,  dt_depth5.predict(X_test)),  4),
        "Test RMSE":        round(root_mean_squared_error(y_test, dt_depth5.predict(X_test)), 4),
    })

# -- Day 8: Random Forest -----------------------------------------------------

print("\n" + "=" * 60)
print("  DAY 8 - Random Forest (Ensemble)")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_train_rf = rf.predict(X_train)
y_pred_test_rf  = rf.predict(X_test)

print_metrics("Random Forest (unscaled)", y_train, y_pred_train_rf, y_test, y_pred_test_rf)

# Scale-invariance demonstration
rf_scaled = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_scaled.fit(X_train_scaled, y_train)

test_r2_rf_scaled = r2_score(y_test, rf_scaled.predict(X_test_scaled))
test_r2_rf_raw    = r2_score(y_test, y_pred_test_rf)

print("Scale-invariance check:")
print(f"  RF on raw features    -> Test R2: {test_r2_rf_raw:.4f}")
print(f"  RF on scaled features -> Test R2: {test_r2_rf_scaled:.4f}")
diff = abs(test_r2_rf_raw - test_r2_rf_scaled)
print(f"  Delta = {diff:.6f}  {'[OK] Virtually identical - tree-based models are scale-invariant.' if diff < 0.005 else '[WARN] Unexpected difference.'}")

summary_rows.append({
    "Model":            "Random Forest",
    "Scaling Required": "No",
    "Train R2":         round(r2_score(y_train, y_pred_train_rf), 4),
    "Test R2":          round(test_r2_rf_raw, 4),
    "Test RMSE":        round(root_mean_squared_error(y_test, y_pred_test_rf), 4),
})

# -- Day 9: XGBoost -----------------------------------------------------------

print("\n" + "=" * 60)
print("  DAY 9 - Gradient Boosting (XGBoost)")
print("=" * 60)

from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train, y_train)

y_pred_train_xgb = xgb.predict(X_train)
y_pred_test_xgb  = xgb.predict(X_test)

print_metrics("XGBoost", y_train, y_pred_train_xgb, y_test, y_pred_test_xgb)

summary_rows.append({
    "Model":            "XGBoost",
    "Scaling Required": "No",
    "Train R2":         round(r2_score(y_train, y_pred_train_xgb), 4),
    "Test R2":          round(r2_score(y_test,  y_pred_test_xgb),  4),
    "Test RMSE":        round(root_mean_squared_error(y_test, y_pred_test_xgb), 4),
})

print("Week 2 script completed successfully.")
print("See README.md for the comparative performance table and reflection.")
