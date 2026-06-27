"""
Week 3: Real-World Sensor Data -- UCI Air Quality Dataset
=========================================================
  Day 11 -- Data ingestion & schema cleaning (European CSV format)
  Day 12 -- Exploratory data analysis on raw sensor signals
  Day 13 -- Missing-value handling & imputation strategies
  Day 14 -- Temporal splitting & cyclical feature engineering
  Day 15 -- Modelling (Random Forest) + data-leakage experiment

Run:
    python week3_sensor_pipeline.py

Dependencies:
    scikit-learn, pandas, numpy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

os.makedirs("plots", exist_ok=True)

# ============================================================================
#  DAY 11: Data Ingestion & Schema Cleaning
# ============================================================================

print("=" * 65)
print("  DAY 11 -- Data Ingestion & Schema Cleaning")
print("=" * 65)

# Wrong — will produce garbage columns
print("\n--- Broken parse (default pd.read_csv) ---")
df_broken = pd.read_csv("AirQualityUCI.csv")
print(df_broken.head())
print(df_broken.shape)

# Correct parse with European formatting
print("\n--- Correct parse (sep=';', decimal=',') ---")
df = pd.read_csv(
    "AirQualityUCI.csv",
    sep=';',
    decimal=',',
).dropna(how='all', axis=1).dropna(how='all', axis=0)

print(df.shape)
print(df.dtypes)
print(df.head())

# Column mapping
print("\n--- Column Mapping ---")
column_map = {
    "Date":          "Date of measurement (DD/MM/YYYY)",
    "Time":          "Time of measurement (HH.MM.SS)",
    "CO(GT)":        "Ground-truth CO concentration (mg/m3)",
    "PT08.S1(CO)":   "Tin-oxide sensor response -- targets CO (raw sensor resistance)",
    "NMHC(GT)":      "Ground-truth Non-Methane Hydrocarbons (ug/m3)",
    "C6H6(GT)":      "Ground-truth Benzene concentration (ug/m3)",
    "PT08.S2(NMHC)": "Titania sensor response -- targets NMHC (raw sensor resistance)",
    "NOx(GT)":       "Ground-truth NOx concentration (ppb)",
    "PT08.S3(NOx)":  "Tungsten-oxide sensor response -- targets NOx (raw sensor resistance)",
    "NO2(GT)":       "Ground-truth NO2 concentration (ug/m3)",
    "PT08.S4(NO2)":  "Tungsten-oxide sensor response -- targets NO2 (raw sensor resistance)",
    "PT08.S5(O3)":   "Indium-oxide sensor response -- targets O3 (raw sensor resistance)",
    "T":             "Temperature (deg C)",
    "RH":            "Relative Humidity (%)",
    "AH":            "Absolute Humidity",
}
for col, desc in column_map.items():
    if col in df.columns:
        print(f"  {col:20s}  ->  {desc}")

# ============================================================================
#  DAY 12: EDA on Sensor Data
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 12 -- EDA on Sensor Data")
print("=" * 65)

# Parse a combined datetime index
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                 format='%d/%m/%Y %H.%M.%S')
df = df.set_index('Datetime').sort_index()
df.drop(columns=['Date', 'Time'], inplace=True)

# Plot raw Benzene concentration over time
df['C6H6(GT)'].plot(figsize=(14, 4), title="Raw Benzene Concentration Over Time")
plt.ylabel("C₆H₆ (μg/m³)")
plt.tight_layout()
plt.savefig("plots/benzene_raw.png", dpi=150)
plt.close()
print("\n  -> Saved: plots/benzene_raw.png")

# Observations
print("\n  Look for:")
print("   * The -200 sentinel values (sensor failure) visible as sharp downward spikes")
print("   * Any long runs of constant values (stuck sensors)")
print("   * The overall shape of the time series (daily, seasonal patterns)")
print("\n  What does -200 represent physically?")
print("  A gas sensor can NEVER genuinely read a negative concentration.")
print("  -200 is a manufacturer-designated error code.")

# ============================================================================
#  DAY 13: Missing Value Handling
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 13 -- Missing Value Handling")
print("=" * 65)

# Replace the -200 sentinel values with NaN
df.replace(-200, np.nan, inplace=True)

# Quantify the missingness
missing_pct = df.isnull().mean() * 100
print("\n  Missingness per column (% NaN):")
print(missing_pct.sort_values(ascending=False))

# Compare three imputation strategies on a single sensor column
sensor_col = 'PT08.S1(CO)'
print(f"\n  Imputation strategy comparison on: {sensor_col}")

# Strategy 1: Drop all rows with any NaN
df_dropped = df.dropna(subset=[sensor_col])

# Strategy 2: Forward-fill
df_ffill = df.copy()
df_ffill[sensor_col] = df_ffill[sensor_col].ffill()

# Strategy 3: Rolling mean imputation (window of 3 hours)
df_rolling = df.copy()
df_rolling[sensor_col] = df_rolling[sensor_col].fillna(
    df_rolling[sensor_col].rolling(window=3, min_periods=1).mean()
)

print(f"\n  Strategy 1 (Drop NaN rows): {len(df_dropped)} rows kept")
print(f"  Strategy 2 (Forward-fill):  {len(df_ffill)} rows, {df_ffill[sensor_col].isna().sum()} NaN remaining")
print(f"  Strategy 3 (Rolling mean):  {len(df_rolling)} rows, {df_rolling[sensor_col].isna().sum()} NaN remaining")

print("\n  Trade-offs:")
print("   * Dropping rows: loses data and can introduce gaps in a time series.")
print("   * Forward-fill: is simple but can propagate stale values for hours,")
print("     artificially smoothing out real events.")
print("   * Rolling mean: is more representative but still assumes a local")
print("     continuity that may not exist during rapid pollution events.")
print("   * There is no universally correct answer. The best strategy depends")
print("     on how frequent and how long the sensor failures are.")

# Apply ffill + bfill for downstream modelling
df.ffill(inplace=True)
df.bfill(inplace=True)

# ============================================================================
#  DAY 14: Temporal Splitting and Feature Engineering
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 14 -- Temporal Splitting and Feature Engineering")
print("=" * 65)

# Why random splitting is wrong for time-series data
print("\n  [!] DO NOT use train_test_split(shuffle=True) for sequential data!")
print("  If you shuffle rows, future observations end up in the training set")
print("  and past observations in the test set. Test performance looks great,")
print("  but it measures your model's ability to interpolate within the data,")
print("  not to generalise forward in time.")

# Correct approach — chronological split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

print(f"\n  Training period: {train.index.min()} -> {train.index.max()}")
print(f"  Test period:     {test.index.min()} -> {test.index.max()}")

# Cyclical hour features
df['Hour'] = df.index.hour
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

print("\n  Cyclical hour encoding added:")
print("    Hour_sin = sin(2*pi * Hour / 24)")
print("    Hour_cos = cos(2*pi * Hour / 24)")
print("  This ensures Hour 23 is adjacent to Hour 0.")

# Re-split after adding features
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

# ============================================================================
#  DAY 15: Modelling and Week 3 Deliverable
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 15 -- Modelling and Week 3 Deliverable")
print("=" * 65)

target = 'C6H6(GT)'
feature_cols = [c for c in df.columns if c != target and c != 'Hour']

# --- Temporal split model (honest evaluation) ---
print("\n--- Temporal split model (honest) ---")

X_train = train[feature_cols].ffill()
y_train = train[target]
X_test  = test[feature_cols].ffill()
y_test  = test[target]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

r2_temporal = r2_score(y_test, rf.predict(X_test))
print(f"  Temporal split -- Test R2: {r2_temporal:.3f}")

# --- Controlled leakage experiment ---
print("\n--- Random shuffle split model (leaky) ---")

X_all = df[feature_cols].ffill()
y_all = df[target]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_all, y_all, test_size=0.2, shuffle=True, random_state=42
)

rf_random = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_random.fit(X_train_r, y_train_r)

r2_random = r2_score(y_test_r, rf_random.predict(X_test_r))
print(f"  Random shuffle -- Test R2: {r2_random:.3f}")

# --- Comparison ---
print("\n--- Data Leakage Comparison ---")
print(f"\n  {'Split Strategy':<22}  {'Test R2':>10}  {'Honest?':>10}")
print(f"  {'---------------':<22}  {'-------':>10}  {'-------':>10}")
print(f"  {'Random shuffle':<22}  {r2_random:>10.3f}  {'No':>10}")
print(f"  {'Chronological':<22}  {r2_temporal:>10.3f}  {'Yes':>10}")

# ============================================================================
#  REFLECTION
# ============================================================================

print("Week 3 script completed successfully.")