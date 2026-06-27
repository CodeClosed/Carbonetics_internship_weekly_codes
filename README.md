# Carbonetics ML Internship — Weekly Codes

This repository contains the weekly implementations, codes, and projects developed during the Carbonetics Machine Learning Internship.

---

## 📅 Roadmap by Weeks

### 🔍 Week 1: Exploratory Data Analysis (EDA)
- **Focus**: Dataset understanding, distribution visualization, feature correlation, train/test splitting, and feature scaling.
- **Files**:
  - [week1_eda.py](file:///c:/Users/vihaa/OneDrive/Desktop/carbonetics_ml_internship/week1_eda.py): Python script performing EDA and preprocessing steps.
- **Details**:
  1. **Dataset**: Loaded California Housing dataset using `sklearn.datasets`.
  2. **Visualization**: Plotted feature distributions, target variable distribution (`MedHouseVal`), and a correlation heatmap under the `plots/` folder.
  3. **Preprocessing**: Performed a standard train/test split (80-20) and scaled features using `StandardScaler`.

### 📝 Week 1 Discussion & Reflection

#### Discussion Questions
* **Which feature correlates most strongly with house value?**
  * **Answer**: `MedInc` (Median Income) correlates most strongly with `MedHouseVal` (Median House Value) with a correlation coefficient of approximately `0.69`.
* **Look at the scales of each feature. MedInc is in tens of thousands; Population is in the thousands. Why might this be a problem for some models?**
  * **Answer**: Distance-based models (e.g., K-Nearest Neighbors, Support Vector Machines) and gradient-descent-based models (e.g., Linear Regression, Neural Networks) are highly sensitive to feature scales. Features with larger ranges (like `Population`) will dominate the distance calculations or gradients, rendering features with smaller scales (like `MedInc`) less influential. Scaling ensures all features contribute proportionally. Tree-based models (e.g., Random Forests, XGBoost) are scale-invariant and do not suffer from this issue.

#### Reflection
* **If you called `scaler.fit_transform(X_test)` instead of `scaler.transform(X_test)`, what would go wrong? What information would have leaked?**
  * **Answer**: Calling `scaler.fit_transform(X_test)` would calculate the mean and standard deviation of the test set and scale it using these new parameters. 
    1. **Inconsistent Scaling**: The features in the test set would be scaled using different parameters ($\mu_{test}$ and $\sigma_{test}$) than those used for the training set ($\mu_{train}$ and $\sigma_{train}$). This means the same numerical value would map to different scaled values between train and test, confusing the model.
    2. **Data Leakage / Real-world Infeasibility**: In production, new data arrives one sample at a time, making it impossible to compute a meaningful mean or standard deviation for fitting. Preprocessing must rely strictly on parameters learned from the training set. Fitting the scaler on the test set is a form of data leakage because the preprocessing pipeline adapts to the distribution of the unseen test data.

---

### 📈 Week 2: Regression, Overfitting & Ensembles

- **Focus**: Train and evaluate regression models of increasing complexity. Understand overfitting and how ensemble methods address it.
- **Files**:
  - [week2_regression_comparison.py](./week2_regression_comparison.py): Full pipeline covering Linear Regression, Decision Trees, Random Forest, and XGBoost.
- **Key Concepts**: Linear Regression metrics (MAE, RMSE, R²) · Decision Tree regularisation · Random Forest bagging · XGBoost gradient boosting · Bias-variance trade-off

#### Day 6 — Linear Regression
- Trained `LinearRegression` on `StandardScaler`-scaled features (required for gradient-descent-based models).
- Printed train/test MAE, RMSE, and R² and sorted feature coefficients to identify which features push house prices up or down.
- **Runtime Results**: Train MAE `0.5286` · Test MAE `0.5332` · Train RMSE `0.7197` · Test RMSE `0.7456` · Train R² `0.6126` · **Test R² `0.5758`**
- **Top coefficients**: `MedInc` (+0.854), `Longitude` (−0.870), `Latitude` (−0.897)

#### Day 7 — Decision Trees & Overfitting
- Trained an unlimited `DecisionTreeRegressor`: Train R² ≈ 1.0, Test R² much lower → classic memorisation / overfitting.
- Swept `max_depth` ∈ {2, 3, 5, 8, 10, None} to observe the bias-variance trade-off.
- **Runtime Results**:

  | max_depth | Train R² | Test R² |
  |-----------|----------|---------|
  | 2 | 0.452 | 0.424 |
  | 3 | 0.538 | 0.510 |
  | 5 | 0.638 | 0.600 |
  | 8 | 0.760 | 0.678 |
  | 10 | 0.835 | 0.683 |
  | None (unlimited) | 1.000 | 0.622 |

  → Sweet spot: Test R² peaks around `max_depth=5–8` before overfitting returns.

#### Day 8 — Random Forest (Ensemble)
- Trained `RandomForestRegressor(n_estimators=100)` — builds trees on random bootstrap samples and averages predictions (bagging) to reduce variance.
- **Scale-invariance demo**: Identical test R² whether raw or scaled features are used, confirming tree-based models are indifferent to feature scale.
- **Runtime Results**: Train MAE `0.1221` · Test MAE `0.3275` · Train RMSE `0.1880` · Test RMSE `0.5053` · Train R² `0.9736` · **Test R² `0.8051`**
- Scale-invariance confirmed: raw vs scaled delta = `0.000152` (virtually identical).

#### Day 9 — XGBoost (Gradient Boosting)
- Trained `XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)` — trees are built **sequentially**, each correcting residual errors of the previous ensemble.
- **Runtime Results**: Train MAE `0.2922` · Test MAE `0.3251` · Train RMSE `0.4222` · Test RMSE `0.4878` · Train R² `0.8667` · **Test R² `0.8184`**

#### Day 10 — Comparative Performance Table

| Model | Scaling Required | Train R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|---|
| Linear Regression | ✅ Yes | 0.6126 | 0.5758 | 0.5332 | 0.7456 |
| Decision Tree (unlimited) | ❌ No | 1.0000 | 0.6220 | — | — |
| Decision Tree (max_depth=5) | ❌ No | 0.6380 | 0.6000 | — | — |
| Decision Tree (max_depth=8) | ❌ No | 0.7600 | 0.6780 | — | — |
| Random Forest (100 trees) | ❌ No | 0.9736 | 0.8051 | 0.3275 | 0.5053 |
| XGBoost (200 trees) | ❌ No | 0.8667 | 0.8184 | 0.3251 | 0.4878 |

*(Values above are exact runtime results — dataset: California Housing, train: 16,512 · test: 4,128 rows.)*

### 📝 Week 2 Discussion & Reflection

#### Discussion Questions

**Q: What is the fundamental architectural difference between Random Forest and XGBoost?**

**A:**
- **Random Forest** builds trees **in parallel** on independent bootstrap samples (bagging). Each tree is a full strong learner; predictions are averaged. This primarily reduces **variance** (overfitting).
- **XGBoost** builds trees **sequentially**. Each new tree fits the residual pseudo-gradients of the previous ensemble — the model iteratively corrects its own mistakes. This reduces both **bias** and **variance**, yielding higher accuracy on well-prepared tabular data.

**Q: When might you prefer one over the other?**

| Scenario | Prefer |
|---|---|
| Quick, robust baseline with minimal tuning | **Random Forest** |
| Noisy data / many irrelevant features | **Random Forest** |
| Squeezing maximum accuracy on clean tabular data | **XGBoost** |
| Missing values in raw data | **XGBoost** (handles natively) |
---

### 🛠️ Week 3: Real-World Sensor Data — UCI Air Quality

- **Focus**: Non-standard CSV parsing, sentinel value handling, imputation strategies, time-series data leakage, cyclical feature engineering.
- **Files**:
  - [week3_sensor_pipeline.py](./week3_sensor_pipeline.py): Full pipeline covering Days 11–15.
- **Dataset**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality) — 9,357 hourly records from metal-oxide chemical sensors in an Italian city (Mar 2004 – Apr 2005).
- **Key Concepts**: European CSV formatting · Sensor failure sentinel values (`-200`) · Imputation trade-offs · Chronological train/test split · Cyclical time encoding · Data leakage detection

#### Day 11 — Data Ingestion & Schema Cleaning
- Demonstrated that a naïve `pd.read_csv()` silently misparses the file (produces 1 garbled column instead of 15) because the dataset uses **semicolons** (`;`) as delimiters and **commas** (`,`) as decimal separators (European convention).
- Correct parse: `pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',')`
- **Runtime Results**: Broken shape `(9471, 1)` → Correct shape `(9357, 15)` with proper numeric dtypes.
- Mapped all 15 columns to physical meanings: 5 ground-truth chemical concentrations, 5 raw sensor resistances, temperature (T), relative humidity (RH), and absolute humidity (AH).

#### Day 12 — EDA on Sensor Data
- Created a datetime index spanning `2004-03-10 18:00` → `2005-04-04 14:00`.
- Plotted raw benzene concentration and all sensor channels over time.
- **Key Observations**:
  - Sharp downward spikes to `-200` are visible across many channels — these are sensor-failure sentinel values, not real measurements.
  - A gas sensor cannot produce a negative concentration reading; `-200` is a manufacturer-designated error code.
  - Daily (diurnal) patterns are visible: benzene peaks during morning and evening traffic rush hours.

#### Day 13 — Missing Value Handling
- Replaced all `-200` sentinel values with `NaN`.
- **Missingness Report**:

  | Column | % NaN |
  |---|---|
  | NMHC(GT) | 90.2% |
  | CO(GT) | 18.0% |
  | NO2(GT), NOx(GT) | 17.5% |
  | All sensor channels (PT08.*), T, RH, AH | 3.9% |

- Compared three imputation strategies on `PT08.S1(CO)`:

  | Strategy | Rows Kept | NaN Remaining |
  |---|---|---|
  | Drop NaN rows | 8,991 | 0 |
  | Forward-fill | 9,357 | 0 |
  | Rolling mean (w=3) | 9,357 | 337 |

  → Dropping loses 366 rows (3.9% of data).

- **Trade-offs**:
  - *Dropping rows*: loses data and introduces gaps in the time series.
  - *Forward-fill*: simple but propagates stale values, artificially smoothing real pollution events.
  - *Rolling mean*: more representative but assumes local continuity that may not exist during rapid pollution spikes.
- Applied `ffill()` + `bfill()` for downstream modelling → **0 NaN remaining**.

#### Day 14 — Temporal Splitting & Feature Engineering
- **Why random splitting is wrong**: shuffling rows leaks future observations into the training set, inflating test performance by measuring *interpolation* rather than *generalisation*.
- **Chronological split (80/20)**:
  - Training period: `2004-03-10 18:00` → `2005-01-16 14:00` (7,485 rows)
  - Test period: `2005-01-16 15:00` → `2005-04-04 14:00` (1,872 rows)
- **Cyclical hour encoding**: `Hour_sin = sin(2π × Hour / 24)`, `Hour_cos = cos(2π × Hour / 24)` — ensures Hour 23 is adjacent to Hour 0 on the unit circle.

#### Day 15 — Modelling & Data-Leakage Experiment
- Trained `RandomForestRegressor(n_estimators=100)` on both temporal and random splits.
- **Runtime Results**:

  | Split Strategy | Train R² | Test R² | Test RMSE | Test MAE | Honest? |
  |---|---|---|---|---|---|
  | Temporal (chronological) | 0.9999 | 0.9999 | 0.0458 | 0.0128 | ✅ Yes |
  | Random shuffle | 0.9999 | 0.9999 | 0.0927 | 0.0147 | ❌ No |

- **Note**: Both R² values are ~0.9999 because `PT08.S2(NMHC)` (the titania sensor) has a near-perfect chemical correlation with benzene — it alone accounts for 99.96% of feature importance. The leakage effect is visible in RMSE: the temporal model has **2× lower RMSE** (0.0458 vs 0.0927) than the shuffled model, meaning the honest temporal model actually generalises *better*.

- **Top Feature Importances** (temporal model):

  | Feature | Importance |
  |---|---|
  | PT08.S2(NMHC) | 0.9996 |
  | AH | 0.0001 |
  | Hour_cos | 0.0001 |
  | PT08.S3(NOx) | 0.0001 |
  | PT08.S1(CO) | 0.0001 |

- Generated 6 plots in `plots/`:
  - `week3_raw_benzene.png` — Raw benzene concentration time series
  - `week3_raw_sensors.png` — All sensor channels with -200 spikes
  - `week3_imputation_comparison.png` — Side-by-side imputation strategies
  - `week3_cyclical_hours.png` — Sin/cos encoding + unit circle visualisation
  - `week3_feature_importance.png` — RandomForest feature importances
  - `week3_actual_vs_predicted.png` — Temporal vs random split scatter plots

### 📝 Week 3 Discussion & Reflection

#### Discussion Questions

**Q: What does the `-200` value represent physically? Can a gas sensor ever genuinely read a negative concentration?**

**A:** No. A gas sensor measures resistance changes caused by gas molecules adsorbing onto a metal-oxide surface — this always produces a positive electrical signal. A concentration of `-200 μg/m³` is physically impossible. The value `-200` is a **manufacturer-designated error code** (sentinel value) indicating that the sensor was malfunctioning, uncalibrated, or reporting an out-of-range reading at that time. Treating these as real numbers would corrupt any analysis (e.g., pulling the mean downward, confusing regression models).

**Q: Why is the R² so high (~0.9999) for both splits?**

**A:** The `PT08.S2(NMHC)` sensor (a titania metal-oxide sensor targeting Non-Methane Hydrocarbons) has an extremely strong chemical cross-sensitivity to benzene. In urban air, benzene is a dominant NMHC component, so the sensor response is nearly a direct proxy for benzene concentration. This results in a near-perfect R² regardless of split strategy. However, the **RMSE** reveals the difference: the temporal split (0.0458) outperforms the shuffled split (0.0927) by 2×, showing that even with a dominant feature, honest temporal evaluation matters.

#### Reflection

**Your colleague runs `train_test_split(shuffle=True)` on sensor data and reports R² = 0.92. Your temporal split gives R² = 0.76. Who has the better model? Who has the better number?**

**Answer:** Your colleague has the **better number** — but you have the **better model**.

The R² = 0.92 from a random shuffle is *inflated by data leakage*. When time-series data is shuffled, the model sees observations from 2 PM and 4 PM in training, then is asked to predict 3 PM in testing. It learns to *interpolate* — trivially filling in gaps between known neighbours — rather than to *extrapolate* forward in time, which is what deployment requires.

Your R² = 0.76 from a chronological split is an **honest estimate** of how the model will perform on genuinely unseen future data. In production, the model will never have access to tomorrow's readings to help predict today's. The temporal split simulates exactly this constraint.

In practice:
- The shuffled model may crash to R² ≈ 0.76 (or worse) when deployed.
- Your model will perform close to R² ≈ 0.76 in deployment — no unpleasant surprises.

**The best model is the one whose offline evaluation matches its real-world performance.** A flattering number on a leaky split is worse than useless — it gives false confidence and can lead to costly deployment failures.

---

### 🚀 Week 4: Model Deployment & Integration (Placeholder)
- **Focus**: Packaging model pipelines, API creation, or integration.
- **Files**: *To be updated*
