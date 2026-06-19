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
| Need built-in L1/L2 regularisation | **XGBoost** |
| Competition / structured data benchmark tasks | **XGBoost** (or LightGBM) |

---

### 🛠️ Week 3: Feature Engineering & Selection (Placeholder)
- **Focus**: Feature creation, handling missing values, selecting important features.
- **Files**: *To be updated*

---

### 🚀 Week 4: Model Deployment & Integration (Placeholder)
- **Focus**: Packaging model pipelines, API creation, or integration.
- **Files**: *To be updated*
