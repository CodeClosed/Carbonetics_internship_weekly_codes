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

### 📈 Week 2: Model Training & Evaluation (Placeholder)
- **Focus**: Implementing baseline models, hyperparameter tuning, and metric evaluation.
- **Files**: *To be updated*

---

### 🛠️ Week 3: Feature Engineering & Selection (Placeholder)
- **Focus**: Feature creation, handling missing values, selecting important features.
- **Files**: *To be updated*

---

### 🚀 Week 4: Model Deployment & Integration (Placeholder)
- **Focus**: Packaging model pipelines, API creation, or integration.
- **Files**: *To be updated*
