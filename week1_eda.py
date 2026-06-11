import os
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# 1. Load California Housing
print("--- Section 1: Load California Housing ---")
data = fetch_california_housing(as_frame=True)
df = data.frame
print(f"Dataset Shape: {df.shape}")
print("\nDataset Describe:")
print(df.describe())

# 2. Exploratory Data Analysis (distributions, correlation)
print("\n--- Section 2: Exploratory Data Analysis ---")
# Plot feature distributions
print("Plotting feature distributions...")
df.hist(bins=30, figsize=(12, 8))
plt.suptitle("Feature Distributions — California Housing")
plt.tight_layout()
plt.savefig("plots/feature_distributions.png")
plt.close()

# Plot target variable distribution
print("Plotting target distribution...")
plt.figure(figsize=(8, 5))
df['MedHouseVal'].hist(bins=50)
plt.title("Target Distribution: Median House Value")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()

# Create correlation heatmap
print("Plotting correlation heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# 3. Train/Test Split
print("\n--- Section 3: Train/Test Split ---")
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

# 4. Feature Scaling with StandardScaler
print("\n--- Section 4: Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std from training data
X_test_scaled  = scaler.transform(X_test)        # Apply learned stats — do NOT fit again
print("Feature scaling completed successfully.")
