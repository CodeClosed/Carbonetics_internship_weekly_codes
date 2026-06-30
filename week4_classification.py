"""
Week 4: Classification & Class Imbalance — AI4I Predictive Maintenance
======================================================================
  Day 16 -- Load dataset & EDA (class distribution, feature boxplots)
  Day 17 -- Why accuracy is misleading (trivial baseline, metric defs)
  Day 18 -- Addressing class imbalance (no adjustment, class_weight, SMOTE)
  Day 19 -- Stratified cross-validation
  Day 20 -- Final deliverable (confusion matrix, ROC curve, summary)

Run:
    python week4_classification.py

Dependencies:
    scikit-learn, pandas, numpy, matplotlib, imbalanced-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
)
from imblearn.over_sampling import SMOTE

os.makedirs("plots", exist_ok=True)

# ============================================================================
#  DAY 16: Load Dataset & EDA
# ============================================================================

print("=" * 65)
print("  DAY 16 -- Load Dataset & EDA")
print("=" * 65)

df = pd.read_csv("ai4i2020.csv")
print(f"\n  Shape: {df.shape}")
print(f"\n  Null counts:\n{df.isnull().sum()}")

# Class distribution
print("\n  Class distribution (absolute):")
print(df['Machine failure'].value_counts())
print("\n  Class distribution (normalised):")
print(df['Machine failure'].value_counts(normalize=True))

# Feature boxplots split by failure label
features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, feat in zip(axes, features):
    df.boxplot(column=feat, by='Machine failure', ax=ax)
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel('Machine failure')
plt.suptitle("Feature Distributions by Failure Label", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("plots/week4_feature_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  -> Saved: plots/week4_feature_boxplots.png")

# ============================================================================
#  DAY 17: Why Accuracy Is a Misleading Metric
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 17 -- Why Accuracy Is a Misleading Metric")
print("=" * 65)

X = df[features]
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"  Train failure rate: {y_train.mean():.4f}")
print(f"  Test  failure rate: {y_test.mean():.4f}")

# Trivial baseline: always predict 0 (No Failure)
y_pred_trivial = np.zeros(len(y_test), dtype=int)
trivial_acc = accuracy_score(y_test, y_pred_trivial)

print(f"\n  Trivial baseline (always predict 'No Failure'):")
print(f"  Accuracy: {trivial_acc:.3f}")
print("\n  Classification report:")
print(classification_report(
    y_test, y_pred_trivial,
    target_names=['No Failure', 'Failure'],
    zero_division=0,
))

# ============================================================================
#  DAY 18: Addressing Class Imbalance — Three Strategies
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 18 -- Addressing Class Imbalance — Three Strategies")
print("=" * 65)

# ---------- Strategy 1: No adjustment (baseline) ----------
print("\n--- Strategy 1: RandomForest — No adjustment (baseline) ---")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)
print(classification_report(
    y_test, y_pred_baseline,
    target_names=['No Failure', 'Failure'],
    zero_division=0,
))

# ---------- Strategy 2: class_weight='balanced' ----------
print("--- Strategy 2: RandomForest — class_weight='balanced' ---")
rf_weighted = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42
)
rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)
print(classification_report(
    y_test, y_pred_weighted,
    target_names=['No Failure', 'Failure'],
    zero_division=0,
))

# ---------- Strategy 3: SMOTE oversampling ----------
print("--- Strategy 3: SMOTE + RandomForest ---")
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"  After  SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}\n")

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = rf_smote.predict(X_test)
print(classification_report(
    y_test, y_pred_smote,
    target_names=['No Failure', 'Failure'],
    zero_division=0,
))

# ---------- Comparative bar chart ----------
from sklearn.metrics import precision_recall_fscore_support

strategies = ['Baseline', 'class_weight', 'SMOTE']
preds_list = [y_pred_baseline, y_pred_weighted, y_pred_smote]

precisions, recalls, f1s = [], [], []
for preds in preds_list:
    p, r, f, _ = precision_recall_fscore_support(
        y_test, preds, pos_label=1, average='binary', zero_division=0
    )
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

x = np.arange(len(strategies))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#4C72B0')
bars2 = ax.bar(x,         recalls,    width, label='Recall',    color='#DD8452')
bars3 = ax.bar(x + width, f1s,        width, label='F1-Score',  color='#55A868')

ax.set_ylabel('Score')
ax.set_title('Failure-Class Metrics by Imbalance Strategy')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylim(0, 1.05)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("plots/week4_strategy_comparison.png", dpi=150)
plt.close()
print("  -> Saved: plots/week4_strategy_comparison.png")

# Strategy comparison summary
print("\n  Failure-class summary:")
print(f"  {'Strategy':<15}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, p, r, f in zip(strategies, precisions, recalls, f1s):
    print(f"  {name:<15}  {p:>10.3f}  {r:>10.3f}  {f:>10.3f}")

# ============================================================================
#  DAY 19: Stratified Cross-Validation
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 19 -- Stratified Cross-Validation")
print("=" * 65)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42
)

f1_scores = cross_val_score(rf_cv, X, y, cv=skf, scoring='f1')
print(f"  5-Fold Stratified CV — F1 (Failure class):")
for i, score in enumerate(f1_scores, 1):
    print(f"    Fold {i}: {score:.3f}")
print(f"\n  Mean F1: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# ============================================================================
#  DAY 20: Final Deliverable — Confusion Matrix, ROC Curve, Summary
# ============================================================================

print("\n" + "=" * 65)
print("  DAY 20 -- Final Deliverable")
print("=" * 65)

# Select the best model based on Day 18 results
# (Using class_weight='balanced' as the best overall strategy)
best_model = rf_weighted
best_name = "RandomForest (class_weight='balanced')"

print(f"\n  Best model: {best_name}")
print(f"\n  Final classification report on held-out test set:")
print(classification_report(
    y_test, best_model.predict(X_test),
    target_names=['No Failure', 'Failure'],
))

# --- Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=['No Failure', 'Failure'],
    cmap='Blues',
    ax=ax,
)
ax.set_title("Confusion Matrix — Predictive Maintenance Classifier")
plt.tight_layout()
plt.savefig("plots/week4_confusion_matrix.png", dpi=150)
plt.close()
print("  -> Saved: plots/week4_confusion_matrix.png")

# --- ROC Curve ---
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(
    best_model, X_test, y_test,
    name=best_name,
    ax=ax,
)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random chance')
ax.set_title("ROC Curve — Predictive Maintenance Classifier")
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("plots/week4_roc_curve.png", dpi=150)
plt.close()
print("  -> Saved: plots/week4_roc_curve.png")

# --- Final Summary ---
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)

models = {
    'Trivial (all 0)':      y_pred_trivial,
    'RF Baseline':          y_pred_baseline,
    'RF class_weight':      y_pred_weighted,
    'RF + SMOTE':           y_pred_smote,
}

print(f"\n  {'Model':<20}  {'Accuracy':>10}  {'F1 (Fail)':>10}  {'Recall (Fail)':>14}  {'Precision (Fail)':>16}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*16}")

for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, preds, pos_label=1, average='binary', zero_division=0
    )
    print(f"  {name:<20}  {acc:>10.3f}  {f:>10.3f}  {r:>14.3f}  {p:>16.3f}")


print("Week 4 script completed successfully.")
