"""
ML Assignment 2 - Model Training Script
Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: sklearn.datasets / UCI ML Repository
Features: 30 (mean, se, worst of radius, texture, perimeter, area, smoothness,
           compactness, concavity, concave_points, symmetry, fractal_dimension)
Instances: 569 | Classes: 2 (Malignant=1, Benign=0)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")   # 0 = malignant, 1 = benign

print(f"Dataset: Breast Cancer Wisconsin Diagnostic")
print(f"Shape  : {X.shape[0]} instances × {X.shape[1]} features")
print(f"Classes: {dict(zip(data.target_names, np.bincount(y)))}\n")

# ── 2. Train/Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler
os.makedirs("model", exist_ok=True)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save test data as CSV (used in Streamlit app)
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"test_data.csv saved  ({len(test_df)} rows)\n")

# ── 4. Define Models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
}

# ── 5. Train, Evaluate, Save ──────────────────────────────────────────────────
results = []

print(f"{'Model':<25} {'Acc':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred      = model.predict(X_test_scaled)
    y_prob      = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": name, "Accuracy": acc, "AUC": auc,
        "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
    })

    print(f"{name:<25} {acc:>6.3f} {auc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {mcc:>6.3f}")

    model_filename = name.lower().replace(" ", "_") + ".pkl"
    with open(f"model/{model_filename}", "wb") as f:
        pickle.dump(model, f)

# ── 6. Save Results Summary ───────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("model/results_summary.csv", index=False)

print("\nAll models saved to model/")
print("Results saved to model/results_summary.csv")
