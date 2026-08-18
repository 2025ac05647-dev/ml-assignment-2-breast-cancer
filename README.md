# ML Assignment 2 — Breast Cancer Classification

## Problem Statement
Build and evaluate multiple machine learning classification models to predict whether a breast tumour is **malignant (0)** or **benign (1)** using the Wisconsin Diagnostic Breast Cancer dataset. Deploy an interactive Streamlit web application that allows users to upload test data, select a model, and view evaluation metrics and visual comparisons.

---

## Dataset Description
| Property | Value |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| **Instances** | 569 |
| **Features** | 30 (numerical) |
| **Target Classes** | 2 — Malignant (0), Benign (1) |
| **Class Distribution** | 212 Malignant / 357 Benign |
| **Task** | Binary Classification |

**Features**: Mean, standard error, and worst values of 10 cell-nucleus measurements:
*radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension*

---

## GitHub Repository Link
> **GitHub:** https://github.com/2025ac05647-dev/ml-assignment-2-breast-cancer
>
> **Streamlit App (Live):** https://ml-assignment-2-breast-cancer-kfwjgwfu5x3ze2ifwrmrsu.streamlit.app/

---

## Models Used

### Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9952 | 0.9859 | 0.9861 | 0.9860 | 0.9621 |
| Decision Tree | 0.9123 | 0.9162 | 0.9556 | 0.9028 | 0.9285 | 0.8168 |
| KNN | 0.9561 | 0.9791 | 0.9589 | 0.9722 | 0.9655 | 0.9050 |
| Naive Bayes | 0.9298 | 0.9872 | 0.9444 | 0.9444 | 0.9444 | 0.8494 |
| Random Forest | 0.9561 | 0.9942 | 0.9589 | 0.9722 | 0.9655 | 0.9050 |
| SVM | 0.9825 | 0.9952 | 0.9859 | 0.9861 | 0.9860 | 0.9621 |

*(Actual values are shown live in the Streamlit app after uploading test_data.csv)*

---

## Model Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Simple linear classifier. Performs well due to near-linear separability. Fast, interpretable. |
| Decision Tree | Captures non-linear boundaries but prone to overfitting. Lower AUC vs ensembles. |
| KNN | Distance-based, sensitive to scaling (handled via StandardScaler). Moderate performance. |
| Naive Bayes | Assumes feature independence. Reasonable accuracy despite correlated features in this dataset. |
| Random Forest | Best performer — bagging reduces overfitting. High AUC shows strong probability calibration. |
| SVM | Excellent for high-dimensional data. Finds optimal margin hyperplane. Competitive with Random Forest. |

**Overall Winner: Logistic Regression / SVM** (tied highest Accuracy 98.25%, AUC 99.52%, F1 98.60%)

---

## Project Structure

```
ml_assignment2/
├── app.py                # Streamlit application
├── train_models.py       # Model training & saving script
├── requirements.txt      # Python dependencies
├── test_data.csv         # Test split used for evaluation
├── README.md             # This file
└── model/
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── svm.pkl
```

---

## Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/2025ac05647-dev/ml-assignment-2-breast-cancer.git
cd ml-assignment-2-breast-cancer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models (generates model/*.pkl and test_data.csv)
python train_models.py

# 5. Launch Streamlit app
streamlit run app.py
```

---

## Streamlit App Features
- **CSV Upload**: Upload test data via sidebar; falls back to bundled `test_data.csv`
- **Model Selection**: Dropdown to choose from 6 classification models
- **Decision Threshold Slider**: Adjust probability cut-off (0.10–0.90) — all metrics update live
- **Evaluation Metrics**: Accuracy, AUC, Precision, Recall, F1, MCC displayed as metric cards
- **Diagnostics tab**:
  - Confusion Matrix with TP/TN/FP/FN + Sensitivity, Specificity, PPV, NPV
  - Classification Report (per-class precision, recall, F1)
  - ROC Curve with AUC for selected model
  - All-Model ROC Overlay — all 6 curves on one chart
- **Model Comparison tab**: Comparison table (best cells highlighted) + grouped bar chart + radar charts per model
- **Feature Importance tab**: Top-15 feature importance bar chart (RF, DT, LR, SVM)
- **Live Predictor tab**: Pick any test sample, see all 6 models vote with ✓/✗, confidence gauge + probability bar

---

