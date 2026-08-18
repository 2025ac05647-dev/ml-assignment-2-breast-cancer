"""
ML Assignment 2 - Streamlit Application
Breast Cancer Classification - Interactive ML Dashboard
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Assignment 2 | Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "KNN":                 "knn.pkl",
    "Naive Bayes":         "naive_bayes.pkl",
    "Random Forest":       "random_forest.pkl",
    "SVM":                 "svm.pkl",
}
CLASS_NAMES = ["Malignant (0)", "Benign (1)"]

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Load a pickled object
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Compute all metrics for a single model
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/BITS_Pilani-Logo.svg/320px-BITS_Pilani-Logo.svg.png",
                 width=180)
st.sidebar.title("🔬 ML Assignment 2")
st.sidebar.markdown("**Breast Cancer Classification**")
st.sidebar.markdown("---")

# ── a. CSV Upload ─────────────────────────────────────────────────────────────
st.sidebar.subheader("📁 Upload Test Data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload your test_data.csv",
    type=["csv"],
    help="CSV must contain the 30 Breast Cancer features + a 'target' column (0/1)."
)

st.sidebar.markdown("---")

# ── b. Model Selection Dropdown ───────────────────────────────────────────────
st.sidebar.subheader("🤖 Select Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODEL_FILES.keys()),
    index=4,   # default: Random Forest
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: Breast Cancer Wisconsin Diagnostic  \n"
    "**Source**: UCI ML Repository  \n"
    "**Instances**: 569 | **Features**: 30  \n"
    "**Task**: Binary Classification"
)

# ──────────────────────────────────────────────────────────────────────────────
# Main Layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Classification - ML Dashboard")
st.markdown(
    "This app evaluates **6 classification models** on the "
    "Breast Cancer Wisconsin Diagnostic dataset.  \n"
    "Upload test data via the sidebar, select a model, and explore the results."
)

# ──────────────────────────────────────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
elif os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    st.info("ℹ️ Using bundled **test_data.csv**. You can upload your own via the sidebar.")
else:
    st.warning("⚠️ No test data found. Please upload a CSV file from the sidebar.")
    st.stop()

# Validate columns
if "target" not in df.columns:
    st.error("The CSV must have a **'target'** column (0 = Malignant, 1 = Benign).")
    st.stop()

X_test = df.drop(columns=["target"])
y_test  = df["target"]

# ──────────────────────────────────────────────────────────────────────────────
# Load Scaler & Scale Features
# ──────────────────────────────────────────────────────────────────────────────
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    st.error("Scaler not found. Please run `train_models.py` first.")
    st.stop()

scaler      = load_pickle(scaler_path)
X_test_sc   = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────────────────────
# Show Dataset Preview
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📊 Test Data Preview", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Malignant (0)", int((y_test == 0).sum()))
    col3.metric("Benign (1)",    int((y_test == 1).sum()))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Load Selected Model & Predict
# ──────────────────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model_name])
if not os.path.exists(model_path):
    st.error(f"Model file not found: `{model_path}`. Run `train_models.py` first.")
    st.stop()

model  = load_pickle(model_path)
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

metrics = compute_metrics(y_test, y_pred, y_prob)

# ──────────────────────────────────────────────────────────────────────────────
# c. Single Model Metrics Display
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"📈 Evaluation Metrics — {selected_model_name}")

cols = st.columns(6)
metric_labels = list(metrics.keys())
metric_values = list(metrics.values())

for i, col in enumerate(cols):
    col.metric(label=metric_labels[i], value=f"{metric_values[i]:.4f}")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# d. Confusion Matrix + Classification Report
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"🧩 Confusion Matrix & Classification Report — {selected_model_name}")

col_left, col_right = st.columns([1, 1.4])

with col_left:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Malignant", "Benign"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{selected_model_name}", fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col_right:
    st.markdown("**Classification Report**")
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Malignant", "Benign"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    # Drop 'support' from class rows for cleaner display
    st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=0),
                 use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# All Models Comparison Table
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🏆 All Models Comparison on Test Data")

all_results = []
for model_name, model_file in MODEL_FILES.items():
    mp = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(mp):
        continue
    m       = load_pickle(mp)
    yp      = m.predict(X_test_sc)
    yprob   = m.predict_proba(X_test_sc)[:, 1]
    row     = {"Model": model_name}
    row.update(compute_metrics(y_test, yp, yprob))
    all_results.append(row)

comparison_df = pd.DataFrame(all_results).set_index("Model")

# Highlight best value per metric
def highlight_best(col):
    if col.name == "MCC":
        is_best = col == col.max()
    else:
        is_best = col == col.max()
    return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_best]

styled = (
    comparison_df.style
    .apply(highlight_best, axis=0)
    .format("{:.4f}")
)
st.dataframe(styled, use_container_width=True)

# Bar chart comparison
st.markdown("**Visual Comparison of Metrics**")
chart_df = comparison_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
selected_metrics = st.multiselect(
    "Select metrics to visualize:",
    list(comparison_df.columns),
    default=["Accuracy", "AUC", "F1 Score"],
)
if selected_metrics:
    chart_data = comparison_df[selected_metrics]
    st.bar_chart(chart_data, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Model Observations
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📝 Model Observations")

observations = {
    "Logistic Regression": "Simple linear classifier. Performs well on this dataset due to the near-linear separability of classes. Fast to train and highly interpretable.",
    "Decision Tree":       "Captures non-linear boundaries. Prone to overfitting without pruning. Lower AUC than ensemble models.",
    "KNN":                 "Non-parametric distance-based model. Sensitive to feature scaling (handled by StandardScaler). Moderately good performance.",
    "Naive Bayes":         "Assumes feature independence. Despite this strong assumption, achieves reasonable accuracy on medical data with many correlated features.",
    "Random Forest":       "Best overall performer. Reduces overfitting through bagging and feature subsampling. High AUC indicates strong probability calibration.",
    "SVM":                 "Excellent for high-dimensional data. Finds optimal hyperplane margin. Competitive AUC with Random Forest.",
}

obs_df = pd.DataFrame([
    {"Model": k, "Observation": v}
    for k, v in observations.items()
])
st.dataframe(obs_df, use_container_width=True, hide_index=True)

# Winner
best_model = comparison_df["F1 Score"].idxmax()
st.success(f"🥇 **Overall Winner**: **{best_model}** (highest F1 Score on test data)")

st.markdown("---")
st.caption(
    "ML Assignment 2 | BITS Pilani WILP | "
    "Breast Cancer Wisconsin Diagnostic Dataset (UCI ML Repository)"
)
