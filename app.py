import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve,
)

st.set_page_config(
    page_title="BC-Classifier | ML Assignment 2",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT  = "#4F8EF7"
ACCENT2 = "#22C55E"
MUTED   = "#6B7280"
BG_CARD = "#F8FAFC"
BORDER  = "#E2E8F0"
FONT    = "'Inter', 'Segoe UI', sans-serif"

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "KNN":                 "knn.pkl",
    "Naive Bayes":         "naive_bayes.pkl",
    "Random Forest":       "random_forest.pkl",
    "SVM":                 "svm.pkl",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  section[data-testid="stSidebar"] {{
    background: #0F172A;
    border-right: 1px solid #1E293B;
  }}
  section[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{ color: #F1F5F9 !important; }}
  section[data-testid="stSidebar"] label {{
    color: #94A3B8 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
  }}

  .page-header {{
    padding: 1.25rem 0 0.25rem 0;
    border-bottom: 2px solid {BORDER};
    margin-bottom: 1.5rem;
  }}
  .page-header h1 {{
    font-size: 1.4rem; font-weight: 700; color: #0F172A;
    margin: 0; letter-spacing: -0.025em;
  }}
  .page-header p {{ color: {MUTED}; font-size: 0.83rem; margin: 0.2rem 0 0 0; }}

  .sec-label {{
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 0.5rem;
  }}

  .mcard {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 0.8rem 0.75rem; text-align: center;
  }}
  .mcard:hover {{ border-color: {ACCENT}; }}
  .mcard .mlabel {{
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {MUTED}; margin-bottom: 0.25rem;
  }}
  .mcard .mval {{
    font-size: 1.5rem; font-weight: 700; color: #0F172A;
    font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
  }}
  .mcard .msub {{ font-size: 0.68rem; color: {MUTED}; margin-top: 0.1rem; }}

  .winner {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px;
    padding: 0.45rem 0.875rem; font-size: 0.83rem; font-weight: 600;
    color: #15803D; margin-top: 0.75rem;
  }}

  .block-container {{ padding-top: 1rem !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }


def mcard(label, value, sub=""):
    v = f"{value:.4f}" if isinstance(value, float) else str(value)
    return (f"<div class='mcard'><div class='mlabel'>{label}</div>"
            f"<div class='mval'>{v}</div>"
            + (f"<div class='msub'>{sub}</div>" if sub else "")
            + "</div>")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### BC-Classifier")
    st.markdown("<span style='font-size:0.72rem;color:#64748B'>Breast Cancer · Binary Classification</span>",
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E293B;margin:0.75rem 0'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#475569;margin-bottom:0.4rem'>Test Data</p>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"],
                                     label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E293B;margin:0.75rem 0'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
                "text-transform:uppercase;color:#475569;margin-bottom:0.4rem'>Model</p>",
                unsafe_allow_html=True)
    selected_model_name = st.selectbox("Choose model", list(MODEL_FILES.keys()),
                                       index=4, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E293B;margin:0.75rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#475569;line-height:2'>
      <span style='color:#64748B'>Dataset</span><br>Breast Cancer Wisconsin (UCI)<br>
      <span style='color:#64748B'>Instances</span><br>569 &nbsp;·&nbsp; 80/20 split<br>
      <span style='color:#64748B'>Features</span><br>30 numerical<br>
      <span style='color:#64748B'>Task</span><br>Binary · Malignant / Benign
    </div>""", unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>Breast Cancer Classification</h1>
  <p>ML Assignment 2 &nbsp;·&nbsp; 6-model comparison &nbsp;·&nbsp;
     Accuracy · AUC · Precision · Recall · F1 · MCC</p>
</div>""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV — {e}")
        st.stop()
elif os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    st.caption("Using bundled test_data.csv  ·  Upload a different file via the sidebar to override.")
else:
    st.warning("No test data found. Upload a CSV file from the sidebar.")
    st.stop()

if "target" not in df.columns:
    st.error("CSV must contain a `target` column (0 = Malignant, 1 = Benign).")
    st.stop()

X_test = df.drop(columns=["target"])
y_test  = df["target"]

scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    st.error("scaler.pkl not found — run `train_models.py` first.")
    st.stop()

scaler    = load_pickle(scaler_path)
X_test_sc = scaler.transform(X_test)


# ── Dataset stats ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(mcard("Test samples", len(df)), unsafe_allow_html=True)
c2.markdown(mcard("Malignant", int((y_test == 0).sum()), "class 0"), unsafe_allow_html=True)
c3.markdown(mcard("Benign",    int((y_test == 1).sum()), "class 1"), unsafe_allow_html=True)
c4.markdown(mcard("Features",  X_test.shape[1]),         unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)


# ── Load selected model ───────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model_name])
if not os.path.exists(model_path):
    st.error(f"Model file missing: `{model_path}` — run train_models.py")
    st.stop()

model  = load_pickle(model_path)
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]
m      = compute_metrics(y_test, y_pred, y_prob)


# ── Metrics row ───────────────────────────────────────────────────────────────
st.markdown(f"<p class='sec-label'>{selected_model_name} — metrics</p>",
            unsafe_allow_html=True)
mc = st.columns(6)
for col, (label, val) in zip(mc, m.items()):
    col.markdown(mcard(label, val), unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ── Diagnostics tabs ──────────────────────────────────────────────────────────
st.markdown("<p class='sec-label'>Diagnostics</p>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "ROC Curve"])

with tab1:
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_facecolor("white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"],
                linewidths=0.5, linecolor="#E2E8F0",
                cbar=False, ax=ax, annot_kws={"size": 15, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=9, labelpad=8, color="#374151")
    ax.set_ylabel("Actual",    fontsize=9, labelpad=8, color="#374151")
    ax.set_title(selected_model_name, fontsize=10, fontweight="600", pad=12, color="#0F172A")
    ax.tick_params(labelsize=8, colors="#6B7280")
    for sp in ax.spines.values():
        sp.set_edgecolor("#E2E8F0")
    plt.tight_layout()

    left, _, right = st.columns([1.2, 0.15, 1])
    left.pyplot(fig, use_container_width=True)
    with right:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        for label, val, ok in [
            ("True Positive",  tp, True),
            ("True Negative",  tn, True),
            ("False Positive", fp, False),
            ("False Negative", fn, False),
        ]:
            color = "#15803D" if ok else "#B91C1C"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.4rem 0;border-bottom:1px solid #F1F5F9;font-size:0.83rem'>"
                f"<span style='color:#374151'>{label}</span>"
                f"<span style='font-weight:700;color:{color}'>{int(val)}</span></div>",
                unsafe_allow_html=True)

with tab2:
    report = classification_report(y_test, y_pred,
                                   target_names=["Malignant", "Benign"],
                                   output_dict=True)
    rdf = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore").round(4)
    st.dataframe(
        rdf.style
           .background_gradient(cmap="Blues", axis=0, vmin=0.5, vmax=1.0)
           .format("{:.4f}"),
        use_container_width=True)

with tab3:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("white")
    ax2.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {m['AUC']:.4f}")
    ax2.plot([0, 1], [0, 1], "--", color="#CBD5E1", lw=1)
    ax2.fill_between(fpr, tpr, alpha=0.06, color=ACCENT)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
    ax2.set_xlabel("False Positive Rate", fontsize=9, color="#374151")
    ax2.set_ylabel("True Positive Rate",  fontsize=9, color="#374151")
    ax2.set_title(f"ROC — {selected_model_name}", fontsize=10, fontweight="600", color="#0F172A")
    ax2.tick_params(labelsize=8, colors="#6B7280")
    ax2.legend(fontsize=9, framealpha=0.6)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#E2E8F0")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.35, color="#E2E8F0")
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ── All-model comparison ──────────────────────────────────────────────────────
st.markdown("<p class='sec-label'>All models — test set comparison</p>",
            unsafe_allow_html=True)

all_results = []
for name, fname in MODEL_FILES.items():
    mp = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(mp):
        continue
    mm    = load_pickle(mp)
    yp    = mm.predict(X_test_sc)
    yprob = mm.predict_proba(X_test_sc)[:, 1]
    row   = {"Model": name}
    row.update(compute_metrics(y_test, yp, yprob))
    all_results.append(row)

cmp_df  = pd.DataFrame(all_results).set_index("Model")
best_f1 = cmp_df["F1"].idxmax()


def style_cmp(df):
    s = df.style
    for col in df.columns:
        best_val = df[col].max()
        s = s.applymap(
            lambda v, bv=best_val: (
                "background-color:#F0FDF4;color:#15803D;font-weight:700" if v == bv else ""
            ),
            subset=[col],
        )
    return s.format("{:.4f}")


st.dataframe(style_cmp(cmp_df), use_container_width=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

chosen = st.multiselect("Metrics to plot", list(cmp_df.columns),
                        default=["Accuracy", "AUC", "F1"])

if chosen:
    models_list = list(cmp_df.index)
    x           = np.arange(len(models_list))
    bar_w       = 0.75 / len(chosen)
    palette     = [ACCENT, ACCENT2, "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6"]

    fig3, ax3 = plt.subplots(figsize=(10, 3.6))
    fig3.patch.set_facecolor("white")
    for i, metric in enumerate(chosen):
        vals = [cmp_df.loc[mn, metric] for mn in models_list]
        ax3.bar(x + i * bar_w, vals, width=bar_w * 0.88,
                color=palette[i % len(palette)], label=metric, zorder=3, alpha=0.9)
    ax3.set_xticks(x + bar_w * (len(chosen) - 1) / 2)
    ax3.set_xticklabels(models_list, fontsize=8.5, color="#374151")
    ax3.set_ylim(0, 1.09)
    ax3.set_ylabel("Score", fontsize=9, color="#374151")
    ax3.tick_params(axis="y", labelsize=8, colors="#6B7280")
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4, color="#E2E8F0", zorder=0)
    ax3.set_axisbelow(True)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax3.spines[sp].set_edgecolor("#E2E8F0")
    ax3.legend(fontsize=8.5, framealpha=0.6, loc="lower right")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

st.markdown(
    f"<div class='winner'>Best by F1 — {best_f1} ({cmp_df.loc[best_f1,'F1']:.4f})</div>",
    unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ── Observations ──────────────────────────────────────────────────────────────
with st.expander("Model observations", expanded=False):
    obs = {
        "Logistic Regression": "Near-optimal on this near-linearly-separable dataset. Fast, interpretable, strong baseline. Tied best AUC.",
        "Decision Tree":       "Lowest AUC — captures non-linear splits but overfits without pruning. Useful for rule extraction.",
        "KNN":                 "Instance-based; sensitive to scale (StandardScaler applied). Reasonable at k=5.",
        "Naive Bayes":         "Assumes feature independence (violated here) yet achieves useful recall. Fastest inference.",
        "Random Forest":       "Bagging reduces variance. Stable AUC. Good when interpretability is less critical.",
        "SVM":                 "Tied best with LR. Max-margin classifier excels in high-dimensional feature spaces.",
    }
    obs_df = pd.DataFrame({"Model": obs.keys(), "Observation": obs.values()})
    st.dataframe(obs_df, hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:#E2E8F0;margin-top:2rem'>"
    "<p style='font-size:0.7rem;color:#94A3B8;text-align:center;padding-bottom:0.5rem'>"
    "BITS Pilani WILP · NSP4 ML Assignment 2 · "
    "Breast Cancer Wisconsin Diagnostic · UCI ML Repository</p>",
    unsafe_allow_html=True)
