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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BC-Classifier | ML Assignment 2",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
ACCENT  = "#4F8EF7"
ACCENT2 = "#22C55E"
MUTED   = "#6B7280"
BG_CARD = "#F8FAFC"
BORDER  = "#E2E8F0"
FONT    = "'Inter', 'Segoe UI', sans-serif"
PALETTE = ["#4F8EF7", "#22C55E", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6"]

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


def style_cmp(df):
    """Highlight best-per-column. Uses .apply (pandas 2.1+ compatible — applymap removed)."""
    def _best(s):
        best = s.max()
        return ["background-color:#F0FDF4;color:#15803D;font-weight:700"
                if v == best else "" for v in s]
    return df.style.apply(_best, axis=0).format("{:.4f}")


def get_feature_importance(model, feature_names):
    """Return (top_names, top_importances) or None if not supported."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        elif hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "coef_"):
                imp = np.abs(base.coef_[0])
            else:
                return None
        else:
            return None
        n   = min(15, len(feature_names))
        idx = np.argsort(imp)[::-1][:n]
        return [feature_names[i] for i in idx], imp[idx]
    except Exception:
        return None


def radar_chart(metrics_dict, title=""):
    """Polar radar chart for a single model's metrics."""
    labels = list(metrics_dict.keys())
    # Normalise MCC from [-1,1] → [0,1] for display
    vals = [(v + 1) / 2 if k == "MCC" else v for k, v in metrics_dict.items()]
    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    vals  += [vals[0]]

    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")
    ax.plot(angles, vals, color=ACCENT, lw=2)
    ax.fill(angles, vals, color=ACCENT, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7, color="#374151")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], size=0)
    ax.grid(color="#E2E8F0", linestyle="--", alpha=0.6)
    ax.spines["polar"].set_edgecolor("#E2E8F0")
    if title:
        ax.set_title(title, size=8, fontweight="600", color="#0F172A", pad=10)
    plt.tight_layout()
    return fig


# ── Sidebar label / divider helpers ──────────────────────────────────────────
def _slbl(text):
    return ("<p style='font-size:0.68rem;font-weight:600;letter-spacing:0.1em;"
            f"text-transform:uppercase;color:#475569;margin-bottom:0.4rem'>{text}</p>")


_hr = "<hr style='border-color:#1E293B;margin:0.75rem 0'>"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### BC-Classifier")
    st.markdown("<span style='font-size:0.72rem;color:#64748B'>Breast Cancer · Binary Classification</span>",
                unsafe_allow_html=True)
    st.markdown(_hr, unsafe_allow_html=True)

    st.markdown(_slbl("Test Data"), unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

    st.markdown(_hr, unsafe_allow_html=True)
    st.markdown(_slbl("Model"), unsafe_allow_html=True)
    selected_model_name = st.selectbox("Model", list(MODEL_FILES.keys()),
                                       index=4, label_visibility="collapsed")

    st.markdown(_hr, unsafe_allow_html=True)
    st.markdown(_slbl("Decision Threshold"), unsafe_allow_html=True)
    threshold = st.slider(
        "Threshold", min_value=0.10, max_value=0.90,
        value=0.50, step=0.01, label_visibility="collapsed",
        help="Probability cut-off for Benign class",
    )
    st.markdown(
        f"<p style='font-size:0.72rem;color:#64748B;margin-top:-0.4rem'>"
        f"p &ge; <b style='color:#94A3B8'>{threshold:.2f}</b> &rarr; Benign</p>",
        unsafe_allow_html=True)

    st.markdown(_hr, unsafe_allow_html=True)
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
feature_names = list(X_test.columns)


# ── Dataset stats strip ───────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(mcard("Test samples", len(df)),                               unsafe_allow_html=True)
c2.markdown(mcard("Malignant",    int((y_test == 0).sum()), "class 0"),   unsafe_allow_html=True)
c3.markdown(mcard("Benign",       int((y_test == 1).sum()), "class 1"),   unsafe_allow_html=True)
c4.markdown(mcard("Features",     X_test.shape[1]),                       unsafe_allow_html=True)
st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)


# ── Load selected model & predict (threshold-aware) ───────────────────────────
model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model_name])
if not os.path.exists(model_path):
    st.error(f"Model file missing: `{model_path}` — run train_models.py")
    st.stop()

model  = load_pickle(model_path)
y_prob = model.predict_proba(X_test_sc)[:, 1]
y_pred = (y_prob >= threshold).astype(int)
m      = compute_metrics(y_test, y_pred, y_prob)


# ── Metrics row ───────────────────────────────────────────────────────────────
st.markdown(
    f"<p class='sec-label'>{selected_model_name} — metrics &nbsp;·&nbsp; threshold {threshold:.2f}</p>",
    unsafe_allow_html=True)
mc = st.columns(6)
for col, (label, val) in zip(mc, m.items()):
    col.markdown(mcard(label, val), unsafe_allow_html=True)
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_diag, tab_cmp, tab_imp, tab_pred = st.tabs([
    "Diagnostics", "Model Comparison", "Feature Importance", "Live Predictor",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Diagnostics
# ──────────────────────────────────────────────────────────────────────────────
with tab_diag:
    dtab1, dtab2, dtab3, dtab4 = st.tabs([
        "Confusion Matrix", "Classification Report", "ROC Curve", "All-Model ROC",
    ])

    with dtab1:
        cm_arr         = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm_arr.ravel()

        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor("white")
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"],
                    linewidths=0.5, linecolor="#E2E8F0",
                    cbar=False, ax=ax, annot_kws={"size": 15, "weight": "bold"})
        ax.set_xlabel("Predicted", fontsize=9, labelpad=8, color="#374151")
        ax.set_ylabel("Actual",    fontsize=9, labelpad=8, color="#374151")
        ax.set_title(f"{selected_model_name}  (threshold = {threshold:.2f})",
                     fontsize=10, fontweight="600", pad=12, color="#0F172A")
        ax.tick_params(labelsize=8, colors="#6B7280")
        for sp in ax.spines.values():
            sp.set_edgecolor("#E2E8F0")
        plt.tight_layout()

        left, _, right = st.columns([1.2, 0.15, 1])
        left.pyplot(fig, use_container_width=True)
        with right:
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            # TP / TN / FP / FN
            for lbl, val, ok in [
                ("True Positive",  tp, True),
                ("True Negative",  tn, True),
                ("False Positive", fp, False),
                ("False Negative", fn, False),
            ]:
                color = "#15803D" if ok else "#B91C1C"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:0.4rem 0;border-bottom:1px solid #F1F5F9;font-size:0.83rem'>"
                    f"<span style='color:#374151'>{lbl}</span>"
                    f"<span style='font-weight:700;color:{color}'>{int(val)}</span></div>",
                    unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            # Derived clinical stats
            for lbl, val in [
                ("Sensitivity",  f"{tp/(tp+fn):.4f}" if (tp+fn) > 0 else "—"),
                ("Specificity",  f"{tn/(tn+fp):.4f}" if (tn+fp) > 0 else "—"),
                ("PPV (Precision)", f"{tp/(tp+fp):.4f}" if (tp+fp) > 0 else "—"),
                ("NPV",           f"{tn/(tn+fn):.4f}" if (tn+fn) > 0 else "—"),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:0.3rem 0;font-size:0.77rem;color:#64748B'>"
                    f"<span>{lbl}</span>"
                    f"<span style='font-weight:600;color:#374151'>{val}</span></div>",
                    unsafe_allow_html=True)

    with dtab2:
        report      = classification_report(y_test, y_pred,
                                            target_names=["Malignant", "Benign"],
                                            output_dict=True)
        # Drop scalar 'accuracy' key before building DataFrame
        rdf_rows    = {k: v for k, v in report.items() if isinstance(v, dict)}
        rdf         = pd.DataFrame(rdf_rows).T.drop(columns=["support"], errors="ignore").round(4)
        st.dataframe(
            rdf.style
               .background_gradient(cmap="Blues", axis=None, vmin=0.5, vmax=1.0)
               .format("{:.4f}"),
            use_container_width=True)

    with dtab3:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2   = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("white")
        ax2.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC = {m['AUC']:.4f}")
        ax2.plot([0, 1], [0, 1], "--", color="#CBD5E1", lw=1)
        ax2.fill_between(fpr, tpr, alpha=0.07, color=ACCENT)
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

    with dtab4:
        # All 6 ROC curves overlaid
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        fig_roc.patch.set_facecolor("white")
        for i, (nm, fn) in enumerate(MODEL_FILES.items()):
            mp = os.path.join(MODEL_DIR, fn)
            if not os.path.exists(mp):
                continue
            mm       = load_pickle(mp)
            yprob_i  = mm.predict_proba(X_test_sc)[:, 1]
            auc_i    = roc_auc_score(y_test, yprob_i)
            fpr_i, tpr_i, _ = roc_curve(y_test, yprob_i)
            is_sel   = nm == selected_model_name
            ax_roc.plot(fpr_i, tpr_i,
                        color=PALETTE[i], lw=2.5 if is_sel else 1.5,
                        linestyle="-" if is_sel else "--",
                        label=f"{nm}  ({auc_i:.3f})")
        ax_roc.plot([0, 1], [0, 1], ":", color="#CBD5E1", lw=1)
        ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
        ax_roc.set_xlabel("False Positive Rate", fontsize=9, color="#374151")
        ax_roc.set_ylabel("True Positive Rate",  fontsize=9, color="#374151")
        ax_roc.set_title("All-Model ROC Curves", fontsize=10, fontweight="600", color="#0F172A")
        ax_roc.tick_params(labelsize=8, colors="#6B7280")
        ax_roc.legend(fontsize=8, framealpha=0.7, loc="lower right")
        ax_roc.yaxis.grid(True, linestyle="--", alpha=0.3, color="#E2E8F0")
        ax_roc.set_axisbelow(True)
        ax_roc.spines["top"].set_visible(False)
        ax_roc.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax_roc.spines[sp].set_edgecolor("#E2E8F0")
        fig_roc.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)
        st.caption(f"Solid line = {selected_model_name}  ·  dashed = others")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Comparison
# ──────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    all_results = []
    for name, fname in MODEL_FILES.items():
        mp = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(mp):
            continue
        mm    = load_pickle(mp)
        yprob = mm.predict_proba(X_test_sc)[:, 1]
        yp    = (yprob >= threshold).astype(int)
        row   = {"Model": name}
        row.update(compute_metrics(y_test, yp, yprob))
        all_results.append(row)

    cmp_df  = pd.DataFrame(all_results).set_index("Model")
    best_f1 = cmp_df["F1"].idxmax()

    st.markdown(f"<p class='sec-label'>All models — threshold {threshold:.2f}</p>",
                unsafe_allow_html=True)
    st.dataframe(style_cmp(cmp_df), use_container_width=True)
    st.markdown(
        f"<div class='winner'>Best by F1 — {best_f1} ({cmp_df.loc[best_f1,'F1']:.4f})</div>",
        unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # Grouped bar chart
    st.markdown("<p class='sec-label'>Grouped bar — select metrics</p>", unsafe_allow_html=True)
    chosen = st.multiselect("Metrics", list(cmp_df.columns),
                            default=["Accuracy", "AUC", "F1"], key="bar_metrics")
    if chosen:
        models_list = list(cmp_df.index)
        x     = np.arange(len(models_list))
        bar_w = 0.75 / len(chosen)
        fig3, ax3 = plt.subplots(figsize=(10, 3.6))
        fig3.patch.set_facecolor("white")
        for i, metric in enumerate(chosen):
            vals = [cmp_df.loc[mn, metric] for mn in models_list]
            ax3.bar(x + i * bar_w, vals, width=bar_w * 0.88,
                    color=PALETTE[i % len(PALETTE)], label=metric, zorder=3, alpha=0.9)
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

    # Radar charts — one per model
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<p class='sec-label'>Metric radar — per model</p>", unsafe_allow_html=True)
    radar_cols = st.columns(3)
    for i, (model_nm, row) in enumerate(cmp_df.iterrows()):
        fig_r = radar_chart(row.to_dict(), title=model_nm)
        radar_cols[i % 3].pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    # Observations expander
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.expander("Model observations", expanded=False):
        obs = {
            "Logistic Regression": "Near-optimal on this near-linearly-separable dataset. Fast, interpretable, strong baseline.",
            "Decision Tree":       "Lowest AUC — captures non-linear splits but overfits without pruning.",
            "KNN":                 "Instance-based; sensitive to scale (StandardScaler applied). Reasonable at k=5.",
            "Naive Bayes":         "Assumes feature independence (violated here) yet achieves useful recall.",
            "Random Forest":       "Bagging reduces variance. Stable AUC. Good when interpretability is less critical.",
            "SVM":                 "Max-margin classifier. Excels in high-dimensional feature spaces. Competitive AUC.",
        }
        obs_df = pd.DataFrame({"Model": obs.keys(), "Observation": obs.values()})
        st.dataframe(obs_df, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
with tab_imp:
    st.markdown(f"<p class='sec-label'>{selected_model_name} — feature importance</p>",
                unsafe_allow_html=True)
    imp_result = get_feature_importance(model, feature_names)
    if imp_result is None:
        st.info(
            f"**{selected_model_name}** does not expose direct feature importances.\n\n"
            "Switch to **Random Forest**, **Decision Tree**, **Logistic Regression**, "
            "or **SVM** to see feature rankings."
        )
    else:
        top_names, top_imp = imp_result
        imp_df = pd.DataFrame({"Feature": top_names, "Importance": top_imp})

        fig4, ax4 = plt.subplots(figsize=(8, max(4, len(top_names) * 0.38)))
        fig4.patch.set_facecolor("white")
        bar_colors = [ACCENT if i == 0 else "#93C5FD" for i in range(len(top_names))]
        bars = ax4.barh(top_names[::-1], top_imp[::-1], color=bar_colors[::-1], alpha=0.9)
        for bar, val in zip(bars, top_imp[::-1]):
            ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=7.5, color="#374151")
        ax4.set_xlabel("Importance Score", fontsize=9, color="#374151")
        ax4.set_title(f"Top {len(top_names)} Features — {selected_model_name}",
                      fontsize=10, fontweight="600", color="#0F172A")
        ax4.tick_params(labelsize=8, colors="#6B7280")
        ax4.xaxis.grid(True, linestyle="--", alpha=0.4, color="#E2E8F0", zorder=0)
        ax4.set_axisbelow(True)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax4.spines[sp].set_edgecolor("#E2E8F0")
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)

        st.dataframe(
            imp_df.style
                  .bar(subset=["Importance"], color="#93C5FD")
                  .format({"Importance": "{:.5f}"}),
            use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Live Predictor
# ──────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("<p class='sec-label'>Pick a test sample — get a live prediction from all models</p>",
                unsafe_allow_html=True)

    # Session-state-backed index so "Random" button works
    if "live_idx" not in st.session_state:
        st.session_state.live_idx = 0

    ctrl_col, _, res_col = st.columns([1, 0.1, 1.5])

    with ctrl_col:
        if st.button("Pick random sample"):
            st.session_state.live_idx = int(np.random.randint(0, len(df)))

        sample_idx = st.number_input(
            f"Sample index  (0 – {len(df) - 1})",
            min_value=0, max_value=len(df) - 1,
            value=int(st.session_state.live_idx), step=1,
        )
        st.session_state.live_idx = int(sample_idx)

        true_label = int(y_test.iloc[sample_idx])
        lstr       = "Benign (1)" if true_label == 1 else "Malignant (0)"
        lcol       = "#15803D"    if true_label == 1 else "#B91C1C"
        st.markdown(
            f"<p style='font-size:0.78rem;color:{MUTED};margin-top:0.5rem'>"
            f"True label: <span style='font-weight:700;color:{lcol}'>{lstr}</span></p>",
            unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem;font-weight:600;color:#475569;"
            "text-transform:uppercase;letter-spacing:0.08em'>All-model votes</p>",
            unsafe_allow_html=True)

        sample_sc = X_test_sc[sample_idx].reshape(1, -1)
        for nm, fn in MODEL_FILES.items():
            mp = os.path.join(MODEL_DIR, fn)
            if not os.path.exists(mp):
                continue
            mm      = load_pickle(mp)
            prob_v  = mm.predict_proba(sample_sc)[0, 1]
            pred_v  = "Benign" if prob_v >= threshold else "Malignant"
            correct = (1 if prob_v >= threshold else 0) == true_label
            c_icon  = "✓" if correct else "✗"
            c_color = "#15803D" if correct else "#B91C1C"
            bold    = "font-weight:700;" if nm == selected_model_name else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.3rem 0;border-bottom:1px solid #F8FAFC;font-size:0.78rem;{bold}'>"
                f"<span style='color:#374151'>{nm}</span>"
                f"<span style='color:{c_color}'>{c_icon}&nbsp;{pred_v} ({prob_v:.2f})</span></div>",
                unsafe_allow_html=True)

    with res_col:
        prob_sel = float(model.predict_proba(sample_sc)[0, 1])
        pred_sel = "Benign"    if prob_sel >= threshold else "Malignant"
        pred_col = "#15803D"   if pred_sel == "Benign"  else "#B91C1C"
        pred_bg  = "#F0FDF4"   if pred_sel == "Benign"  else "#FFF1F2"
        pred_bdr = "#86EFAC"   if pred_sel == "Benign"  else "#FECDD3"
        conf_pct = max(prob_sel, 1 - prob_sel)

        st.markdown(
            f"<div style='background:{pred_bg};border:1.5px solid {pred_bdr};"
            f"border-radius:10px;padding:1rem 1.5rem;text-align:center;margin-bottom:0.75rem'>"
            f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.1em;"
            f"text-transform:uppercase;color:{MUTED}'>Prediction — {selected_model_name}</div>"
            f"<div style='font-size:2.2rem;font-weight:800;color:{pred_col};margin:0.4rem 0'>{pred_sel}</div>"
            f"<div style='font-size:0.83rem;color:{MUTED}'>Confidence: "
            f"<span style='font-weight:700;color:{pred_col}'>{conf_pct:.1%}</span></div>"
            f"</div>",
            unsafe_allow_html=True)

        benign_pct = prob_sel * 100
        malig_pct  = (1 - prob_sel) * 100
        st.markdown(
            f"<div style='font-size:0.72rem;color:{MUTED};margin-bottom:0.25rem'>"
            f"Malignant &nbsp;<b>{malig_pct:.1f}%</b> &nbsp;|&nbsp; "
            f"Benign &nbsp;<b>{benign_pct:.1f}%</b></div>",
            unsafe_allow_html=True)
        st.progress(float(prob_sel))

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.72rem;font-weight:600;color:#475569;"
            "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem'>"
            "Sample feature values</p>",
            unsafe_allow_html=True)
        samp_df = (
            pd.DataFrame(X_test.iloc[sample_idx].values.reshape(1, -1),
                         columns=feature_names)
            .T.rename(columns={0: "Value"})
        )
        samp_df["Value"] = samp_df["Value"].round(4)
        st.dataframe(samp_df, use_container_width=True, height=300)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:#E2E8F0;margin-top:2rem'>"
    "<p style='font-size:0.7rem;color:#94A3B8;text-align:center;padding-bottom:0.5rem'>"
    "BITS Pilani WILP · NSP4 ML Assignment 2 · "
    "Breast Cancer Wisconsin Diagnostic · UCI ML Repository</p>",
    unsafe_allow_html=True)
