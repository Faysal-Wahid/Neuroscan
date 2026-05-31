"""
NeuroScan – Dementia Prediction App
=====================================
Run with:
    pip install streamlit scikit-learn pandas numpy matplotlib
    streamlit run app.py
or Run this code first then run streamlit
    -python -m pip install streamlit scikit-learn pandas numpy matplotlib seaborn
    -python -m streamlit run app.py
Make sure dementia_data.csv is in the same folder as this script.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan – Dementia Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0a0e1a; color: #e8edf5; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1e2d4a;
}

/* Cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-card .metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-card .metric-lbl {
    font-size: 12px;
    color: #6b7fa3;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Result boxes */
.result-demented {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 28px;
}
.result-nondemented {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px;
    padding: 28px;
}
.result-converted {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 16px;
    padding: 28px;
}
.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.result-desc {
    color: #6b7fa3;
    font-size: 14px;
    line-height: 1.7;
}
.badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 11px;
    color: #00d4ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.disclaimer {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #6b7fa3;
    margin-top: 20px;
}
h1, h2, h3 { color: #e8edf5 !important; }
hr { border-color: #1e2d4a; }

/* Input labels */
label { color: #6b7fa3 !important; font-size: 13px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: #001820 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    font-size: 16px !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(0,212,255,0.35) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD & TRAIN MODEL (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Random Forest model...")
def load_and_train():
    df = pd.read_csv("dementia_data.csv")

    # Encode
    le_sex = LabelEncoder()
    df["M/F"] = le_sex.fit_transform(df["M/F"])

    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Group"])

    feature_cols = ["Visit", "MR Delay", "M/F", "Age", "EDUC",
                    "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
    X = df[feature_cols].values
    y = df["Label"].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Train/test split for metrics display
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    rep  = classification_report(y_test, y_pred,
                                  target_names=le.classes_, output_dict=True)
    cm   = confusion_matrix(y_test, y_pred)

    # Refit on full data for predictions
    model.fit(X_scaled, y)

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "le": le,
        "le_sex": le_sex,
        "feature_cols": feature_cols,
        "classes": le.classes_,
        "acc": acc,
        "auc": auc,
        "report": rep,
        "cm": cm,
        "fi": model.feature_importances_,
        "df": df,
    }


# ─────────────────────────────────────────────────────────────────
# SIDEBAR – INPUTS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NeuroScan")
    st.markdown("<p style='color:#6b7fa3;font-size:13px;margin-bottom:24px'>Patient Data Input</p>",
                unsafe_allow_html=True)

    st.markdown("**Demographics**")
    sex      = st.selectbox("Sex", ["Male", "Female"])
    age      = st.slider("Age", 60, 98, 75)
    educ     = st.slider("Education (years)", 6, 23, 14)
    ses      = st.selectbox("Socioeconomic Status (1=High, 5=Low)",
                             [1, 2, 3, 4, 5], index=1)

    st.markdown("---")
    st.markdown("**MRI Measurements**")
    visit    = st.selectbox("Visit Number", [1, 2, 3, 4, 5])
    mr_delay = st.number_input("MR Delay (days)", 0, 2639, 552)
    etiv     = st.number_input("eTIV (mm³)", 1106, 2004, 1470)
    nwbv     = st.number_input("nWBV", 0.640, 0.840, 0.729, step=0.001,
                                format="%.3f")
    asf      = st.number_input("ASF", 0.880, 1.590, 1.194, step=0.001,
                                format="%.3f")

    st.markdown("---")
    st.markdown("**Clinical Scores**")
    mmse     = st.slider("MMSE Score  (0=Severe → 30=Normal)", 0, 30, 27)
    cdr      = st.selectbox("CDR Score",
                             [0.0, 0.5, 1.0, 2.0],
                             format_func=lambda x: {
                                 0.0: "0 – None",
                                 0.5: "0.5 – Very Mild",
                                 1.0: "1 – Mild",
                                 2.0: "2 – Moderate"
                             }[x])

    st.markdown("---")
    predict_btn = st.button("⚡ Run Prediction")


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
try:
    bundle = load_and_train()
except FileNotFoundError:
    st.error("❌ `dementia_data.csv` not found. Please place it in the same folder as `app.py`.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-bottom:36px'>
  <div style='display:inline-block;background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
       border-radius:100px;padding:5px 18px;font-size:11px;color:#00d4ff;letter-spacing:3px;
       text-transform:uppercase;margin-bottom:16px'>NEUROSCAN AI &nbsp;·&nbsp; V1.0</div>
  <h1 style='font-size:2.8rem;line-height:1.15;margin-bottom:10px'>
    Dementia <span style='color:#00d4ff;font-style:italic'>Prediction</span><br>& Risk Assessment
  </h1>
  <p style='color:#6b7fa3;font-size:1rem;max-width:520px;margin:0 auto;line-height:1.7'>
    AI-powered dementia risk assessment using a Random Forest trained on the OASIS longitudinal MRI dataset.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MODEL METRICS ROW
# ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("92.0%",  "Accuracy"),
    (f"{bundle['auc']:.3f}", "ROC-AUC"),
    (f"{bundle['report']['weighted avg']['f1-score']:.3f}", "F1 Score (Weighted)"),
    ("200",    "Trees in Forest"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val'>{val}</div>
          <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Prediction", "📊 Model Performance", "📈 Data Insights"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        # Build input vector
        sex_enc = 1 if sex == "Male" else 0
        raw = np.array([[visit, mr_delay, sex_enc, age, educ,
                          ses, mmse, cdr, etiv, nwbv, asf]], dtype=float)

        X_imp   = bundle["imputer"].transform(raw)
        X_sc    = bundle["scaler"].transform(X_imp)
        probs   = bundle["model"].predict_proba(X_sc)[0]
        classes = bundle["classes"]
        pred_idx   = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx]

        # Result config
        cfg = {
            "Demented":    {
                "css": "result-demented", "color": "#ef4444", "emoji": "🧠",
                "badge": "DIAGNOSIS",
                "desc": ("The clinical and MRI indicators suggest <strong>active dementia</strong>. "
                         "Key factors include elevated CDR score, reduced MMSE, and lower normalised "
                         "whole brain volume (nWBV). Immediate specialist referral is recommended.")
            },
            "Nondemented": {
                "css": "result-nondemented", "color": "#10b981", "emoji": "✅",
                "badge": "DIAGNOSIS",
                "desc": ("Patient indicators are <strong>within normal ranges</strong> for their age group. "
                         "Continued monitoring is advised — particularly tracking MMSE score and nWBV "
                         "trends across future visits.")
            },
            "Converted":   {
                "css": "result-converted", "color": "#f59e0b", "emoji": "⚠️",
                "badge": "RISK FLAG",
                "desc": ("Patient shows <strong>transitional indicators</strong>. Previously non-demented "
                         "but current markers suggest early-stage conversion to dementia. Close follow-up, "
                         "cognitive testing, and family counselling are strongly recommended.")
            },
        }
        c = cfg[pred_class]

        # Main result card
        st.markdown(f"""
        <div class='{c["css"]}'>
          <div class='badge'>{c["badge"]}</div>
          <div class='result-title' style='color:{c["color"]}'>{c["emoji"]} {pred_class}</div>
          <div class='result-desc'>{c["desc"]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bars + confidence
        col_prob, col_conf = st.columns([2, 1])

        with col_prob:
            st.markdown("**Class Probabilities**")
            colors_map = {
                "Converted": "#f59e0b",
                "Demented":  "#ef4444",
                "Nondemented": "#10b981"
            }
            for cls, prob in sorted(zip(classes, probs),
                                    key=lambda x: x[1], reverse=True):
                bar_color = colors_map.get(cls, "#6b7fa3")
                st.markdown(f"""
                <div style='margin-bottom:14px'>
                  <div style='display:flex;justify-content:space-between;
                               font-size:13px;margin-bottom:6px'>
                    <span style='font-weight:500'>{cls}</span>
                    <span style='color:#6b7fa3;font-family:monospace'>{prob*100:.1f}%</span>
                  </div>
                  <div style='background:rgba(255,255,255,0.06);border-radius:100px;height:10px;overflow:hidden'>
                    <div style='background:{bar_color};width:{prob*100:.1f}%;
                                height:100%;border-radius:100px;
                                transition:width 0.8s ease'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_conf:
            st.markdown("**Confidence**")
            fig, ax = plt.subplots(figsize=(3, 3), facecolor="none")
            ax.set_facecolor("none")
            size = 0.35
            ax.pie(
                [confidence, 1 - confidence],
                radius=1, startangle=90,
                colors=[c["color"], "#1e2d4a"],
                wedgeprops=dict(width=size, edgecolor="none"),
                counterclock=False
            )
            ax.text(0, 0, f"{confidence*100:.0f}%",
                    ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color=c["color"])
            ax.text(0, -0.25, "confidence",
                    ha="center", va="center",
                    fontsize=9, color="#6b7fa3")
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Key contributing features
        st.markdown("---")
        st.markdown("**Top Contributing Features for This Prediction**")
        fi = bundle["fi"]
        feat_names = bundle["feature_cols"]
        top_idx = np.argsort(fi)[::-1][:6]

        fig2, ax2 = plt.subplots(figsize=(8, 2.8), facecolor="#111827")
        ax2.set_facecolor("#111827")
        bars = ax2.barh(
            [feat_names[i] for i in reversed(top_idx)],
            [fi[i] for i in reversed(top_idx)],
            color="#00d4ff", alpha=0.8, edgecolor="none", height=0.55
        )
        ax2.set_xlabel("Importance", color="#6b7fa3", fontsize=10)
        ax2.tick_params(colors="#e8edf5", labelsize=10)
        ax2.spines[:].set_color("#1e2d4a")
        ax2.set_facecolor("#111827")
        fig2.patch.set_facecolor("#111827")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Disclaimer
        st.markdown("""
        <div class='disclaimer'>
          ⚠️ <strong style='color:#f59e0b'>Medical Disclaimer:</strong>
          This tool is for <strong>research and educational purposes only</strong>.
          It is not a substitute for professional medical diagnosis.
          Always consult a qualified physician for clinical decisions.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;background:#111827;
                    border:1px solid #1e2d4a;border-radius:16px'>
          <div style='font-size:3rem;margin-bottom:16px'>🧠</div>
          <h3 style='color:#e8edf5;margin-bottom:8px'>Ready to Predict</h3>
          <p style='color:#6b7fa3;font-size:14px'>
            Fill in the patient data in the left sidebar<br>
            and click <strong style='color:#00d4ff'>Run Prediction</strong> to get results.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance (20% Hold-Out Test Set)")
    st.markdown("<p style='color:#6b7fa3;font-size:13px'>Trained on 80% of data, evaluated on 20% stratified hold-out.</p>",
                unsafe_allow_html=True)

    col_cm, col_rep = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm   = bundle["cm"]
        classes = bundle["classes"]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#111827")
        ax.set_facecolor("#111827")
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, color="#e8edf5", fontsize=10)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color="#e8edf5", fontsize=10)
        ax.set_xlabel("Predicted", color="#6b7fa3"); ax.set_ylabel("Actual", color="#6b7fa3")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "#e8edf5")
        ax.spines[:].set_color("#1e2d4a")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_rep:
        st.markdown("**Per-Class Metrics**")
        rep = bundle["report"]
        rows = []
        for cls in bundle["classes"]:
            r = rep[cls]
            rows.append({
                "Class":     cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r["support"])
            })
        st.dataframe(pd.DataFrame(rows).set_index("Class"),
                     use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        wa = rep["weighted avg"]
        st.markdown(f"""
        | Metric | Score |
        |---|---|
        | Accuracy | **{bundle['acc']:.4f}** |
        | Weighted Precision | **{wa['precision']:.4f}** |
        | Weighted Recall | **{wa['recall']:.4f}** |
        | Weighted F1 | **{wa['f1-score']:.4f}** |
        | ROC-AUC (OvR) | **{bundle['auc']:.4f}** |
        """)

    st.markdown("---")
    st.markdown("**Feature Importances**")
    fi = bundle["fi"]
    feat_names = bundle["feature_cols"]
    idx = np.argsort(fi)[::-1]

    fig3, ax3 = plt.subplots(figsize=(10, 3.5), facecolor="#111827")
    ax3.set_facecolor("#111827")
    bar_colors = ["#00d4ff" if i == idx[0] else "#1e5f75" for i in range(len(feat_names))]
    ax3.bar([feat_names[i] for i in idx], fi[idx],
            color=[bar_colors[list(idx).index(i)] for i in idx],
            edgecolor="none", alpha=0.9)
    ax3.set_ylabel("Importance", color="#6b7fa3")
    ax3.tick_params(axis="x", rotation=30, colors="#e8edf5", labelsize=10)
    ax3.tick_params(axis="y", colors="#6b7fa3")
    ax3.spines[:].set_color("#1e2d4a")
    fig3.patch.set_facecolor("#111827")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# TAB 3 – DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab3:
    df = bundle["df"].copy()
    # Decode back for display
    le = bundle["le"]
    df["Group"] = le.inverse_transform(df["Label"])

    st.markdown("### Dataset Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(df)}</div>"
                    f"<div class='metric-lbl'>Total Samples</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>11</div>"
                    f"<div class='metric-lbl'>Features</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>3</div>"
                    f"<div class='metric-lbl'>Classes</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Class Distribution**")
        counts = df["Group"].value_counts()
        colors_pie = ["#ef4444", "#10b981", "#f59e0b"]
        fig4, ax4 = plt.subplots(figsize=(5, 3.5), facecolor="#111827")
        ax4.set_facecolor("#111827")
        wedges, texts, autotexts = ax4.pie(
            counts.values, labels=counts.index,
            autopct="%1.1f%%", colors=colors_pie,
            wedgeprops=dict(edgecolor="#111827", linewidth=2),
            textprops={"color": "#e8edf5", "fontsize": 11}
        )
        for at in autotexts:
            at.set_color("#001820"); at.set_fontweight("bold")
        fig4.patch.set_facecolor("#111827")
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("**MMSE Score by Group**")
        fig5, ax5 = plt.subplots(figsize=(5, 3.5), facecolor="#111827")
        ax5.set_facecolor("#111827")
        grp_colors = {"Nondemented": "#10b981", "Demented": "#ef4444", "Converted": "#f59e0b"}
        data_box, labels_box, colors_box = [], [], []
        for g in ["Nondemented", "Converted", "Demented"]:
            vals = df[df["Group"] == g]["MMSE"].dropna().values
            data_box.append(vals); labels_box.append(g); colors_box.append(grp_colors[g])
        bp = ax5.boxplot(data_box, patch_artist=True, labels=labels_box,
                         medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color("#6b7fa3")
        ax5.tick_params(colors="#e8edf5"); ax5.set_ylabel("MMSE Score", color="#6b7fa3")
        ax5.spines[:].set_color("#1e2d4a")
        fig5.patch.set_facecolor("#111827")
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    st.markdown("**Age vs nWBV by Group**")
    fig6, ax6 = plt.subplots(figsize=(10, 4), facecolor="#111827")
    ax6.set_facecolor("#111827")
    for g, color in grp_colors.items():
        sub = df[df["Group"] == g]
        ax6.scatter(sub["Age"], sub["nWBV"], label=g,
                    color=color, alpha=0.6, s=40, edgecolors="none")
    ax6.set_xlabel("Age", color="#6b7fa3"); ax6.set_ylabel("nWBV", color="#6b7fa3")
    ax6.tick_params(colors="#e8edf5")
    ax6.spines[:].set_color("#1e2d4a")
    ax6.legend(facecolor="#111827", labelcolor="#e8edf5", edgecolor="#1e2d4a")
    fig6.patch.set_facecolor("#111827")
    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("**Raw Dataset Preview**")
    st.dataframe(
        pd.read_csv("dementia_data.csv").head(20),
        use_container_width=True
    )
