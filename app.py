import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="MYTRA AI | Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HELPERS
# ============================================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def risk_band(bad_prob):
    if bad_prob < 20:
        return "LOW RISK", "Low", "🟢"
    if bad_prob < 50:
        return "MEDIUM RISK", "Medium", "🟡"
    return "HIGH RISK", "High", "🔴"


def credit_score(bad_prob):
    # Presentation score only; it is not a bureau score.
    score = int(round(850 - (bad_prob / 100) * 550))
    return max(300, min(850, score))


def score_label(score):
    if score >= 750:
        return "Excellent"
    if score >= 700:
        return "Good"
    if score >= 650:
        return "Fair"
    if score >= 600:
        return "Weak"
    return "Poor"


def ratio_label(value):
    if value <= 20:
        return "Healthy"
    if value <= 35:
        return "Moderate"
    return "Elevated"


def render_progress(label, value, suffix="%", help_text=""):
    value = max(0, min(100, float(value)))
    st.markdown(
        f"""
        <div class="health-row">
            <div class="health-top">
                <span>{label}</span>
                <strong>{value:.1f}{suffix}</strong>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{value}%"></div>
            </div>
            {f'<small>{help_text}</small>' if help_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_display_name(name):
    return name.replace("_", " ").replace("Credit Amount", "Credit Amount").title()


def get_shap_explanation(model, explainer, input_data, n_features):
    """Normalize SHAP output across SHAP/CatBoost versions with a native fallback."""
    values = None
    base_value = None

    try:
        try:
            explanation = explainer(input_data, check_additivity=False)
            values = getattr(explanation, "values", None)
            base = getattr(explanation, "base_values", None)
            if base is not None:
                b = np.asarray(base)
                if b.ndim >= 2 and b.shape[-1] > 1:
                    base_value = float(b.reshape(-1, b.shape[-1])[0, 1])
                else:
                    base_value = float(b.reshape(-1)[0])
        except Exception:
            try:
                raw = explainer.shap_values(input_data, check_additivity=False)
            except TypeError:
                raw = explainer.shap_values(input_data)

            if isinstance(raw, list):
                values = np.asarray(raw[1]) if len(raw) > 1 else np.asarray(raw[0])
            else:
                values = np.asarray(raw)
    except Exception:
        # 3) Native CatBoost SHAP fallback. It returns one extra column for the base value.
        if not hasattr(model, "get_feature_importance"):
            raise
        native = np.asarray(
            model.get_feature_importance(type="ShapValues", data=input_data),
            dtype=float,
        )
        if native.ndim != 2 or native.shape[1] != n_features + 1:
            raise ValueError(f"Native CatBoost SHAP returned shape {native.shape}")
        values = native[:, :n_features]
        base_value = float(native[0, -1])

    arr = np.asarray(values)

    if arr.ndim == 3:
        # Normal multiclass/binary layout: rows x features x outputs.
        if arr.shape[1] == n_features and arr.shape[2] > 1:
            arr = arr[:, :, 1]
        # Alternate layout: rows x outputs x features.
        elif arr.shape[2] == n_features and arr.shape[1] > 1:
            arr = arr[:, 1, :]
        else:
            raise ValueError(f"Unsupported SHAP shape: {arr.shape}")
    elif arr.ndim == 2:
        # CatBoost-native SHAP sometimes has one extra base-value column.
        if arr.shape[1] == n_features + 1:
            arr = arr[:, :n_features]
        elif arr.shape[1] != n_features:
            raise ValueError(
                f"SHAP returned {arr.shape[1]} values for {n_features} features."
            )
    elif arr.ndim == 1:
        if arr.size == n_features + 1:
            arr = arr[:n_features]
        elif arr.size != n_features:
            raise ValueError(
                f"SHAP returned {arr.size} values for {n_features} features."
            )
        arr = arr.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported SHAP output dimensions: {arr.ndim}")

    if base_value is None:
        expected = getattr(explainer, "expected_value", 0.0)
        e = np.asarray(expected)
        if e.ndim > 0 and e.size > 1:
            base_value = float(e.reshape(-1)[1])
        else:
            base_value = float(e.reshape(-1)[0])

    return arr.astype(float), float(base_value)


# ============================================================
# MODERN FINTECH CSS
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0f172a;
    --muted: #64748b;
    --line: #dbe4f0;
    --surface: rgba(255,255,255,0.94);
    --surface-2: rgba(248,250,252,0.92);
    --blue: #2563eb;
    --cyan: #0891b2;
    --purple: #7c3aed;
    --green: #059669;
    --red: #dc2626;
    --amber: #d97706;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Main background ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 7% 7%, rgba(59,130,246,.22), transparent 22%),
        radial-gradient(circle at 93% 10%, rgba(124,58,237,.16), transparent 23%),
        radial-gradient(circle at 80% 78%, rgba(6,182,212,.10), transparent 26%),
        linear-gradient(135deg, #eef4ff 0%, #f8fbff 46%, #edf5ff 100%);
    position: relative;
}

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    z-index: 0;
    pointer-events: none;
    border-radius: 999px;
    filter: blur(3px);
}

[data-testid="stAppViewContainer"]::before {
    width: 360px;
    height: 360px;
    top: -160px;
    right: -100px;
    background: radial-gradient(circle, rgba(37,99,235,.14), transparent 68%);
}

[data-testid="stAppViewContainer"]::after {
    width: 300px;
    height: 300px;
    bottom: 8%;
    left: -130px;
    background: radial-gradient(circle, rgba(124,58,237,.10), transparent 68%);
}

.main .block-container,
[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 1;
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,0);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.35rem;
    padding-bottom: 4rem;
}

#MainMenu, footer { visibility: hidden; }

/* ---------- Global text — explicitly dark so widgets don't inherit white ---------- */
h1, h2, h3, h4, h5, h6,
p, span, label, small,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stCaptionContainer"] {
    color: var(--ink) !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: #64748b !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08111f 0%, #0f1d35 60%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e5e7eb !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #94a3b8 !important;
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:24px;
}

.brand-icon {
    width:44px;
    height:44px;
    border-radius:14px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    box-shadow:0 10px 25px rgba(37,99,235,.3);
    font-size:21px;
}

.brand-title { font-size:20px; font-weight:800; color:white !important; }
.brand-sub { color:#94a3b8 !important; font-size:11px; margin-top:2px; }

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
    background: rgba(255,255,255,.97) !important;
    border: 1px solid #d7e1ee !important;
    color: #0f172a !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    color:#0f172a !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: #2563eb !important;
    border-color: #2563eb !important;
}

/* ---------- Hero ---------- */
.hero {
    position:relative;
    overflow:hidden;
    padding:34px 38px;
    border-radius:24px;
    margin-bottom:24px;
    background:
        radial-gradient(circle at 86% 20%, rgba(96,165,250,.28), transparent 24%),
        radial-gradient(circle at 68% 110%, rgba(124,58,237,.25), transparent 28%),
        linear-gradient(135deg,#07111f 0%,#142a57 52%,#1d4ed8 100%);
    box-shadow:0 18px 45px rgba(15,23,42,.18);
    color:white;
}

.hero:after {
    content:"";
    position:absolute;
    width:280px;
    height:280px;
    right:-90px;
    bottom:-145px;
    border:1px solid rgba(255,255,255,.14);
    border-radius:50%;
    box-shadow:0 0 0 26px rgba(255,255,255,.03),0 0 0 52px rgba(255,255,255,.025);
}

.hero-badge {
    display:inline-flex;
    padding:7px 12px;
    border:1px solid rgba(147,197,253,.35);
    background:rgba(59,130,246,.15);
    color:#bfdbfe !important;
    border-radius:999px;
    font-size:11px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.08em;
}

.hero h1 { color:#ffffff !important; font-size:38px; margin:10px 0 7px; font-weight:800; text-shadow:0 2px 12px rgba(0,0,0,.20); }
.credit-model-line { color:#dbeafe !important; font-size:14px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin-top:14px; }
.hero p { color:#dbeafe !important; max-width:800px; font-size:15px; line-height:1.6; margin:0; }
.hero-meta { margin-top:20px; display:flex; gap:18px; flex-wrap:wrap; color:#dbeafe; font-size:12px; }
.hero-meta span { color:#dbeafe !important; }

/* ---------- Cards ---------- */
.section-card {
    background: var(--surface);
    border: 1px solid rgba(217,226,239,.95);
    border-radius:18px;
    padding:22px;
    margin-bottom:18px;
    box-shadow:0 10px 30px rgba(30,64,175,.065);
    backdrop-filter: blur(14px);
}

.section-title { display:flex; align-items:center; gap:10px; margin-bottom:15px; }
.section-title h3 { margin:0; font-size:18px; color:#0f172a !important; }
.section-title span { color:inherit !important; }

.kpi-card {
    background:rgba(255,255,255,.95);
    border:1px solid var(--line);
    border-radius:18px;
    padding:19px 20px;
    min-height:125px;
    box-shadow:0 8px 24px rgba(15,23,42,.055);
    transition:transform .2s ease, box-shadow .2s ease;
}
.kpi-card:hover { transform:translateY(-3px); box-shadow:0 14px 32px rgba(15,23,42,.10); }
.kpi-label { color:#64748b !important; font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
.kpi-value { color:#0f172a !important; font-size:28px; font-weight:800; margin-top:8px; }
.kpi-sub { color:#94a3b8 !important; font-size:11px; margin-top:4px; }

/* ---------- Decision ---------- */
.decision {
    padding:22px 24px;
    border-radius:18px;
    border:1px solid;
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:20px;
    margin-bottom:18px;
    box-shadow:0 12px 30px rgba(15,23,42,.06);
}
.decision h2 { margin:0; font-size:27px; }
.decision h2, .decision p, .decision span { color:inherit; }
.decision p { margin:6px 0 0; color:#475569 !important; }
.decision-score { text-align:right; }
.decision-score .number { font-size:35px; font-weight:800; }

/* ---------- Score ---------- */
.score-circle {
    width:160px;
    height:160px;
    border-radius:50%;
    margin:auto;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    background:radial-gradient(circle,#fff 58%,transparent 59%),
               conic-gradient(#2563eb 0deg,#38bdf8 220deg,#e2e8f0 220deg);
    box-shadow:0 12px 32px rgba(37,99,235,.14);
}
.score-number { font-size:35px; font-weight:800; color:#0f172a !important; }
.score-caption { color:#64748b !important; font-size:11px; font-weight:600; }

/* ---------- Health / factors ---------- */
.health-row { margin:13px 0 18px; }
.health-top { display:flex; justify-content:space-between; color:#334155 !important; font-size:12px; margin-bottom:7px; }
.health-top span, .health-top strong { color:#334155 !important; }
.progress-track { height:8px; border-radius:999px; background:#e5edf7; overflow:hidden; }
.progress-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#06b6d4,#7c3aed); }
.health-row small { color:#94a3b8 !important; font-size:10px; }

.insight { border-left:4px solid #2563eb; padding:12px 14px; background:#eff6ff; border-radius:0 10px 10px 0; margin:8px 0; }
.insight strong { color:#1e3a8a !important; }

.factor { display:flex; align-items:center; justify-content:space-between; padding:12px 0; border-bottom:1px solid #eef2f7; gap:15px; }
.factor:last-child { border-bottom:0; }
.factor-name { font-size:12px; color:#334155 !important; font-weight:600; }
.factor-impact { font-size:12px; font-weight:800; }
.positive { color:#dc2626 !important; }
.negative { color:#059669 !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap:6px;
    background:rgba(255,255,255,.64);
    border:1px solid #dbe4f0;
    padding:5px;
    border-radius:14px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:9px;
    padding:9px 15px;
    color:#475569 !important;
}
.stTabs [aria-selected="true"] {
    background:white !important;
    color:#1d4ed8 !important;
    box-shadow:0 4px 12px rgba(15,23,42,.07);
}
.stTabs [data-baseweb="tab-highlight"] { background:#2563eb !important; }

/* ---------- Buttons ---------- */
div.stButton > button {
    border-radius:13px !important;
    border:0 !important;
    background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    color:white !important;
    font-weight:700 !important;
    min-height:48px !important;
    box-shadow:0 9px 22px rgba(37,99,235,.22) !important;
    transition:all .2s ease !important;
}
div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 13px 27px rgba(37,99,235,.30) !important; }

/* ---------- Tables / expanders ---------- */
[data-testid="stExpander"] {
    background:rgba(255,255,255,.84);
    border:1px solid #dbe4f0;
    border-radius:16px;
}

[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
}

.footer { text-align:center; color:#64748b !important; font-size:11px; padding:22px; }
.footer * { color:#64748b !important; }

@media (max-width: 900px) {
    .hero h1 { font-size:30px; }
    .decision { align-items:flex-start; flex-direction:column; }
    .decision-score { text-align:left; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA MAPPINGS
# ============================================================
status_map = {
    "No Checking Account": 0, "< 0 DM": 1, "0 to 200 DM": 2,
    ">= 200 DM (Salary Account)": 3
}
history_map = {
    "No credits taken / All paid back": 0,
    "All credits at this bank paid back duly": 1,
    "Existing credits paid back duly till now": 2,
    "Delay in paying off in the past": 3,
    "Critical account / Other credits existing": 4
}
purpose_map = {
    "Car (New)": 0, "Car (Used)": 1, "Furniture/Equipment": 2,
    "Radio/Television": 3, "Domestic Appliances": 4, "Repairs": 5,
    "Education": 6, "Vacation": 7, "Retraining": 8, "Business": 9, "Others": 10
}
savings_map = {
    "Unknown / No savings account": 0, "< 100 DM": 1,
    "100 to 500 DM": 2, "500 to 1000 DM": 3, ">= 1000 DM": 4
}
emp_map = {
    "Unemployed": 0, "< 1 year": 1, "1 to 4 years": 2,
    "4 to 7 years": 3, ">= 7 years": 4
}
sex_map = {
    "Male: Divorced/Separated": 0,
    "Female: Divorced/Separated/Married": 1,
    "Male: Single": 2,
    "Male: Married/Widowed": 3
}
debtor_map = {"None": 0, "Co-applicant": 1, "Guarantor": 2}
property_map = {
    "Real Estate": 0, "Life Insurance / Building Society": 1,
    "Car / Other": 2, "Unknown / No Property": 3
}
install_map = {"Bank": 0, "Stores": 1, "None": 2}
housing_map = {"Rent": 0, "Own": 1, "For Free": 2}
job_map = {
    "Unemployed / Unskilled (Non-resident)": 0,
    "Unskilled (Resident)": 1, "Skilled Employee": 2,
    "Management / Self-Employed": 3
}
foreign_map = {"Yes (Foreign Worker)": 0, "No": 1}

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("models/credit_risk_model.pkl")
        explainer = shap.TreeExplainer(model)
        return model, explainer, None
    except Exception as e:
        return None, None, str(e)


model, explainer, model_error = load_ml_assets()
model_loaded = model is not None

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">💳</div>
            <div>
                <div class="brand-title">MYTRA AI</div>
                <div class="brand-sub">Credit Intelligence Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Risk Controls")

    risk_threshold = st.slider(
        "Default Risk Threshold (%)",
        10.0, 90.0, 50.0, 5.0,
        help="Probability above this threshold is classified as high risk."
    )

    st.markdown("---")
    st.markdown("### 🧩 Model Information")

    st.markdown(
        """
        **Algorithm**  
        CatBoost Classifier

        **Input Variables**  
        23 features

        **Explainability**  
        SHAP TreeExplainer

        **Dataset**  
        German Credit
        """
    )

    if model_loaded:
        st.success("● Model Online")
    else:
        st.error("● Model Offline")
        if model_error:
            st.caption(model_error)

    st.markdown("---")
    st.caption("MYTRA AI • Risk Core v2.0")
    st.caption("Machine Learning Underwriting Engine")

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">MYTRA AI • RISK CORE v2.0</span>
        <div class="credit-model-line">Credit Risk Model • AI Underwriting &amp; Explainability</div>
        <h1 style="color:#ffffff !important;">Credit Intelligence, Reimagined.</h1>
        <p>
            An AI-powered credit risk assessment workspace combining
            predictive modeling, financial health indicators and
            explainable AI to analyze applicant profiles.
        </p>
        <div class="hero-meta">
            <span>⚡ Real-time inference</span>
            <span>🧠 SHAP explainability</span>
            <span>📊 23-feature risk model</span>
            <span>🔒 Local model inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TOP OVERVIEW
# ============================================================
overview_cols = st.columns(4)
overview_data = [
    ("Model", "CatBoost", "Classification engine"),
    ("Features", "23", "Applicant variables"),
    ("Explainability", "SHAP", "Local attribution"),
    ("Status", "ONLINE" if model_loaded else "OFFLINE", "Model availability"),
]

for col, (label, value, sub) in zip(overview_cols, overview_data):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ============================================================
# INPUT WORKSPACE
# ============================================================
st.markdown("## <span style='color:#0f172a'>🔍 Credit Assessment</span>", unsafe_allow_html=True)
st.caption("Enter the applicant profile below and run the AI assessment.")

tab_fin, tab_demo, tab_assets = st.tabs(
    ["💳 Financial Profile", "👤 Demographics & Work", "🏠 Assets & Credit History"]
)

with tab_fin:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        v_amount = st.number_input("Credit Amount ($)", min_value=250, value=5000, step=250)
        v_duration = st.slider("Loan Term (Months)", 4, 72, 24)
        v_monthly = st.number_input(
            "Estimated Monthly Payment ($)", min_value=0.0, value=220.0, step=10.0
        )

    with c2:
        v_status = st.selectbox("Checking Account Status", list(status_map.keys()))
        v_savings = st.selectbox("Savings Balance", list(savings_map.keys()))

    with c3:
        v_purpose = st.selectbox("Loan Purpose", list(purpose_map.keys()))
        v_install_rate = st.slider("Installment Rate (%)", 1, 4, 2)
        v_high_cred = st.selectbox("Exposure Flag", ["Standard", "High Exposure"])

    st.markdown("</div>", unsafe_allow_html=True)

with tab_demo:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)

    with c4:
        v_age = st.number_input("Applicant Age", 18, 90, 35)
        v_age_grp = st.selectbox(
            "Age Group",
            ["0 (Youth)", "1 (Young Adult)", "2 (Adult)", "3 (Senior)"]
        )
        v_sex = st.selectbox("Civil Status & Gender", list(sex_map.keys()))

    with c5:
        v_emp = st.selectbox("Employment Duration", list(emp_map.keys()))
        v_job = st.selectbox("Job Classification", list(job_map.keys()))

    with c6:
        v_liable = st.slider("Number of Dependents", 1, 2, 1)
        v_foreign = st.selectbox("Foreign Worker Status", list(foreign_map.keys()))
        v_phone = st.selectbox(
            "Contact Verification", ["Unverified/None", "Verified / Registered"]
        )

    st.markdown("</div>", unsafe_allow_html=True)

with tab_assets:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)

    with c7:
        v_history = st.selectbox("Payment History", list(history_map.keys()))
        v_exist_cred = st.slider("Active Credit Lines", 1, 4, 1)

    with c8:
        v_housing = st.selectbox("Housing Status", list(housing_map.keys()))
        v_residence = st.slider("Years at Current Residence", 1, 4, 2)
        v_property = st.selectbox("Primary Asset", list(property_map.keys()))

    with c9:
        v_other_debt = st.selectbox("Other Debtors / Guarantors", list(debtor_map.keys()))
        v_other_inst = st.selectbox("External Installment Plans", list(install_map.keys()))

    st.markdown("</div>", unsafe_allow_html=True)

predict = st.button("🚀 Run AI Credit Assessment", use_container_width=True)

# ============================================================
# PREDICTION
# ============================================================
# Streamlit reruns the full script after widget interactions.
# Keep the latest successful assessment in session_state so the
# dashboard and What-If simulator continue working after reruns.
if predict and model_loaded:
    input_data = pd.DataFrame([{
        "Status": status_map[v_status],
        "Duration": v_duration,
        "Credit_History": history_map[v_history],
        "Purpose": purpose_map[v_purpose],
        "Credit_Amount": v_amount,
        "Savings": savings_map[v_savings],
        "Employment": emp_map[v_emp],
        "Installment_Rate": v_install_rate,
        "Personal_Status_Sex": sex_map[v_sex],
        "Other_Debtors": debtor_map[v_other_debt],
        "Residence_Since": v_residence,
        "Property": property_map[v_property],
        "Age": v_age,
        "Other_Installment": install_map[v_other_inst],
        "Housing": housing_map[v_housing],
        "Existing_Credits": v_exist_cred,
        "Job": job_map[v_job],
        "People_Liable": v_liable,
        "Telephone": 1 if v_phone == "Verified / Registered" else 0,
        "Foreign_Worker": foreign_map[v_foreign],
        "Monthly_Loan": v_monthly,
        "Age_Group": int(v_age_grp.split(" ")[0]),
        "High_Credit": 1 if v_high_cred == "High Exposure" else 0,
    }])

    try:
        with st.spinner("Running CatBoost inference..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

        bad_prob = float(probability[0]) * 100
        good_prob = float(probability[1]) * 100
        confidence = max(good_prob, bad_prob)
        band, band_short, band_icon = risk_band(bad_prob)
        score = credit_score(bad_prob)
        score_text = score_label(score)

        payment_ratio = (v_monthly / v_amount * 100) if v_amount else 0
        savings_strength = (savings_map[v_savings] / 4) * 100
        employment_strength = (emp_map[v_emp] / 4) * 100
        history_strength = max(0, 100 - (history_map[v_history] / 4) * 100)

        result = {
            "time": datetime.now().strftime("%d %b %Y, %H:%M"),
            "amount": v_amount,
            "risk": bad_prob,
            "approval": good_prob,
            "decision": "Eligible" if bad_prob < risk_threshold else "Manual Review / High Risk",
            "score": score,
        }
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:10]
        st.session_state.last_result = result
        st.session_state.assessment = {
            "input_data": input_data.copy(),
            "bad_prob": bad_prob,
            "good_prob": good_prob,
            "confidence": confidence,
            "band": band,
            "band_short": band_short,
            "band_icon": band_icon,
            "score": score,
            "score_text": score_text,
            "payment_ratio": payment_ratio,
            "savings_strength": savings_strength,
            "employment_strength": employment_strength,
            "history_strength": history_strength,
            "baseline_amount": int(v_amount),
            "baseline_monthly": int(v_monthly),
            "threshold": float(risk_threshold),
        }
        st.session_state.simulation = None

    except Exception as e:
        st.error(
            "Execution Error: your model must expect exactly the 23 features "
            f"used by this application.\n\nDetails: {e}"
        )
        st.stop()

elif predict and not model_loaded:
    st.error("The ML model is not loaded. Check models/credit_risk_model.pkl.")

# ============================================================
# PERSISTED ASSESSMENT DASHBOARD
# ============================================================
assessment = st.session_state.get("assessment")
if assessment is not None and model_loaded:
    input_data = assessment["input_data"].copy()
    bad_prob = assessment["bad_prob"]
    good_prob = assessment["good_prob"]
    confidence = assessment["confidence"]
    band = assessment["band"]
    band_short = assessment["band_short"]
    band_icon = assessment["band_icon"]
    score = assessment["score"]
    score_text = assessment["score_text"]
    payment_ratio = assessment["payment_ratio"]
    savings_strength = assessment["savings_strength"]
    employment_strength = assessment["employment_strength"]
    history_strength = assessment["history_strength"]
    baseline_amount = assessment["baseline_amount"]
    baseline_monthly = assessment["baseline_monthly"]
    assessment_threshold = assessment["threshold"]

    # DECISION HEADER
    # ========================================================
    st.markdown("---")

    if bad_prob >= assessment_threshold:
        decision_color = "#dc2626"
        decision_bg = "#fef2f2"
        decision_border = "#fecaca"
        decision_text = "MANUAL REVIEW / HIGH RISK"
        decision_subtext = (
            f"Estimated default probability is {bad_prob:.2f}%, "
            f"above the configured {assessment_threshold:.0f}% risk threshold."
        )
    else:
        decision_color = "#059669"
        decision_bg = "#ecfdf5"
        decision_border = "#a7f3d0"
        decision_text = "LOWER RISK / ELIGIBLE"
        decision_subtext = (
            f"Estimated default probability is {bad_prob:.2f}%, "
            "below the configured risk threshold."
        )

    st.markdown(
        f"""
        <div class="decision"
             style="background:{decision_bg};border-color:{decision_border};">
            <div>
                <h2 style="color:{decision_color};">{band_icon} {decision_text}</h2>
                <p>{decision_subtext}</p>
            </div>
            <div class="decision-score">
                <div class="number" style="color:{decision_color};">{score}</div>
                <div style="color:#64748b;font-size:11px;font-weight:700;">
                    PRESENTATION CREDIT SCORE
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # KPI ROW
    # ========================================================
    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        ("Credit Score", f"{score}", score_text),
        ("Default Risk", f"{bad_prob:.2f}%", band),
        ("Approval Probability", f"{good_prob:.2f}%", "Model estimate"),
        ("Model Confidence", f"{confidence:.2f}%", "Prediction confidence"),
    ]

    for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # ========================================================
    # FINANCIAL HEALTH + SCORE
    # ========================================================
    left, right = st.columns([1, 1.5])

    with left:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">
                    <span>💠</span><h3>Risk Score</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Build a score gauge using progress-style HTML.
        st.markdown(
            f"""
            <div class="score-circle">
                <div class="score-number">{score}</div>
                <div class="score-caption">{score_text}</div>
            </div>
            <br>
            <div style="text-align:center;color:#64748b;font-size:11px;">
                Presentation score derived from model default probability.
                <br>Not a bureau/FICO score.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">
                    <span>📊</span><h3>Financial Health Snapshot</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        render_progress(
            "Payment-to-credit ratio",
            payment_ratio,
            help_text=ratio_label(payment_ratio),
        )
        render_progress(
            "Savings strength",
            savings_strength,
            help_text="Higher indicates a stronger savings category.",
        )
        render_progress(
            "Employment stability",
            employment_strength,
            help_text="Based on the selected employment-duration category.",
        )
        render_progress(
            "Credit history strength",
            history_strength,
            help_text="Higher indicates a stronger selected history category.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # AI INSIGHTS
    # ========================================================
    st.markdown("## <span style='color:#0f172a'>🧠 AI Decision Intelligence</span>", unsafe_allow_html=True)

    # ========================================================
    # ROBUST SHAP FEATURE ATTRIBUTION
    # ========================================================
    target_shap = None
    base_value = None
    shap_error = None

    try:
        with st.spinner("Generating SHAP feature attribution..."):
            target_shap, base_value = get_shap_explanation(
                model, explainer, input_data, input_data.shape[1]
            )
    except Exception as e:
        shap_error = str(e)
        st.error(
            "Feature attribution could not be generated. "
            f"SHAP details: {shap_error}"
        )

    if target_shap is not None and target_shap.shape[1] == input_data.shape[1]:
        shap_row = target_shap[0]
        feature_names = list(input_data.columns)
        feature_values = input_data.iloc[0].values

        factor_df = pd.DataFrame({
            "feature": feature_names,
            "value": feature_values,
            "impact": shap_row,
        })
        factor_df["abs_impact"] = factor_df["impact"].abs()
        factor_df = factor_df.sort_values("abs_impact", ascending=False)

        # ----------------------------------------------------
        # TOP ATTRIBUTIONS
        # ----------------------------------------------------
        f1, f2 = st.columns([1.05, 1])

        with f1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-title">
                    <span>🎯</span><h3>Top Feature Contributions</h3>
                </div>
                <div style="font-size:11px;color:#64748b;margin-bottom:8px;">
                    Positive values push the model toward class 1; negative values
                    push it away. SHAP values are contributions in the model's output space.
                </div>
                """,
                unsafe_allow_html=True,
            )

            for _, row in factor_df.head(8).iterrows():
                impact = float(row["impact"])
                cls = "positive" if impact >= 0 else "negative"
                sign = "+" if impact >= 0 else ""
                st.markdown(
                    f"""
                    <div class="factor">
                        <div>
                            <div class="factor-name">{feature_display_name(row['feature'])}</div>
                            <div style="font-size:10px;color:#94a3b8;margin-top:2px;">
                                Applicant value: {row['value']}
                            </div>
                        </div>
                        <span class="factor-impact {cls}">{sign}{impact:.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with f2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-title">
                    <span>💡</span><h3>Model Reasoning</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            positive = factor_df[factor_df["impact"] > 0].head(3)
            negative = factor_df[factor_df["impact"] < 0].head(3)

            if len(positive):
                for _, row in positive.iterrows():
                    st.markdown(
                        f"""
                        <div class="insight">
                            <strong>↗ {feature_display_name(row['feature'])}</strong><br>
                            Positive contribution: <b>{row['impact']:.4f}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if len(negative):
                for _, row in negative.iterrows():
                    st.markdown(
                        f"""
                        <div class="insight" style="border-left-color:#059669;background:#ecfdf5;">
                            <strong style="color:#065f46;">↘ {feature_display_name(row['feature'])}</strong><br>
                            Negative contribution: <b>{row['impact']:.4f}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # EXPLAINABLE AI WORKSPACE
        # ----------------------------------------------------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">
                <span>🔬</span><h3>Explainable AI Workspace</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        explain_tab, bar_tab, table_tab = st.tabs(
            ["Local Explanation", "Feature Attribution", "All 23 Features"]
        )

        with explain_tab:
            st.caption(
                "Local explanation for this applicant. SHAP shows how each feature moves the class-1 model output away from the baseline."
            )
            try:
                force = shap.force_plot(
                    base_value,
                    shap_row,
                    input_data.iloc[0, :],
                    matplotlib=False,
                )
                st_shap(force, height=230)
            except Exception as e:
                st.warning(f"Local SHAP plot unavailable: {e}")

        with bar_tab:
            st.caption("Horizontal ranking of the 10 strongest local feature contributions.")

            # Plotly is used here instead of Matplotlib so the horizontal chart
            # reliably renders inside Streamlit tabs and remains interactive.
            plot_df = factor_df.head(10).copy().sort_values("impact")
            labels = [feature_display_name(x) for x in plot_df["feature"]]
            values = plot_df["impact"].astype(float).to_numpy()
            applicant_values = plot_df["value"].astype(str).tolist()
            bar_colors = ["#ef4444" if v >= 0 else "#10b981" for v in values]
            text_values = [f"{v:+.4f}" for v in values]

            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        line=dict(color="rgba(255,255,255,0.9)", width=1),
                    ),
                    text=text_values,
                    textposition="outside",
                    cliponaxis=False,
                    customdata=applicant_values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "SHAP contribution: %{x:+.4f}<br>"
                        "Applicant value: %{customdata}"
                        "<extra></extra>"
                    ),
                    opacity=0.92,
                )
            )

            max_abs = max(float(np.max(np.abs(values))) if len(values) else 0.01, 0.01)
            x_pad = max_abs * 0.30

            fig.update_layout(
                height=max(470, 56 * len(labels) + 110),
                margin=dict(l=8, r=70, t=70, b=65),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, Arial, sans-serif", color="#334155"),
                title=dict(
                    text="<b>Top Feature Attribution</b>",
                    x=0,
                    xanchor="left",
                    font=dict(size=18, color="#0f172a"),
                ),
                xaxis=dict(
                    title=dict(
                        text="SHAP contribution to class-1 output",
                        font=dict(size=11, color="#64748b"),
                    ),
                    range=[-max_abs - x_pad, max_abs + x_pad],
                    zeroline=True,
                    zerolinecolor="#94a3b8",
                    zerolinewidth=1.5,
                    gridcolor="#e5e7eb",
                    tickfont=dict(size=10, color="#64748b"),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=11, color="#334155"),
                    gridcolor="rgba(0,0,0,0)",
                ),
                showlegend=False,
                hoverlabel=dict(bgcolor="#0f172a", font_color="white"),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

            st.markdown(
                '<div style="display:flex;gap:22px;font-size:11px;margin:4px 0 0;">'
                '<span style="color:#dc2626;font-weight:700;">■ Positive contribution toward class 1</span>'
                '<span style="color:#059669;font-weight:700;">■ Negative contribution away from class 1</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        with table_tab:
            display_df = factor_df.copy()
            display_df["Feature"] = display_df["feature"].map(feature_display_name)
            display_df["Applicant Value"] = display_df["value"].astype(str)
            display_df["SHAP Impact"] = display_df["impact"].map(lambda x: f"{x:+.5f}")
            display_df["Direction"] = np.where(
                display_df["impact"] >= 0,
                "Positive / toward class 1",
                "Negative / away from class 1",
            )
            display_df = display_df[[
                "Feature", "Applicant Value", "SHAP Impact", "Direction"
            ]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="section-card" style="border-color:#fecaca;background:#fff7f7;">'
            '<b style="color:#b91c1c;">Feature attribution unavailable</b><br>'
            '<span style="color:#64748b;font-size:12px;">'
            'The prediction completed, but SHAP did not return a feature vector matching the 23 model inputs.'
            '</span></div>',
            unsafe_allow_html=True,
        )

    # ========================================================
    # WHAT-IF SIMULATOR
    # ========================================================
    st.markdown("## <span style='color:#0f172a'>🧪 What-If Simulator</span>", unsafe_allow_html=True)
    st.caption(
        "Change the credit amount or monthly payment and compare the new model estimate "
        "with the last assessed applicant."
    )

    sim1, sim2 = st.columns(2)
    with sim1:
        sim_amount = st.slider(
            "Simulated Credit Amount ($)",
            min_value=250,
            max_value=50000,
            value=int(baseline_amount),
            step=250,
            key="sim_amount_slider",
        )
    with sim2:
        sim_monthly = st.slider(
            "Simulated Monthly Payment ($)",
            min_value=0,
            max_value=10000,
            value=int(baseline_monthly),
            step=50,
            key="sim_monthly_slider",
        )

    if st.button("🔄 Simulate Scenario", use_container_width=True, key="simulate_scenario_btn"):
        sim_input = input_data.copy()
        sim_input["Credit_Amount"] = sim_amount
        sim_input["Monthly_Loan"] = sim_monthly

        try:
            with st.spinner("Running simulated scenario..."):
                sim_prob = model.predict_proba(sim_input)[0]

            sim_bad = float(sim_prob[0]) * 100
            sim_good = float(sim_prob[1]) * 100
            delta_risk = sim_bad - bad_prob
            sim_band, _, sim_icon = risk_band(sim_bad)
            sim_score = credit_score(sim_bad)

            st.session_state.simulation = {
                "amount": sim_amount,
                "monthly": sim_monthly,
                "bad": sim_bad,
                "good": sim_good,
                "delta": delta_risk,
                "band": sim_band,
                "icon": sim_icon,
                "score": sim_score,
            }
        except Exception as e:
            st.session_state.simulation = None
            st.error(
                "Simulation failed. The model must accept the same 23 features as the baseline assessment. "
                f"Details: {e}"
            )

    simulation = st.session_state.get("simulation")

    if simulation:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">
                <span>📈</span><h3>Scenario Comparison</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric(
                "Simulated Default Risk",
                f"{simulation['bad']:.2f}%",
                f"{simulation['delta']:+.2f}% vs baseline",
            )
        with s2:
            st.metric(
                "Simulated Approval Probability",
                f"{simulation['good']:.2f}%",
                f"{simulation['good'] - good_prob:+.2f}% vs baseline",
            )
        with s3:
            st.metric("Simulated Risk Band", f"{simulation['icon']} {simulation['band']}")
        with s4:
            st.metric(
                "Simulated Score",
                f"{simulation['score']}",
                f"{simulation['score'] - score:+d} vs baseline",
            )

        # Two donut charts make the comparison visually clear while keeping
        # each scenario internally consistent (Default + Approval = 100%).
        compare_fig = go.Figure()

        compare_fig.add_trace(go.Pie(
            labels=["Default Risk", "Approval Probability"],
            values=[bad_prob, good_prob],
            hole=0.62,
            domain={"x": [0.02, 0.48], "y": [0.02, 0.98]},
            marker=dict(colors=["#ef4444", "#10b981"], line=dict(color="#ffffff", width=3)),
            textinfo="percent",
            textfont=dict(color="#0f172a", size=13),
            hovertemplate="%{label}: %{value:.2f}%<extra>Baseline</extra>",
            sort=False,
            direction="clockwise",
        ))

        compare_fig.add_trace(go.Pie(
            labels=["Default Risk", "Approval Probability"],
            values=[simulation["bad"], simulation["good"]],
            hole=0.62,
            domain={"x": [0.52, 0.98], "y": [0.02, 0.98]},
            marker=dict(colors=["#ef4444", "#10b981"], line=dict(color="#ffffff", width=3)),
            textinfo="percent",
            textfont=dict(color="#0f172a", size=13),
            hovertemplate="%{label}: %{value:.2f}%<extra>Simulated</extra>",
            sort=False,
            direction="clockwise",
            showlegend=False,
        ))

        compare_fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=85, b=25),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.5,
                y=-0.03,
                xanchor="center",
                yanchor="top",
                font=dict(size=12, color="#334155"),
            ),
            annotations=[
                dict(
                    text=f"<b>BASELINE</b><br><span style='font-size:20px'>{bad_prob:.2f}%</span> risk",
                    x=0.25, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13, color="#0f172a")
                ),
                dict(
                    text=f"<b>SIMULATED</b><br><span style='font-size:20px'>{simulation['bad']:.2f}%</span> risk",
                    x=0.75, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13, color="#0f172a")
                ),
                dict(
                    text="<b>Risk Composition</b>",
                    x=0.5, y=1.10, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=18, color="#0f172a")
                ),
            ],
        )

        st.plotly_chart(
            compare_fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        st.markdown(
            f"<div class='insight'><strong>Scenario:</strong> ${simulation['amount']:,.0f} credit amount "
            f"with ${simulation['monthly']:,.0f} monthly payment. "
            f"The model estimates <b>{simulation['bad']:.2f}%</b> default probability for this scenario.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # APPLICATION DATA / EXPORT
    st.markdown("## <span style='color:#0f172a'>📁 Application Record</span>", unsafe_allow_html=True)

    with st.expander("View model payload & export", expanded=False):
        st.dataframe(input_data, use_container_width=True, hide_index=True)

        csv_data = input_data.to_csv(index=False)
        json_data = input_data.to_json(orient="records", indent=2)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                "mytra_application.csv",
                "text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download JSON",
                json_data,
                "mytra_application.json",
                "application/json",
                use_container_width=True,
            )

# ============================================================
# APPLICATION HISTORY
# ============================================================
st.markdown("---")
st.markdown("## <span style='color:#0f172a'>🗂️ Recent Assessments</span>", unsafe_allow_html=True)

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    history_df.columns = [
        "Time", "Loan Amount", "Default Risk",
        "Approval Probability", "Model Decision", "Credit Score"
    ]
    history_df["Loan Amount"] = history_df["Loan Amount"].map(lambda x: f"${x:,.0f}")
    history_df["Default Risk"] = history_df["Default Risk"].map(lambda x: f"{x:.2f}%")
    history_df["Approval Probability"] = history_df["Approval Probability"].map(lambda x: f"{x:.2f}%")
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div class="history-empty">No assessments yet. Run your first AI credit assessment above.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        MYTRA AI • Credit Intelligence Platform • Risk Core v2.0<br>
        Developed by Ishan Arora • CatBoost + SHAP Explainable AI
    </div>
    """,
    unsafe_allow_html=True,
)
