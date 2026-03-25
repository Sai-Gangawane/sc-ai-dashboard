import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Supply Chain Intelligence Hub", layout="wide", page_icon="📦")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f0f4f8; }
.top-bar {
    background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
    padding: 28px 40px; border-radius: 12px; margin-bottom: 24px;
    display: flex; align-items: center; justify-content: space-between;
}
.top-bar-title { font-size: 1.6rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: 0.5px; }
.top-bar-sub { font-size: 0.82rem; color: #a8c8e8; margin-top: 4px; font-weight: 400; }
.top-bar-badge {
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 6px 16px; color: #ffffff; font-size: 0.75rem; font-weight: 500;
}
.kpi-card {
    background: #ffffff; border-radius: 10px; padding: 20px 24px;
    border-left: 4px solid #2d6a9f; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-card.green { border-left-color: #1d9e75; }
.kpi-card.amber { border-left-color: #d97706; }
.kpi-card.red   { border-left-color: #dc2626; }
.kpi-label { font-size: 0.72rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1a3c5e; line-height: 1; }
.kpi-sub { font-size: 0.75rem; color: #9ca3af; margin-top: 4px; }
.section-title {
    font-size: 0.8rem; font-weight: 700; color: #1a3c5e; text-transform: uppercase;
    letter-spacing: 1.2px; padding: 6px 0 6px 12px; border-left: 3px solid #2d6a9f; margin: 28px 0 16px 0;
}
.anomaly-banner {
    background: #fff5f5; border: 1px solid #fecaca; border-left: 4px solid #dc2626;
    border-radius: 10px; padding: 20px 24px; text-align: center;
}
.anomaly-num { font-size: 2.4rem; font-weight: 700; color: #dc2626; line-height: 1; }
.normal-banner {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #1d9e75;
    border-radius: 10px; padding: 20px 24px; text-align: center;
}
.normal-num { font-size: 2.4rem; font-weight: 700; color: #1d9e75; line-height: 1; }
.ai-box {
    background: #ffffff; border-radius: 10px; padding: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #2d6a9f;
    color: #374151; line-height: 1.8; font-size: 0.9rem; margin-top: 16px;
}
.scorecard-header {
    background: #1a3c5e; color: white; padding: 14px 20px; border-radius: 10px 10px 0 0;
    font-weight: 600; font-size: 0.85rem; letter-spacing: 0.3px;
}
.upload-box {
    background: #ffffff; border: 2px dashed #2d6a9f40; border-radius: 10px;
    padding: 30px; text-align: center; margin-bottom: 20px;
}
.upload-title { font-weight: 600; color: #1a3c5e; font-size: 0.95rem; margin-bottom: 6px; }
.upload-sub { color: #6b7280; font-size: 0.82rem; }
.stTextArea textarea {
    background: #ffffff !important; border: 1px solid #d1d5db !important;
    border-radius: 8px !important; color: #374151 !important; font-size: 0.9rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3c5e, #2d6a9f) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.9rem !important; padding: 12px 28px !important;
}
div[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e5e7eb !important; }
.footer-bar {
    background: #1a3c5e; border-radius: 10px; padding: 16px 24px; text-align: center;
    color: #a8c8e8; font-size: 0.75rem; margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div>
        <p class="top-bar-title">📦 Supply Chain Intelligence Hub</p>
        <p class="top-bar-sub">Anomaly Detection &nbsp;·&nbsp; Demand Analytics &nbsp;·&nbsp; Supplier Scorecard &nbsp;·&nbsp; AI Root Cause Analysis</p>
    </div>
    <div class="top-bar-badge">● Live Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ── About ────────────────────────────────────────────────────
st.markdown("""
<div style="background:#ffffff;border-radius:10px;padding:20px 28px;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:20px;display:flex;gap:40px;align-items:flex-start;">
    <div style="flex:2;border-right:1px solid #e5e7eb;padding-right:32px;">
        <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">About This Tool</div>
        <div style="font-size:0.88rem;color:#374151;line-height:1.7;">
            Built by <strong>Sai Santosh Gangawane</strong> — MS Engineering Management, Syracuse University · Lean Six Sigma Black Belt.<br>
            This tool applies <strong>ML anomaly detection</strong>, <strong>supplier analytics</strong>, and <strong>AI-powered DMAIC root cause analysis</strong> to real supply chain data — helping operations teams identify risk, reduce defects, and optimize supplier performance.
        </div>
    </div>
    <div style="flex:1;padding-right:32px;border-right:1px solid #e5e7eb;">
        <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">Tech Stack</div>
        <div style="font-size:0.82rem;color:#374151;line-height:1.9;">
            🐍 Python · Pandas · Scikit-learn<br>
            📊 Plotly · Streamlit<br>
            🤖 Llama 3.3 70B (Groq API)<br>
            🔬 Isolation Forest · DMAIC
        </div>
    </div>
    <div style="flex:1;">
        <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">How to Use</div>
        <div style="font-size:0.82rem;color:#374151;line-height:1.9;">
            1️⃣ Upload your CSV or use default data<br>
            2️⃣ Review KPIs & supplier scorecard<br>
            3️⃣ Check flagged anomalies on the map<br>
            4️⃣ Describe an issue → get AI analysis
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Data Upload ───────────────────────────────────────────────
st.markdown('<div class="section-title">Data Source</div>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-box">
    <div class="upload-title">📂 Upload Your Supply Chain Dataset</div>
    <div class="upload-sub">Upload any CSV with supply chain data — or use the default dataset below</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

@st.cache_data
def load_default():
    df = pd.read_csv("supply_chain_data.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_uploaded(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

if uploaded:
    df = load_uploaded(uploaded)
    st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows, {len(df.columns)} columns")
else:
    df = load_default()
    st.info("📊 Using default supply chain dataset — 100 SKUs across 3 product lines")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("### 🔧 Filters")
if "Product type" in df.columns:
    product_types = st.sidebar.multiselect(
        "Product Type", df["Product type"].unique(), default=df["Product type"].unique()
    )
    df = df[df["Product type"].isin(product_types)]

# ── KPIs ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)

with k1:
    rev = df['Revenue generated'].sum() if 'Revenue generated' in df.columns else 0
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">${rev/1000:.0f}K</div>
        <div class="kpi-sub">Across all product lines</div>
    </div>""", unsafe_allow_html=True)
with k2:
    lt = df['Lead times'].mean() if 'Lead times' in df.columns else 0
    st.markdown(f"""<div class="kpi-card green">
        <div class="kpi-label">Avg Lead Time</div>
        <div class="kpi-value">{lt:.1f} <span style="font-size:1rem;font-weight:400">days</span></div>
        <div class="kpi-sub">Supplier to warehouse</div>
    </div>""", unsafe_allow_html=True)
with k3:
    dr = df['Defect rates'].mean() if 'Defect rates' in df.columns else 0
    st.markdown(f"""<div class="kpi-card amber">
        <div class="kpi-label">Avg Defect Rate</div>
        <div class="kpi-value">{dr:.2f}</div>
        <div class="kpi-sub">Quality control index</div>
    </div>""", unsafe_allow_html=True)
with k4:
    sold = df['Number of products sold'].sum() if 'Number of products sold' in df.columns else 0
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Units Sold</div>
        <div class="kpi-value">{sold:,}</div>
        <div class="kpi-sub">Total demand fulfilled</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

COLORS = ["#1a3c5e", "#0f6e56", "#b45309", "#5b21b6", "#991b1b"]

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f9fafb",
        font=dict(color="#1a3c5e", family="Inter", size=11),
        title_font=dict(color="#1a3c5e", family="Inter", size=13),
       legend=dict(bgcolor="#ffffff", bordercolor="#e5e7eb", borderwidth=1, font=dict(color="#1a3c5e", size=12)),
        xaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb"),
        yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb"),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

# ── Performance Charts ────────────────────────────────────────
st.markdown('<div class="section-title">Supply Chain Performance</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    if "Product type" in df.columns and "Revenue generated" in df.columns:
        fig = px.bar(
            df.groupby("Product type")["Revenue generated"].sum().reset_index(),
            x="Product type", y="Revenue generated",
            title="Revenue by Product Type",
            color="Product type", color_discrete_sequence=COLORS
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

with c2:
    if "Lead times" in df.columns and "Defect rates" in df.columns:
        fig2 = px.scatter(
            df, x="Lead times", y="Defect rates",
            color="Product type" if "Product type" in df.columns else None,
            size="Revenue generated" if "Revenue generated" in df.columns else None,
            title="Lead Time vs Defect Rate",
            color_discrete_sequence=COLORS,
            hover_data=[c for c in ["Supplier name", "SKU"] if c in df.columns]
        )
        st.plotly_chart(style_fig(fig2), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    if "Shipping carriers" in df.columns and "Shipping times" in df.columns:
        fig3 = px.box(
            df, x="Shipping carriers", y="Shipping times",
            title="Shipping Performance by Carrier",
            color="Shipping carriers", color_discrete_sequence=COLORS
        )
        st.plotly_chart(style_fig(fig3), use_container_width=True)

with c4:
    if "Transportation modes" in df.columns:
        fig4 = px.pie(
            df, names="Transportation modes",
            title="Transportation Mode Distribution",
            hole=0.5, color_discrete_sequence=COLORS
        )
        st.plotly_chart(style_fig(fig4), use_container_width=True)

# ── Supplier Scorecard ────────────────────────────────────────
st.markdown('<div class="section-title">Supplier Performance Scorecard</div>', unsafe_allow_html=True)

if "Supplier name" in df.columns:
    scorecard_cols = {}
    if "Defect rates" in df.columns:      scorecard_cols["Avg Defect Rate"] = ("Defect rates", "mean")
    if "Lead times" in df.columns:        scorecard_cols["Avg Lead Time (days)"] = ("Lead times", "mean")
    if "Shipping times" in df.columns:    scorecard_cols["Avg Shipping Time (days)"] = ("Shipping times", "mean")
    if "Revenue generated" in df.columns: scorecard_cols["Total Revenue ($)"] = ("Revenue generated", "sum")
    if "Order quantities" in df.columns:  scorecard_cols["Total Orders"] = ("Order quantities", "sum")

    agg = {v[0]: v[1] for v in scorecard_cols.values()}
    scorecard = df.groupby("Supplier name").agg(agg).reset_index()
    scorecard.columns = ["Supplier"] + list(scorecard_cols.keys())

    # Round and format
    for col in scorecard.columns[1:]:
        if "Revenue" in col:
            scorecard[col] = scorecard[col].apply(lambda x: f"${x:,.0f}")
        elif "Rate" in col or "Time" in col:
            scorecard[col] = scorecard[col].round(2)
        else:
            scorecard[col] = scorecard[col].round(0).astype(int)

    # Risk flag
    if "Avg Defect Rate" in scorecard.columns:
        avg_defect = df["Defect rates"].mean() if "Defect rates" in df.columns else 0
        scorecard["Risk Level"] = scorecard["Avg Defect Rate"].apply(
            lambda x: "🔴 High" if float(x) > avg_defect * 1.3
            else ("🟡 Medium" if float(x) > avg_defect else "🟢 Low")
        )

    st.dataframe(scorecard, use_container_width=True, height=250)

    # Supplier bar chart
    if "Defect rates" in df.columns:
        sup_fig = px.bar(
            df.groupby("Supplier name")["Defect rates"].mean().reset_index().sort_values("Defect rates", ascending=False),
            x="Supplier name", y="Defect rates",
            title="Defect Rate by Supplier — Higher is worse",
            color="Defect rates",
            color_continuous_scale=["#1d9e75", "#d97706", "#dc2626"]
        )
        st.plotly_chart(style_fig(sup_fig), use_container_width=True)

# ── Anomaly Detection ────────────────────────────────────────
st.markdown('<div class="section-title">AI Anomaly Detection — Isolation Forest Model</div>', unsafe_allow_html=True)

anomaly_cols = [c for c in ["Lead times", "Defect rates", "Shipping times", "Stock levels", "Order quantities"] if c in df.columns]

if len(anomaly_cols) >= 2:
    features = df[anomaly_cols].dropna()
    model = IsolationForest(contamination=0.05, random_state=42)
    df.loc[features.index, "anomaly"] = model.fit_predict(features)
    df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "⚠️ Anomaly"})

    anomalies = df[df["anomaly"] == -1]
    normals = df[df["anomaly"] == 1]

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"""<div class="anomaly-banner">
            <div class="kpi-label" style="color:#dc2626;margin-bottom:8px;">Anomalies Detected</div>
            <div class="anomaly-num">{len(anomalies)}</div>
            <div class="kpi-sub" style="color:#ef4444;margin-top:6px;">{len(anomalies)/len(df)*100:.1f}% of total records flagged</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div class="normal-banner">
            <div class="kpi-label" style="color:#1d9e75;margin-bottom:8px;">Normal Records</div>
            <div class="normal-num">{len(normals)}</div>
            <div class="kpi-sub" style="color:#1d9e75;margin-top:6px;">{len(normals)/len(df)*100:.1f}% operating within thresholds</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if "Lead times" in df.columns and "Defect rates" in df.columns:
        fig5 = px.scatter(
            df, x="Lead times", y="Defect rates",
            color="anomaly_label",
            color_discrete_map={"Normal": "#1d9e75", "⚠️ Anomaly": "#dc2626"},
            title="Anomaly Map — Lead Time vs Defect Rate",
            hover_data=[c for c in ["Supplier name", "SKU", "Product type"] if c in df.columns],
            size="Shipping costs" if "Shipping costs" in df.columns else None
        )
        st.plotly_chart(style_fig(fig5), use_container_width=True)

    if len(anomalies) > 0:
        st.markdown("**🚨 Flagged Records requiring attention:**")
        show_cols = [c for c in ["SKU", "Product type", "Supplier name", "Lead times", "Defect rates", "Shipping times"] if c in anomalies.columns]
        st.dataframe(anomalies[show_cols].head(10), use_container_width=True)

# ── AI Root Cause Analysis ───────────────────────────────────
st.markdown('<div class="section-title">AI Root Cause Analysis — DMAIC Framework</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#6b7280;font-size:0.85rem;margin-bottom:16px;">Describe any supply chain issue and receive a structured Lean Six Sigma analysis powered by Llama 3.3 70B.</p>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

problem = st.text_area(
    "Describe the supply chain issue:",
    placeholder="e.g. Defect rates have increased by 15% over the last 3 weeks for SKU haircare products from Supplier 3.",
    height=120
)

if st.button("🔍 Run AI Analysis", key="btn_analyze"):
    if problem:
        with st.spinner("Analyzing with Lean Six Sigma DMAIC framework..."):
            try:
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

                anomaly_summary = ""
                if "anomaly" in df.columns:
                    anomalies_local = df[df["anomaly"] == -1]
                    anomaly_summary = f"""
                    Dataset context:
                    - Anomalies detected: {len(anomalies_local)}
                    - Avg defect rate in anomalies: {round(anomalies_local['Defect rates'].mean(), 2) if 'Defect rates' in anomalies_local.columns else 'N/A'}
- Avg lead time in anomalies: {round(anomalies_local['Lead times'].mean(), 1) if 'Lead times' in anomalies_local.columns else 'N/A'} days
                    """

                message = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=1024,
                    messages=[{
                        "role": "system",
                        "content": "You are a Lean Six Sigma Black Belt and Supply Chain expert. Analyze problems using DMAIC. Be specific and actionable."
                    }, {
                        "role": "user",
                        "content": f"""Problem: {problem}
                        {anomaly_summary}
                        Provide:
                        1. ROOT CAUSES (top 3, 5 Whys thinking)
                        2. IMPACT ASSESSMENT (quantify if possible)
                        3. IMMEDIATE ACTIONS (next 48 hours)
                        4. LONG-TERM FIXES (DMAIC recommendations)
                        5. KPIs TO MONITOR"""
                    }]
                )

                result = message.choices[0].message.content
                st.session_state.chat_history.append({"problem": problem, "analysis": result})
                st.markdown(f'<div class="ai-box">{result.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please describe a supply chain issue to analyze.")

# ── Analysis History ─────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown('<div class="section-title">Previous Analyses</div>', unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
        with st.expander(f"Analysis #{len(st.session_state.chat_history) - i}: {entry['problem'][:80]}..."):
            st.markdown(f'<div class="ai-box">{entry["analysis"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    Built by <strong>Sai Santosh Gangawane</strong> &nbsp;·&nbsp; MS Engineering Management, Syracuse University &nbsp;·&nbsp; Lean Six Sigma Black Belt (ICBB)
</div>
""", unsafe_allow_html=True)
