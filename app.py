import pandas as pd
import streamlit as st

from utils.preprocessing import preprocess
from utils.Scoring import generate_scores

RESULT_PREVIEW_ROWS = 500
UPLOAD_PREVIEW_ROWS = 100
RISK_HISTOGRAM_BINS = 25


def _compact_number(value):
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


st.set_page_config(page_title="CashFlowSense", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

:root {
  --bg1: #07131f;
  --bg2: #0b1c2f;
  --ink: #e8f1fb;
  --muted: #8ca3bb;
  --accent: #1e88e5;
  --accent2: #4cc9f0;
  --card: #0c2135;
  --card2: #12314d;
  --border: rgba(124, 167, 211, 0.18);
  --line: rgba(124, 167, 211, 0.12);
}

html, body, [class*="css"]  {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, Segoe UI, sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(900px 500px at 0% 0%, rgba(76, 201, 240, 0.12), transparent 60%),
    radial-gradient(700px 400px at 100% 0%, rgba(30, 136, 229, 0.16), transparent 55%),
    linear-gradient(180deg, var(--bg2) 0%, var(--bg1) 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #163f72 0%, #0b2746 100%);
  border-right: 1px solid rgba(112, 162, 214, 0.18);
}

[data-testid="stSidebar"] * {
  color: #eef6ff;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #d4e5f7;
}

.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 20px;
  font-weight: 700;
  color: #f4f9ff;
  margin: 26px 0 8px 0;
  display: flex;
  align-items: center;
  gap: 10px;
  letter-spacing: 0.2px;
}

.section-title span {
  display: inline-flex;
  align-items: center;
}

.hero {
  padding: 28px 32px;
  border-radius: 22px;
  background:
    radial-gradient(420px 160px at 85% 10%, rgba(76, 201, 240, 0.16), transparent 55%),
    linear-gradient(135deg, #0b2239 0%, #07192d 100%);
  color: #ffffff;
  border: 1px solid rgba(120, 168, 218, 0.16);
  box-shadow: 0 24px 60px rgba(3, 12, 22, 0.35);
}

.hero-head {
  display: flex;
  align-items: center;
  gap: 18px;
  margin: 0 0 10px 0;
}

.brand-logo {
  width: 68px;
  height: 68px;
  border-radius: 20px;
  position: relative;
  background: linear-gradient(145deg, rgba(76, 201, 240, 0.18), rgba(30, 136, 229, 0.28));
  border: 1px solid rgba(160, 216, 255, 0.18);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 14px 28px rgba(1, 8, 16, 0.22);
  overflow: hidden;
}

.brand-logo:before,
.brand-logo:after {
  content: "";
  position: absolute;
  border-radius: 999px;
}

.brand-logo:before {
  width: 44px;
  height: 44px;
  left: 12px;
  top: 12px;
  border: 5px solid #56d0ff;
  border-right-color: transparent;
  border-bottom-color: rgba(86, 208, 255, 0.35);
  transform: rotate(45deg);
}

.brand-logo:after {
  width: 18px;
  height: 18px;
  right: 10px;
  bottom: 10px;
  background: linear-gradient(135deg, #60d8ff 0%, #2a95f7 100%);
  box-shadow: 0 0 18px rgba(76, 201, 240, 0.35);
}

.hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 42px;
  margin: 0;
  letter-spacing: 0.3px;
}

.hero p {
  margin: 0;
  max-width: 720px;
  color: #c9dbef;
  font-size: 20px;
}

.badges {
  margin-top: 18px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.badge {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.16);
  color: #edf6ff;
  padding: 9px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
}

.panel {
  background: linear-gradient(180deg, rgba(14, 35, 56, 0.94) 0%, rgba(10, 26, 42, 0.96) 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px 12px 18px;
  box-shadow: 0 18px 32px rgba(1, 8, 16, 0.24);
}

.panel-tight {
  padding-bottom: 18px;
}

.divider {
  height: 1px;
  background: var(--line);
  margin: 20px 0 8px 0;
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--card2) 0%, var(--card) 100%);
  border: 1px solid var(--border);
  padding: 14px 16px;
  border-radius: 16px;
  color: var(--ink);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

[data-testid="stMetricLabel"] {
  color: #9eb3c7 !important;
}

[data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-size: 2rem !important;
  line-height: 1.1 !important;
}

[data-testid="stMetricDelta"] {
  color: #61d0ff !important;
}

[data-testid="stFileUploader"] {
  padding: 8px 0;
}

[data-testid="stFileUploader"] > div {
  background: linear-gradient(180deg, rgba(14, 35, 56, 0.98) 0%, rgba(10, 26, 42, 0.98) 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  box-shadow: 0 18px 28px rgba(2, 10, 18, 0.24) !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.02) !important;
  border: 1px dashed rgba(116, 166, 218, 0.22) !important;
  border-radius: 14px !important;
}

[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploader"] svg {
  color: #61d0ff !important;
  fill: #61d0ff !important;
}

[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
  color: #f3f8fe !important;
  font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] small {
  color: #9db6cc !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileData"] {
  color: #f4f9ff !important;
}

[data-testid="stFileUploader"] section {
  color: #f4f9ff !important;
}

[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, #2a95f7 0%, #1c77d2 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover {
  background: linear-gradient(135deg, #49a8ff 0%, #2f87df 100%) !important;
  border-color: rgba(255, 255, 255, 0.36) !important;
}

.small-muted {
  color: var(--muted);
  font-size: 12px;
}

.section-note {
  color: #97afc7;
  font-size: 13px;
  margin: -2px 0 12px 0;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
}

[data-testid="stDataFrame"] div {
  color: #e7f0fa;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlockBorderWrapper"] {
  width: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("Decision Policy ⚙️")
approve_at = st.sidebar.slider("Approve at or above", min_value=60, max_value=95, value=80)
review_at = st.sidebar.slider("Review at or above", min_value=30, max_value=80, value=50)

st.sidebar.markdown("---")
st.sidebar.markdown("Tune the policy thresholds below to make the engine more strict or more flexible.")

st.markdown(
    """
<div class="hero">
  <div class="hero-head">
    <div class="brand-logo"></div>
    <h1>CashFlowSense</h1>
  </div>
  <p>AI based credit risk intelligence for transaction-level decisions.</p>
  <div class="badges">
    <span class="badge">⚡ Model scoring</span>
    <span class="badge">🧪 Data quality checks</span>
    <span class="badge">🛡️ Policy thresholds</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='section-title'>Upload Data ⬆️</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='panel panel-tight'><div class='section-note'>Upload the transaction CSV to run scoring, decisioning, and dashboard analytics in one flow.</div>",
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("Transaction CSV", type=["csv"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    try:
        with st.spinner("Scoring transactions..."):
            df = pd.read_csv(uploaded_file)
            X_scaled, processed_df, quality = preprocess(df)
            results = generate_scores(
                X_scaled, processed_df, approve_at=approve_at, review_at=review_at
            )

        st.markdown("<div class='section-title'>Executive Summary 📊</div>", unsafe_allow_html=True)

        total_rows = int(quality["rows_out"])
        filtered_rows = int(quality["rows_filtered"])
        avg_score = float(results["Risk_Score"].mean())
        avg_stability = float(results["Stability_Score"].mean())
        approve_rate = float((results["Credit_Decision"] == "Approve").mean() * 100)
        review_rate = float((results["Credit_Decision"] == "Review").mean() * 100)
        reject_rate = float((results["Credit_Decision"] == "Reject").mean() * 100)
        high_risk_rate = float((results["Risk_Label"] == "High Risk").mean() * 100)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        top_metrics = st.columns(3)
        top_metrics[0].metric("Scored Rows 📈", _compact_number(total_rows))
        top_metrics[1].metric("Filtered Rows 🧹", _compact_number(filtered_rows))
        top_metrics[2].metric("Avg Risk Score 🎯", f"{avg_score:.2f}")

        bottom_metrics = st.columns(3)
        bottom_metrics[0].metric("Avg Stability 🧭", f"{avg_stability:.2f}")
        bottom_metrics[1].metric("Approve Rate ✅", f"{approve_rate:.1f}%")
        bottom_metrics[2].metric("Reject Rate ⛔", f"{reject_rate:.1f}%")

        st.caption(
            f"Filtered out {quality['filter_rate_pct']}% of rows during preprocessing. "
            f"Invalid numeric rows removed: {quality['invalid_numeric_rows']:,}."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Portfolio Snapshot 🧠</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel panel-tight'>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.metric("Review Rate 🟨", f"{review_rate:.1f}%")
        p2.metric("High Risk Share 🚨", f"{high_risk_rate:.1f}%")
        p3.metric(
            "Avg Suggested Limit 💳", f"{float(results['Suggested_Credit_Limit'].mean()):,.2f}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        type_summary = (
            results.groupby("type")
            .agg(
                Transactions=("amount", "size"),
                Avg_Risk_Score=("Risk_Score", "mean"),
                Avg_Stability_Score=("Stability_Score", "mean"),
                Avg_Suggested_Limit=("Suggested_Credit_Limit", "mean"),
            )
            .round(2)
        )

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("<div class='section-title'>Risk Distribution 📉</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='panel'><div class='section-note'>This chart shows how transaction risk scores are distributed across score bands, helping you see where the portfolio is concentrated.</div>",
                unsafe_allow_html=True,
            )
            histogram = (
                pd.cut(results["Risk_Score"], bins=RISK_HISTOGRAM_BINS, include_lowest=True)
                .value_counts(sort=False)
                .rename_axis("Risk Band")
                .reset_index(name="Transactions")
            )
            histogram["Risk Band"] = histogram["Risk Band"].astype(str)
            st.bar_chart(histogram.set_index("Risk Band"))
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_right:
            st.markdown("<div class='section-title'>Decision Mix 🧾</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='panel'><div class='section-note'>This chart compares how many transactions were approved, reviewed, or rejected under the current policy thresholds.</div>",
                unsafe_allow_html=True,
            )
            decision_mix = (
                results["Credit_Decision"]
                .value_counts()
                .rename_axis("Decision")
                .reset_index(name="Count")
            )
            st.bar_chart(decision_mix.set_index("Decision"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Transaction Type Analysis 🔍</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel'><div class='section-note'>This table compares CASH_OUT and TRANSFER transactions to show which type has weaker risk quality, lower stability, and lower suggested credit capacity.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(type_summary)
        st.markdown("</div>", unsafe_allow_html=True)

        if "step" in processed_df.columns:
            st.markdown("<div class='section-title'>Cash-Flow Trend Over Time 📊</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='panel'><div class='section-note'>This graph tracks transaction volume and average balance over time steps so you can spot spikes, irregular movement, and balance instability.</div>",
                unsafe_allow_html=True,
            )
            trend = (
                processed_df.assign(
                    amount=processed_df["amount"].astype(float),
                    balance=processed_df["newbalanceOrig"].astype(float),
                )
                .groupby("step")
                .agg(Total_Amount=("amount", "sum"), Avg_Balance=("balance", "mean"))
                .reset_index()
            )
            st.line_chart(trend.set_index("step"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Explainability Layer 💬</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel'><div class='section-note'>This table explains the predicted outcome for each transaction using risk probability, risk label, stability score, credit decision, and recommendation fields.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            results[
                [
                    "amount",
                    "balance",
                    "Risk_Probability",
                    "Risk_Label",
                    "Risk_Score",
                    "Stability_Score",
                    "Credit_Decision",
                    "Suggested_Credit_Limit",
                    "Suggested_Tenure",
                    "AI_Explanation",
                ]
            ].head(RESULT_PREVIEW_ROWS)
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Results Preview 🧾</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel'><div class='section-note'>This is the final scored output preview showing the model prediction and business decision for a sample of processed transactions.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(results.head(RESULT_PREVIEW_ROWS))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Uploaded Data Preview 🗂️</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel'><div class='section-note'>This is a sample of the original uploaded CSV so you can compare the raw input with the scored output shown above.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(UPLOAD_PREVIEW_ROWS))
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Failed to score file: {exc}")
