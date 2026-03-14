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
  --bg1: #f7fbff;
  --bg2: #e9f2ff;
  --ink: #0b1320;
  --muted: #556070;
  --accent: #0f4c81;
  --accent2: #18a0fb;
  --card: #ffffff;
  --border: #d8e3f0;
}

html, body, [class*="css"]  {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, Segoe UI, sans-serif;
  color: var(--ink);
}

.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, var(--bg2), var(--bg1));
}

.hero {
  padding: 28px 32px;
  border-radius: 16px;
  background: linear-gradient(135deg, #0f4c81 0%, #0b2b4a 100%);
  color: #ffffff;
  box-shadow: 0 12px 30px rgba(12, 30, 53, 0.2);
}

.hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 34px;
  margin: 0 0 6px 0;
  letter-spacing: 0.2px;
}

.hero p {
  margin: 0;
  color: #dbe7f3;
}

.badges {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.badge {
  background: rgba(255, 255, 255, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.25);
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
}

.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: var(--accent);
  margin: 28px 0 8px 0;
  display: flex;
  align-items: center;
  gap: 10px;
  letter-spacing: 0.2px;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.35);
}

.section-title span {
  display: inline-flex;
  align-items: center;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 18px rgba(13, 44, 77, 0.06);
}

[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 12px 14px;
  border-radius: 12px;
  color: var(--ink);
}

[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
  color: var(--ink) !important;
  font-size: 2.15rem !important;
  line-height: 1.1 !important;
}

[data-testid="stMetricDelta"] {
  color: var(--accent) !important;
}

[data-testid="stFileUploader"] {
  padding: 8px 0;
}

[data-testid="stFileUploader"] > div {
  background: linear-gradient(135deg, #eef5ff 0%, #dceaff 100%) !important;
  border: 1px solid rgba(15, 76, 129, 0.16) !important;
  border-radius: 18px !important;
  box-shadow: 0 14px 30px rgba(18, 38, 63, 0.08) !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.72) !important;
  border: 1px dashed rgba(15, 76, 129, 0.26) !important;
  border-radius: 14px !important;
}

[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploader"] svg {
  color: #0f4c81 !important;
  fill: #0f4c81 !important;
}

[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
  color: #0b1320 !important;
  font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] small {
  color: #556070 !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileData"] {
  color: #0b1320 !important;
}

[data-testid="stFileUploader"] section {
  color: #0b1320 !important;
}

[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, #2b8be7 0%, #176fc1 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255, 255, 255, 0.24) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover {
  background: linear-gradient(135deg, #46a3ff 0%, #2381d8 100%) !important;
  border-color: rgba(255, 255, 255, 0.36) !important;
}

.small-muted {
  color: var(--muted);
  font-size: 12px;
}

.section-note {
  color: var(--muted);
  font-size: 13px;
  margin: -2px 0 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>CashFlowSense</h1>
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

st.sidebar.header("Decision Policy ⚙️")
approve_at = st.sidebar.slider("Approve at or above", min_value=60, max_value=95, value=80)
review_at = st.sidebar.slider("Review at or above", min_value=30, max_value=80, value=50)

st.sidebar.markdown("---")
st.sidebar.markdown("This demo scores each transaction and applies the policy thresholds above.")

st.markdown("<div class='section-title'>Upload Data ⬆️</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Transaction CSV", type=["csv"])

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

        st.markdown("<div class='section-title'>Portfolio Snapshot 🧠</div>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.metric("Review Rate 🟨", f"{review_rate:.1f}%")
        p2.metric("High Risk Share 🚨", f"{high_risk_rate:.1f}%")
        p3.metric(
            "Avg Suggested Limit 💳", f"{float(results['Suggested_Credit_Limit'].mean()):,.2f}"
        )

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

        st.markdown("<div class='section-title'>Risk Distribution 📉</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>This chart shows how transaction risk scores are distributed across score bands, helping you see whether the portfolio is concentrated in low-, medium-, or high-score ranges.</div>",
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

        st.markdown("<div class='section-title'>Decision Mix 🧾</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>This chart compares how many transactions were approved, sent for review, or rejected under the current policy thresholds.</div>",
            unsafe_allow_html=True,
        )
        decision_mix = (
            results["Credit_Decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
        )
        st.bar_chart(decision_mix.set_index("Decision"))

        st.markdown("<div class='section-title'>Transaction Type Analysis 🔍</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>This table compares CASH_OUT and TRANSFER transactions to show which type has weaker risk quality, lower stability, and lower suggested credit capacity.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(type_summary)

        if "step" in processed_df.columns:
            st.markdown("<div class='section-title'>Cash-Flow Trend Over Time 📊</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-note'>This graph tracks transaction volume and average balance over time steps so you can spot spikes, irregular movement, and balance instability.</div>",
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

        st.markdown("<div class='section-title'>Explainability Layer 💬</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>This table explains the predicted outcome for each transaction using risk probability, risk label, stability score, credit decision, and recommendation fields.</div>",
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

        st.markdown("<div class='section-title'>Results Preview 🧾</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>This is the final scored output preview showing the model prediction and business decision for a sample of processed transactions.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            results.head(RESULT_PREVIEW_ROWS)
        )

        st.markdown("<div class='section-title'>Uploaded Data Preview 🗂️</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>This is a sample of the original uploaded CSV so you can compare the raw input with the scored output shown above.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(UPLOAD_PREVIEW_ROWS))
    except Exception as exc:
        st.error(f"Failed to score file: {exc}")
