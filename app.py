import pandas as pd
import streamlit as st

from utils.preprocessing import preprocess
from utils.scoring import generate_scores

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
  font-size: 18px;
  margin: 22px 0 6px 0;
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
}

[data-testid="stFileUploader"] {
  padding: 8px 0;
}

.small-muted {
  color: var(--muted);
  font-size: 12px;
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

st.markdown(
    "<div class='small-muted'>Max upload size configured to 1 GB. Restart Streamlit if you just changed this setting.</div>",
    unsafe_allow_html=True,
)
st.info(
    "If the uploader still shows a 200 MB limit, restart Streamlit from the project root or set "
    "STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024 in the terminal before running the app."
)

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
        approve_rate = float((results["Credit_Decision"] == "Approve").mean() * 100)
        review_rate = float((results["Credit_Decision"] == "Review").mean() * 100)
        reject_rate = float((results["Credit_Decision"] == "Reject").mean() * 100)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Scored Rows 📈", f"{total_rows:,}")
        c2.metric("Filtered Rows 🧹", f"{filtered_rows:,}")
        c3.metric("Avg Risk Score 🎯", f"{avg_score:.2f}")
        c4.metric("Approve Rate ✅", f"{approve_rate:.1f}%")
        c5.metric("Reject Rate ⛔", f"{reject_rate:.1f}%")

        st.caption(f"Filtered out {quality['filter_rate_pct']}% of rows during preprocessing.")

        st.markdown("<div class='section-title'>Risk Distribution 📉</div>", unsafe_allow_html=True)
        st.bar_chart(results["Risk_Score"])

        st.markdown("<div class='section-title'>Results Preview 🧾</div>", unsafe_allow_html=True)
        st.dataframe(
            results[["amount", "Risk_Probability", "Risk_Score", "Credit_Decision"]].head(500)
        )

        st.markdown("<div class='section-title'>Uploaded Data Preview 🗂️</div>", unsafe_allow_html=True)
        st.dataframe(df.head(100))
    except Exception as exc:
        st.error(f"Failed to score file: {exc}")
