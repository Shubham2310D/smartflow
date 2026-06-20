"""
SmartFlow — Event-Driven Traffic Intelligence Platform
Streamlit entry point (home page).

Run from the smartflow/ directory:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Make src/ importable from all pages
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title  = "SmartFlow | Traffic Intelligence",
    page_icon   = "🚦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("SmartFlow — Event-Driven Traffic Intelligence")
st.caption("Bengaluru Traffic Management · Flipkart Hackathon Round 2")

st.markdown(
    """
    **SmartFlow** ingests historical Bengaluru traffic event data and provides four capabilities:

    | Capability | What it does |
    |---|---|
    | **Predict** | Classify event severity (High/Medium/Low) and forecast clearance time |
    | **Detect** | Surface chronic hotspot junctions using spatial clustering |
    | **Recommend** | Tell traffic officers how many personnel to deploy and from where |
    | **Learn** | Log decisions and track predicted-vs-actual to retrain over time |

    Use the sidebar to navigate between modules.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Key metrics from features.csv
# ---------------------------------------------------------------------------

feats_path = _ROOT / "data" / "processed" / "features.csv"

if feats_path.exists():
    @st.cache_data(show_spinner=False)
    def _load_feats():
        return pd.read_csv(feats_path)

    df = _load_feats()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events",        f"{len(df):,}")
    c2.metric("Corridors Monitored", df["corridor"].nunique())
    c3.metric("Junctions Tracked",   df[df["junction"] != "unknown"]["junction"].nunique())
    c4.metric("High-Severity Events",
              f"{(df['severity_class'] == 'High').sum():,}",
              delta=f"{100*(df['severity_class']=='High').mean():.1f}% of total",
              delta_color="inverse")
    median_dur = df["duration_minutes"].dropna().median()
    c5.metric("Median Clearance", f"{median_dur:.0f} min")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Severity Breakdown")
        sev_counts = df["severity_class"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        order = ["High", "Medium", "Low"]
        sev_counts["Severity"] = pd.Categorical(sev_counts["Severity"], categories=order, ordered=True)
        sev_counts = sev_counts.sort_values("Severity")
        import plotly.express as px
        fig = px.bar(
            sev_counts, x="Severity", y="Count",
            color="Severity",
            color_discrete_map={"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"},
            text_auto=True,
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Top 5 Corridors by Events")
        top_corr = (
            df[df["corridor"] != "Non-corridor"]["corridor"]
            .value_counts().head(5).reset_index()
        )
        top_corr.columns = ["Corridor", "Events"]
        fig2 = px.bar(
            top_corr, x="Events", y="Corridor",
            orientation="h", text_auto=True,
            color_discrete_sequence=["#0d6efd"],
        )
        fig2.update_layout(height=280, margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info(
        "Feature data not yet generated.  "
        "Run `python src/feature_engineering.py` from the smartflow/ directory first."
    )

st.divider()
st.caption(
    "Data source: Astram Traffic Events Dataset, Bengaluru (8,057 incidents after cleaning) · "
    "Models: XGBoost · Clustering: DBSCAN (haversine)"
)
