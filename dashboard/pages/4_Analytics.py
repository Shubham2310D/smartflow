"""
Page 4 — Analytics
Six charts:
  1. Event cause distribution (bar)
  2. Events by hour of day (line — AM/PM peaks)
  3. Events by day of week (bar — Mon–Sun pattern)
  4. Top 10 junctions by event count (horizontal bar)
  5. Corridor congestion score over time (line)
  6. Average resolution time by cause (bar)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import CAUSE_DISPLAY, is_peak_hour

st.set_page_config(page_title="Analytics | SmartFlow", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_FEATS = _ROOT / "data" / "processed" / "features.csv"


@st.cache_data(show_spinner=False)
def load_df():
    if not _FEATS.exists():
        return None
    df = pd.read_csv(_FEATS, parse_dates=["start_datetime"])
    df["cause_label"] = df["event_cause"].map(CAUSE_DISPLAY).fillna(df["event_cause"])
    df["date"]        = df["start_datetime"].dt.date
    return df


st.title("Analytics")
st.caption("Historical patterns from 8,057 Bengaluru traffic incidents")

df = load_df()
if df is None:
    st.error("Run `python src/feature_engineering.py` from smartflow/ to generate analytics data.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    all_zones = sorted(df["zone"].dropna().unique())
    sel_zones = st.multiselect("Zone", all_zones, default=all_zones)

    all_causes = sorted(df["event_cause"].unique())
    sel_causes = st.multiselect("Event Cause", all_causes, default=all_causes)

    if df["start_datetime"].notna().any():
        dt  = pd.to_datetime(df["start_datetime"])
        min_d = dt.min().date()
        max_d = dt.max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    else:
        date_range = None

# Apply filters
mask = df["zone"].isin(sel_zones) & df["event_cause"].isin(sel_causes)
if date_range and len(date_range) == 2:
    mask &= pd.to_datetime(df["start_datetime"]).dt.date.between(date_range[0], date_range[1])

fdf = df[mask].copy()

if len(fdf) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# Top-line metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Events",         f"{len(fdf):,}")
m2.metric("High Severity",  f"{(fdf['severity_class']=='High').sum():,}")
m3.metric("With Duration",  f"{fdf['duration_minutes'].notna().sum():,}")
m4.metric("Median Clearance", f"{fdf['duration_minutes'].dropna().median():.0f} min")

st.divider()

# ---------------------------------------------------------------------------
# Chart 1 + 2 (row 1)
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Event Cause Distribution")
    cause_counts = (
        fdf["cause_label"].value_counts().reset_index()
    )
    cause_counts.columns = ["Cause", "Events"]
    fig1 = px.bar(
        cause_counts, x="Cause", y="Events",
        color="Events", color_continuous_scale="Blues",
        text_auto=True,
    )
    fig1.update_layout(
        coloraxis_showscale=False,
        height=350, margin=dict(t=10, b=10),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Events by Hour of Day")
    hourly = fdf.groupby("hour_of_day").size().reset_index(name="Events")
    hourly["Load"] = hourly["hour_of_day"].apply(
        lambda h: "High-incident" if is_peak_hour(h) else "Off-peak"
    )
    fig2 = px.bar(
        hourly, x="hour_of_day", y="Events",
        color="Load",
        color_discrete_map={"High-incident": "#dc3545", "Off-peak": "#6ea8fe"},
        labels={"hour_of_day": "Hour", "Events": "Event Count"},
    )
    fig2.update_layout(height=350, margin=dict(t=10, b=10), showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 + 4 (row 2)
# ---------------------------------------------------------------------------

col3, col4 = st.columns(2)

with col3:
    st.subheader("Events by Day of Week")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = fdf.groupby("day_of_week").size().reset_index(name="Events")
    dow["Day"] = dow["day_of_week"].map(dict(enumerate(days)))
    dow["Weekend"] = dow["day_of_week"].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
    fig3 = px.bar(
        dow, x="Day", y="Events",
        color="Weekend",
        color_discrete_map={"Weekend": "#fd7e14", "Weekday": "#0d6efd"},
        text_auto=True,
        category_orders={"Day": days},
    )
    fig3.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Top 10 Junctions by Event Count")
    junc_counts = (
        fdf[fdf["junction"] != "unknown"]["junction"]
        .value_counts().head(10).reset_index()
    )
    junc_counts.columns = ["Junction", "Events"]
    fig4 = px.bar(
        junc_counts, x="Events", y="Junction",
        orientation="h", text_auto=True,
        color="Events", color_continuous_scale="Reds",
    )
    fig4.update_layout(
        coloraxis_showscale=False,
        height=350, margin=dict(t=10, b=10),
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 5: Corridor congestion score over time
# ---------------------------------------------------------------------------

st.subheader("Corridor Congestion Score Over Time (Top 6 Corridors)")

top_corridors = (
    fdf[fdf["corridor"] != "Non-corridor"]["corridor"]
    .value_counts().head(6).index.tolist()
)

if top_corridors:
    corr_df = fdf[fdf["corridor"].isin(top_corridors)].copy()
    corr_df["week"] = pd.to_datetime(corr_df["start_datetime"]).dt.to_period("W").dt.start_time
    weekly = (
        corr_df.groupby(["week", "corridor"])["corridor_7d_score"]
        .mean().reset_index()
    )
    weekly.columns = ["Week", "Corridor", "Avg 7d Score"]
    fig5 = px.line(
        weekly, x="Week", y="Avg 7d Score", color="Corridor",
        markers=True,
    )
    fig5.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("No named corridor data available in current filter.")

# ---------------------------------------------------------------------------
# Chart 6: Average resolution time by cause
# ---------------------------------------------------------------------------

st.subheader("Resolution Time by Event Cause (median — mean inflated by unclosed tickets)")

dur_df = (
    fdf.dropna(subset=["duration_minutes"])
    .groupby("cause_label")["duration_minutes"]
    .agg(["mean", "median", "count"])
    .reset_index()
)
dur_df.columns = ["Cause", "Mean (min)", "Median (min)", "N"]
dur_df = dur_df.sort_values("Median (min)", ascending=False)

fig6 = go.Figure()
fig6.add_trace(go.Bar(
    name="Median", x=dur_df["Cause"], y=dur_df["Median (min)"],
    marker_color="#0d6efd", text=dur_df["Median (min)"].round(0), textposition="auto",
))
fig6.add_trace(go.Bar(
    name="Mean",   x=dur_df["Cause"], y=dur_df["Mean (min)"],
    marker_color="#6ea8fe", opacity=0.6,
))
fig6.update_layout(
    barmode="group", height=380,
    margin=dict(t=10, b=10),
    yaxis_title="Minutes",
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig6, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 7: Event type mined from free-text descriptions (NLP)
# ---------------------------------------------------------------------------

if "event_semantic_type" in fdf.columns:
    st.subheader("Event Type Mined from Free-Text Descriptions")
    st.caption(
        "Derived by a bilingual (English + Kannada) keyword pass over the "
        "`description` field — recovers event semantics the structured cause "
        "column misses (sports events, utility work, VIP movement, processions)."
    )
    sem = fdf[fdf["event_semantic_type"] != "other"]["event_semantic_type"]
    if len(sem) > 0:
        sem_counts = sem.value_counts().reset_index()
        sem_counts.columns = ["Event Type", "Events"]
        sem_counts["Event Type"] = sem_counts["Event Type"].str.replace("_", " ").str.title()
        fig7 = px.bar(
            sem_counts, x="Event Type", y="Events",
            color="Events", color_continuous_scale="Teal", text_auto=True,
        )
        fig7.update_layout(
            coloraxis_showscale=False, height=350,
            margin=dict(t=10, b=10), xaxis_tickangle=-30,
        )
        st.plotly_chart(fig7, use_container_width=True)
        matched_pct = 100 * len(sem) / len(fdf)
        st.caption(
            f"{len(sem):,} of {len(fdf):,} events ({matched_pct:.0f}%) matched a "
            "text pattern. The rest are too short or generic to classify."
        )
    else:
        st.info("No event types matched in the current filter selection.")

st.caption(f"Based on {len(fdf):,} events matching current filters.")
