"""
Page 1 — Hotspot Map
Folium map with:
  Layer 1: KDE heatmap (density surface)
  Layer 2: DBSCAN cluster polygons (clickable)
  Layer 3: Individual event markers (color by severity, optional)
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Hotspot Map | SmartFlow", page_icon="🗺️", layout="wide")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FEATS       = _ROOT / "data" / "processed" / "features.csv"
_HOTSPOTS    = _ROOT / "data" / "processed" / "hotspots.geojson"
_HEATMAP_PTS = _ROOT / "data" / "processed" / "heatmap_points.csv"
_SUMMARY     = _ROOT / "data" / "processed" / "hotspot_summary.csv"

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_features():
    if not _FEATS.exists():
        return None
    df = pd.read_csv(_FEATS, parse_dates=["start_datetime"])
    return df

@st.cache_data(show_spinner=False)
def load_geojson():
    if not _HOTSPOTS.exists():
        return None
    with open(_HOTSPOTS) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_heatmap():
    if not _HEATMAP_PTS.exists():
        return None
    df = pd.read_csv(_HEATMAP_PTS)
    return df[["lat", "lon", "weight"]].values.tolist()

@st.cache_data(show_spinner=False)
def load_summary():
    if not _SUMMARY.exists():
        return None
    return pd.read_csv(_SUMMARY)

# ---------------------------------------------------------------------------
# Severity colours
# ---------------------------------------------------------------------------

SEV_COLOR = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}

def _dot_color(sev: str) -> str:
    return SEV_COLOR.get(sev, "#6c757d")

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Hotspot Map")
st.caption("Spatial clustering of Bengaluru traffic incidents")

df        = load_features()
geojson   = load_geojson()
heat_pts  = load_heatmap()
summary   = load_summary()

if df is None:
    st.error("Run `python src/hotspot_engine.py` from smartflow/ to generate map data.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    all_causes = sorted(df["event_cause"].unique().tolist())
    sel_causes = st.multiselect("Event Cause", all_causes, default=all_causes)

    all_zones = sorted(df["zone"].dropna().unique().tolist())
    sel_zones = st.multiselect("Zone", all_zones, default=all_zones)

    show_markers = st.checkbox("Show individual event markers", value=False)
    show_clusters = st.checkbox("Show cluster polygons", value=True)

    if df["start_datetime"].notna().any():
        dt = pd.to_datetime(df["start_datetime"])
        min_date = dt.min().date()
        max_date = dt.max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
    else:
        date_range = None

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

mask = df["event_cause"].isin(sel_causes) & df["zone"].isin(sel_zones)
if date_range and len(date_range) == 2:
    dt_col = pd.to_datetime(df["start_datetime"])
    mask &= dt_col.dt.date.between(date_range[0], date_range[1])

fdf = df[mask]

# ---------------------------------------------------------------------------
# Metrics strip
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered Events", f"{len(fdf):,}")
c2.metric("High Severity",   f"{(fdf['severity_class']=='High').sum():,}")
c3.metric("Corridors",       fdf["corridor"].nunique())
c4.metric("Clusters",
          len(geojson["features"]) if geojson else "—")

st.divider()

# ---------------------------------------------------------------------------
# Build Folium map
# ---------------------------------------------------------------------------

m = folium.Map(
    location=[12.97, 77.59],
    zoom_start=12,
    tiles="CartoDB dark_matter",
)

# Layer 1: KDE heatmap
if heat_pts:
    HeatMap(
        heat_pts,
        name="Density Heatmap",
        min_opacity=0.3,
        max_zoom=16,
        radius=18,
        blur=15,
        gradient={0.2: "blue", 0.45: "lime", 0.65: "yellow", 1.0: "red"},
    ).add_to(m)

# Layer 2: DBSCAN cluster polygons
if geojson and show_clusters:
    def _style(feature):
        count = feature["properties"].get("event_count", 0)
        opacity = min(0.7, 0.2 + count / 200)
        return {"fillColor": "#ff6b35", "color": "#ff6b35",
                "weight": 2, "fillOpacity": opacity}

    folium.GeoJson(
        geojson,
        name="Hotspot Clusters",
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["top_junction", "event_count", "dominant_cause",
                    "avg_duration_minutes", "zone"],
            aliases=["Junction:", "Events:", "Main Cause:",
                     "Avg Duration (min):", "Zone:"],
            localize=True,
        ),
    ).add_to(m)

# Layer 3: Individual event markers (via MarkerCluster for performance)
if show_markers and len(fdf) > 0:
    cluster_layer = MarkerCluster(name="Individual Events").add_to(m)
    sample = fdf.sample(min(500, len(fdf)), random_state=42)
    for _, row in sample.iterrows():
        sev  = row.get("severity_class", "Low")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=_dot_color(sev),
            fill=True,
            fill_opacity=0.8,
            tooltip=(
                f"<b>{row.get('junction','—')}</b><br>"
                f"Cause: {row.get('event_cause','—')}<br>"
                f"Severity: {sev}<br>"
                f"Duration: {row.get('duration_minutes','—')} min"
            ),
        ).add_to(cluster_layer)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=None, height=600, returned_objects=[])

# ---------------------------------------------------------------------------
# Top hotspot table
# ---------------------------------------------------------------------------

if summary is not None and len(summary) > 0:
    st.subheader("Top Hotspot Junctions")
    display_cols = [c for c in
        ["junction", "event_count", "dominant_cause",
         "avg_duration_minutes", "dominant_severity", "zone"]
        if c in summary.columns]
    st.dataframe(
        summary[display_cols].rename(columns={
            "junction":             "Junction",
            "event_count":          "Events",
            "dominant_cause":       "Main Cause",
            "avg_duration_minutes": "Avg Duration (min)",
            "dominant_severity":    "Severity",
            "zone":                 "Zone",
        }),
        use_container_width=True,
        hide_index=True,
    )
