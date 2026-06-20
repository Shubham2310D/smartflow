"""
Page 6 — Roster Optimizer (multi-event conflict view)

The per-event Resource Plan answers "how many officers does this event need?".
This page answers the operational question it can't: when several events are
active at once and officers are scarce, who gets them? A min-cost-flow allocator
distributes a fixed roster across the concurrent set, serving high-priority
events first and minimising travel.

Two inputs are ASSUMPTIONS (the data has neither) and are surfaced as such:
  • the officer roster / station capacity, and
  • the concurrency scenario (events that started within one clock-hour).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from roster_optimizer import (
    build_scenario,
    optimize,
    roster_capacities,
    station_locations,
)
from utils import CAUSE_DISPLAY

st.set_page_config(page_title="Roster Optimizer | SmartFlow", page_icon="🚓", layout="wide")

_FEATS = _ROOT / "data" / "processed" / "features.csv"

_SEV_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}


@st.cache_data(show_spinner=False)
def load_features():
    if not _FEATS.exists():
        return None
    return pd.read_csv(_FEATS, parse_dates=["start_datetime"])


@st.cache_data(show_spinner=False)
def busiest_windows(_df, n=15):
    pool = _df[_df["status"] == "active"].dropna(subset=["latitude", "longitude", "start_datetime"])
    h = pool["start_datetime"].dt.tz_localize(None).dt.floor("h")
    vc = h.value_counts().head(n)
    return [(ts, int(c)) for ts, c in vc.items()]


st.title("🚓 Roster Optimizer — Multi-Event Allocation")
st.caption(
    "Allocates a fixed officer roster across simultaneously-active events via "
    "min-cost flow: high-priority events first, travel minimised."
)

df = load_features()
if df is None or "status" not in df.columns:
    st.error("features.csv not found or missing `status`. Run the pipeline first.")
    st.stop()

st.warning(
    "**Two inputs are modelling assumptions — the dataset contains neither.** "
    "**(1) Roster:** there is no staffing table, so station capacity is illustrative "
    "(tune it below). **(2) Concurrency:** 5 months of incidents aren't naturally "
    "simultaneous, so the scenario is the set of events that *started within one "
    "clock-hour*. Everything else — demand, travel, the allocation — is real."
)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
locs = station_locations(df)
windows = busiest_windows(df)

c1, c2 = st.columns([2, 1])
with c1:
    labels = [f"{ts:%Y-%m-%d %H:%M}  ·  {c} events active" for ts, c in windows]
    pick = st.selectbox("Concurrency scenario (hour window)", range(len(windows)),
                        format_func=lambda i: labels[i])
    when = windows[pick][0]
with c2:
    ops = st.slider("Officers per station (assumed roster)", 1, 15, 6,
                    help="Uniform capacity across all stations. Lower it to see "
                         "the allocator triage under scarcity.")

scen = build_scenario(df, when=when)
events = scen["events"]
caps = roster_capacities(list(locs.keys()), officers_per_station=ops)
result = optimize(events, caps, locs)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
total_unmet = sum(result["unmet"].values())
under = sum(1 for e in events if result["unmet"][e["id"]] > 0)
pct_met = (result["met"] / result["total_demand"] * 100) if result["total_demand"] else 100.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active events", len(events))
m2.metric("Officer demand", result["total_demand"])
m3.metric("Roster size", result["roster_size"])
m4.metric("Demand met", f"{pct_met:.0f}%", f"{total_unmet} short" if total_unmet else "fully covered",
          delta_color="inverse")
m5.metric("Total travel", f"{result['total_travel_km']:.0f} km")

if total_unmet:
    st.error(
        f"⚠️ **{under} event(s) under-resourced** — {total_unmet} officers short of demand. "
        "The allocator left the lowest-priority demand unmet first."
    )
else:
    st.success("✅ Roster covers all concurrent demand at minimum travel.")

# ---------------------------------------------------------------------------
# Map: events (color = fulfilment) + stations (size = capacity)
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Allocation map")
    center = [df["latitude"].mean(), df["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    for s, (lat, lon) in locs.items():
        cap = caps.get(s, 0)
        folium.CircleMarker(
            [lat, lon], radius=5 + cap, color="#1f6feb", fill=True, fill_opacity=0.6,
            tooltip=f"Station: {s} · capacity {cap}",
        ).add_to(fmap)

    for e in events:
        short = result["unmet"][e["id"]]
        if short == 0:
            color = "#28a745"           # fully met
        elif short < e["demand"]:
            color = "#ffc107"           # partially met
        else:
            color = "#dc3545"           # unmet
        sent = sum(result["allocations"][e["id"]].values())
        folium.CircleMarker(
            [e["lat"], e["lon"]], radius=4 + e["demand"],
            color=color, fill=True, fill_opacity=0.75,
            tooltip=(f"{CAUSE_DISPLAY.get(e['cause'], e['cause'])} · {e['severity']} · "
                     f"demand {e['demand']} · sent {sent} · short {short}"),
        ).add_to(fmap)

    st_folium(fmap, width=None, height=460, returned_objects=[])

with right:
    st.subheader("Demand vs. allocation by severity")
    rows = []
    for e in events:
        rows.append({"Severity": e["severity"],
                     "Officers sent": sum(result["allocations"][e["id"]].values()),
                     "Short": result["unmet"][e["id"]]})
    agg = pd.DataFrame(rows).groupby("Severity")[["Officers sent", "Short"]].sum().reset_index()
    order = [s for s in ["High", "Medium", "Low"] if s in set(agg["Severity"])]
    fig = px.bar(agg.melt(id_vars="Severity", var_name="Type", value_name="Officers"),
                 x="Severity", y="Officers", color="Type",
                 category_orders={"Severity": order},
                 color_discrete_map={"Officers sent": "#28a745", "Short": "#dc3545"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-event allocation table
# ---------------------------------------------------------------------------
st.subheader("Per-event allocation")
table = []
for e in events:
    alloc = result["allocations"][e["id"]]
    table.append({
        "Event": e["id"],
        "Severity": _SEV_ICON.get(e["severity"], e["severity"]),
        "Cause": CAUSE_DISPLAY.get(e["cause"], e["cause"]),
        "Zone": e["zone"],
        "Demand": e["demand"],
        "Sent": sum(alloc.values()),
        "Short": result["unmet"][e["id"]],
        "Dispatched from": ", ".join(f"{s}×{n}" for s, n in sorted(alloc.items())) or "—",
    })
tbl = pd.DataFrame(table).sort_values(["Short", "Demand"], ascending=[False, False])
st.dataframe(tbl, use_container_width=True, hide_index=True)
