"""
Page 8 — Live Operations Console (command center)

The map-first view a control-room operator actually watches: every currently-
active incident, colour- and size-coded by severity, each with its recommended
deployment and nearest dispatch station — plus an alerts panel for the events
most likely to need a barricade. This is the operational counterpart to the
analytical pages.

Active events are served from the SQLite event store (seeded here from the
historical active incidents); the store is the same one the real-time API writes
to, so this console reflects live state, not a static snapshot.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from event_store import active_events, count, seed_from_features
from impact_score import impact_score
from resource_recommender import recommend
from utils import CAUSE_DISPLAY, severity_badge

st.set_page_config(page_title="Live Ops | SmartFlow", page_icon="🛰️", layout="wide")

_FEATS = _ROOT / "data" / "processed" / "features.csv"
_CLO   = _ROOT / "models" / "closure_predictor.pkl"
_SEV_COLOR = {"High": "#dc3545", "Medium": "#fd7e14", "Low": "#28a745"}


@st.cache_data(show_spinner=False)
def load_active():
    """Active events (rich features) + a live closure probability per event."""
    seed_from_features(_ROOT)                      # populate the store if empty
    n_store = count(_ROOT)
    if not _FEATS.exists():
        return None, n_store
    df = pd.read_csv(_FEATS)
    df = df[df.get("status") == "active"].copy() if "status" in df.columns else df
    if df.empty:
        return df, n_store
    if _CLO.exists():
        clo = joblib.load(_CLO)
        feats = [f for f in clo["features"] if f in df.columns]
        if len(feats) == len(clo["features"]):
            df["closure_prob"] = clo["model"].predict_proba(
                df[clo["features"]].fillna(0).astype(float))[:, 1]
            df.attrs["barricade_threshold"] = clo.get("barricade_threshold", 0.15)
    return df, n_store


st.title("🛰️ Live Operations Console")
st.caption("Active incidents, their recommended deployment, and barricade alerts — one operational pane.")

df, n_store = load_active()
if df is None or df.empty:
    st.info("No active events found. Run the pipeline (status='active' rows feed this console).")
    st.stop()

thr = df.attrs.get("barricade_threshold", 0.15)

# Per-event recommendation + impact (rule-based, fast).
recs = []
for _, r in df.iterrows():
    cp = float(r["closure_prob"]) if "closure_prob" in df.columns and pd.notna(r.get("closure_prob")) else None
    rec = recommend(
        severity_class=str(r.get("severity_class", "Medium")),
        event_cause=r.get("event_cause", "other"),
        requires_road_closure=bool(r.get("road_closure_binary", 0)),
        hour_of_day=int(r.get("hour_of_day", 12)),
        zone=r.get("zone", "Unknown"),
        closure_probability=cp,
        barricade_threshold=thr,
    )
    imp = impact_score(
        road_closure=bool(r.get("road_closure_binary", 0)),
        corridor_7d=r.get("corridor_7d_score", 0),
        is_peak=bool(r.get("is_peak_hour", 0)),
        closure_prob=cp,
        cluster_closure_rate=r.get("cluster_closure_rate"),
        road_class_rank=r.get("road_class_rank"),
        lane_count=r.get("lane_count"),
    )
    recs.append({
        "id": r.get("id"), "lat": r.get("latitude"), "lon": r.get("longitude"),
        "severity": str(r.get("severity_class", "Medium")),
        "cause": r.get("event_cause", "other"), "zone": r.get("zone", "Unknown"),
        "personnel": rec["personnel_count"], "barricade": rec["barricade_required"],
        "station": rec["dispatch_from"], "closure_prob": cp,
        "impact": imp["score"], "impact_label": imp["label"],
    })
rdf = pd.DataFrame(recs)

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Active incidents", len(rdf))
k2.metric("High severity", int((rdf["severity"] == "High").sum()))
k3.metric("Barricade recommended", int(rdf["barricade"].sum()))
k4.metric("Officers needed", int(rdf["personnel"].sum()))
k5.metric("In live store", n_store)

# ---------------------------------------------------------------------------
# Alerts — events most likely to need a barricade / highest impact
# ---------------------------------------------------------------------------
alerts = rdf[(rdf["barricade"]) | (rdf["impact"] >= 50)].sort_values(
    ["impact", "closure_prob"], ascending=False)
if len(alerts):
    st.error(f"🚨 {len(alerts)} priority alert(s) — high impact or barricade-likely. "
             "Pre-position crews before queues form.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Active incident map")
    fmap = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()],
                      zoom_start=11, tiles="cartodbpositron")
    for _, e in rdf.iterrows():
        if pd.isna(e["lat"]) or pd.isna(e["lon"]):
            continue
        radius = {"High": 9, "Medium": 6, "Low": 4}.get(e["severity"], 5)
        cp_txt = f"{e['closure_prob']*100:.0f}%" if e["closure_prob"] is not None else "n/a"
        folium.CircleMarker(
            [e["lat"], e["lon"]], radius=radius,
            color=_SEV_COLOR.get(e["severity"], "#666"), fill=True, fill_opacity=0.8,
            tooltip=(f"{CAUSE_DISPLAY.get(e['cause'], e['cause'])} · {severity_badge(e['severity'])}<br>"
                     f"Impact {e['impact']} ({e['impact_label']}) · closure {cp_txt}<br>"
                     f"Deploy {e['personnel']} from {e['station']}"
                     + (" · BARRICADE" if e["barricade"] else "")),
        ).add_to(fmap)
    st_folium(fmap, width=None, height=460, returned_objects=[])

with right:
    st.subheader("Priority alerts")
    show = (alerts if len(alerts) else rdf.sort_values("impact", ascending=False)).head(40)
    show = show.assign(Severity=show["severity"].map(severity_badge),
                       Cause=show["cause"].map(lambda c: CAUSE_DISPLAY.get(c, c)),
                       Barricade=show["barricade"].map({True: "YES", False: "—"}))
    st.dataframe(
        show[["Severity", "Cause", "zone", "impact", "personnel", "station", "Barricade"]]
        .rename(columns={"zone": "Zone", "impact": "Impact", "personnel": "Officers",
                         "station": "Dispatch"}),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# Diversion planner — a REAL reroute around a chosen blockage (OSM road graph)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧭 Diversion planner")
st.caption(
    "For incidents where a diversion is warranted, this computes an **actual "
    "reroute** around the blockage on the OpenStreetMap road graph — not a "
    "yes/no flag. Pick an incident to see the detour and its added distance."
)


@st.cache_data(show_spinner="Computing reroute…")
def _diversion(lat: float, lon: float):
    from diversion import plan_diversion
    return plan_diversion(lat, lon, project_root=_ROOT)


# Offer the incidents a diversion actually matters for: barricade-likely or
# high-impact, and with a known location.
divertable = rdf[(rdf["lat"].notna()) & (rdf["lon"].notna()) &
                 ((rdf["barricade"]) | (rdf["impact"] >= 50))]
if divertable.empty:
    st.info("No barricade-likely / high-impact incidents with a location to reroute right now.")
else:
    labels = {
        f"{CAUSE_DISPLAY.get(r['cause'], r['cause'])} · {r['zone']} · impact {r['impact']}"
        f"  [{r['lat']:.4f}, {r['lon']:.4f}]": (r["lat"], r["lon"])
        for _, r in divertable.head(40).iterrows()
    }
    pick = st.selectbox("Incident to reroute", list(labels.keys()))
    lat, lon = labels[pick]
    plan = _diversion(float(lat), float(lon))

    if not plan.get("feasible"):
        st.warning(f"No reroute available: {plan.get('reason', 'unknown')}.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Detour length", f"{plan['detour_m']:.0f} m")
        m2.metric("Added distance", f"+{plan['extra_m']:.0f} m",
                  delta=f"{plan['extra_pct']:.0f}% longer" if plan.get("extra_pct") else None,
                  delta_color="inverse")
        m3.metric("Segments", plan["n_segments"])

        dmap = folium.Map(location=plan["blocked_point"], zoom_start=15,
                          tiles="cartodbpositron")
        folium.Marker(plan["blocked_point"], tooltip="Blockage",
                      icon=folium.Icon(color="red", icon="ban", prefix="fa")).add_to(dmap)
        folium.Circle(plan["blocked_point"], radius=plan["closure_radius_m"],
                      color="#dc3545", fill=True, fill_opacity=0.15,
                      tooltip="Closure zone").add_to(dmap)
        folium.PolyLine(plan["detour_path"], color="#1f77b4", weight=5, opacity=0.85,
                        tooltip=plan["summary"]).add_to(dmap)
        folium.Marker(plan["detour_path"][0], tooltip="Detour entry",
                      icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(dmap)
        folium.Marker(plan["detour_path"][-1], tooltip="Detour exit",
                      icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(dmap)
        st_folium(dmap, width=None, height=420, returned_objects=[])
        st.caption(plan["summary"])

st.caption(
    "Recommendations use the same rule engine as the Resource Plan; closure "
    "likelihood is the calibrated model; impact is the transparent heuristic; the "
    "diversion is routed on the OSM road graph. Active events are served from the "
    "SQLite store the real-time API also writes to."
)
