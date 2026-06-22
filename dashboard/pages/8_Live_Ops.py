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

from utils import inject_responsive_css  # noqa: E402
inject_responsive_css()

_FEATS = _ROOT / "data" / "processed" / "features.csv"
_CLO   = _ROOT / "models" / "closure_predictor.pkl"
_SEV_COLOR = {"High": "#dc3545", "Medium": "#fd7e14", "Low": "#28a745"}


@st.cache_data(show_spinner=False, ttl=10)
def load_active():
    """Active events (rich features) + a live closure probability per event."""
    seed_from_features(_ROOT)                      # populate the store if empty
    n_store = count(_ROOT)
    if not _FEATS.exists():
        return None, n_store
    df = pd.read_csv(_FEATS)
    df = df[df.get("status") == "active"].copy() if "status" in df.columns else df

    # Merge LIVE incidents added through the API (event store rows that aren't
    # already in features.csv) so real-time events show on the map and feed the
    # convex hulls — without a pipeline re-run, and without double-counting the
    # seeded batch rows (those share features.csv ids).
    live = active_events(_ROOT)
    if not live.empty:
        known = set(df["id"].dropna()) if "id" in df.columns else set()
        adds = live[~live["id"].isin(known)] if "id" in live.columns else live
        if not adds.empty:
            adds = adds.rename(columns={"severity": "severity_class",
                                        "cluster": "cluster_label", "ts": "start_datetime"})
            df = pd.concat([df, adds], ignore_index=True)

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

if st.button("🔄 Refresh live data",
             help="Pull the latest incidents, including ones just added via the real-time API."):
    load_active.clear()
    st.rerun()

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
    # Cap markers: thousands of CircleMarkers bloat the Folium HTML and memory.
    # Show the highest-impact incidents (the ones an operator acts on first).
    _MAP_CAP = 300
    map_rows = rdf.dropna(subset=["lat", "lon"]).sort_values("impact", ascending=False)
    if len(map_rows) > _MAP_CAP:
        st.caption(f"Showing the {_MAP_CAP} highest-impact of {len(map_rows)} "
                   "located incidents (map performance).")
        map_rows = map_rows.head(_MAP_CAP)
    for _, e in map_rows.iterrows():
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
        f"  [{r['lat']:.4f}, {r['lon']:.4f}]": r.to_dict()
        for _, r in divertable.head(40).iterrows()
    }
    pick = st.selectbox("Incident to reroute", list(labels.keys()))
    row = labels[pick]
    lat, lon = float(row["lat"]), float(row["lon"])
    key = f"{lat:.5f},{lon:.5f}"

    # Routing loads the OSM road graph into memory, so compute ONLY on demand
    # (a click), never on every page render — this keeps the console lightweight
    # on a small host (the graph isn't touched until someone asks for a reroute).
    if st.button("🧭 Compute reroute", type="primary"):
        st.session_state["divplan"] = {"key": key, "plan": _diversion(lat, lon), "row": row}

    state = st.session_state.get("divplan")
    if not state or state["key"] != key:
        st.caption("Pick an incident and click **Compute reroute** — the road graph "
                   "loads on demand to keep the console lightweight.")
    else:
        plan, row = state["plan"], state["row"]
        if not plan.get("feasible"):
            st.warning(f"No reroute available: {plan.get('reason', 'unknown')}.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Detour length", f"{plan['detour_m']:.0f} m")
            m2.metric("Added distance", f"+{plan['extra_m']:.0f} m",
                      delta=f"{plan['extra_pct']:.0f}% longer" if plan.get("extra_pct") else None,
                      delta_color="inverse")
            m3.metric("Segments", plan["n_segments"])

            show_div_hull = st.checkbox(
                "Show diversion-affected area (convex hull)", value=True,
                help="Convex hull over the reroute geometry — the area this diversion spans.")

            dmap = folium.Map(location=plan["blocked_point"], zoom_start=15,
                              tiles="cartodbpositron")
            folium.Marker(plan["blocked_point"], tooltip="Blockage",
                          icon=folium.Icon(color="red", icon="ban", prefix="fa")).add_to(dmap)
            folium.Circle(plan["blocked_point"], radius=plan["closure_radius_m"],
                          color="#dc3545", fill=True, fill_opacity=0.15,
                          tooltip="Closure zone").add_to(dmap)
            # Convex hull over the detour + blockage = the area the diversion affects.
            # Drawn under the route so the line stays crisp on top.
            if show_div_hull:
                from scipy.spatial import ConvexHull  # noqa: PLC0415
                pts = [list(p) for p in plan["detour_path"]] + [list(plan["blocked_point"])]
                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)          # raises if route is collinear
                        ring = [pts[v] for v in hull.vertices]
                        folium.Polygon(ring, color="#6f42c1", weight=2, fill=True,
                                       fill_opacity=0.10,
                                       tooltip="Diversion-affected area (convex hull)").add_to(dmap)
                    except Exception:
                        pass                            # degenerate route — skip the hull
            folium.PolyLine(plan["detour_path"], color="#1f77b4", weight=5, opacity=0.85,
                            tooltip=plan["summary"]).add_to(dmap)
            folium.Marker(plan["detour_path"][0], tooltip="Detour entry",
                          icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(dmap)
            folium.Marker(plan["detour_path"][-1], tooltip="Detour exit",
                          icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(dmap)
            st_folium(dmap, width=None, height=420, returned_objects=[])
            st.caption(plan["summary"])

        # --- Push the alert to officers' phones (Telegram) -------------------
        from notifications import notify_incident, notify_status, invite_link
        n_ok = notify_status(_ROOT)["available"]
        link = invite_link(_ROOT)
        bcol1, bcol2 = st.columns([1, 2])
        with bcol1:
            send = st.button("🔔 Alert officers", use_container_width=True, disabled=not n_ok,
                             type="primary")
        with bcol2:
            if not n_ok:
                st.caption("Telegram alerts off — set `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` "
                           "(env or config) to enable.")
            elif link:
                st.caption(f"📲 [Join the alerts group]({link}) to receive these "
                           "notifications on your phone.")
        if send:
            res = notify_incident({
                "severity": row.get("severity"), "event_cause": row.get("cause"),
                "zone": row.get("zone"), "closure_prob": row.get("closure_prob"),
                "personnel": row.get("personnel"), "dispatch_from": row.get("station"),
                "barricade": bool(row.get("barricade")),
                "diversion_summary": plan["summary"] if plan.get("feasible") else None,
                "location": (lat, lon),
            }, project_root=_ROOT)
            if res.get("sent"):
                st.success("📲 Alert sent to the officers' Telegram group.")
            else:
                st.error(f"Alert not sent: {res.get('reason', 'unknown error')}")

st.caption(
    "Recommendations use the same rule engine as the Resource Plan; closure "
    "likelihood is the calibrated model; impact is the transparent heuristic; the "
    "diversion is routed on the OSM road graph. Active events are served from the "
    "SQLite store the real-time API also writes to. Officer alerts go out over "
    "Telegram (opt-in)."
)
