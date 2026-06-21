"""
Page 7 — Event Planner (advance what-if for an upcoming event)

The forecastable half of the brief in one pane: pick an event TYPE, PLACE and
DATE/TIME, and get a full advance plan — case-based impact forecast (expected
severity, road-closure likelihood, clearance range) plus the deployment plan
(personnel, barricade, diversion, dispatch station) — before the event happens.

Everything is grounded in similar past events with an explicit confidence; no
model is invented for the thin gathering data.
"""

import sys
from datetime import date, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from event_analog import PLANNED_OBSTRUCTIONS, PLANNABLE_EVENTS
from event_planner import plan_event
from outcomes_log import log_decision
from utils import ALL_ZONES, CAUSE_DISPLAY, severity_badge

st.set_page_config(page_title="Event Planner | SmartFlow", page_icon="📅", layout="wide")

from utils import inject_responsive_css  # noqa: E402
inject_responsive_css()

_CONF = {
    "high":   ("#198754", "High confidence", "backed by several closely-matched past events"),
    "medium": ("#fd7e14", "Medium confidence", "a moderate number of analogs, or a citywide match"),
    "low":    ("#dc3545", "Low confidence", "few analogs — treat as a rough prior, not a forecast"),
    "none":   ("#6c757d", "No evidence", "no comparable past events of this type"),
}
_SEV_COLOR = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}


def _card(label, value, unit="", color="#0d6efd"):
    st.markdown(
        f"""
        <div style="background:{color};color:white;padding:16px 20px;border-radius:10px;
                    text-align:center;margin-bottom:4px;">
          <div style="font-size:0.85em;opacity:0.85;">{label}</div>
          <div style="font-size:1.7em;font-weight:700;">{value}{(' ' + unit) if unit else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("📅 Event Planner — Advance Plan for an Upcoming Event")
st.caption(
    "Pick a planned event's type, place and time → a full deployment plan and "
    "impact forecast, grounded in similar past events (with a confidence rating)."
)

# ---------------------------------------------------------------------------
# Planned-event calendar (Layer-B forecasting seam). The dataset's planned
# events are a *log of the past*; a real calendar (fixtures, permits, rallies)
# is what makes this true advance forecasting. It's optional and off by default.
# ---------------------------------------------------------------------------
with st.expander("📆 Planned-event calendar (advance forecasting source)"):
    from external_feeds import load_calendar
    cal = load_calendar(_ROOT)
    if cal["available"]:
        st.success(f"{cal['n']} planned events loaded. Each can be run through the "
                   "planner below for a batch advance deployment plan.")
        st.dataframe(cal["events"], use_container_width=True, hide_index=True)
    else:
        st.info(
            f"No calendar wired in ({cal['reason']}). Without it, the planner uses "
            "the manual inputs below + case-retrieval from past events — it does not "
            "invent a forecast. Populate `data/external/event_calendar.csv` "
            "(stadium fixtures, festival/permit/rally schedules) to drive batch "
            "advance planning. See `src/external_feeds.py`."
        )

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([2, 2, 1.4, 1])
with c1:
    def _label(c):
        tag = "  🚧 obstruction" if c in PLANNED_OBSTRUCTIONS else "  👥 gathering"
        return CAUSE_DISPLAY.get(c, c) + tag
    cause = st.selectbox("Event type", PLANNABLE_EVENTS, format_func=_label)
with c2:
    zone = st.selectbox("Place (zone)", ALL_ZONES,
                        index=ALL_ZONES.index("Central Zone 1") if "Central Zone 1" in ALL_ZONES else 0)
with c3:
    ev_date = st.date_input("Date", value=date(2026, 6, 21))
with c4:
    ev_time = st.time_input("Start time", value=time(18, 0))

hour = ev_time.hour
plan = plan_event(cause, zone=zone, hour=hour)
analog = plan["analog"]
rec = plan["recommendation"]

# ---------------------------------------------------------------------------
# Confidence banner
# ---------------------------------------------------------------------------
conf_color, conf_label, conf_note = _CONF.get(plan["confidence"], _CONF["none"])
kind = "sustained obstruction (no crowd)" if plan["is_obstruction"] else "crowd / gathering event"
st.markdown(
    f"""
    <div style="background:{conf_color};color:white;padding:12px 20px;border-radius:10px;
                margin:6px 0 14px;">
      <b>{conf_label}</b> — {conf_note}.
      &nbsp;|&nbsp; {analog.get('found', 0)} analog(s), matched {analog.get('scope', 'n/a')}
      &nbsp;|&nbsp; treated as a {kind}.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Impact forecast (from history)
# ---------------------------------------------------------------------------
st.subheader("Forecast impact (from similar past events)")
f1, f2, f3, f4 = st.columns(4)
with f1:
    _card("Expected severity", severity_badge(plan["expected_severity"]),
          color=_SEV_COLOR.get(plan["expected_severity"], "#6c757d"))
with f2:
    cl = plan["closure_likelihood"]
    _card("Road-closure likelihood", f"{cl*100:.0f}%" if cl is not None else "—",
          color="#6f42c1")
with f3:
    mc = analog.get("median_clearance")
    rng = (f"{analog.get('p25_clearance',0):.0f}–{analog.get('p75_clearance',0):.0f}"
           if mc is not None else "")
    _card("Typical clearance", f"{mc:.0f}" if mc is not None else "—",
          f"min ({rng})" if rng else "", "#198754")
with f4:
    _card("Analogs found", analog.get("found", 0), color="#0dcaf0")

# ---------------------------------------------------------------------------
# Deployment plan (rule engine, seeded with the forecast)
# ---------------------------------------------------------------------------
st.subheader("Recommended deployment")
d1, d2, d3, d4 = st.columns(4)
with d1:
    _card("Personnel", rec["personnel_count"], "officers", "#0d6efd")
with d2:
    _card("Barricade", "YES" if rec["barricade_required"] else "NO",
          color="#dc3545" if rec["barricade_required"] else "#6c757d")
with d3:
    _card("Diversion", "YES" if rec["diversion_recommended"] else "NO",
          color="#fd7e14" if rec["diversion_recommended"] else "#6c757d")
with d4:
    _card("Dispatch from", rec["dispatch_from"], color="#20c997")

st.markdown(f"> {rec['rationale']}")
st.caption(
    "Impact is the median outcome of matched past events (closure rate → "
    "barricade decision); clearance is historical close-time, not a model forecast."
)

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
if analog.get("samples") is not None and len(analog["samples"]):
    with st.expander(f"Evidence — the {analog['found']} matched past events"):
        st.dataframe(analog["samples"], use_container_width=True, hide_index=True)
elif not analog.get("found"):
    st.info("No comparable past events of this type — plan shown is a generic prior.")

# ---------------------------------------------------------------------------
# Log the plan into the learning loop
# ---------------------------------------------------------------------------
st.divider()
if st.button("Log this plan (learning loop)", use_container_width=True):
    log_decision({
        "event_cause":             cause,
        "zone":                    zone,
        "hour_of_day":             hour,
        "predicted_severity":      plan["expected_severity"],
        "predicted_clearance_min": rec["estimated_clearance_minutes"],
        "recommended_personnel":   rec["personnel_count"],
    })
    st.success("Plan logged. Compare against the real outcome on the Feedback Loop page.")
