"""
Page 3 — Resource Plan
Shows deployment recommendation card based on the last prediction (Page 2)
or a standalone form if no prediction is in session_state.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from event_analog import PLANNED_CAUSES, find_analogs
from outcomes_log import log_decision
from resource_recommender import recommend
from utils import ALL_CAUSES, ALL_ZONES, CAUSE_DISPLAY

st.set_page_config(page_title="Resource Plan | SmartFlow", page_icon="🚔", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLAG_COLOR = {"URGENT": "#dc3545", "PRIORITY": "#ffc107", "ROUTINE": "#28a745"}
_SEV_COLOR  = {"High":   "#dc3545", "Medium":   "#ffc107", "Low":     "#28a745"}


def _card(label: str, value, unit: str = "", color: str = "#0d6efd"):
    st.markdown(
        f"""
        <div style="background:{color};color:white;padding:16px 20px;
                    border-radius:10px;text-align:center;margin-bottom:4px;">
          <div style="font-size:0.85em;opacity:0.85;">{label}</div>
          <div style="font-size:1.8em;font-weight:700;">{value}{(' ' + unit) if unit else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Populate inputs from session_state (last prediction) or defaults
# ---------------------------------------------------------------------------

prev = st.session_state.get("last_prediction")

st.title("Resource Deployment Plan")
st.caption("Personnel and logistics recommendation for traffic officers")

# ---------------------------------------------------------------------------
# Optional override form (sidebar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Override Inputs")
    st.caption("Values pre-filled from the last prediction on Page 2.")

    if prev:
        default_sev     = prev["severity"]
        default_cause   = prev["event_cause"]
        default_closure = prev["road_closure"]
        default_hour    = prev["hour_of_day"]
        default_zone    = prev["zone"]
        default_dur     = prev["duration_minutes"]
    else:
        default_sev, default_cause, default_closure = "Medium", "vehicle_breakdown", False
        default_hour, default_zone, default_dur     = 9, "Central Zone 1", None

    sev_choice = st.selectbox(
        "Severity", ["High", "Medium", "Low"],
        index=["High", "Medium", "Low"].index(default_sev),
    )

    cause_display_list = [CAUSE_DISPLAY.get(c, c) for c in ALL_CAUSES]
    default_cause_display = CAUSE_DISPLAY.get(default_cause, default_cause)
    if default_cause_display not in cause_display_list:
        default_cause_display = cause_display_list[0]
    cause_sel = st.selectbox(
        "Event Cause", cause_display_list,
        index=cause_display_list.index(default_cause_display),
    )
    cause_key = ALL_CAUSES[cause_display_list.index(cause_sel)]

    closure_sel = st.checkbox("Road Closure", value=bool(default_closure))
    hour_sel    = st.slider("Hour of Day", 0, 23, int(default_hour))
    zone_sel    = st.selectbox(
        "Zone", ALL_ZONES,
        index=ALL_ZONES.index(default_zone) if default_zone in ALL_ZONES else 0,
    )

    dur_override = st.number_input(
        "Predicted duration (min, 0 = auto)",
        min_value=0, max_value=10080,
        value=int(default_dur) if default_dur else 0,
    )

    recalc = st.button("Recalculate", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Compute recommendation
# ---------------------------------------------------------------------------

duration_for_rec = float(dur_override) if dur_override > 0 else (
    default_dur if default_dur else None
)

# Carry the calibrated closure probability from the last prediction (Page 2)
closure_prob = prev.get("closure_probability") if prev else None

rec = recommend(
    severity_class        = sev_choice,
    event_cause           = cause_key,
    requires_road_closure = closure_sel,
    hour_of_day           = hour_sel,
    zone                  = zone_sel,
    duration_minutes      = duration_for_rec,
    closure_probability   = closure_prob,
)

# ---------------------------------------------------------------------------
# Priority banner
# ---------------------------------------------------------------------------

flag       = rec["priority_flag"]
flag_color = _FLAG_COLOR.get(flag, "#6c757d")
sev_color  = _SEV_COLOR.get(sev_choice, "#6c757d")

st.markdown(
    f"""
    <div style="background:{flag_color};color:white;padding:14px 24px;
                border-radius:12px;text-align:center;font-size:1.5em;
                font-weight:800;letter-spacing:0.1em;margin-bottom:20px;">
        {flag}
    </div>
    """,
    unsafe_allow_html=True,
)

if prev and not recalc:
    st.info(
        f"Showing recommendation for the prediction from Page 2 — "
        f"**{prev['severity']}** severity, cause: **{CAUSE_DISPLAY.get(prev['event_cause'], prev['event_cause'])}**"
    )

# ---------------------------------------------------------------------------
# Main cards
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    _card("Personnel to Deploy", rec["personnel_count"], "officers", "#0d6efd")

with c2:
    bar_val   = "YES" if rec["barricade_required"] else "NO"
    bar_color = "#dc3545" if rec["barricade_required"] else "#6c757d"
    _card("Barricade Required", bar_val, color=bar_color)

with c3:
    div_val   = "YES" if rec["diversion_recommended"] else "NO"
    div_color = "#fd7e14" if rec["diversion_recommended"] else "#6c757d"
    _card("Diversion Recommended", div_val, color=div_color)

with c4:
    _card(
        "Typical Clearance",
        f"{rec['estimated_clearance_minutes']}",
        f"min  ({rec.get('clearance_low','?')}–{rec.get('clearance_high','?')})",
        "#198754",
    )

st.caption(
    "Clearance is the **median historical close-time** for this cause (with IQR), "
    "not a model forecast — see the Feedback Loop page for why."
    + (f"  ·  Model road-closure likelihood: **{closure_prob*100:.0f}%**."
       if closure_prob is not None else "")
)

st.divider()

# ---------------------------------------------------------------------------
# Planned-event historical analogs (the forecastable half of the brief)
# ---------------------------------------------------------------------------

if cause_key in PLANNED_CAUSES:
    st.divider()
    st.subheader("Similar Past Events (Case-Based Forecast)")
    st.caption(
        "Planned events have a known type and place ahead of time. Here is what "
        "the most similar past events actually required — grounded in history, not a model."
    )
    analog = find_analogs(cause_key, zone=zone_sel)
    if analog.get("found"):
        a1, a2, a3 = st.columns(3)
        a1.metric(f"Past {CAUSE_DISPLAY.get(cause_key, cause_key)}s", analog["found"],
                  help=f"Scope: {analog['scope']}")
        if analog.get("median_clearance") is not None:
            a2.metric("Their median clearance",
                      f"{analog['median_clearance']:.0f} min",
                      delta=f"range {analog.get('p25_clearance',0):.0f}–{analog.get('p75_clearance',0):.0f}",
                      delta_color="off")
        if analog.get("closure_rate") is not None:
            a3.metric("Needed road closure", f"{analog['closure_rate']:.0f}%")
        if analog.get("samples") is not None and len(analog["samples"]):
            with st.expander("View the matched past events"):
                st.dataframe(analog["samples"], use_container_width=True, hide_index=True)
    else:
        st.info("No comparable past events of this type in the dataset.")

# ---------------------------------------------------------------------------
# Dispatch & context
# ---------------------------------------------------------------------------

d1, d2, d3 = st.columns(3)

def _info_box(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding:12px 16px;border-radius:8px;
                    background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);">
          <div style="font-size:0.8em;color:#aaa;margin-bottom:4px;">{label}</div>
          <div style="font-size:1.2em;font-weight:600;word-wrap:break-word;
                      overflow-wrap:break-word;white-space:normal;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with d1:
    _info_box("Dispatch From", rec["dispatch_from"])

with d2:
    _info_box("Event Cause", CAUSE_DISPLAY.get(cause_key, cause_key))

with d3:
    _info_box("High-Incident Window", "Yes" if rec["is_peak_hour"] else "No")

# Rationale
st.divider()
st.subheader("Decision Rationale")
st.markdown(
    f"> {rec['rationale']}"
)

# ---------------------------------------------------------------------------
# Scoring breakdown table
# ---------------------------------------------------------------------------

with st.expander("Scoring Breakdown"):
    base = {"Low": 2, "Medium": 3, "High": 5}[sev_choice]
    peak_bonus    = 1 if rec["is_peak_hour"] and sev_choice in {"Medium", "High"} else 0
    closure_bonus = 1 if closure_sel else 0
    cause_bonus   = 1 if cause_key in {"accident", "flood", "water_logging"} else 0

    import pandas as pd
    breakdown = pd.DataFrame({
        "Factor":     ["Base personnel", "High-incident window bonus", "Road closure bonus", "High-risk cause bonus", "TOTAL"],
        "Points":     [base, peak_bonus, closure_bonus, cause_bonus, base + peak_bonus + closure_bonus + cause_bonus],
    })
    st.table(breakdown)

# ---------------------------------------------------------------------------
# Record this decision into the learning loop
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Record Decision (Learning Loop)")
st.caption(
    "Log this recommendation so it can be compared against the real outcome later. "
    "Recorded decisions feed the **Feedback Loop** page and future model retraining."
)

with st.form("log_decision_form"):
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        actual_clearance = st.number_input(
            "Actual clearance (min, 0 = not yet known)",
            min_value=0, max_value=10080, value=0,
        )
    with lc2:
        actual_sev = st.selectbox(
            "Actual severity (optional)", ["(unknown)", "High", "Medium", "Low"],
        )
    with lc3:
        followed = st.selectbox("Plan followed?", ["(unknown)", "yes", "no"])

    log_clicked = st.form_submit_button("Log this decision", use_container_width=True)

if log_clicked:
    log_decision({
        "event_cause":             cause_key,
        "zone":                    zone_sel,
        "hour_of_day":             hour_sel,
        "predicted_severity":      sev_choice,
        "predicted_clearance_min": rec["estimated_clearance_minutes"],
        "recommended_personnel":   rec["personnel_count"],
        "actual_clearance_min":    actual_clearance if actual_clearance > 0 else "",
        "actual_severity":         "" if actual_sev == "(unknown)" else actual_sev,
        "followed":                "" if followed == "(unknown)" else followed,
    })
    st.success("Decision logged. See the **Feedback Loop** page for predicted-vs-actual tracking.")
