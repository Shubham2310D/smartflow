"""
utils.py — Shared helpers: config loading, zone→station lookup, project root.
"""

from __future__ import annotations

import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the smartflow/ project root regardless of CWD."""
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg_cache: dict = {}


def load_config(project_root: Path | None = None) -> dict:
    """Load config.yaml and cache it."""
    if project_root is None:
        project_root = get_project_root()
    key = str(project_root)
    if key not in _cfg_cache:
        with open(project_root / "config.yaml") as f:
            _cfg_cache[key] = yaml.safe_load(f)
    return _cfg_cache[key]


# ---------------------------------------------------------------------------
# Peak hours — single source of truth for train AND serve.
#
# Defined in config.yaml as a data-derived high-incident-load window (the hours
# whose volume exceeds the daily mean), NOT the assumed 08-10/17-20 commuter
# rush. The raw timestamps' wall-clock already behaves as Bengaluru local time,
# so hour_of_day is used as-is at both training and inference (no UTC->IST
# conversion — that would empty the evening peak and invent a 2 AM one).
# Centralising the definition here guarantees feature_engineering (training),
# the recommender, the API, and the dashboard all label "peak" identically.
# ---------------------------------------------------------------------------

# Fallback if config is missing/unreadable: above-mean freight window.
_DEFAULT_PEAK_HOURS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23})


def peak_hours() -> frozenset[int]:
    """Return the set of hours (0-23) treated as high-incident-load 'peak'."""
    try:
        hrs = load_config().get("resource_rules", {}).get("peak_hours")
        if hrs:
            return frozenset(int(h) for h in hrs)
    except Exception:
        pass
    return _DEFAULT_PEAK_HOURS


def is_peak_hour(hour: int) -> bool:
    """True if the given hour-of-day falls in the data-derived peak window."""
    return int(hour) in peak_hours()


# ---------------------------------------------------------------------------
# Severity display — colour-blind-safe cues.
# Colour alone (red/amber/green) is invisible to ~8% of men, so every severity
# display pairs the colour with a distinct GEOMETRIC SHAPE and the text label.
# ---------------------------------------------------------------------------

SEVERITY_SHAPE = {"High": "▲", "Medium": "●", "Low": "■"}


def severity_badge(sev: str) -> str:
    """'High' -> '▲ High' — a shape + text cue that needs no colour to read."""
    return f"{SEVERITY_SHAPE.get(sev, '◆')} {sev}"


# ---------------------------------------------------------------------------
# Responsive / mobile UI — injected on every dashboard page.
#
# Streamlit is desktop-first: columns stay side-by-side on phones (squishing
# metrics and cards into unreadable slivers), the content padding wastes space,
# and headings are oversized for a 360 px screen. A field officer opening this on
# a phone needs the layout to STACK. This injects one stylesheet that, below a
# tablet/phone breakpoint, forces column rows to wrap to full width, trims
# padding, scales headings, and keeps maps/tables/charts inside the viewport so
# the page never scrolls sideways. Call it once per page, right after
# st.set_page_config(). Lazy streamlit import keeps utils usable by the
# (non-Streamlit) data pipeline.
# ---------------------------------------------------------------------------

_RESPONSIVE_CSS = """
<style>
/* Never let the page scroll sideways on a phone. */
[data-testid="stAppViewContainer"] { overflow-x: hidden; }

/* Tablets and phones: let column rows wrap instead of squishing. */
@media (max-width: 900px) {
  .block-container { padding-left: 1.1rem !important; padding-right: 1.1rem !important;
                     padding-top: 2.6rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.6rem !important; }
  /* Each column drops to at least half-width (two-up) on tablets. */
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
      flex: 1 1 48% !important; min-width: 48% !important;
  }
}

/* Phones: full-width stacking + tighter type. */
@media (max-width: 640px) {
  .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important;
                     padding-bottom: 3rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
      flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important;
  }
  h1 { font-size: 1.5rem !important; line-height: 1.25 !important; }
  h2 { font-size: 1.2rem !important; }
  h3 { font-size: 1.05rem !important; }
  /* Metric values can overflow their box on narrow screens. */
  [data-testid="stMetricValue"] { font-size: 1.35rem !important; }
  /* Wide tables/maps scroll within their own box, not the page. */
  [data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto !important; }
}

/* Custom HTML cards (impact/severity/plan tiles) shouldn't overflow their column. */
div[style*="border-radius"] { max-width: 100%; box-sizing: border-box; word-wrap: break-word; }

/* Keep folium/plotly inside the viewport width. */
iframe, .stPlotlyChart { max-width: 100% !important; }
</style>
"""


def inject_responsive_css() -> None:
    """Inject the mobile/responsive stylesheet. Call after st.set_page_config()."""
    import streamlit as st  # lazy: only when a dashboard page calls it
    st.markdown(_RESPONSIVE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Zone → police station mapping
# (covers all zone values seen in the Astram dataset)
# ---------------------------------------------------------------------------

ZONE_STATIONS: dict[str, list[str]] = {
    # North
    "North Zone 1":                    ["Hebbala", "Hennuru", "Byatarayanapura"],
    "North Zone 2":                    ["Kodigehalli", "Byatarayanapura", "Hebbala"],
    "Bengaluru North Corporation":     ["Hebbala", "Byatarayanapura", "Hennuru"],
    # Central
    "Central Zone 1":                  ["Sadashivanagar", "Cubbon Park", "Halasur"],
    "Central Zone 2":                  ["Cubbon Park", "Sadashivanagar", "Halasur"],
    "Bengaluru Central Corporation":   ["Cubbon Park", "Sadashivanagar", "Halasur"],
    # South
    "South Zone 1":                    ["Jayanagara", "Wilson Garden", "Madiwala"],
    "South Zone 2":                    ["Madiwala", "Jayanagara", "HSR Layout"],
    "Bengaluru South Corporation":     ["Jayanagara", "Madiwala", "Wilson Garden"],
    # East
    "East Zone 1":                     ["K.R. Pura", "Mahadevapura", "Hennuru"],
    "East Zone 2":                     ["Mahadevapura", "K.R. Pura", "Halasur"],
    "Bengaluru East Corporation":      ["K.R. Pura", "Mahadevapura"],
    # West
    "West Zone 1":                     ["Peenya", "Kengeri", "Byatarayanapura"],
    "West Zone 2":                     ["Kengeri", "Peenya", "Byatarayanapura"],
    "Bengaluru West Corporation":      ["Peenya", "Kengeri"],
    # Fallback
    "Unknown":                         ["Cubbon Park"],
}


def get_nearest_station(zone: str) -> str:
    """Return the primary (nearest) police station for a zone."""
    return ZONE_STATIONS.get(zone, ZONE_STATIONS["Unknown"])[0]


def get_all_stations(zone: str) -> list[str]:
    """Return all police stations for a zone."""
    return ZONE_STATIONS.get(zone, ZONE_STATIONS["Unknown"])


# ---------------------------------------------------------------------------
# Cause display name mapping
# ---------------------------------------------------------------------------

CAUSE_DISPLAY: dict[str, str] = {
    "vehicle_breakdown": "Vehicle Breakdown",
    "accident":          "Accident",
    "tree_fall":         "Tree Fall",
    "water_logging":     "Water Logging",
    "pot_holes":         "Pot Holes",
    "public_event":      "Public Event",
    "procession":        "Procession",
    "vip_movement":      "VIP Movement",
    "protest":           "Protest",
    "construction":      "Construction",
    "congestion":        "Congestion",
    "road_conditions":   "Road Conditions",
    "flood":             "Flood",
    "other":             "Other",
}

# Event-driven / gathering causes (the planned & unplanned event categories
# the problem statement is centred on).  Used for filtering and analytics.
EVENT_DRIVEN_CAUSES = ["procession", "vip_movement", "protest", "public_event"]

ALL_CAUSES = list(CAUSE_DISPLAY.keys())
ALL_ZONES  = sorted(ZONE_STATIONS.keys())
