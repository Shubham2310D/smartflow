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
    "construction":      "Construction",
    "flood":             "Flood",
    "other":             "Other",
}

ALL_CAUSES = list(CAUSE_DISPLAY.keys())
ALL_ZONES  = sorted(ZONE_STATIONS.keys())
