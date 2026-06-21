"""
test_external_feeds.py — Prove the roadmap scaffolds are real, not vaporware.

The speed feed and event calendar need an external source we don't ship, but the
*plumbing* around them is fully implemented and tested here: unconfigured calls
degrade to a clear "unavailable" (never fabricated numbers), the calendar
template/load/filter round-trips, and the impact-vs-speed validation actually
computes the expected correlation the moment a speed column exists.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import external_feeds as ef


def test_speed_feed_unconfigured_is_honest():
    s = ef.speed_provider_status()
    assert s["available"] is False and "reason" in s


def test_calendar_template_roundtrip(tmp_path):
    path = ef.write_calendar_template(tmp_path)
    assert path.exists()
    # Empty template → not available, but with correct headers.
    assert ef.load_calendar(tmp_path)["available"] is False
    cols = pd.read_csv(path).columns.tolist()
    assert cols == ef.CALENDAR_COLUMNS


def test_calendar_load_and_filter(tmp_path):
    path = ef.write_calendar_template(tmp_path)
    pd.DataFrame([
        {"date": "2024-04-15", "start_time": "19:00", "venue": "Chinnaswamy",
         "latitude": 12.9788, "longitude": 77.5996, "event_type": "sports_event",
         "expected_attendance": 40000, "notes": "IPL"},
        {"date": "2024-04-20", "start_time": "10:00", "venue": "Palace Grounds",
         "latitude": 13.0, "longitude": 77.59, "event_type": "public_event",
         "expected_attendance": 15000, "notes": ""},
    ]).to_csv(path, index=False)

    cal = ef.load_calendar(tmp_path)
    assert cal["available"] is True and cal["n"] == 2

    on = ef.upcoming_events("2024-04-15", tmp_path)
    assert len(on) == 1 and on.iloc[0]["venue"] == "Chinnaswamy"
    assert ef.upcoming_events("2024-04-16", tmp_path).empty


def test_impact_validation_runs_when_speed_present():
    # No speed column → honestly unavailable.
    df0 = pd.DataFrame({"impact_score": [10, 50, 90]})
    assert ef.validate_impact_against_speed(df0)["available"] is False

    # Synthetic measured speed: higher impact ⇒ bigger slowdown (lower ratio).
    rng = np.random.default_rng(0)
    impact = rng.uniform(0, 100, 200)
    speed_ratio = np.clip(1 - impact / 120 + rng.normal(0, 0.05, 200), 0.1, 1.0)
    df = pd.DataFrame({"impact_score": impact, "speed_ratio": speed_ratio})
    res = ef.validate_impact_against_speed(df)
    assert res["available"] is True
    assert res["spearman_rho"] < -0.2     # impact tracks measured slowdown
