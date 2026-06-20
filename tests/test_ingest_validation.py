"""
test_ingest_validation.py — Guard the ingest schema/range check.

Asserts a clean feed passes silently, out-of-range coordinates are flagged, and
a missing required column is a hard (critical) failure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from data_pipeline import validate_schema


def _good():
    return pd.DataFrame({
        "id": ["a", "b"], "latitude": [12.97, 13.0], "longitude": [77.59, 77.6],
        "start_datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "event_cause": ["accident", "procession"], "status": ["closed", "active"],
    })


def test_clean_feed_has_no_issues():
    assert validate_schema(_good()) == []


def test_out_of_range_coords_flagged():
    df = _good()
    df.loc[0, "latitude"] = 28.6  # Delhi, not Bengaluru
    issues = validate_schema(df)
    assert any("latitude" in i for i in issues)


def test_missing_required_column_is_critical():
    df = _good().drop(columns=["status"])
    with pytest.raises(ValueError):
        validate_schema(df)
