"""
test_leakage.py — Guard the leakage-free history features.

These two features are the ones most likely to silently leak the future into the
past, so they get an explicit assertion that they are strictly backward-looking.
Run with:  pytest tests/  (or: python -m pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from feature_engineering import corridor_7d_score, junction_repeat_count


def test_junction_repeat_count_counts_only_prior_events():
    df = pd.DataFrame({
        "junction": ["MG Road", "MG Road", "MG Road", "BTM"],
        "start_datetime": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05"], utc=True),
    })
    out = junction_repeat_count(df.copy())
    # MG Road events in time order see 0, then 1, then 2 prior events.
    assert out.loc[0, "junction_repeat_count"] == 0
    assert out.loc[1, "junction_repeat_count"] == 1
    assert out.loc[2, "junction_repeat_count"] == 2
    # First (and only) BTM event has no priors.
    assert out.loc[3, "junction_repeat_count"] == 0


def test_junction_repeat_count_zeros_unknown_bucket():
    df = pd.DataFrame({
        "junction": ["unknown", "unknown", "unknown"],
        "start_datetime": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
    })
    out = junction_repeat_count(df.copy())
    # "unknown" is a catch-all, never a chronic-hotspot signal → always 0.
    assert (out["junction_repeat_count"] == 0).all()


def test_corridor_7d_score_is_backward_looking_and_windowed():
    df = pd.DataFrame({
        "corridor": ["ORR", "ORR", "ORR", "ORR"],
        "start_datetime": pd.to_datetime(
            ["2024-01-01", "2024-01-03", "2024-01-06", "2024-01-20"], utc=True),
    })
    out = corridor_7d_score(df.sort_values("start_datetime").reset_index(drop=True))
    out = out.sort_values("start_datetime").reset_index(drop=True)
    # Event 0: no prior. Event 1: 1 prior (Jan 1). Event 2: 2 prior (Jan 1, 3).
    assert out.loc[0, "corridor_7d_score"] == 0
    assert out.loc[1, "corridor_7d_score"] == 1
    assert out.loc[2, "corridor_7d_score"] == 2
    # Event 3 (Jan 20) is >7 days after all others → window empty.
    assert out.loc[3, "corridor_7d_score"] == 0


if __name__ == "__main__":
    test_junction_repeat_count_counts_only_prior_events()
    test_junction_repeat_count_zeros_unknown_bucket()
    test_corridor_7d_score_is_backward_looking_and_windowed()
    print("All leakage tests passed.")
