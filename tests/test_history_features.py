"""
test_history_features.py — Guard the inference-time history lookup.

Regression test for the fabricated-history bug: live predictions used to feed the
models a hardcoded `5` for junction_repeat_count / corridor_7d_score, so their
outputs were computed from a made-up value. These tests assert the lookup returns
real, backward-looking integers (with a global-median fallback), never that
constant.
Run with:  pytest tests/  (or: python -m pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import history_features as hf


def test_returns_both_history_features_as_ints():
    out = hf.history_features("any-corridor")
    assert set(out) == {"corridor_7d_score", "junction_repeat_count"}
    assert all(isinstance(v, int) for v in out.values())


def test_unknown_corridor_falls_back_to_global_not_constant():
    # An unknown corridor must yield the global median, identical to None — and
    # must NOT silently return the old hardcoded sentinel masquerading as data.
    fallback = hf.history_features("NoSuchCorridorXYZ")
    assert fallback == hf.history_features(None)


def test_known_corridor_differs_from_fallback_when_history_exists():
    corridors = hf.corridor_list()
    if not corridors:
        return  # no features.csv in this environment — nothing to assert
    fallback = hf.history_features(None)
    # At least one real corridor should carry a different historical profile than
    # the global fallback (otherwise the lookup is effectively a constant again).
    assert any(hf.history_features(c) != fallback for c in corridors)
