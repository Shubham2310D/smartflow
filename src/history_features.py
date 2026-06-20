"""
history_features.py — Backward-looking history features for inference.

A genuinely new event has no `junction_repeat_count` or `corridor_7d_score` of
its own (those are computed from prior events). Rather than feed the model a
fabricated constant — the old behaviour was a hardcoded `5`, which silently made
every live prediction depend on a made-up value — we look up the historical
**median** of each feature for the event's corridor, with a global-median
fallback for unknown corridors. This is the same backward-looking discipline
used for clearance ranges, and it is the single source of truth shared by the
dashboard Predict page and the real-time API, so both compute history features
the same honest way.

Note on honesty: for a new event "now", the best backward-looking estimate of
"events on this corridor in the prior 7 days" is the historical median of that
quantity on that corridor. It is a typical-rate proxy, not a live count — that
would need a real event store (roadmap), which this deliberately is not.
"""
from __future__ import annotations

import pandas as pd

from utils import get_project_root

# Features this module supplies. Both are backward-looking counts that an
# incoming event cannot know about itself at inference time.
_HISTORY_FEATURES = ["corridor_7d_score", "junction_repeat_count"]

_cache: dict | None = None


def _features_path():
    return get_project_root() / "data" / "processed" / "features.csv"


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    path = _features_path()
    if not path.exists():
        # No history available — neutral zeros (an honest "unknown"), never a
        # magic constant.
        _cache = {"per_corridor": {}, "global": {f: 0 for f in _HISTORY_FEATURES}}
        return _cache

    wanted = set(_HISTORY_FEATURES) | {"corridor"}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    per_corridor = {
        corridor: {f: int(round(float(grp[f].median()))) for f in _HISTORY_FEATURES}
        for corridor, grp in df.groupby("corridor")
    }
    global_med = {f: int(round(float(df[f].median()))) for f in _HISTORY_FEATURES}
    _cache = {"per_corridor": per_corridor, "global": global_med}
    return _cache


def corridor_list() -> list[str]:
    """Sorted corridors that have historical stats (for UI dropdowns)."""
    return sorted(_load()["per_corridor"].keys())


def history_features(corridor: str | None = None) -> dict:
    """
    Backward-looking {corridor_7d_score, junction_repeat_count} for a NEW event.

    Uses the historical median for `corridor`; falls back to the global median
    when the corridor is unknown / None. Never a hardcoded constant.
    """
    data = _load()
    if corridor and corridor in data["per_corridor"]:
        return dict(data["per_corridor"][corridor])
    return dict(data["global"])
