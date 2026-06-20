"""
feature_engineering.py — Derive all model input features from the clean dataset.

New columns produced
--------------------
  duration_minutes       : float  — regression target (NaN when unresolved)
  severity_class         : str    — classification target (High / Medium / Low)
  hour_of_day            : int    — 0–23
  day_of_week            : int    — 0=Monday … 6=Sunday
  month                  : int    — 1–12
  is_peak_hour           : int    — 1 if 08-10 or 17-20, else 0
  is_weekend             : int    — 1 if Saturday / Sunday
  junction_repeat_count  : int    — global event count at this junction
  corridor_7d_score      : int    — events on same corridor in prior 7 days
  cause_severity_weight  : int    — numeric risk weight for event_cause
  road_closure_binary    : int    — 1 if requires_road_closure else 0
  veh_type_encoded       : int    — ordinal encoding by traffic disruption impact
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Free-text event-type extraction (description field)
# ---------------------------------------------------------------------------
# The `description` field (83% populated, mixed English / Kannada / transliteration)
# carries the real event semantics the structured `event_cause` column misses —
# "Cricket Match at Chinnaswamy", "BWSSB work", "BMTC bus broken down", tree falls.
# This is a lightweight bilingual keyword pass that derives an event_semantic_type.
# Order matters: the first matching pattern wins (most specific first).
SEMANTIC_PATTERNS: list[tuple[str, str]] = [
    ("sports_event",      r"cricket|match|stadium|chinnaswamy|ಕ್ರಿಕೆಟ್|ಪಂದ್ಯ|ಕ್ರೀಡಾ"),
    ("vip_movement",      r"\bvip\b|\bvvip\b|minister|cm\b|convoy|ಗಣ್ಯ|ಸಚಿವ"),
    ("protest",           r"protest|dharna|strike|bandh|agitation|ಪ್ರತಿಭಟನೆ|ಧರಣಿ|ಮುಷ್ಕರ"),
    ("procession",        r"procession|rally|march|festival|jatre|ಮೆರವಣಿಗೆ|ಜಾತ್ರೆ|ಹಬ್ಬ"),
    ("utility_work",      r"bwssb|kride|bescom|bbmp|cement|drainage|chamber|pipe|"
                          r"ಒಳಚರಂಡಿ|ಸಿಮೆಂಟ್|ಪೈಪ್|ಕೆಲಸ|ವರ್ಕ್|ಕಾಮಗಾರಿ"),
    ("tree_fall",         r"tree\s*fall|tree\s*fell|fallen tree|ಮರ\s*ಬಿದ್ದ|ಮರ ಬಿದ"),
    ("water_logging",     r"water\s*logg|woter|flood|drainage leak|ನೀರು|ಮಳೆ|ಜಲ"),
    ("accident",          r"accident|collision|crash|hit|ಅಪಘಾತ|ಡಿಕ್ಕಿ"),
    ("vehicle_breakdown", r"break\s*down|breakdown|broke|offload|off road|off\s* road|"
                          r"mechanic|clutch|gear\s*box|punctur|panchar|ಪಂಚರ್|tyre|tyear|"
                          r"tire|blost|blast|ಕೆಟ್ಟು|ಬ್ರೇಕ್\s*ಡೌನ್|ವೆಹಿಕಲ್"),
    ("pothole",           r"pot\s*hole|pothole|gundi|ಗುಂಡಿ"),
    ("congestion",        r"congest|traffic jam|slow mov|ನಿಧಾನ|ಜಾಮ್"),
]
_SEMANTIC_COMPILED = [(t, re.compile(p, re.IGNORECASE)) for t, p in SEMANTIC_PATTERNS]

# Anonymisation placeholders to strip before matching
_ANON_RE = re.compile(r"\[(LOCATION|PERSON|PHONE|EMAIL)\]", re.IGNORECASE)

# Ordinal encoding for the semantic type (tree/numeric input for XGBoost)
SEMANTIC_TYPE_ORDER = [
    "other", "congestion", "pothole", "vehicle_breakdown", "water_logging",
    "tree_fall", "accident", "utility_work", "procession", "protest",
    "vip_movement", "sports_event",
]
_SEMANTIC_MAP: dict[str, int] = {t: i for i, t in enumerate(SEMANTIC_TYPE_ORDER)}


def extract_semantic_type(text) -> str:
    """Return the event semantic type inferred from a free-text description."""
    if not isinstance(text, str) or not text.strip():
        return "other"
    cleaned = _ANON_RE.sub(" ", text)
    for sem_type, pattern in _SEMANTIC_COMPILED:
        if pattern.search(cleaned):
            return sem_type
    return "other"

# Numeric risk weight per cause (used as XGBoost feature)
CAUSE_SEVERITY_WEIGHT: dict[str, int] = {
    "accident": 3,
    "flood": 3,
    "vip_movement": 3,     # rolling closures, high disruption
    "protest": 3,          # unplanned gathering, unpredictable spread
    "water_logging": 2,
    "tree_fall": 2,
    "public_event": 2,
    "procession": 2,       # planned gathering, partial closures
    "construction": 2,
    "pot_holes": 1,
    "vehicle_breakdown": 1,
    "congestion": 1,
    "road_conditions": 1,
    "other": 1,
}

# Ordinal encoding: higher index = more disruptive to traffic
_VEH_TYPE_ORDER = [
    "heavy_vehicle",
    "ksrtc_bus",
    "bmtc_bus",
    "private_bus",
    "lcv",
    "private_car",
    "two_wheeler",
    "unknown",
]
_VEH_TYPE_MAP: dict[str, int] = {v: i for i, v in enumerate(_VEH_TYPE_ORDER)}

# Maximum plausible event duration (7 days).  Beyond this the ticket was
# likely forgotten open rather than a real congestion event.
_MAX_DURATION_MINUTES = 7 * 24 * 60  # 10 080

# Severity scoring weights
_PRIORITY_SCORE = {"High": 2, "Medium": 1, "Low": 0}
_CAUSE_SEVERITY_SCORE: dict[str, int] = {
    "accident": 2,
    "flood": 2,
    "vip_movement": 2,
    "protest": 2,
    "water_logging": 1,
    "tree_fall": 1,
    "public_event": 1,
    "procession": 1,
    "construction": 1,
    "vehicle_breakdown": 0,
    "pot_holes": 0,
    "congestion": 0,
    "road_conditions": 0,
    "other": 0,
}


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer resolved_datetime, fall back to closed_datetime, then end_datetime.
    Negative durations (data-entry errors) and values > 7 days become NaN.
    """
    resolution = df["resolved_datetime"].copy()

    if "closed_datetime" in df.columns:
        resolution = resolution.combine_first(df["closed_datetime"])

    if "end_datetime" in df.columns:
        resolution = resolution.combine_first(df["end_datetime"])

    delta_minutes = (resolution - df["start_datetime"]).dt.total_seconds() / 60.0
    delta_minutes = delta_minutes.where(delta_minutes > 0, other=np.nan)
    delta_minutes = delta_minutes.where(delta_minutes <= _MAX_DURATION_MINUTES, other=np.nan)

    df["duration_minutes"] = delta_minutes
    n_valid = delta_minutes.notna().sum()
    logger.info("duration_minutes: %d / %d rows have a valid value", n_valid, len(df))
    return df


def compute_severity_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score = priority(0-2) + road_closure(0-2) + cause(0-2)
    High   : score >= 4
    Medium : score >= 2
    Low    : score <  2
    """
    priority_score = df["priority"].map(_PRIORITY_SCORE).fillna(0)
    closure_score = df["requires_road_closure"].astype(float).fillna(0) * 2
    cause_score = df["event_cause"].map(_CAUSE_SEVERITY_SCORE).fillna(0)

    total = priority_score + closure_score + cause_score

    df["severity_class"] = np.select(
        [total >= 4, total >= 2],
        ["High", "Medium"],
        default="Low",
    )

    dist = df["severity_class"].value_counts()
    logger.info("severity_class distribution:\n%s", dist.to_string())
    return df


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["start_datetime"].dt
    df["hour_of_day"] = dt.hour
    df["day_of_week"] = dt.dayofweek          # 0=Monday
    df["month"] = dt.month

    peak_hours = set(range(8, 11)) | set(range(17, 21))   # 08-10, 17-20
    df["is_peak_hour"] = df["hour_of_day"].isin(peak_hours).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def junction_repeat_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-looking count of prior events at the same junction.

    Previously this used a global value_counts() over the entire dataset, which
    leaks future information into the past (an event "knows" about events that
    happened after it).  We instead count, for each event, how many events
    occurred at the same junction *strictly before* it — a leakage-free chronic
    hotspot signal computed exactly the same way as corridor_7d_score.
    """
    ordered = df.sort_values("start_datetime", kind="stable")
    prior = ordered.groupby("junction").cumcount()   # 0,1,2,… prior events
    counts = prior.reindex(df.index).astype(int)
    # "unknown" is a catch-all bucket, not a real junction — a running count over
    # it is just a time index, so force it to 0 (no known chronic-hotspot signal).
    counts = counts.where(df["junction"].str.lower() != "unknown", 0)
    df["junction_repeat_count"] = counts.astype(int)
    logger.info(
        "junction_repeat_count (backward-looking) computed (max=%d at a named junction)",
        int(df["junction_repeat_count"].max()),
    )
    return df


def corridor_7d_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event e, count events on the same corridor whose start_datetime
    falls in [e.start_datetime - 7 days, e.start_datetime).

    Uses vectorised numpy searchsorted per corridor group for speed —
    O(k log k) per corridor, runs in well under a second for 8k rows.
    """
    df = df.sort_values("start_datetime").reset_index(drop=True)

    # Convert timezone-aware datetimes to int64 nanoseconds for arithmetic
    ts_naive = df["start_datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    ts_ns: np.ndarray = ts_naive.astype("int64").values

    seven_days_ns = int(7 * 24 * 3600 * 1e9)
    scores = np.zeros(len(df), dtype=np.int32)

    for _, group in df.groupby("corridor", sort=False):
        idx = group.index.values          # positions in the sorted df
        times = ts_ns[idx]                # already sorted because df is sorted

        # Vectorised: for every event in this corridor group at once
        window_starts = times - seven_days_ns
        lo_arr = np.searchsorted(times, window_starts, side="left")
        hi_arr = np.arange(len(times), dtype=np.int32)
        scores[idx] = hi_arr - lo_arr

    df["corridor_7d_score"] = scores
    logger.info("corridor_7d_score computed (max=%d)", scores.max())
    return df


def encode_cause_weight(df: pd.DataFrame) -> pd.DataFrame:
    df["cause_severity_weight"] = (
        df["event_cause"].map(CAUSE_SEVERITY_WEIGHT).fillna(1).astype(int)
    )
    return df


def encode_road_closure(df: pd.DataFrame) -> pd.DataFrame:
    df["road_closure_binary"] = (
        df["requires_road_closure"].astype(float).fillna(0).astype(int)
    )
    return df


def encode_veh_type(df: pd.DataFrame) -> pd.DataFrame:
    df["veh_type_encoded"] = (
        df["veh_type"]
        .map(_VEH_TYPE_MAP)
        .fillna(len(_VEH_TYPE_ORDER) - 1)
        .astype(int)
    )
    return df


def encode_semantic_type(df: pd.DataFrame) -> pd.DataFrame:
    """Derive event_semantic_type + its ordinal encoding from the description text."""
    text = df["description"] if "description" in df.columns else pd.Series([""] * len(df))
    df["event_semantic_type"] = text.map(extract_semantic_type)
    df["event_semantic_encoded"] = (
        df["event_semantic_type"].map(_SEMANTIC_MAP).fillna(0).astype(int)
    )
    coverage = (df["event_semantic_type"] != "other").mean() * 100
    logger.info(
        "event_semantic_type: %.0f%% of events matched a text pattern\n%s",
        coverage, df["event_semantic_type"].value_counts().to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Columns written to model_ready.csv
_MODEL_COLS = [
    "id",
    # identifiers / metadata
    "event_type", "status", "authenticated",
    "latitude", "longitude",
    "corridor", "junction", "zone", "police_station",
    # raw inputs
    "event_cause", "priority", "requires_road_closure", "veh_type",
    "start_datetime",
    # engineered features (XGBoost inputs)
    "cause_severity_weight",
    "road_closure_binary",
    "veh_type_encoded",
    "event_semantic_type",
    "event_semantic_encoded",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_peak_hour",
    "is_weekend",
    "junction_repeat_count",
    "corridor_7d_score",
    # targets
    "severity_class",
    "duration_minutes",
]


def run_feature_engineering(
    df: pd.DataFrame | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    Run all feature derivation steps.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Output of data_pipeline.run_pipeline().  If None, the pipeline is
        run automatically to produce the clean DataFrame.
    project_root : Path, optional
        Root of the smartflow project.  Defaults to two levels above this file.

    Returns
    -------
    pd.DataFrame
        Full DataFrame with all engineered columns.
        Also writes data/processed/features.csv and data/processed/model_ready.csv.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    if df is None:
        # Lazy import avoids circular issues when both files are imported together
        from data_pipeline import run_pipeline  # noqa: PLC0415
        df = run_pipeline(project_root)

    df = compute_duration(df)
    df = compute_severity_class(df)
    df = temporal_features(df)
    df = junction_repeat_count(df)

    logger.info("Computing corridor_7d_score…")
    df = corridor_7d_score(df)

    df = encode_cause_weight(df)
    df = encode_road_closure(df)
    df = encode_veh_type(df)
    df = encode_semantic_type(df)

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "features.csv", index=False)
    logger.info("features.csv → %d rows", len(df))

    available = [c for c in _MODEL_COLS if c in df.columns]
    model_df = df[available].copy()
    model_df.to_csv(out_dir / "model_ready.csv", index=False)
    logger.info(
        "model_ready.csv → %d rows total, %d with duration",
        len(model_df),
        model_df["duration_minutes"].notna().sum(),
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_feature_engineering()
