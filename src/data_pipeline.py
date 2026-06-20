"""
data_pipeline.py — Load and clean the raw Astram events CSV.

Responsibilities:
  - Locate the raw CSV (data/raw/events.csv or the original Astram filename)
  - Replace "NULL" string tokens with NaN
  - Zero values in endlatitude/endlongitude → NaN (sentinel for "point event")
  - Parse all datetime columns as UTC-aware Timestamps
  - Convert requires_road_closure to pandas BooleanDtype
  - Standardize event_cause to a fixed vocabulary
  - Drop rows where start_datetime is missing
  - Normalise priority / corridor / zone / veh_type
  - Write data/processed/clean.csv and return a DataFrame
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Original filename as downloaded from Astram
_ASTRAM_FILENAME = (
    "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
)

# Tokens the CSV uses for missing values
_NULL_TOKENS = ["NULL", "null", "None", "none", "N/A", "n/a", "\\N", ""]

# Columns that should be parsed as UTC-aware datetimes
_DATETIME_COLS = [
    "start_datetime",
    "end_datetime",
    "resolved_datetime",
    "closed_datetime",
    "modified_datetime",
    "created_date",
]

# Canonical event cause vocabulary (source token → canonical label)
#
# The event-driven causes the problem statement cares about — procession,
# vip_movement, protest, public_event — are kept as first-class labels rather
# than collapsed into "other".  congestion and road_conditions are also
# surfaced because they carry real operational signal.  Only genuinely
# residual tokens (debris, test_demo, fog) fall through to "other".
_CAUSE_MAP: dict[str, str] = {
    "vehicle_breakdown": "vehicle_breakdown",
    "accident": "accident",
    "tree_fall": "tree_fall",
    "water_logging": "water_logging",
    "pot_holes": "pot_holes",
    "public_event": "public_event",
    "construction": "construction",
    "flood": "flood",
    # event-driven / gathering causes (were silently dropped into "other")
    "procession": "procession",
    "vip_movement": "vip_movement",
    "protest": "protest",
    # condition causes worth keeping distinct
    "congestion": "congestion",
    "road_conditions": "road_conditions",
    # residual
    "others": "other",
    "other": "other",
}


# ---------------------------------------------------------------------------
# Locating the raw CSV
# ---------------------------------------------------------------------------

def _find_raw_csv(project_root: Path) -> Path:
    """Return the path to the raw CSV, searching in priority order."""
    candidates = [
        project_root / "data" / "raw" / "events.csv",
        project_root / _ASTRAM_FILENAME,
        project_root.parent / _ASTRAM_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Raw CSV not found. Copy it to:\n"
        f"  {project_root / 'data' / 'raw' / 'events.csv'}\n"
        f"or leave the original file at:\n"
        f"  {project_root.parent / _ASTRAM_FILENAME}"
    )


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def _load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        na_values=_NULL_TOKENS,
        keep_default_na=True,
        low_memory=False,
    )
    logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), csv_path.name)
    return df


def _fix_zero_coords(df: pd.DataFrame) -> pd.DataFrame:
    """endlatitude / endlongitude of 0 are missing-value sentinels, not coordinates."""
    for col in ("endlatitude", "endlongitude"):
        if col in df.columns:
            df[col] = df[col].replace(0.0, np.nan)
    return df


def _parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    # The raw timestamps carry a "+00" tag, but their wall-clock already behaves
    # as Bengaluru LOCAL time (converting to IST empties the evening peak and
    # invents a 2 AM one — verified empirically). So we parse with utc=True to
    # interpret the tag, then strip it to NAIVE LOCAL here, ONCE, at ingestion.
    # Every downstream consumer then works in plain local time with no tz juggling
    # — and nobody is tempted to "correct" a UTC tag that was never truly UTC.
    for col in _DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def _clean_road_closure(df: pd.DataFrame) -> pd.DataFrame:
    """Map TRUE/FALSE strings → pandas BooleanDtype (supports NA)."""
    def _to_bool(val):
        s = str(val).strip().upper()
        if s == "TRUE":
            return True
        if s == "FALSE":
            return False
        return pd.NA

    df["requires_road_closure"] = (
        df["requires_road_closure"].map(_to_bool).astype("boolean")
    )
    return df


def _standardize_cause(df: pd.DataFrame) -> pd.DataFrame:
    df["event_cause"] = (
        df["event_cause"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: _CAUSE_MAP.get(x, "other"))
    )
    return df


def _drop_missing_start(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["start_datetime"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing start_datetime", dropped)
    return df


def _clean_priority(df: pd.DataFrame) -> pd.DataFrame:
    valid = {"High", "Medium", "Low"}
    mask = ~df["priority"].isin(valid)
    if mask.any():
        logger.info("Replacing %d invalid priority values with 'Low'", mask.sum())
    df["priority"] = df["priority"].where(df["priority"].isin(valid), other="Low")
    return df


def _clean_corridor(df: pd.DataFrame) -> pd.DataFrame:
    df["corridor"] = df["corridor"].fillna("Non-corridor").str.strip()
    return df


def _clean_zone(df: pd.DataFrame) -> pd.DataFrame:
    df["zone"] = df["zone"].fillna("Unknown").str.strip()
    return df


def _clean_veh_type(df: pd.DataFrame) -> pd.DataFrame:
    df["veh_type"] = (
        df["veh_type"].fillna("unknown").str.strip().str.lower()
    )
    return df


def _clean_junction(df: pd.DataFrame) -> pd.DataFrame:
    df["junction"] = df["junction"].fillna("unknown").str.strip()
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Bengaluru bounding box (deg) — points outside this are bad coordinates.
_BLR_BBOX = {"lat": (12.5, 13.5), "lon": (77.2, 77.9)}
_REQUIRED_COLS = ["id", "latitude", "longitude", "start_datetime", "event_cause", "status"]


def validate_schema(df: pd.DataFrame, raise_on_critical: bool = True) -> list[str]:
    """
    Lightweight schema/range check on the cleaned feed — catches a dirty or
    schema-drifted upload before it silently poisons features and models.

    Raises on a *critical* problem (a required column missing entirely); returns
    a list of non-fatal warnings (out-of-range coords, high null rates, unknown
    statuses) for everything else. A Pandera schema could formalise this later;
    this keeps it dependency-free.
    """
    issues: list[str] = []

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        msg = f"missing required columns: {missing}"
        if raise_on_critical:
            raise ValueError(f"Ingest validation failed — {msg}")
        issues.append(msg)
        return issues

    for axis, col in (("lat", "latitude"), ("lon", "longitude")):
        lo, hi = _BLR_BBOX[axis]
        vals = pd.to_numeric(df[col], errors="coerce")
        out = int(((vals < lo) | (vals > hi)).sum())
        if out:
            issues.append(f"{col}: {out} value(s) outside Bengaluru range [{lo}, {hi}]")

    nat = int(pd.to_datetime(df["start_datetime"], errors="coerce").isna().sum())
    if nat:
        issues.append(f"start_datetime: {nat} unparseable/NaT row(s)")

    for col in ("event_cause", "status", "id"):
        nulls = int(df[col].isna().sum())
        if nulls:
            issues.append(f"{col}: {nulls} null(s)")

    for msg in issues:
        logger.warning("Ingest validation: %s", msg)
    if not issues:
        logger.info("Ingest validation: clean (%d rows, all checks passed)", len(df))
    return issues


def run_pipeline(project_root: Path | None = None) -> pd.DataFrame:
    """
    Run the full cleaning pipeline.

    Parameters
    ----------
    project_root : Path, optional
        Root of the smartflow project (parent of src/).
        Defaults to the directory two levels above this file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame also written to data/processed/clean.csv.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    csv_path = _find_raw_csv(project_root)
    logger.info("Using raw CSV: %s", csv_path)

    df = _load_raw(csv_path)
    df = _fix_zero_coords(df)
    df = _parse_datetimes(df)
    df = _clean_road_closure(df)
    df = _standardize_cause(df)
    df = _drop_missing_start(df)
    df = _clean_priority(df)
    df = _clean_corridor(df)
    df = _clean_zone(df)
    df = _clean_veh_type(df)
    df = _clean_junction(df)

    validate_schema(df)

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clean.csv"
    df.to_csv(out_path, index=False)
    logger.info("Clean data → %d rows, saved to %s", len(df), out_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_pipeline()
