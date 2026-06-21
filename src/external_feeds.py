"""
external_feeds.py — Integration scaffolds for the two roadmap data joins.

The audit identifies two STRUCTURAL gaps that the incident dataset alone cannot
close, because the data simply isn't in it:

  1. There is no measured congestion outcome (speed / delay / queue), so the
     impact score can never be *validated* — only asserted. A live/typical-speed
     feed (TomTom, HERE, Google Roads) at the incident point+time is the missing
     ground truth.
  2. Planned-event forecasting (Layer B) needs to know events are *coming*. The
     ~191 planned-event rows are a log of events that already happened; a real
     event calendar (stadium fixtures, festival/permit/rally schedules) is what
     turns retrieval into advance forecasting.

Both need an external source we deliberately do NOT fabricate. This module is the
honest seam where a real feed plugs in:

  * It is OFF by default and the whole system runs without it.
  * When unconfigured, every call returns a clear `{"available": False, "reason":
    ...}` instead of inventing numbers.
  * The validation path (validate_impact_against_speed) is fully implemented and
    runs the moment a speed column exists — so the claim "this would validate
    impact" is demonstrable, not hand-waved.

This is the single highest-leverage upgrade on the roadmap, scoped so wiring a
key/CSV is a drop-in, not a rewrite.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _config(project_root: Path) -> dict:
    try:
        import yaml
        cfg = yaml.safe_load((project_root / "config.yaml").read_text()) or {}
        return cfg.get("external", {}) or {}
    except Exception as exc:
        logger.warning("external config not read (%s)", exc)
        return {}


# ===========================================================================
# 1. Speed feed — the measured-impact ground truth
# ===========================================================================

def speed_provider_status(project_root: Path | None = None) -> dict:
    """Report whether a speed feed is configured, without calling out."""
    root = project_root or Path(__file__).resolve().parents[1]
    cfg = _config(root).get("speed", {}) or {}
    provider = (cfg.get("provider") or "none").lower()
    key = cfg.get("api_key") or os.environ.get("SMARTFLOW_SPEED_API_KEY", "")
    if provider == "none":
        return {"available": False, "reason": "no speed provider configured "
                "(set external.speed.provider in config.yaml)"}
    if not key:
        return {"available": False, "provider": provider,
                "reason": f"{provider} selected but no api_key / "
                "SMARTFLOW_SPEED_API_KEY set"}
    return {"available": True, "provider": provider}


def speed_context(lat: float, lon: float, project_root: Path | None = None) -> dict:
    """
    Typical-vs-current speed at a point — the basis for a *measured* impact.

    Returns {"available": False, "reason": ...} until a provider + key are set.
    When configured, an implementer fills in the single provider call below; the
    return contract is {available, free_flow_kmph, current_kmph, speed_ratio}.
    """
    status = speed_provider_status(project_root)
    if not status["available"]:
        return status
    # --- Implementer plugs the real call in here (provider-specific) ----------
    # e.g. TomTom Flow Segment Data:
    #   GET .../flowSegmentData/absolute/10/json?point={lat},{lon}&key={key}
    #   free_flow = resp["flowSegmentData"]["freeFlowSpeed"]
    #   current   = resp["flowSegmentData"]["currentSpeed"]
    # We intentionally do NOT ship a half-real HTTP call with no key to test it.
    raise NotImplementedError(
        f"speed_context: provider '{status['provider']}' is configured but the "
        "provider HTTP call is left for the integrator (one request, documented "
        "inline). The rest of the pipeline is ready to consume it."
    )


def validate_impact_against_speed(df: pd.DataFrame,
                                  speed_col: str = "speed_ratio",
                                  impact_col: str = "impact_score") -> dict:
    """
    THE validation the dataset can't currently support: correlate the heuristic
    impact score against a measured speed drop. This is fully implemented — it
    runs as soon as a speed feed has populated `speed_col` (current ÷ free-flow;
    lower = worse). A strong negative correlation (high impact ↔ big slowdown)
    is the evidence that the impact heuristic tracks reality.

    Returns {"available": False, ...} if the speed column isn't present yet.
    """
    if speed_col not in df.columns or impact_col not in df.columns:
        return {"available": False,
                "reason": f"need both '{impact_col}' and a measured '{speed_col}' "
                "column (populate via a speed feed) to validate impact"}
    sub = df[[impact_col, speed_col]].dropna()
    if len(sub) < 30:
        return {"available": False, "reason": f"only {len(sub)} paired rows (need ≥30)"}
    # Speed ratio is inverse to disruption, so we expect negative correlation.
    pearson = float(sub[impact_col].corr(sub[speed_col], method="pearson"))
    spearman = float(sub[impact_col].corr(sub[speed_col], method="spearman"))
    return {
        "available": True,
        "n": int(len(sub)),
        "pearson_r": round(pearson, 3),
        "spearman_rho": round(spearman, 3),
        "interpretation": (
            "impact tracks measured slowdown (expected negative correlation)"
            if spearman < -0.2 else
            "weak/inconsistent link — revisit impact weights against measured speed"
        ),
    }


# ===========================================================================
# 2. Event calendar — advance signal for planned-event forecasting (Layer B)
# ===========================================================================

# Columns the planner consumes. Kept minimal and operator-fillable.
CALENDAR_COLUMNS = [
    "date", "start_time", "venue", "latitude", "longitude",
    "event_type", "expected_attendance", "notes",
]


def calendar_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    rel = (_config(root).get("event_calendar", {}) or {}).get(
        "path", "data/external/event_calendar.csv")
    return root / rel


def write_calendar_template(project_root: Path | None = None) -> Path:
    """Write an empty, correctly-headed calendar CSV for operators to fill in."""
    path = calendar_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=CALENDAR_COLUMNS).to_csv(path, index=False)
        logger.info("Wrote event-calendar template → %s", path)
    return path


def load_calendar(project_root: Path | None = None) -> dict:
    """
    Load the planned-event calendar if an operator has populated it.

    Returns {"available": False, ...} when the CSV is missing/empty — the planner
    then falls back to its case-retrieval mode (analogous past events) instead of
    inventing a forecast.
    """
    path = calendar_path(project_root)
    if not path.exists():
        return {"available": False,
                "reason": f"no event calendar at {path} "
                "(run write_calendar_template() and fill it in)"}
    df = pd.read_csv(path)
    if df.empty:
        return {"available": False, "reason": "event calendar is empty"}
    missing = [c for c in ("date", "event_type") if c not in df.columns]
    if missing:
        return {"available": False, "reason": f"calendar missing columns: {missing}"}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return {"available": True, "n": int(len(df)), "events": df}


def upcoming_events(on_date, project_root: Path | None = None) -> pd.DataFrame:
    """Planned events on a given date (empty frame if no calendar / none that day)."""
    cal = load_calendar(project_root)
    if not cal["available"]:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    df = cal["events"]
    target = pd.to_datetime(on_date, errors="coerce")
    if pd.isna(target):
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    return df[df["date"].dt.date == target.date()].reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    print("speed feed:", speed_provider_status())
    print("calendar  :", {k: v for k, v in load_calendar().items() if k != "events"})
