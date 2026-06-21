"""
hotspot_validation.py — Does a hotspot map fit on the PAST predict the FUTURE?

Moran's I (in hotspot_engine.py) already confirms the clusters are spatially real.
But "real" is descriptive — it says clusters exist, not that they forecast where
the next incidents land. This module closes that gap with a strict
spatial-temporal holdout, exactly as a hotspot-policing study would:

    1. Split events by calendar month: TRAIN = all months but the last,
       TEST = the final month (a future the model never saw).
    2. Build the hotspot footprint from TRAIN only (grid cells whose historical
       count clears a threshold).
    3. Score it against TEST with two standard metrics:
         hit_rate = share of TEST incidents that fall inside the TRAIN hotspots
         PAI      = hit_rate ÷ (hotspot area share)         [Chainey et al. 2008]
       PAI > 1 means the hotspots concentrate future incidents better than
       chance; PAI = 10 means 10× denser than a random patch of the same size.

A grid (≈500 m cells) is used instead of DBSCAN circles so "area share" is
unambiguous and the metric is reproducible. Output → data/processed/
hotspot_validation.json, surfaced on the Hotspot Map page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CELL_KM = 0.5                       # grid resolution (~500 m, matches DBSCAN eps scale)
_KM_PER_DEG_LAT = 110.574
# Thresholds (min TRAIN incidents per cell to call it a hotspot) to sweep —
# spanning a wide footprint (low threshold) to a tight, patrollable one (high).
_THRESHOLDS = (3, 5, 8, 12, 20, 35, 60)


def _cell_size_deg(lat0: float) -> tuple[float, float]:
    dlat = _CELL_KM / _KM_PER_DEG_LAT
    dlon = _CELL_KM / (111.320 * np.cos(np.radians(lat0)))
    return dlat, dlon


def _assign_cells(df: pd.DataFrame, dlat: float, dlon: float,
                  lat_min: float, lon_min: float) -> pd.Series:
    r = ((df["latitude"] - lat_min) / dlat).astype(int)
    c = ((df["longitude"] - lon_min) / dlon).astype(int)
    return r.astype(str) + "_" + c.astype(str)


def validate_hotspots(project_root: Path | None = None,
                      thresholds: tuple[int, ...] = _THRESHOLDS) -> dict:
    """
    Run the spatial-temporal holdout and return a result dict (also saved to JSON).
    Degrades gracefully (returns {"status": ...}) if there isn't enough data.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    src = project_root / "data" / "processed" / "features.csv"
    if not src.exists():
        src = project_root / "data" / "processed" / "clean.csv"
    df = pd.read_csv(src, usecols=lambda c: c in {"latitude", "longitude", "start_datetime"})
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "start_datetime"])

    # Chronological split on calendar month: last month = test, the rest = train.
    df["ym"] = df["start_datetime"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    if len(months) < 2:
        return {"status": "insufficient temporal coverage (need ≥2 months)"}
    test_month = months[-1]
    train = df[df["ym"] < test_month]
    test = df[df["ym"] == test_month]
    if len(train) < 50 or len(test) < 20:
        return {"status": f"too few events (train={len(train)}, test={len(test)})"}

    # Grid frame fixed from the TRAIN extent (the only thing known at fit time).
    lat0 = float(train["latitude"].mean())
    dlat, dlon = _cell_size_deg(lat0)
    lat_min, lon_min = float(train["latitude"].min()), float(train["longitude"].min())

    train_cells = _assign_cells(train, dlat, dlon, lat_min, lon_min)
    test_cells = _assign_cells(test, dlat, dlon, lat_min, lon_min)
    train_counts = train_cells.value_counts()
    occupied = int((train_counts > 0).sum())   # active-area denominator (cells ever seen)

    results = []
    for t in thresholds:
        hot = set(train_counts[train_counts >= t].index)
        if not hot:
            continue
        hits = int(test_cells.isin(hot).sum())
        hit_rate = hits / len(test)
        area_share = len(hot) / occupied
        pai = (hit_rate / area_share) if area_share else float("nan")
        results.append({
            "min_train_events_per_cell": t,
            "n_hotspot_cells": len(hot),
            "area_share": round(area_share, 4),
            "future_hit_rate": round(hit_rate, 4),
            "pai": round(pai, 2),
        })

    if not results:
        return {"status": "no hotspot cells at any threshold"}

    # Headline: the operating point closest to ~5% area coverage — a realistically
    # patrollable footprint — so PAI isn't cherry-picked from a tiny dense cell.
    headline = min(results, key=lambda r: abs(r["area_share"] - 0.05))
    out = {
        "status": "ok",
        "cell_km": _CELL_KM,
        "train_months": [str(m) for m in months[:-1]],
        "test_month": str(test_month),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "occupied_cells": occupied,
        "headline": headline,
        "by_threshold": results,
        "interpretation": (
            f"Hotspots fit on {months[0]}–{months[-2]} covering "
            f"{headline['area_share']*100:.1f}% of the active area captured "
            f"{headline['future_hit_rate']*100:.0f}% of {test_month}'s incidents "
            f"— {headline['pai']:.1f}× denser than chance (PAI>1 ⇒ predictive)."
        ),
    }
    dest = project_root / "data" / "processed" / "hotspot_validation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    logger.info(out["interpretation"])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    res = validate_hotspots()
    print(json.dumps(res, indent=2))
