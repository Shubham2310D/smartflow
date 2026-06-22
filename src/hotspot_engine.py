"""
hotspot_engine.py — Spatial analysis pipeline.

Stages
------
1. DBSCAN  — cluster incident points (eps ≈ 500 m, min_samples = 5)
2. KDE     — build density surface for Folium HeatMap
3. Moran's I — confirm spatial autocorrelation is statistically significant
4. GeoJSON — convex-hull polygons per cluster for map overlay
5. Rank    — top-N hotspot junctions by event count

Outputs (data/processed/)
  hotspots.geojson          — cluster polygons with properties
  hotspot_summary.csv       — ranked hotspot table
  heatmap_points.csv        — (lat, lon, weight) for Folium HeatMap
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# Keywords that indicate a meaningful landmark in an address part
_LANDMARK_RE = re.compile(
    r"\b(junction|junc|circle|cross|flyover|bridge|signal|toll|gate|layout|nagar|road)\b",
    re.IGNORECASE,
)


def _label_from_address(address: str) -> str:
    """Extract a short readable location label from a full address string."""
    if not isinstance(address, str) or not address.strip():
        return ""
    parts = [p.strip() for p in address.split(",") if p.strip()]
    # Prefer any part that contains a landmark keyword
    for part in parts:
        if _LANDMARK_RE.search(part):
            label = part.split(".")[0].strip()   # drop trailing ". Pin-..."
            return label[:40]
    # Fall back to first part (usually road/area name)
    return parts[0][:40] if parts else ""


def _best_junction_label(junction: str, address: str, cluster_id: int) -> str:
    """Return the best available name: junction → address extract → Cluster-N."""
    if junction and junction.lower() != "unknown":
        return junction
    label = _label_from_address(address)
    return label if label else f"Cluster-{cluster_id}"

# ---------------------------------------------------------------------------
# Zone imputation
# ---------------------------------------------------------------------------

def _build_zone_centroids(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["zone"].notna() & (df["zone"] != "Unknown")]
    if known.empty:
        return pd.DataFrame()
    return known.groupby("zone")[["latitude", "longitude"]].mean().reset_index()


def _impute_zones(df: pd.DataFrame, zone_centroids: pd.DataFrame) -> pd.DataFrame:
    """Replace zone='Unknown' with nearest zone centroid (vectorized)."""
    mask = (df["zone"] == "Unknown") | df["zone"].isna()
    if not mask.any() or zone_centroids.empty:
        return df
    df = df.copy()
    lats = df.loc[mask, "latitude"].values
    lons = df.loc[mask, "longitude"].values
    zc_lats = zone_centroids["latitude"].values
    zc_lons = zone_centroids["longitude"].values
    dists = (lats[:, None] - zc_lats[None, :]) ** 2 + (lons[:, None] - zc_lons[None, :]) ** 2
    nearest = zone_centroids["zone"].values[dists.argmin(axis=1)]
    df.loc[mask, "zone"] = nearest
    logger.info("Imputed zone for %d Unknown events via nearest centroid", int(mask.sum()))
    return df


# ---------------------------------------------------------------------------
# Config (read from config.yaml; falls back to these defaults)
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM   = 6371.0
_DEFAULTS = {"dbscan_eps_km": 0.2, "dbscan_min_samples": 5,
             "kde_bandwidth": 0.04, "cluster_buffer_km": 0.2}


def _hotspot_cfg(project_root: Path) -> dict:
    """Load the hotspot section of config.yaml (single source of truth)."""
    cfg = dict(_DEFAULTS)
    try:
        import yaml
        loaded = yaml.safe_load((project_root / "config.yaml").read_text()) or {}
        for k, v in (loaded.get("hotspot", {}) or {}).items():
            if k in cfg:
                cfg[k] = v
    except Exception as exc:
        logger.warning("config.yaml not read (%s); using defaults", exc)
    return cfg


# Module-level defaults (overridden per-run from config)
DBSCAN_EPS         = _DEFAULTS["dbscan_eps_km"] / _EARTH_RADIUS_KM  # radians
DBSCAN_MIN_SAMPLES = _DEFAULTS["dbscan_min_samples"]


# ---------------------------------------------------------------------------
# Online cluster assignment — give a NEW (e.g. real-time) incident a
# cluster_label without re-running DBSCAN on the whole dataset.
#
# DBSCAN itself is batch; but for a live event we just need "does this point
# fall inside an existing hotspot's footprint?". We precompute each cluster's
# centroid and radius (from the committed features.csv) once, then snap a new
# point to the nearest cluster it lies within. Outside every footprint → noise
# (-1), exactly as DBSCAN would treat an isolated point.
# ---------------------------------------------------------------------------

_CENTROIDS_CACHE: list | None = None


def _haversine_km(lat0: float, lon0: float, lats, lons):
    R = _EARTH_RADIUS_KM
    p0 = np.radians(lat0)
    p = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dl = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def cluster_centroids(project_root: Path | None = None) -> list[dict]:
    """Centroid + footprint radius (km) per DBSCAN cluster, from features.csv. Cached."""
    global _CENTROIDS_CACHE
    if _CENTROIDS_CACHE is not None:
        return _CENTROIDS_CACHE
    root = project_root or Path(__file__).resolve().parents[1]
    feats = root / "data" / "processed" / "features.csv"
    if not feats.exists():
        _CENTROIDS_CACHE = []
        return _CENTROIDS_CACHE
    df = pd.read_csv(feats, usecols=lambda c: c in {"latitude", "longitude", "cluster_label"})
    df = df.dropna(subset=["latitude", "longitude", "cluster_label"])
    out = []
    for cid, g in df[df["cluster_label"] >= 0].groupby("cluster_label"):
        clat, clon = float(g["latitude"].mean()), float(g["longitude"].mean())
        d = _haversine_km(clat, clon, g["latitude"].to_numpy(), g["longitude"].to_numpy())
        out.append({"cluster": int(cid), "lat": clat, "lon": clon,
                    "radius_km": float(d.max()) if len(d) else 0.0})
    _CENTROIDS_CACHE = out
    return out


def assign_cluster(lat: float | None, lon: float | None,
                   project_root: Path | None = None, margin_km: float = 0.2) -> int:
    """
    Assign a new point to the nearest existing cluster whose footprint (radius +
    a small margin) contains it; -1 (noise) if it's outside every cluster.
    """
    if lat is None or lon is None:
        return -1
    cents = cluster_centroids(project_root)
    if not cents:
        return -1
    best, best_d = -1, float("inf")
    for c in cents:
        d = float(_haversine_km(c["lat"], c["lon"], np.array([lat]), np.array([lon]))[0])
        if d <= c["radius_km"] + margin_km and d < best_d:
            best, best_d = c["cluster"], d
    return int(best)


# ---------------------------------------------------------------------------
# 1. DBSCAN clustering
# ---------------------------------------------------------------------------

def run_dbscan(df: pd.DataFrame, eps_rad: float = DBSCAN_EPS,
               min_samples: int = DBSCAN_MIN_SAMPLES) -> pd.DataFrame:
    coords = df[["latitude", "longitude"]].values
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine",
                algorithm="ball_tree", n_jobs=-1)
    # haversine expects radians
    coords_rad = np.radians(coords)
    labels = db.fit_predict(coords_rad)
    df = df.copy()
    df["cluster_label"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    logger.info(
        "DBSCAN: %d clusters, %d noise points (%.1f%% coverage)",
        n_clusters, n_noise, 100 * (1 - n_noise / len(df))
    )
    return df


# ---------------------------------------------------------------------------
# 2. KDE heatmap points
# ---------------------------------------------------------------------------

def build_heatmap_points(df: pd.DataFrame) -> list[list[float]]:
    """Return [[lat, lon, weight], ...] for Folium HeatMap."""
    from scipy.stats import gaussian_kde

    lats = df["latitude"].values
    lons = df["longitude"].values
    pts  = np.vstack([lons, lats])

    try:
        kde = gaussian_kde(pts, bw_method=0.04)
        # Evaluate on a 60×60 grid
        lo_lon, hi_lon = lons.min() - 0.02, lons.max() + 0.02
        lo_lat, hi_lat = lats.min() - 0.02, lats.max() + 0.02
        glon, glat = np.meshgrid(
            np.linspace(lo_lon, hi_lon, 60),
            np.linspace(lo_lat, hi_lat, 60)
        )
        z = kde(np.vstack([glon.ravel(), glat.ravel()]))
        zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
        heat = [
            [float(la), float(lo), float(w)]
            for la, lo, w in zip(glat.ravel(), glon.ravel(), zn)
            if w > 0.08
        ]
    except Exception as exc:
        logger.warning("KDE failed (%s), falling back to raw points", exc)
        heat = [[float(la), float(lo), 1.0] for la, lo in zip(lats, lons)]

    logger.info("Heatmap: %d grid cells with weight > 0.08", len(heat))
    return heat


# ---------------------------------------------------------------------------
# 3. Moran's I
# ---------------------------------------------------------------------------

def compute_morans_i(df: pd.DataFrame) -> dict:
    """
    Test whether cluster membership is spatially autocorrelated.
    Returns dict with I, p_value, and interpretation.
    Falls back gracefully if pysal/esda is not installed.
    """
    result = {"I": None, "p_value": None, "significant": False,
              "interpretation": "esda/libpysal not installed — skipped"}

    try:
        from esda.moran import Moran
        from libpysal.weights import KNN

        coords = df[["latitude", "longitude"]].values
        is_clustered = (df["cluster_label"] >= 0).astype(float).values

        w = KNN.from_array(coords, k=8)
        w.transform = "r"

        mi = Moran(is_clustered, w)
        result = {
            "I":              float(mi.I),
            "p_value":        float(mi.p_norm),
            "significant":    bool(mi.p_norm < 0.05),
            "interpretation": (
                "Significant positive spatial autocorrelation — clusters are real"
                if mi.p_norm < 0.05
                else "Not significant — may be random"
            ),
        }
        logger.info(
            "Moran's I = %.4f  p = %.4f  → %s",
            mi.I, mi.p_norm, result["interpretation"]
        )
    except ImportError:
        logger.warning("esda/libpysal not available — Moran's I skipped")
    except Exception as exc:
        logger.warning("Moran's I failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# 4. GeoJSON polygon builder
# ---------------------------------------------------------------------------

def _circle_coords(lat: float, lon: float, radius_km: float = 0.2, n: int = 28) -> list:
    """
    GeoJSON ring approximating a circle of `radius_km` around (lat, lon).

    A fixed-radius circle around the cluster centroid is an honest depiction of
    the affected area.  A convex hull over road-aligned incidents produces giant
    triangles that wildly overstate the footprint, so we deliberately avoid it.
    """
    dlat = radius_km / 110.574                                   # km per deg latitude
    dlon = radius_km / (111.320 * np.cos(np.radians(lat)) + 1e-9)
    ring = [
        [lon + dlon * np.cos(t), lat + dlat * np.sin(t)]
        for t in np.linspace(0, 2 * np.pi, n)
    ]
    ring.append(ring[0])
    return [ring]


def build_geojson(df: pd.DataFrame, buffer_km: float = 0.2) -> dict:
    features = []
    for cid, grp in df[df["cluster_label"] >= 0].groupby("cluster_label"):
        lats = grp["latitude"].values
        lons = grp["longitude"].values

        dominant_cause = grp["event_cause"].mode().iloc[0] if len(grp) else "other"
        raw_junction   = grp["junction"].mode().iloc[0] if len(grp) else "unknown"
        top_address    = grp["address"].mode().iloc[0] if "address" in grp.columns and grp["address"].notna().any() else ""
        top_junction   = _best_junction_label(raw_junction, top_address, int(cid))
        avg_duration   = (
            grp["duration_minutes"].dropna().median()
            if "duration_minutes" in grp.columns
            else None
        )
        zone = grp["zone"].mode().iloc[0] if "zone" in grp.columns else "Unknown"

        feat = {
            "type": "Feature",
            "geometry": {
                "type":        "Polygon",
                "coordinates": _circle_coords(float(lats.mean()), float(lons.mean()), buffer_km),
            },
            "properties": {
                "cluster_id":           int(cid),
                "event_count":          int(len(grp)),
                "dominant_cause":       dominant_cause,
                "avg_duration_minutes": round(float(avg_duration), 1) if avg_duration else None,
                "top_junction":         top_junction,
                "zone":                 zone,
                "centroid_lat":         round(float(lats.mean()), 5),
                "centroid_lon":         round(float(lons.mean()), 5),
            },
        }
        features.append(feat)

    geojson = {"type": "FeatureCollection", "features": features}
    logger.info("GeoJSON: %d cluster polygons", len(features))
    return geojson


# ---------------------------------------------------------------------------
# 5. Hotspot ranking
# ---------------------------------------------------------------------------

def rank_hotspots(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top-N hotspot junctions, excluding 'unknown'."""
    clustered = df[df["cluster_label"] >= 0].copy()
    if clustered.empty:
        return pd.DataFrame()

    rows = []
    for cid, grp in clustered.groupby("cluster_label"):
        raw_junc  = grp["junction"].mode().iloc[0]
        top_addr  = grp["address"].mode().iloc[0] if "address" in grp.columns and grp["address"].notna().any() else ""
        dom_cause = grp["event_cause"].mode().iloc[0]
        avg_sev   = grp.get("severity_class", pd.Series(dtype=str)).mode()
        avg_dur   = grp["duration_minutes"].dropna().median() if "duration_minutes" in grp.columns else None
        rows.append({
            "cluster_id":           int(cid),
            "junction":             _best_junction_label(raw_junc, top_addr, int(cid)),
            "event_count":          len(grp),
            "dominant_cause":       dom_cause,
            "dominant_severity":    avg_sev.iloc[0] if len(avg_sev) else "Medium",
            "avg_duration_minutes": round(float(avg_dur), 1) if avg_dur else None,
            "zone":                 grp["zone"].mode().iloc[0] if "zone" in grp.columns else "Unknown",
            "centroid_lat":         round(float(grp["latitude"].mean()), 5),
            "centroid_lon":         round(float(grp["longitude"].mean()), 5),
        })

    all_hs = pd.DataFrame(rows).sort_values("event_count", ascending=False)
    hotspots = all_hs.head(top_n).reset_index(drop=True)
    logger.info("Top hotspot junctions:\n%s", hotspots[["junction","event_count","dominant_cause"]].to_string())
    return hotspots


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hotspot_engine(project_root: Path | None = None) -> dict:
    """
    Run the full hotspot pipeline.
    Returns dict with keys: hotspots, morans_i, geojson, heatmap_points.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    clean_csv = project_root / "data" / "processed" / "clean.csv"
    if not clean_csv.exists():
        import sys
        sys.path.insert(0, str(project_root / "src"))
        from data_pipeline import run_pipeline
        run_pipeline(project_root)

    df = pd.read_csv(clean_csv, parse_dates=["start_datetime"])

    # Bring in duration_minutes if features.csv exists
    feats_csv = project_root / "data" / "processed" / "features.csv"
    if feats_csv.exists():
        feats = pd.read_csv(feats_csv, usecols=["id", "duration_minutes", "severity_class"])
        df = df.merge(feats, on="id", how="left")

    # Fill Unknown zones using nearest zone centroid derived from known events
    zone_centroids = _build_zone_centroids(df)
    df = _impute_zones(df, zone_centroids)

    cfg = _hotspot_cfg(project_root)
    eps_rad = cfg["dbscan_eps_km"] / _EARTH_RADIUS_KM
    logger.info("Hotspot config: eps=%.0f m, min_samples=%d, buffer=%.0f m",
                cfg["dbscan_eps_km"] * 1000, cfg["dbscan_min_samples"],
                cfg["cluster_buffer_km"] * 1000)

    df = run_dbscan(df, eps_rad=eps_rad, min_samples=cfg["dbscan_min_samples"])
    heatmap_pts = build_heatmap_points(df)
    morans      = compute_morans_i(df)
    geojson     = build_geojson(df, buffer_km=cfg["cluster_buffer_km"])
    hotspots    = rank_hotspots(df)

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "hotspots.geojson", "w") as f:
        json.dump(geojson, f)
    logger.info("Saved hotspots.geojson")

    hotspots.to_csv(out_dir / "hotspot_summary.csv", index=False)

    # Save heatmap points
    pd.DataFrame(heatmap_pts, columns=["lat", "lon", "weight"]).to_csv(
        out_dir / "heatmap_points.csv", index=False
    )

    return {
        "hotspots":     hotspots,
        "morans_i":     morans,
        "geojson":      geojson,
        "heatmap_points": heatmap_pts,
        "df_with_labels": df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_hotspot_engine()
