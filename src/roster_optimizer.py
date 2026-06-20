"""
roster_optimizer.py — Allocate a fixed officer roster across simultaneous events.

The per-event recommender answers "how many officers does THIS event need?". It
cannot answer the real operational question: when several events are active at
once and officers are scarce, *who gets them?* That is a constrained allocation
problem, solved here as a min-cost flow.

Two inputs are ASSUMPTIONS, stated plainly because the dataset contains neither:
  1. Officer roster / station capacity — there is no staffing table in the data,
     so capacities are illustrative (configurable in config.yaml `roster` and in
     the UI). The optimiser is only as real as this supply side.
  2. Concurrency — five months of incidents are not naturally simultaneous, so we
     construct a scenario: the events that started within one clock-hour window
     (default: the busiest such window). This is a built scenario, not a live
     snapshot.

Given those, the allocation itself is a genuine optimisation:

    minimise   sum(travel_cost · officers_sent)  +  sum(unmet_penalty · officers_short)

    s.t.       each event receives exactly its demand (officers OR an explicit
               "unmet" shortfall);
               each station sends at most its capacity.

Modelled as min-cost flow:  SOURCE → stations (cap = roster) → events (cost =
travel km) and SOURCE → UNMET → events (cost = severity-weighted penalty). The
penalty exceeds any travel cost, so a real officer always beats leaving demand
unmet; and High-severity penalties exceed Low, so when the roster runs out the
solver leaves the *lowest-priority* demand short first.
"""
from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from resource_recommender import recommend
from utils import get_nearest_station, load_config

# Severity → unmet-penalty weight. All exceed the max plausible travel cost
# (~600 = 60 km · 10), so officers are always preferred over a shortfall; the
# ordering makes the solver sacrifice Low before Medium before High.
_UNMET_PENALTY = {"High": 9000, "Medium": 3000, "Low": 1000}
_TRAVEL_SCALE = 10  # km → integer weight (min-cost flow needs integer weights)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def station_locations(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Data-derived station coordinates: the centroid of all events whose zone maps
    that station as nearest. The dataset has no station coordinates, so this
    grounds each station's position in where its incidents actually occur.
    """
    pts = df.dropna(subset=["latitude", "longitude", "zone"]).copy()
    pts["station"] = pts["zone"].map(get_nearest_station)
    locs = pts.groupby("station")[["latitude", "longitude"]].mean()
    return {s: (float(r.latitude), float(r.longitude)) for s, r in locs.iterrows()}


def roster_capacities(stations: list[str], officers_per_station: int | None = None) -> dict[str, int]:
    """Assumed officer capacity per station (uniform default, config-overridable)."""
    cfg = {}
    try:
        cfg = load_config().get("roster", {}) or {}
    except Exception:
        pass
    default = officers_per_station if officers_per_station is not None \
        else int(cfg.get("default_officers_per_station", 6))
    overrides = cfg.get("stations", {}) or {}
    return {s: int(overrides.get(s, default)) for s in stations}


def build_scenario(df: pd.DataFrame, when: pd.Timestamp | None = None,
                   status: str = "active") -> dict:
    """
    Construct a concurrency scenario: events that started within one clock-hour.

    `when` selects the hour window; if None, the busiest such window is used.
    Returns the window and a list of event dicts with demand from the recommender.
    """
    pool = df.copy()
    if status and "status" in pool.columns:
        pool = pool[pool["status"] == status]
    pool = pool.dropna(subset=["latitude", "longitude", "start_datetime"])
    hour = pool["start_datetime"].dt.tz_localize(None).dt.floor("h")
    pool = pool.assign(_hour=hour)

    if when is None:
        when = pool["_hour"].value_counts().idxmax()
    else:
        when = pd.Timestamp(when).floor("h")

    rows = pool[pool["_hour"] == when]
    events = []
    for _, r in rows.iterrows():
        sev = str(r.get("severity_class", "Medium")) or "Medium"
        rec = recommend(
            severity_class=sev,
            event_cause=r.get("event_cause", "other"),
            requires_road_closure=bool(r.get("road_closure_binary", 0)),
            hour_of_day=int(when.hour),
            zone=r.get("zone", "Unknown"),
        )
        events.append({
            "id": r.get("id"),
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "zone": r.get("zone", "Unknown"),
            "cause": r.get("event_cause", "other"),
            "severity": sev,
            "demand": int(rec["personnel_count"]),
            "nearest_station": rec["dispatch_from"],
            "priority": rec["priority_flag"],
        })
    return {"window": when, "events": events}


def optimize(events: list[dict], capacities: dict[str, int],
             station_locs: dict[str, tuple[float, float]]) -> dict:
    """
    Min-cost-flow allocation of the roster across the scenario's events.

    Returns per-event allocations (station → officers), per-event shortfall, and
    scenario totals (demand met, officers used, total travel km).
    """
    total_demand = sum(e["demand"] for e in events)
    stations = [s for s in capacities if s in station_locs and capacities[s] > 0]

    if total_demand == 0 or not stations:
        return {
            "allocations": {e["id"]: {} for e in events},
            "unmet": {e["id"]: e["demand"] for e in events},
            "total_demand": total_demand, "met": 0,
            "officers_used": 0, "total_travel_km": 0.0,
            "roster_size": sum(capacities.get(s, 0) for s in stations),
        }

    G = nx.DiGraph()
    G.add_node("SRC", demand=-total_demand)
    G.add_node("UNMET", demand=0)
    G.add_edge("SRC", "UNMET", capacity=total_demand, weight=0)
    for s in stations:
        G.add_edge("SRC", f"ST::{s}", capacity=int(capacities[s]), weight=0)

    for e in events:
        ev = f"EV::{e['id']}"
        G.add_node(ev, demand=int(e["demand"]))
        penalty = _UNMET_PENALTY.get(e["severity"], _UNMET_PENALTY["Medium"])
        G.add_edge("UNMET", ev, capacity=int(e["demand"]), weight=penalty)
        for s in stations:
            slat, slon = station_locs[s]
            km = _haversine_km(slat, slon, e["lat"], e["lon"])
            G.add_edge(f"ST::{s}", ev,
                       capacity=int(e["demand"]),
                       weight=int(round(km * _TRAVEL_SCALE)))

    flow = nx.min_cost_flow(G)

    allocations, unmet = {}, {}
    officers_used, total_travel_km = 0, 0.0
    for e in events:
        ev = f"EV::{e['id']}"
        alloc = {}
        for s in stations:
            f = flow.get(f"ST::{s}", {}).get(ev, 0)
            if f > 0:
                alloc[s] = f
                officers_used += f
                total_travel_km += _haversine_km(*station_locs[s], e["lat"], e["lon"]) * f
        allocations[e["id"]] = alloc
        unmet[e["id"]] = flow.get("UNMET", {}).get(ev, 0)

    return {
        "allocations": allocations,
        "unmet": unmet,
        "total_demand": total_demand,
        "met": total_demand - sum(unmet.values()),
        "officers_used": officers_used,
        "total_travel_km": round(total_travel_km, 1),
        "roster_size": sum(capacities.get(s, 0) for s in stations),
    }
