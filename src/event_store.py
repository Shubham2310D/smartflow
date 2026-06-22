"""
event_store.py — Lightweight SQLite store for the real-time path.

The API previously had no memory of past events, so its history features
(junction_repeat_count / corridor_7d_score) had to be looked up from static
corridor medians. A genuinely real-time system needs live state: this is a
minimal, dependency-free SQLite store that

  • records each incoming event (record_event),
  • computes REAL backward-looking history for a new event from what it has seen
    (live_history) — no more static proxy once the store is warm,
  • serves the currently-active set for the live operations console
    (active_events), and
  • can be seeded from the historical active incidents (seed_from_features) so a
    demo console isn't empty.

SQLite (one file, no server) is enough for a demo; Postgres is the next step.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def db_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "data" / "processed" / "events.db"


def _conn(project_root: Path | None = None) -> sqlite3.Connection:
    path = db_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id            TEXT,
            ts            TEXT,        -- event start (ISO)
            event_cause   TEXT,
            zone          TEXT,
            corridor      TEXT,
            junction      TEXT,
            latitude      REAL,
            longitude     REAL,
            severity      TEXT,
            closure_prob  REAL,
            status        TEXT,
            cluster       INTEGER,     -- assigned DBSCAN cluster (online), -1 = noise
            logged_at     TEXT
        )
    """)
    # Migrate older DBs that predate the `cluster` column.
    existing = {row[1] for row in con.execute("PRAGMA table_info(events)").fetchall()}
    if "cluster" not in existing:
        con.execute("ALTER TABLE events ADD COLUMN cluster INTEGER")
    con.execute("CREATE INDEX IF NOT EXISTS ix_corridor ON events(corridor)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_status ON events(status)")
    con.commit()
    return con


def record_event(rec: dict, project_root: Path | None = None) -> None:
    """Insert one event into the live store."""
    cols = ["id", "ts", "event_cause", "zone", "corridor", "junction",
            "latitude", "longitude", "severity", "closure_prob", "status",
            "cluster", "logged_at"]
    row = {c: rec.get(c) for c in cols}
    if not row.get("logged_at"):
        row["logged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with _conn(project_root) as con:
        con.execute(
            f"INSERT INTO events ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )


def live_history(corridor: str | None, junction: str | None = None,
                 project_root: Path | None = None) -> dict:
    """
    REAL backward-looking history from the store (not a static median): events on
    this corridor in the trailing 7 days, and prior events at this junction.
    Returns {} when the store is empty so the caller can fall back gracefully.
    """
    with _conn(project_root) as con:
        n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if not n:
            return {}
        latest = con.execute("SELECT MAX(ts) FROM events").fetchone()[0]
        ref = pd.to_datetime(latest, errors="coerce")
        since = (ref - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ref) else "0"
        c7d = con.execute(
            "SELECT COUNT(*) FROM events WHERE corridor=? AND ts>=?",
            (corridor, since),
        ).fetchone()[0] if corridor else 0
        jrc = con.execute(
            "SELECT COUNT(*) FROM events WHERE junction=? AND junction<>'unknown'",
            (junction,),
        ).fetchone()[0] if junction else 0
    return {"corridor_7d_score": int(c7d), "junction_repeat_count": int(jrc)}


def active_events(project_root: Path | None = None, limit: int = 500) -> pd.DataFrame:
    """Currently-active events for the live operations console."""
    with _conn(project_root) as con:
        return pd.read_sql_query(
            "SELECT * FROM events WHERE status='active' ORDER BY ts DESC LIMIT ?",
            con, params=(limit,),
        )


def count(project_root: Path | None = None) -> int:
    with _conn(project_root) as con:
        return int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])


def seed_from_features(project_root: Path | None = None, force: bool = False) -> int:
    """
    Populate the store from the historical *active* incidents so the live console
    has something to show. Idempotent: skips if already seeded (unless force).
    """
    root = project_root or Path(__file__).resolve().parents[1]
    if not force and count(root) > 0:
        return 0
    feats = root / "data" / "processed" / "features.csv"
    if not feats.exists():
        return 0
    cols = ["id", "start_datetime", "event_cause", "zone", "corridor", "junction",
            "latitude", "longitude", "severity_class", "status"]
    df = pd.read_csv(feats, usecols=lambda c: c in cols)
    df = df[df.get("status") == "active"] if "status" in df.columns else df
    if df.empty:
        return 0
    with _conn(root) as con:
        if force:
            con.execute("DELETE FROM events")
        rows = [
            (r.get("id"), str(r.get("start_datetime")), r.get("event_cause"),
             r.get("zone"), r.get("corridor"), r.get("junction"),
             r.get("latitude"), r.get("longitude"), r.get("severity_class"),
             None, "active",
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
            for _, r in df.iterrows()
        ]
        con.executemany(
            "INSERT INTO events (id,ts,event_cause,zone,corridor,junction,"
            "latitude,longitude,severity,closure_prob,status,logged_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows,
        )
    return len(rows)
