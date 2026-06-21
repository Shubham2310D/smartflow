# SmartFlow — Event-Driven Traffic Intelligence Platform

> Flipkart Hackathon Round 2 | Intelligent Traffic Management System for Bengaluru

---

## Dashboard Preview

| Home — KPIs & Charts | Hotspot Map |
|---|---|
| ![Home](snapshots/home.png) | ![Hotspot Map](snapshots/hotspot_map.png) |

| Predict Event + SHAP | Resource Plan |
|---|---|
| ![Predict](snapshots/predict_result.png) | ![Resource](snapshots/resource.png) |

| Analytics | Roster Optimizer |
|---|---|
| ![Analytics](snapshots/analytics.png) | ![Roster Optimizer](snapshots/roster_optimizer.png) |

| Event Planner | Live Ops Console |
|---|---|
| ![Event Planner](snapshots/event_planner.png) | ![Live Ops](snapshots/live_ops.png) |

---

## Problem

Bengaluru processes thousands of road incidents every month — accidents, floods, vehicle breakdowns, construction blocks. For Flipkart's last-mile logistics, each undetected hotspot or delayed response costs delivery SLA time and fleet efficiency. Operators currently lack a unified system to predict incident severity, identify recurring hotspots, and deploy resources before queues form.

---

## Approach — two layers, scoped to what the data supports

The brief spans two distinct regimes, and so does this dataset. Rather than build one model that pretends they're the same problem, SmartFlow is **explicitly two layers**:

- **Layer A — Background Incident Operations** (≈92% of the data: vehicle breakdowns, potholes, water-logging). These are *reactive*: an incident is logged, then triaged and cleared. This layer does real-time triage, hotspot detection, **road-closure prediction** (the headline model — a real observed target, now lifted by an OSM road-class join), **real diversion routing**, and resource dispatch.
- **Layer B — Planned-Event Impact** (≈8%: `public_event`, `procession`, `vip_movement`, `protest`, plus `construction`). These have a known type, place, and time *ahead of time*, so they're handled by **case-based retrieval** — pull the most similar past events and surface what they actually required — rather than a deep learned forecaster the ~130 gathering rows can't support.

This split is a deliberate scoping choice, not a limitation we're hiding: the dataset is incident-heavy, so we make the incident layer genuinely operational and answer the planned-event half with honest case-based evidence. Where the brief asks for a measured *congestion impact* (queue length, speed drop), the data contains none — see [Honest Limitations](#honest-limitations).

---

## Solution

SmartFlow is built on 8,057 real Bengaluru traffic incidents (Astram dataset, cleaned from 8,173 raw rows):

| Module | What it does |
|---|---|
| **Data Pipeline** | Cleans raw Astram CSV — null handling, timestamps normalised to naive Bengaluru-local time at ingestion (the `+00` tag was never truly UTC — see Design Decisions), event-aware cause vocabulary (procession / VIP / protest kept as first-class causes) |
| **OSM Road Join** | Snaps every incident to the nearest **major road** on the cached OpenStreetMap network → `road_class` + `lane_count` (~99% of incidents matched). The single biggest external upgrade — see metrics |
| **ML Models** | **Road-closure predictor** (calibrated, ROC-AUC 0.720 single-split / 0.671±0.051 walk-forward on a *real observed* target) + severity triage classifier (81.3% chronological holdout vs 63.2% baseline) — both lifted by the road-class join |
| **Free-text Mining** | Bilingual (English + Kannada) keyword pass over `description` → event semantic type (sports event, utility work, VIP movement, …) |
| **Hotspot Engine** | DBSCAN spatial clustering (config-driven 200 m radius) + KDE density heatmap + a **spatial-temporal holdout** proving the hotspots forecast next-month incidents (PAI up to 4.7× chance) |
| **Resource Recommender** | Config-driven scoring → personnel, barricade, diversion, dispatch station + empirical clearance range |
| **Diversion Router** | A **real reroute** around a blockage on the OSM road graph (networkx) — detour path + added distance, not a yes/no flag |
| **Roster Optimizer** | Min-cost-flow allocation of a fixed officer roster across simultaneous events — high-priority first, travel minimised (the *optimal* in "optimal deployment") |
| **Live Ops Console** | Map-first command center: active incidents (SQLite store) with per-event deployment, nearest station, barricade alerts, and an on-map diversion planner |
| **Impact Heuristic** | Transparent weighted composite (closure × corridor-pressure × **road-capacity** × high-incident-window) — the road term uses measured OSM class/lanes |
| **External Feeds** | Drop-in scaffolds for the two roadmap joins — a speed feed (measured-impact ground truth) and an event calendar (advance forecasting) — OFF by default, honest "unavailable" until wired |
| **Event Planner** | Advance what-if for a known upcoming event (type + place + date/time) → impact forecast + full deployment plan, grounded in similar past events with a confidence rating |
| **Case-Based Forecast** | Retrieves similar past events (graceful backoff: nearby → type-in-zone → type-citywide → similar) and what they actually required, with a confidence tier |
| **Feedback Loop** | Logs each decision, tracks predicted-vs-actual, and **closes the loop**: `learning_loop.py` re-fits and records a drift snapshot to `metrics_history.csv` |
| **Real-time API** | `POST /event` (FastAPI) runs the same predict → recommend → log path for streaming use |

---

## Key Metrics

| Metric | Value |
|---|---|
| Events processed | 8,057 (from 8,173 raw) |
| Incidents snapped to a major OSM road | **~99%** (≈7,945 / 8,057) |
| **Road-closure model ROC-AUC** (real observed target, calibrated) | **0.720** single-split; **0.671 ± 0.051** walk-forward (PR-AUC 0.21) vs 7.4% base rate — *up from 0.695 / 0.17 before the road-class join* |
| Barricade threshold | **0.15** — *derived* from a cost tradeoff (missed closure = 5× a wasted barricade), not asserted |
| Severity triage accuracy (chronological holdout) | 81.3% — vs 63.2% majority baseline (*up from 79.8% with the road-class join*) |
| Severity triage accuracy (random 5-fold CV) | 86%+ (optimistic — ignores time order) |
| Clearance predictor vs median baseline | MAE 102.7 min vs 106.2 median — **+3.5 min lift** (modest, reported honestly) |
| **Hotspot predictiveness** (spatial-temporal holdout) | densest **2.3%** of the city captured **11%** of *next month's* incidents (**4.7× chance**); 18% captured 49% (2.7×) |
| Median incident clearance | 57 min |
| Top hotspot | Sankey Road — 764 events |

> **Which model is the "real" one?** The **road-closure predictor** is — it
> learns a genuinely *observed* outcome (`requires_road_closure`) and is
> calibrated, so its probability is trustworthy. Severity is a *rules-based
> triage* label (priority + closure + cause), so its classifier is framed as
> triage, not impact prediction. The clearance regressor does **not** beat a
> naive median, so we show it as an empirical range, not a forecast.

---

## Dashboard Pages

1. **Home** — KPI metrics, severity distribution, top corridors
2. **Hotspot Map** — Folium heatmap + DBSCAN cluster circles with tooltips, plus the **spatial-temporal validation** banner (do past hotspots predict the future?)
3. **Predict Event** — Severity triage + **calibrated road-closure likelihood** + typical clearance range + road-aware impact + SHAP + live text mining
4. **Resource Plan** — Deployment recommendation, **case-based analog panel** for planned events, decision logging
5. **Analytics** — 7 Plotly charts (incl. free-text event type)
6. **Feedback Loop** — Predicted-vs-actual backtest (with median baseline) + live operator decision log
7. **Roster Optimizer** — Multi-event conflict view: allocate a fixed officer roster across simultaneously-active events (min-cost flow), with under-resourced events flagged
8. **Event Planner** — Advance what-if simulator: pick an upcoming event's type, place and date/time → full deployment plan + impact forecast + analog evidence, in one pane
9. **Live Ops Console** — Map-first command center: active incidents with per-event recommendation, nearest station, barricade alerts (served from the SQLite store), and an on-map **diversion planner** that routes a real detour around a chosen blockage

---

## Project Structure

```
smartflow/
├── src/
│   ├── data_pipeline.py       # Load & clean raw CSV
│   ├── feature_engineering.py # Features + bilingual text mining + OSM road join
│   ├── osm_features.py        # OpenStreetMap road-class / lane-count join (cached)
│   ├── model_training.py      # Closure predictor, severity triage, clearance stats
│   ├── hotspot_engine.py      # DBSCAN + KDE + GeoJSON (config-driven)
│   ├── hotspot_validation.py  # Spatial-temporal holdout (hit-rate + PAI)
│   ├── diversion.py           # Real reroute around a blockage (OSM graph, networkx)
│   ├── resource_recommender.py# Config-driven per-event deployment logic
│   ├── roster_optimizer.py    # Min-cost-flow allocation across concurrent events
│   ├── event_analog.py        # Case-based retrieval (backoff + confidence)
│   ├── event_planner.py       # Advance plan for an upcoming event (impact + deployment)
│   ├── impact_score.py        # Transparent disruption-impact heuristic (road-aware)
│   ├── external_feeds.py      # Speed-feed + event-calendar scaffolds (roadmap joins)
│   ├── notifications.py       # Telegram officer push alerts (opt-in, off by default)
│   ├── event_store.py         # SQLite real-time event store (live history + active set)
│   ├── history_features.py    # Backward-looking history lookup for inference
│   ├── outcomes_log.py        # Decision/outcome log (append-only CSV)
│   ├── learning_loop.py       # Closes the loop: retrain + metrics_history drift
│   └── utils.py               # Zone/station mappings, constants
├── dashboard/
│   ├── app.py                 # Home page
│   └── pages/
│       ├── 1_Hotspot_Map.py
│       ├── 2_Predict_Event.py
│       ├── 3_Resource_Plan.py
│       ├── 4_Analytics.py
│       ├── 5_Feedback_Loop.py
│       ├── 6_Roster_Optimizer.py
│       ├── 7_Event_Planner.py
│       └── 8_Live_Ops.py
├── api/
│   └── main.py                # FastAPI real-time endpoint
├── tests/
│   ├── test_leakage.py        # Asserts history features are backward-looking
│   ├── test_history_features.py # Asserts inference uses real history, not a constant
│   ├── test_roster_optimizer.py # Asserts allocation conserves demand & triages by priority
│   ├── test_event_planner.py  # Asserts confidence backoff & obstruction handling
│   ├── test_learning_loop.py  # Asserts the loop reads outcomes back & records drift
│   ├── test_model_versioning.py # Asserts version-skew detection for pickled models
│   ├── test_impact_score.py   # Asserts the impact heuristic is bounded & monotonic
│   ├── test_ingest_validation.py # Asserts schema/range checks on the feed
│   ├── test_event_store.py    # Asserts live history & active-set from the store
│   ├── test_model_performance.py # Asserts the shipped models clear a quality floor
│   ├── test_diversion.py      # Asserts the reroute is feasible & degrades gracefully
│   ├── test_external_feeds.py # Asserts the roadmap scaffolds work (calendar + validation)
│   └── test_notifications.py  # Asserts officer alerts no-op safely + format correctly
├── data/
│   ├── raw/                   # Place events.csv here
│   └── processed/             # Auto-generated outputs
├── models/                    # Trained .pkl files
├── .github/workflows/ci.yml   # CI: pytest + Docker build on push/PR
├── Dockerfile                 # Dashboard container (out-of-the-box demo)
├── config.yaml               # Single source of truth (wired into code)
└── requirements.txt
```

---

## Instructions to Run

### 1. Prerequisites

- Python 3.10 or higher (tested on 3.13)
- The raw Astram events CSV (`events.csv`) placed in `data/raw/`

### 2. Install dependencies

```bash
cd smartflow
pip install --prefer-binary -r requirements.txt
pip install shap --prefer-binary
```

### 3. Run the data pipeline

```bash
python src/data_pipeline.py
python src/feature_engineering.py   # also runs the OSM road-class join
```

> **OSM road join:** the first feature-engineering run fetches the Bengaluru **major-road** network from Overpass (one query) and caches it slimmed+gzipped to `data/processed/osm_roads.json.gz` (~1 MB, committed). Every later run is offline. If Overpass is unreachable and no cache exists, the join degrades to safe defaults rather than failing.

### 4. Train the ML models

```bash
python src/model_training.py
```

### 5. Generate hotspot data

```bash
python src/hotspot_engine.py
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Quick start (all steps in one):** The dashboard auto-runs the pipeline on first load if processed files are missing. Just run `streamlit run dashboard/app.py` after placing `events.csv` in `data/raw/`.

### Run with Docker

The processed data and trained models are committed, so the dashboard runs out of the box — no raw CSV or training step needed:

```bash
docker build -t smartflow .
docker run -p 8501:8501 smartflow      # → http://localhost:8501
```

### Tests & CI

```bash
pytest tests/        # 46 tests: leakage, history, optimizer, planner, learning loop, versioning, impact, ingest, event store, model-performance floor, diversion, external feeds, notifications
```

[GitHub Actions](.github/workflows/ci.yml) runs the test suite **and** a Docker image build on every push / PR.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy, scipy |
| Machine learning | XGBoost, scikit-learn |
| Spatial analysis | DBSCAN (sklearn, haversine), KDE (scipy) |
| Explainability | SHAP (TreeExplainer, waterfall chart) |
| Calibration | scikit-learn (isotonic) for the closure model |
| Dashboard / API | Streamlit, Plotly, Folium / FastAPI |
| Config | PyYAML (wired into the code, not decorative) |

---

## Design Decisions

- **The real model predicts a real target: road closure.** `requires_road_closure` is *observed* (not a derived label) and is exactly what drives barricading/diversion — so it is genuinely learnable and operationally meaningful. We predict it with a class-weighted, **isotonic-calibrated** XGBoost on a chronological split. At a 7.4% base rate, accuracy is meaningless, so we report **ROC-AUC 0.70 / PR-AUC** and present the calibrated probability against the base rate. This is the headline model, replacing the synthetic-label classifier as the thing to trust.
- **Severity is a triage classifier, not a congestion-impact predictor** — its label is rules-derived (priority + closure + cause) and real-world impact (queue length, delay) is never measured in this data. So it is framed as *triage*. The honest result still stands: **79.8% on a chronological holdout vs a 63.2% baseline** from spatial-temporal context alone (the leakage-free spatial cluster feature lifted it from ~75%).
- **Contextual-only severity features (no leakage, no text)** — cause/closure are excluded because the label is derived from them; the text-derived `event_semantic_type` is *also* excluded from the classifier because it is a cause-proxy that would re-introduce the same circularity. It is used only by the duration model, where cause-like features are legitimate.
- **Chronological validation, not random** — operational forecasting must train on the past and predict the future. We split by time (earlier 80% / later 20%) instead of a random split. This is why the headline accuracy (79.8%) is lower than the conventional random CV (~86%) — and more trustworthy.
- **`month` dropped as a feature** — coverage is only **Nov 2023–Apr 2024 (~5 months)**, so under a chronological split the test months barely appear in training: `month` can't learn seasonality from this window and acts as a mild leak. We verified removal is neutral-to-positive — severity accuracy held and the closure model's minority-class metrics *improved* (PR-AUC 0.16 → 0.17, recall 0.02 → 0.06, precision 0.20 → 0.44) — so it's gone from all three models.
- **Hotspot output fed back as features — and kept only where it measurably helps.** The DBSCAN clusters were computed but never used by the models. We added two **leakage-free, backward-looking** cluster signals (prior events in the cluster; prior closure rate), then *measured* their effect per model rather than assuming it: they lift severity (**0.75 → 0.80**) and duration, but **hurt** the chronological-holdout closure AUC (0.695 → 0.63 — a per-cluster rate is too noisy at a 7% base rate), so they are excluded from closure. Measured, not assumed.
- **Walk-forward validation, not a lone split.** A single 80/20 chronological cut is one noisy estimate on ~600 rows. The closure model is also backtested over 5 expanding-window folds: **ROC-AUC 0.666 ± 0.057** (the single-split 0.695 sits at the optimistic end). Shown with the band on the Feedback Loop page.
- **The barricade threshold is derived from a cost tradeoff, not asserted.** A missed real closure (no barricade when a road shuts) is weighted **5× a wasted barricade**; the cost-minimising threshold on the holdout is **0.15** (lower than a naive 0.30 — catching closures matters more). The precision/recall/cost curve is shown on the Feedback Loop page and stamped into the model.
- **The biggest single upgrade: an OSM road-class join.** The raw data has lat/lon but no road context — a closure on a 6-lane trunk road and one on a residential lane looked identical to the models. `osm_features.py` does one Overpass query for the Bengaluru **major-road** network (motorway→tertiary; residential is dropped — it's 90% of ways, irrelevant to arterial diversion, and 5× the memory — so the cache is ~1 MB gzip, committed for offline rebuilds), then snaps every incident to its nearest road (KD-tree, **~99% matched**) → `road_class` + `lane_count`. *Measured*, not assumed: it lifts the headline closure model (**ROC-AUC 0.695 → 0.720, PR-AUC 0.17 → 0.21**) and severity (**79.8% → 81.3%**), and is neutral for duration (kept there for the SHAP story). The real-time API snaps each event to its exact road; the dashboard uses the corridor's typical road class.
- **Hotspots are validated as predictive, not just described.** Moran's I confirms the clusters are spatially *real*; `hotspot_validation.py` goes further with a strict **spatial-temporal holdout** (cluster on every month but the last, score the held-out final month) using the standard hotspot-policing metrics — hit-rate and PAI (hit-rate ÷ area-share, Chainey et al. 2008). The densest **2.3%** of the city captured **11%** of the next month's incidents (**4.7× chance**); tightening trades coverage for precision monotonically (PAI 1.5 → 7.0). Shown on the Hotspot Map page.
- **Diversion is now routed, not just flagged.** `diversion.py` builds a road graph from the cached OSM network (networkx), models the closure as a zone around the incident, finds the road's two opposite exits, and runs Dijkstra around the closure — a **real detour with an added-distance cost** (e.g. ORR @ Marathahalli: a 2.5 km reroute). Surfaced on-map in Live Ops and returned by the API when a location is given. The boolean rule still decides *whether* to divert; this answers *how*.
- **Impact is a transparent heuristic, now grounded in measured road capacity.** The data has no measured congestion outcome (queue/speed/delay), so "impact" still can't be fully learned — but the heuristic composite now adds a **road-capacity term from the OSM class + lane count** (closure × corridor-pressure × **road-capacity** × high-incident-window), so a closure on an arterial scores above one on a lane. Every weight is visible and the UI labels it a heuristic; `external_feeds.validate_impact_against_speed()` is implemented and runs the moment a speed feed supplies the ground truth.
- **A model-performance regression test guards the learning loop.** `test_model_performance.py` fails CI if a retrain drops the closure model below an AUC floor, lets severity fall to the majority baseline, lets duration lose to the median, or silently drops the road-context features — so a bad retrain can't ship quietly.
- **Officer push alerts close the last-mile gap.** A barricade recommendation that needs someone watching a dashboard isn't operational. `notifications.py` pushes a **Telegram alert to an officers' group** when a barricade is warranted — fired automatically by the real-time API `/event` path and on-demand from a "🔔 Alert officers" button on Live Ops (with the incident's severity, location, recommended crew, and the diversion summary). Opt-in and OFF by default: credentials come from env vars (`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`, never committed), and with none set every call is a silent no-op, so it never blocks a request or breaks the demo. Telegram over SMS/web-push: free, no number verification, real phone push, 5-minute setup.
- **The two structural gaps have honest, drop-in scaffolds.** A measured-impact speed feed and an advance event calendar need an external source we don't fabricate. `external_feeds.py` is the seam: OFF by default, a clear `{"available": False, "reason": ...}` when unconfigured (never invented numbers), config-driven keys/paths, and the calendar load/filter + impact-validation paths fully implemented and tested — so "this would close the gap" is demonstrable, not hand-waved.
- **Real-time path has live state.** A lightweight **SQLite event store** records each event the API sees and computes *real* backward-looking history (replacing the static corridor-median proxy once warm). The **Live Operations Console** reads the active set for a map-first command center with per-event deployment and barricade alerts.
- **Defensive ingest + concurrency.** Ingestion runs a schema/range check (Bengaluru coordinate bounds, required columns, null/NaT rates) that fails hard on a missing column and warns on dirt; the decisions log is written under a portable file lock with an atomic replace, so the Streamlit app and API can't corrupt it with concurrent writes.
- **Colour-blind-safe severity.** Severity is never conveyed by colour alone — every display pairs the red/amber/green with a distinct shape (▲/●/■) and the text label, and map markers also encode it by size.
- **Responsive, mobile-usable UI.** Streamlit is desktop-first — columns stay side-by-side on a phone, squishing metrics and cards into unreadable slivers. A single shared stylesheet (`utils.inject_responsive_css()`, injected on every page) fixes that: below a 900 px / 640 px breakpoint, column rows wrap to full width, padding tightens, headings scale, and maps/tables/charts stay inside the viewport so the page never scrolls sideways. A field officer can open the Live Ops console or the Predict form on a phone and actually use it — verified at a 390 px viewport.
- **Leakage-free junction history** — `junction_repeat_count` counts only events that occurred *before* each event at the same junction (an expanding count), zeroed for the catch-all "unknown" junction so it can't act as a disguised time index.
- **Inference uses real history, not a fabricated constant** — a new event has no `junction_repeat_count` / `corridor_7d_score` of its own, and these features *do* move the prediction. Earlier code fed a hardcoded `5`, so every live prediction depended on a made-up value. Both the Predict page and the API now look these up from the corridor's **historical medians** (global-median fallback for unknown corridors) via one shared `history_features` module; an explicit override is still accepted for when a real event store can supply a live count. This is a typical-rate proxy, not a live count — a true live store is on the roadmap.
- **Timezone normalised once at ingestion; "high-incident window" is data-derived, not assumed.** The raw timestamps carry a `+00` tag, but their wall-clock already behaves as Bengaluru **local** time: converting to IST empties the evening rush (18–21h drops to ~8–52 events) and invents a 2 AM peak. So the pipeline strips the misleading tag **once, at ingestion** (`data_pipeline._parse_datetimes` → naive local), and every downstream consumer works in plain local time — no per-file tz juggling and nobody tempted to "correct" a UTC tag that was never truly UTC. And because this feed is ~60% truck breakdowns, incident volume peaks in the **freight window (evening + pre-dawn)**, *not* the textbook 08–10 / 17–20 commuter rush. So the high-incident window is defined **from the data** (hours whose volume exceeds the daily mean), lives in `config.yaml`, and is shared by training, the recommender, the API and the dashboard through one helper. The user-facing label is **"high-incident window"**, not "peak hour", so an operator doesn't misread it as commuter rush. (Measured honestly: the `is_peak_hour` flag is redundant with raw `hour_of_day` for the models — closure AUC unchanged at 0.695 — so its real value is the recommender's personnel bonus now firing on the actual load pattern, e.g. a 21:00 incident, not the wrong commuter hours.)
- **Event-aware cause vocabulary** — `procession`, `vip_movement`, `protest`, `public_event` are kept as first-class causes rather than collapsed into "other", because planned/unplanned gatherings are exactly the event types this problem targets.
- **Free-text mining (English + Kannada)** — the `description` field (83% populated, bilingual) is mined with a keyword pass into an `event_semantic_type` (sports event, utility work, VIP movement, …). It powers the auto-categorisation chart in Analytics and the live extraction on the Predict page. **We measured its effect honestly: it does *not* improve clearance-time MAE** (see below) — but it recovers event semantics the structured cause column misses.
- **The clearance regressor is reported against its baseline** — on the chronological holdout the XGBoost MAE (~106 min) is **statistically indistinguishable from "always predict the median" (~106 min)**. Clearance time here is dominated by unobserved operational factors (crew dispatch, on-scene complexity), so we present the estimate as a rough prior, show the baseline beside it on the Feedback Loop page, and have the Resource Plan fall back to cause-based medians. We'd rather show this than dress up a constant as a model.
- **Planned events get a case-based forecast, not a model** — processions, VIP movement, protests and public events have a known type and place ahead of time. The **Event Planner** page turns that into a single advance what-if: pick type + place + date/time and get the impact forecast (expected severity, road-closure likelihood, clearance range) *and* the deployment plan (personnel, barricade, diversion, dispatch) in one pane, before the event. It directly answers the "recommend manpower for an upcoming event" half of the brief without inventing a model the ~130 gathering rows can't support.
- **Retrieval backs off gracefully and rates its own confidence** — a query keys on *type-and-place*, but ~15 protests citywide means an exact local match can return nothing. So retrieval degrades **nearby → type-in-zone → type-citywide → similar-events**, surfaces how many analogs it found, and labels the result **high / medium / low confidence** so a thin match reads as a rough prior, not a confident answer on one data point. **Construction is handled as its own sub-case** — a sustained obstruction, not a crowd — so it doesn't inherit crowd-control personnel logic.
- **"Optimal" deployment is an actual optimisation, not a scoring table** — the per-event recommender says how many officers *one* event needs; it can't resolve the real constraint, which is scarce officers across *simultaneous* events. The Roster Optimizer models that as a **min-cost flow**: a fixed roster (station capacities) flows to events at a cost of travel distance, with a severity-weighted penalty path for unmet demand. The result minimises travel while serving high-priority events first, and leaves the lowest-priority demand short when the roster can't cover everything — a genuine "optimal given the constraints" plan. Two inputs are assumed and labelled as such (roster capacity and the concurrency scenario — see Limitations); everything else (demand, travel, allocation) is real.
- **Config is the single source of truth** — `config.yaml` is read by the hotspot engine (DBSCAN eps, min samples, KDE bandwidth, cluster radius) and the resource recommender (personnel, bonuses, barricade causes, closure threshold, **data-derived peak hours**). It is no longer decorative; change the file and behaviour changes.
- **Honest cluster footprints** — clusters render as fixed-radius circles around the centroid, not convex hulls. Hulls over road-aligned incidents produce giant triangles that overstate the affected area.
- **DBSCAN over k-means** — no need to pre-specify cluster count; naturally handles noise; 200 m haversine radius tuned to Bengaluru block size.
- **Median over mean for clearance** — 9.7% of tickets were never properly closed, inflating the mean to 552 min. Median (57 min) reflects real operational close-time.
- **Models record the versions they were trained with** — a pickled sklearn/xgboost model can silently break or mis-load under a different library version. Each model payload stamps its `scikit-learn` / `xgboost` / `numpy` (and python/pandas) versions; `check_lib_versions()` compares them to the runtime and the API `/health` endpoint and the dashboard surface a non-fatal warning on skew, so a version-mismatched environment is flagged rather than failing mysteriously.
- **The post-event learning loop is closed in code, not just logged** — the brief names "no post-event learning system" as a core pain. Logging alone is an open loop, so `learning_loop.py` reads `decisions_log.csv` back to measure recommendation-vs-actual accuracy (clearance MAE + severity accuracy), re-fits the models, and appends a snapshot to `metrics_history.csv`. Run on a schedule, that file makes **drift across retrains** visible on the Feedback Loop page — the difference between a loop that's real and one that's merely scaffolded. (Honest scope: the re-fit reads the canonical feature set, which grows as the pipeline ingests resolved events; a raw log row lacks the engineered features to train on directly.)

---

## Honest Limitations

We'd rather state these than have a reviewer find them:

- **Dataset is incident-heavy.** ~92% of records are incidents (vehicle breakdowns, potholes, water-logging). True event/gathering rows (procession, VIP, protest, public event) are ~130 events. SmartFlow surfaces them as first-class, but the data can't support a deep event-specific forecaster yet.
- **No measured congestion outcome.** The dataset records no queue length, speed drop, or delay. "Severity" is a rules-derived label, not an observed impact — so the classifier is framed as *triage*, and the only genuine outcome we model is **clearance time** (`duration_minutes`).
- **Clearance is mostly clerical, and unpredictable from these features.** Of ~2,760 usable durations, only **69** come from a real `resolved_datetime`; the rest come from `closed_datetime`, an administrative ticket-close that is often batched. So the target is largely paperwork timing — which is why no regressor beats the median, and why we present clearance as an empirical range labelled "close-time", not "time to clear".
- **The duration target is selection-biased.** Only **43% (3,061 of 7,058)** of closed/resolved tickets carry a usable close-time delta — **57% (3,997)** were never properly closed. That missingness is **not random**: clean, fast-resolving tickets tend to get closed properly, while messy long-running ones are abandoned open, so the trainable subset skews toward *shorter* events and the reported MAE is optimistic. We surface this rather than hide it, and lean on the (bias-aware) empirical median instead of a point forecast.
- **No *measured* congestion impact (partly addressed).** The CSV has no queue length, speed drop, or delay, so impact still can't be directly *measured* from it. We've closed half the gap — the OSM road-class / lane-count join is now live and feeds a road-capacity term into the impact heuristic — but a true measured impact needs a live/typical-speed feed. That integration is **scaffolded and validation-ready** (`external_feeds.py`), pending an API key; the correlation test that would validate impact against measured slowdown is implemented and tested on synthetic data.
- **Seasonality blind spot.** Coverage is roughly **Nov–Apr only** — it misses the Jun–Sep monsoon when water-logging and tree-fall spike. Any generalisation claim should be fenced to the non-monsoon window.
- **`comment` field is empty (0%); `description` is noisy.** Only `description` carries signal, and ~36% of descriptions are too short/generic ("Starting problem", "no") to classify.
- **Closure model is modest.** ROC-AUC ~0.72 is a real signal (and improved by the road-class join) but not a strong classifier; at a 7.4% base rate recall at the 0.5 threshold is low, so it is used as a *likelihood/ranking* signal, not a hard yes/no.
- **Moran's I is optional.** It needs `esda`/`libpysal`, which aren't in the core install, so the "clusters are statistically significant" check silently skips in most environments. (The spatial-temporal holdout, which needs neither, runs everywhere.)
- **Diversion routes on the *mapped* network, around a fixed-radius closure.** The reroute is real (OSM graph + Dijkstra), but it models the blockage as an ~80 m zone and routes on the cached major-road network, not turn-by-turn with live signals/one-ways. It's a planning-grade detour and added-distance estimate, not a navigation product.
- **"Real-time" is stream-*ready*, not a live feed.** The `POST /event` API + SQLite store run the full predict→recommend→route→log path and accumulate live history, but no external incident stream is wired in — it's the streaming entry point, demonstrated with synthetic/manual events, not a production ingest.
- **Resource quantities are illustrative, and can't be validated from this data.** The personnel numbers (2/3/5 + bonuses) are operational rules of thumb, not learned — and they're *unvalidatable* here because `assigned_to_police_id` is ~98% null, so there's no record of what was actually deployed to score against. We surface them as recommendations, not empirical optima.
- **Hardcoded Bengaluru geography.** Zone/station mappings and coordinate bounds in `utils.py` / `config.yaml` are Bengaluru-specific, so the system isn't portable to another city without re-supplying those tables. Fine for this brief; a one-time config swap otherwise.
- **No roles/auth.** The dashboard and API are open — fine for a demo; an operator/admin split (commander-at-desk vs officer-on-phone) and endpoint auth are a deployment next step.
- **The roster optimizer rests on two assumed inputs.** The dataset has no officer roster or station capacity, so the supply side is illustrative (configurable). And 5 months of incidents aren't naturally simultaneous, so concurrency is *constructed* — the events that started within one clock-hour. The allocation maths is real; the scenario it runs on is a scaffold for demonstration, not a live operational feed. Station locations are themselves derived (centroid of the events each station serves), since the data has no station coordinates.

---

## Roadmap

- **Wire a live/typical-speed feed** (TomTom / HERE / Google Roads) into the `external_feeds.py` scaffold — the one remaining external join, and the only path to a *measured* (not heuristic) impact score. The validation correlation is already implemented and tested.
- **Populate the event calendar** (`external_feeds.py`) with stadium fixtures / festival & rally permits so Layer B does true advance forecasting rather than case-retrieval.
- Schedule `learning_loop.py` (cron) and auto-ingest newly-resolved logged events into the feature set — the loop is now closed in code (retrain + `metrics_history.csv` drift tracking); what remains is automating the cadence
- A dedicated role-separated officer view (commander-at-desk vs officer-on-phone); the UI is already responsive/mobile-usable and officer push alerts ship over Telegram — see Design Decisions
- Stronger multilingual text models (embeddings) over `description`; extend coverage across the monsoon season

*Done since the last review: the OSM road-class join, predictive hotspot validation, real diversion routing, road-aware impact, and a model-performance regression test — see Design Decisions.*
