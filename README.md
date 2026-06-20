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

| Analytics |
|---|
| ![Analytics](snapshots/analytics.png) |

---

## Problem

Bengaluru processes thousands of road incidents every month — accidents, floods, vehicle breakdowns, construction blocks. For Flipkart's last-mile logistics, each undetected hotspot or delayed response costs delivery SLA time and fleet efficiency. Operators currently lack a unified system to predict incident severity, identify recurring hotspots, and deploy resources before queues form.

---

## Solution

SmartFlow is a five-module intelligence platform built on 8,057 real Bengaluru traffic incidents (Astram dataset, cleaned from 8,173 raw rows):

| Module | What it does |
|---|---|
| **Data Pipeline** | Cleans raw Astram CSV — null handling, UTC datetimes, event-aware cause vocabulary (procession / VIP / protest kept as first-class causes) |
| **ML Models** | XGBoost severity classifier (75.5% on a chronological holdout vs 63% baseline) + clearance-time predictor |
| **Hotspot Engine** | DBSCAN spatial clustering (219 clusters, 200 m radius) + KDE density heatmap |
| **Resource Recommender** | Rule-based scoring → personnel count, barricade, diversion, dispatch station, clearance estimate |
| **Feedback Loop** | Logs each decision and tracks predicted-vs-actual clearance — the basis for periodic retraining |

---

## Key Metrics

| Metric | Value |
|---|---|
| Events processed | 8,057 (from 8,173 raw) |
| Hotspot clusters found | 219 |
| Severity accuracy (chronological holdout) | 75.5% — vs 63% majority baseline |
| Severity accuracy (random 5-fold CV) | 86.0% (optimistic — ignores time order) |
| Clearance predictor MAE (holdout) | 105.7 min |
| Median incident clearance | 57 min |
| Top hotspot | Sankey Road — 764 events |

> **On the two accuracy numbers:** the **75.5%** figure is the honest one — the
> model is trained on earlier events and tested on later ones, so it never sees
> the future. The 86% random-split number is reported only because it is the
> conventional metric; it is optimistic for an operational forecasting system.

---

## Dashboard Pages

1. **Home** — 5 KPI metrics, severity distribution, top corridors
2. **Hotspot Map** — Folium heatmap + DBSCAN cluster polygons with tooltips
3. **Predict Event** — Input form → severity + clearance forecast + SHAP waterfall
4. **Resource Plan** — Deployment recommendation with audit breakdown + decision logging
5. **Analytics** — 7 Plotly charts (cause, hourly, daily, junctions, corridor, clearance, + free-text event type)
6. **Feedback Loop** — Predicted-vs-actual backtest (with median baseline) + live operator decision log

---

## Project Structure

```
smartflow/
├── src/
│   ├── data_pipeline.py       # Load & clean raw CSV
│   ├── feature_engineering.py # Derive 12 features
│   ├── model_training.py      # Train XGBoost models
│   ├── hotspot_engine.py      # DBSCAN + KDE + GeoJSON
│   ├── resource_recommender.py# Rule-based deployment logic
│   └── utils.py               # Zone/station mappings, constants
├── dashboard/
│   ├── app.py                 # Home page
│   └── pages/
│       ├── 1_Hotspot_Map.py
│       ├── 2_Predict_Event.py
│       ├── 3_Resource_Plan.py
│       └── 4_Analytics.py
├── data/
│   ├── raw/                   # Place events.csv here
│   └── processed/             # Auto-generated outputs
├── models/                    # Trained .pkl files
├── config.yaml
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
python src/feature_engineering.py
```

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

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy, scipy |
| Machine learning | XGBoost, scikit-learn |
| Spatial analysis | DBSCAN (sklearn), KDE (scipy), ConvexHull (scipy) |
| Explainability | SHAP (TreeExplainer, waterfall chart) |
| Dashboard | Streamlit, Plotly, Folium |
| Config | PyYAML |

---

## Design Decisions

- **Severity is a triage classifier, not a congestion-impact predictor** — the label is rules-derived (priority + closure + cause), and real-world impact (queue length, delay) is never measured in this data. So the model is presented as a contextual *triage* score. The honest result still stands: **75.5% on a chronological holdout vs a 63% majority baseline — a real +12 points** from spatial-temporal context alone.
- **Contextual-only severity features (no leakage, no text)** — cause/closure are excluded because the label is derived from them; the text-derived `event_semantic_type` is *also* excluded from the classifier because it is a cause-proxy that would re-introduce the same circularity. It is used only by the duration model, where cause-like features are legitimate.
- **Chronological validation, not random** — operational forecasting must train on the past and predict the future. We split by time (earlier 80% / later 20%) instead of a random split. This is why the headline accuracy (75.5%) is lower than the conventional random CV (86%) — and more trustworthy.
- **Leakage-free junction history** — `junction_repeat_count` counts only events that occurred *before* each event at the same junction (an expanding count), zeroed for the catch-all "unknown" junction so it can't act as a disguised time index.
- **Event-aware cause vocabulary** — `procession`, `vip_movement`, `protest`, `public_event` are kept as first-class causes rather than collapsed into "other", because planned/unplanned gatherings are exactly the event types this problem targets.
- **Free-text mining (English + Kannada)** — the `description` field (83% populated, bilingual) is mined with a keyword pass into an `event_semantic_type` (sports event, utility work, VIP movement, …). It powers the auto-categorisation chart in Analytics and the live extraction on the Predict page. **We measured its effect honestly: it does *not* improve clearance-time MAE** (see below) — but it recovers event semantics the structured cause column misses.
- **The clearance regressor is reported against its baseline** — on the chronological holdout the XGBoost MAE (~106 min) is **statistically indistinguishable from "always predict the median" (~106 min)**. Clearance time here is dominated by unobserved operational factors (crew dispatch, on-scene complexity), so we present the estimate as a rough prior, show the baseline beside it on the Feedback Loop page, and have the Resource Plan fall back to cause-based medians. We'd rather show this than dress up a constant as a model.
- **DBSCAN over k-means** — no need to pre-specify cluster count; naturally handles noise; 200 m radius tuned to Bengaluru block size.
- **Median over mean for clearance** — 9.7% of tickets were never properly closed, inflating the mean to 552 min. Median (57 min) reflects real operational clearance.

---

## Honest Limitations

We'd rather state these than have a reviewer find them:

- **Dataset is incident-heavy.** ~92% of records are incidents (vehicle breakdowns, potholes, water-logging). True event/gathering rows (procession, VIP, protest, public event) are ~130 events. SmartFlow surfaces them as first-class, but the data can't support a deep event-specific forecaster yet.
- **No measured congestion outcome.** The dataset records no queue length, speed drop, or delay. "Severity" is a rules-derived label, not an observed impact — so the classifier is framed as *triage*, and the only genuine outcome we model is **clearance time** (`duration_minutes`).
- **Clearance is effectively unpredictable from the available features.** Only ~38% of events have a usable resolution time, the target is dominated by unobserved operational factors, and the regressor (even with the text feature) does not beat the median baseline. We report this openly rather than hiding the MAE next to no baseline.
- **`comment` field is empty (0%); `description` is noisy.** Only `description` carries signal, and ~36% of descriptions are too short/generic ("Starting problem", "no") to classify — so the text feature matches ~36% of events.
- **Temporal coverage is sporadic.** Events cluster on a limited set of reporting days, so rolling-window features are built on an irregular sample.
- **Batch, not real-time (yet).** The dashboard reads a static cleaned CSV. The Feedback Loop + decision log are the scaffolding for streaming ingestion and retraining, which is the next step.

---

## Roadmap

- Real-time ingestion endpoint (FastAPI `/event`) feeding the same prediction + logging path
- Stronger text models (multilingual embeddings) and richer operational features (crew availability, on-scene reports) to give the clearance estimate something the median doesn't already have
- Calibrated probabilities (Platt/isotonic) so the triage score becomes a true probability
- Scheduled retraining keyed to the accumulated decision log
