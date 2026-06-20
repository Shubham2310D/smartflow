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

SmartFlow is built on 8,057 real Bengaluru traffic incidents (Astram dataset, cleaned from 8,173 raw rows):

| Module | What it does |
|---|---|
| **Data Pipeline** | Cleans raw Astram CSV — null handling, UTC datetimes, event-aware cause vocabulary (procession / VIP / protest kept as first-class causes) |
| **ML Models** | **Road-closure predictor** (calibrated, ROC-AUC 0.70 on a *real observed* target) + severity triage classifier (75.5% chronological holdout vs 63% baseline) |
| **Free-text Mining** | Bilingual (English + Kannada) keyword pass over `description` → event semantic type (sports event, utility work, VIP movement, …) |
| **Hotspot Engine** | DBSCAN spatial clustering (219 clusters, config-driven 200 m radius) + KDE density heatmap |
| **Resource Recommender** | Config-driven scoring → personnel, barricade, diversion, dispatch station + empirical clearance range |
| **Case-Based Forecast** | For planned events (procession / VIP / protest / public event), retrieves similar past events and what they actually required |
| **Feedback Loop** | Logs each decision and tracks predicted-vs-actual — the basis for periodic retraining |
| **Real-time API** | `POST /event` (FastAPI) runs the same predict → recommend → log path for streaming use |

---

## Key Metrics

| Metric | Value |
|---|---|
| Events processed | 8,057 (from 8,173 raw) |
| **Road-closure model ROC-AUC** (real observed target, calibrated) | **0.70** vs 7.4% base rate |
| Severity triage accuracy (chronological holdout) | 75.5% — vs 63% majority baseline |
| Severity triage accuracy (random 5-fold CV) | 86.0% (optimistic — ignores time order) |
| Clearance predictor vs median baseline | ~106 min — **no lift over the median** (reported honestly) |
| Median incident clearance | 57 min |
| Hotspot clusters found | 219 |
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
2. **Hotspot Map** — Folium heatmap + DBSCAN cluster circles with tooltips
3. **Predict Event** — Severity triage + **calibrated road-closure likelihood** + typical clearance range + SHAP + live text mining
4. **Resource Plan** — Deployment recommendation, **case-based analog panel** for planned events, decision logging
5. **Analytics** — 7 Plotly charts (incl. free-text event type)
6. **Feedback Loop** — Predicted-vs-actual backtest (with median baseline) + live operator decision log

---

## Project Structure

```
smartflow/
├── src/
│   ├── data_pipeline.py       # Load & clean raw CSV
│   ├── feature_engineering.py # Features + bilingual text mining
│   ├── model_training.py      # Closure predictor, severity triage, clearance stats
│   ├── hotspot_engine.py      # DBSCAN + KDE + GeoJSON (config-driven)
│   ├── resource_recommender.py# Config-driven deployment logic
│   ├── event_analog.py        # Case-based recommender for planned events
│   ├── outcomes_log.py        # Decision/outcome log (learning loop)
│   └── utils.py               # Zone/station mappings, constants
├── dashboard/
│   ├── app.py                 # Home page
│   └── pages/
│       ├── 1_Hotspot_Map.py
│       ├── 2_Predict_Event.py
│       ├── 3_Resource_Plan.py
│       ├── 4_Analytics.py
│       └── 5_Feedback_Loop.py
├── api/
│   └── main.py                # FastAPI real-time endpoint
├── tests/
│   └── test_leakage.py        # Asserts history features are backward-looking
├── data/
│   ├── raw/                   # Place events.csv here
│   └── processed/             # Auto-generated outputs
├── models/                    # Trained .pkl files
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
| Spatial analysis | DBSCAN (sklearn, haversine), KDE (scipy) |
| Explainability | SHAP (TreeExplainer, waterfall chart) |
| Calibration | scikit-learn (isotonic) for the closure model |
| Dashboard / API | Streamlit, Plotly, Folium / FastAPI |
| Config | PyYAML (wired into the code, not decorative) |

---

## Design Decisions

- **The real model predicts a real target: road closure.** `requires_road_closure` is *observed* (not a derived label) and is exactly what drives barricading/diversion — so it is genuinely learnable and operationally meaningful. We predict it with a class-weighted, **isotonic-calibrated** XGBoost on a chronological split. At a 7.4% base rate, accuracy is meaningless, so we report **ROC-AUC 0.70 / PR-AUC** and present the calibrated probability against the base rate. This is the headline model, replacing the synthetic-label classifier as the thing to trust.
- **Severity is a triage classifier, not a congestion-impact predictor** — its label is rules-derived (priority + closure + cause) and real-world impact (queue length, delay) is never measured in this data. So it is framed as *triage*. The honest result still stands: **75.5% on a chronological holdout vs a 63% baseline** from spatial-temporal context alone.
- **Contextual-only severity features (no leakage, no text)** — cause/closure are excluded because the label is derived from them; the text-derived `event_semantic_type` is *also* excluded from the classifier because it is a cause-proxy that would re-introduce the same circularity. It is used only by the duration model, where cause-like features are legitimate.
- **Chronological validation, not random** — operational forecasting must train on the past and predict the future. We split by time (earlier 80% / later 20%) instead of a random split. This is why the headline accuracy (75.5%) is lower than the conventional random CV (86%) — and more trustworthy.
- **Leakage-free junction history** — `junction_repeat_count` counts only events that occurred *before* each event at the same junction (an expanding count), zeroed for the catch-all "unknown" junction so it can't act as a disguised time index.
- **Event-aware cause vocabulary** — `procession`, `vip_movement`, `protest`, `public_event` are kept as first-class causes rather than collapsed into "other", because planned/unplanned gatherings are exactly the event types this problem targets.
- **Free-text mining (English + Kannada)** — the `description` field (83% populated, bilingual) is mined with a keyword pass into an `event_semantic_type` (sports event, utility work, VIP movement, …). It powers the auto-categorisation chart in Analytics and the live extraction on the Predict page. **We measured its effect honestly: it does *not* improve clearance-time MAE** (see below) — but it recovers event semantics the structured cause column misses.
- **The clearance regressor is reported against its baseline** — on the chronological holdout the XGBoost MAE (~106 min) is **statistically indistinguishable from "always predict the median" (~106 min)**. Clearance time here is dominated by unobserved operational factors (crew dispatch, on-scene complexity), so we present the estimate as a rough prior, show the baseline beside it on the Feedback Loop page, and have the Resource Plan fall back to cause-based medians. We'd rather show this than dress up a constant as a model.
- **Planned events get a case-based forecast, not a model** — processions, VIP movement, protests and public events have a known type and place ahead of time. The Resource Plan retrieves the most similar past events of that type and shows what they actually required (median clearance, closure rate). This directly answers the "recommend manpower for an upcoming event" half of the brief without inventing a model the data can't support.
- **Config is the single source of truth** — `config.yaml` is read by the hotspot engine (DBSCAN eps, min samples, KDE bandwidth, cluster radius) and the resource recommender (personnel, bonuses, barricade causes, closure threshold). It is no longer decorative; change the file and behaviour changes.
- **Honest cluster footprints** — clusters render as fixed-radius circles around the centroid, not convex hulls. Hulls over road-aligned incidents produce giant triangles that overstate the affected area.
- **DBSCAN over k-means** — no need to pre-specify cluster count; naturally handles noise; 200 m haversine radius tuned to Bengaluru block size.
- **Median over mean for clearance** — 9.7% of tickets were never properly closed, inflating the mean to 552 min. Median (57 min) reflects real operational close-time.

---

## Honest Limitations

We'd rather state these than have a reviewer find them:

- **Dataset is incident-heavy.** ~92% of records are incidents (vehicle breakdowns, potholes, water-logging). True event/gathering rows (procession, VIP, protest, public event) are ~130 events. SmartFlow surfaces them as first-class, but the data can't support a deep event-specific forecaster yet.
- **No measured congestion outcome.** The dataset records no queue length, speed drop, or delay. "Severity" is a rules-derived label, not an observed impact — so the classifier is framed as *triage*, and the only genuine outcome we model is **clearance time** (`duration_minutes`).
- **Clearance is mostly clerical, and unpredictable from these features.** Of ~2,760 usable durations, only **69** come from a real `resolved_datetime`; ~2,460 come from `closed_datetime`, an administrative ticket-close that is often batched. So the target is largely paperwork timing — which is why no regressor beats the median, and why we present clearance as an empirical range labelled "close-time", not "time to clear".
- **No measured congestion impact.** The data has no queue length, speed drop, or delay — so true "event impact forecasting" is structurally impossible from this CSV alone. It would need an external signal (OSM road class / lane count, or a typical-speed feed); that join is future work, not faked here.
- **Seasonality blind spot.** Coverage is roughly **Nov–Apr only** — it misses the Jun–Sep monsoon when water-logging and tree-fall spike. Any generalisation claim should be fenced to the non-monsoon window.
- **`comment` field is empty (0%); `description` is noisy.** Only `description` carries signal, and ~36% of descriptions are too short/generic ("Starting problem", "no") to classify.
- **Closure model is modest.** ROC-AUC 0.70 is a real signal but not a strong classifier; at a 7.4% base rate recall at the 0.5 threshold is low, so it is used as a *likelihood/ranking* signal, not a hard yes/no.
- **Moran's I is optional.** It needs `esda`/`libpysal`, which aren't in the core install, so the "clusters are statistically significant" check silently skips in most environments.

---

## Roadmap

- External impact signal (OSM lane count / typical-speed feed) so a closure on an 8-lane arterial scores differently from a side street — the missing piece for true impact forecasting
- Consume `decisions_log.csv`: nightly re-fit of the resource-rule constants and model retraining (close the learning loop, not just log it)
- Stronger multilingual text models (embeddings) over `description`
- Extend coverage across the monsoon season
