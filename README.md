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

SmartFlow is a four-module intelligence platform built on 8,057 real Bengaluru traffic incidents (Astram dataset, Jan 2023 – Mar 2024):

| Module | What it does |
|---|---|
| **Data Pipeline** | Cleans raw Astram CSV — null handling, UTC datetimes, standardised cause vocabulary |
| **ML Models** | XGBoost severity classifier (80% accuracy from contextual signals) + duration predictor (MAE 88.6 min) |
| **Hotspot Engine** | DBSCAN spatial clustering (219 clusters, 200 m radius) + KDE density heatmap |
| **Resource Recommender** | Rule-based scoring → personnel count, barricade, diversion, dispatch station, clearance estimate |

---

## Key Metrics

| Metric | Value |
|---|---|
| Events processed | 8,057 |
| Hotspot clusters found | 219 |
| Severity classifier accuracy | 80% (86.7% 5-fold CV) |
| Duration predictor MAE | 88.6 min |
| Median incident clearance | 57 min |
| Top hotspot | Sankey Road — 764 events |

---

## Dashboard Pages

1. **Home** — 5 KPI metrics, severity distribution, top corridors
2. **Hotspot Map** — Folium heatmap + DBSCAN cluster polygons with tooltips
3. **Predict Event** — Input form → severity + clearance forecast + SHAP waterfall
4. **Resource Plan** — Deployment recommendation with audit breakdown
5. **Analytics** — 6 Plotly charts (cause, hourly, daily, junctions, corridor, clearance)

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

- **Contextual-only severity classifier** — cause and road closure are excluded from ML features because severity labels are derived from them (circular). The model predicts severity from spatial-temporal context alone: junction repeat frequency, corridor pressure, time of day, vehicle type.
- **Log1p transform on duration** — duration is heavily right-skewed (median 57 min, some stale tickets >7 days). Log transform + 24h cap gives honest MAE of 88.6 min.
- **DBSCAN over k-means** — no need to pre-specify cluster count; naturally handles noise points; 200 m radius tuned to Bengaluru block size.
- **Median over mean for clearance** — 9.7% of tickets were never properly closed, inflating the mean to 552 min. Median (57 min) reflects real operational clearance.
