"""
Page 5 — Feedback Loop (post-event learning)
Closes the loop the problem statement asks for:

  1. Model Backtest      — honest out-of-sample predicted-vs-actual clearance
                           (chronological holdout, never seen in training).
  2. Live Operator Log   — decisions logged from the Resource Plan page, with
                           predicted-vs-actual once real outcomes are recorded.
  3. Retraining hook     — how accumulated outcomes feed the next model.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from learning_loop import evaluate_decisions, load_history
from outcomes_log import load_decisions, summary

st.set_page_config(page_title="Feedback Loop | SmartFlow", page_icon="🔁", layout="wide")

_BACKTEST = _ROOT / "data" / "processed" / "duration_backtest.csv"
_DUR_PKL  = _ROOT / "models" / "duration_predictor.pkl"


@st.cache_resource(show_spinner=False)
def _load_dur_meta():
    if not _DUR_PKL.exists():
        return {}
    pkg = joblib.load(_DUR_PKL)
    return {k: pkg.get(k) for k in ("baseline_mae", "baseline_median", "lift_vs_baseline")}

st.title("Feedback Loop")
st.caption("Predicted-vs-actual tracking and the post-event learning cycle")

st.markdown(
    """
    A forecast is only as good as what it does in the field. This page compares
    **what SmartFlow predicted** against **what actually happened** — both on a
    held-out historical sample and on live decisions logged by operators.
    """
)

# ---------------------------------------------------------------------------
# Helper: predicted-vs-actual scatter with y=x reference
# ---------------------------------------------------------------------------

def _pva_scatter(actual, predicted, title: str):
    hi = float(max(actual.max(), predicted.max())) if len(actual) else 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode="markers",
        marker=dict(color="#0d6efd", size=6, opacity=0.5),
        name="events",
    ))
    fig.add_trace(go.Scatter(
        x=[0, hi], y=[0, hi], mode="lines",
        line=dict(color="#dc3545", dash="dash"),
        name="perfect prediction",
    ))
    fig.update_layout(
        title=title, height=400, margin=dict(t=40, b=10),
        xaxis_title="Actual clearance (min)",
        yaxis_title="Predicted clearance (min)",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Model backtest (honest holdout)
# ---------------------------------------------------------------------------

st.header("1 · Model Backtest (held-out history)")

if _BACKTEST.exists():
    bt = pd.read_csv(_BACKTEST)
    actual, pred = bt["actual_minutes"], bt["predicted_minutes"]
    err = (actual - pred).abs()
    mae  = err.mean()
    rmse = np.sqrt((err ** 2).mean())
    within30 = ((err / actual.clip(lower=1)) <= 0.30).mean() * 100

    meta = _load_dur_meta()
    baseline_mae = meta.get("baseline_mae")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Holdout events", f"{len(bt):,}")
    b2.metric("Model MAE", f"{mae:.0f} min")
    b3.metric(
        "Predict-median baseline",
        f"{baseline_mae:.0f} min" if baseline_mae else "—",
        delta=(f"{mae - baseline_mae:+.0f} min vs baseline" if baseline_mae else None),
        delta_color="inverse",
    )
    b4.metric("Within ±30%", f"{within30:.0f}%")

    st.plotly_chart(
        _pva_scatter(actual, pred, "Predicted vs Actual Clearance — chronological holdout"),
        use_container_width=True,
    )
    st.warning(
        "**Honest read:** the model's MAE essentially *equals* the naive "
        "\"always predict the median\" baseline — clearance time here is dominated "
        "by operational factors the data doesn't capture (crew dispatch, on-scene "
        "complexity). We report the estimate as a rough prior, not a precise "
        "forecast, and the Resource Plan falls back to cause-based medians. The "
        "**severity** model, by contrast, beats its baseline by a real ~12 points. "
        "This panel exists so that distinction is visible, not buried."
    )
else:
    st.info(
        "No backtest file yet. Run `python src/model_training.py` from smartflow/ "
        "to generate `duration_backtest.csv`."
    )

st.divider()

# ---------------------------------------------------------------------------
# 2. Live operator decisions
# ---------------------------------------------------------------------------

st.header("2 · Live Operator Decisions")

dec = load_decisions()
stats = summary()

if dec.empty:
    st.info(
        "No decisions logged yet. Go to **Resource Plan**, generate a "
        "recommendation, and click **Log this decision** — they appear here and "
        "accumulate into the retraining set."
    )
else:
    ev = evaluate_decisions()
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Decisions logged", f"{stats['total']:,}")
    d2.metric("With recorded outcome", f"{stats['with_outcome']:,}")
    d3.metric("Clearance MAE",
              f"{stats['mae']:.0f} min" if stats["mae"] is not None else "—")
    d4.metric("Within ±30%",
              f"{stats['within_30pct']:.0f}%" if stats.get("within_30pct") is not None else "—")
    d5.metric("Severity accuracy",
              f"{ev['severity_accuracy']:.0f}%" if ev.get("severity_accuracy") is not None else "—",
              help="Logged predicted severity vs the operator-recorded actual severity.")

    closed = dec.copy()
    closed["actual_clearance_min"] = pd.to_numeric(
        closed["actual_clearance_min"], errors="coerce")
    closed["predicted_clearance_min"] = pd.to_numeric(
        closed["predicted_clearance_min"], errors="coerce")
    closed = closed.dropna(subset=["actual_clearance_min", "predicted_clearance_min"])

    if len(closed) >= 1:
        st.plotly_chart(
            _pva_scatter(
                closed["actual_clearance_min"], closed["predicted_clearance_min"],
                "Predicted vs Actual — logged operator decisions",
            ),
            use_container_width=True,
        )

    st.subheader("Decision log")
    st.dataframe(
        dec.sort_values("logged_at", ascending=False),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Retraining hook
# ---------------------------------------------------------------------------

st.header("3 · Retrain & drift tracking")
st.markdown(
    """
    The loop is **closed in code**, not just described:
    1. **Predict** → 2. **Deploy** → 3. **Record** (above) →
    4. **Retrain** — `python src/learning_loop.py` re-fits the models and appends
       a metrics + recommendation-accuracy snapshot to `metrics_history.csv`.
       Run it on a schedule and the chart below shows drift across retrains.
    """
)

hist = load_history()
if hist.empty:
    st.info(
        "No retrain snapshots yet. Run `python src/learning_loop.py` from smartflow/ "
        "(add `--no-retrain` to only record current metrics) to populate "
        "`metrics_history.csv`."
    )
else:
    metric_cols = {
        "closure_roc_auc":   "Closure ROC-AUC",
        "severity_test_acc": "Severity accuracy",
        "rec_severity_accuracy": "Recommendation severity acc (logged, %)",
    }
    present = [c for c in metric_cols if c in hist.columns and hist[c].notna().any()]
    fig = go.Figure()
    for c in present:
        fig.add_trace(go.Scatter(
            x=hist["recorded_at"], y=hist[c], mode="lines+markers", name=metric_cols[c]))
    fig.update_layout(
        title="Model metrics across retrains (drift)", height=360,
        margin=dict(t=40, b=10), xaxis_title="Retrain snapshot",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(hist)} snapshot(s) recorded. Each `learning_loop.py` run adds one — "
        "stable lines mean no drift yet; a drop flags that real outcomes have moved "
        "away from the model and a retrain on fresh data is due."
    )
    with st.expander("Raw metrics history"):
        st.dataframe(hist.sort_values("recorded_at", ascending=False),
                     use_container_width=True, hide_index=True)
