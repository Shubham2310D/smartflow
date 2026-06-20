"""
Page 2 — Predict Event
Input form → severity classification + duration regression + SHAP waterfall.
Prediction result is stored in st.session_state for Page 3.
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

from feature_engineering import (
    CAUSE_SEVERITY_WEIGHT,
    _SEMANTIC_MAP,
    _VEH_TYPE_MAP,
    _VEH_TYPE_ORDER,
    extract_semantic_type,
)
from model_training import FEATURES, SEVERITY_COLORS, SEVERITY_INVERSE_MAP
from utils import ALL_CAUSES, ALL_ZONES, CAUSE_DISPLAY, get_nearest_station

st.set_page_config(page_title="Predict Event | SmartFlow", page_icon="🔮", layout="wide")

# ---------------------------------------------------------------------------
# Load models (cached)
# ---------------------------------------------------------------------------

_CLF_PATH = _ROOT / "models" / "severity_classifier.pkl"
_DUR_PATH = _ROOT / "models" / "duration_predictor.pkl"
_FEATS_PATH = _ROOT / "data" / "processed" / "features.csv"


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    if not _CLF_PATH.exists() or not _DUR_PATH.exists():
        return None, None
    clf_pkg = joblib.load(_CLF_PATH)
    dur_pkg = joblib.load(_DUR_PATH)
    return clf_pkg, dur_pkg


@st.cache_data(show_spinner=False)
def load_feature_stats():
    if not _FEATS_PATH.exists():
        return {}
    df = pd.read_csv(_FEATS_PATH)
    stats = {}
    for corridor, grp in df.groupby("corridor"):
        stats[corridor] = {
            "median_7d":  grp["corridor_7d_score"].median(),
            "median_jrc": grp["junction_repeat_count"].median(),
        }
    return stats


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Predict Event")
st.caption("Enter event details to get severity classification + clearance time forecast")

clf_pkg, dur_pkg = load_models()
if clf_pkg is None:
    st.error(
        "Models not found. Run `python src/model_training.py` from smartflow/ first."
    )
    st.stop()

clf_model = clf_pkg["model"]
dur_model = dur_pkg["model"]
feat_stats = load_feature_stats()

# Known corridors for the dropdown
all_corridors = sorted(feat_stats.keys()) if feat_stats else ["Non-corridor"]

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("predict_form"):
    st.subheader("Event Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        cause_display = st.selectbox(
            "Event Cause",
            options=[CAUSE_DISPLAY.get(c, c) for c in ALL_CAUSES],
        )
        cause = ALL_CAUSES[[CAUSE_DISPLAY.get(c, c) for c in ALL_CAUSES].index(cause_display)]

        road_closure = st.checkbox("Road Closure Required", value=False)

        veh_type_options = [v.replace("_", " ").title() for v in _VEH_TYPE_ORDER]
        veh_type_sel = st.selectbox("Vehicle Type", veh_type_options)
        veh_type = _VEH_TYPE_ORDER[veh_type_options.index(veh_type_sel)]

    with col2:
        hour = st.slider("Hour of Day (24h)", 0, 23, 8)
        day  = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)

        month = st.slider("Month", 1, 12, 3)

    with col3:
        corridor = st.selectbox("Corridor", all_corridors)
        zone = st.selectbox("Zone", ALL_ZONES)

    desc_text = st.text_area(
        "Event description (optional — English or Kannada)",
        placeholder="e.g. 'Cricket match at M Chinnaswamy Stadium' or 'BMTC bus broken down'",
        help="Free text is mined for an event type (sports event, utility work, "
             "VIP movement, …) that feeds the clearance-time estimate.",
    )

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Compute features + predict
# ---------------------------------------------------------------------------

if submitted:
    # Look up corridor/junction stats
    c_stats = feat_stats.get(corridor, {})
    corridor_7d   = int(c_stats.get("median_7d", 5))
    junction_rpt  = int(c_stats.get("median_jrc", 5))

    is_peak   = int(hour in range(8, 11) or hour in range(17, 21))
    is_weekend = int(day_idx >= 5)

    # Mine the free-text description for an event semantic type (bilingual).
    # Falls back to "other" when no description is given.
    semantic_type = extract_semantic_type(desc_text)
    semantic_encoded = _SEMANTIC_MAP.get(semantic_type, 0)

    # Full feature dict — each model picks its own columns from the pkl's feature list.
    # CLF features: contextual only (no cause/closure/text) to avoid label leakage.
    # REG features: includes cause/closure/semantic (duration is observed, not circular).
    all_feat_vals = {
        "cause_severity_weight":  CAUSE_SEVERITY_WEIGHT.get(cause, 1),
        "road_closure_binary":    int(road_closure),
        "event_semantic_encoded": semantic_encoded,
        "hour_of_day":            hour,
        "day_of_week":            day_idx,
        "month":                  month,
        "is_peak_hour":           is_peak,
        "is_weekend":             is_weekend,
        "junction_repeat_count":  junction_rpt,
        "corridor_7d_score":      corridor_7d,
        "veh_type_encoded":       _VEH_TYPE_MAP.get(veh_type, len(_VEH_TYPE_ORDER) - 1),
    }
    X_clf = pd.DataFrame([all_feat_vals])[clf_pkg["features"]]
    X_dur = pd.DataFrame([all_feat_vals])[dur_pkg["features"]]

    # --- Classification ---
    clf_probs     = clf_model.predict_proba(X_clf)[0]
    clf_class_idx = int(np.argmax(clf_probs))
    severity      = SEVERITY_INVERSE_MAP[clf_class_idx]
    confidence    = float(clf_probs[clf_class_idx])

    # --- Regression ---
    raw_pred      = float(dur_model.predict(X_dur)[0])
    if dur_pkg.get("log_transform"):
        import numpy as _np
        duration_pred = float(_np.expm1(raw_pred))
    else:
        duration_pred = raw_pred
    duration_pred = max(1.0, min(duration_pred, 1440.0))

    # Store in session_state for Page 3
    st.session_state["last_prediction"] = {
        "severity":         severity,
        "confidence":       confidence,
        "duration_minutes": duration_pred,
        "event_cause":      cause,
        "road_closure":     road_closure,
        "hour_of_day":      hour,
        "zone":             zone,
        "corridor":         corridor,
        "X_input":          X_clf.to_dict("records")[0],
    }

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------

    sev_color = SEVERITY_COLORS.get(severity, "#6c757d")

    st.divider()
    st.subheader("Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        f"<div style='background:{sev_color};padding:20px;border-radius:10px;text-align:center;"
        f"color:white;font-size:1.4em;font-weight:bold;'>"
        f"Severity: {severity}</div>",
        unsafe_allow_html=True,
    )
    r2.metric("Estimated Clearance", f"{duration_pred:.0f} min",
              delta=f"{duration_pred/60:.1f} hrs")
    r3.metric("Model Score", f"{confidence*100:.1f}%",
              help="Raw XGBoost class probability (uncalibrated) — a relative "
                   "confidence indicator, not a true probability.")

    if desc_text and desc_text.strip():
        if semantic_type != "other":
            st.caption(
                f"🔎 Detected event type from description: "
                f"**{semantic_type.replace('_', ' ').title()}**"
            )
        else:
            st.caption("🔎 No specific event type matched in the description.")

    # Probability bar
    st.divider()
    st.subheader("Class Probabilities")
    prob_fig = go.Figure(go.Bar(
        x=["Low", "Medium", "High"],
        y=[clf_probs[0], clf_probs[1], clf_probs[2]],
        marker_color=["#28a745", "#ffc107", "#dc3545"],
        text=[f"{p*100:.1f}%" for p in [clf_probs[0], clf_probs[1], clf_probs[2]]],
        textposition="auto",
    ))
    prob_fig.update_layout(
        yaxis_range=[0, 1], yaxis_tickformat=".0%",
        height=280, margin=dict(t=10, b=10),
        yaxis_title="Probability",
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # SHAP feature contributions
    st.divider()
    st.subheader("Feature Contributions (SHAP)")
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        @st.cache_resource(show_spinner=False)
        def get_clf_explainer(_m):
            return shap.TreeExplainer(_m)

        clf_features = clf_pkg["features"]
        explainer  = get_clf_explainer(clf_model)
        shap_vals  = explainer.shap_values(X_clf)

        if isinstance(shap_vals, list):
            sv = shap_vals[clf_class_idx][0]
            ev = explainer.expected_value[clf_class_idx]
        else:
            sv = shap_vals[0]
            ev = explainer.expected_value

        shap_exp = shap.Explanation(
            values         = sv,
            base_values    = float(ev),
            data           = X_clf.values[0],
            feature_names  = clf_features,
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        shap.plots.waterfall(shap_exp, show=False, max_display=10)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    except Exception as exc:
        # Fallback: simple horizontal bar chart of "feature importance × value"
        clf_features = clf_pkg["features"]
        feat_vals = X_clf.iloc[0].tolist()
        try:
            importance = clf_model.feature_importances_
        except Exception:
            importance = np.ones(len(clf_features))
        contrib = [float(imp * val) for imp, val in zip(importance, feat_vals)]
        fig_fb = go.Figure(go.Bar(
            x=contrib, y=clf_features, orientation="h",
            marker_color=["#dc3545" if v > 0 else "#28a745" for v in contrib],
        ))
        fig_fb.update_layout(
            title=f"Feature Importance × Value (SHAP unavailable: {exc})",
            height=350, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_fb, use_container_width=True)

    st.success(
        f"Prediction complete. Go to **Resource Plan** to see the deployment recommendation."
    )

elif "last_prediction" not in st.session_state:
    st.info("Fill in the event details above and click **Predict**.")
else:
    last = st.session_state["last_prediction"]
    st.info(
        f"Showing last prediction — Severity: **{last['severity']}**, "
        f"Duration: **{last['duration_minutes']:.0f} min**.  "
        "Submit the form again to update."
    )
