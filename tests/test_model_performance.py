"""
test_model_performance.py — Quality floor for the shipped models.

The learning loop retrains models. Without a guard, a bad retrain (corrupted
features, a leakage fix that removes real signal, an upstream data change) could
silently ship a model that's worse than chance, and nothing would catch it. These
tests assert the committed model payloads clear an explicit floor — set safely
below the current measured numbers so normal run-to-run noise passes, but a real
regression fails CI.

They read the stamped metrics in each .pkl (chronological-holdout numbers written
at train time), so they're fast and check exactly what ships. If a model file is
absent (e.g. a fresh checkout before training), the test skips rather than fails.
"""

import sys
from pathlib import Path

import joblib
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "models"
sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str) -> dict:
    path = _MODELS / name
    if not path.exists():
        pytest.skip(f"{name} not present — run python src/model_training.py")
    return joblib.load(path)


def test_closure_model_beats_chance():
    """The headline closure model must stay meaningfully above random."""
    p = _load("closure_predictor.pkl")
    # Single-split ROC-AUC (current ≈0.715). Floor leaves headroom for noise.
    assert p["roc_auc"] >= 0.66, f"closure ROC-AUC regressed to {p['roc_auc']:.3f}"
    # PR-AUC must beat the positive base rate (what a random ranker scores).
    assert p["pr_auc"] > p["base_rate"], (
        f"closure PR-AUC {p['pr_auc']:.3f} not above base rate {p['base_rate']:.3f}"
    )


def test_closure_walk_forward_is_stable():
    """Walk-forward mean AUC (current ≈0.667) is the honest, variance-aware number."""
    wf = _load("closure_predictor.pkl").get("walk_forward", {})
    if not wf.get("n_folds"):
        pytest.skip("no walk-forward record in payload")
    assert wf["roc_auc_mean"] >= 0.60, (
        f"closure walk-forward AUC regressed to {wf['roc_auc_mean']:.3f}"
    )


def test_severity_beats_majority_baseline():
    """Severity triage must beat always-predicting-the-majority-class."""
    p = _load("severity_classifier.pkl")
    assert p["test_accuracy"] > p["baseline_accuracy"], (
        f"severity accuracy {p['test_accuracy']:.3f} no better than majority "
        f"baseline {p['baseline_accuracy']:.3f}"
    )
    assert p["test_macro_f1"] >= 0.55, (
        f"severity macro-F1 regressed to {p['test_macro_f1']:.3f}"
    )


def test_duration_beats_median_baseline():
    """The duration regressor must beat predicting the training median (MAE)."""
    p = _load("duration_predictor.pkl")
    assert p["mae"] <= p["baseline_mae"], (
        f"duration MAE {p['mae']:.1f} worse than median baseline {p['baseline_mae']:.1f}"
    )


def test_models_carry_road_context_features():
    """
    Guard the OSM road join: closure & severity must keep using road context
    (its removal silently reverted the measured lift), and inference code must be
    able to supply it. A missing feature here means a feature-list regression.
    """
    for name in ("closure_predictor.pkl", "severity_classifier.pkl"):
        feats = _load(name)["features"]
        assert "road_class_rank" in feats and "lane_count" in feats, (
            f"{name} lost road-context features: {feats}"
        )
