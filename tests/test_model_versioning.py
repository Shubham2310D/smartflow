"""
test_model_versioning.py — Guard the model reproducibility check.

Asserts that the version-compatibility check flags a sklearn/xgboost/numpy skew
between a model's training environment and the runtime, matches cleanly when they
agree, and is explicit when a model predates version stamping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_training import check_lib_versions, lib_versions


def test_matching_versions_produce_no_warnings():
    payload = {"lib_versions": lib_versions()}
    assert check_lib_versions(payload) == []


def test_mismatch_is_flagged_for_compat_libs():
    bad = lib_versions() | {"xgboost": "0.0.1"}
    warns = check_lib_versions({"lib_versions": bad})
    assert len(warns) == 1 and "xgboost" in warns[0]


def test_noncompat_lib_drift_is_ignored():
    # pandas isn't pickle-compat-critical, so a pandas skew must NOT warn.
    drift = lib_versions() | {"pandas": "0.0.1"}
    assert check_lib_versions({"lib_versions": drift}) == []


def test_unstamped_model_is_called_out():
    warns = check_lib_versions({})
    assert warns and "no recorded training versions" in warns[0]
