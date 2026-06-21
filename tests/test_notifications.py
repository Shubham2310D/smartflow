"""
test_notifications.py — Guard the officer-alert (Telegram) channel.

Asserts it stays OFF and silent when no credentials are configured (never raises,
never makes a network call), reports its status honestly, and formats an incident
alert with the operationally-important fields. A real send is NOT tested here —
it needs a live bot token — but the no-op contract and message formatting are.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import notifications as nt


@pytest.fixture
def unconfigured(tmp_path, monkeypatch):
    """A project root with no config + no env credentials."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    return tmp_path


def test_status_unconfigured(unconfigured):
    s = nt.notify_status(unconfigured)
    assert s["available"] is False and "reason" in s


def test_send_is_silent_noop_when_unconfigured(unconfigured):
    # Must not raise and must not attempt a network call.
    res = nt.send_message("hello", project_root=unconfigured)
    assert res["sent"] is False and "reason" in res


def test_notify_incident_noop_when_unconfigured(unconfigured):
    res = nt.notify_incident({"severity": "High", "event_cause": "accident"},
                             project_root=unconfigured)
    assert res["sent"] is False


def test_env_credentials_flip_status_available(unconfigured, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100999")
    assert nt.notify_status(unconfigured)["available"] is True


def test_alert_message_contains_key_fields():
    msg = nt.format_incident_alert({
        "severity": "High", "event_cause": "vip_movement", "zone": "Central Zone 1",
        "closure_probability": 0.62, "personnel": 5, "dispatch_from": "Cubbon Park",
        "barricade": True, "diversion_summary": "2.5 km reroute (+2327 m)",
        "location": (12.97, 77.59),
    })
    assert "High" in msg
    assert "Vip Movement" in msg          # cause prettified
    assert "Central Zone 1" in msg
    assert "62%" in msg                    # closure likelihood
    assert "Barricade" in msg
    assert "reroute" in msg                # diversion summary carried through
    assert "maps.google.com" in msg        # location link
