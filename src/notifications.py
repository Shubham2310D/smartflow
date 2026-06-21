"""
notifications.py — Optional Telegram push alerts for field officers.

The system already *decides* when an alert matters — `recommend()` flags a
barricade, Live Ops ranks priority incidents — but until now that decision only
lived on a dashboard someone had to be watching. This is the missing delivery
channel: when a high-priority incident comes through, an officer's phone buzzes.

Design (same honesty discipline as external_feeds.py):
  * OFF by default. Credentials come from env vars (preferred) or config.yaml;
    with none set, every call is a silent no-op returning {"sent": False,
    "reason": ...} — it NEVER raises and never blocks the request path.
  * Secrets stay out of git: read from TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
    (e.g. Render env vars), or a gitignored config entry — never committed.
  * Telegram chosen over SMS/web-push: free, no paid account or number
    verification, real phone push, and a 5-minute bot setup. See README.

Setup (one time): create a bot via @BotFather → token; add it to an officers'
group → chat id (from getUpdates). Set the two env vars. Done.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


def _cfg(project_root: Path) -> dict:
    try:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load((project_root / "config.yaml").read_text()) or {}
        return cfg.get("notifications", {}) or {}
    except Exception:
        return {}


def _credentials(project_root: Path) -> tuple[str, str]:
    """Token + chat id, env first (so a deploy never needs the secret in the repo)."""
    cfg = _cfg(project_root)
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token") or ""
    chat = os.environ.get("TELEGRAM_CHAT_ID") or str(cfg.get("telegram_chat_id") or "")
    return token.strip(), chat.strip()


def invite_link(project_root: Path | None = None) -> str:
    """Public group invite link (for the demo 'join & watch alerts' link). May be ''."""
    root = project_root or Path(__file__).resolve().parents[1]
    return (os.environ.get("TELEGRAM_INVITE_LINK")
            or _cfg(root).get("telegram_invite_link") or "").strip()


def notify_status(project_root: Path | None = None) -> dict:
    """Report whether Telegram alerts are configured, without sending anything."""
    root = project_root or Path(__file__).resolve().parents[1]
    token, chat = _credentials(root)
    if not token or not chat:
        return {"available": False,
                "reason": "set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
                          "(env or config.yaml → notifications) to enable alerts"}
    return {"available": True}


def send_message(text: str, project_root: Path | None = None,
                 timeout: int = 10) -> dict:
    """
    Send a Telegram message. Returns {"sent": bool, ...}; never raises so the
    caller (API request / dashboard button) is never broken by a delivery hiccup.
    """
    root = project_root or Path(__file__).resolve().parents[1]
    status = notify_status(root)
    if not status["available"]:
        return {"sent": False, **status}

    token, chat = _credentials(root)
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(_API.format(token=token), data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
        if body.get("ok"):
            return {"sent": True, "message_id": body["result"].get("message_id")}
        return {"sent": False, "reason": f"telegram error: {body.get('description')}"}
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return {"sent": False, "reason": f"request failed: {exc}"}


# ---------------------------------------------------------------------------
# Incident → alert message
# ---------------------------------------------------------------------------

_SEV_ICON = {"High": "🔺", "Medium": "🟠", "Low": "🟩"}


def format_incident_alert(incident: dict) -> str:
    """Build a compact HTML alert message from an incident/recommendation dict."""
    sev = incident.get("severity", "—")
    cause = incident.get("event_cause", incident.get("cause", "incident"))
    zone = incident.get("zone", "—")
    lines = [
        f"🚨 <b>SmartFlow alert</b> — {_SEV_ICON.get(sev, '⚠️')} <b>{sev}</b>",
        f"<b>Cause:</b> {str(cause).replace('_', ' ').title()}",
        f"<b>Zone:</b> {zone}",
    ]
    cp = incident.get("closure_probability", incident.get("closure_prob"))
    if cp is not None:
        lines.append(f"<b>Road-closure likelihood:</b> {float(cp) * 100:.0f}%")
    if incident.get("personnel") is not None:
        frm = f" from {incident['dispatch_from']}" if incident.get("dispatch_from") else ""
        lines.append(f"<b>Deploy:</b> {incident['personnel']} officers{frm}")
    if incident.get("barricade"):
        lines.append("<b>⛔ Barricade recommended</b>")
    div = incident.get("diversion_summary")
    if div:
        lines.append(f"<b>Diversion:</b> {div}")
    loc = incident.get("location")
    if loc:
        lat, lon = loc
        lines.append(f'📍 <a href="https://maps.google.com/?q={lat},{lon}">map</a>')
    return "\n".join(lines)


def notify_incident(incident: dict, project_root: Path | None = None) -> dict:
    """Format + send an incident alert (no-op when unconfigured)."""
    return send_message(format_incident_alert(incident), project_root=project_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    print("status:", notify_status())
    demo = {"severity": "High", "event_cause": "accident", "zone": "East Zone",
            "closure_probability": 0.62, "personnel": 5, "dispatch_from": "Mahadevapura",
            "barricade": True, "diversion_summary": "2.5 km reroute (+2327 m)",
            "location": (12.9568, 77.7011)}
    print("send:", notify_incident(demo))
