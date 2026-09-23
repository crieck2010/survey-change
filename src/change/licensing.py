"""License-key and update-check hooks.

Stubbed integration points, per suite convention: the engine never phones
home on its own. Host applications call :func:`check_license` before
running paid workflows and :func:`check_for_updates` on a user-triggered
"check for updates" action. Both are safe no-ops until a vendor wires in
real backends.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, Optional

PRODUCT = "survey-change"
CURRENT_VERSION = "0.1.0"
# Vendors: point this at your release feed (e.g. GitHub releases API).
UPDATE_URL = "https://api.github.com/repos/crieck2010/survey-change/releases/latest"


def check_license(key: Optional[str] = None) -> Dict[str, object]:
    """Validate a license key.

    Returns ``{"valid": bool, "tier": str, "message": str}``. The default
    implementation accepts any key (including None) as a valid
    community-tier license; vendors replace this with a real check
    against their merchant of record (Gumroad / Lemon Squeezy).
    """
    key = key or os.environ.get("SURVEY_CHANGE_LICENSE", "")
    if not key:
        return {"valid": True, "tier": "community",
                "message": "No license key: running under the community tier."}
    return {"valid": True, "tier": "licensed",
            "message": "License key accepted (stub validation)."}


def check_for_updates(timeout: float = 10.0) -> Dict[str, object]:
    """Query the release feed for a newer version.

    Never raises: network failures return ``{"available": False, ...}``
    with an ``error`` message. Call only from an explicit user action.
    """
    try:
        req = urllib.request.Request(
            UPDATE_URL, headers={"User-Agent": f"{PRODUCT}/{CURRENT_VERSION}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest = str(data.get("tag_name", "")).lstrip("v")
        return {
            "available": _newer(latest, CURRENT_VERSION),
            "current": CURRENT_VERSION,
            "latest": latest or None,
            "url": data.get("html_url"),
        }
    except Exception as exc:  # noqa: BLE001 - network failures are data
        return {"available": False, "current": CURRENT_VERSION,
                "latest": None, "error": str(exc)}


def _newer(latest: str, current: str) -> bool:
    def _parts(v: str):
        return [int(p) for p in v.split(".") if p.isdigit()]
    return _parts(latest) > _parts(current)
