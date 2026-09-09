"""Public capability flags — booleans only, never authorization.

The frontend uses these to decide WHICH controls to render (the
「開發者」 button, the 「在本機執行」 button, the simulation banner
plumbing). Every developer endpoint still verifies the X-Developer-Token
header server-side; hiding a button is convenience, the 403 is the
boundary.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from ..helper import auto_fix, devfix_queue
from ._auth import dev_token_configured

router = APIRouter()

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in _TRUTHY


def capabilities_payload() -> dict[str, bool]:
    local_runs, _reason = devfix_queue.local_runs_status()
    token = dev_token_configured()
    evaluate_switch = (os.getenv("HELPER_DASHBOARD_EVALUATE") or "1").strip().lower()
    return {
        "dev_token_configured": token,
        "devfix_local_runs": bool(local_runs),
        "auto_fix_enabled": bool(auto_fix.is_enabled()),
        "simulation_available": _truthy("HELPER_DASHBOARD_ALLOW_SIMULATION"),
        "evaluate_enabled": bool(token and evaluate_switch not in _FALSY),
    }


@router.get("")
@router.get("/")
def read_capabilities() -> dict[str, bool]:
    return capabilities_payload()
