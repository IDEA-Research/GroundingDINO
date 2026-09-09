"""Developer-only simulation switch (`/api/dev/simulation`).

Server-side gated: every route depends on `require_dev` (X-Developer-Token
must match HELPER_DASHBOARD_DEV_TOKEN) AND switching a scenario on also
requires the env kill switch `HELPER_DASHBOARD_ALLOW_SIMULATION=1`.
State is in-process only — a restart is always "off".
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..services import simulation
from ._auth import require_dev


router = APIRouter(dependencies=[Depends(require_dev)])


class ScenarioRequest(BaseModel):
    scenario: Literal[
        "off", "normal", "nodata", "stale", "partial", "offline", "error"
    ]


@router.get("")
@router.get("/")
def get_simulation() -> dict:
    return simulation.state()


@router.post("")
@router.post("/")
def set_simulation(req: ScenarioRequest) -> dict:
    try:
        return simulation.set_scenario(req.scenario)
    except simulation.SimulationDisabled as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "模擬功能未啟用",
                "reason": "simulation_disabled",
                "hint": str(exc),
            },
        ) from exc
