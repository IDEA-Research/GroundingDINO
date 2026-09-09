"""Shared authorization gate for developer-only API routes."""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status


def dev_token_configured() -> bool:
    """Return whether the server has a non-empty developer token."""
    return bool(os.getenv("HELPER_DASHBOARD_DEV_TOKEN"))


def require_dev(
    x_developer_token: str | None = Header(
        default=None,
        alias="X-Developer-Token",
    ),
) -> None:
    """Reject access unless the supplied token matches the configured token."""
    configured_token = os.getenv("HELPER_DASHBOARD_DEV_TOKEN")
    if (
        not configured_token
        or not x_developer_token
        or not hmac.compare_digest(configured_token, x_developer_token)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Developer access forbidden",
        )
