"""Spec validator.

Pydantic already does most of the heavy lifting. This module:

1. Wraps parsing so callers get one exception type (`SpecValidationError`)
   with a list of readable error strings.
2. Adds the extra semantic checks that aren't natural in Pydantic
   (cross-widget layout, forbidden option keys when passed as raw
   dicts, etc.).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..specs import DashboardSpec, PatchSpec
from ..specs.widget_spec import FORBIDDEN_WIDGET_FIELDS, WidgetSpec


class SpecValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _pydantic_errors(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "invalid")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def _scan_for_forbidden(payload: Any, path: str = "") -> list[str]:
    """Walk a raw dict/list tree and flag forbidden keys.

    This runs *before* Pydantic so the error message is obvious even
    when Pydantic would otherwise only say "extra forbidden".
    """
    errors: list[str] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in FORBIDDEN_WIDGET_FIELDS:
                errors.append(
                    f"{path + '.' if path else ''}{k}: forbidden field"
                )
            errors.extend(_scan_for_forbidden(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            errors.extend(_scan_for_forbidden(item, f"{path}[{i}]"))
    return errors


class SpecValidator:
    def validate_dashboard(self, payload: dict) -> DashboardSpec:
        errors = _scan_for_forbidden(payload)
        if errors:
            raise SpecValidationError(errors)
        try:
            return DashboardSpec.model_validate(payload)
        except ValidationError as exc:
            raise SpecValidationError(_pydantic_errors(exc)) from exc

    def validate_patch(self, payload: dict) -> PatchSpec:
        errors = _scan_for_forbidden(payload)
        if errors:
            raise SpecValidationError(errors)
        try:
            return PatchSpec.model_validate(payload)
        except ValidationError as exc:
            raise SpecValidationError(_pydantic_errors(exc)) from exc

    def validate_widget(self, payload: dict) -> WidgetSpec:
        errors = _scan_for_forbidden(payload)
        if errors:
            raise SpecValidationError(errors)
        try:
            return WidgetSpec.model_validate(payload)
        except ValidationError as exc:
            raise SpecValidationError(_pydantic_errors(exc)) from exc
