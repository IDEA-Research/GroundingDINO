"""Normalize screen_analysis medical_values: key aliases, value cleaning, composites, corrections, primary choice."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .aliases import AliasRegistry, MetricSpec, get_registry, normalize_key_text
from .models import (
    FLAG_EMPTY,
    FLAG_INVALID_MEDICAL_VALUES,
    FLAG_UNMAPPED_CORRECTION,
    Q_ALIAS_CONFLICT,
    Q_AMBIGUOUS,
    Q_CORRECTED,
    Q_NON_NUMERIC,
    Q_OK,
    Q_OUT_OF_RANGE,
    Q_PARSED_COMPOSITE,
    Q_UNIT_NORMALIZED,
    Q_UNIT_STRIPPED,
    NormalizedValues,
)

__all__ = [
    "NormalizedValues", "normalize_medical_values", "choose_primary", "assign_primary",
    "primary_sort_key", "screen_group_key", "parse_number", "parse_ratio",
]

_NUM = r"[-+]?\d+(?:\.\d+)?"
_RATIO_RE = re.compile(rf"^\s*({_NUM})\s*:\s*({_NUM})\s*$")
_BP_STR_RE = re.compile(rf"^\s*({_NUM})\s*/\s*({_NUM})\s*(?:\(\s*({_NUM})\s*\))?\s*$")
_NULL_STRINGS = {"null", "none"}
# L3: a number followed by a unit, e.g. "12%", "45 mmHg", "5 cmH2O", "3.2 L/min"
_UNIT_STR_RE = re.compile(
    rf"^\s*({_NUM})\s*(%|mmhg|cmh2o|cm\s*h2o|mbar|bpm|/min|l/min|ml|ml/kg|s|sec)\s*$", re.IGNORECASE
)

# parse outcomes
_NULL = "null"
_NUMERIC = "numeric"
_NONNUM = "non_numeric"


def parse_number(value: Any) -> Tuple[str, Optional[float]]:
    """Classify a raw value as ('null'|'numeric'|'non_numeric', float|None); numeric strings become floats."""
    if value is None:
        return _NULL, None
    if isinstance(value, bool):
        return _NONNUM, None
    if isinstance(value, (int, float)):
        f = float(value)
        return (_NUMERIC, f) if math.isfinite(f) else (_NONNUM, None)
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in _NULL_STRINGS:
            return _NULL, None
        try:
            f = float(s)
        except ValueError:
            return _NONNUM, None
        return (_NUMERIC, f) if math.isfinite(f) else (_NONNUM, None)
    return _NONNUM, None


def parse_ratio(value: Any) -> Tuple[str, Optional[float]]:
    """Parse an I:E ratio such as '1:2.0' -> 0.5; plain numbers pass through."""
    if isinstance(value, str):
        m = _RATIO_RE.match(value)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if b == 0:
                return _NONNUM, None
            return _NUMERIC, a / b
    return parse_number(value)


def _parse_for(spec: MetricSpec, value: Any) -> Tuple[str, Optional[float], List[str]]:
    """Parse a value with the metric's parser; returns (status, float, extra q flags).

    v2 additions (REVIEW L3): a plain number for a ratio metric is flagged `ambiguous` (it may be
    only the E of 1:E); a number with a trailing unit is parsed and flagged `unit_stripped`.
    """
    extra: List[str] = []
    if spec.parser == "ratio":
        status, f = parse_ratio(value)
        is_ratio_text = isinstance(value, str) and _RATIO_RE.match(value) is not None
        if status == _NUMERIC and not is_ratio_text:
            extra.append(Q_AMBIGUOUS)
    else:
        status, f = parse_number(value)
    if status == _NONNUM and isinstance(value, str):
        m = _UNIT_STR_RE.match(value)
        if m:
            f2 = float(m.group(1))
            if math.isfinite(f2):
                status, f = _NUMERIC, f2
                extra.append(Q_UNIT_STRIPPED)
    return status, f, extra


def _finish_value(spec: MetricSpec, v: float) -> Tuple[float, List[str]]:
    """Apply unit normalization and range check; return (value, flags)."""
    flags: List[str] = []
    if spec.fraction_to_percent and 0 < v <= 1.0:
        v = round(v * 100.0, 6)
        flags.append(Q_UNIT_NORMALIZED)
    if spec.range is not None and not (spec.range[0] <= v <= spec.range[1]):
        flags.append(Q_OUT_OF_RANGE)
    return v, flags


def _set_value(out: NormalizedValues, spec: MetricSpec, v: float, extra: Sequence[str], raw: Any) -> None:
    """Write a numeric value into m with fresh q flags (and raw for string inputs)."""
    v, flags = _finish_value(spec, v)
    out.m[spec.name] = v
    out.q[spec.name] = []
    for f in list(extra) + flags:
        out.add_q(spec.name, f)
    if not out.q[spec.name]:
        out.q[spec.name] = [Q_OK]
    if isinstance(raw, str):
        out.raw[spec.name] = raw
    else:
        out.raw.pop(spec.name, None)


def _set_non_numeric(out: NormalizedValues, canon: str, extra: Sequence[str], raw: Any) -> None:
    """Record a non-numeric value: no m entry, q non_numeric, raw text kept."""
    out.m.pop(canon, None)
    out.q[canon] = []
    out.add_q(canon, Q_NON_NUMERIC)
    for f in extra:
        out.add_q(canon, f)
    out.raw[canon] = raw if isinstance(raw, str) else repr(raw)[:200]


def _split_composite(reg: AliasRegistry, name: str, value: Any) -> Optional[Dict[str, float]]:
    """Split a composite (blood_pressure) string/object into {canonical: float}; None if not splittable."""
    comp = reg.composites[name]
    parts: Dict[str, float] = {}
    if isinstance(value, str):
        m = _BP_STR_RE.match(value)
        if not m:
            return None
        canon_order = list(comp.parts)  # sys, dia, map order as declared
        for canon, grp in zip(canon_order, m.groups()):
            if grp is not None:
                parts[canon] = float(grp)
        return parts
    if isinstance(value, Mapping):
        lookup = {normalize_key_text(k): v for k, v in value.items() if isinstance(k, str)}
        for canon, subkeys in comp.parts.items():
            for sk in subkeys:
                if sk in lookup:
                    status, f = parse_number(lookup[sk])
                    if status == _NUMERIC and f is not None:
                        parts[canon] = f
                    break
        return parts or None
    return None


def normalize_medical_values(
    medical_values: Any,
    corrected_medical_values: Any,
    family: str,
    registry: Optional[AliasRegistry] = None,
) -> NormalizedValues:
    """Normalize one document's medical_values and corrections for a device family (DESIGN section 4)."""
    reg = registry or get_registry()
    out = NormalizedValues()
    source_empty = False

    if medical_values is None or medical_values == {} or medical_values == []:
        source_empty = True
        items: List[Tuple[str, Any]] = []
    elif isinstance(medical_values, Mapping):
        items = list(medical_values.items())
    else:
        source_empty = True
        out.add_flag(FLAG_INVALID_MEDICAL_VALUES)
        items = []

    # pass 1: group candidate values per canonical metric
    scalars: Dict[str, List[Tuple[str, Any]]] = {}
    composites: List[Tuple[str, str, Any]] = []
    for raw_key, value in items:
        out.raw_keys.append(str(raw_key))
        if not isinstance(raw_key, str) or reg.is_ignored(raw_key):
            continue
        canon = reg.normalize_key(family, raw_key)
        if canon is None:
            out.unmapped[raw_key] = value
        elif reg.is_composite(canon):
            composites.append((canon, raw_key, value))
        else:
            scalars.setdefault(canon, []).append((raw_key, value))

    # pass 2: resolve scalars (first numeric wins; conflicting numeric aliases are flagged)
    for canon, cands in scalars.items():
        spec = reg.metrics[canon]
        parsed = [(rk, v) + _parse_for(spec, v) for rk, v in cands]
        numeric = [(rk, v, f, ex) for rk, v, st, f, ex in parsed if st == _NUMERIC]
        if numeric:
            _, raw_v, f, ex = numeric[0]
            conflict = any(abs(f2 - f) > 1e-9 for _, _, f2, _ in numeric[1:])
            _set_value(out, spec, f, ([Q_ALIAS_CONFLICT] if conflict else []) + ex, raw_v)
            continue
        nonnum = [(rk, v) for rk, v, st, _, _ in parsed if st == _NONNUM]
        if nonnum:
            _set_non_numeric(out, canon, [], nonnum[0][1])

    # pass 3: composites only fill metrics not already given by an explicit key
    for name, raw_key, value in composites:
        if value is None:
            continue
        parts = _split_composite(reg, name, value)
        if parts is None:
            out.unmapped[raw_key] = value  # scalar or unparseable composite: do not guess
            continue
        for canon, f in parts.items():
            if canon in out.m:
                continue
            _set_value(out, reg.metrics[canon], f, [Q_PARSED_COMPOSITE], None)
        if isinstance(value, str):
            out.raw[name] = value

    # pass 4: corrections overlay (values are strings; 'null' removes the value)
    if isinstance(corrected_medical_values, Mapping):
        for raw_key, value in corrected_medical_values.items():
            if not isinstance(raw_key, str) or reg.is_ignored(raw_key):
                continue
            canon = reg.normalize_key(family, raw_key)
            if canon is None:
                out.unmapped[raw_key] = value
                out.add_flag(FLAG_UNMAPPED_CORRECTION)
                continue
            if reg.is_composite(canon):
                if parse_number(value)[0] == _NULL or (isinstance(value, str) and not value.strip()):
                    for pc in reg.composites[canon].parts:
                        out.m.pop(pc, None)
                        out.raw.pop(pc, None)
                        out.q[pc] = [Q_CORRECTED]
                    out.raw.pop(canon, None)
                    continue
                parts = _split_composite(reg, canon, value)
                if parts is None:
                    out.unmapped[raw_key] = value
                    out.add_flag(FLAG_UNMAPPED_CORRECTION)
                    continue
                for pc, f in parts.items():
                    _set_value(out, reg.metrics[pc], f, [Q_PARSED_COMPOSITE, Q_CORRECTED], None)
                if isinstance(value, str):
                    out.raw[canon] = value
                continue
            spec = reg.metrics[canon]
            if isinstance(value, str) and value.strip() == "":
                status, f, ex = _NULL, None, []
            else:
                status, f, ex = _parse_for(spec, value)
            if status == _NULL:
                out.m.pop(canon, None)
                out.raw.pop(canon, None)
                out.q[canon] = [Q_CORRECTED]
            elif status == _NUMERIC and f is not None:
                _set_value(out, spec, f, [Q_CORRECTED] + ex, value)
            else:
                _set_non_numeric(out, canon, [Q_CORRECTED], value)

    if source_empty and not out.m:
        out.add_flag(FLAG_EMPTY)
    return out


def _get(c: Any, name: str) -> Any:
    """Read a field from a mapping or an object attribute."""
    if isinstance(c, Mapping):
        return c.get(name)
    return getattr(c, name, None)


def _ts_key(dt: Any) -> float:
    """Sort key for analyzed_at: aware/naive datetimes compared as UTC, missing sorts first."""
    if not isinstance(dt, datetime):
        return float("-inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _id_key(v: Any) -> str:
    """Tie-break key for _id / src_id: ObjectId hex (or str) compares in creation order."""
    return str(v) if v is not None else ""


def primary_sort_key(c: Any) -> Tuple[bool, bool, float, str]:
    """Primary ranking: is_corrected > success > latest analyzed_at > larger _id (REVIEW H8 tie-break)."""
    oid = _get(c, "_id")
    if oid is None:
        oid = _get(c, "src_id")
    return (bool(_get(c, "is_corrected")), _get(c, "success") is True, _ts_key(_get(c, "analyzed_at")), _id_key(oid))


def choose_primary(candidates: Sequence[Any]) -> int:
    """Return the index of the primary result: is_corrected > success > latest analyzed_at > larger _id.

    Candidates without any _id/src_id keep the v1 behaviour (ties resolve to the first)."""
    if not candidates:
        raise ValueError("choose_primary() needs at least one candidate")
    best_i = 0
    best_key: Optional[Tuple[bool, bool, float, str]] = None
    for i, c in enumerate(candidates):
        key = primary_sort_key(c)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def screen_group_key(c: Any) -> Tuple[str, Any]:
    """Primary group: (frame_result_id, original_screen_number ?? screen_number)."""
    osn = _get(c, "original_screen_number")
    if osn is None:
        osn = _get(c, "screen_number")
    return (_id_key(_get(c, "frame_result_id")), osn)


def assign_primary(candidates: Sequence[Any]) -> List[bool]:
    """Return one bool per candidate: exactly one True per screen group (H8: recomputed per group)."""
    groups: Dict[Tuple[str, Any], List[int]] = {}
    for i, c in enumerate(candidates):
        groups.setdefault(screen_group_key(c), []).append(i)
    out = [False] * len(candidates)
    for idxs in groups.values():
        best = idxs[choose_primary([candidates[i] for i in idxs])]
        out[best] = True
    return out
