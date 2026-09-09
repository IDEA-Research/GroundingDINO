"""Threshold direction semantics — shared, deterministic resolver.

A widget's thresholds are a list of `{value, color, label?, direction?}`
bands. The bands alone do not say whether HIGH or LOW readings are the
alarmed side: an HR ladder `[100 amber, 120 red]` alarms ABOVE, while
the live SpO2 ladder `[94 amber 偏低 <95%, 90 red 危急 <90%]` alarms
BELOW. Rendering (stat card state text, gauge bands, chart reference
lines) must agree on one effective direction per widget, and the
frontend helper `frontend/lib/threshold.ts` mirrors THIS logic exactly.

Rules, in order (first hit wins):

1. explicit — any threshold carries `direction`: the first non-null
   `direction` in list order is used for the whole widget.
2. severity ordering — with >= 2 thresholds, compare the most severe
   band (red > amber/orange > yellow > other, by colour name or hue)
   with the least severe one: if the more severe band has the LOWER
   value the widget alarms "below"; if it has the HIGHER value, "above".
   Equal severities fall through.
3. label hint — any label containing "<", "≤" or "低於" -> "below".
4. default -> "above".

Clinical VALUES are never touched here: this module only decides which
side of an existing band is the alarmed side.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal

Direction = Literal["above", "below"]

_BELOW_LABEL_HINTS: tuple[str, ...] = ("<", "≤", "低於")

# Colour names -> severity rank. Mirrors the frontend colour normaliser
# (chart-theme.ts alarmColorForSpec) so both sides rank a band the same.
_NAME_RANK: dict[str, int] = {
    "red": 3, "crimson": 3, "rose": 3, "danger": 3, "critical": 3,
    "amber": 2, "orange": 2, "warn": 2, "warning": 2,
    "yellow": 1, "gold": 1,
}

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3,8})$")


def _hex_to_rgb(hex_body: str) -> tuple[float, float, float] | None:
    body = hex_body
    if len(body) in (3, 4):
        body = "".join(ch * 2 for ch in body[:3])
    elif len(body) in (6, 8):
        body = body[:6]
    else:
        return None
    try:
        r = int(body[0:2], 16) / 255.0
        g = int(body[2:4], 16) / 255.0
        b = int(body[4:6], 16) / 255.0
    except ValueError:
        return None
    return r, g, b


def _hue_sat(rgb: tuple[float, float, float]) -> tuple[float, float]:
    r, g, b = rgb
    mx = max(r, g, b)
    mn = min(r, g, b)
    delta = mx - mn
    if delta == 0:
        return 0.0, 0.0
    if mx == r:
        h = ((g - b) / delta) % 6
    elif mx == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    hue = (h * 60.0) % 360.0
    sat = delta / mx if mx else 0.0
    return hue, sat


def color_severity(color: str | None) -> int:
    """Rank a threshold colour: 3 red, 2 amber/orange, 1 yellow, 0 other.

    Names are matched case-insensitively; hex colours are ranked by hue
    (red < 15 or >= 340, amber/orange 15..45, yellow 45..70). Greys
    (saturation < 0.25) and greens/blues rank 0.
    """
    if not color:
        return 0
    text = color.strip().lower()
    if text in _NAME_RANK:
        return _NAME_RANK[text]
    m = _HEX_RE.match(text)
    if not m:
        return 0
    rgb = _hex_to_rgb(m.group(1))
    if rgb is None:
        return 0
    hue, sat = _hue_sat(rgb)
    if sat < 0.25:
        return 0
    if hue < 15 or hue >= 340:
        return 3
    if hue < 45:
        return 2
    if hue < 70:
        return 1
    return 0


def _field(t: Any, name: str) -> Any:
    if isinstance(t, dict):
        return t.get(name)
    return getattr(t, name, None)


def resolve_direction(thresholds: Iterable[Any] | None) -> Direction:
    """Effective alarm direction for a widget's threshold list.

    Accepts `WidgetThreshold` models or plain dicts (raw spec JSON).
    See the module docstring for the rule order.
    """
    items = [t for t in (thresholds or []) if t is not None]
    if not items:
        return "above"

    # 1. explicit
    for t in items:
        d = _field(t, "direction")
        if d in ("above", "below"):
            return d  # type: ignore[return-value]

    # 2. severity ordering
    if len(items) >= 2:
        ranked: list[tuple[int, float]] = []
        for t in items:
            try:
                value = float(_field(t, "value"))
            except (TypeError, ValueError):
                continue
            ranked.append((color_severity(_field(t, "color")), value))
        if len(ranked) >= 2:
            most = max(ranked, key=lambda rv: rv[0])
            least = min(ranked, key=lambda rv: rv[0])
            if most[0] > least[0]:
                if most[1] < least[1]:
                    return "below"
                if most[1] > least[1]:
                    return "above"

    # 3. label hint
    for t in items:
        label = str(_field(t, "label") or "")
        if any(h in label for h in _BELOW_LABEL_HINTS):
            return "below"

    # 4. default
    return "above"


def active_threshold(value: float | None, thresholds: Iterable[Any] | None) -> Any | None:
    """The band a reading currently sits in, honouring the effective
    direction: for "above" the highest threshold with `value >= t.value`,
    for "below" the lowest threshold with `value <= t.value`. None when
    the reading is on the un-alarmed side of every band or is missing."""
    if value is None:
        return None
    items = [t for t in (thresholds or []) if t is not None]
    if not items:
        return None
    direction = resolve_direction(items)
    hit = None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if direction == "above":
        for t in sorted(items, key=lambda t: float(_field(t, "value"))):
            if v >= float(_field(t, "value")):
                hit = t
    else:
        for t in sorted(items, key=lambda t: -float(_field(t, "value"))):
            if v <= float(_field(t, "value")):
                hit = t
    return hit


DIRECTION_RULES_DOC = (
    "Effective direction per widget: (1) any threshold's explicit "
    "`direction` (first non-null wins); else (2) with >= 2 thresholds, "
    "if the most severe colour (red > amber/orange > yellow > other) sits "
    "at the LOWER value the widget alarms `below`, at the HIGHER value "
    "`above`; else (3) a label containing `<`, `≤` or `低於` means "
    "`below`; else (4) `above`."
)
