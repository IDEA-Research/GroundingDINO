"""Prometheus metric catalog + medical vocabulary.

Ground truth for metric names used by spec generation and the LLM
prompts. The catalog is fetched live from Prometheus
`/api/v1/label/__name__/values` with a short TTL cache.

There is NO fallback metric list: when discovery fails, `list_metrics`
raises `PrometheusUnavailable` so callers degrade EXPLICITLY (a note in
the generated spec, an honest report to the user) instead of silently
substituting metrics that may not exist on this box.

Clinical vocabulary lives here too: `MEDICAL_SYNONYMS` maps 中文 and
English clinician vocabulary ("心率", "spo2", "呼吸器"…) to the real
`patient_*` / `ventilator_*` / `cerebral_*` metric families exported by
the camera-stream pipeline. Synonym hits are deliberately kept even when
the metric is currently absent from the live catalog — patient metrics
vanish whenever no stream runs, but they are still the RIGHT answer for
"幫我盯心率" (they will exist the moment a stream starts).
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from urllib import request as urllib_request

from .client import PROM_URL, REQUEST_TIMEOUT, PrometheusUnavailable


# ---------------------------------------------------------------------------
# Clinical metric families and vocabulary
# ---------------------------------------------------------------------------

# Metrics under these prefixes carry a free-text `device_model` label
# (LLM output, MULTIPLE variants per physical device — deliberate, do
# not "fix" upstream). One patient's vital therefore appears as N
# parallel series; queries must aggregate `max by (patient_id) (...)`.
CLINICAL_METRIC_PREFIXES: tuple[str, ...] = (
    "patient_",
    "ventilator_",
    "cerebral_",
)


def is_clinical_metric(metric: str) -> bool:
    return metric.startswith(CLINICAL_METRIC_PREFIXES)


_HR = "patient_heart_rate_bpm"
_SPO2 = "patient_spo2_percentage"
_RESP = "patient_respiration_rate_per_minute"
_BP_SYS = "patient_blood_pressure_systolic_mmhg"
_BP_DIA = "patient_blood_pressure_diastolic_mmhg"
_RSO2 = ("cerebral_rso2_left_percentage", "cerebral_rso2_right_percentage")

# Vocabulary (lowercase-normalized, 中文 + English) -> real metric names.
# Matching rules (see `resolve_synonyms`):
#   - ASCII keys match on word-ish boundaries so "hr" never fires inside
#     "threshold" and "pip" never fires inside "pipeline";
#   - CJK keys match by substring;
#   - longer keys are consumed first so "呼吸器" (ventilator) does not
#     also fire the "呼吸" (respiration) key.
MEDICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    # heart rate
    "心率": (_HR,),
    "心跳": (_HR,),
    "heart rate": (_HR,),
    "hr": (_HR,),
    "pulse": (_HR,),
    # oxygen saturation
    "血氧": (_SPO2,),
    "氧飽和": (_SPO2,),
    "飽和度": (_SPO2,),
    "oxygen saturation": (_SPO2,),
    "spo2": (_SPO2,),
    # respiration
    "呼吸率": (_RESP,),
    "呼吸": (_RESP,),
    "respiration": (_RESP,),
    "rr": (_RESP,),
    "breathing": (_RESP,),
    # blood pressure
    "血壓": (_BP_SYS,),
    "收縮壓": (_BP_SYS,),
    "blood pressure": (_BP_SYS,),
    "bp": (_BP_SYS,),
    "systolic": (_BP_SYS,),
    "舒張壓": (_BP_DIA,),
    "diastolic": (_BP_DIA,),
    # cerebral oximetry (both hemispheres)
    "腦氧": _RSO2,
    "近紅外": _RSO2,
    "rso2": _RSO2,
    "cerebral": _RSO2,
    "nirs": _RSO2,
    # ventilator
    "呼吸器": ("ventilator_fio2_percentage",),
    "吸氧濃度": ("ventilator_fio2_percentage",),
    "ventilator": ("ventilator_fio2_percentage",),
    "fio2": ("ventilator_fio2_percentage",),
    "peep": ("ventilator_peep_cmh2o",),
    "pip": ("ventilator_pip_cmh2o",),
    # vitals trio
    "生命徵象": (_HR, _SPO2, _RESP),
    "生理數值": (_HR, _SPO2, _RESP),
    "vitals": (_HR, _SPO2, _RESP),
}

# Longest key first so overlapping vocabulary ("呼吸器" vs "呼吸",
# "blood pressure" vs "bp") is consumed by the most specific entry.
_SYNONYM_KEYS_BY_LENGTH: list[str] = sorted(
    MEDICAL_SYNONYMS, key=len, reverse=True
)

_ASCII_KEY = re.compile(r"^[\x00-\x7f]+$")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def resolve_synonyms(text: str) -> list[str]:
    """Map free text (中文/English) to clinical metric names.

    Pure and network-free. Returns metric names deduplicated in
    first-match order. Presence in the live catalog is NOT checked here
    — callers decide (clinical metrics are the right answer even while
    absent, because they disappear whenever no camera stream runs).
    """
    working = _normalize(text)
    if not working:
        return []
    out: list[str] = []
    for key in _SYNONYM_KEYS_BY_LENGTH:
        if _ASCII_KEY.match(key):
            # Word-ish boundaries: not preceded/followed by [a-z0-9].
            # (`\b` fails next to CJK — CJK chars count as \w — so
            # "看hr和血氧" would not match with plain \b.)
            pattern = re.compile(
                rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])"
            )
            if not pattern.search(working):
                continue
            working = pattern.sub("\x00", working)
        else:
            if key not in working:
                continue
            working = working.replace(key, "\x00")
        out.extend(MEDICAL_SYNONYMS[key])
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Live catalog
# ---------------------------------------------------------------------------

_DEFAULT_TTL_SECONDS = 30.0


def _ttl_from_env() -> float:
    raw = os.getenv("HELPER_DASHBOARD_CATALOG_TTL_SECONDS", "")
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TTL_SECONDS
    return value if value >= 0 else _DEFAULT_TTL_SECONDS


@dataclass
class MetricSuggestion:
    metric: str
    # True/False when the live catalog answered; the clinical synonym
    # hits report present=False while no stream is running (still the
    # right answer — see module docstring).
    present: bool
    reason: str


class MetricCatalog:
    """TTL-cached live metric catalog. No fallback list, ever."""

    def __init__(
        self,
        base_url: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or PROM_URL).rstrip("/")
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else _ttl_from_env()
        )
        self._cache: list[str] | None = None
        self._fetched_at: float = 0.0

    # -- fetch / cache ---------------------------------------------
    def _fetch_names(self) -> list[str]:
        """GET /api/v1/label/__name__/values. Raises PrometheusUnavailable."""
        url = f"{self.base_url}/api/v1/label/__name__/values"
        try:
            with urllib_request.urlopen(  # noqa: S310
                url, timeout=REQUEST_TIMEOUT
            ) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise PrometheusUnavailable(
                f"metric catalog discovery failed: {exc}"
            ) from exc
        values = raw.get("data") or []
        return sorted(str(v) for v in values)

    def list_metrics(self) -> list[str]:
        """Live metric names, cached for `ttl_seconds`.

        Raises `PrometheusUnavailable` on discovery failure — callers
        must degrade explicitly (never a silent substitute list).
        """
        now = time.monotonic()
        if (
            self._cache is not None
            and (now - self._fetched_at) < self.ttl_seconds
        ):
            return self._cache
        self._cache = self._fetch_names()
        self._fetched_at = time.monotonic()
        return self._cache

    def invalidate(self) -> None:
        self._cache = None
        self._fetched_at = 0.0

    # -- queries ---------------------------------------------------
    def has(self, metric: str) -> bool:
        """Raises PrometheusUnavailable when the catalog is unreachable."""
        return metric in self.list_metrics()

    def suggest(self, hint: str, limit: int = 10) -> list[str]:
        """Metric names for a free-text hint (names only).

        Raises `PrometheusUnavailable` when the live catalog cannot be
        fetched — callers degrade explicitly.
        """
        return [s.metric for s in self.suggest_detailed(hint, limit=limit)]

    def suggest_detailed(
        self, hint: str, limit: int = 10
    ) -> list[MetricSuggestion]:
        """Synonym hits first, then live-catalog substring matches.

        - Medical synonym hits are ALWAYS kept, even when absent from
          the live catalog (patient_* metrics vanish while no stream
          runs); such hits carry ``present=False``.
        - Catalog substring matches are present by construction.
        - Deduplicated preserving order; raises `PrometheusUnavailable`
          when the catalog is unreachable.
        """
        live = self.list_metrics()
        live_set = set(live)
        normalized = _normalize(hint).strip()
        out: list[MetricSuggestion] = []
        seen: set[str] = set()
        for metric in resolve_synonyms(hint):
            if metric in seen:
                continue
            seen.add(metric)
            out.append(
                MetricSuggestion(
                    metric=metric,
                    present=metric in live_set,
                    reason="medical synonym",
                )
            )
        if normalized:
            for metric in live:
                if len(out) >= limit:
                    break
                if metric in seen:
                    continue
                if normalized in metric.lower():
                    seen.add(metric)
                    out.append(
                        MetricSuggestion(
                            metric=metric,
                            present=True,
                            reason="catalog match",
                        )
                    )
        return out[:limit]


# ---------------------------------------------------------------------------
# Shared default instance
# ---------------------------------------------------------------------------

_DEFAULT_CATALOG: MetricCatalog | None = None


def get_catalog() -> MetricCatalog:
    """Process-wide shared catalog (one TTL cache for all callers)."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = MetricCatalog()
    return _DEFAULT_CATALOG
