"""Load metric_aliases.json / device_families.json and expose key and device normalization lookups."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Pattern, Sequence, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
METRIC_ALIASES_PATH = DATA_DIR / "metric_aliases.json"
DEVICE_FAMILIES_PATH = DATA_DIR / "device_families.json"

UNKNOWN_FAMILY = "unknown"
METRIC_FAMILIES = ("monitor", "invos", "osypka", "ventilator")
VALID_PARSERS = ("number", "ratio")
# contract v2 #14 defaults (overridable in metric_aliases.json)
DEFAULT_UNKNOWN_FAMILY_DENY = ("rr", "r", "l", "map", "f", "mv_set")
DEFAULT_UNKNOWN_FAMILY_MIN_LEN = 3

_WS_HYPHEN_RE = re.compile(r"[\s\-]+")


class AliasConfigError(ValueError):
    """Raised when an alias / device-family JSON file is structurally invalid."""


def normalize_key_text(raw_key: str) -> str:
    """Apply the key normalization rule: strip -> lower -> spaces/hyphens to underscores."""
    return _WS_HYPHEN_RE.sub("_", str(raw_key).strip().lower())


@dataclass(frozen=True)
class MetricSpec:
    """One canonical metric definition from metric_aliases.json."""

    name: str
    family: str
    unit: str
    range: Optional[Tuple[float, float]]
    aliases: Tuple[str, ...]
    fraction_to_percent: bool = False
    parser: str = "number"


@dataclass(frozen=True)
class CompositeSpec:
    """A composite key (e.g. blood_pressure) that is split into several canonical metrics."""

    name: str
    family: str
    aliases: Tuple[str, ...]
    parts: Mapping[str, Tuple[str, ...]]  # canonical -> accepted sub-keys of an object value


def _require(obj: Mapping[str, Any], key: str, typ: Any, where: str) -> Any:
    """Return obj[key] after checking presence and type, else raise AliasConfigError."""
    if not isinstance(obj, Mapping) or key not in obj:
        raise AliasConfigError(f"{where}: missing required field {key!r}")
    val = obj[key]
    if not isinstance(val, typ):
        raise AliasConfigError(f"{where}.{key}: expected {typ}, got {type(val).__name__}")
    return val


def _str_list(val: Any, where: str, allow_empty: bool = False) -> List[str]:
    """Validate a list of non-empty strings."""
    if not isinstance(val, list) or (not allow_empty and not val):
        raise AliasConfigError(f"{where}: expected a non-empty list of strings")
    for item in val:
        if not isinstance(item, str) or not item.strip():
            raise AliasConfigError(f"{where}: contains a non-string or empty entry {item!r}")
    return val


class AliasRegistry:
    """Lookup tables built from the two JSON data files; immutable after construction."""

    def __init__(self, metric_doc: Mapping[str, Any], device_doc: Mapping[str, Any]) -> None:
        """Validate both JSON documents and build per-family lookup tables."""
        self.metrics_version: str = _require(metric_doc, "version", str, "metric_aliases")
        self.devices_version: str = _require(device_doc, "version", str, "device_families")
        self.metrics: Dict[str, MetricSpec] = {}
        self.composites: Dict[str, CompositeSpec] = {}
        self._by_family: Dict[str, Dict[str, str]] = {f: {} for f in METRIC_FAMILIES}
        self._unknown: Dict[str, str] = {}
        self.ignore: Set[str] = set()
        self._load_metrics(metric_doc)
        self._family_res: List[Tuple[str, List[Pattern[str]]]] = []
        self._model_res: List[Tuple[str, List[Pattern[str]]]] = []
        self._load_devices(device_doc)

    # ------------------------------------------------------------------ loading
    def _add_alias(self, family: str, alias: str, target: str, where: str) -> None:
        """Register one normalized alias in a family table, rejecting conflicts."""
        key = normalize_key_text(alias)
        table = self._by_family[family]
        if key in table and table[key] != target:
            raise AliasConfigError(
                f"{where}: alias {alias!r} (normalized {key!r}) maps to both "
                f"{table[key]!r} and {target!r} in family {family!r}"
            )
        table[key] = target

    def _load_metrics(self, doc: Mapping[str, Any]) -> None:
        """Validate and index metric_aliases.json."""
        metrics = _require(doc, "metrics", dict, "metric_aliases")
        if not metrics:
            raise AliasConfigError("metric_aliases.metrics is empty")
        for name, spec in metrics.items():
            where = f"metric_aliases.metrics.{name}"
            family = _require(spec, "family", str, where)
            if family not in METRIC_FAMILIES:
                raise AliasConfigError(f"{where}.family: unknown family {family!r}")
            unit = _require(spec, "unit", str, where)
            if "range" not in spec:
                raise AliasConfigError(f"{where}: missing required field 'range'")
            rng = spec["range"]
            if rng is not None:
                if (
                    not isinstance(rng, list)
                    or len(rng) != 2
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rng)
                    or rng[0] > rng[1]
                ):
                    raise AliasConfigError(f"{where}.range: expected null or [lo, hi] with lo <= hi")
                rng = (float(rng[0]), float(rng[1]))
            aliases = _str_list(_require(spec, "aliases", list, where), f"{where}.aliases")
            parser = spec.get("parser", "number")
            if parser not in VALID_PARSERS:
                raise AliasConfigError(f"{where}.parser: must be one of {VALID_PARSERS}")
            ftp = spec.get("fraction_to_percent", False)
            if not isinstance(ftp, bool):
                raise AliasConfigError(f"{where}.fraction_to_percent: must be a boolean")
            self.metrics[name] = MetricSpec(name, family, unit, rng, tuple(aliases), ftp, parser)
            for alias in aliases:
                self._add_alias(family, alias, name, where)
            extra = spec.get("extra_family_aliases", {})
            if not isinstance(extra, dict):
                raise AliasConfigError(f"{where}.extra_family_aliases must be an object")
            for fam, extra_aliases in extra.items():
                if fam not in METRIC_FAMILIES:
                    raise AliasConfigError(f"{where}.extra_family_aliases: unknown family {fam!r}")
                for alias in _str_list(extra_aliases, f"{where}.extra_family_aliases.{fam}"):
                    self._add_alias(fam, alias, name, where)

        composites = doc.get("composites", {})
        if not isinstance(composites, dict):
            raise AliasConfigError("metric_aliases.composites must be an object")
        for name, spec in composites.items():
            where = f"metric_aliases.composites.{name}"
            if name in self.metrics:
                raise AliasConfigError(f"{where}: name collides with a metric")
            family = _require(spec, "family", str, where)
            if family not in METRIC_FAMILIES:
                raise AliasConfigError(f"{where}.family: unknown family {family!r}")
            aliases = _str_list(_require(spec, "aliases", list, where), f"{where}.aliases")
            parts_raw = _require(spec, "parts", dict, where)
            parts: Dict[str, Tuple[str, ...]] = {}
            for canon, subkeys in parts_raw.items():
                if canon not in self.metrics:
                    raise AliasConfigError(f"{where}.parts: {canon!r} is not a defined metric")
                parts[canon] = tuple(
                    normalize_key_text(s) for s in _str_list(subkeys, f"{where}.parts.{canon}")
                )
            self.composites[name] = CompositeSpec(name, family, tuple(aliases), parts)
            for alias in aliases:
                self._add_alias(family, alias, name, where)

        self.ignore = {
            normalize_key_text(k) for k in _str_list(_require(doc, "ignore", list, "metric_aliases"),
                                                     "metric_aliases.ignore", allow_empty=True)
        }
        deny_raw = doc.get("unknown_family_deny", list(DEFAULT_UNKNOWN_FAMILY_DENY))
        deny = {normalize_key_text(k) for k in _str_list(deny_raw, "metric_aliases.unknown_family_deny",
                                                        allow_empty=True)}
        min_len = doc.get("unknown_family_min_len", DEFAULT_UNKNOWN_FAMILY_MIN_LEN)
        if not isinstance(min_len, int) or isinstance(min_len, bool) or min_len < 1:
            raise AliasConfigError("metric_aliases.unknown_family_min_len must be a positive integer")
        self.unknown_family_deny = frozenset(deny)
        # unknown family (contract v2 #14): accept an alias if its normalized text maps to ONE canonical
        # across all families, is at least min_len long, and is not in the deny list.
        global_map: Dict[str, Set[str]] = {}
        for table in self._by_family.values():
            for key, target in table.items():
                global_map.setdefault(key, set()).add(target)
        self._unknown = {
            key: next(iter(targets))
            for key, targets in global_map.items()
            if len(targets) == 1 and len(key) >= min_len and key not in deny
            and next(iter(targets)) not in self.composites
        }
        overlap = self.ignore & set(global_map)
        if overlap:
            raise AliasConfigError(f"ignore list overlaps aliases: {sorted(overlap)}")

    def _load_devices(self, doc: Mapping[str, Any]) -> None:
        """Validate and compile device_families.json."""
        families = _require(doc, "families", dict, "device_families")
        models = _require(doc, "models", dict, "device_families")
        for fam, patterns in families.items():
            if fam not in METRIC_FAMILIES and fam != UNKNOWN_FAMILY:
                raise AliasConfigError(f"device_families.families: unknown family {fam!r}")
            self._family_res.append((fam, self._compile(patterns, f"device_families.families.{fam}")))
        for model, patterns in models.items():
            self._model_res.append((model, self._compile(patterns, f"device_families.models.{model}")))

    @staticmethod
    def _compile(patterns: Any, where: str) -> List[Pattern[str]]:
        """Compile a list of case-insensitive regexes."""
        out = []
        for p in _str_list(patterns, where):
            try:
                out.append(re.compile(p, re.IGNORECASE))
            except re.error as exc:
                raise AliasConfigError(f"{where}: invalid regex {p!r}: {exc}") from exc
        return out

    # ------------------------------------------------------------------ lookups
    @property
    def version(self) -> str:
        """Combined version string of both data files."""
        return f"metrics:{self.metrics_version}+devices:{self.devices_version}"

    def is_ignored(self, raw_key: str) -> bool:
        """True if the key is a known junk key that should be dropped silently."""
        return normalize_key_text(raw_key) in self.ignore

    def is_composite(self, name: Optional[str]) -> bool:
        """True if name is a composite key (e.g. blood_pressure) rather than a scalar metric."""
        return name in self.composites

    def normalize_key(self, family: str, raw_key: str) -> Optional[str]:
        """Map a raw medical_values key to a canonical metric (or composite) name for a family, else None."""
        if not isinstance(raw_key, str):
            return None
        key = normalize_key_text(raw_key)
        if not key or key in self.ignore:
            return None
        table = self._by_family.get(family)
        if table is None:
            return self._unknown.get(key)
        return table.get(key)

    def classify_device(
        self, detected_model_raw: Optional[str], corrected_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """Return (device_family, device_model) from corrected_model (if set) else detected_model."""
        text = corrected_model if isinstance(corrected_model, str) and corrected_model.strip() else detected_model_raw
        if not isinstance(text, str) or not text.strip():
            return UNKNOWN_FAMILY, "unknown"
        family = UNKNOWN_FAMILY
        for fam, regs in self._family_res:
            if any(r.search(text) for r in regs):
                family = fam
                break
        if family == UNKNOWN_FAMILY:
            return UNKNOWN_FAMILY, "unknown"
        for model, regs in self._model_res:
            if any(r.search(text) for r in regs):
                return family, model
        return family, f"{family}_other"

    def known_raw_aliases(self, family: str) -> Sequence[str]:
        """Return the normalized alias keys accepted for a family (for diagnostics)."""
        table = self._by_family.get(family, self._unknown)
        return sorted(table)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise AliasConfigError(f"{path.name}: top level must be a JSON object")
    return data


def load_registry(
    metric_path: Optional[Path] = None, device_path: Optional[Path] = None
) -> AliasRegistry:
    """Load and validate both JSON files into an AliasRegistry."""
    return AliasRegistry(
        _read_json(metric_path or METRIC_ALIASES_PATH),
        _read_json(device_path or DEVICE_FAMILIES_PATH),
    )


_REGISTRY: Optional[AliasRegistry] = None


def get_registry() -> AliasRegistry:
    """Return the process-wide registry, loading it on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def normalize_key(family: str, raw_key: str) -> Optional[str]:
    """Module-level shortcut for get_registry().normalize_key."""
    return get_registry().normalize_key(family, raw_key)


def classify_device(detected_model_raw: Optional[str], corrected_model: Optional[str] = None) -> Tuple[str, str]:
    """Module-level shortcut for get_registry().classify_device."""
    return get_registry().classify_device(detected_model_raw, corrected_model)


def __getattr__(name: str) -> Any:
    """Expose `aliases.version` lazily (PEP 562)."""
    if name == "version":
        return get_registry().version
    raise AttributeError(name)
