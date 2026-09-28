"""Mongo access: a read-only view of the source DB and a guarded handle on the derived QE DB.

Safety rules (DESIGN 12.4, REVIEW M4/M6):
- the source client is always ``MongoClient(tz_aware=False)`` so source Dates come back as the
  naive wall-clock values they really are (see tz.source_wallclock_to_utc);
- the source is only reachable through ``ReadOnlySource`` / ``ReadOnlyCollection``, which expose
  find / find_one / aggregate / count_documents / distinct / estimated_document_count and nothing
  else, always with maxTimeMS, and refuse aggregation stages that write ($out / $merge);
- the QE handle refuses any DB name that is not ``medical_monitor_qe(_test)`` or equals the source.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import Settings, guard_db_names

_WRITE_STAGES = frozenset({"$out", "$merge"})


class ReadOnlyViolation(RuntimeError):
    """Raised when code tries to write through the read-only source view."""


def _check_pipeline(pipeline: Sequence[Mapping[str, Any]]) -> None:
    """Reject aggregation pipelines that contain a writing stage."""
    for stage in pipeline:
        if not isinstance(stage, Mapping):
            raise ReadOnlyViolation("pipeline stages must be mappings")
        bad = _WRITE_STAGES.intersection(stage.keys())
        if bad:
            raise ReadOnlyViolation(f"stage {sorted(bad)} is not allowed on the source DB")


class ReadOnlyCollection:
    """Read-only facade over one source collection. Every call carries maxTimeMS."""

    __slots__ = ("_coll", "_ms", "name")

    def __init__(self, coll: Any, max_time_ms: int) -> None:
        """Wrap a pymongo Collection; no write method is exposed."""
        object.__setattr__(self, "_coll", coll)
        object.__setattr__(self, "_ms", int(max_time_ms))
        object.__setattr__(self, "name", coll.name)

    def __setattr__(self, key: str, value: Any) -> None:
        """Immutable."""
        raise ReadOnlyViolation("ReadOnlyCollection is immutable")

    def find(self, filter: Optional[Mapping[str, Any]] = None, projection: Any = None, **kwargs: Any):
        """collection.find(...) with max_time_ms applied."""
        kwargs.pop("max_time_ms", None)
        return self._coll.find(filter or {}, projection, max_time_ms=self._ms, **kwargs)

    def find_one(self, filter: Optional[Mapping[str, Any]] = None, projection: Any = None, **kwargs: Any):
        """collection.find_one(...) with max_time_ms applied."""
        kwargs.pop("max_time_ms", None)
        return self._coll.find_one(filter or {}, projection, max_time_ms=self._ms, **kwargs)

    def aggregate(self, pipeline: Sequence[Mapping[str, Any]], **kwargs: Any):
        """collection.aggregate(...) with maxTimeMS; $out / $merge are refused."""
        _check_pipeline(pipeline)
        kwargs.pop("maxTimeMS", None)
        return self._coll.aggregate(list(pipeline), maxTimeMS=self._ms, **kwargs)

    def count_documents(self, filter: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> int:
        """collection.count_documents(...) with maxTimeMS."""
        kwargs.pop("maxTimeMS", None)
        return self._coll.count_documents(filter or {}, maxTimeMS=self._ms, **kwargs)

    def distinct(self, key: str, filter: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> List[Any]:
        """collection.distinct(...) with maxTimeMS."""
        kwargs.pop("maxTimeMS", None)
        return self._coll.distinct(key, filter or {}, maxTimeMS=self._ms, **kwargs)

    def estimated_document_count(self) -> int:
        """collection.estimated_document_count() with maxTimeMS."""
        return self._coll.estimated_document_count(maxTimeMS=self._ms)


class ReadOnlySource:
    """Read-only facade over the source database: ``src["screen_analysis"].find(...)``."""

    __slots__ = ("_db", "_ms", "name")

    def __init__(self, db: Any, max_time_ms: int) -> None:
        """Wrap a pymongo Database."""
        object.__setattr__(self, "_db", db)
        object.__setattr__(self, "_ms", int(max_time_ms))
        object.__setattr__(self, "name", db.name)

    def __setattr__(self, key: str, value: Any) -> None:
        """Immutable."""
        raise ReadOnlyViolation("ReadOnlySource is immutable")

    def __getitem__(self, name: str) -> ReadOnlyCollection:
        """Return a read-only collection view."""
        return ReadOnlyCollection(self._db[name], self._ms)

    def ping(self) -> bool:
        """Server ping (read-only admin command)."""
        self._db.client.admin.command("ping")
        return True


def make_source_client(uri: str, timeout_ms: int = 5000):
    """MongoClient for the source DB: tz_aware=False is mandatory (REVIEW M4)."""
    from pymongo import MongoClient

    return MongoClient(
        uri, tz_aware=False, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms,
        appname="nicu_qe-source-ro",
    )


def make_qe_client(uri: str, timeout_ms: int = 5000):
    """MongoClient for the derived DB: returns aware UTC datetimes (values are stored as true UTC)."""
    from pymongo import MongoClient

    return MongoClient(
        uri, tz_aware=True, tzinfo=timezone.utc, serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms, appname="nicu_qe",
    )


class Mongo:
    """Holds the source (read-only) and QE (read-write, name-guarded) handles."""

    def __init__(self, settings: Settings, source_client: Any = None, qe_client: Any = None) -> None:
        """Build clients lazily-connecting clients; raises UnsafeConfigError on a bad QE_DB."""
        guard_db_names(settings.db, settings.source_db)
        self.settings = settings
        self._source_client = source_client or make_source_client(settings.source_uri)
        self._qe_client = qe_client or make_qe_client(settings.source_uri)
        self.source = ReadOnlySource(self._source_client[settings.source_db], settings.mongo_max_time_ms)
        self.qe = self._qe_client[settings.db]
        if self.qe.name != settings.db or self.qe.name == settings.source_db:
            raise RuntimeError("QE database handle does not match the guarded name")

    @property
    def max_time_ms(self) -> int:
        """Per-operation time limit."""
        return self.settings.mongo_max_time_ms

    def rename_into(self, src_coll: str, dst_coll: str) -> None:
        """renameCollection(dropTarget=True) inside the QE DB only."""
        guard_db_names(self.qe.name, self.settings.source_db)
        self._qe_client.admin.command(
            "renameCollection", f"{self.qe.name}.{src_coll}", to=f"{self.qe.name}.{dst_coll}", dropTarget=True
        )

    def ping_source(self) -> bool:
        """Ping the source server."""
        return self.source.ping()

    def ping_qe(self) -> bool:
        """Ping via the QE client."""
        self._qe_client.admin.command("ping")
        return True

    def close(self) -> None:
        """Close both clients."""
        for c in (self._source_client, self._qe_client):
            try:
                c.close()
            except Exception:
                pass


def chunked(seq: Iterable[Any], n: int):
    """Yield lists of at most n items."""
    buf: List[Any] = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def as_dict(doc: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a plain dict (empty for None)."""
    return dict(doc) if doc else {}
