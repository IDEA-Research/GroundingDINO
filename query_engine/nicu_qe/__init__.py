"""nicu_qe: long-term metric query engine over medical_monitor_db (normalization, time axis, ETL, queries)."""

__version__ = "0.1.0"

__all__ = ["QueryEngine", "__version__"]


def __getattr__(name):
    """Lazy `from nicu_qe import QueryEngine` (keeps the pure-function layer import-light)."""
    if name == "QueryEngine":
        from .engine import QueryEngine

        return QueryEngine
    raise AttributeError(name)
