"""Integration fixtures: only run with QE_TEST_MONGO=1 against QE_DB=medical_monitor_qe_test (orin11)."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip every integration test unless explicitly enabled."""
    if os.environ.get("QE_TEST_MONGO") == "1" and os.environ.get("QE_DB") == "medical_monitor_qe_test":
        return
    skip = pytest.mark.skip(reason="set QE_TEST_MONGO=1 and QE_DB=medical_monitor_qe_test to run")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def settings():
    from nicu_qe.config import Settings

    st = Settings.from_env()
    assert st.db == "medical_monitor_qe_test", "integration tests only write to the _test DB"
    return st


@pytest.fixture(scope="session")
def mongo(settings):
    from nicu_qe.mongo import Mongo

    m = Mongo(settings)
    yield m
    m.close()


@pytest.fixture(scope="session")
def engine(settings, mongo):
    from nicu_qe.engine import QueryEngine

    return QueryEngine(settings, mongo=mongo)


@pytest.fixture(scope="session")
def etl(settings, mongo):
    from nicu_qe.etl import Etl

    return Etl(mongo, settings)
