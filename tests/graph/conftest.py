"""Fixtures for the Phase 2 graph tests. These run against the LIVE dev database.

Why live rather than a mock: the whole point of Phase 2 is that the Cypher is
correct, that MERGE really is idempotent and that the composite key constraints
really exist. A fake driver would assert only that we can format strings.

Neo4j Community exposes exactly one user database, so there is no test database
to create. Isolation is by ``run_id`` instead: every test gets its own
``pytest-<hex>`` namespace, which the fixtures delete afterwards, plus a
session-level sweep that clears anything an interrupted run left behind. Nothing
under the ``dev`` run - the namespace `python -m graph.ingestion_pipeline` writes
to - is ever touched.

If no database is reachable, every test here skips with the reason. `pytest
tests/` therefore still passes on a machine that has not run `docker compose up`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from graph.connection import Neo4jSettings, connect
from graph.ingestion_pipeline import NODE_LABELS, GraphIngestionPipeline
from graph.migrate import apply_migrations

#: Every node these tests create carries a run_id starting with this.
TEST_RUN_PREFIX = "pytest-"

#: Migration-runner tests use versions from here up, and this scratch label, so
#: they can never be confused with a real migration.
TEST_MIGRATION_FLOOR = 900
TEST_MIGRATION_LABEL = "_PytestMigrationProbe"


def _sweep(driver: Any, database: str) -> None:
    """Delete every artefact these tests can create. Safe to call any time."""
    for label in NODE_LABELS:
        driver.execute_query(
            f"MATCH (n:{label}) WHERE n.run_id STARTS WITH $prefix DETACH DELETE n",
            {"prefix": TEST_RUN_PREFIX},
            database_=database,
        )
    driver.execute_query(
        f"MATCH (n:{TEST_MIGRATION_LABEL}) DETACH DELETE n", database_=database
    )
    driver.execute_query(
        "MATCH (m:_AegisMigration) WHERE m.version >= $floor DETACH DELETE m",
        {"floor": TEST_MIGRATION_FLOOR},
        database_=database,
    )


@pytest.fixture(scope="session")
def neo4j_settings() -> Neo4jSettings:
    try:
        return Neo4jSettings.from_env()
    except RuntimeError as exc:  # no credentials configured at all
        pytest.skip(f"Neo4j not configured: {exc}")


@pytest.fixture(scope="session")
def driver(neo4j_settings: Neo4jSettings) -> Iterator[Any]:
    """A session-wide driver against the dev database, with migrations applied."""
    try:
        drv = connect(neo4j_settings)
    except Exception as exc:  # container down, wrong password, wrong port
        pytest.skip(f"Neo4j unreachable at {neo4j_settings.uri}: {type(exc).__name__}")
    # The schema under test is the one the migration runner produces - not a
    # separate copy set up by the fixture.
    apply_migrations(drv, neo4j_settings.database)
    _sweep(drv, neo4j_settings.database)
    try:
        yield drv
    finally:
        _sweep(drv, neo4j_settings.database)
        drv.close()


@pytest.fixture
def database(neo4j_settings: Neo4jSettings) -> str:
    return neo4j_settings.database


@pytest.fixture
def run_id() -> str:
    """A namespace unique to one test."""
    return f"{TEST_RUN_PREFIX}{uuid.uuid4().hex[:12]}"


@pytest.fixture
def pipeline(
    driver: Any, run_id: str, database: str
) -> Iterator[GraphIngestionPipeline]:
    """A pipeline scoped to this test's namespace; its nodes are deleted after."""
    pipe = GraphIngestionPipeline(driver, run_id, database=database)
    try:
        yield pipe
    finally:
        pipe.clear_run()


@pytest.fixture
def env() -> Any:
    """A reset simulator at the Phase 1 defaults: 12 services, 6 nodes, 48 pod slots."""
    from simulator.cluster_env import ClusterEnv

    cluster = ClusterEnv()
    cluster.reset(seed=1234)
    return cluster


def step_random(cluster: Any, rng: Any) -> None:
    """Advance one tick with uniformly random actions, resetting at episode end."""
    from simulator.cluster_env import N_ACTIONS

    if not cluster.agents:
        cluster.reset()
        return
    actions = rng.integers(0, N_ACTIONS, len(cluster.agents))
    cluster.step({a: int(x) for a, x in zip(cluster.agents, actions)})
