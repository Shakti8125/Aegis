"""Idempotency, proved by counting - not by reading the Cypher and trusting MERGE.

PLAN.md Phase 2: "The ingestion pipeline reads the simulator's event stream and
MERGEs updates into Neo4j - idempotent by construction, since the simulator can
replay or restart."  Each test here does the same thing twice (or from a fresh
pipeline, or after a simulator reset) and asserts the graph is byte-for-byte the
same size afterwards, with zero creations on the second pass.
"""

from __future__ import annotations

import numpy as np

from graph.ingestion_pipeline import GraphIngestionPipeline
from tests.graph.conftest import step_random


def test_ingesting_the_same_tick_twice_changes_nothing(pipeline, env):
    snapshot = env.graph_snapshot()

    first = pipeline.ingest(snapshot)
    before = pipeline.counts()
    assert first.nodes_created > 0, "first ingest should build the graph"
    assert first.relationships_created > 0

    second = pipeline.ingest(snapshot)
    after = pipeline.counts()

    assert after == before, f"counts moved on replay: {before} -> {after}"
    assert second.nodes_created == 0
    assert second.nodes_deleted == 0
    assert second.relationships_created == 0
    assert second.relationships_deleted == 0


def test_ingesting_the_same_tick_ten_times_changes_nothing(pipeline, env):
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    before = pipeline.counts()
    for _ in range(9):
        pipeline.ingest(snapshot)
    assert pipeline.counts() == before


def test_a_fresh_pipeline_does_not_duplicate(driver, database, pipeline, env):
    """A restarted process has no structure cache, so it rewrites every edge.

    That full rewrite must still be a no-op against an already-populated graph -
    this is the "simulator restarts" case, and the one a bare CREATE would fail.
    """
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    before = pipeline.counts()

    restarted = GraphIngestionPipeline(driver, pipeline.run_id, database=database)
    stats = restarted.ingest(snapshot)

    assert stats.structure_written is True, "a fresh pipeline must rewrite structure"
    assert stats.nodes_created == 0
    assert stats.relationships_created == 0
    assert restarted.counts() == before


def test_replaying_a_whole_episode_converges(pipeline, env):
    """Ingest 20 ticks, then replay the identical 20 snapshots. Same graph."""
    rng = np.random.default_rng(4321)
    snapshots = []
    for _ in range(20):
        snapshot = env.graph_snapshot()
        snapshots.append(snapshot)
        pipeline.ingest(snapshot)
        step_random(env, rng)

    after_first_pass = pipeline.counts()

    pipeline.resync()
    for snapshot in snapshots:
        pipeline.ingest(snapshot)

    assert pipeline.counts() == after_first_pass


def test_simulator_restart_does_not_duplicate(pipeline, env):
    """env.reset() with the same seed replays tick 0 - the graph must not grow."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        pipeline.ingest(env.graph_snapshot())
        step_random(env, rng)

    env.reset(seed=1234)
    pipeline.resync()
    stats = pipeline.ingest(env.graph_snapshot())
    baseline = pipeline.counts()

    env.reset(seed=1234)
    pipeline.resync()
    repeat = pipeline.ingest(env.graph_snapshot())

    assert pipeline.counts() == baseline
    assert repeat.nodes_created == 0
    assert repeat.relationships_created == 0
    assert stats.tick == repeat.tick == 0


def test_structure_cache_does_not_change_the_result(driver, database, env, run_id):
    """The cache is an optimisation only: cached and uncached must agree exactly."""
    rng = np.random.default_rng(2024)
    snapshots = []
    for _ in range(15):
        snapshots.append(env.graph_snapshot())
        step_random(env, rng)

    cached = GraphIngestionPipeline(driver, run_id + "-on", database=database)
    uncached = GraphIngestionPipeline(
        driver, run_id + "-off", database=database, structure_cache=False
    )
    try:
        for snapshot in snapshots:
            cached.ingest(snapshot)
            uncached.ingest(snapshot)
        assert cached.counts() == uncached.counts()
    finally:
        cached.clear_run()
        uncached.clear_run()


def test_no_bare_create_in_the_ingestion_statements():
    """The structural guarantee behind all of the above, asserted on the Cypher."""
    from graph.ingestion_pipeline import TICK_QUERIES

    assert len(TICK_QUERIES) == 3
    for query in TICK_QUERIES.values():
        for line in query.splitlines():
            assert not line.strip().upper().startswith("CREATE"), line
        assert "MERGE" in query
