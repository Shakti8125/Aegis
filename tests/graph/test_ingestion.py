"""Ingestion correctness against the live dev database.

The claim under test is PLAN.md Phase 2's first half: the graph *is* the
simulator's state - the four patterns, the right properties, and nothing extra
left over from a tick that has already passed.

Every test here works inside its own ``pytest-<hex>`` run_id namespace (see
conftest.py) and deletes it afterwards, so nothing is left in the dev graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from graph.ingestion_pipeline import (
    MERGE_NODES,
    MERGE_RELS,
    NODE_LABELS,
    PRUNE_PODS,
    DROP_RELS,
    RECONCILE_RELS,
    REL_ENDPOINTS,
    GraphIngestionPipeline,
    _check_props,
)
from tests.graph.conftest import step_random


def _expected_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    return {
        **{label: len(snapshot["nodes"][label]) for label in NODE_LABELS},
        **{rel: len(snapshot["relationships"][rel]) for rel in REL_ENDPOINTS},
    }


# ------------------------------------------------------------------ the shape


def test_one_tick_reproduces_the_snapshot_exactly(pipeline, env):
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    assert pipeline.counts() == _expected_counts(snapshot)


def test_the_four_plan_patterns_connect_the_right_labels(pipeline, env, driver, database):
    """(Pod)-[:INSTANCE_OF]->(Service) and friends, checked from the database side."""
    pipeline.ingest(env.graph_snapshot())
    for rel, (src, dst) in REL_ENDPOINTS.items():
        wrong = driver.execute_query(
            f"MATCH (a)-[r:{rel}]->(b) WHERE a.run_id = $run "
            f"AND (NOT a:{src} OR NOT b:{dst}) RETURN count(r) AS c",
            {"run": pipeline.run_id},
            database_=database,
        ).records[0]["c"]
        assert wrong == 0, f"{rel} connects labels other than ({src})->({dst})"


def test_plan_mandated_node_properties_are_present(pipeline, env, driver, database):
    """PLAN.md: health, cpu_pct, mem_pct, restart_count update every tick."""
    pipeline.ingest(env.graph_snapshot())
    for label in NODE_LABELS:
        records = driver.execute_query(
            f"MATCH (n:{label} {{run_id: $run}}) RETURN properties(n) AS p",
            {"run": pipeline.run_id},
            database_=database,
        ).records
        assert records, f"no {label} nodes written"
        for record in records:
            props = record["p"]
            for key in ("id", "run_id", "tick", "health", "cpu_pct", "mem_pct", "restart_count"):
                assert key in props, f"{label} is missing {key}"


def test_calls_edges_carry_latency_and_error_rate(pipeline, env, driver, database):
    """PLAN.md: (:Service)-[:CALLS {p99_latency_ms, error_rate}]->(:Service)."""
    pipeline.ingest(env.graph_snapshot())
    records = driver.execute_query(
        "MATCH (:Service {run_id: $run})-[r:CALLS]->(:Service) RETURN properties(r) AS p",
        {"run": pipeline.run_id},
        database_=database,
    ).records
    assert records
    for record in records:
        props = record["p"]
        assert props["p99_latency_ms"] >= 0.0
        assert 0.0 <= props["error_rate"] <= 1.0


def test_property_values_match_the_snapshot(pipeline, env, driver, database):
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    expected = {row["id"]: row for row in snapshot["nodes"]["Service"]}
    records = driver.execute_query(
        "MATCH (s:Service {run_id: $run}) RETURN s.id AS id, properties(s) AS p",
        {"run": pipeline.run_id},
        database_=database,
    ).records
    assert len(records) == len(expected)
    for record in records:
        source = expected[record["id"]]
        stored = record["p"]
        for key, value in source.items():
            if isinstance(value, float):
                assert stored[key] == pytest.approx(value, rel=1e-6, abs=1e-9)
            else:
                assert stored[key] == value


def test_tick_advances_on_every_node(pipeline, env, driver, database):
    pipeline.ingest(env.graph_snapshot())
    rng = np.random.default_rng(5)
    for _ in range(3):
        step_random(env, rng)
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    ticks = driver.execute_query(
        "MATCH (s:Service {run_id: $run}) RETURN collect(DISTINCT s.tick) AS t",
        {"run": pipeline.run_id},
        database_=database,
    ).records[0]["t"]
    assert ticks == [snapshot["tick"]]


# ------------------------------------------------------------ staying in sync


def test_pods_that_disappear_are_deleted_not_orphaned(pipeline, env, driver, database):
    """graph_snapshot() lists only alive pods, so scaled-down pods must vanish."""
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    before = pipeline.counts()
    assert before["Pod"] > 2

    # Drop two pods from the snapshot, exactly as a scale_down would.
    dropped = {row["id"] for row in snapshot["nodes"]["Pod"][:2]}
    trimmed = {
        **snapshot,
        "nodes": {
            **snapshot["nodes"],
            "Pod": [r for r in snapshot["nodes"]["Pod"] if r["id"] not in dropped],
        },
        "relationships": {
            rel: [
                r
                for r in rows
                if r.get("source") not in dropped and r.get("target") not in dropped
            ]
            for rel, rows in snapshot["relationships"].items()
        },
    }
    pipeline.ingest(trimmed)

    after = pipeline.counts()
    assert after["Pod"] == before["Pod"] - 2
    assert after["INSTANCE_OF"] == before["INSTANCE_OF"] - 2
    assert after["RUNS_ON"] == before["RUNS_ON"] - 2

    leftover = driver.execute_query(
        "MATCH (p:Pod {run_id: $run}) WHERE p.id IN $dropped RETURN count(p) AS c",
        {"run": pipeline.run_id, "dropped": sorted(dropped)},
        database_=database,
    ).records[0]["c"]
    assert leftover == 0


def test_pods_that_come_back_are_re_linked(pipeline, env):
    """Scale down then up: the pod and both of its edges return, once each."""
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    full = pipeline.counts()

    victim = snapshot["nodes"]["Pod"][0]["id"]
    trimmed = {
        **snapshot,
        "nodes": {
            **snapshot["nodes"],
            "Pod": [r for r in snapshot["nodes"]["Pod"] if r["id"] != victim],
        },
        "relationships": {
            rel: [r for r in rows if victim not in (r.get("source"), r.get("target"))]
            for rel, rows in snapshot["relationships"].items()
        },
    }
    pipeline.ingest(trimmed)
    assert pipeline.counts()["Pod"] == full["Pod"] - 1

    pipeline.ingest(snapshot)
    assert pipeline.counts() == full


def test_stale_service_edges_are_pruned(pipeline, env):
    """A dependency that disappears from the topology must leave the graph."""
    snapshot = env.graph_snapshot()
    pipeline.ingest(snapshot)
    before = pipeline.counts()
    assert before["CALLS"] > 1

    trimmed = {
        **snapshot,
        "relationships": {
            **snapshot["relationships"],
            "CALLS": snapshot["relationships"]["CALLS"][1:],
        },
    }
    pipeline.ingest(trimmed)
    assert pipeline.counts()["CALLS"] == before["CALLS"] - 1


def test_graph_tracks_a_whole_episode(pipeline, env):
    """After every tick of a churning episode the graph equals the snapshot."""
    rng = np.random.default_rng(99)
    for _ in range(25):
        snapshot = env.graph_snapshot()
        pipeline.ingest(snapshot)
        assert pipeline.counts() == _expected_counts(snapshot)
        step_random(env, rng)


# --------------------------------------------------------------- run isolation


def test_two_runs_do_not_share_nodes(driver, database, env, run_id):
    """The same service names under two run_ids stay two separate clusters."""
    other = run_id + "-b"
    first = GraphIngestionPipeline(driver, run_id, database=database)
    second = GraphIngestionPipeline(driver, other, database=database)
    try:
        snapshot = env.graph_snapshot()
        first.ingest(snapshot)
        second.ingest(snapshot)
        assert first.counts() == second.counts() == _expected_counts(snapshot)

        first.clear_run()
        assert first.counts()["Service"] == 0
        assert second.counts() == _expected_counts(snapshot)
    finally:
        first.clear_run()
        second.clear_run()


def test_clear_run_removes_everything(pipeline, env):
    pipeline.ingest(env.graph_snapshot())
    assert pipeline.clear_run() > 0
    assert set(pipeline.counts().values()) == {0}


# ------------------------------------------------------------- query planning


@pytest.mark.parametrize(
    "query",
    [
        *MERGE_NODES.values(),
        *MERGE_RELS.values(),
        *DROP_RELS.values(),
        *RECONCILE_RELS.values(),
        PRUNE_PODS,
    ],
)
def test_every_ingestion_fragment_uses_an_index(driver, database, query, run_id):
    """No label scans on the per-tick path - the reason schema.cypher adds no
    indexes beyond the three constraint-backed ones."""

    def operators(plan: dict[str, Any]) -> list[str]:
        found = [plan["operatorType"]]
        for child in plan.get("children", []):
            found += operators(child)
        return found

    params: dict[str, Any] = {"run_id": run_id, "tick": 0}
    for label in NODE_LABELS:
        params[f"rows_{label}"] = []
        params[f"ids_{label}"] = []
    for rel in REL_ENDPOINTS:
        params[f"rows_{rel}"] = []
        params[f"keep_{rel}"] = []
        params[f"drop_{rel}"] = []

    summary = driver.execute_query(
        "EXPLAIN " + query, params, database_=database
    ).summary
    ops = operators(summary.plan)
    assert not any("NodeByLabelScan" in op for op in ops), (
        f"{query.splitlines()[0]} falls back to a label scan: {ops}"
    )
    assert any("IndexSeek" in op for op in ops), f"no index seek in plan: {ops}"


# --------------------------------------------------------------- input guards


def test_nested_properties_are_rejected_by_name():
    with pytest.raises(TypeError, match="Service.meta"):
        _check_props("Service", {"id": "x", "meta": {"nested": 1}})


def test_lists_of_scalars_are_allowed():
    _check_props("Service", {"id": "x", "labels": ["a", "b"], "empty": None})


def test_empty_run_id_is_rejected(driver, database):
    with pytest.raises(ValueError, match="run_id"):
        GraphIngestionPipeline(driver, "", database=database)


def test_settings_never_print_the_password():
    from graph.connection import Neo4jSettings

    settings = Neo4jSettings(uri="bolt://x:7687", username="neo4j", password="hunter2")
    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings)
    assert "hunter2" not in f"{settings}"
