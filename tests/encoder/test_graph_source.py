"""The optional Neo4j read path produces the same tensors as the simulator path.

The encoder's gate must never need a database - Phase 4 depends on Phase 3 being
runnable anywhere - so every test here skips when none is reachable, exactly as
``tests/graph/`` does. What it checks when a database *is* there is that
simulator -> ingestion -> Cypher read -> HeteroData lands on the same numbers as
simulator -> HeteroData, so there is only ever one feature implementation.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

pytest.importorskip("torch", reason="Phase 3 needs torch (see requirements.txt)")
pytest.importorskip(
    "torch_geometric", reason="Phase 3 needs torch-geometric (see requirements.txt)"
)

import torch  # noqa: E402

from encoder.features import NODE_TYPES, RELATIONS, snapshot_to_hetero_data  # noqa: E402
from encoder.graph_source import snapshot_from_env, snapshot_from_neo4j  # noqa: E402
from simulator.cluster_env import N_ACTIONS, ClusterEnv  # noqa: E402

TEST_RUN_PREFIX = "pytest-encoder-"


@pytest.fixture(scope="module")
def driver_and_database():
    """A live dev driver with migrations applied, or a skip. Never a mock."""
    pytest.importorskip("neo4j", reason="Phase 2 driver not installed")
    from graph.connection import Neo4jSettings, connect
    from graph.migrate import apply_migrations

    try:
        settings = Neo4jSettings.from_env()
    except RuntimeError as exc:
        pytest.skip(f"Neo4j not configured: {exc}")
    try:
        driver = connect(settings)
    except Exception as exc:
        pytest.skip(f"Neo4j unreachable at {settings.uri}: {type(exc).__name__}")

    apply_migrations(driver, settings.database)
    try:
        yield driver, settings.database
    finally:
        driver.close()


@pytest.fixture
def pipeline(driver_and_database) -> Iterator:
    from graph.ingestion_pipeline import GraphIngestionPipeline

    driver, database = driver_and_database
    run_id = f"{TEST_RUN_PREFIX}{uuid.uuid4().hex[:12]}"
    pipe = GraphIngestionPipeline(driver, run_id, database=database)
    try:
        yield pipe
    finally:
        pipe.clear_run()


def _sorted_by_id(snapshot: dict) -> dict:
    """Neo4j has no row order; compare both sides in a canonical one."""
    out = dict(snapshot)
    out["nodes"] = {
        label: sorted(rows, key=lambda row: str(row["id"]))
        for label, rows in snapshot["nodes"].items()
    }
    out["relationships"] = {
        rel: sorted(rows, key=lambda row: (str(row["source"]), str(row["target"])))
        for rel, rows in snapshot["relationships"].items()
    }
    return out


def test_neo4j_round_trip_gives_identical_encoder_input(pipeline, driver_and_database):
    driver, database = driver_and_database
    env = ClusterEnv()
    env.reset(seed=1234)
    generator = torch.Generator().manual_seed(1234)
    for _ in range(12):
        actions = torch.randint(0, N_ACTIONS, (len(env.agents),), generator=generator)
        env.step({a: int(x) for a, x in zip(env.agents, actions)})

    from_simulator = snapshot_from_env(env)
    pipeline.ingest(from_simulator)
    from_graph = snapshot_from_neo4j(driver, pipeline.run_id, database=database)

    expected = snapshot_to_hetero_data(_sorted_by_id(from_simulator))
    actual = snapshot_to_hetero_data(_sorted_by_id(from_graph))

    for ntype in NODE_TYPES:
        assert actual[ntype].id == expected[ntype].id, ntype
        torch.testing.assert_close(actual[ntype].x, expected[ntype].x)
        torch.testing.assert_close(actual[ntype].y, expected[ntype].y)
    for relation in RELATIONS:
        torch.testing.assert_close(
            actual[relation].edge_index, expected[relation].edge_index
        )
        attr = expected[relation].get("edge_attr")
        if attr is not None:
            torch.testing.assert_close(actual[relation].edge_attr, attr)


def test_a_frozen_encoder_reads_the_live_graph(pipeline, driver_and_database):
    """End to end: whatever Phase 6 serves, the encoder can already consume."""
    from encoder.gnn_model import AegisGraphEncoder, EncoderConfig

    driver, database = driver_and_database
    env = ClusterEnv()
    env.reset(seed=99)
    pipeline.ingest(snapshot_from_env(env))

    encoder = AegisGraphEncoder(EncoderConfig(hidden_dim=16, embed_dim=16, global_dim=32))
    encoder.eval()
    out = encoder.encode_snapshot(
        snapshot_from_neo4j(driver, pipeline.run_id, database=database)
    )
    assert out.agent_observations.shape == (env.n_services, encoder.embed_dim)
    assert out.global_embedding.shape == (1, encoder.global_dim)
    assert torch.isfinite(out.global_embedding).all()
