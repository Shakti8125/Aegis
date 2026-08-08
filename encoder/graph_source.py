"""Where a snapshot comes from: the simulator (default) or, optionally, Neo4j.

Phase 3 - owned by the gnn-architect subagent. See PLAN.md section 3, Phase 3.

The encoder consumes exactly one shape - the PLAN.md section 3 Phase 2 snapshot
dict - and two things produce it:

* :func:`snapshot_from_env` - ``ClusterEnv.graph_snapshot()``. This is the
  default and the *only* path Phase 3's gate uses.
* :func:`snapshot_from_neo4j` - the same dict, read back out of the graph
  ``graph/ingestion_pipeline.py`` writes.

**The Neo4j path is optional by construction.** ``graph`` and ``neo4j`` are
imported inside the function, so ``import encoder`` works on a machine that has
never run ``docker compose up``; training and ``python -m encoder.probe`` never
touch this module.  ``tests/graph/`` already skips itself when no database is
reachable and the encoder gate must not be the one thing that cannot, since
Phase 4 depends on it.

Why the read path exists at all: it is what lets a trained encoder run against
the live graph in Phase 6-8 (and against a real cluster in the Phase 8 demo)
without a second feature implementation that could silently drift from this one.
``tests/encoder/test_graph_source.py`` round-trips simulator -> Neo4j -> encoder
and asserts the two sources produce identical tensors - when a database is
there, and skips when it is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from neo4j import Driver

    from simulator.cluster_env import ClusterEnv

__all__ = ["snapshot_from_env", "snapshot_from_neo4j"]

#: Relationship type -> the property keys the snapshot carries for it.
_REL_PROPERTIES: dict[str, tuple[str, ...]] = {
    "DEPENDS_ON": (),
    "INSTANCE_OF": (),
    "RUNS_ON": (),
    "CALLS": ("p99_latency_ms", "error_rate", "traffic_share"),
}

_NODE_QUERY = """
MATCH (n:{label} {{run_id: $run_id}})
RETURN properties(n) AS props
"""

_REL_QUERY = """
MATCH (a:{src} {{run_id: $run_id}})-[r:{rel}]->(b:{dst} {{run_id: $run_id}})
RETURN a.id AS source, b.id AS target, properties(r) AS props
"""

_REL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "DEPENDS_ON": ("Service", "Service"),
    "INSTANCE_OF": ("Pod", "Service"),
    "RUNS_ON": ("Pod", "Node"),
    "CALLS": ("Service", "Service"),
}


def snapshot_from_env(env: "ClusterEnv") -> dict[str, Any]:
    """The default source. A thin alias, so call sites name their source."""
    return env.graph_snapshot()


def snapshot_from_neo4j(
    driver: "Driver", run_id: str, *, database: str | None = None
) -> dict[str, Any]:
    """Read one run's graph back out of Neo4j in ``graph_snapshot()`` shape.

    The bookkeeping properties ingestion adds (``run_id``, ``tick``) are dropped
    from node rows so the result is byte-comparable with what the simulator
    emitted; ``tick`` is lifted to the top level where the snapshot keeps it.
    Requires a reachable database - callers that must not depend on one should
    use :func:`snapshot_from_env`.
    """
    from graph.connection import DEFAULT_DATABASE
    from graph.ingestion_pipeline import NODE_LABELS

    db = database or DEFAULT_DATABASE
    nodes: dict[str, list[dict[str, Any]]] = {}
    tick = 0
    for label in NODE_LABELS:
        records, _, _ = driver.execute_query(
            _NODE_QUERY.format(label=label), {"run_id": run_id}, database_=db
        )
        rows = []
        for record in records:
            props = dict(record["props"])
            props.pop("run_id", None)
            tick = max(tick, int(props.pop("tick", 0) or 0))
            rows.append(props)
        rows.sort(key=lambda row: str(row.get("id", "")))
        nodes[label] = rows

    relationships: dict[str, list[dict[str, Any]]] = {}
    for rel, (src, dst) in _REL_ENDPOINTS.items():
        records, _, _ = driver.execute_query(
            _REL_QUERY.format(src=src, dst=dst, rel=rel), {"run_id": run_id}, database_=db
        )
        rows = []
        for record in records:
            props = dict(record["props"])
            row: dict[str, Any] = {
                "source": record["source"],
                "target": record["target"],
            }
            for key in _REL_PROPERTIES[rel]:
                if key in props:
                    row[key] = props[key]
            rows.append(row)
        rows.sort(key=lambda row: (str(row["source"]), str(row["target"])))
        relationships[rel] = rows

    return {
        "tick": tick,
        "nodes": nodes,
        "relationships": relationships,
        "active_faults": [],
    }
