"""Aegis knowledge graph - Neo4j schema, migrations and simulator ingestion.

Phase 2 - owned by the graph-engineer subagent. See PLAN.md section 3, Phase 2.

    (:Service)-[:DEPENDS_ON]->(:Service)
    (:Pod)-[:INSTANCE_OF]->(:Service)
    (:Pod)-[:RUNS_ON]->(:Node)
    (:Service)-[:CALLS {p99_latency_ms, error_rate}]->(:Service)

Layout:
    schema.cypher          current schema as one readable page (documentation)
    migrations/            numbered, applied by `python -m graph.migrate --apply`
    migrate.py             migration runner + applied-migration ledger
    ingestion_pipeline.py  ClusterEnv.graph_snapshot() -> Neo4j, per tick
    connection.py          credentials from env/.env, never hardcoded
    benchmark.py           measured per-tick sync latency

Typical use::

    from graph import GraphIngestionPipeline

    with GraphIngestionPipeline.from_env("dev") as pipeline:
        pipeline.ingest(env.graph_snapshot())

The re-exports below are resolved lazily. Importing them eagerly would import
``graph.migrate`` as a side effect of ``python -m graph.migrate`` (RuntimeWarning
about a module already in sys.modules) and would pull in the neo4j driver for
anyone who only wanted, say, ``graph.migrate.split_statements``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - re-exported for type checkers only
    from graph.connection import Neo4jSettings, neo4j_driver
    from graph.ingestion_pipeline import (
        DEFAULT_RUN_ID,
        GraphIngestionPipeline,
        IngestStats,
        stream_episode,
    )
    from graph.migrate import Migration, apply_migrations, discover_migrations

_EXPORTS: dict[str, str] = {
    "DEFAULT_RUN_ID": "graph.ingestion_pipeline",
    "GraphIngestionPipeline": "graph.ingestion_pipeline",
    "IngestStats": "graph.ingestion_pipeline",
    "stream_episode": "graph.ingestion_pipeline",
    "Migration": "graph.migrate",
    "apply_migrations": "graph.migrate",
    "discover_migrations": "graph.migrate",
    "Neo4jSettings": "graph.connection",
    "neo4j_driver": "graph.connection",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return __all__
