"""Keeps the Neo4j knowledge graph in sync with the simulator, one tick at a time.

Phase 2 - owned by the graph-engineer subagent. See PLAN.md section 3, Phase 2.

Input contract
--------------
The single input is ``ClusterEnv.graph_snapshot()`` (simulator/cluster_env.py),
which already emits the PLAN.md section 3 Phase 2 shape - labels
``Service``/``Pod``/``Node``, relationships ``DEPENDS_ON``/``INSTANCE_OF``/
``RUNS_ON``/``CALLS``, node properties health/cpu_pct/mem_pct/restart_count,
``CALLS`` properties p99_latency_ms/error_rate.  There is deliberately no second
event format to keep in step with the simulator; the contract is pinned by
``tests/simulator/test_topology.py::test_graph_snapshot_matches_phase2_schema``.

Idempotent by construction
--------------------------
Every write is a ``MERGE`` on the ``(run_id, id)`` node key - never a bare
``CREATE`` - so replaying a tick, restarting the simulator or re-running a whole
episode converges on the same graph instead of duplicating it.  Node and
relationship counts after ingesting a tick twice are identical;
``tests/graph/test_ingestion.py`` asserts that against the live database rather
than by inspection.

*Sync*, not *append*: pods that scale down or crash out disappear from the
snapshot (``graph_snapshot`` lists only alive pods), so each tick also sweeps
away pods of this run that the snapshot no longer contains.  Otherwise the graph
would monotonically accumulate ghosts and every downstream consumer - the Phase 3
encoder especially - would read a cluster larger than the one that exists.

Latency
-------
PLAN.md's Phase 2 bar is "the graph stays in sync with simulator state at low
latency", so this was measured (``graph/benchmark.py``) rather than guessed, and
the shape of the code follows what the measurement showed:

1. **UNWIND batching, not a query per node.** One subquery per label and per
   relationship type, parameterized with the whole row list.
2. **One round trip per tick.** Those subqueries are composed into a *single*
   Cypher statement of unit ``CALL {}`` blocks, run in one transaction. This is
   the big one: profiled against the dev container, a round trip costs ~2.5 ms
   and server-side work for a whole tick costs ~6 ms, so the twelve-statement
   version spent three quarters of its time waiting on the wire (42.7 ms/tick
   versus 11.9 ms/tick for the identical writes composed into one statement).
3. **Remembering the topology.** The pipeline keeps the last edge set it wrote,
   which collapses a tick into one of three shapes:

   ``properties`` - topology unchanged, so only node properties and ``CALLS``
   metrics are written. Nothing structural, nothing removed.

   ``delta`` - topology changed, and the remembered set says exactly *how*. The
   structural edges are re-MERGEd and the removals are named explicitly, so the
   removal step is an ``UNWIND`` over a list that is usually empty rather than a
   traversal of every edge in the run.

   ``reconcile`` - the first ingest of a pipeline, where nothing can be assumed
   about what the database already holds, so the removal step is a full sweep
   ("delete every edge of this run I did not just send"). This is the expensive
   shape, and it runs once per pipeline rather than once per tick.

   Correctness never depends on what is remembered: an empty memory means
   ``reconcile``, and every structural difference is written. If a database is
   wiped underneath a live pipeline, call :meth:`resync` to force one.

One thing was measured and deliberately *not* done: sending only the properties
whose values changed since the previous tick. Only ~34% of node property values
move per tick, but cutting the payload to a quarter of its size did not move the
per-tick time outside run-to-run noise, so it does not pay for a per-node write
cache that could silently disagree with the database.

A note on the ``delta``/``reconcile`` split, since the numbers do not justify it:
interleaved A/B/C runs put the delta form, the reconcile sweep and the original
label-scan sweep within noise of each other at this cluster size. The split is
kept because it is the one that *scales* - a label scan reads every Service in
the database including other runs', and the reconcile sweep's cost tracks the run
rather than the tick's actual change - not because it measured faster today.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from graph.connection import DEFAULT_DATABASE, Neo4jSettings, connect

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from neo4j import Driver, ManagedTransaction

    from simulator.cluster_env import ClusterEnv

DEFAULT_RUN_ID = "dev"

#: Labels, in the order graph_snapshot() nests them under "nodes".
NODE_LABELS: tuple[str, ...] = ("Service", "Pod", "Node")

#: Relationship type -> (source label, target label).
REL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "DEPENDS_ON": ("Service", "Service"),
    "INSTANCE_OF": ("Pod", "Service"),
    "RUNS_ON": ("Pod", "Node"),
    "CALLS": ("Service", "Service"),
}

#: Structure-only relationships: rewritten only when the topology changes.
#: CALLS is excluded because its properties (p99_latency_ms, error_rate) move
#: every tick even when the edge set does not.
STRUCTURAL_RELS: tuple[str, ...] = ("DEPENDS_ON", "INSTANCE_OF", "RUNS_ON")

_PROP_SCALARS = (str, bool, int, float)

#: Remembered topology: relationship type -> set of (source id, target id).
Structure = dict[str, frozenset[tuple[str, str]]]

__all__ = [
    "DEFAULT_RUN_ID",
    "NODE_LABELS",
    "REL_ENDPOINTS",
    "GraphIngestionPipeline",
    "IngestStats",
    "stream_episode",
]


# ---------------------------------------------------------------------------
# Cypher fragments
#
# Labels and relationship types cannot be parameterized in Cypher, so these are
# built once, at import, from the constant tables above - never from input. Each
# fragment carries a distinct parameter name (`$rows_Pod`, `$keep_CALLS`, ...) so
# that composing them into one statement cannot collide.
#
# Every fragment is a valid statement on its own, which is what lets
# tests/graph/test_ingestion.py EXPLAIN them individually and assert that each
# one seeks the (run_id, id) index instead of scanning a label.
# ---------------------------------------------------------------------------
def _merge_nodes_query(label: str) -> str:
    return (
        f"UNWIND $rows_{label} AS row\n"
        f"MERGE (n:{label} {{run_id: $run_id, id: row.id}})\n"
        f"SET n += row, n.tick = $tick"
    )


def _merge_rels_query(rel_type: str) -> str:
    src, dst = REL_ENDPOINTS[rel_type]
    return (
        f"UNWIND $rows_{rel_type} AS row\n"
        f"MATCH (a:{src} {{run_id: $run_id, id: row.source}})\n"
        f"MATCH (b:{dst} {{run_id: $run_id, id: row.target}})\n"
        f"MERGE (a)-[r:{rel_type}]->(b)\n"
        f"SET r += row.props, r.tick = $tick"
    )


def _drop_rels_query(rel_type: str) -> str:
    """Delete exactly the edges that vanished since the last tick.

    The normal way to keep an edge set in sync is a sweep - "delete every edge of
    this run that is not in the list I just sent". This does the opposite: the
    pipeline already remembers the previous topology, so it can name the removals
    and delete only those. The list is empty on almost every tick, which makes
    this an UNWIND over nothing rather than a traversal over everything.

    The sweep still exists as :func:`_reconcile_rels_query` for the one case this
    cannot cover: a pipeline that has no memory of a previous tick.
    """
    src, dst = REL_ENDPOINTS[rel_type]
    return (
        f"UNWIND $drop_{rel_type} AS row\n"
        f"MATCH (a:{src} {{run_id: $run_id, id: row.source}})"
        f"-[r:{rel_type}]->"
        f"(b:{dst} {{run_id: $run_id, id: row.target}})\n"
        f"DELETE r"
    )


def _reconcile_rels_query(rel_type: str) -> str:
    """Sweep: drop every edge of this run the current snapshot does not contain.

    Only used on a pipeline's first ingest, where there is no remembered topology
    to diff against and the database may hold anything.

    Anchored by UNWINDing the source ids rather than written as the more obvious
    ``MATCH (a:Service {run_id: $run_id})-[r]->(b:Service {run_id: $run_id})``.
    That form plans as a ``NodeByLabelScan`` - Neo4j will not use the composite
    index for a bare run_id prefix here, and an index hint is rejected in this
    context - which would scan *every* Service in the database, including every
    other run's. Seeding the seek with the ids we already hold makes it a
    ``NodeUniqueIndexSeek`` whose cost tracks this run and no one else's - the
    right trade for a statement that runs once per pipeline rather than once per
    tick, and one that measured no slower than the label scan at this size.
    """
    src, dst = REL_ENDPOINTS[rel_type]
    return (
        f"UNWIND $ids_{src} AS source_id\n"
        f"MATCH (a:{src} {{run_id: $run_id, id: source_id}})-[r:{rel_type}]->(b:{dst})\n"
        f"WHERE NOT a.id + '>' + b.id IN $keep_{rel_type}\n"
        f"DELETE r"
    )


MERGE_NODES: dict[str, str] = {label: _merge_nodes_query(label) for label in NODE_LABELS}
MERGE_RELS: dict[str, str] = {rel: _merge_rels_query(rel) for rel in REL_ENDPOINTS}
DROP_RELS: dict[str, str] = {rel: _drop_rels_query(rel) for rel in REL_ENDPOINTS}
RECONCILE_RELS: dict[str, str] = {
    rel: _reconcile_rels_query(rel) for rel in REL_ENDPOINTS
}

#: graph_snapshot() lists only alive pods; any other Pod in this run is a ghost.
#: This one *is* a run_id prefix seek on the composite index - the planner takes
#: it here because there is no relationship pattern to cost against it.
PRUNE_PODS = (
    "MATCH (p:Pod {run_id: $run_id})\nWHERE NOT p.id IN $ids_Pod\nDETACH DELETE p"
)

DELETE_RUN = "MATCH (n:{label} {{run_id: $run_id}}) DETACH DELETE n"
COUNT_NODES = "MATCH (n:{label} {{run_id: $run_id}}) RETURN count(n) AS c"
COUNT_RELS = (
    "MATCH (:{src} {{run_id: $run_id}})-[r:{rel}]->(:{dst} {{run_id: $run_id}}) "
    "RETURN count(r) AS c"
)


def _compose(fragments: list[str]) -> str:
    """Wrap each fragment in a unit ``CALL () {}`` block and concatenate.

    Unit subqueries run in clause order for the single implicit input row, which
    is what the ordering relies on: nodes exist before the relationships that
    MATCH them, and the stale-pod sweep happens before the relationship writes.

    ``CALL ()`` is the explicit empty variable-scope clause: these subqueries
    import nothing from the outer scope, only parameters. The bare ``CALL {}``
    form is deprecated as of Neo4j 5.23 and warns once per statement, which at
    one statement per tick would be a torrent - so the floor here is Neo4j 5.23,
    comfortably under the 5.26 dev container and AuraDB.
    """
    blocks = []
    for fragment in fragments:
        body = "\n".join("  " + line for line in fragment.splitlines())
        blocks.append(f"CALL () {{\n{body}\n}}")
    return "\n".join(blocks)


#: One of three shapes a tick can take. See :meth:`GraphIngestionPipeline._statement`.
MODE_PROPERTIES = "properties"  # topology unchanged: node properties + CALLS
MODE_DELTA = "delta"  # topology changed, and we know how it changed
MODE_RECONCILE = "reconcile"  # first ingest: assume nothing about the database


def _tick_query(mode: str) -> str:
    fragments = [MERGE_NODES[label] for label in NODE_LABELS]
    fragments.append(PRUNE_PODS)
    if mode == MODE_PROPERTIES:
        fragments.append(MERGE_RELS["CALLS"])
        return _compose(fragments)
    fragments += [MERGE_RELS[rel] for rel in (*STRUCTURAL_RELS, "CALLS")]
    removals = DROP_RELS if mode == MODE_DELTA else RECONCILE_RELS
    fragments += [removals[rel] for rel in REL_ENDPOINTS]
    return _compose(fragments)


#: The only three statements this pipeline ever sends. Precomputed so Neo4j's
#: plan cache hits on every tick after the first of each shape.
TICK_QUERIES: dict[str, str] = {
    mode: _tick_query(mode)
    for mode in (MODE_PROPERTIES, MODE_DELTA, MODE_RECONCILE)
}


@dataclass(frozen=True)
class IngestStats:
    """What one :meth:`GraphIngestionPipeline.ingest` call did, and how long it took.

    ``duration_ms`` is wall-clock around the whole write transaction - parameter
    marshalling, the round trip and the commit - measured with
    ``time.perf_counter``. It is the number PLAN.md's "stays in sync at low
    latency" is judged on, so it deliberately includes everything the caller
    waits for.
    """

    run_id: str
    tick: int
    duration_ms: float
    subqueries: int
    mode: str
    rows: dict[str, int] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def structure_written(self) -> bool:
        """Did this tick write the topology, or only node properties and CALLS?"""
        return self.mode != MODE_PROPERTIES

    @property
    def nodes_created(self) -> int:
        return self.counters.get("nodes_created", 0)

    @property
    def nodes_deleted(self) -> int:
        return self.counters.get("nodes_deleted", 0)

    @property
    def relationships_created(self) -> int:
        return self.counters.get("relationships_created", 0)

    @property
    def relationships_deleted(self) -> int:
        return self.counters.get("relationships_deleted", 0)

    @property
    def properties_set(self) -> int:
        return self.counters.get("properties_set", 0)

    def __str__(self) -> str:
        return (
            f"tick {self.tick:>4}  {self.duration_ms:6.2f} ms  "
            f"{self.subqueries:>2} subqueries  "
            f"+{self.nodes_created}n/-{self.nodes_deleted}n  "
            f"+{self.relationships_created}r/-{self.relationships_deleted}r  "
            f"{self.properties_set} props  [{self.mode}]"
        )


def _split_rel_row(row: dict[str, Any]) -> dict[str, Any]:
    """``{source, target, **props}`` -> ``{source, target, props}``.

    Generic on purpose: a new CALLS property added by the simulator flows into
    the graph without a change here.
    """
    return {
        "source": row["source"],
        "target": row["target"],
        "props": {k: v for k, v in row.items() if k not in ("source", "target")},
    }


def _check_props(label: str, row: dict[str, Any]) -> None:
    """Neo4j properties must be scalars or homogeneous lists of scalars.

    Raise naming the offending key rather than let the driver fail mid-transaction
    with a message that does not say which property caused it.
    """
    for key, value in row.items():
        if value is None or isinstance(value, _PROP_SCALARS):
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(v, _PROP_SCALARS) for v in value
        ):
            continue
        raise TypeError(
            f"{label}.{key} is {type(value).__name__}, which Neo4j cannot store as a "
            f"property. graph_snapshot() must emit flat scalar properties."
        )


class GraphIngestionPipeline:
    """MERGEs ``ClusterEnv.graph_snapshot()`` into Neo4j, once per simulation tick.

    ``run_id`` namespaces one simulation run inside the single Neo4j Community
    database (PLAN.md section 10 deploys to AuraDB Free, which is also single
    database). Two pipelines with different run_ids never touch each other's
    nodes; two pipelines with the *same* run_id converge on the same graph.
    """

    def __init__(
        self,
        driver: "Driver",
        run_id: str = DEFAULT_RUN_ID,
        *,
        database: str = DEFAULT_DATABASE,
        structure_cache: bool = True,
        owns_driver: bool = False,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.driver = driver
        self.run_id = run_id
        self.database = database
        self.structure_cache = structure_cache
        self._owns_driver = owns_driver
        self._structure: Structure | None = None

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def from_env(
        cls,
        run_id: str = DEFAULT_RUN_ID,
        *,
        settings: Neo4jSettings | None = None,
        **kwargs: Any,
    ) -> "GraphIngestionPipeline":
        """Build a pipeline from NEO4J_* in the environment or repo-root .env."""
        cfg = settings or Neo4jSettings.from_env()
        driver = connect(cfg)
        kwargs.setdefault("database", cfg.database)
        return cls(driver, run_id, owns_driver=True, **kwargs)

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def __enter__(self) -> "GraphIngestionPipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def resync(self) -> None:
        """Forget the structure cache, so the next ingest rewrites every edge.

        Call this when the database may have changed behind the pipeline's back
        (a manual wipe, a restored dump, a crashed run) or when the simulator has
        been reset onto a different topology.
        """
        self._structure = None

    # ---------------------------------------------------------------- ingest
    def ingest(self, snapshot: dict[str, Any]) -> IngestStats:
        """Sync one ``graph_snapshot()`` into the graph. Returns timing + counters."""
        tick = int(snapshot.get("tick", 0))
        nodes = snapshot["nodes"]
        rels = snapshot["relationships"]

        node_rows: dict[str, list[dict[str, Any]]] = {}
        for label in NODE_LABELS:
            rows = list(nodes.get(label, []))
            for row in rows:
                _check_props(label, row)
            node_rows[label] = rows

        rel_rows: dict[str, list[dict[str, Any]]] = {
            rel: [_split_rel_row(r) for r in rels.get(rel, [])] for rel in REL_ENDPOINTS
        }

        structure = self._structure_signature(rel_rows)
        mode = self._mode(structure)
        query, params = self._statement(tick, node_rows, rel_rows, structure, mode)

        started = time.perf_counter()
        with self.driver.session(database=self.database) as session:
            counters = session.execute_write(_run_tick, query, params)
        duration_ms = (time.perf_counter() - started) * 1000.0

        self._structure = structure
        return IngestStats(
            run_id=self.run_id,
            tick=tick,
            duration_ms=duration_ms,
            subqueries=query.count("CALL () {"),
            mode=mode,
            rows={
                **{label: len(rows) for label, rows in node_rows.items()},
                **{rel: len(rows) for rel, rows in rel_rows.items()},
            },
            counters=counters,
        )

    def ingest_env(self, env: "ClusterEnv") -> IngestStats:
        """Convenience wrapper: ``pipeline.ingest(env.graph_snapshot())``."""
        return self.ingest(env.graph_snapshot())

    # ------------------------------------------------------------- internals
    def _mode(self, structure: "Structure") -> str:
        """Which of the three tick shapes this snapshot needs.

        ``reconcile`` only on the first ingest of a pipeline, where nothing is
        known about what the database already holds. After that a remembered
        topology makes the cheaper ``delta`` removal correct, and an unchanged
        topology makes removal unnecessary altogether.
        """
        if self._structure is None:
            return MODE_RECONCILE
        if not self.structure_cache or structure != self._structure:
            return MODE_DELTA
        return MODE_PROPERTIES

    def _statement(
        self,
        tick: int,
        node_rows: dict[str, list[dict[str, Any]]],
        rel_rows: dict[str, list[dict[str, Any]]],
        structure: "Structure",
        mode: str,
    ) -> tuple[str, dict[str, Any]]:
        """Pick the tick statement and build exactly the parameters it references."""
        params: dict[str, Any] = {
            "run_id": self.run_id,
            "tick": tick,
            # Doubles as the alive-pod list PRUNE_PODS filters against.
            "ids_Pod": [row["id"] for row in node_rows["Pod"]],
        }
        for label in NODE_LABELS:
            params[f"rows_{label}"] = node_rows[label]
        if mode == MODE_PROPERTIES:
            params["rows_CALLS"] = rel_rows["CALLS"]
            return TICK_QUERIES[mode], params

        for rel in REL_ENDPOINTS:
            params[f"rows_{rel}"] = rel_rows[rel]
        if mode == MODE_RECONCILE:
            params["ids_Service"] = [row["id"] for row in node_rows["Service"]]
            for rel in REL_ENDPOINTS:
                params[f"keep_{rel}"] = [f"{s}>{t}" for s, t in structure[rel]]
            return TICK_QUERIES[mode], params

        previous = self._structure or {}
        for rel in REL_ENDPOINTS:
            gone = previous.get(rel, frozenset()) - structure[rel]
            params[f"drop_{rel}"] = [{"source": s, "target": t} for s, t in sorted(gone)]
        return TICK_QUERIES[mode], params

    @staticmethod
    def _structure_signature(
        rel_rows: dict[str, list[dict[str, Any]]],
    ) -> "Structure":
        """Topology fingerprint: which edges exist, ignoring their properties."""
        return {
            rel: frozenset((row["source"], row["target"]) for row in rel_rows[rel])
            for rel in REL_ENDPOINTS
        }

    # --------------------------------------------------------------- queries
    def counts(self) -> dict[str, int]:
        """Node and relationship counts for this run - the idempotency assertion."""
        run_id = self.run_id

        def work(tx: "ManagedTransaction") -> dict[str, int]:
            # Built inside the unit of work, which a managed transaction may retry.
            out: dict[str, int] = {}
            for label in NODE_LABELS:
                out[label] = tx.run(
                    COUNT_NODES.format(label=label), run_id=run_id
                ).single()["c"]
            for rel, (src, dst) in REL_ENDPOINTS.items():
                out[rel] = tx.run(
                    COUNT_RELS.format(rel=rel, src=src, dst=dst), run_id=run_id
                ).single()["c"]
            return out

        with self.driver.session(database=self.database) as session:
            return session.execute_read(work)

    def clear_run(self, run_id: str | None = None) -> int:
        """Delete every node of a run (and, by DETACH, its relationships)."""
        target = run_id or self.run_id

        def work(tx: "ManagedTransaction") -> int:
            deleted = 0
            for label in NODE_LABELS:
                summary = tx.run(DELETE_RUN.format(label=label), run_id=target).consume()
                deleted += summary.counters.nodes_deleted
            return deleted

        with self.driver.session(database=self.database) as session:
            deleted = session.execute_write(work)
        if target == self.run_id:
            self.resync()
        return deleted

    def run_ids(self) -> list[str]:
        """Every run_id currently present on a :Service node."""
        result = self.driver.execute_query(
            "MATCH (s:Service) RETURN DISTINCT s.run_id AS run_id ORDER BY run_id",
            database_=self.database,
        )
        return [record["run_id"] for record in result.records]


_COUNTER_FIELDS = (
    "nodes_created",
    "nodes_deleted",
    "relationships_created",
    "relationships_deleted",
    "properties_set",
    "labels_added",
)


def _run_tick(
    tx: "ManagedTransaction", query: str, params: dict[str, Any]
) -> dict[str, int]:
    """One tick = one statement = one commit."""
    counters = tx.run(query, params).consume().counters
    out: dict[str, int] = {}
    for name in _COUNTER_FIELDS:
        value = getattr(counters, name, 0)
        if value:
            out[name] = value
    return out


# ---------------------------------------------------------------------------
# Driving the simulator into the graph
# ---------------------------------------------------------------------------
def stream_episode(
    env: "ClusterEnv",
    pipeline: GraphIngestionPipeline,
    ticks: int,
    *,
    seed: int | None = None,
    rng: Any = None,
    reset_on_done: bool = True,
) -> Iterator[IngestStats]:
    """Step the simulator and ingest every tick. Yields per-tick stats.

    Uses uniformly random actions so scale_up/scale_down actually fire and the
    pod set churns - the case that exercises the stale-pod sweep and defeats the
    structure cache. A no-op policy would make ingestion look easier than it is.
    """
    import numpy as np

    from simulator.cluster_env import N_ACTIONS

    generator = rng if rng is not None else np.random.default_rng(seed)
    env.reset(seed=seed)
    yield pipeline.ingest(env.graph_snapshot())
    for _ in range(ticks):
        if not env.agents:
            if not reset_on_done:
                return
            env.reset()
            pipeline.resync()  # new episode: topology and pod set start over
        actions = {
            agent: int(action)
            for agent, action in zip(
                env.agents, generator.integers(0, N_ACTIONS, len(env.agents))
            )
        }
        env.step(actions)
        yield pipeline.ingest(env.graph_snapshot())


def main(argv: list[str] | None = None) -> int:
    """Stream a live episode into the dev graph: ``python -m graph.ingestion_pipeline``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--ticks", type=int, default=50, help="simulation ticks to ingest")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--services", type=int, default=None)
    parser.add_argument("--clear", action="store_true", help="wipe the run and exit")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from simulator.cluster_env import ClusterConfig, ClusterEnv

    with GraphIngestionPipeline.from_env(args.run_id) as pipeline:
        if args.clear:
            print(f"deleted {pipeline.clear_run()} nodes from run {args.run_id!r}")
            return 0
        cfg = (
            ClusterConfig()
            if args.services is None
            else ClusterConfig(n_services=args.services)
        )
        env = ClusterEnv(cfg)
        durations: list[float] = []
        for stats in stream_episode(env, pipeline, args.ticks, seed=args.seed):
            durations.append(stats.duration_ms)
            if not args.quiet:
                print(stats)
        durations.sort()
        mean = sum(durations) / len(durations)
        p99 = durations[min(len(durations) - 1, int(0.99 * len(durations)))]
        print(
            f"\n{len(durations)} ticks ingested into run {args.run_id!r}: "
            f"mean {mean:.2f} ms, p99 {p99:.2f} ms"
        )
        print(f"graph now holds {pipeline.counts()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
