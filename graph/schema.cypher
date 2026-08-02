// Aegis knowledge-graph schema. Phase 2 - owned by the graph-engineer subagent.
// See PLAN.md section 3, Phase 2.
//
// (:Service)-[:DEPENDS_ON]->(:Service)
// (:Pod)-[:INSTANCE_OF]->(:Service)
// (:Pod)-[:RUNS_ON]->(:Node)
// (:Service)-[:CALLS {p99_latency_ms, error_rate}]->(:Service)
//
// ---------------------------------------------------------------------------
// THIS FILE IS DOCUMENTATION, NOT THE DEPLOY PATH.
// It is the current state of the schema, kept as one readable page. The thing
// that actually runs against a database is the numbered set in
// graph/migrations/, applied by `python -m graph.migrate --apply`
// (CLAUDE.md: "Cypher migrations are numbered files under graph/migrations/").
// Every statement below is byte-equivalent to a statement in some migration;
// tests/graph/test_migrations.py asserts the two never drift apart.
// ---------------------------------------------------------------------------
//
// Node identity
// -------------
// Key is the pair (run_id, id).
//   id     - the simulator's own name: `svc-03-mid`, `pod-03-1`, `node-4`.
//            Comes straight from ClusterEnv.graph_snapshot(); never invented here.
//   run_id - namespaces one simulation run. Neo4j Community exposes a single
//            user database, so the dev stream, pytest and any Phase 4 eval sweep
//            share it. Without run_id they would collide on `id`.
//
// Node properties (PLAN.md: updated every simulation tick)
// -------------------------------------------------------
//   all labels  : health, cpu_pct, mem_pct, restart_count, tick
//   :Service    : + name, tier, replicas, ready_replicas, p99_latency_ms,
//                   error_rate, isolated, sla_violating
//   :Pod        : + name, status  (Running | NotReady | Restarting | CrashLoopBackOff)
//   :Node       : + name, pod_count, pod_capacity
// Ingestion copies whatever graph_snapshot() emits (`SET n += row`), so a new
// simulator property lands in the graph without a schema change. The properties
// above are what the Phase 1 simulator emits today.
//
// Relationship properties
// -----------------------
//   :CALLS      : p99_latency_ms, error_rate, traffic_share, tick
//   :DEPENDS_ON, :INSTANCE_OF, :RUNS_ON : tick only (pure structure)
//
// Constraints
// -----------
// Composite uniqueness, one per label. Each is backed by a composite range index
// that MERGE seeks on, and whose run_id prefix also serves the per-tick stale-pod
// sweep `MATCH (p:Pod {run_id: $run_id}) WHERE NOT p.id IN $ids_Pod`. Verified
// against the live dev database (Neo4j Kernel 5.26.28 Community): every
// statement graph/ingestion_pipeline.py sends per tick plans as a
// NodeUniqueIndexSeek, none as a NodeByLabelScan, and
// tests/graph/test_ingestion.py::test_every_ingestion_fragment_uses_an_index
// EXPLAINs each one to keep it that way.

CREATE CONSTRAINT aegis_service_key IF NOT EXISTS
FOR (s:Service) REQUIRE (s.run_id, s.id) IS UNIQUE;

CREATE CONSTRAINT aegis_pod_key IF NOT EXISTS
FOR (p:Pod) REQUIRE (p.run_id, p.id) IS UNIQUE;

CREATE CONSTRAINT aegis_node_key IF NOT EXISTS
FOR (n:Node) REQUIRE (n.run_id, n.id) IS UNIQUE;

// Indexes
// -------
// None beyond the three constraint-backed composite indexes above. Every lookup
// graph/ingestion_pipeline.py performs per tick is an equality or prefix match
// on (run_id, id), so an extra index would cost write throughput on a path that
// writes ~500 properties per tick and buy nothing.
//
// One case needed work rather than an index: the stale-edge sweep. Written the
// obvious way it planned as a label scan over every Service/Pod in the database
// - all runs, not just this one - and Neo4j rejects an index hint in that
// context. The fix was to anchor the sweep by UNWINDing the source ids the
// pipeline already holds, which seeks the existing index. Adding a
// single-property run_id index would have been the other way to fix it; the
// query rewrite costs nothing on the write path, so it won.
//
// Add an index here - as a new numbered migration - when a real query needs it.
//
// Not modelled, on purpose
// ------------------------
// graph_snapshot() also carries `active_faults`. Fault provenance is genuinely
// useful to Phase 5's narrator, but PLAN.md section 3 Phase 2 specifies exactly
// the four patterns above and nothing else, so faults stay out of the graph
// until a phase actually needs them - at which point they arrive as a numbered
// migration.
//
// Bookkeeping label
// -----------------
// `(:_AegisMigration {version, name, checksum, applied_at})` records which
// migrations have run. Its uniqueness constraint on `version` is created by the
// runner itself (graph/migrate.py) before the first migration is applied - it
// cannot be a migration without a chicken-and-egg problem. The leading
// underscore marks it as infrastructure, not part of the domain model.
