// 001_initial_schema
// Phase 2 initial schema: node identity for the four PLAN.md section 3 patterns.
//
//   (:Service)-[:DEPENDS_ON]->(:Service)
//   (:Pod)-[:INSTANCE_OF]->(:Service)
//   (:Pod)-[:RUNS_ON]->(:Node)
//   (:Service)-[:CALLS {p99_latency_ms, error_rate}]->(:Service)
//
// A node's key is (run_id, id):
//   * `id` is the simulator's own name - `svc-03-mid`, `pod-03-1`, `node-4`.
//   * `run_id` namespaces one simulation run. Neo4j Community has a single user
//     database, so concurrent runs (dev stream, pytest, a Phase 4 eval sweep)
//     share it; without run_id they would collide on `id` and MERGE would fuse
//     two different clusters into one graph.
//
// Each constraint is backed by a composite range index, which also serves the
// prefix predicate `(:Pod {run_id: $run_id})` that the per-tick stale-pod sweep
// uses - verified with EXPLAIN, plan is NodeUniqueIndexSeek, not a label scan.
// That is why this file adds no further indexes: ingestion has no per-tick
// lookup these three do not already cover.
//
// Property existence constraints (`REQUIRE n.run_id IS NOT NULL`) would be the
// natural companion here but are Enterprise-only; graph/ingestion_pipeline.py
// always writes both key properties in the MERGE pattern instead.

CREATE CONSTRAINT aegis_service_key IF NOT EXISTS
FOR (s:Service) REQUIRE (s.run_id, s.id) IS UNIQUE;

CREATE CONSTRAINT aegis_pod_key IF NOT EXISTS
FOR (p:Pod) REQUIRE (p.run_id, p.id) IS UNIQUE;

CREATE CONSTRAINT aegis_node_key IF NOT EXISTS
FOR (n:Node) REQUIRE (n.run_id, n.id) IS UNIQUE;
