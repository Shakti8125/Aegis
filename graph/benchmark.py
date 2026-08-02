"""Per-tick sync-latency harness for the knowledge graph.

Beyond PLAN.md section 2's file list, on purpose: Phase 2 is done when "the graph
stays in sync with simulator state **at low latency**", and that is a claim about
a number.  This script produces the number so it can be re-checked after any
change to graph/ingestion_pipeline.py, rather than asserted once and forgotten.

Method (each of these changes the answer, so each is stated):

* **warm-up ticks discarded** - they pay for the first bolt handshake, Cypher
  plan compilation for both tick statements, and Neo4j's page cache warming.
* **fixed seed**, so the same episode is replayed by every variant.
* **best of N repeats**, the same protocol simulator/benchmark.py uses, and here
  it is not optional: a Neo4j container on a desktop is noisy enough that the
  cost of an *empty* write transaction was observed to swing between 2.2 ms and
  8.5 ms across consecutive runs. Every repeat's mean is printed, so the spread
  stays visible instead of being hidden behind the winner.
* ``time.perf_counter`` around the whole write transaction: parameter
  marshalling, the round trip and the commit. That is what a caller waits for.
* two action policies, because they stress different halves of the pipeline:

  - ``random`` - uniform over all six actions, so ``scale_up``/``scale_down``
    fire constantly, pods appear and disappear, and almost every tick is a
    *structural* write plus a stale-pod sweep. This is the pessimistic case.
  - ``noop`` - no actions; the pod set only changes when a fault kills a pod, so
    most ticks are property-only writes and the structure cache holds. This is
    the steady-state case a trained Phase 4 policy looks more like.

* a **floor** measurement - an empty write transaction and a one-row write -
  so the reported latency can be read as "round trip + commit + real work"
  rather than as an opaque total.

Scope note: at ~15 ms/tick this pipeline sustains ~65-70 ticks/sec, while the
Phase 1 simulator runs several thousand ticks/sec. That gap is fine and intended -
the graph is the *live state* surface for the dashboard, the Phase 3 encoder and
the Phase 5 narrator, not something to run inside the Phase 4 training loop.
Training reads the simulator's array state directly. If Phase 3 ever does want a
graph per training step, the answer is to read snapshots straight from the env,
not to make Neo4j keep up.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.connection import Neo4jSettings, neo4j_driver  # noqa: E402
from graph.ingestion_pipeline import GraphIngestionPipeline  # noqa: E402
from simulator.cluster_env import N_ACTIONS, ClusterConfig, ClusterEnv  # noqa: E402

BENCH_RUN_ID = "benchmark"


@dataclass
class Latencies:
    policy: str
    samples: list[float]
    structural_ticks: int

    def pct(self, q: float) -> float:
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(q * len(ordered)))
        return ordered[index]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples)

    @property
    def ticks_per_sec(self) -> float:
        return 1000.0 / self.mean


def _measure(
    pipeline: GraphIngestionPipeline,
    cfg: ClusterConfig,
    policy: str,
    ticks: int,
    warmup: int,
    seed: int,
) -> Latencies:
    env = ClusterEnv(cfg)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    pipeline.clear_run()
    pipeline.resync()

    samples: list[float] = []
    structural = 0
    for i in range(warmup + ticks):
        stats = pipeline.ingest(env.graph_snapshot())
        if i >= warmup:
            samples.append(stats.duration_ms)
            structural += int(stats.structure_written)
        if not env.agents:
            env.reset()
            pipeline.resync()
        if policy == "random":
            actions = rng.integers(0, N_ACTIONS, len(env.agents))
        else:
            actions = np.zeros(len(env.agents), dtype=int)
        env.step({a: int(x) for a, x in zip(env.agents, actions)})
    return Latencies(policy=policy, samples=samples, structural_ticks=structural)


def _floor(pipeline: GraphIngestionPipeline, repeats: int = 60) -> tuple[float, float]:
    """Empty-transaction and one-property-write cost, as a latency floor."""
    driver, database = pipeline.driver, pipeline.database

    def timeit(fn) -> float:
        for _ in range(5):
            fn()
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            times.append((time.perf_counter() - started) * 1000.0)
        return statistics.median(times)

    with driver.session(database=database) as session:
        empty = timeit(lambda: session.execute_write(lambda tx: None))
        one = timeit(
            lambda: session.execute_write(
                lambda tx: tx.run(
                    "MERGE (n:Service {run_id: $r, id: '__floor__'}) SET n.tick = $t",
                    r=BENCH_RUN_ID,
                    t=int(time.time()),
                ).consume()
            )
        )
    return empty, one


def _best_of(
    pipeline: GraphIngestionPipeline,
    cfg: ClusterConfig,
    policy: str,
    ticks: int,
    warmup: int,
    seed: int,
    repeats: int,
) -> tuple[Latencies, list[float]]:
    runs = [
        _measure(pipeline, cfg, policy, ticks, warmup, seed) for _ in range(repeats)
    ]
    return min(runs, key=lambda r: r.mean), [r.mean for r in runs]


def _report(result: Latencies, ticks: int, all_means: list[float]) -> None:
    print(f"-- {result.policy} actions ------------------------------------------------")
    print(
        "  repeat means           : "
        + ", ".join(f"{m:.2f}" for m in all_means)
        + " ms  (best reported below)"
    )
    print(f"  mean                   : {result.mean:12.2f} ms")
    print(f"  p50                    : {result.pct(0.50):12.2f} ms")
    print(f"  p95                    : {result.pct(0.95):12.2f} ms")
    print(f"  p99                    : {result.pct(0.99):12.2f} ms")
    print(f"  max                    : {max(result.samples):12.2f} ms")
    print(f"  min                    : {min(result.samples):12.2f} ms")
    print(f"  sustained rate         : {result.ticks_per_sec:12.1f} ticks/sec")
    print(
        f"  structural ticks       : {result.structural_ticks:12d} "
        f"of {ticks} ({100.0 * result.structural_ticks / ticks:.0f}% wrote topology)"
    )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=300, help="timed ticks per policy")
    parser.add_argument("--warmup", type=int, default=20, help="discarded ticks")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per policy")
    parser.add_argument("--seed", type=int, default=20240202)
    parser.add_argument("--services", type=int, default=None)
    parser.add_argument("--run-id", default=BENCH_RUN_ID)
    args = parser.parse_args(argv)

    cfg = (
        ClusterConfig()
        if args.services is None
        else ClusterConfig(n_services=args.services)
    )
    settings = Neo4jSettings.from_env()

    with neo4j_driver(settings) as driver:
        version = driver.execute_query(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name + ' ' + versions[0] + ' ' + edition AS v",
            database_=settings.database,
        ).records[0]["v"]
        pipeline = GraphIngestionPipeline(
            driver, args.run_id, database=settings.database
        )

        print("=" * 74)
        print("Aegis knowledge-graph ingestion latency")
        print("=" * 74)
        print(f"  machine        : {platform.platform()}")
        print(f"  neo4j          : {version} at {settings.uri}")
        print(
            f"  cluster        : {cfg.n_services} services, {cfg.n_nodes} nodes, "
            f"<= {cfg.n_services * cfg.max_replicas} pod slots"
        )
        print(
            f"  protocol       : {args.warmup} warm-up ticks discarded, best of "
            f"{args.repeats} x {args.ticks} timed ticks, seed {args.seed}"
        )
        print("  timed          : one ingest() = one Cypher statement = one commit")
        print()

        try:
            results = [
                _best_of(
                    pipeline,
                    cfg,
                    policy,
                    args.ticks,
                    args.warmup,
                    args.seed,
                    args.repeats,
                )
                for policy in ("random", "noop")
            ]
            for result, means in results:
                _report(result, args.ticks, means)

            empty, one = _floor(pipeline)
            print("-- latency floor on this machine -------------------------------------")
            print(f"  empty write transaction: {empty:12.2f} ms  (round trip + commit)")
            print(f"  one-property write     : {one:12.2f} ms")
            print()
            pessimistic = results[0][0]
            print(
                f"  So of the {pessimistic.mean:.1f} ms mean above, ~{empty:.1f} ms is "
                f"the wire and the commit;\n  the rest is the ~500 property writes and "
                f"the topology churn a tick actually carries."
            )
        finally:
            deleted = pipeline.clear_run()
            print(f"\n  cleaned up: {deleted} nodes removed from run {args.run_id!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
