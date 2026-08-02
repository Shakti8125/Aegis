"""Sync latency, measured against the live database.

PLAN.md Phase 2 is done when "the graph stays in sync with simulator state **at
low latency**". These are regression guards, not the benchmark - `graph/benchmark.py`
is the harness that produces the reported numbers, with warm-up, percentiles, a
best-of-N protocol and a floor measurement.

The guard asserts on the **median**, not the mean or a tail percentile. Neo4j in
a desktop container stalls occasionally - a single 1000 ms tick was observed
during development, presumably a transaction-log checkpoint - and over the ~50
samples a test can afford, one stall moves the mean by 20 ms and *is* the p99.
Asserting on those would make the suite fail for reasons that have nothing to do
with this code. The median at a loose bound catches what actually matters here:
a change that quietly turns one batched round trip per tick back into many.
``test_a_tick_is_one_round_trip`` covers the same regression structurally, and
without any timing at all.
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from tests.graph.conftest import step_random

WARMUP_TICKS = 8
TIMED_TICKS = 50

#: Measured on the dev container (Neo4j 5.26 community in Docker, Windows 11):
#: median 12-13 ms per tick with uniformly random actions, ~1.5 ms of which is
#: the round trip and commit. The bound is ~10x that because the same benchmark
#: read 2x slower on a loaded machine - the floor moved with it, so that is the
#: host, not this code. See graph/benchmark.py for the method.
MAX_MEDIAN_MS = 150.0


@pytest.fixture
def latency_samples(pipeline, env):
    rng = np.random.default_rng(31337)
    for _ in range(WARMUP_TICKS):  # bolt handshake, plan compilation, page cache
        pipeline.ingest(env.graph_snapshot())
        step_random(env, rng)
    samples = []
    for _ in range(TIMED_TICKS):
        samples.append(pipeline.ingest(env.graph_snapshot()).duration_ms)
        step_random(env, rng)
    return samples


def test_per_tick_latency_is_low(latency_samples):
    ordered = sorted(latency_samples)
    median = statistics.median(ordered)
    print(
        f"\ningest latency over {len(ordered)} ticks: "
        f"median {median:.2f} ms, mean {statistics.mean(ordered):.2f} ms, "
        f"p95 {ordered[int(0.95 * len(ordered))]:.2f} ms, max {ordered[-1]:.2f} ms"
    )
    assert median < MAX_MEDIAN_MS, f"median {median:.1f} ms exceeds {MAX_MEDIAN_MS} ms"


def test_a_tick_is_one_round_trip(pipeline, env):
    """The optimisation that bought most of the latency: one statement per tick.

    Twelve separate statements in one transaction measured 42.7 ms/tick against
    11.9 ms for the same writes composed into a single statement, because a round
    trip to the dev container costs ~2.5 ms. If someone splits these back up, the
    latency thresholds above are loose enough to miss it - this is not.
    """
    stats = pipeline.ingest(env.graph_snapshot())
    assert stats.subqueries == 12, "structural tick: 3 labels + pod sweep + 4 rels + 4 prunes"
    again = pipeline.ingest(env.graph_snapshot())
    assert again.subqueries == 5, "unchanged topology: 3 labels + pod sweep + CALLS"
    assert again.structure_written is False


def test_ingest_reports_its_own_duration(pipeline, env):
    stats = pipeline.ingest(env.graph_snapshot())
    assert stats.duration_ms > 0.0
    assert stats.tick == env.graph_snapshot()["tick"]
    assert stats.run_id == pipeline.run_id
    assert stats.rows["Service"] == env.n_services
