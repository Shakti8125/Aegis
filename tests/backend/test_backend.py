"""Tests for backend/ — models, REST endpoints, and WebSocket stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
from backend.models import (
    ActionEvent,
    ClusterSnapshot,
    EdgeSnapshot,
    EpisodeSummary,
    FaultSnapshot,
    HealthResponse,
    HealthStatus,
    NodeSnapshot,
    ServiceSnapshot,
    TrainingMetricPoint,
    TrainingMetrics,
    TrainingRun,
    WsFrame,
    WsFrameType,
)


class TestHealthStatus:
    def test_healthy(self):
        assert HealthStatus.from_value(1.0) is HealthStatus.HEALTHY
        assert HealthStatus.from_value(0.85) is HealthStatus.HEALTHY

    def test_degraded(self):
        assert HealthStatus.from_value(0.6) is HealthStatus.DEGRADED
        assert HealthStatus.from_value(0.40) is HealthStatus.DEGRADED

    def test_critical(self):
        assert HealthStatus.from_value(0.39) is HealthStatus.CRITICAL
        assert HealthStatus.from_value(0.0) is HealthStatus.CRITICAL


class TestServiceSnapshot:
    def test_serialisation_roundtrip(self):
        s = ServiceSnapshot(
            id="svc-03",
            health=0.45,
            status=HealthStatus.DEGRADED,
            cpu_pct=0.82,
            mem_pct=0.6,
            p99_latency_ms=312.5,
            error_rate=0.08,
            replicas=2,
            ready_replicas=1,
        )
        d = s.model_dump()
        assert d["id"] == "svc-03"
        assert d["status"] == "degraded"
        s2 = ServiceSnapshot.model_validate(d)
        assert s2.health == 0.45


class TestClusterSnapshot:
    def test_full_snapshot(self):
        snap = ClusterSnapshot(
            tick=42,
            services=[
                ServiceSnapshot(
                    id="svc-00", health=1.0, status=HealthStatus.HEALTHY,
                    cpu_pct=0.1, mem_pct=0.2, p99_latency_ms=10.0,
                    error_rate=0.0, replicas=2, ready_replicas=2,
                ),
            ],
            nodes=[
                NodeSnapshot(id="node-0", cpu_pct=0.3, mem_pct=0.4,
                             pod_count=4, pod_capacity=8, health=0.9),
            ],
            edges=[
                EdgeSnapshot(source="svc-00", target="svc-01", p99_latency_ms=15.0),
            ],
        )
        data = json.loads(snap.model_dump_json())
        assert data["tick"] == 42
        assert len(data["services"]) == 1
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1


class TestWsFrame:
    def test_tick_frame(self):
        frame = WsFrame(
            type=WsFrameType.TICK,
            tick=10,
            cluster=ClusterSnapshot(tick=10, services=[], nodes=[], edges=[]),
            actions=[],
        )
        data = json.loads(frame.model_dump_json())
        assert data["type"] == "tick"
        assert data["tick"] == 10

    def test_connected_frame(self):
        frame = WsFrame(type=WsFrameType.CONNECTED, message="Hello")
        data = json.loads(frame.model_dump_json())
        assert data["type"] == "connected"
        assert data["message"] == "Hello"

    def test_episode_end_frame(self):
        summary = EpisodeSummary(
            episode_id="test-1", seed=42, scenario="mixed",
            length=100, recovered=True, terminal_reason="recovered",
        )
        frame = WsFrame(
            type=WsFrameType.EPISODE_END,
            tick=100,
            episode_summary=summary,
        )
        data = json.loads(frame.model_dump_json())
        assert data["type"] == "episode_end"
        assert data["episode_summary"]["recovered"] is True


class TestActionEvent:
    def test_action_event(self):
        e = ActionEvent(
            tick=42, agent_id="service_3", action="restart",
            target_service="svc-03", narration="Restarting pod.",
            was_vetoed=False,
        )
        d = e.model_dump()
        assert d["action"] == "restart"
        assert d["narration"] == "Restarting pod."

    def test_vetoed_action_event(self):
        e = ActionEvent(
            tick=42, agent_id="service_0", action="isolate",
            target_service="svc-00", was_vetoed=True,
            veto_reason="Protected service", veto_policy="protected_service",
        )
        assert e.was_vetoed is True
        assert e.veto_policy == "protected_service"


class TestTrainingMetrics:
    def test_training_metrics_model(self):
        tm = TrainingMetrics(
            run=TrainingRun(run_id="test-run", total_updates=100),
            metrics=[
                TrainingMetricPoint(
                    update=1, env_steps=1024, reward_total=-0.5,
                    loss_value=0.3, loss_policy=-0.01,
                ),
            ],
        )
        data = json.loads(tm.model_dump_json())
        assert data["run"]["run_id"] == "test-run"
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["update"] == 1


# ---------------------------------------------------------------------------
# REST endpoints (requires fastapi + httpx)
# ---------------------------------------------------------------------------
try:
    from httpx import ASGITransport, AsyncClient
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")
class TestRestEndpoints:
    @pytest.fixture
    def client(self):
        from backend.main import app
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.anyio
    async def test_health(self, client):
        async with client:
            r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.anyio
    async def test_scenarios(self, client):
        async with client:
            r = await client.get("/api/scenarios")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "mixed" in names
        assert "pod_crash" in names

    @pytest.mark.anyio
    async def test_runs_empty(self, client):
        async with client:
            r = await client.get("/api/runs")
        assert r.status_code == 200
        # May or may not have runs, but should be a list
        assert isinstance(r.json(), list)

    @pytest.mark.anyio
    async def test_metrics_no_runs(self, client):
        async with client:
            r = await client.get("/api/metrics")
        assert r.status_code == 200
        # Returns null if no runs exist
        # (unless there are actual checkpoint dirs)

    @pytest.mark.anyio
    async def test_metrics_missing_run(self, client):
        async with client:
            r = await client.get("/api/metrics/nonexistent-run")
        assert r.status_code == 200
        assert r.json() is None


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
from backend.ws import ConnectionManager, _extract_cluster_snapshot, SimulationRunner


class TestConnectionManager:
    def test_initial_count(self):
        mgr = ConnectionManager()
        assert mgr.count == 0


class TestExtractClusterSnapshot:
    def test_snapshot_from_env(self):
        """Verify we can extract a snapshot from a real ClusterEnv."""
        from simulator.cluster_env import ClusterEnv
        env = ClusterEnv()
        env.reset(seed=42)
        snap = _extract_cluster_snapshot(env)
        assert snap.tick == 0
        assert len(snap.services) == 12
        assert len(snap.nodes) == 6
        assert len(snap.edges) > 0
        assert 0.0 <= snap.mean_health <= 1.0

    def test_snapshot_after_steps(self):
        from simulator.cluster_env import ClusterEnv, ACTION_NOOP
        env = ClusterEnv()
        obs, _ = env.reset(seed=123)
        for _ in range(5):
            actions = {a: ACTION_NOOP for a in env.agents}
            env.step(actions)
        snap = _extract_cluster_snapshot(env)
        assert snap.tick == 5
        # Check service fields are valid
        for svc in snap.services:
            assert 0.0 <= svc.health <= 1.0
            assert svc.replicas >= 0

    def test_snapshot_json_serialisable(self):
        from simulator.cluster_env import ClusterEnv
        env = ClusterEnv()
        env.reset(seed=42)
        snap = _extract_cluster_snapshot(env)
        data = json.loads(snap.model_dump_json())
        assert isinstance(data["services"], list)
        assert isinstance(data["nodes"], list)


class TestSimulationRunner:
    @pytest.mark.anyio
    async def test_runner_produces_frames(self):
        runner = SimulationRunner(
            scenario="pod_crash", seed=42, max_cycles=10, tick_delay_ms=0,
        )
        frames = []
        async for frame in runner.run():
            frames.append(frame)
        assert len(frames) > 0
        # Last frame should be episode_end
        assert frames[-1].type == WsFrameType.EPISODE_END
        # All other frames should be ticks
        for f in frames[:-1]:
            assert f.type == WsFrameType.TICK
            assert f.cluster is not None
            assert f.actions is not None

    @pytest.mark.anyio
    async def test_runner_includes_narrations(self):
        runner = SimulationRunner(
            scenario="pod_crash", seed=42, max_cycles=5, tick_delay_ms=0,
        )
        has_narration = False
        async for frame in runner.run():
            if frame.actions:
                for a in frame.actions:
                    if a.narration:
                        has_narration = True
                        break
        assert has_narration, "Expected at least one narrated action"
