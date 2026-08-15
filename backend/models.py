"""Pydantic models for the backend API and WebSocket payloads.

Phase 6 — owned by the main session. See PLAN.md section 3.

Every model here is a serialisation contract between the backend and the
frontend. WebSocket frames use the ``WsFrame`` envelope (type-tagged JSON);
REST responses use the specific model directly.

Naming convention: ``*Snapshot`` for point-in-time state,
``*Event`` for something that happened, ``*Summary`` for aggregates.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ==========================================================================
# WebSocket commands
# ==========================================================================
class WsStartCommand(BaseModel):
    command: str
    scenario: str = "mixed"
    seed: int = 42
    max_cycles: int = 200
    tick_delay_ms: int = 100

    @field_validator("max_cycles", mode="before")
    @classmethod
    def clamp_max_cycles(cls, v: Any) -> int:
        try:
            return max(1, min(500, int(v)))
        except (ValueError, TypeError):
            return 200

    @field_validator("tick_delay_ms", mode="before")
    @classmethod
    def clamp_tick_delay(cls, v: Any) -> int:
        try:
            return max(20, min(2000, int(v)))
        except (ValueError, TypeError):
            return 100


# ==========================================================================
# Enums
# ==========================================================================
class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

    @classmethod
    def from_value(cls, health: float) -> "HealthStatus":
        if health >= 0.85:
            return cls.HEALTHY
        if health >= 0.40:
            return cls.DEGRADED
        return cls.CRITICAL


class ActionName(str, Enum):
    NO_OP = "no-op"
    RESTART = "restart"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ISOLATE = "isolate"
    REROUTE = "reroute"


class WsFrameType(str, Enum):
    """Discriminator for WebSocket JSON frames."""
    TICK = "tick"           # Full cluster state snapshot
    ACTION = "action"       # Agent action + narration
    VETO = "veto"           # Safety supervisor veto
    EPISODE_END = "episode_end"
    ERROR = "error"
    CONNECTED = "connected"


# ==========================================================================
# Cluster state snapshots (per-tick)
# ==========================================================================
class ServiceSnapshot(BaseModel):
    """One service at a single tick."""
    id: str
    name: str = ""
    tier: str = ""
    health: float
    status: HealthStatus
    cpu_pct: float
    mem_pct: float
    p99_latency_ms: float
    error_rate: float
    replicas: int
    ready_replicas: int
    isolated: bool = False
    sla_violating: bool = False


class NodeSnapshot(BaseModel):
    """One cluster node at a single tick."""
    id: str
    name: str = ""
    cpu_pct: float
    mem_pct: float
    pod_count: int
    pod_capacity: int
    health: float


class EdgeSnapshot(BaseModel):
    """One service-to-service call edge."""
    source: str
    target: str
    relation: str = "CALLS"
    p99_latency_ms: float | None = None
    error_rate: float | None = None
    traffic_share: float | None = None


class FaultSnapshot(BaseModel):
    """An active fault at a given tick."""
    fault_type: str
    target: str
    tick_start: int = 0
    duration: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ClusterSnapshot(BaseModel):
    """Complete cluster state at one tick — the main WebSocket payload."""
    tick: int
    services: list[ServiceSnapshot]
    nodes: list[NodeSnapshot]
    edges: list[EdgeSnapshot]
    active_faults: list[FaultSnapshot] = Field(default_factory=list)
    sla_violation_rate: float = 0.0
    mean_health: float = 1.0
    min_health: float = 1.0


# ==========================================================================
# Action and narration events
# ==========================================================================
class ActionEvent(BaseModel):
    """One agent action at a tick, with its narration and veto status."""
    tick: int
    agent_id: str
    action: str
    target_service: str
    narration: str = ""
    was_vetoed: bool = False
    veto_reason: str = ""
    veto_policy: str = ""
    cited_edge_source: str | None = None
    cited_edge_target: str | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)



# ==========================================================================
# WebSocket frame envelope
# ==========================================================================
class WsFrame(BaseModel):
    """Type-tagged envelope for all WebSocket messages.

    The frontend dispatches on ``type`` and reads the payload from the
    matching field.
    """
    type: WsFrameType
    tick: int | None = None
    cluster: ClusterSnapshot | None = None
    actions: list[ActionEvent] | None = None
    message: str | None = None
    episode_summary: "EpisodeSummary | None" = None


# ==========================================================================
# REST: episode replay
# ==========================================================================
class TickRecord(BaseModel):
    """One tick in an episode replay."""
    tick: int
    cluster: ClusterSnapshot
    actions: list[ActionEvent] = Field(default_factory=list)


class EpisodeSummary(BaseModel):
    """Summary of a completed episode."""
    episode_id: str
    seed: int = 0
    scenario: str = ""
    length: int = 0
    recovered: bool = False
    collapsed: bool = False
    terminal_reason: str = ""
    ttr: float = 0.0
    sla_service_ticks: int = 0
    mean_health: float = 0.0
    total_reward: float = 0.0
    action_counts: dict[str, int] = Field(default_factory=dict)
    reward_components: dict[str, float] = Field(default_factory=dict)


class EpisodeReplay(BaseModel):
    """Full replay of a completed episode (REST response)."""
    summary: EpisodeSummary
    ticks: list[TickRecord] = Field(default_factory=list)


# ==========================================================================
# REST: training metrics
# ==========================================================================
class TrainingMetricPoint(BaseModel):
    """One data point in a training curve."""
    update: int
    env_steps: int
    wall_clock_s: float = 0.0
    reward_total: float = 0.0
    reward_components: dict[str, float] = Field(default_factory=dict)
    loss_value: float = 0.0
    loss_policy: float = 0.0
    loss_entropy: float = 0.0
    explained_variance: float = 0.0
    episode_length: float = 0.0
    recovery_rate: float = 0.0
    sla_service_ticks: float = 0.0


class TrainingRun(BaseModel):
    """Metadata for one training run."""
    run_id: str
    seed: int = 0
    total_updates: int = 0
    total_env_steps: int = 0
    device: str = "cpu"
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingMetrics(BaseModel):
    """Training curves for the dashboard (REST response)."""
    run: TrainingRun
    metrics: list[TrainingMetricPoint] = Field(default_factory=list)


# ==========================================================================
# REST: health check
# ==========================================================================
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    active_connections: int = 0


# Forward reference resolution
WsFrame.model_rebuild()
