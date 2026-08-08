"""/ws/live — WebSocket stream of cluster state, actions, and narratives.

Phase 6 — owned by the main session. See PLAN.md section 3.

Architecture
------------
:class:`ConnectionManager` tracks every active WebSocket and broadcasts
typed JSON frames (:class:`WsFrame`) to all of them. The simulation loop
runs in a background ``asyncio.Task`` (one per active simulation) and
pushes frames through the manager.

:class:`SimulationRunner` wraps a ``ClusterEnv`` + a controller + the
Phase 5 ops layer (narrator + safety supervisor) into a single async
generator that yields one ``WsFrame`` per tick. The ``/ws/live`` endpoint
consumes this generator and forwards every frame to all connected clients.

Flow::

    Client connects → /ws/live
    Client sends: {"command": "start", "scenario": "mixed", "seed": 42}
    Server: starts SimulationRunner, broadcasts tick frames
    Client sends: {"command": "stop"}  |  disconnect
    Server: cancels the runner
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Sequence

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from simulator.cluster_env import (
    ACTION_NAMES,
    N_ACTIONS,
    ClusterConfig,
    ClusterEnv,
)
from simulator.fault_injection import FaultType
from marl.baseline import BaselineConfig, RuleBasedController
from marl.reward import RewardConfig, RewardShaper
from marl.vec_env import scenario_overrides

from ops_layer.narrator import (
    ActionContext,
    DependencyEdge,
    Narrator,
    ServiceSnapshot as NarratorServiceSnapshot,
)
from ops_layer.safety_supervisor import SafetySupervisor, VetoDecision

from backend.models import (
    ActionEvent,
    ClusterSnapshot,
    EdgeSnapshot,
    EpisodeSummary,
    FaultSnapshot,
    HealthStatus,
    NodeSnapshot,
    ServiceSnapshot,
    WsFrame,
    WsFrameType,
)

logger = logging.getLogger(__name__)


# ==========================================================================
# Connection manager
# ==========================================================================
class ConnectionManager:
    """Tracks active WebSocket connections and broadcasts frames."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    @property
    def count(self) -> int:
        return len(self._connections)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket connected (%d active)", self.count)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket disconnected (%d active)", self.count)

    async def broadcast(self, frame: WsFrame) -> None:
        """Send a frame to every connected client, dropping broken ones."""
        data = frame.model_dump_json()
        broken: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                broken.append(ws)
        for ws in broken:
            self.disconnect(ws)

    async def send(self, ws: WebSocket, frame: WsFrame) -> None:
        """Send a frame to a single client."""
        await ws.send_text(frame.model_dump_json())


# ==========================================================================
# Env → model converters
# ==========================================================================
def _extract_cluster_snapshot(env: ClusterEnv) -> ClusterSnapshot:
    """Read the public state arrays of a ``ClusterEnv`` into a model."""
    topo = env.topology
    cfg = env.cfg
    tier_names = [
        "front" if i == 0 else ("back" if i == cfg.n_tiers - 1 else f"mid-{i}")
        for i in range(cfg.n_tiers)
    ]

    services: list[ServiceSnapshot] = []
    for i in range(env.n_services):
        h = float(env.svc_health[i])
        services.append(ServiceSnapshot(
            id=f"svc-{i:02d}",
            name=topo.service_names[i] if hasattr(topo, "service_names") else f"svc-{i:02d}",
            tier=tier_names[int(topo.service_tier[i])] if topo.service_tier is not None else "",
            health=h,
            status=HealthStatus.from_value(h),
            cpu_pct=float(env.svc_util[i]),
            mem_pct=float(env.pod_mem[i, :int(env.svc_replicas[i])].mean())
            if int(env.svc_replicas[i]) > 0
            else 0.0,
            p99_latency_ms=float(env.svc_latency[i]),
            error_rate=float(env.svc_error[i]),
            replicas=int(env.svc_replicas[i]),
            ready_replicas=int(env.svc_ready[i]),
            isolated=bool(env.svc_isolate_timer[i] > 0),
            sla_violating=bool(env._sla_violating[i]),
        ))

    nodes: list[NodeSnapshot] = []
    for j in range(env.n_nodes):
        pod_mask = env._pod_node_flat == j
        pod_count = int(env.pod_alive.reshape(-1)[pod_mask].sum()) if pod_mask.any() else 0
        node_h = max(0.0, 1.0 - float(env.node_cpu[j]) - float(env.node_cpu_pressure[j]) * 0.5)
        nodes.append(NodeSnapshot(
            id=f"node-{j}",
            name=f"node-{j}",
            cpu_pct=float(env.node_cpu[j]),
            mem_pct=float(env.node_mem[j]),
            pod_count=pod_count,
            pod_capacity=int(topo.node_pod_capacity),
            health=node_h,
        ))

    edges: list[EdgeSnapshot] = []
    for src, dst in env._calls_pairs:
        edges.append(EdgeSnapshot(
            source=f"svc-{src:02d}",
            target=f"svc-{dst:02d}",
            relation="CALLS",
            p99_latency_ms=float(env.svc_latency[dst]),
            error_rate=float(env.svc_error[dst]),
            traffic_share=float(env.call_weight[src, dst]),
        ))

    active_faults: list[FaultSnapshot] = []
    for fault in env.active_faults:
        target_str = f"svc-{fault.target[0]:02d}" if isinstance(fault.target, tuple) else str(fault.target)
        ft_str = fault.fault_type.name.lower() if hasattr(fault.fault_type, "name") else str(fault.fault_type)
        active_faults.append(FaultSnapshot(
            fault_type=ft_str,
            target=target_str,
            tick_start=int(fault.tick),
            duration=int(fault.duration),
        ))

    return ClusterSnapshot(
        tick=env.t,
        services=services,
        nodes=nodes,
        edges=edges,
        active_faults=active_faults,
        sla_violation_rate=float(env.sla_violation_rate),
        mean_health=float(env.svc_health.mean()),
        min_health=float(env.svc_health.min()),
    )


def _build_action_context(
    env: ClusterEnv, agent_idx: int, action: int
) -> ActionContext:
    """Build a narrator ActionContext from the env's live state."""
    topo = env.topology
    cfg = env.cfg
    tier_names = [
        "front" if i == 0 else ("back" if i == cfg.n_tiers - 1 else f"mid-{i}")
        for i in range(cfg.n_tiers)
    ]
    i = agent_idx
    svc = NarratorServiceSnapshot(
        service_id=f"svc-{i:02d}",
        health=float(env.svc_health[i]),
        cpu_pct=float(env.svc_util[i]),
        mem_pct=float(env.pod_mem[i, :max(1, int(env.svc_replicas[i]))].mean()),
        p99_latency_ms=float(env.svc_latency[i]),
        error_rate=float(env.svc_error[i]),
        replicas=int(env.svc_replicas[i]),
        ready_replicas=int(env.svc_ready[i]),
        tier=tier_names[int(topo.service_tier[i])],
        isolated=bool(env.svc_isolate_timer[i] > 0),
        sla_violating=bool(env._sla_violating[i]),
    )

    deps = []
    for _, dst in env._calls_pairs:
        if _ == i:
            deps.append(DependencyEdge(
                source_id=f"svc-{i:02d}",
                target_id=f"svc-{dst:02d}",
                p99_latency_ms=float(env.svc_latency[dst]),
                error_rate=float(env.svc_error[dst]),
            ))

    dependents = []
    for src, _ in env._calls_pairs:
        if _ == i:
            dependents.append(DependencyEdge(
                source_id=f"svc-{src:02d}",
                target_id=f"svc-{i:02d}",
                p99_latency_ms=float(env.svc_latency[i]),
                error_rate=float(env.svc_error[i]),
            ))

    fault_dicts = []
    for f in env.active_faults:
        ft_str = f.fault_type.name.lower() if hasattr(f.fault_type, "name") else str(f.fault_type)
        fault_dicts.append({
            "fault_type": ft_str,
            "target": str(f.target),
            "tick": f.tick,
        })

    return ActionContext(
        tick=env.t,
        agent_id=f"service_{i}",
        action=action,
        target_service=svc,
        dependencies=deps,
        dependents=dependents,
        active_faults=fault_dicts,
    )


# ==========================================================================
# Simulation runner
# ==========================================================================
class SimulationRunner:
    """Drives one episode and yields WsFrames for streaming.

    Composes the simulator, a controller, the narrator, and the safety
    supervisor into a single loop.
    """

    def __init__(
        self,
        scenario: str = "mixed",
        seed: int = 42,
        max_cycles: int = 200,
        tick_delay_ms: int = 100,
        narrator: Narrator | None = None,
        supervisor: SafetySupervisor | None = None,
    ) -> None:
        overrides = scenario_overrides(scenario)
        self.env = ClusterEnv(
            ClusterConfig(max_cycles=max_cycles, **overrides)
        )
        self.seed = seed
        self.scenario = scenario
        self.tick_delay = tick_delay_ms / 1000.0
        self.narrator = narrator or Narrator()
        self.supervisor = supervisor or SafetySupervisor()
        self.controller = RuleBasedController(BaselineConfig())
        self._episode_id = f"{scenario}-{seed}-{int(time.time())}"
        self._tick_records: list[dict] = []

    async def run(self) -> AsyncIterator[WsFrame]:
        """Async generator yielding one frame per tick."""
        observations, infos = self.env.reset(seed=self.seed)
        self.controller.reset(self.env)
        shaper = RewardShaper()

        while self.env.agents:
            # 1. Controller proposes actions
            actions = self.controller.act(observations, infos, self.env)

            # 2. Safety supervisor checks each action
            action_events: list[ActionEvent] = []
            final_actions: dict[str, int] = {}

            for agent_name, action_int in actions.items():
                idx = self.env.agent_name_to_index[agent_name]
                ctx = _build_action_context(self.env, idx, action_int)

                veto_result = self.supervisor.check(ctx)
                if veto_result.vetoed:
                    ctx.was_vetoed = True
                    ctx.veto_reason = veto_result.reason
                    final_actions[agent_name] = 0  # Force no-op
                else:
                    final_actions[agent_name] = action_int

                narration = self.narrator.narrate(ctx)

                action_events.append(ActionEvent(
                    tick=self.env.t + 1,
                    agent_id=agent_name,
                    action=ACTION_NAMES[action_int],
                    target_service=f"svc-{idx:02d}",
                    narration=narration.text,
                    was_vetoed=veto_result.vetoed,
                    veto_reason=veto_result.reason if veto_result.vetoed else "",
                    veto_policy=veto_result.policy_name if veto_result.vetoed else "",
                ))

            # 3. Step the environment with final (possibly vetoed) actions
            observations, _, terminations, truncations, infos = self.env.step(final_actions)

            # 4. Build and yield the tick frame
            snapshot = _extract_cluster_snapshot(self.env)
            yield WsFrame(
                type=WsFrameType.TICK,
                tick=self.env.t,
                cluster=snapshot,
                actions=action_events,
            )

            await asyncio.sleep(self.tick_delay)

        # Episode finished
        term_reason = self.env.terminal_reason or "max_cycles"
        recovered = term_reason == "recovered"
        collapsed = term_reason == "collapsed"

        summary = EpisodeSummary(
            episode_id=self._episode_id,
            seed=self.seed,
            scenario=self.scenario,
            length=self.env.t,
            recovered=recovered,
            collapsed=collapsed,
            terminal_reason=term_reason,
            mean_health=float(self.env.svc_health.mean()),
        )
        yield WsFrame(
            type=WsFrameType.EPISODE_END,
            tick=self.env.t,
            episode_summary=summary,
            message=f"Episode ended: {term_reason} at tick {self.env.t}",
        )
