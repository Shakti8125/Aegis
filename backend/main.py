"""FastAPI app: /ws/live, /api/episodes/{id} replay, /api/metrics training curves.

Phase 6 — owned by the main session. See PLAN.md section 3.

Endpoints
---------
``GET  /health``                Health check, active connection count.
``GET  /api/metrics``           Training curves from the latest run.
``GET  /api/metrics/{run_id}``  Training curves from a specific run.
``GET  /api/runs``              List available training runs.
``GET  /api/scenarios``         List available fault scenarios.
``WS   /ws/live``               Live simulation stream.

The WebSocket protocol is:
1. Client connects.
2. Server sends ``{"type": "connected", "message": "..."}``.
3. Client sends ``{"command": "start", "scenario": "mixed", "seed": 42}``.
4. Server streams ``tick`` frames until the episode ends or the client
   sends ``{"command": "stop"}`` / disconnects.
5. Client can send ``"start"`` again for another episode.

CORS is wide-open (``allow_origins=["*"]``) — this is a dev/portfolio
server, not production.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    EpisodeSummary,
    HealthResponse,
    TrainingMetricPoint,
    TrainingMetrics,
    TrainingRun,
    WsFrame,
    WsFrameType,
    WsStartCommand,
)
from pydantic import ValidationError
from backend.ws import ConnectionManager, SimulationRunner
from marl.vec_env import SCENARIOS
from ops_layer.llm_client import make_auto_client, make_client
from ops_layer.narrator import Narrator
from ops_layer.safety_supervisor import SafetySupervisor

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "marl" / "checkpoints"


# ==========================================================================
# App
# ==========================================================================
app = FastAPI(
    title="Aegis Backend",
    description="Multi-Agent RL for Cluster Self-Healing — API & WebSocket server",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()
_active_simulations: dict[int, asyncio.Task] = {}


# ==========================================================================
# Health
# ==========================================================================
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(active_connections=manager.count)


# ==========================================================================
# Scenarios
# ==========================================================================
@app.get("/api/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    """List available fault scenarios with their configurations."""
    return [
        {
            "name": name,
            "enabled_faults": [f.value for f in overrides.get("enabled_faults", [])],
            "n_faults_range": list(overrides.get("n_faults_range", (1, 4))),
        }
        for name, overrides in SCENARIOS.items()
    ]


# ==========================================================================
# Training runs & metrics
# ==========================================================================
def _find_runs() -> list[Path]:
    """Find all run directories that contain a config.json."""
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(
        [d for d in CHECKPOINT_DIR.iterdir() if d.is_dir() and (d / "config.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def _load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List available training runs."""
    runs = []
    for run_dir in _find_runs():
        cfg = _load_run_config(run_dir)
        train_cfg = cfg.get("train", {})
        runs.append({
            "run_id": run_dir.name,
            "seed": train_cfg.get("seed", 0),
            "updates": train_cfg.get("updates", 0),
            "total_env_steps": train_cfg.get("updates", 0) * train_cfg.get("rollout_steps", 128) * train_cfg.get("n_envs", 8),
            "device": train_cfg.get("device", "cpu"),
            "scenario": train_cfg.get("train_scenario", "mixed"),
        })
    return runs


@app.get("/api/metrics", response_model=TrainingMetrics | None)
async def get_latest_metrics() -> TrainingMetrics | None:
    """Training curves from the most recent run."""
    runs = _find_runs()
    if not runs:
        return None
    return _build_training_metrics(runs[0])


@app.get("/api/metrics/{run_id}", response_model=TrainingMetrics | None)
async def get_run_metrics(run_id: str) -> TrainingMetrics | None:
    """Training curves from a specific run."""
    run_dir = CHECKPOINT_DIR / run_id
    if not run_dir.is_dir() or not (run_dir / "config.json").exists():
        return None
    return _build_training_metrics(run_dir)


def _build_training_metrics(run_dir: Path) -> TrainingMetrics:
    cfg = _load_run_config(run_dir)
    train_cfg = cfg.get("train", {})
    raw_metrics = _load_metrics(run_dir)

    run = TrainingRun(
        run_id=run_dir.name,
        seed=train_cfg.get("seed", 0),
        total_updates=train_cfg.get("updates", 0),
        total_env_steps=train_cfg.get("updates", 0) * train_cfg.get("rollout_steps", 128) * train_cfg.get("n_envs", 8),
        device=train_cfg.get("device", "cpu"),
        config=cfg,
    )

    metrics = []
    for r in raw_metrics:
        metrics.append(TrainingMetricPoint(
            update=r.get("update", 0),
            env_steps=r.get("env_steps", 0),
            wall_clock_s=r.get("wall_clock_s", 0.0),
            reward_total=r.get("reward/total", 0.0),
            reward_components={
                k: r.get(f"reward/{k}", 0.0)
                for k in ("sla_violation", "latency", "availability", "action_cost", "invalid_action", "terminal")
            },
            loss_value=r.get("loss/value", 0.0),
            loss_policy=r.get("loss/policy", 0.0),
            loss_entropy=r.get("loss/entropy", 0.0),
            explained_variance=r.get("loss/explained_variance", 0.0),
            episode_length=r.get("episode/length", 0.0),
            recovery_rate=r.get("episode/recovery_rate", 0.0),
            sla_service_ticks=r.get("episode/sla_service_ticks", 0.0),
        ))

    return TrainingMetrics(run=run, metrics=metrics)


# ==========================================================================
# WebSocket: /ws/live
# ==========================================================================
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await manager.connect(ws)
    ws_id = id(ws)

    try:
        # Send connected acknowledgement
        await manager.send(ws, WsFrame(
            type=WsFrameType.CONNECTED,
            message="Connected to Aegis live stream. Send {\"command\": \"start\"} to begin.",
        ))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send(ws, WsFrame(
                    type=WsFrameType.ERROR,
                    message="Invalid JSON",
                ))
                continue

            command = msg.get("command", "")

            if command == "start":
                # Cancel any existing simulation for this client
                if ws_id in _active_simulations:
                    _active_simulations[ws_id].cancel()
                    del _active_simulations[ws_id]

                try:
                    cmd_data = WsStartCommand(**msg)
                except ValidationError as e:
                    await manager.send(ws, WsFrame(
                        type=WsFrameType.ERROR,
                        message=f"Validation error: {e}",
                    ))
                    continue

                scenario = cmd_data.scenario
                if scenario not in SCENARIOS:
                    await manager.send(ws, WsFrame(
                        type=WsFrameType.ERROR,
                        message=f"Unknown scenario '{scenario}'. Valid options: {list(list(SCENARIOS.keys()))}",
                    ))
                    continue

                seed = cmd_data.seed
                max_cycles = cmd_data.max_cycles
                tick_delay_ms = cmd_data.tick_delay_ms

                # Build ops layer components dynamically
                llm = make_auto_client()
                narrator = Narrator(llm_client=llm)
                supervisor = SafetySupervisor()

                runner = SimulationRunner(
                    scenario=scenario,
                    seed=seed,
                    max_cycles=max_cycles,
                    tick_delay_ms=tick_delay_ms,
                    narrator=narrator,
                    supervisor=supervisor,
                )

                async def _run_sim(r: SimulationRunner = runner, target_ws: WebSocket = ws) -> None:
                    try:
                        async for frame in r.run():
                            await manager.send(target_ws, frame)
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.exception("Simulation error: %s", exc)
                        await manager.send(target_ws, WsFrame(
                            type=WsFrameType.ERROR,
                            message=f"Simulation error: {exc}",
                        ))

                _active_simulations[ws_id] = asyncio.create_task(_run_sim())

            elif command == "stop":
                if ws_id in _active_simulations:
                    _active_simulations[ws_id].cancel()
                    del _active_simulations[ws_id]

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
    finally:
        if ws_id in _active_simulations:
            _active_simulations[ws_id].cancel()
            del _active_simulations[ws_id]
        manager.disconnect(ws)
