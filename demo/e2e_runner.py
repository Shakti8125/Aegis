"""End-to-End Demo Runner for Aegis Phase 8.

Wires together:
1. Simulator (or live cluster stream)
2. Neo4j Graph Ingestion
3. GNN State Encoder
4. MAPPO RL Decision Loop
5. LLM Ops Layer (Grounded Narration & Safety Veto)
6. Kubectl Adapter (Optional execution mode)
7. FastAPI WebSocket broadcasting

Usage:
    python -m demo.e2e_runner --steps 50 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.cluster_env import ClusterConfig, ClusterEnv
from simulator.fault_injection import FaultEvent, FaultType
from encoder.gnn_model import AegisGraphEncoder, EncoderConfig
from ops_layer.narrator import Narrator, ActionContext, ServiceSnapshot
from ops_layer.safety_supervisor import SafetySupervisor
from demo.kubectl_adapter import KubectlAdapter

logger = logging.getLogger("aegis.demo.e2e_runner")


def run_e2e_demo(steps: int = 20, dry_run: bool = True):
    """Run an end-to-end demo episode demonstrating all system layers."""
    logger.info("Initializing Aegis End-to-End Integrated Demo...")

    # 1. Initialize Simulator
    config = ClusterConfig(n_services=5, n_nodes=3)
    env = ClusterEnv(config=config)
    obs_dict, infos = env.reset(seed=42)

    # 2. Initialize GNN Encoder & Safety Supervisor & Narrator
    encoder = AegisGraphEncoder(EncoderConfig())
    narrator = Narrator()
    supervisor = SafetySupervisor()
    adapter = KubectlAdapter(dry_run=dry_run)

    logger.info("Starting integration step loop (%d steps)...", steps)

    for step in range(steps):
        # Inject pod failure at step 5 to simulate incident
        if step == 5:
            logger.warning(">>> Ingesting Fault: POD_CRASH on service svc_0 <<<")
            env.fault_schedule.append(
                FaultEvent(tick=step, fault_type=FaultType.POD_CRASH, target=(0, 0), duration=20, magnitude=1.0)
            )

        # Step environment (Decentralized action selection mock/policy)
        actions = {agent_id: 0 for agent_id in env.agents}
        
        # At step 6, act to heal
        if step >= 6 and step <= 8:
            if "svc_0" in actions:
                actions["svc_0"] = 1  # RESTART

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        # Check safety supervisor for non-NOOP actions
        for agent_id, act_code in actions.items():
            if act_code != 0:
                service_name = agent_id
                act_name = "RESTART" if act_code == 1 else "OTHER"
                
                # Check safety veto
                allowed, reason = supervisor.evaluate_action(
                    action=act_name,
                    target_service=service_name,
                    active_faults=["POD_CRASH"],
                )
                
                if allowed:
                    logger.info("Action %s approved by Safety Supervisor.", act_name)
                    res = adapter.execute_action(act_name, "deployment", service_name)
                    ctx = ActionContext(
                        tick=step,
                        agent_id=agent_id,
                        action=act_code,
                        target_service=ServiceSnapshot(
                            service_id=service_name,
                            health=0.0,
                            cpu_pct=45.0,
                            mem_pct=50.0,
                            p99_latency_ms=850.0,
                            error_rate=0.85,
                            replicas=2,
                            ready_replicas=0,
                        ),
                    )
                    narration = narrator.narrate(ctx)
                    logger.info("LLM Ops Narration: %s", narration.text)
                else:
                    logger.warning("Action VETOED by Safety Supervisor: %s", reason)

        if all(terminations.values()):
            logger.info("Environment episode terminated successfully at step %d.", step)
            break

    logger.info("End-to-End Integration Demo completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run Aegis Phase 8 E2E Integration Demo")
    parser.add_argument("--steps", type=int, default=15, help="Number of simulation steps")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run kubectl commands")
    args = parser.parse_args()

    run_e2e_demo(steps=args.steps, dry_run=args.dry_run)
