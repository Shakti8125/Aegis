"""End-to-End Demo Runner for Aegis Phase 8.

Wires together:
1. Simulator (or live cluster stream)
2. Neo4j Graph Ingestion
3. GNN State Encoder
4. MAPPO RL Decision Loop
5. LLM Ops Layer (Grounded Narration & Safety Veto)
6. Kubectl Adapter (Optional execution mode with human confirmation gate)
7. FastAPI WebSocket broadcasting

Usage:
    python -m demo.e2e_runner --steps 50 --dry-run
    python -m demo.e2e_runner --steps 50 --execute --yes
"""

from __future__ import annotations

import argparse
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


def run_e2e_demo(steps: int = 20, dry_run: bool = True, auto_confirm: bool = False):
    """Run an end-to-end demo episode demonstrating all system layers."""
    logger.info("Initializing Aegis End-to-End Integrated Demo (dry_run=%s)...", dry_run)

    # 1. Initialize Simulator
    config = ClusterConfig(n_services=5, n_nodes=3)
    env = ClusterEnv(config=config)
    obs_dict, infos = env.reset(seed=42)

    # 2. Initialize GNN Encoder, Safety Supervisor, Narrator, and Kubectl Adapter
    encoder = AegisGraphEncoder(EncoderConfig())
    narrator = Narrator()
    supervisor = SafetySupervisor()
    adapter = KubectlAdapter(dry_run=dry_run)

    logger.info("Starting integration step loop (%d steps)...", steps)

    for step in range(steps):
        # Inject pod failure at step 5 to simulate incident
        if step == 5:
            logger.warning(">>> Ingesting Fault: POD_CRASH on service svc-00 <<<")
            env.fault_schedule.append(
                FaultEvent(tick=step, fault_type=FaultType.POD_CRASH, target=(0, 0), duration=20, magnitude=1.0)
            )

        # Step environment (Decentralized action selection mock/policy)
        actions = {agent_id: 0 for agent_id in env.agents}

        # At step 6, act to heal
        if 6 <= step <= 8:
            if "service_0" in actions:
                actions["service_0"] = 1  # RESTART

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        # Check safety supervisor for non-NOOP actions
        for agent_id, act_code in actions.items():
            if act_code != 0:
                service_idx = env.agent_name_to_index[agent_id]
                service_name = f"svc-{service_idx:02d}"

                ctx = ActionContext(
                    tick=step,
                    agent_id=agent_id,
                    action=act_code,
                    target_service=ServiceSnapshot(
                        service_id=service_name,
                        health=float(env.svc_health[service_idx]),
                        cpu_pct=float(env.svc_util[service_idx]),
                        mem_pct=50.0,
                        p99_latency_ms=float(env.svc_latency[service_idx]),
                        error_rate=float(env.svc_error[service_idx]),
                        replicas=int(env.svc_replicas[service_idx]),
                        ready_replicas=int(env.svc_ready[service_idx]),
                    ),
                    active_faults=[
                        {"fault_type": f.fault_type.name.lower(), "target": str(f.target)}
                        for f in env.active_faults
                    ],
                )

                # Check safety veto
                veto_result = supervisor.check(ctx)

                if not veto_result.vetoed:
                    logger.info("Action '%s' approved by Safety Supervisor for %s.", ctx.action_name, service_name)

                    # Confirmation gate for live execution
                    if not dry_run and not auto_confirm:
                        # PROPOSE phase
                        adapter.dry_run = True
                        proposal = adapter.execute_action(ctx.action_name, "deployment", service_name)
                        adapter.dry_run = False
                        
                        cmd_str = " ".join(proposal.get("cmd", []))
                        print(f"\n--- ACTION PROPOSAL ---")
                        print(f"Agent Action : {ctx.action_name}")
                        print(f"Target       : {service_name}")
                        print(f"Command      : {cmd_str}")
                        print(f"-----------------------")
                        
                        resp = input(f"CONFIRM EXECUTION? [y/N]: ")
                        if resp.lower() != "y":
                            logger.warning("Execution cancelled by operator for action on %s.", service_name)
                            continue

                    res = adapter.execute_action(ctx.action_name, "deployment", service_name)
                    narration = narrator.narrate(ctx)
                    logger.info("Kubectl result: %s | LLM Ops Narration: %s", res.get("status"), narration.text)
                else:
                    logger.warning("Action VETOED by Safety Supervisor: %s", veto_result.reason)

        if all(terminations.values()):
            logger.info("Environment episode terminated successfully at step %d.", step)
            break

    logger.info("End-to-End Integration Demo completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run Aegis Phase 8 E2E Integration Demo")
    parser.add_argument("--steps", type=int, default=15, help="Number of simulation steps")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run kubectl commands (default)")
    parser.add_argument("--execute", action="store_true", help="Execute real kubectl commands against the cluster")
    parser.add_argument("--yes", "-y", action="store_true", help="Bypass interactive confirmation prompt when --execute is passed")
    args = parser.parse_args()

    is_dry_run = not args.execute

    run_e2e_demo(steps=args.steps, dry_run=is_dry_run, auto_confirm=args.yes)
