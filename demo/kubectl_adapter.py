"""Kubectl Action Adapter for Aegis Phase 8 Demo.

Translates MAPPO multi-agent decisions into real Kubernetes actions executed via `kubectl`.
Decoupled from training; used exclusively during live cluster integration and demo sessions.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aegis.demo.kubectl_adapter")


class KubectlAdapter:
    """Maps Aegis MARL agent actions to kubectl commands."""

    def __init__(self, namespace: str = "aegis-demo", dry_run: bool = False):
        self.namespace = namespace
        self.dry_run = dry_run

    def execute_action(
        self,
        action_name: str,
        target_type: str,
        target_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a MARL action against the Kubernetes cluster.
        
        Action Space:
          - NOOP (0)
          - RESTART (1): kubectl rollout restart deployment <target_name>
          - SCALE_UP (2): kubectl scale deployment <target_name> --replicas=+1
          - SCALE_DOWN (3): kubectl scale deployment <target_name> --replicas=-1
          - ISOLATE (4): kubectl label pod <target_name> aegis.io/isolated=true
          - REROUTE (5): kubectl annotate service <target_name> aegis.io/rerouted=true
        """
        params = params or {}
        cmd: List[str] = []

        if action_name == "NOOP" or action_name == "0":
            logger.info("NOOP action received; taking no cluster action.")
            return {"status": "success", "action": "NOOP", "cmd": []}

        elif action_name in ("RESTART", "1"):
            cmd = [
                "kubectl", "rollout", "restart",
                f"deployment/{target_name}",
                "-n", self.namespace,
            ]

        elif action_name in ("SCALE_UP", "2"):
            current_replicas = params.get("current_replicas", 1)
            cmd = [
                "kubectl", "scale", f"deployment/{target_name}",
                f"--replicas={current_replicas + 1}",
                "-n", self.namespace,
            ]

        elif action_name in ("SCALE_DOWN", "3"):
            current_replicas = params.get("current_replicas", 2)
            new_replicas = max(1, current_replicas - 1)
            cmd = [
                "kubectl", "scale", f"deployment/{target_name}",
                f"--replicas={new_replicas}",
                "-n", self.namespace,
            ]

        elif action_name in ("ISOLATE", "4"):
            cmd = [
                "kubectl", "label", "pod", target_name,
                "aegis.io/status=isolated", "--overwrite",
                "-n", self.namespace,
            ]

        elif action_name in ("REROUTE", "5"):
            cmd = [
                "kubectl", "annotate", "service", target_name,
                "aegis.io/traffic-rerouted=true", "--overwrite",
                "-n", self.namespace,
            ]

        else:
            raise ValueError(f"Unknown action_name: {action_name}")

        cmd_str = " ".join(cmd)
        logger.info("Executing Kubectl Command: %s (dry_run=%s)", cmd_str, self.dry_run)

        if self.dry_run:
            return {"status": "dry_run", "action": action_name, "cmd": cmd}

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {
                "status": "success",
                "action": action_name,
                "cmd": cmd,
                "stdout": res.stdout.strip(),
            }
        except subprocess.CalledProcessError as exc:
            logger.error("Kubectl command failed: %s | Error: %s", cmd_str, exc.stderr)
            return {
                "status": "failed",
                "action": action_name,
                "cmd": cmd,
                "error": exc.stderr.strip(),
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    adapter = KubectlAdapter(dry_run=True)
    res = adapter.execute_action("RESTART", "deployment", "payment-service")
    print("Dry run result:", res)
