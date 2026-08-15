"""Kubectl Action Adapter for Aegis Phase 8 Demo.

Translates MAPPO multi-agent decisions into real Kubernetes actions executed via `kubectl`.
Decoupled from training; used exclusively during live cluster integration and demo sessions.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aegis.demo.kubectl_adapter")

# Kubernetes DNS-1123 subdomain validation regex
K8S_NAME_REGEX = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

DEFAULT_SERVICE_MAP: Dict[str, str] = {
    "svc-00": "gateway-service",
    "svc-01": "auth-service",
    "svc-02": "user-service",
    "svc-03": "payment-service",
    "svc-04": "inventory-service",
    "svc-05": "notification-service",
    "svc-06": "analytics-service",
    "svc-07": "recommendation-service",
    "svc-08": "database-primary",
    "svc-09": "database-replica",
    "svc-10": "cache-redis",
    "svc-11": "search-es",
}


class KubectlAdapter:
    """Maps Aegis MARL agent actions to kubectl commands with input validation."""

    def __init__(
        self,
        namespace: str = "aegis-demo",
        dry_run: bool = True,
        service_map: Optional[Dict[str, str]] = None,
    ):
        if not K8S_NAME_REGEX.fullmatch(namespace):
            raise ValueError(f"Invalid Kubernetes namespace name: '{namespace}'")
        self.namespace = namespace
        self.dry_run = dry_run
        self.service_map = service_map or DEFAULT_SERVICE_MAP

    def _resolve_target_name(self, raw_name: str) -> str:
        name = self.service_map.get(raw_name, raw_name)
        if not K8S_NAME_REGEX.fullmatch(name):
            raise ValueError(
                f"Invalid Kubernetes resource name '{name}' (resolved from '{raw_name}'). "
                "Must match RFC 1123 DNS subdomain format."
            )
        return name

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

        try:
            target_name = self._resolve_target_name(target_name)
        except ValueError as err:
            logger.error("Target name validation failed: %s", err)
            return {"status": "failed", "action": action_name, "error": str(err)}

        if action_name in ("NOOP", "0", "no-op"):
            logger.info("NOOP action received; taking no cluster action.")
            return {"status": "success", "action": "NOOP", "cmd": []}

        elif action_name in ("RESTART", "1", "restart"):
            cmd = [
                "kubectl", "rollout", "restart",
                f"deployment/{target_name}",
                "-n", self.namespace,
            ]

        elif action_name in ("SCALE_UP", "2", "scale_up"):
            current_replicas = params.get("current_replicas", 1)
            cmd = [
                "kubectl", "scale", f"deployment/{target_name}",
                f"--replicas={current_replicas + 1}",
                "-n", self.namespace,
            ]

        elif action_name in ("SCALE_DOWN", "3", "scale_down"):
            current_replicas = params.get("current_replicas", 2)
            new_replicas = max(1, current_replicas - 1)
            cmd = [
                "kubectl", "scale", f"deployment/{target_name}",
                f"--replicas={new_replicas}",
                "-n", self.namespace,
            ]

        elif action_name in ("ISOLATE", "4", "isolate"):
            cmd = [
                "kubectl", "label", "pod", target_name,
                "aegis.io/status=isolated", "--overwrite",
                "-n", self.namespace,
            ]

        elif action_name in ("REROUTE", "5", "reroute"):
            cmd = [
                "kubectl", "annotate", "service", target_name,
                "aegis.io/traffic-rerouted=true", "--overwrite",
                "-n", self.namespace,
            ]

        else:
            return {
                "status": "failed",
                "action": action_name,
                "error": f"Unknown action_name: {action_name}",
            }

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
    res = adapter.execute_action("RESTART", "deployment", "svc-03")
    print("Dry run result:", res)
