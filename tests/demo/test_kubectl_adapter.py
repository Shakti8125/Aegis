"""Unit tests for Kubectl Action Adapter in Phase 8 demo."""

from __future__ import annotations

import pytest
from demo.kubectl_adapter import KubectlAdapter


def test_kubectl_adapter_dry_run_all_actions():
    adapter = KubectlAdapter(namespace="aegis-demo", dry_run=True)

    # 0. NOOP
    res_noop = adapter.execute_action("NOOP", "deployment", "svc-00")
    assert res_noop["status"] == "success"
    assert res_noop["action"] == "NOOP"
    assert res_noop["cmd"] == []

    # 1. RESTART
    res_restart = adapter.execute_action("RESTART", "deployment", "svc-01")
    assert res_restart["status"] == "dry_run"
    assert res_restart["cmd"] == [
        "kubectl", "rollout", "restart", "deployment/auth-service", "-n", "aegis-demo"
    ]

    # 2. SCALE_UP
    res_up = adapter.execute_action("SCALE_UP", "deployment", "svc-02", {"current_replicas": 2})
    assert res_up["status"] == "dry_run"
    assert res_up["cmd"] == [
        "kubectl", "scale", "deployment/user-service", "--replicas=3", "-n", "aegis-demo"
    ]

    # 3. SCALE_DOWN
    res_down = adapter.execute_action("SCALE_DOWN", "deployment", "svc-03", {"current_replicas": 3})
    assert res_down["status"] == "dry_run"
    assert res_down["cmd"] == [
        "kubectl", "scale", "deployment/payment-service", "--replicas=2", "-n", "aegis-demo"
    ]

    # 4. ISOLATE
    res_isolate = adapter.execute_action("ISOLATE", "pod", "svc-04")
    assert res_isolate["status"] == "dry_run"
    assert res_isolate["cmd"] == [
        "kubectl", "label", "pod", "inventory-service", "aegis.io/status=isolated", "--overwrite", "-n", "aegis-demo"
    ]

    # 5. REROUTE
    res_reroute = adapter.execute_action("REROUTE", "service", "svc-05")
    assert res_reroute["status"] == "dry_run"
    assert res_reroute["cmd"] == [
        "kubectl", "annotate", "service", "notification-service", "aegis.io/traffic-rerouted=true", "--overwrite", "-n", "aegis-demo"
    ]


def test_kubectl_adapter_validation_errors():
    # Invalid namespace
    with pytest.raises(ValueError, match="Invalid Kubernetes namespace name"):
        KubectlAdapter(namespace="Invalid_Namespace!")

    adapter = KubectlAdapter(dry_run=True)

    # Invalid service name
    res_bad = adapter.execute_action("RESTART", "deployment", "bad_name$$")
    assert res_bad["status"] == "failed"
    assert "Invalid Kubernetes resource name" in res_bad["error"]

    # Unknown action
    res_unknown = adapter.execute_action("UNKNOWN_ACTION", "deployment", "svc-00")
    assert res_unknown["status"] == "failed"
    assert "Unknown action_name" in res_unknown["error"]
