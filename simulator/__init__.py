"""Aegis Phase 1 - the simulated cluster environment.

Public surface:

* :class:`ClusterEnv`   - PettingZoo ParallelEnv, one agent per Service
* :class:`ClusterConfig` - every tunable knob, frozen
* :func:`make_env`      - factory
* :class:`Topology` / :func:`generate_topology` - static cluster structure
* :class:`FaultType` / :class:`FaultEvent`      - the reproducible fault schedule
"""

from simulator.cluster_env import (
    ACTION_ISOLATE,
    ACTION_NAMES,
    ACTION_NOOP,
    ACTION_REROUTE,
    ACTION_RESTART,
    ACTION_SCALE_DOWN,
    ACTION_SCALE_UP,
    N_ACTIONS,
    ClusterConfig,
    ClusterEnv,
    make_env,
)
from simulator.fault_injection import FaultEvent, FaultType, build_fault_schedule
from simulator.topology_generator import Topology, generate_topology

__all__ = [
    "ClusterEnv",
    "ClusterConfig",
    "make_env",
    "Topology",
    "generate_topology",
    "FaultType",
    "FaultEvent",
    "build_fault_schedule",
    "ACTION_NAMES",
    "N_ACTIONS",
    "ACTION_NOOP",
    "ACTION_RESTART",
    "ACTION_SCALE_UP",
    "ACTION_SCALE_DOWN",
    "ACTION_ISOLATE",
    "ACTION_REROUTE",
]
