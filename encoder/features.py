"""Turns a ``graph_snapshot()`` dict into a PyG ``HeteroData`` the encoder can read.

Phase 3 - owned by the gnn-architect subagent. See PLAN.md section 3, Phase 3.

Input contract
--------------
The one input is the PLAN.md section 3 Phase 2 snapshot shape - labels
``Service``/``Pod``/``Node``, relationships ``DEPENDS_ON``/``INSTANCE_OF``/
``RUNS_ON``/``CALLS``, node properties health/cpu_pct/mem_pct/restart_count,
``CALLS`` properties p99_latency_ms/error_rate.  That is exactly what
``ClusterEnv.graph_snapshot()`` emits and exactly what
``graph/ingestion_pipeline.py`` writes to Neo4j, so the encoder reads the same
shape whether the snapshot came from the simulator or from a Cypher read
(``encoder/graph_source.py``).  **No live Neo4j is required** to train or probe
the encoder - the simulator path is the default.

The label is withheld from the input features
---------------------------------------------
``health`` is the *target* of the Phase 3 linear probe, so it is deliberately
**not** a node feature.  Feeding it in would make the probe a test of whether a
linear layer can copy one input column, which is no test at all.  The encoder
sees only the underlying telemetry (cpu/mem, latency, error rate, replica
counts, pod status, node occupancy) and has to reconstruct health from it - for
a Service that genuinely requires the graph, because service health is 40% an
aggregate of its pods' health and those pods are only reachable across an
``INSTANCE_OF`` edge.  ``tests/encoder/test_features.py`` asserts the label
property never appears in any feature vector.

Everything else the snapshot carries is kept, including ``sla_violating`` and
pod ``status``: those are real signals an operator sees, they are useful to
Phase 4, and they are not the label - they are thresholded views of features
that are already present.  Note the consequence, since it shows up in the probe
report: pod health is close to a linear function of a pod's *own* telemetry (a
pod that is not ``Running`` reports cpu/mem of exactly 0), so the Pod rows are
the easy part of the probe and the Service rows are the informative part.

Size invariance
---------------
Every feature is either a per-node intensity (cpu fraction, error rate) or a
local ratio (ready/replicas, pods/capacity).  Nothing is normalised by the
number of nodes in the graph, so a feature vector means the same thing in a
6-service cluster and a 28-service one.  This is what lets the probe be scored
on cluster sizes the encoder never trained on.

Reverse edges
-------------
The Phase 2 schema is directed and, as written, information can only flow one
way: ``(:Pod)-[:INSTANCE_OF]->(:Service)`` means a Service is never a message
*destination* for its own pods.  A Service embedding that cannot see its pods
is useless, so every relation is also materialised in the reverse direction as
its own relation type (``REV_INSTANCE_OF`` and friends) with its own weights.
Adding reverse *types* rather than making the graph undirected keeps
"depends on" and "is depended on by" distinguishable, which matters for
cascades - they run one way.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch_geometric.data import HeteroData

#: Node labels, in the order ``graph_snapshot()`` nests them under "nodes".
NODE_TYPES: tuple[str, ...] = ("Service", "Pod", "Node")

#: The property the linear probe predicts. Never a feature - see module docstring.
LABEL_PROPERTY = "health"

#: Probe classes, in label order (0, 1, 2).
HEALTH_CLASSES: tuple[str, ...] = ("healthy", "degraded", "critical")

#: Operating thresholds on ``health`` in [0, 1]:
#: ``healthy >= 0.85 > degraded >= 0.50 > critical``.
HEALTHY_MIN = 0.85
DEGRADED_MIN = 0.50

#: Service tier vocabulary (simulator/topology_generator.py ``tier_names(3)``).
#: A tier outside it lands in the trailing "other" slot rather than crashing, so
#: an n_tiers != 3 config still encodes.
TIER_VOCAB: tuple[str, ...] = ("edge", "mid", "data")

#: Pod lifecycle states emitted by ``graph_snapshot()``.
POD_STATUS_VOCAB: tuple[str, ...] = (
    "Running",
    "NotReady",
    "Restarting",
    "CrashLoopBackOff",
)

CALLS_EDGE = ("Service", "CALLS", "Service")
REV_CALLS_EDGE = ("Service", "REV_CALLS", "Service")

#: Every relation the encoder message-passes over: the four Phase 2 schema
#: relations plus a distinctly-typed reverse of each.
RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("Service", "DEPENDS_ON", "Service"),
    ("Service", "REV_DEPENDS_ON", "Service"),
    CALLS_EDGE,
    REV_CALLS_EDGE,
    ("Pod", "INSTANCE_OF", "Service"),
    ("Service", "REV_INSTANCE_OF", "Pod"),
    ("Pod", "RUNS_ON", "Node"),
    ("Node", "REV_RUNS_ON", "Pod"),
)

#: Snapshot relationship name -> (forward relation, reverse relation).
_SCHEMA_RELS: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
    "DEPENDS_ON": (
        ("Service", "DEPENDS_ON", "Service"),
        ("Service", "REV_DEPENDS_ON", "Service"),
    ),
    "CALLS": (CALLS_EDGE, REV_CALLS_EDGE),
    "INSTANCE_OF": (
        ("Pod", "INSTANCE_OF", "Service"),
        ("Service", "REV_INSTANCE_OF", "Pod"),
    ),
    "RUNS_ON": (
        ("Pod", "RUNS_ON", "Node"),
        ("Node", "REV_RUNS_ON", "Pod"),
    ),
}

#: Relation properties. Only ``CALLS`` carries any in the Phase 2 schema; the
#: structural relations are pure topology and get no edge vector.
EDGE_FEATURE_NAMES: dict[tuple[str, str, str], tuple[str, ...]] = {
    CALLS_EDGE: ("p99_latency_norm", "error_rate", "traffic_share"),
    REV_CALLS_EDGE: ("p99_latency_norm", "error_rate", "traffic_share"),
}

#: Latency divisor, matching ``ClusterConfig.max_latency_ms``. Only a scale -
#: the encoder standardises features afterwards - but it keeps the raw vectors
#: readable when debugging.
_LATENCY_SCALE = 1000.0


def _one_hot(value: str, vocab: Sequence[str]) -> list[float]:
    """One-hot with a trailing catch-all slot, so an unknown value is encodable."""
    out = [0.0] * (len(vocab) + 1)
    try:
        out[vocab.index(value)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


def _service_features(row: Mapping[str, Any]) -> list[float]:
    replicas = float(row.get("replicas", 0.0))
    ready = float(row.get("ready_replicas", 0.0))
    latency = max(float(row.get("p99_latency_ms", 0.0)), 0.0)
    return [
        float(row["cpu_pct"]) / 100.0,
        float(row["mem_pct"]) / 100.0,
        math.log1p(max(float(row.get("restart_count", 0)), 0.0)),
        replicas,
        ready,
        ready / replicas if replicas > 0.0 else 0.0,
        latency / _LATENCY_SCALE,
        math.log1p(latency),
        float(row.get("error_rate", 0.0)),
        float(bool(row.get("isolated", False))),
        float(bool(row.get("sla_violating", False))),
        *_one_hot(str(row.get("tier", "")), TIER_VOCAB),
    ]


def _pod_features(row: Mapping[str, Any]) -> list[float]:
    return [
        float(row["cpu_pct"]) / 100.0,
        float(row["mem_pct"]) / 100.0,
        math.log1p(max(float(row.get("restart_count", 0)), 0.0)),
        *_one_hot(str(row.get("status", "")), POD_STATUS_VOCAB),
    ]


def _node_features(row: Mapping[str, Any]) -> list[float]:
    pod_count = float(row.get("pod_count", 0.0))
    capacity = float(row.get("pod_capacity", 0.0))
    return [
        float(row["cpu_pct"]) / 100.0,
        float(row["mem_pct"]) / 100.0,
        math.log1p(max(float(row.get("restart_count", 0)), 0.0)),
        math.log1p(pod_count),
        pod_count / capacity if capacity > 0.0 else 0.0,
    ]


_EXTRACTORS = {
    "Service": _service_features,
    "Pod": _pod_features,
    "Node": _node_features,
}

#: Feature names per node type, in vector order. The lengths here are the
#: encoder's per-type input dimensions.
FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "Service": (
        "cpu",
        "mem",
        "log1p_restart_count",
        "replicas",
        "ready_replicas",
        "ready_fraction",
        "p99_latency_norm",
        "log1p_p99_latency",
        "error_rate",
        "isolated",
        "sla_violating",
        *(f"tier_{t}" for t in TIER_VOCAB),
        "tier_other",
    ),
    "Pod": (
        "cpu",
        "mem",
        "log1p_restart_count",
        *(f"status_{s}" for s in POD_STATUS_VOCAB),
        "status_other",
    ),
    "Node": (
        "cpu",
        "mem",
        "log1p_restart_count",
        "log1p_pod_count",
        "pod_occupancy",
    ),
}

#: Per-node-type input width. Fixed; independent of how many nodes exist.
FEATURE_DIMS: dict[str, int] = {k: len(v) for k, v in FEATURE_NAMES.items()}

#: Per-relation edge-feature width (0 for the structural relations).
EDGE_DIMS: dict[tuple[str, str, str], int] = {
    rel: len(EDGE_FEATURE_NAMES.get(rel, ())) for rel in RELATIONS
}

__all__ = [
    "CALLS_EDGE",
    "DEGRADED_MIN",
    "EDGE_DIMS",
    "EDGE_FEATURE_NAMES",
    "FEATURE_DIMS",
    "FEATURE_NAMES",
    "HEALTHY_MIN",
    "HEALTH_CLASSES",
    "LABEL_PROPERTY",
    "NODE_TYPES",
    "POD_STATUS_VOCAB",
    "RELATIONS",
    "REV_CALLS_EDGE",
    "TIER_VOCAB",
    "class_counts",
    "health_class",
    "snapshot_to_hetero_data",
]


def health_class(health: float) -> int:
    """``health`` in [0, 1] -> 0 healthy / 1 degraded / 2 critical."""
    if health >= HEALTHY_MIN:
        return 0
    if health >= DEGRADED_MIN:
        return 1
    return 2


def class_counts(labels: Iterable[int]) -> list[int]:
    counts = [0] * len(HEALTH_CLASSES)
    for value in labels:
        counts[int(value)] += 1
    return counts


def _calls_edge_features(row: Mapping[str, Any]) -> list[float]:
    return [
        max(float(row.get("p99_latency_ms", 0.0)), 0.0) / _LATENCY_SCALE,
        float(row.get("error_rate", 0.0)),
        float(row.get("traffic_share", 0.0)),
    ]


def snapshot_to_hetero_data(
    snapshot: Mapping[str, Any], *, with_labels: bool = True
) -> HeteroData:
    """Build a ``HeteroData`` from one Phase 2 snapshot.

    Node order within a type is the snapshot's own order, so row *i* of
    ``data['Service'].x`` is ``snapshot['nodes']['Service'][i]`` - which is the
    simulator's service index, i.e. Phase 4's agent index.  ``data['Service'].id``
    keeps the ids so a caller never has to rely on that by convention.
    """
    data = HeteroData()
    index: dict[str, dict[str, int]] = {}

    for label in NODE_TYPES:
        rows = list(snapshot["nodes"].get(label, ()))
        index[label] = {str(row["id"]): i for i, row in enumerate(rows)}
        extract = _EXTRACTORS[label]
        store = data[label]
        # A type with no instances still needs a well-formed (0, F) matrix.
        store.x = (
            torch.tensor([extract(row) for row in rows], dtype=torch.float32)
            if rows
            else torch.zeros((0, FEATURE_DIMS[label]), dtype=torch.float32)
        )
        store.num_nodes = len(rows)
        store.id = [str(row["id"]) for row in rows]
        if with_labels:
            store.y = torch.tensor(
                [health_class(float(row[LABEL_PROPERTY])) for row in rows],
                dtype=torch.long,
            )

    rels = snapshot.get("relationships", {})
    for schema_name, (fwd, rev) in _SCHEMA_RELS.items():
        src_type, _, dst_type = fwd
        rows = list(rels.get(schema_name, ()))
        src_idx = index[src_type]
        dst_idx = index[dst_type]
        pairs = [
            (src_idx[str(r["source"])], dst_idx[str(r["target"])])
            for r in rows
            if str(r["source"]) in src_idx and str(r["target"]) in dst_idx
        ]
        edge_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros((2, 0), dtype=torch.long)
        )
        data[fwd].edge_index = edge_index
        data[rev].edge_index = edge_index.flip(0)

        if fwd in EDGE_FEATURE_NAMES:
            attrs = [
                _calls_edge_features(r)
                for r in rows
                if str(r["source"]) in src_idx and str(r["target"]) in dst_idx
            ]
            edge_attr = (
                torch.tensor(attrs, dtype=torch.float32)
                if attrs
                else torch.zeros((0, len(EDGE_FEATURE_NAMES[fwd])), dtype=torch.float32)
            )
            data[fwd].edge_attr = edge_attr
            data[rev].edge_attr = edge_attr.clone()

    data.tick = int(snapshot.get("tick", 0))
    return data
