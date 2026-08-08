"""Snapshot -> HeteroData: schema conformance, label withholding, size variability."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="Phase 3 needs torch (see requirements.txt)")
pytest.importorskip(
    "torch_geometric", reason="Phase 3 needs torch-geometric (see requirements.txt)"
)

import torch  # noqa: E402

from encoder.features import (  # noqa: E402
    DEGRADED_MIN,
    EDGE_FEATURE_NAMES,
    FEATURE_DIMS,
    FEATURE_NAMES,
    HEALTHY_MIN,
    HEALTH_CLASSES,
    LABEL_PROPERTY,
    NODE_TYPES,
    RELATIONS,
    health_class,
    snapshot_to_hetero_data,
)
from simulator.cluster_env import N_ACTIONS, ClusterEnv  # noqa: E402


def _env(seed: int = 17, **overrides) -> ClusterEnv:
    env = ClusterEnv(**overrides)
    env.reset(seed=seed)
    return env


def _snapshot(seed: int = 17, ticks: int = 0, **overrides) -> dict:
    env = _env(seed, **overrides)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(ticks):
        if not env.agents:
            break
        actions = torch.randint(0, N_ACTIONS, (len(env.agents),), generator=rng)
        env.step({a: int(x) for a, x in zip(env.agents, actions)})
    return env.graph_snapshot()


# --------------------------------------------------------- the label is withheld
def test_the_probe_label_is_never_an_input_feature():
    """The Phase 3 gate is meaningless if `health` is fed straight in."""
    for ntype, names in FEATURE_NAMES.items():
        assert LABEL_PROPERTY not in names, f"{ntype} feeds the probe label back in"
        assert not any(LABEL_PROPERTY in name for name in names), (
            f"{ntype} has a feature derived by name from {LABEL_PROPERTY!r}: {names}"
        )
    for rel, names in EDGE_FEATURE_NAMES.items():
        assert not any(LABEL_PROPERTY in name for name in names), rel


def test_feature_dims_match_the_declared_names():
    data = snapshot_to_hetero_data(_snapshot())
    for ntype in NODE_TYPES:
        assert FEATURE_DIMS[ntype] == len(FEATURE_NAMES[ntype])
        assert data[ntype].x.shape[1] == FEATURE_DIMS[ntype]


# ------------------------------------------------------------- schema conformance
def test_every_snapshot_node_and_relationship_survives_the_conversion():
    snapshot = _snapshot(ticks=8)
    data = snapshot_to_hetero_data(snapshot)

    for ntype in NODE_TYPES:
        assert data[ntype].num_nodes == len(snapshot["nodes"][ntype])
        assert data[ntype].y.shape == (data[ntype].num_nodes,)

    for schema_name, relation in (
        ("DEPENDS_ON", ("Service", "DEPENDS_ON", "Service")),
        ("CALLS", ("Service", "CALLS", "Service")),
        ("INSTANCE_OF", ("Pod", "INSTANCE_OF", "Service")),
        ("RUNS_ON", ("Pod", "RUNS_ON", "Node")),
    ):
        expected = len(snapshot["relationships"][schema_name])
        assert data[relation].edge_index.shape == (2, expected), relation


def test_all_eight_relations_are_present_and_in_bounds():
    data = snapshot_to_hetero_data(_snapshot(ticks=5))
    assert set(data.edge_types) == set(RELATIONS)
    for relation in RELATIONS:
        src_type, _, dst_type = relation
        edge_index = data[relation].edge_index
        assert edge_index.dtype == torch.long
        if edge_index.numel() == 0:
            continue
        assert int(edge_index[0].max()) < data[src_type].num_nodes
        assert int(edge_index[1].max()) < data[dst_type].num_nodes
        assert int(edge_index.min()) >= 0


def test_reverse_relations_are_exact_transposes():
    """A Service must be able to see its own pods; the schema only points the other way."""
    data = snapshot_to_hetero_data(_snapshot(ticks=6))
    for forward, reverse in (
        (("Service", "DEPENDS_ON", "Service"), ("Service", "REV_DEPENDS_ON", "Service")),
        (("Service", "CALLS", "Service"), ("Service", "REV_CALLS", "Service")),
        (("Pod", "INSTANCE_OF", "Service"), ("Service", "REV_INSTANCE_OF", "Pod")),
        (("Pod", "RUNS_ON", "Node"), ("Node", "REV_RUNS_ON", "Pod")),
    ):
        torch.testing.assert_close(
            data[forward].edge_index.flip(0), data[reverse].edge_index
        )


def test_calls_carries_its_schema_edge_properties():
    snapshot = _snapshot(ticks=10)
    data = snapshot_to_hetero_data(snapshot)
    relation = ("Service", "CALLS", "Service")
    edge_attr = data[relation].edge_attr
    assert edge_attr.shape == (
        len(snapshot["relationships"]["CALLS"]),
        len(EDGE_FEATURE_NAMES[relation]),
    )
    assert torch.isfinite(edge_attr).all()
    # error_rate is column 1 and is a probability.
    assert float(edge_attr[:, 1].min()) >= 0.0
    assert float(edge_attr[:, 1].max()) <= 1.0


def test_every_feature_is_finite():
    for ticks in (0, 12, 40):
        data = snapshot_to_hetero_data(_snapshot(seed=3, ticks=ticks))
        for ntype in NODE_TYPES:
            assert torch.isfinite(data[ntype].x).all(), ntype


# ------------------------------------------------------------------------ labels
@pytest.mark.parametrize(
    "health, expected",
    [
        (1.0, 0),
        (HEALTHY_MIN, 0),
        (HEALTHY_MIN - 1e-6, 1),
        (0.7, 1),
        (DEGRADED_MIN, 1),
        (DEGRADED_MIN - 1e-6, 2),
        (0.0, 2),
    ],
)
def test_health_class_thresholds(health, expected):
    assert health_class(health) == expected


def test_labels_agree_with_the_snapshot_health_values():
    snapshot = _snapshot(ticks=15)
    data = snapshot_to_hetero_data(snapshot)
    for ntype in NODE_TYPES:
        expected = [health_class(float(r[LABEL_PROPERTY])) for r in snapshot["nodes"][ntype]]
        assert data[ntype].y.tolist() == expected
        assert set(data[ntype].y.tolist()) <= set(range(len(HEALTH_CLASSES)))


def test_labels_can_be_omitted_for_inference():
    data = snapshot_to_hetero_data(_snapshot(), with_labels=False)
    for ntype in NODE_TYPES:
        assert "y" not in data[ntype]


# ------------------------------------------------------- why the encoder must be inductive
def test_pod_count_moves_within_a_single_episode():
    """Pods scale in and out mid-episode, so even a fixed config has no fixed size."""
    env = _env(seed=5)
    rng = torch.Generator().manual_seed(5)
    counts = set()
    for _ in range(60):
        if not env.agents:
            break
        actions = torch.randint(0, N_ACTIONS, (len(env.agents),), generator=rng)
        env.step({a: int(x) for a, x in zip(env.agents, actions)})
        counts.add(snapshot_to_hetero_data(env.graph_snapshot())["Pod"].num_nodes)
    assert len(counts) > 1, f"pod count never moved: {counts}"


def test_service_rows_keep_the_simulator_agent_order():
    """Row i of the Service block is agent i - Phase 4 indexes on this."""
    snapshot = _snapshot(ticks=4)
    data = snapshot_to_hetero_data(snapshot)
    assert data["Service"].id == [r["id"] for r in snapshot["nodes"]["Service"]]
