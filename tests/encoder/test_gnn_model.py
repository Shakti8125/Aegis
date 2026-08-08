"""The encoder's two heads, its inductivity, and its size-invariant pooling.

PLAN.md Phase 3 makes two structural promises about this model: nothing may be
dimensioned by node count, and the pooled embedding must not change meaning when
the cluster changes size. Both are asserted here as properties rather than left
as claims in a docstring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="Phase 3 needs torch (see requirements.txt)")
pytest.importorskip(
    "torch_geometric", reason="Phase 3 needs torch-geometric (see requirements.txt)"
)

import torch  # noqa: E402
from torch_geometric.data import Batch, HeteroData  # noqa: E402
from torch_geometric.nn import SAGEConv  # noqa: E402

from encoder.dataset import ClusterSize, RolloutConfig, collect_graphs  # noqa: E402
from encoder.features import NODE_TYPES, RELATIONS  # noqa: E402
from encoder.gnn_model import (  # noqa: E402
    AegisGraphEncoder,
    EncoderConfig,
    SAGEConvWithEdgeAttr,
)

SMALL = EncoderConfig(hidden_dim=24, embed_dim=16, global_dim=32, num_layers=2)


def _graphs(n_services: int, n_nodes: int, seed: int = 0, episodes: int = 1):
    return collect_graphs(
        ClusterSize(n_services, n_nodes),
        RolloutConfig(episodes=episodes, max_cycles=60, seed=seed),
    )


@pytest.fixture(scope="module")
def fitted_encoder():
    """Built and normalised at ONE size - 12 services - and never refitted."""
    torch.manual_seed(0)
    encoder = AegisGraphEncoder(SMALL)
    encoder.fit_normalization(_graphs(12, 6, seed=1))
    encoder.eval()
    return encoder


# --------------------------------------------------------------- the two heads
def test_forward_returns_per_node_and_pooled_embeddings(fitted_encoder):
    graph = _graphs(12, 6, seed=2)[0]
    with torch.no_grad():
        out = fitted_encoder(graph)

    assert set(out.node_embeddings) == set(NODE_TYPES)
    for ntype in NODE_TYPES:
        assert out.node_embeddings[ntype].shape == (
            graph[ntype].num_nodes,
            fitted_encoder.embed_dim,
        )
    assert out.global_embedding.shape == (1, fitted_encoder.global_dim)
    assert torch.isfinite(out.global_embedding).all()


def test_agent_observations_are_the_service_rows(fitted_encoder):
    """Phase 4 has one agent per Service; the actor reads these rows."""
    graph = _graphs(12, 6, seed=2)[0]
    with torch.no_grad():
        out = fitted_encoder(graph)
    assert out.agent_observations.shape == (
        graph["Service"].num_nodes,
        fitted_encoder.embed_dim,
    )
    torch.testing.assert_close(out.agent_observations, out.node_embeddings["Service"])


# ----------------------------------------------------------------- inductivity
@pytest.mark.parametrize(
    "n_services, n_nodes", [(6, 4), (8, 4), (12, 6), (20, 10), (28, 14)]
)
def test_runs_at_sizes_it_was_never_fitted_on(fitted_encoder, n_services, n_nodes):
    graph = _graphs(n_services, n_nodes, seed=7)[0]
    with torch.no_grad():
        out = fitted_encoder(graph)
    assert out.node_embeddings["Service"].shape == (n_services, fitted_encoder.embed_dim)
    assert out.node_embeddings["Node"].shape == (n_nodes, fitted_encoder.embed_dim)
    assert out.global_embedding.shape == (1, fitted_encoder.global_dim)
    for value in out.node_embeddings.values():
        assert torch.isfinite(value).all()


def test_no_parameter_is_dimensioned_by_node_count(fitted_encoder):
    """The whole reason PLAN.md picks GraphSAGE over a transductive model."""
    before = {
        name: tuple(p.shape) for name, p in fitted_encoder.named_parameters()
    }
    for n_services, n_nodes in ((6, 4), (16, 8), (28, 14)):
        with torch.no_grad():
            fitted_encoder(_graphs(n_services, n_nodes, seed=11)[0])
    after = {name: tuple(p.shape) for name, p in fitted_encoder.named_parameters()}
    assert before == after

    # And a freshly built encoder has the same shapes, since nothing about the
    # constructor depends on a cluster size either.
    fresh = AegisGraphEncoder(SMALL)
    assert {n: tuple(p.shape) for n, p in fresh.named_parameters()} == before


def test_embeddings_of_a_node_do_not_depend_on_unrelated_cluster_growth(fitted_encoder):
    """A node's embedding is a function of its neighbourhood, not of graph size."""
    graph = _graphs(12, 6, seed=13)[0]
    with torch.no_grad():
        alone = fitted_encoder(graph).node_embeddings["Service"]
        together = fitted_encoder(_disjoint_union(graph, 3)).node_embeddings["Service"]
    torch.testing.assert_close(alone, together[: graph["Service"].num_nodes], atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------- pooling
def _disjoint_union(graph: HeteroData, copies: int) -> HeteroData:
    """``copies`` independent copies of ``graph`` inside ONE graph.

    Mean/max pooling is invariant to this; sum pooling is not. That is precisely
    the property that lets a critic trained at 12 services read a 28-service
    cluster without every dimension reading as more extreme.
    """
    union = HeteroData()
    counts = {ntype: graph[ntype].num_nodes for ntype in NODE_TYPES}
    for ntype in NODE_TYPES:
        union[ntype].x = graph[ntype].x.repeat(copies, 1)
        union[ntype].num_nodes = counts[ntype] * copies
    for relation in RELATIONS:
        src_type, _, dst_type = relation
        edge_index = graph[relation].edge_index
        shifted = [
            edge_index
            + torch.tensor([[k * counts[src_type]], [k * counts[dst_type]]])
            for k in range(copies)
        ]
        union[relation].edge_index = torch.cat(shifted, dim=1)
        attr = graph[relation].get("edge_attr")
        if attr is not None:
            union[relation].edge_attr = attr.repeat(copies, 1)
    return union


@pytest.mark.parametrize("copies", [2, 3, 5])
def test_pooled_embedding_is_invariant_to_graph_size(fitted_encoder, copies):
    graph = _graphs(12, 6, seed=3)[0]
    with torch.no_grad():
        once = fitted_encoder(graph).global_embedding
        many = fitted_encoder(_disjoint_union(graph, copies)).global_embedding
    torch.testing.assert_close(once, many, atol=1e-5, rtol=1e-4)


def test_pooled_embedding_is_invariant_to_node_ordering(fitted_encoder):
    graph = _graphs(12, 6, seed=4)[0]
    generator = torch.Generator().manual_seed(4)
    permuted = HeteroData()
    perms = {}
    for ntype in NODE_TYPES:
        count = graph[ntype].num_nodes
        perms[ntype] = torch.randperm(count, generator=generator)
        permuted[ntype].x = graph[ntype].x[perms[ntype]]
        permuted[ntype].num_nodes = count
    # inverse[old] = new position
    inverse = {
        ntype: torch.argsort(perm) for ntype, perm in perms.items()
    }
    for relation in RELATIONS:
        src_type, _, dst_type = relation
        edge_index = graph[relation].edge_index
        permuted[relation].edge_index = torch.stack(
            (inverse[src_type][edge_index[0]], inverse[dst_type][edge_index[1]])
        )
        attr = graph[relation].get("edge_attr")
        if attr is not None:
            permuted[relation].edge_attr = attr.clone()

    with torch.no_grad():
        base = fitted_encoder(graph)
        shuffled = fitted_encoder(permuted)
    torch.testing.assert_close(
        base.global_embedding, shuffled.global_embedding, atol=1e-5, rtol=1e-4
    )
    # And the per-node head is equivariant, not merely invariant.
    torch.testing.assert_close(
        base.node_embeddings["Service"][perms["Service"]],
        shuffled.node_embeddings["Service"],
        atol=1e-5,
        rtol=1e-4,
    )


def test_batched_forward_matches_individual_forwards(fitted_encoder):
    graphs = _graphs(12, 6, seed=6)[:4]
    with torch.no_grad():
        singly = torch.cat([fitted_encoder(g).global_embedding for g in graphs])
        batched = fitted_encoder(Batch.from_data_list(graphs)).global_embedding
    assert batched.shape == (len(graphs), fitted_encoder.global_dim)
    torch.testing.assert_close(singly, batched, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------- it really is GraphSAGE
def test_conv_reduces_to_pyg_sageconv_without_edge_features():
    """edge_dim=0 must be numerically SAGEConv(mean, root weight, no norm)."""
    torch.manual_seed(0)
    mine = SAGEConvWithEdgeAttr(8, 5)
    reference = SAGEConv(
        8, 5, aggr="mean", root_weight=True, normalize=False, project=False
    )
    with torch.no_grad():
        reference.lin_l.weight.copy_(mine.lin_l.weight)
        reference.lin_l.bias.copy_(mine.lin_l.bias)
        reference.lin_r.weight.copy_(mine.lin_r.weight)

    x = torch.randn(9, 8)
    edge_index = torch.randint(0, 9, (2, 24))
    torch.testing.assert_close(mine(x, edge_index), reference(x, edge_index))


def test_conv_handles_bipartite_relations():
    """Pod -> Service message passing: different node counts on each side."""
    conv = SAGEConvWithEdgeAttr((6, 4), 5)
    x_src = torch.randn(11, 6)
    x_dst = torch.randn(3, 4)
    edge_index = torch.stack(
        (torch.randint(0, 11, (17,)), torch.randint(0, 3, (17,)))
    )
    assert conv((x_src, x_dst), edge_index).shape == (3, 5)


def test_calls_edge_properties_actually_reach_the_embeddings(fitted_encoder):
    """Guards against silently dropping p99_latency_ms / error_rate."""
    graph = _graphs(12, 6, seed=8)[0]
    relation = ("Service", "CALLS", "Service")
    assert graph[relation].edge_attr.numel() > 0

    altered = graph.clone()
    altered[relation].edge_attr = altered[relation].edge_attr + 5.0
    with torch.no_grad():
        base = fitted_encoder(graph).node_embeddings["Service"]
        moved = fitted_encoder(altered).node_embeddings["Service"]
    assert not torch.allclose(base, moved), "CALLS edge properties are being ignored"


def test_empty_relation_does_not_break_the_forward(fitted_encoder):
    graph = _graphs(12, 6, seed=9)[0]
    graph[("Service", "CALLS", "Service")].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    graph[("Service", "CALLS", "Service")].edge_attr = torch.zeros((0, 3))
    graph[("Service", "REV_CALLS", "Service")].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    graph[("Service", "REV_CALLS", "Service")].edge_attr = torch.zeros((0, 3))
    with torch.no_grad():
        out = fitted_encoder(graph)
    assert torch.isfinite(out.global_embedding).all()
    assert torch.isfinite(out.node_embeddings["Service"]).all()


# ------------------------------------------------------------- normalisation
def test_normalization_statistics_travel_with_the_checkpoint():
    """Phase 4 must scale a held-out cluster with the training-time numbers."""
    encoder = AegisGraphEncoder(SMALL)
    encoder.fit_normalization(_graphs(12, 6, seed=1))
    reloaded = AegisGraphEncoder(SMALL)
    reloaded.load_state_dict(encoder.state_dict())

    assert bool(reloaded.get_buffer("normalization_fitted"))
    for ntype in NODE_TYPES:
        torch.testing.assert_close(
            reloaded.get_buffer(f"x_mean__{ntype}"), encoder.get_buffer(f"x_mean__{ntype}")
        )
        assert float(reloaded.get_buffer(f"x_std__{ntype}").min()) > 0.0
