"""GraphSAGE state encoder: per-node embeddings (agent observations) + pooled global embedding (critic input).

Phase 3 - owned by the gnn-architect subagent. See PLAN.md section 3.

Why GraphSAGE and not GAT
-------------------------
PLAN.md fixes this and the reason is structural, not stylistic: pods scale in
and out *within a single episode* (``graph_snapshot()`` lists only alive pods,
so ``scale_up``/``scale_down``/a crash change the node count between two
consecutive ticks), and Phase 4 will train on some cluster sizes and be
demonstrated on others.  GraphSAGE learns an aggregator over a neighbourhood
rather than a table indexed by node, so the same weights apply to a cluster of
any size.  Nothing here is dimensioned by node count:

* the input projections are ``per-node-type feature width -> hidden``;
* the convolutions are ``hidden -> hidden`` per *relation*, of which there are
  eight regardless of cluster size;
* the pooling head consumes fixed-width per-type mean/max summaries.

``tests/encoder/test_gnn_model.py`` pins that as a property test - it builds the
model for one size, runs it at several others and asserts every parameter shape
is identical - rather than leaving it as a claim in a docstring.

Heterogeneity: typed message passing, not a collapsed homogeneous graph
-----------------------------------------------------------------------
Service, Pod and Node are genuinely different objects: they carry different
properties (a Pod has a lifecycle status, a Node has an occupancy ratio, a
Service has replica counts and SLA state), so their raw feature vectors are not
even the same width.  Collapsing them into one untyped graph would mean padding
all three into a shared vector and asking one weight matrix to mean three
different things.  Instead:

1. **Per-type input encoders.** One ``Linear -> LayerNorm -> ReLU`` per node
   type maps its own feature width into a shared ``hidden`` space.  After this
   point the three types live in one space, which is what lets a *single*
   linear probe - and later a single actor network - read any of them.
2. **Per-relation convolutions.** Each layer is a
   ``torch_geometric.nn.HeteroConv`` holding one :class:`SAGEConvWithEdgeAttr`
   per relation in ``features.RELATIONS`` (the four Phase 2 relations plus a
   distinctly-typed reverse of each), summed at the destination node.  So
   "what my pods tell me" and "what the service I depend on tells me" arrive
   through different weights, and ``DEPENDS_ON`` stays distinguishable from
   ``REV_DEPENDS_ON``.

The alternative, ``torch_geometric.nn.to_hetero()`` on a homogeneous module,
produces roughly this graph automatically.  It is written out explicitly here
because the reverse-relation set and the ``CALLS`` edge-feature path are both
choices worth reading in the source rather than inferring from a transform.

Edge properties
---------------
The schema puts ``p99_latency_ms`` and ``error_rate`` on ``CALLS``, and those
are exactly the signals that make a dependency edge interesting.  Plain
``SAGEConv`` has nowhere to put them, so :class:`SAGEConvWithEdgeAttr` folds a
projected edge vector into the message before aggregation.  With ``edge_dim=0``
it reduces *numerically* to ``SAGEConv(aggr='mean', root_weight=True,
normalize=False)``; that equivalence is asserted against PyG's own
implementation in ``tests/encoder/test_gnn_model.py`` so "this is still
GraphSAGE" is a checked claim.

The two outputs
---------------
:meth:`AegisGraphEncoder.forward` returns an :class:`EncoderOutput` with

* ``node_embeddings`` - ``{node_type: (n_nodes_of_that_type, embed_dim)}``.
  ``node_embeddings["Service"]`` is the Phase 4 **agent observation**: one row
  per Service, and ``simulator/cluster_env.py`` gives one agent per Service in
  the same index order.
* ``global_embedding`` - ``(num_graphs, global_dim)``, the Phase 4 **centralized
  critic input**.

Pooling is per-type **mean and max**, concatenated in the fixed
``features.NODE_TYPES`` order and projected.  Both reductions are invariant to
how many nodes are present - unlike sum, whose magnitude would scale with
cluster size and would make a critic trained at 12 services read a 24-service
cluster as uniformly twice as extreme.  Keeping the types in separate slots
means a fixed-width vector that still says *which* layer of the cluster is
degraded, instead of averaging a Node's occupancy into a Service's error rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.utils import scatter

from encoder.features import (
    EDGE_DIMS,
    FEATURE_DIMS,
    NODE_TYPES,
    RELATIONS,
)

__all__ = [
    "AegisGraphEncoder",
    "EncoderConfig",
    "EncoderOutput",
    "SAGEConvWithEdgeAttr",
]

EdgeType = tuple[str, str, str]


class SAGEConvWithEdgeAttr(MessagePassing):
    r"""Mean-aggregator GraphSAGE with an optional edge-feature term.

    .. math::
        h_i' = W_{\text{root}} x_i
             + W_{\text{neigh}} \cdot \operatorname{mean}_{j \in N(i)}
               \bigl( x_j + W_e e_{ji} \bigr)

    With ``edge_dim=0`` the :math:`W_e e_{ji}` term disappears and this is
    exactly ``torch_geometric.nn.SAGEConv(aggr='mean', root_weight=True,
    normalize=False, project=False)`` - asserted numerically in the tests.  The
    edge term is injected additively into the message, the same way
    :class:`~torch_geometric.nn.GINEConv` handles edge features, which keeps the
    aggregation semantics of GraphSAGE untouched.

    Bipartite by design: ``x`` may be a ``(x_src, x_dst)`` pair, which is what
    :class:`~torch_geometric.nn.HeteroConv` hands it for a relation whose source
    and destination types differ (``Pod -> Service``).
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        *,
        edge_dim: int = 0,
        aggr: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__(aggr=aggr)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels = in_channels
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)

        self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)
        self.lin_edge = (
            nn.Linear(self.edge_dim, in_channels[0], bias=False)
            if self.edge_dim > 0
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Tensor | tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        x_src, x_dst = (x, x) if isinstance(x, Tensor) else x
        aggregated = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            size=(x_src.size(0), x_dst.size(0)),
        )
        out = self.lin_l(aggregated)
        return out + self.lin_r(x_dst)

    def message(self, x_j: Tensor, edge_attr: Tensor | None) -> Tensor:
        if self.lin_edge is None or edge_attr is None:
            return x_j
        return x_j + self.lin_edge(edge_attr)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}({self.in_channels} -> {self.out_channels}, "
            f"edge_dim={self.edge_dim}, aggr={self.aggr})"
        )


@dataclass(frozen=True)
class EncoderConfig:
    """Shape of the encoder. Everything here is independent of cluster size."""

    hidden_dim: int = 64
    embed_dim: int = 64
    global_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    relation_aggr: str = "sum"


class EncoderOutput(NamedTuple):
    """The two heads PLAN.md asks for.

    ``node_embeddings["Service"]`` is the per-agent observation (Phase 4 actor);
    ``global_embedding`` is the pooled critic input, shape ``(num_graphs, global_dim)``.
    """

    node_embeddings: dict[str, Tensor]
    global_embedding: Tensor

    @property
    def agent_observations(self) -> Tensor:
        """Per-Service rows - one agent per Service, in simulator index order."""
        return self.node_embeddings["Service"]


def _mean_max_pool(
    x: Tensor, batch: Tensor | None, num_graphs: int
) -> tuple[Tensor, Tensor]:
    """Size-invariant pooling. Empty node set -> zeros, never NaN."""
    width = x.size(-1)
    if x.size(0) == 0:
        zeros = x.new_zeros((num_graphs, width))
        return zeros, zeros
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    mean = scatter(x, batch, dim=0, dim_size=num_graphs, reduce="mean")
    amax = scatter(x, batch, dim=0, dim_size=num_graphs, reduce="max")
    # scatter-max over an empty group yields the reduction identity; clamp it to
    # zero so a graph with no nodes of this type does not inject -inf/-3e38.
    amax = torch.where(torch.isfinite(amax), amax, torch.zeros_like(amax))
    return torch.nan_to_num(mean), amax


class AegisGraphEncoder(nn.Module):
    """Heterogeneous GraphSAGE over the Phase 2 cluster graph.

    Parameters
    ----------
    config:
        Widths and depth. Defaults are the Phase 3 shape.
    feature_dims / edge_dims:
        Per-node-type and per-relation input widths. Defaults come from
        ``encoder.features``; overriding is only for tests.
    """

    def __init__(
        self,
        config: EncoderConfig | None = None,
        *,
        feature_dims: Mapping[str, int] | None = None,
        edge_dims: Mapping[EdgeType, int] | None = None,
        node_types: Sequence[str] = NODE_TYPES,
        relations: Sequence[EdgeType] = RELATIONS,
    ) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.node_types = tuple(node_types)
        self.relations = tuple(relations)
        self.feature_dims = dict(feature_dims or FEATURE_DIMS)
        self.edge_dims = dict(edge_dims or EDGE_DIMS)

        hidden = self.config.hidden_dim
        embed = self.config.embed_dim

        self.input_encoders = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(self.feature_dims[ntype], hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(inplace=True),
                )
                for ntype in self.node_types
            }
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.config.num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        rel: SAGEConvWithEdgeAttr(
                            hidden, hidden, edge_dim=self.edge_dims.get(rel, 0)
                        )
                        for rel in self.relations
                    },
                    aggr=self.config.relation_aggr,
                )
            )
            self.norms.append(
                nn.ModuleDict(
                    {ntype: nn.LayerNorm(hidden) for ntype in self.node_types}
                )
            )

        self.dropout = nn.Dropout(self.config.dropout)
        # Jumping-knowledge concatenation: the head reads the input encoding and
        # every layer's output, not just the last one. Without it the encoder
        # measured *worse* than a linear model on the raw node features - each
        # LayerNorm+ReLU discards some of a node's own telemetry, and for a Pod
        # or a Node health is almost entirely a function of that telemetry. The
        # concat makes "my own reading" and "what my neighbourhood says" both one
        # linear map away from the embedding instead of trading off against each
        # other. Width is still fixed by depth, never by cluster size.
        self.output_heads = nn.ModuleDict(
            {
                ntype: nn.Linear(hidden * (self.config.num_layers + 1), embed)
                for ntype in self.node_types
            }
        )
        self.pool_head = nn.Sequential(
            nn.Linear(2 * embed * len(self.node_types), self.config.global_dim),
            nn.LayerNorm(self.config.global_dim),
        )

        # Standardisation statistics, fitted on the training-size graphs by
        # :meth:`fit_normalization` and carried in the checkpoint so held-out
        # sizes - and Phase 4 - are scaled by exactly the same numbers.
        for ntype in self.node_types:
            width = self.feature_dims[ntype]
            self.register_buffer(f"x_mean__{ntype}", torch.zeros(width))
            self.register_buffer(f"x_std__{ntype}", torch.ones(width))
        for rel in self.relations:
            width = self.edge_dims.get(rel, 0)
            if width:
                key = _rel_key(rel)
                self.register_buffer(f"e_mean__{key}", torch.zeros(width))
                self.register_buffer(f"e_std__{key}", torch.ones(width))
        self.register_buffer("normalization_fitted", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------------ shapes
    @property
    def embed_dim(self) -> int:
        """Width of one node embedding = width of one Phase 4 agent observation."""
        return self.config.embed_dim

    @property
    def global_dim(self) -> int:
        """Width of the pooled embedding = width of the Phase 4 critic input."""
        return self.config.global_dim

    # ----------------------------------------------------------- normalisation
    @torch.no_grad()
    def fit_normalization(self, graphs: Iterable[HeteroData]) -> None:
        """Fit per-feature mean/std over a set of graphs.

        Standardisation is a property of the *feature*, not of the cluster, so
        the statistics are pooled over every node of a type across every graph
        in the fitting set. Constant features get std 1 rather than 0.
        """
        graphs = list(graphs)
        for ntype in self.node_types:
            chunks = [g[ntype].x for g in graphs if g[ntype].num_nodes]
            if not chunks:
                continue
            stacked = torch.cat(chunks, dim=0)
            self.get_buffer(f"x_mean__{ntype}").copy_(stacked.mean(0))
            self.get_buffer(f"x_std__{ntype}").copy_(stacked.std(0).clamp_min(1e-3))
        for rel in self.relations:
            if not self.edge_dims.get(rel, 0):
                continue
            chunks = [
                g[rel].edge_attr
                for g in graphs
                if rel in g.edge_types and g[rel].get("edge_attr") is not None
                and g[rel].edge_attr.numel()
            ]
            if not chunks:
                continue
            stacked = torch.cat(chunks, dim=0)
            key = _rel_key(rel)
            self.get_buffer(f"e_mean__{key}").copy_(stacked.mean(0))
            self.get_buffer(f"e_std__{key}").copy_(stacked.std(0).clamp_min(1e-3))
        self.get_buffer("normalization_fitted").fill_(True)

    def normalize_x(self, ntype: str, x: Tensor) -> Tensor:
        return (x - self.get_buffer(f"x_mean__{ntype}")) / self.get_buffer(
            f"x_std__{ntype}"
        )

    def _normalize_edge(self, rel: EdgeType, edge_attr: Tensor) -> Tensor:
        key = _rel_key(rel)
        return (edge_attr - self.get_buffer(f"e_mean__{key}")) / self.get_buffer(
            f"e_std__{key}"
        )

    def normalized_x_dict(self, data: HeteroData) -> dict[str, Tensor]:
        """Standardised input features, in the order the encoder consumes them."""
        return {
            ntype: self.normalize_x(ntype, data[ntype].x)
            for ntype in self.node_types
            if ntype in data.node_types
        }

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        data: HeteroData,
        *,
        x_dict: Mapping[str, Tensor] | None = None,
    ) -> EncoderOutput:
        """Encode one graph, or a ``Batch`` of graphs.

        ``x_dict`` overrides the (already standardised) input features.  It is
        the hook ``encoder/pretrain.py`` uses to blank out masked nodes; normal
        callers leave it at ``None``.
        """
        num_graphs = int(getattr(data, "num_graphs", 1) or 1)
        h = dict(x_dict) if x_dict is not None else self.normalized_x_dict(data)
        h = {ntype: self.input_encoders[ntype](x) for ntype, x in h.items()}

        edge_index_dict = {
            rel: data[rel].edge_index
            for rel in self.relations
            if rel in data.edge_types
        }
        edge_attr_dict = {}
        for rel in self.relations:
            if not self.edge_dims.get(rel, 0) or rel not in data.edge_types:
                continue
            attr = data[rel].get("edge_attr")
            if attr is not None:
                edge_attr_dict[rel] = self._normalize_edge(rel, attr)

        stacked: dict[str, list[Tensor]] = {ntype: [value] for ntype, value in h.items()}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
            updated = {}
            for ntype, prev in h.items():
                message = out.get(ntype)
                # Residual: a node type that received nothing this layer (an
                # isolated Pod, say) keeps its representation instead of being
                # zeroed out.
                value = prev if message is None else prev + message
                value = norm[ntype](value)
                updated[ntype] = self.dropout(torch.relu(value))
                stacked[ntype].append(updated[ntype])
            h = updated

        node_embeddings = {
            ntype: self.output_heads[ntype](torch.cat(values, dim=-1))
            for ntype, values in stacked.items()
        }

        summaries = []
        for ntype in self.node_types:
            value = node_embeddings.get(ntype)
            if value is None:
                zeros = self.pool_head[0].weight.new_zeros(
                    (num_graphs, self.config.embed_dim)
                )
                summaries.extend((zeros, zeros))
                continue
            batch = getattr(data[ntype], "batch", None)
            mean, amax = _mean_max_pool(value, batch, num_graphs)
            summaries.extend((mean, amax))

        global_embedding = self.pool_head(torch.cat(summaries, dim=-1))
        return EncoderOutput(node_embeddings, global_embedding)

    # -------------------------------------------------------------- utilities
    @torch.no_grad()
    def encode_snapshot(self, snapshot: Mapping[str, object]) -> EncoderOutput:
        """Convenience path from a raw ``graph_snapshot()`` dict to embeddings."""
        from encoder.features import snapshot_to_hetero_data

        self.eval()
        return self(snapshot_to_hetero_data(snapshot, with_labels=False))


def _rel_key(rel: EdgeType) -> str:
    """Buffer-name-safe key for a relation triple."""
    return "__".join(rel)
