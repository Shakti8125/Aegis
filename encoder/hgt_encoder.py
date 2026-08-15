"""Heterogeneous Graph Transformer (HGT) with Dynamic GATv2 Attention & GRU Memory.

Implements type-specific node/edge transformations, multi-head edge attention
over dynamic microservice graphs, and continuous temporal GRU memory blocks
for capturing cascade dynamics across cluster states.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter, softmax

from encoder.features import (
    EDGE_DIMS,
    FEATURE_DIMS,
    NODE_TYPES,
    RELATIONS,
)
from encoder.gnn_model import EncoderConfig, EncoderOutput, _mean_max_pool


class HGTEncoderOutput(NamedTuple):
    node_embeddings: dict[str, Tensor]
    global_embedding: Tensor
    memory: dict[str, Tensor]

    @property
    def agent_observations(self) -> Tensor:
        return self.node_embeddings["Service"]


class HGTLayer(nn.Module):
    """Single HGT layer with GATv2-style dynamic edge attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        node_types: tuple[str, ...] = NODE_TYPES,
        relations: tuple[tuple[str, str, str], ...] = RELATIONS,
        edge_dims: Mapping[tuple[str, str, str], int] | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.node_types = node_types
        self.relations = relations
        edge_dims = edge_dims or EDGE_DIMS

        # Type-specific Query, Key, Value projections
        self.k_linears = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim) for nt in node_types})
        self.q_linears = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim) for nt in node_types})
        self.v_linears = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim) for nt in node_types})

        # Relation-specific edge attribute projections and GATv2 attention vectors
        self.edge_linears = nn.ModuleDict()
        self.attn_vecs = nn.ParameterDict()
        for rel in relations:
            rel_key = "__".join(rel)
            e_dim = edge_dims.get(rel, 0)
            if e_dim > 0:
                self.edge_linears[rel_key] = nn.Linear(e_dim, out_dim)
            else:
                self.edge_linears[rel_key] = None
            self.attn_vecs[rel_key] = nn.Parameter(torch.randn(num_heads, self.head_dim))

        # Type-specific output projections & LayerNorms
        self.out_linears = nn.ModuleDict({nt: nn.Linear(out_dim, out_dim) for nt in node_types})
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(out_dim) for nt in node_types})

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        edge_attr_dict: dict[tuple[str, str, str], Tensor] | None = None,
    ) -> dict[str, Tensor]:
        edge_attr_dict = edge_attr_dict or {}
        out_dict: dict[str, list[Tensor]] = {nt: [] for nt in self.node_types}

        for rel in self.relations:
            src_type, rel_name, dst_type = rel
            rel_key = "__".join(rel)
            if rel not in edge_index_dict:
                continue

            edge_index = edge_index_dict[rel]
            if edge_index.numel() == 0 or edge_index.size(1) == 0:
                continue

            src_idx, dst_idx = edge_index[0], edge_index[1]
            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]

            # Compute Q, K, V
            q = self.q_linears[dst_type](x_dst)[dst_idx].view(-1, self.num_heads, self.head_dim)
            k = self.k_linears[src_type](x_src)[src_idx].view(-1, self.num_heads, self.head_dim)
            v = self.v_linears[src_type](x_src)[src_idx].view(-1, self.num_heads, self.head_dim)

            # Incorporate edge attributes if present
            if self.edge_linears[rel_key] is not None and rel in edge_attr_dict:
                e_attr = edge_attr_dict[rel]
                if e_attr is not None and e_attr.size(0) == edge_index.size(1):
                    e_proj = self.edge_linears[rel_key](e_attr).view(-1, self.num_heads, self.head_dim)
                    k = k + e_proj

            # GATv2 attention scores: a^T * LeakyReLU(Q + K)
            attn_input = F.leaky_relu(q + k, negative_slope=0.2)
            attn_vec = self.attn_vecs[rel_key]  # (num_heads, head_dim)
            scores = (attn_input * attn_vec).sum(dim=-1) / (self.head_dim ** 0.5)  # (E, num_heads)

            # Softmax per dst node
            num_dst = x_dst.size(0)
            attn_weights = torch.zeros_like(scores)
            for h in range(self.num_heads):
                attn_weights[:, h] = softmax(scores[:, h], dst_idx, num_nodes=num_dst)

            # Weighted sum of values
            msg = (v * attn_weights.unsqueeze(-1)).view(-1, self.out_dim)  # (E, out_dim)
            agg = scatter(msg, dst_idx, dim=0, dim_size=num_dst, reduce="sum")  # (N_dst, out_dim)
            out_dict[dst_type].append(agg)

        # Aggregate across relations, apply residual connection + LayerNorm
        res_dict: dict[str, Tensor] = {}
        for nt in self.node_types:
            h_orig = x_dict[nt]
            if out_dict[nt]:
                agg_all = torch.stack(out_dict[nt], dim=0).sum(dim=0)
                h_out = self.out_linears[nt](agg_all)
                res_dict[nt] = self.norms[nt](h_orig + F.gelu(h_out))
            else:
                res_dict[nt] = self.norms[nt](h_orig)

        return res_dict


class ContinuousTemporalGRUMemory(nn.Module):
    """Continuous GRU memory block for dynamic node representations."""

    def __init__(self, embed_dim: int, node_types: tuple[str, ...] = NODE_TYPES) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.node_types = node_types
        self.gru_cells = nn.ModuleDict({nt: nn.GRUCell(embed_dim, embed_dim) for nt in node_types})

    def forward(
        self,
        current_embeddings: dict[str, Tensor],
        prev_memory: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        updated_memory: dict[str, Tensor] = {}
        prev_memory = prev_memory or {}

        for nt in self.node_types:
            curr = current_embeddings[nt]
            if curr.size(0) == 0:
                updated_memory[nt] = curr
                continue

            prev = prev_memory.get(nt, None)
            if prev is None or prev.size(0) != curr.size(0):
                prev = torch.zeros_like(curr)

            updated_memory[nt] = self.gru_cells[nt](curr, prev)

        return updated_memory


class HGTGraphEncoder(nn.Module):
    """Heterogeneous Graph Transformer (HGT) Aegis Graph Encoder."""

    def __init__(
        self,
        config: EncoderConfig | None = None,
        *,
        feature_dims: Mapping[str, int] | None = None,
        edge_dims: Mapping[tuple[str, str, str], int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        feature_dims = feature_dims or FEATURE_DIMS
        edge_dims = edge_dims or EDGE_DIMS

        self.hidden_dim = self.config.hidden_dim
        self.global_dim = self.config.global_dim

        # Input linear projections per node type
        self.input_projections = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(feature_dims[nt], self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                )
                for nt in NODE_TYPES
            }
        )

        # Stack of HGT layers
        self.layers = nn.ModuleList(
            [
                HGTLayer(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    num_heads=4,
                    node_types=NODE_TYPES,
                    relations=RELATIONS,
                    edge_dims=edge_dims,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # GRU memory block
        self.temporal_memory = ContinuousTemporalGRUMemory(self.hidden_dim)

        # Global pooling head: (mean + max per type) = 2 * len(NODE_TYPES) * hidden_dim -> global_dim
        pooled_dim = 2 * len(NODE_TYPES) * self.hidden_dim
        self.global_head = nn.Sequential(
            nn.Linear(pooled_dim, self.global_dim),
            nn.LayerNorm(self.global_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        data: HeteroData,
        prev_memory: dict[str, Tensor] | None = None,
    ) -> HGTEncoderOutput:
        # Input projections
        x_dict: dict[str, Tensor] = {}
        for nt in NODE_TYPES:
            x = data[nt].x
            x_dict[nt] = self.input_projections[nt](x)

        # Extract edge indices and edge attributes
        edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
        edge_attr_dict: dict[tuple[str, str, str], Tensor] = {}
        for rel in RELATIONS:
            if rel in data.edge_types:
                edge_index_dict[rel] = data[rel].edge_index
                if hasattr(data[rel], "edge_attr") and data[rel].edge_attr is not None:
                    edge_attr_dict[rel] = data[rel].edge_attr

        # Pass through HGT layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        # Update continuous temporal GRU memory
        updated_memory = self.temporal_memory(x_dict, prev_memory)

        # Global pooling across node types
        batch_dict = {
            nt: getattr(data[nt], "batch", None) for nt in NODE_TYPES
        }
        num_graphs = data.num_graphs if hasattr(data, "num_graphs") and data.num_graphs is not None else 1

        pooled_chunks: list[Tensor] = []
        for nt in NODE_TYPES:
            mean, amax = _mean_max_pool(updated_memory[nt], batch_dict[nt], num_graphs)
            pooled_chunks.extend([mean, amax])

        pooled = torch.cat(pooled_chunks, dim=-1)
        global_embedding = self.global_head(pooled)

        return HGTEncoderOutput(
            node_embeddings=updated_memory,
            global_embedding=global_embedding,
            memory=updated_memory,
        )
