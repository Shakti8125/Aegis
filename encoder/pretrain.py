"""Self-supervised pretraining for the GraphSAGE encoder - no health labels used.

Phase 3 - owned by the gnn-architect subagent. See PLAN.md section 3, Phase 3.

Why self-supervised
-------------------
The Phase 3 gate is a linear probe on *frozen* embeddings.  That gate only means
something if the encoder was never shown the thing the probe predicts: train it
on health labels and the probe measures nothing but the fact that we trained on
health labels.  So the objective here is label-free, and ``health`` is not even
in the input features (see ``encoder/features.py``).

Three objectives, and why each is there
---------------------------------------
1. **Masked-node feature reconstruction** (GraphMAE-style).  A random
   ``mask_ratio`` of nodes has its whole standardised feature vector replaced by
   zeros - which, post-standardisation, *is* the dataset mean, i.e. "no
   information" - and a **linear** decoder must recover the original vector from
   that node's embedding.  Two deliberate choices:

   * masking the *entire* node vector, not individual columns, means a masked
     node can only be reconstructed from its neighbours, so the loss actually
     pays for message passing rather than for an identity map;
   * the decoder is linear, so the pressure is for the telemetry to be
     *linearly* recoverable from the embedding - the same form the Phase 3
     probe, and Phase 4's linear policy head, will read it in.

2. **Visible-node feature reconstruction**, same linear decoder, on the nodes
   that were *not* masked.  This one was added after measuring: with only the
   masked term, the frozen encoder scored *below* a linear model on the raw node
   features (macro-F1 0.90 vs 0.98) - it had learned to infer a node from its
   neighbourhood while quietly discarding the node's own reading.  For a Pod or a
   cluster Node, health is almost entirely a function of that own reading, so
   that is a straight loss of information.  The visible term makes retaining it
   part of the objective; the jumping-knowledge concat in ``gnn_model.py`` makes
   it cheap to satisfy.

3. **Link prediction** over all eight relations, with in-graph negatives.  This
   is the original GraphSAGE unsupervised objective and it keeps topology in the
   embedding: which service depends on which, which pods belong to which
   service.  Without it the encoder can score well on reconstruction while
   treating the graph as a bag of nodes.

Neither term sees ``health``, ``y``, or the cluster size.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from encoder.features import NODE_TYPES, RELATIONS
from encoder.gnn_model import AegisGraphEncoder, EncoderConfig

__all__ = ["PretrainConfig", "PretrainReport", "pretrain_encoder"]


@dataclass(frozen=True)
class PretrainConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-5
    mask_ratio: float = 0.3
    #: Reconstruct *masked* nodes - only their neighbours can supply the answer,
    #: so this is what pays for message passing.
    recon_masked_weight: float = 1.0
    #: Reconstruct *unmasked* nodes - they can see their own features, so this is
    #: what keeps a node's own telemetry linearly readable off its embedding.
    #: Dropping this term measurably cost the probe: the encoder scored below a
    #: linear model on the raw features, i.e. it was a lossy compressor.
    recon_visible_weight: float = 1.0
    link_weight: float = 0.5
    grad_clip: float = 5.0
    seed: int = 0
    log_every: int = 5


@dataclass
class PretrainReport:
    epochs: int
    seconds: float
    history: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1] if self.history else {}


class _MaskedFeatureDecoder(nn.Module):
    """One linear map per node type, embedding -> that type's feature vector."""

    def __init__(self, embed_dim: int, feature_dims: dict[str, int]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {ntype: nn.Linear(embed_dim, dim) for ntype, dim in feature_dims.items()}
        )

    def forward(self, ntype: str, z: Tensor) -> Tensor:
        return self.heads[ntype](z)


def _sample_masks(
    data: HeteroData, ratio: float, generator: torch.Generator
) -> dict[str, Tensor]:
    """Boolean mask per node type; at least one node masked where any exist."""
    masks: dict[str, Tensor] = {}
    for ntype in NODE_TYPES:
        n = data[ntype].num_nodes
        if not n:
            masks[ntype] = torch.zeros(0, dtype=torch.bool)
            continue
        draw = torch.rand(n, generator=generator)
        mask = draw < ratio
        if not bool(mask.any()):
            mask[int(torch.randint(0, n, (1,), generator=generator))] = True
        masks[ntype] = mask
    return masks


def _in_graph_negatives(
    dst: Tensor,
    num_dst: int,
    dst_batch: Tensor | None,
    ptr: Tensor | None,
    generator: torch.Generator,
) -> Tensor:
    """Resample each edge's destination uniformly from its *own* graph.

    Cross-graph negatives would be trivially separable (a 6-service cluster's
    pods look nothing like a 28-service cluster's), so the negative has to come
    from the same graph for the loss to be about structure rather than about
    which graph a node came from.
    """
    if dst_batch is None or ptr is None:
        return torch.randint(0, max(num_dst, 1), dst.shape, generator=generator)
    graph_of_edge = dst_batch[dst]
    lo = ptr[graph_of_edge]
    span = (ptr[graph_of_edge + 1] - lo).clamp_min(1)
    offset = (torch.rand(dst.shape, generator=generator) * span).long()
    return lo + torch.minimum(offset, span - 1)


def _mean_or_zero(terms: Sequence[Tensor]) -> Tensor:
    if not terms:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(list(terms)).mean()


def _link_loss(
    node_embeddings: dict[str, Tensor],
    data: HeteroData,
    logit_scale: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Cosine-similarity link prediction, positives vs in-graph negatives."""
    losses = []
    for rel in RELATIONS:
        if rel not in data.edge_types:
            continue
        edge_index = data[rel].edge_index
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = rel
        z_src = torch.nn.functional.normalize(node_embeddings[src_type], dim=-1)
        z_dst = torch.nn.functional.normalize(node_embeddings[dst_type], dim=-1)
        src, dst = edge_index[0], edge_index[1]
        neg = _in_graph_negatives(
            dst,
            int(data[dst_type].num_nodes),
            getattr(data[dst_type], "batch", None),
            getattr(data[dst_type], "ptr", None),
            generator,
        )
        pos_logit = (z_src[src] * z_dst[dst]).sum(-1) * logit_scale
        neg_logit = (z_src[src] * z_dst[neg]).sum(-1) * logit_scale
        logits = torch.cat([pos_logit, neg_logit])
        target = torch.cat(
            [torch.ones_like(pos_logit), torch.zeros_like(neg_logit)]
        )
        losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        )
    if not losses:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def pretrain_encoder(
    graphs: Sequence[HeteroData],
    config: PretrainConfig = PretrainConfig(),
    *,
    verbose: bool = True,
) -> tuple[AegisGraphEncoder, PretrainReport]:
    """Fit an encoder on ``graphs`` with the label-free objective above."""
    if not graphs:
        raise ValueError("no graphs to pretrain on")

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)

    encoder = AegisGraphEncoder(config.encoder)
    encoder.fit_normalization(graphs)
    decoder = _MaskedFeatureDecoder(encoder.embed_dim, encoder.feature_dims)
    logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    params = list(encoder.parameters()) + list(decoder.parameters()) + [logit_scale]
    optimizer = torch.optim.AdamW(
        params, lr=config.lr, weight_decay=config.weight_decay
    )
    loader = DataLoader(list(graphs), batch_size=config.batch_size, shuffle=True)

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        encoder.train()
        decoder.train()
        totals = {"loss": 0.0, "recon_masked": 0.0, "recon_visible": 0.0, "link": 0.0}
        batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            x_dict = encoder.normalized_x_dict(batch)
            masks = _sample_masks(batch, config.mask_ratio, generator)
            masked_x = {
                ntype: torch.where(masks[ntype].unsqueeze(-1), torch.zeros_like(x), x)
                for ntype, x in x_dict.items()
            }
            out = encoder(batch, x_dict=masked_x)

            masked_terms: list[Tensor] = []
            visible_terms: list[Tensor] = []
            for ntype, mask in masks.items():
                z = out.node_embeddings[ntype]
                for selector, bucket in ((mask, masked_terms), (~mask, visible_terms)):
                    if not bool(selector.any()):
                        continue
                    bucket.append(
                        torch.nn.functional.mse_loss(
                            decoder(ntype, z[selector]), x_dict[ntype][selector]
                        )
                    )
            recon_masked = _mean_or_zero(masked_terms)
            recon_visible = _mean_or_zero(visible_terms)
            link = _link_loss(
                out.node_embeddings, batch, logit_scale.exp(), generator
            )
            loss = (
                config.recon_masked_weight * recon_masked
                + config.recon_visible_weight * recon_visible
                + config.link_weight * link
            )
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["recon_masked"] += float(recon_masked.detach())
            totals["recon_visible"] += float(recon_visible.detach())
            totals["link"] += float(link.detach())
            batches += 1

        record = {"epoch": float(epoch), **{k: v / max(batches, 1) for k, v in totals.items()}}
        history.append(record)
        if verbose and (epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs):
            print(
                f"  epoch {epoch:3d}/{config.epochs}  loss={record['loss']:.4f}  "
                f"recon(masked)={record['recon_masked']:.4f}  "
                f"recon(visible)={record['recon_visible']:.4f}  "
                f"link={record['link']:.4f}"
            )

    encoder.eval()
    return encoder, PretrainReport(
        epochs=config.epochs,
        seconds=time.perf_counter() - started,
        history=history,
    )
