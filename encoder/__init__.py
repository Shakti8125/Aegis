"""Aegis Phase 3 - the GraphSAGE state encoder.

Owned by the gnn-architect subagent. See PLAN.md section 3, Phase 3.

    graph_snapshot() -> HeteroData -> AegisGraphEncoder -> (node embeddings, pooled embedding)
                                                             |                  |
                                                   Phase 4 actor obs      Phase 4 critic input

Layout:
    features.py    Phase 2 snapshot dict -> PyG HeteroData (health withheld: it is the probe label)
    dataset.py     simulator rollouts, split into training-time and held-out cluster sizes
    gnn_model.py   heterogeneous GraphSAGE; per-node embeddings + size-invariant pooled embedding
    pretrain.py    label-free pretraining (masked-feature reconstruction + link prediction)
    probe.py       the Phase 3 gate: linear probe on frozen embeddings  (`python -m encoder.probe`)
    graph_source.py  optional: read the same snapshot shape back out of Neo4j instead

Nothing here needs a live Neo4j. The gate runs off the simulator.

Typical use (Phase 4)::

    from encoder import AegisGraphEncoder, snapshot_to_hetero_data

    data = snapshot_to_hetero_data(env.graph_snapshot(), with_labels=False)
    out = encoder(data)
    actor_obs = out.agent_observations      # (n_services, embed_dim)
    critic_obs = out.global_embedding       # (1, global_dim)

Re-exports are lazy so that ``import encoder`` does not drag torch and
torch-geometric into a process that only wanted a constant.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - re-exported for type checkers only
    from encoder.dataset import (
        HELDOUT_SIZES,
        TRAIN_SIZES,
        ClusterSize,
        RolloutConfig,
        collect_graphs,
        collect_sized_dataset,
    )
    from encoder.features import (
        HEALTH_CLASSES,
        NODE_TYPES,
        RELATIONS,
        health_class,
        snapshot_to_hetero_data,
    )
    from encoder.gnn_model import (
        AegisGraphEncoder,
        EncoderConfig,
        EncoderOutput,
        SAGEConvWithEdgeAttr,
    )
    from encoder.pretrain import PretrainConfig, pretrain_encoder
    from encoder.probe import ProbeConfig, ProbeReport, run_probe

_EXPORTS: dict[str, str] = {
    "AegisGraphEncoder": "encoder.gnn_model",
    "EncoderConfig": "encoder.gnn_model",
    "EncoderOutput": "encoder.gnn_model",
    "SAGEConvWithEdgeAttr": "encoder.gnn_model",
    "HEALTH_CLASSES": "encoder.features",
    "NODE_TYPES": "encoder.features",
    "RELATIONS": "encoder.features",
    "health_class": "encoder.features",
    "snapshot_to_hetero_data": "encoder.features",
    "ClusterSize": "encoder.dataset",
    "RolloutConfig": "encoder.dataset",
    "TRAIN_SIZES": "encoder.dataset",
    "HELDOUT_SIZES": "encoder.dataset",
    "collect_graphs": "encoder.dataset",
    "collect_sized_dataset": "encoder.dataset",
    "PretrainConfig": "encoder.pretrain",
    "pretrain_encoder": "encoder.pretrain",
    "ProbeConfig": "encoder.probe",
    "ProbeReport": "encoder.probe",
    "run_probe": "encoder.probe",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return __all__
