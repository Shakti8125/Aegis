"""Encoder training data: simulator rollouts -> ``HeteroData`` graphs, grouped by cluster size.

Phase 3 - owned by the gnn-architect subagent. See PLAN.md section 3, Phase 3.

Where the data comes from
-------------------------
Straight from ``simulator/cluster_env.py``.  Phase 3's gate must be runnable on
a laptop with nothing else running - ``tests/graph/`` already skips itself when
no database is reachable, and the encoder gate would be worth little if it were
the one thing that could not - so the default source is the simulator's own
``graph_snapshot()``, which is the *same dict* Phase 2 ingests into Neo4j.
Reading the identical shape back out of a live database is available in
``encoder/graph_source.py`` and is never required here.

Held-out sizes are the point, not an afterthought
-------------------------------------------------
PLAN.md's Phase 3 bar is "the linear probe passes, on both the training-time
graph sizes and a few held-out sizes", so the size split is baked into this
module rather than into a one-off script:

* :data:`TRAIN_SIZES` - 8, 12 and 16 services. The encoder and the probe are
  fitted here and nowhere else.
* :data:`HELDOUT_SIZES` - 6, 20 and 28 services. Never seen by the encoder or
  the probe; bracketing the training range on *both* sides, because
  extrapolating down to a smaller cluster is a different failure mode from
  extrapolating up to a larger one.

Node counts also move *within* an episode regardless of the configured size:
``graph_snapshot()`` lists only alive pods, so a crash or a ``scale_down`` shrinks
the graph between two consecutive ticks and a ``scale_up`` grows it again.  Even
the "training-time sizes" therefore never present a constant node count.

Action policy while collecting
------------------------------
Rollouts are driven by a noisy random policy (``noop_prob`` of no-ops, otherwise
uniform over the six actions) rather than by a trained one, because Phase 4 does
not exist yet and, more importantly, because the encoder needs *coverage* of the
health space rather than the narrow band of states a good policy would visit.
Sampling every ``snapshot_every`` ticks after a short warm-up keeps consecutive
graphs from being near-duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from torch_geometric.data import HeteroData

from encoder.features import snapshot_to_hetero_data
from simulator.cluster_env import N_ACTIONS, ClusterEnv

__all__ = [
    "HELDOUT_SIZES",
    "TRAIN_SIZES",
    "ClusterSize",
    "RolloutConfig",
    "collect_graphs",
    "collect_sized_dataset",
]


@dataclass(frozen=True)
class ClusterSize:
    """One cluster shape. ``label`` is what shows up in the probe report."""

    n_services: int
    n_nodes: int

    @property
    def label(self) -> str:
        return f"{self.n_services}svc/{self.n_nodes}node"

    def env_kwargs(self) -> dict[str, int | bool]:
        return {
            "n_services": self.n_services,
            "n_nodes": self.n_nodes,
            # A fresh topology per episode, so the encoder sees many wiring
            # patterns at each size instead of memorising one.
            "fixed_topology": False,
        }


#: Sizes the encoder and the probe are fitted on.
TRAIN_SIZES: tuple[ClusterSize, ...] = (
    ClusterSize(8, 4),
    ClusterSize(12, 6),
    ClusterSize(16, 8),
)

#: Sizes neither the encoder nor the probe ever sees during fitting.
HELDOUT_SIZES: tuple[ClusterSize, ...] = (
    ClusterSize(6, 4),
    ClusterSize(20, 10),
    ClusterSize(28, 14),
)


@dataclass(frozen=True)
class RolloutConfig:
    """How much data to draw, and how.

    ``seed`` blocks must not overlap between the encoder-fit, probe-fit and
    evaluation sets - the whole gate rests on the evaluation graphs being
    episodes nothing was fitted on.
    """

    episodes: int = 8
    max_cycles: int = 150
    warmup_ticks: int = 4
    snapshot_every: int = 4
    noop_prob: float = 0.5
    seed: int = 0


def _step_once(env: ClusterEnv, rng: np.random.Generator, noop_prob: float) -> bool:
    """One tick with the collection policy. Returns False once the episode ends."""
    if not env.agents:
        return False
    n = len(env.agents)
    actions = rng.integers(1, N_ACTIONS, size=n)
    actions = np.where(rng.random(n) < noop_prob, 0, actions)
    env.step({name: int(a) for name, a in zip(env.agents, actions)})
    return bool(env.agents)


def collect_graphs(
    size: ClusterSize, config: RolloutConfig = RolloutConfig()
) -> list[HeteroData]:
    """Roll out ``config.episodes`` episodes at ``size`` and return the snapshots."""
    graphs: list[HeteroData] = []
    rng = np.random.default_rng(config.seed)
    env = ClusterEnv(max_cycles=config.max_cycles, **size.env_kwargs())
    for episode in range(config.episodes):
        env.reset(seed=config.seed + episode)
        for tick in range(config.max_cycles):
            if not _step_once(env, rng, config.noop_prob):
                break
            if tick >= config.warmup_ticks and tick % config.snapshot_every == 0:
                graphs.append(snapshot_to_hetero_data(env.graph_snapshot()))
    return graphs


def collect_sized_dataset(
    sizes: Sequence[ClusterSize], config: RolloutConfig = RolloutConfig()
) -> dict[str, list[HeteroData]]:
    """``{size.label: graphs}``. Each size gets its own RNG stream off ``config.seed``."""
    out: dict[str, list[HeteroData]] = {}
    for offset, size in enumerate(sizes):
        sized = RolloutConfig(
            episodes=config.episodes,
            max_cycles=config.max_cycles,
            warmup_ticks=config.warmup_ticks,
            snapshot_every=config.snapshot_every,
            noop_prob=config.noop_prob,
            # 1000 apart so two sizes can never draw the same episode seeds.
            seed=config.seed + 1000 * offset,
        )
        out[size.label] = collect_graphs(size, sized)
    return out


def iter_all(dataset: dict[str, list[HeteroData]]) -> Iterator[HeteroData]:
    for graphs in dataset.values():
        yield from graphs
