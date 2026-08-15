"""Linear-probe validation: predict node health from frozen embeddings before wiring the encoder into marl/.

Phase 3 - owned by the gnn-architect subagent. See PLAN.md section 3, Phase 3.

    python -m encoder.probe            # the full gate
    python -m encoder.probe --quick    # a smaller, faster version of the same thing

Exit status is the gate: 0 if the encoder is ready for Phase 4, 1 if it is not.

What is being tested
--------------------
PLAN.md: *"validate it in isolation: train a linear probe that predicts node
health status from the frozen embeddings. If the probe can't beat a trivial
baseline, fix the encoder before you're also debugging RL on top of it."*  So:

1. Pretrain the encoder on :data:`~encoder.dataset.TRAIN_SIZES` with the
   label-free objective in ``encoder/pretrain.py``.
2. **Freeze it** (``eval()`` + ``no_grad``), and embed a *different* set of
   episodes.
3. Fit a **linear** classifier - one ``nn.Linear`` on standardised embeddings,
   no hidden layer - on those embeddings.
4. Score it on episodes nothing was fitted on, at the training sizes *and* at
   :data:`~encoder.dataset.HELDOUT_SIZES`.

Four disjoint seed blocks, so nothing is ever scored on data it was fitted on:
encoder pretrain / probe fit / evaluation at training sizes / evaluation at
held-out sizes.

Why balanced accuracy and macro-F1, never plain accuracy
--------------------------------------------------------
The classes are imbalanced - roughly 60/22/18 healthy/degraded/critical pooled
over the three node types, and far more skewed within a type - so a classifier
that always answers "healthy" scores ~60% accuracy while being worth nothing.
Every headline number here is therefore **balanced accuracy** (mean per-class
recall; a constant predictor scores exactly 1/3) and **macro-F1** (unweighted
mean of per-class F1; a constant predictor cannot exceed ~0.25 on three
classes).  Plain accuracy is printed only as context, and the gate never reads
it.  The majority-class predictor is scored on the same metrics on the same
rows, so "beat the trivial baseline" is a number, not a claim.  The probe is
also fitted with inverse-frequency class weights, so collapsing onto the
majority class is not even a good local minimum for it.

Three baselines are reported
----------------------------
* **majority class** - the trivial baseline the gate is defined against.
* **uniform random** - the other trivial baseline, for orientation.
* **raw features + the same linear probe** - not required by PLAN.md, but it is
  the number that says whether the *graph* earned its keep.  If a linear model
  on the unaggregated per-node telemetry matches the encoder, the encoder is
  just an expensive identity map, and the honest read of that is "passes the
  gate, adds nothing" rather than "passes".

The gate
--------
Applied independently to the training-size split and the held-out-size split;
both must pass, since PLAN.md's bar is explicitly *"on both the training-time
graph sizes and a few held-out sizes"*.

* pooled balanced accuracy >= :data:`GATE_MIN_BALANCED_ACCURACY`
* pooled macro-F1 >= majority-class macro-F1 + :data:`GATE_MIN_MACRO_F1_MARGIN`
* and, per node type, balanced accuracy >= :data:`GATE_MIN_TYPE_BALANCED_ACCURACY`
  and macro-F1 >= that type's own majority-class macro-F1 +
  :data:`GATE_MIN_TYPE_MACRO_F1_MARGIN`

The per-type clause is what stops the pooled number from passing on the easy
rows alone: Pods are ~60% of all graph nodes and a pod's health is close to a
linear function of its own telemetry (a pod that is not ``Running`` reports
cpu/mem of exactly 0), so a pooled-only gate could be cleared while Service
health - the thing Phase 4's agents actually observe - was not learned at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from encoder.dataset import (
    HELDOUT_SIZES,
    TRAIN_SIZES,
    ClusterSize,
    RolloutConfig,
    collect_sized_dataset,
    iter_all,
)
from encoder.features import FEATURE_DIMS, HEALTH_CLASSES, NODE_TYPES
from encoder.gnn_model import AegisGraphEncoder, EncoderConfig
from encoder.pretrain import PretrainConfig, pretrain_encoder

__all__ = [
    "GATE_MIN_BALANCED_ACCURACY",
    "GATE_MIN_MACRO_F1_MARGIN",
    "GATE_MIN_TYPE_BALANCED_ACCURACY",
    "GATE_MIN_TYPE_MACRO_F1_MARGIN",
    "ClassificationMetrics",
    "ProbeConfig",
    "ProbeReport",
    "SplitReport",
    "evaluate_predictions",
    "format_report",
    "gate_passes",
    "run_probe",
]

N_CLASSES = len(HEALTH_CLASSES)

#: Gate thresholds. Chance level on three classes is 1/3 balanced accuracy; a
#: constant predictor tops out near 0.25 macro-F1.
GATE_MIN_BALANCED_ACCURACY = 0.60
GATE_MIN_MACRO_F1_MARGIN = 0.20
GATE_MIN_TYPE_BALANCED_ACCURACY = 0.45
GATE_MIN_TYPE_MACRO_F1_MARGIN = 0.10


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationMetrics:
    """Everything the gate and the report need from one set of predictions."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_f1: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    support: tuple[int, ...]

    @property
    def n(self) -> int:
        return sum(self.support)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES
) -> ClassificationMetrics:
    """Accuracy, balanced accuracy and macro-F1 from a confusion matrix.

    A class with no support contributes neither a recall nor an F1 term, so an
    absent class cannot inflate or deflate the macro averages.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    hits = np.diag(confusion)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, hits / np.maximum(support, 1), np.nan)
        precision = np.where(predicted > 0, hits / np.maximum(predicted, 1), 0.0)
    denom = precision + np.nan_to_num(recall)
    f1 = np.where(denom > 0, 2 * precision * np.nan_to_num(recall) / np.maximum(denom, 1e-12), 0.0)
    present = support > 0

    total = int(support.sum())
    return ClassificationMetrics(
        accuracy=float(hits.sum() / total) if total else 0.0,
        balanced_accuracy=float(np.nanmean(recall[present])) if present.any() else 0.0,
        macro_f1=float(f1[present].mean()) if present.any() else 0.0,
        per_class_f1=tuple(float(v) for v in f1),
        per_class_recall=tuple(float(np.nan_to_num(v)) for v in recall),
        support=tuple(int(v) for v in support),
    )


# ---------------------------------------------------------------------------
# the linear probe
# ---------------------------------------------------------------------------
class LinearProbe:
    """Multinomial logistic regression. One ``nn.Linear``, nothing else.

    Standardisation and the weight matrix are both fitted on the fit split only
    and then applied unchanged to every evaluation split - including the
    held-out cluster sizes, which is the point of the exercise.
    """

    def __init__(self, weight: Tensor, bias: Tensor, mean: Tensor, std: Tensor) -> None:
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.std = std

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    def predict(self, x: Tensor) -> np.ndarray:
        z = (x - self.mean) / self.std
        return (z @ self.weight.T + self.bias).argmax(dim=1).numpy()


def fit_linear_probe(
    x: Tensor,
    y: Tensor,
    *,
    n_classes: int = N_CLASSES,
    epochs: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> LinearProbe:
    """Fit the probe with inverse-frequency class weights.

    The class weighting is what makes "passes by always predicting the majority
    class" impossible rather than merely unlikely: under this loss a constant
    predictor is penalised in proportion to how rare the classes it ignores are.
    """
    torch.manual_seed(seed)
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    z = (x - mean) / std

    counts = torch.bincount(y, minlength=n_classes).float()
    weights = torch.where(
        counts > 0, counts.sum() / (n_classes * counts.clamp_min(1.0)), torch.zeros_like(counts)
    )

    layer = nn.Linear(z.shape[1], n_classes)
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(z), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return LinearProbe(
            layer.weight.detach().clone(),
            layer.bias.detach().clone(),
            mean,
            std,
        )


# ---------------------------------------------------------------------------
# embedding extraction
# ---------------------------------------------------------------------------
#: Layout of the raw-feature control vector: a node-type one-hot followed by one
#: zero-padded block per node type. Zero padding rather than a shared projection
#: keeps it a genuinely *raw* control - no learned parameters at all.
_RAW_OFFSETS: dict[str, int] = {}
_offset = len(NODE_TYPES)
for _ntype in NODE_TYPES:
    _RAW_OFFSETS[_ntype] = _offset
    _offset += FEATURE_DIMS[_ntype]
RAW_FEATURE_DIM = _offset


@dataclass
class EmbeddedSplit:
    """Frozen embeddings, raw-feature controls and labels for one data split."""

    name: str
    embeddings: Tensor
    raw_features: Tensor
    labels: Tensor
    node_type: np.ndarray
    size_label: np.ndarray

    def subset(self, mask: np.ndarray) -> "EmbeddedSplit":
        index = torch.from_numpy(np.flatnonzero(mask))
        return EmbeddedSplit(
            name=self.name,
            embeddings=self.embeddings[index],
            raw_features=self.raw_features[index],
            labels=self.labels[index],
            node_type=self.node_type[mask],
            size_label=self.size_label[mask],
        )


@torch.no_grad()
def embed_split(
    encoder: AegisGraphEncoder,
    dataset: dict[str, list[HeteroData]],
    name: str,
    *,
    batch_size: int = 32,
) -> EmbeddedSplit:
    """Run the frozen encoder over a dataset and stack per-node rows."""
    encoder.eval()
    embeddings: list[Tensor] = []
    raws: list[Tensor] = []
    labels: list[Tensor] = []
    types: list[np.ndarray] = []
    sizes: list[np.ndarray] = []

    for size_label, graphs in dataset.items():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            out = encoder(batch)
            x_dict = encoder.normalized_x_dict(batch)
            for type_index, ntype in enumerate(NODE_TYPES):
                store = batch[ntype]
                count = int(store.num_nodes)
                if not count:
                    continue
                embeddings.append(out.node_embeddings[ntype])
                raw = torch.zeros((count, RAW_FEATURE_DIM), dtype=torch.float32)
                raw[:, type_index] = 1.0
                start = _RAW_OFFSETS[ntype]
                raw[:, start : start + FEATURE_DIMS[ntype]] = x_dict[ntype]
                raws.append(raw)
                labels.append(store.y)
                types.append(np.full(count, type_index, dtype=np.int8))
                sizes.append(np.full(count, size_label, dtype=object))

    return EmbeddedSplit(
        name=name,
        embeddings=torch.cat(embeddings),
        raw_features=torch.cat(raws),
        labels=torch.cat(labels),
        node_type=np.concatenate(types),
        size_label=np.concatenate(sizes),
    )


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def majority_class(y: Tensor, n_classes: int = N_CLASSES) -> int:
    return int(torch.bincount(y, minlength=n_classes).argmax())


def _majority_metrics(y: np.ndarray, klass: int) -> ClassificationMetrics:
    return evaluate_predictions(y, np.full_like(y, klass))


def _random_metrics(y: np.ndarray, seed: int = 0) -> ClassificationMetrics:
    rng = np.random.default_rng(seed)
    return evaluate_predictions(y, rng.integers(0, N_CLASSES, size=y.shape[0]))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass
class SplitReport:
    name: str
    sizes: tuple[str, ...]
    n_nodes: int
    class_counts: tuple[int, ...]
    encoder: ClassificationMetrics
    majority: ClassificationMetrics
    random: ClassificationMetrics
    raw_features: ClassificationMetrics
    per_type_encoder: dict[str, ClassificationMetrics]
    per_type_majority: dict[str, ClassificationMetrics]
    per_type_raw: dict[str, ClassificationMetrics]
    per_size_encoder: dict[str, ClassificationMetrics]
    per_size_majority: dict[str, ClassificationMetrics]

    @property
    def macro_f1_margin(self) -> float:
        return self.encoder.macro_f1 - self.majority.macro_f1

    def failures(self) -> list[str]:
        """Every gate clause this split violates, named. Empty list == pass."""
        problems: list[str] = []
        if self.encoder.balanced_accuracy < GATE_MIN_BALANCED_ACCURACY:
            problems.append(
                f"pooled balanced accuracy {self.encoder.balanced_accuracy:.3f} "
                f"< {GATE_MIN_BALANCED_ACCURACY:.2f}"
            )
        if self.macro_f1_margin < GATE_MIN_MACRO_F1_MARGIN:
            problems.append(
                f"pooled macro-F1 {self.encoder.macro_f1:.3f} beats majority-class "
                f"{self.majority.macro_f1:.3f} by only {self.macro_f1_margin:+.3f} "
                f"(need +{GATE_MIN_MACRO_F1_MARGIN:.2f})"
            )
        for ntype in NODE_TYPES:
            enc = self.per_type_encoder.get(ntype)
            maj = self.per_type_majority.get(ntype)
            if enc is None or maj is None:
                continue
            if enc.balanced_accuracy < GATE_MIN_TYPE_BALANCED_ACCURACY:
                problems.append(
                    f"{ntype} balanced accuracy {enc.balanced_accuracy:.3f} "
                    f"< {GATE_MIN_TYPE_BALANCED_ACCURACY:.2f}"
                )
            margin = enc.macro_f1 - maj.macro_f1
            if margin < GATE_MIN_TYPE_MACRO_F1_MARGIN:
                problems.append(
                    f"{ntype} macro-F1 {enc.macro_f1:.3f} beats majority-class "
                    f"{maj.macro_f1:.3f} by only {margin:+.3f} "
                    f"(need +{GATE_MIN_TYPE_MACRO_F1_MARGIN:.2f})"
                )
        return problems

    @property
    def passed(self) -> bool:
        return not self.failures()


@dataclass
class ProbeReport:
    train_sizes: tuple[str, ...]
    heldout_sizes: tuple[str, ...]
    splits: dict[str, SplitReport]
    embed_dim: int
    global_dim: int
    n_pretrain_graphs: int
    n_probe_fit_nodes: int
    seconds: float
    pretrain_seconds: float

    @property
    def passed(self) -> bool:
        return all(split.passed for split in self.splits.values())

    def failures(self) -> dict[str, list[str]]:
        return {
            name: split.failures()
            for name, split in self.splits.items()
            if split.failures()
        }


def gate_passes(split: SplitReport) -> bool:
    """Public form of the gate, so tests can assert on it directly."""
    return split.passed


# ---------------------------------------------------------------------------
# configuration and the run itself
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeConfig:
    """Four disjoint seed blocks; nothing is scored on data it was fitted on."""

    train_sizes: tuple[ClusterSize, ...] = TRAIN_SIZES
    heldout_sizes: tuple[ClusterSize, ...] = HELDOUT_SIZES
    encoder_rollout: RolloutConfig = RolloutConfig(episodes=8, seed=0)
    probe_fit_rollout: RolloutConfig = RolloutConfig(episodes=6, seed=100_000)
    eval_rollout: RolloutConfig = RolloutConfig(episodes=5, seed=200_000)
    heldout_rollout: RolloutConfig = RolloutConfig(episodes=5, seed=300_000)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    probe_epochs: int = 400
    probe_lr: float = 0.05
    probe_weight_decay: float = 1e-4
    seed: int = 0

    @classmethod
    def quick(cls) -> "ProbeConfig":
        """A small version of the same pipeline - for tests and for iteration."""
        small = EncoderConfig(hidden_dim=32, embed_dim=32, global_dim=64, num_layers=2)
        return cls(
            train_sizes=(ClusterSize(8, 4), ClusterSize(12, 6)),
            heldout_sizes=(ClusterSize(20, 10),),
            encoder_rollout=RolloutConfig(episodes=3, max_cycles=90, seed=0),
            probe_fit_rollout=RolloutConfig(episodes=3, max_cycles=90, seed=100_000),
            eval_rollout=RolloutConfig(episodes=2, max_cycles=90, seed=200_000),
            heldout_rollout=RolloutConfig(episodes=2, max_cycles=90, seed=300_000),
            pretrain=PretrainConfig(encoder=small, epochs=8, log_every=4),
            probe_epochs=250,
        )


def _split_report(
    name: str,
    sizes: Sequence[str],
    split: EmbeddedSplit,
    embedding_probe: LinearProbe,
    raw_probe: LinearProbe,
    majority: int,
    per_type_majority_class: dict[str, int],
    seed: int,
) -> SplitReport:
    y = split.labels.numpy()
    encoder_metrics = evaluate_predictions(y, embedding_probe.predict(split.embeddings))
    raw_metrics = evaluate_predictions(y, raw_probe.predict(split.raw_features))

    per_type_encoder: dict[str, ClassificationMetrics] = {}
    per_type_majority: dict[str, ClassificationMetrics] = {}
    per_type_raw: dict[str, ClassificationMetrics] = {}
    for type_index, ntype in enumerate(NODE_TYPES):
        mask = split.node_type == type_index
        if not mask.any():
            continue
        subset = split.subset(mask)
        y_sub = subset.labels.numpy()
        per_type_encoder[ntype] = evaluate_predictions(
            y_sub, embedding_probe.predict(subset.embeddings)
        )
        per_type_raw[ntype] = evaluate_predictions(
            y_sub, raw_probe.predict(subset.raw_features)
        )
        per_type_majority[ntype] = _majority_metrics(
            y_sub, per_type_majority_class[ntype]
        )

    per_size_encoder: dict[str, ClassificationMetrics] = {}
    per_size_majority: dict[str, ClassificationMetrics] = {}
    for size_label in sizes:
        mask = split.size_label == size_label
        if not mask.any():
            continue
        subset = split.subset(mask)
        y_sub = subset.labels.numpy()
        per_size_encoder[size_label] = evaluate_predictions(
            y_sub, embedding_probe.predict(subset.embeddings)
        )
        per_size_majority[size_label] = _majority_metrics(y_sub, majority)

    return SplitReport(
        name=name,
        sizes=tuple(sizes),
        n_nodes=int(y.shape[0]),
        class_counts=tuple(
            int(v) for v in np.bincount(y, minlength=N_CLASSES)
        ),
        encoder=encoder_metrics,
        majority=_majority_metrics(y, majority),
        random=_random_metrics(y, seed),
        raw_features=raw_metrics,
        per_type_encoder=per_type_encoder,
        per_type_majority=per_type_majority,
        per_type_raw=per_type_raw,
        per_size_encoder=per_size_encoder,
        per_size_majority=per_size_majority,
    )


def run_probe(
    config: ProbeConfig = ProbeConfig(), *, verbose: bool = True
) -> tuple[AegisGraphEncoder, ProbeReport]:
    """Pretrain, freeze, probe, score. Returns the encoder and report; never raises on failure."""
    started = time.perf_counter()

    def say(message: str) -> None:
        if verbose:
            print(message, flush=True)

    say("[1/5] collecting simulator rollouts")
    pretrain_data = collect_sized_dataset(config.train_sizes, config.encoder_rollout)
    probe_fit_data = collect_sized_dataset(config.train_sizes, config.probe_fit_rollout)
    eval_data = collect_sized_dataset(config.train_sizes, config.eval_rollout)
    heldout_data = collect_sized_dataset(config.heldout_sizes, config.heldout_rollout)
    pretrain_graphs = list(iter_all(pretrain_data))
    say(
        f"      pretrain={len(pretrain_graphs)} graphs  "
        f"probe-fit={sum(len(v) for v in probe_fit_data.values())}  "
        f"eval@train-sizes={sum(len(v) for v in eval_data.values())}  "
        f"eval@held-out-sizes={sum(len(v) for v in heldout_data.values())}"
    )

    say("[2/5] pretraining the encoder (self-supervised, no health labels)")
    encoder, pretrain_report = pretrain_encoder(
        pretrain_graphs, config.pretrain, verbose=verbose
    )

    say("[3/5] freezing the encoder and extracting embeddings")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    fit_split = embed_split(encoder, probe_fit_data, "probe-fit")
    eval_split = embed_split(encoder, eval_data, "training sizes")
    heldout_split = embed_split(encoder, heldout_data, "held-out sizes")

    say("[4/5] fitting the linear probe on frozen embeddings")
    embedding_probe = fit_linear_probe(
        fit_split.embeddings,
        fit_split.labels,
        epochs=config.probe_epochs,
        lr=config.probe_lr,
        weight_decay=config.probe_weight_decay,
        seed=config.seed,
    )
    raw_probe = fit_linear_probe(
        fit_split.raw_features,
        fit_split.labels,
        epochs=config.probe_epochs,
        lr=config.probe_lr,
        weight_decay=config.probe_weight_decay,
        seed=config.seed,
    )
    majority = majority_class(fit_split.labels)
    per_type_majority_class = {}
    for type_index, ntype in enumerate(NODE_TYPES):
        mask = fit_split.node_type == type_index
        per_type_majority_class[ntype] = (
            majority_class(fit_split.subset(mask).labels) if mask.any() else 0
        )

    say("[5/5] scoring")
    splits = {
        "training sizes": _split_report(
            "training sizes",
            [s.label for s in config.train_sizes],
            eval_split,
            embedding_probe,
            raw_probe,
            majority,
            per_type_majority_class,
            config.seed,
        ),
        "held-out sizes": _split_report(
            "held-out sizes",
            [s.label for s in config.heldout_sizes],
            heldout_split,
            embedding_probe,
            raw_probe,
            majority,
            per_type_majority_class,
            config.seed,
        ),
    }

    report = ProbeReport(
        train_sizes=tuple(s.label for s in config.train_sizes),
        heldout_sizes=tuple(s.label for s in config.heldout_sizes),
        splits=splits,
        embed_dim=encoder.embed_dim,
        global_dim=encoder.global_dim,
        n_pretrain_graphs=len(pretrain_graphs),
        n_probe_fit_nodes=int(fit_split.labels.shape[0]),
        seconds=time.perf_counter() - started,
        pretrain_seconds=pretrain_report.seconds,
    )
    return encoder, report


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------
def _row(label: str, metrics: ClassificationMetrics) -> str:
    return (
        f"    {label:<28s} bal-acc {metrics.balanced_accuracy:6.3f}   "
        f"macro-F1 {metrics.macro_f1:6.3f}   acc {metrics.accuracy:6.3f}"
    )


def _per_class(metrics: ClassificationMetrics) -> str:
    """recall/support per class - the only way to read a balanced average honestly."""
    return "  ".join(
        f"{name} {recall:.2f}/n={support}"
        for name, recall, support in zip(
            HEALTH_CLASSES, metrics.per_class_recall, metrics.support
        )
    )


def format_report(report: ProbeReport) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("Phase 3 gate - linear probe on frozen GraphSAGE embeddings")
    add("=" * 78)
    add(f"  node embedding dim {report.embed_dim}   pooled global dim {report.global_dim}")
    add(f"  training sizes  : {', '.join(report.train_sizes)}")
    add(f"  held-out sizes  : {', '.join(report.heldout_sizes)}")
    add(
        f"  pretrain graphs {report.n_pretrain_graphs}   "
        f"probe-fit nodes {report.n_probe_fit_nodes}   "
        f"pretrain {report.pretrain_seconds:.1f}s   total {report.seconds:.1f}s"
    )
    add(f"  classes: {', '.join(HEALTH_CLASSES)}")

    for split in report.splits.values():
        add("")
        add("-" * 78)
        add(f"  SPLIT: {split.name}  ({', '.join(split.sizes)})")
        counts = "  ".join(
            f"{name}={count}" for name, count in zip(HEALTH_CLASSES, split.class_counts)
        )
        add(f"    {split.n_nodes} nodes   {counts}")
        add("")
        add(_row("GraphSAGE embeddings", split.encoder))
        add(_row("baseline: majority class", split.majority))
        add(_row("baseline: uniform random", split.random))
        add(_row("control: raw features", split.raw_features))
        add(f"      per-class recall/support: {_per_class(split.encoder)}")
        add("")
        add("    per node type (probe embeddings vs that type's majority class):")
        for ntype in NODE_TYPES:
            enc = split.per_type_encoder.get(ntype)
            maj = split.per_type_majority.get(ntype)
            raw = split.per_type_raw.get(ntype)
            if enc is None or maj is None or raw is None:
                continue
            add(
                f"      {ntype:<8s} n={enc.n:<6d} "
                f"bal-acc {enc.balanced_accuracy:6.3f} (maj {maj.balanced_accuracy:.3f}, "
                f"raw {raw.balanced_accuracy:.3f})   "
                f"macro-F1 {enc.macro_f1:6.3f} (maj {maj.macro_f1:.3f}, "
                f"raw {raw.macro_f1:.3f})"
            )
            add(f"               recall/support: {_per_class(enc)}")
        add("")
        add("    per cluster size (probe embeddings vs majority class):")
        for size_label, enc in split.per_size_encoder.items():
            maj = split.per_size_majority[size_label]
            add(
                f"      {size_label:<14s} n={enc.n:<6d} "
                f"bal-acc {enc.balanced_accuracy:6.3f} (maj {maj.balanced_accuracy:.3f})   "
                f"macro-F1 {enc.macro_f1:6.3f} (maj {maj.macro_f1:.3f})"
            )

        add("")
        verdict = "PASS" if split.passed else "FAIL"
        add(f"    verdict: {verdict}")
        for problem in split.failures():
            add(f"      - {problem}")

    add("")
    add("=" * 78)
    add(f"  PHASE 3 GATE: {'PASS' if report.passed else 'FAIL'}")
    if not report.passed:
        add("  The encoder is NOT ready to wire into marl/ (PLAN.md section 3, Phase 3).")
    add("=" * 78)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smaller rollouts and fewer epochs - same pipeline, same gate",
    )
    parser.add_argument("--epochs", type=int, default=None, help="pretraining epochs")
    parser.add_argument("--seed", type=int, default=None, help="master seed")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = ProbeConfig.quick() if args.quick else ProbeConfig()
    if args.epochs is not None:
        config = replace(config, pretrain=replace(config.pretrain, epochs=args.epochs))
    if args.seed is not None:
        config = replace(
            config, seed=args.seed, pretrain=replace(config.pretrain, seed=args.seed)
        )

    encoder, report = run_probe(config, verbose=not args.quiet)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
