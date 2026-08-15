"""The Phase 3 gate itself: the probe beats the trivial baselines, and cannot be faked.

The important test here is not "the probe scores well" - it is
``test_a_majority_class_predictor_cannot_pass_the_gate``. If a constant
predictor could clear the bar, the bar would be measuring the class imbalance
rather than the encoder.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("torch", reason="Phase 3 needs torch (see requirements.txt)")
pytest.importorskip(
    "torch_geometric", reason="Phase 3 needs torch-geometric (see requirements.txt)"
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from encoder.probe import (  # noqa: E402
    GATE_MIN_BALANCED_ACCURACY,
    GATE_MIN_MACRO_F1_MARGIN,
    N_CLASSES,
    ProbeConfig,
    evaluate_predictions,
    fit_linear_probe,
    format_report,
    gate_passes,
    run_probe,
)


@pytest.fixture(scope="module")
def quick_report():
    """The real pipeline at a smaller scale - same code path as the CLI gate."""
    encoder, report = run_probe(ProbeConfig.quick(), verbose=False)
    return report


# ------------------------------------------------------------------- metrics
def test_metrics_match_a_hand_computed_confusion_matrix():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.support == (3, 2, 1)
    assert metrics.accuracy == pytest.approx(4 / 6)
    # recalls: 2/3, 1/2, 1/1
    assert metrics.balanced_accuracy == pytest.approx((2 / 3 + 1 / 2 + 1.0) / 3)
    # f1: healthy 2*1*(2/3)/(1+2/3)=0.8, degraded 2*.5*.5/1=0.5, critical 2*.5*1/1.5=2/3
    assert metrics.macro_f1 == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_a_constant_predictor_scores_exactly_chance_on_balanced_accuracy():
    y_true = np.array([0] * 90 + [1] * 7 + [2] * 3)
    metrics = evaluate_predictions(y_true, np.zeros_like(y_true))
    assert metrics.accuracy == pytest.approx(0.90)  # the number that would mislead
    assert metrics.balanced_accuracy == pytest.approx(1 / 3)
    assert metrics.macro_f1 < 0.35


def test_absent_classes_do_not_distort_the_macro_averages():
    y_true = np.array([0, 0, 1, 1])
    metrics = evaluate_predictions(y_true, y_true)
    assert metrics.support == (2, 2, 0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)


# ------------------------------------------------------------- the probe is linear
def test_the_probe_is_a_single_linear_map():
    x = torch.randn(200, 8)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()
    probe = fit_linear_probe(x, y, epochs=50)
    assert probe.weight.shape == (N_CLASSES, 8)
    assert probe.bias.shape == (N_CLASSES,)
    assert probe.predict(x).shape == (200,)


def test_the_probe_is_fitted_with_class_weights_so_it_cannot_collapse():
    """A 95/4/1 split must not train the probe into a constant predictor."""
    generator = torch.Generator().manual_seed(0)
    y = torch.tensor([0] * 950 + [1] * 40 + [2] * 10)
    x = torch.randn(1000, 4, generator=generator) + torch.nn.functional.one_hot(
        y, 4
    ).float() * 6.0
    probe = fit_linear_probe(x, y, epochs=300)
    predictions = probe.predict(x)
    assert len(set(predictions.tolist())) == N_CLASSES, "probe collapsed onto one class"
    assert evaluate_predictions(y.numpy(), predictions).balanced_accuracy > 0.9


# ------------------------------------------------------------------ the gate
def test_quick_probe_beats_the_trivial_baselines_at_both_size_regimes(quick_report):
    assert set(quick_report.splits) == {"training sizes", "held-out sizes"}
    for name, split in quick_report.splits.items():
        assert split.n_nodes > 0, name
        assert split.encoder.balanced_accuracy > split.majority.balanced_accuracy, name
        assert split.encoder.balanced_accuracy > split.random.balanced_accuracy, name
        assert split.encoder.macro_f1 > split.majority.macro_f1 + GATE_MIN_MACRO_F1_MARGIN, name
        assert split.encoder.balanced_accuracy >= GATE_MIN_BALANCED_ACCURACY, name


def test_held_out_sizes_are_genuinely_never_fitted_on():
    config = ProbeConfig.quick()
    fitted = {size.label for size in config.train_sizes}
    held_out = {size.label for size in config.heldout_sizes}
    assert fitted.isdisjoint(held_out)
    seeds = {
        config.encoder_rollout.seed,
        config.probe_fit_rollout.seed,
        config.eval_rollout.seed,
        config.heldout_rollout.seed,
    }
    assert len(seeds) == 4, "the four seed blocks must not overlap"


def test_the_gate_passes_on_both_splits(quick_report):
    assert quick_report.passed, quick_report.failures()
    for split in quick_report.splits.values():
        assert gate_passes(split), split.failures()


def test_a_majority_class_predictor_cannot_pass_the_gate(quick_report):
    """Substitute the trivial baseline's own scores for the probe's and re-gate."""
    for split in quick_report.splits.values():
        faked = replace(
            split,
            encoder=split.majority,
            per_type_encoder=dict(split.per_type_majority),
        )
        assert not gate_passes(faked), (
            "a constant predictor cleared the Phase 3 gate - the gate is measuring "
            "class imbalance, not the encoder"
        )
        assert faked.failures()


def test_a_uniform_random_predictor_cannot_pass_the_gate(quick_report):
    for split in quick_report.splits.values():
        faked = replace(
            split,
            encoder=split.random,
            per_type_encoder={k: split.random for k in split.per_type_encoder},
        )
        assert not gate_passes(faked)


def test_report_names_the_sizes_and_the_verdict(quick_report):
    text = format_report(quick_report)
    for label in quick_report.train_sizes + quick_report.heldout_sizes:
        assert label in text
    assert "PHASE 3 GATE" in text
    assert "majority class" in text
    assert ("PASS" if quick_report.passed else "FAIL") in text
