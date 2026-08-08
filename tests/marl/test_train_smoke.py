"""End-to-end smoke: a couple of updates, a checkpoint, and separated reward logs.

Deliberately tiny (a few thousand env steps) so it belongs in ``pytest tests/``.
It is not a training test - it is a wiring test for the three things a run has to
produce to be trustworthy: no NaNs, per-component reward logs, and a checkpoint
that carries the config that produced it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from marl.evaluation import PolicyController, evaluate
from marl.mappo import MAPPOConfig
from marl.reward import COMPONENT_NAMES
from marl.train import TrainConfig, Trainer, config_from_args, build_parser


def _tiny_config(tmp_path, **overrides) -> TrainConfig:
    cfg = TrainConfig(
        run_id="pytest-smoke",
        seed=1234,
        updates=2,
        rollout_steps=16,
        n_envs=2,
        max_cycles=40,
        checkpoint_every=1,
        checkpoint_dir=str(tmp_path),
        eval_episodes=2,
        eval_scenarios=("pod_crash", "mixed"),
        torch_threads=1,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("marl-smoke")
    trainer = Trainer(
        _tiny_config(tmp_path),
        MAPPOConfig(hidden_dim=32, num_minibatches=2, update_epochs=2),
    )
    try:
        summary = trainer.train()
        comparison = trainer.compare()
        yield trainer, summary, comparison
    finally:
        trainer.close()


def test_training_runs_without_nans(smoke_run):
    trainer, _, _ = smoke_run
    for module in (trainer.policy.actor, trainer.policy.critic):
        for p in module.parameters():
            assert torch.isfinite(p).all(), "policy weights went non-finite"
    assert np.all(np.isfinite(trainer.buffer.rewards))
    assert np.all(np.isfinite(trainer.buffer.values))


def test_metrics_jsonl_logs_every_reward_component_separately(smoke_run):
    trainer, _, _ = smoke_run
    lines = (trainer.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == trainer.cfg.updates
    for line in lines:
        record = json.loads(line)
        for key in COMPONENT_NAMES:
            assert f"reward/{key}" in record, f"missing reward/{key}"
            assert np.isfinite(record[f"reward/{key}"])
        # The components must reconstruct the scalar, not replace it.
        assert record["reward/component_sum"] == pytest.approx(
            record["reward/total"], rel=1e-4, abs=1e-5
        )
        for key in ("loss/policy", "loss/value", "loss/entropy", "env_steps_per_sec"):
            assert key in record


def test_checkpoint_is_written_alongside_the_config_that_produced_it(smoke_run):
    trainer, summary, _ = smoke_run
    run_dir = trainer.run_dir

    config_path = run_dir / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for section in ("train", "mappo", "reward", "baseline", "provenance"):
        assert section in config
    assert config["train"]["seed"] == trainer.cfg.seed

    checkpoints = sorted(run_dir.glob("*.pt"))
    assert checkpoints, "no checkpoint written"
    assert (run_dir / "final.pt").exists()

    payload = torch.load(run_dir / "final.pt", weights_only=False)
    # The config travels *inside* the checkpoint too, so a moved file is still
    # reproducible on its own.
    assert payload["config"]["train"]["seed"] == trainer.cfg.seed
    assert payload["config"]["reward"]["w_sla"] > 0
    for key in ("actor", "critic", "encoder", "optimizer"):
        assert key in payload
    assert payload["obs_dim"] == trainer.vec.obs_dim
    assert payload["state_dim"] == trainer.vec.state_dim


def test_checkpoint_reloads_into_a_working_policy(smoke_run):
    trainer, _, _ = smoke_run
    payload = torch.load(trainer.run_dir / "final.pt", weights_only=False)

    from marl.mappo import MAPPO

    restored = MAPPO(
        payload["obs_dim"],
        payload["state_dim"],
        payload["n_agents"],
        payload["n_actions"],
        MAPPOConfig(**payload["config"]["mappo"]),
    )
    restored.load_state_dict(payload)

    rng = np.random.default_rng(0)
    obs = rng.random((1, payload["n_agents"], payload["obs_dim"]), dtype=np.float32)
    state = rng.random((1, payload["state_dim"]), dtype=np.float32)
    a_new, _, _ = restored.act(obs, state, deterministic=True)
    a_old, _, _ = trainer.policy.act(obs, state, deterministic=True)
    np.testing.assert_array_equal(a_new, a_old)


def test_comparison_covers_more_than_one_fault_scenario(smoke_run):
    trainer, _, comparison = smoke_run
    scenarios = {r["scenario"] for r in comparison["reports"]}
    assert len(scenarios) >= 2, "PLAN.md needs more than one fault scenario"
    controllers = {r["controller"] for r in comparison["reports"]}
    assert {"mappo", "baseline", "no-op"} <= controllers

    for report in comparison["reports"]:
        assert set(report["mean_reward_components"]) == set(COMPONENT_NAMES)
        assert np.isfinite(report["mean_ttr"])
        assert report["mean_sla_service_ticks"] >= 0

    assert (trainer.run_dir / "comparison.json").exists()
    for verdict in comparison["verdicts"]:
        for key in ("ttr_delta", "sla_delta", "beats_ttr", "beats_sla", "beats_both"):
            assert key in verdict


def test_evaluation_is_reproducible_for_a_fixed_policy(smoke_run):
    """Identical seeds must replay identical episodes - the whole comparison
    rests on both controllers facing the same faults."""
    trainer, _, _ = smoke_run
    factory = trainer._make_eval_env("pod_crash")
    controller = PolicyController(trainer.policy)
    seeds = [770_100, 770_101]
    first, _ = evaluate(factory, controller, seeds, "pod_crash", trainer.shaper)
    second, _ = evaluate(factory, controller, seeds, "pod_crash", trainer.shaper)
    assert first.mean_ttr == pytest.approx(second.mean_ttr)
    assert first.mean_sla_service_ticks == pytest.approx(second.mean_sla_service_ticks)
    assert first.mean_total_reward == pytest.approx(second.mean_total_reward)


def test_cli_smoke_flag_produces_a_runnable_budget():
    args = build_parser().parse_args(["--smoke"])
    cfg, _ = config_from_args(args)
    assert cfg.updates * cfg.rollout_steps * cfg.n_envs < 5_000
    assert len(cfg.eval_scenarios) >= 2


def test_cli_budget_flags_override_the_defaults():
    args = build_parser().parse_args(
        ["--updates", "7", "--rollout-steps", "64", "--envs", "3", "--seed", "9"]
    )
    cfg, _ = config_from_args(args)
    assert (cfg.updates, cfg.rollout_steps, cfg.n_envs, cfg.seed) == (7, 64, 3, 9)
    assert cfg.total_env_steps == 7 * 64 * 3

    args = build_parser().parse_args(["--total-env-steps", "2048", "--envs", "2",
                                      "--rollout-steps", "32"])
    cfg, _ = config_from_args(args)
    assert cfg.updates == 2048 // (32 * 2)
