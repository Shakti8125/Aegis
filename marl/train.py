"""MAPPO training entrypoint. Checkpoints to marl/checkpoints/ alongside the config that produced them.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.

Usage
-----
    python -m marl.train --smoke                     # ~30 s sanity run
    python -m marl.train --updates 300 --envs 8      # a real run
    python -m marl.train --updates 300 --tune-baseline

What one run produces, all under ``marl/checkpoints/<run-id>/``:

``config.json``
    Every dataclass that shaped the run (train / MAPPO / reward / baseline),
    plus seed, library versions and the git commit.  A run is reproducible from
    this file alone.
``metrics.jsonl``
    One JSON object per update.  **Every reward component appears under its own
    key** (``reward/sla_violation``, ``reward/action_cost``, ...) - never a
    single collapsed scalar (CLAUDE.md).  Machine-readable so the Phase 6
    backend can serve training curves to the dashboard without re-parsing logs.
``update_<n>.pt`` / ``final.pt``
    Model + optimizer state, with the *config embedded in the checkpoint itself*
    so a checkpoint that gets moved is still reproducible.
``comparison.json``
    MAPPO vs the rule-based baseline vs no-op, per fault scenario, on the two
    PLAN.md metrics (time-to-recovery, SLA-violation count).

The training env is built by ``marl.vec_env`` with ``fixed_topology=False`` so
the policy sees a fresh topology every episode; evaluation uses the default
fixed topology per seed, so evaluation topologies are ones training never saw.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from simulator.cluster_env import ACTION_NAMES, N_ACTIONS, ClusterEnv
from marl.baseline import BaselineConfig, RuleBasedController, tune_baseline
from marl.evaluation import (
    NoOpController,
    PolicyController,
    ScenarioReport,
    beats,
    evaluate,
    format_comparison,
    format_reward_components,
)
from marl.mappo import MAPPO, MAPPOConfig, RolloutBuffer, compute_gae
from marl.reward import COMPONENT_NAMES, RewardConfig, RewardShaper
from marl.vec_env import (
    DEFAULT_EVAL_SCENARIOS,
    VecClusterEnv,
    make_scenario_env,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "marl" / "checkpoints"


# ==========================================================================
# Config
# ==========================================================================
@dataclass
class TrainConfig:
    """Everything that defines a run. Serialised verbatim next to the weights."""

    run_id: str = ""
    seed: int = 20240401

    # --- budget -------------------------------------------------------------
    updates: int = 300
    rollout_steps: int = 128
    n_envs: int = 8

    # --- environment --------------------------------------------------------
    train_scenario: str = "mixed"
    max_cycles: int = 200
    fixed_topology: bool = False

    # --- evaluation ---------------------------------------------------------
    eval_scenarios: tuple[str, ...] = DEFAULT_EVAL_SCENARIOS
    eval_episodes: int = 24
    eval_seed_base: int = 900_000
    tune_seed_base: int = 500_000
    tune_baseline: bool = False
    tune_episodes: int = 16
    eval_every: int = 0            # 0 = only at the end
    eval_every_episodes: int = 8   # cheap mid-run probe, "mixed" only

    # --- plumbing -----------------------------------------------------------
    checkpoint_every: int = 25
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    device: str = "cpu"
    torch_threads: int = 4
    log_every: int = 1

    @property
    def env_steps_per_update(self) -> int:
        return self.rollout_steps * self.n_envs

    @property
    def total_env_steps(self) -> int:
        return self.updates * self.env_steps_per_update

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["eval_scenarios"] = list(self.eval_scenarios)
        return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"


def _provenance() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


# ==========================================================================
# Logging
# ==========================================================================
class RunLogger:
    """stdout table + append-only JSONL, both fed from the same record."""

    HEADER = (
        f"{'upd':>5}{'steps':>10}{'sps':>8}"
        f"{'reward':>10}{'sla':>9}{'lat':>8}{'avail':>9}{'cost':>8}{'inval':>8}{'term':>8}"
        f"{'ep_len':>8}{'recov%':>8}{'sla_tk':>8}"
        f"{'v_loss':>9}{'p_loss':>9}{'ent':>7}{'ev':>7}"
    )

    def __init__(self, path: Path, log_every: int = 1) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._n = 0
        self.log_every = max(1, log_every)

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()
        if record.get("update", 0) % self.log_every:
            return
        if self._n % 20 == 0:
            print(self.HEADER, flush=True)
        self._n += 1
        r = record
        print(
            f"{r['update']:>5}{r['env_steps']:>10}{r['env_steps_per_sec']:>8.0f}"
            f"{r['reward/total']:>10.3f}"
            f"{r['reward/sla_violation']:>9.3f}"
            f"{r['reward/latency']:>8.3f}"
            f"{r['reward/availability']:>9.3f}"
            f"{r['reward/action_cost']:>8.3f}"
            f"{r['reward/invalid_action']:>8.3f}"
            f"{r['reward/terminal']:>8.3f}"
            f"{r['episode/length']:>8.1f}"
            f"{100.0 * r['episode/recovery_rate']:>8.1f}"
            f"{r['episode/sla_service_ticks']:>8.1f}"
            f"{r['loss/value']:>9.3f}{r['loss/policy']:>9.4f}"
            f"{r['loss/entropy']:>7.3f}{r['loss/explained_variance']:>7.2f}",
            flush=True,
        )

    def close(self) -> None:
        self._fh.close()


# ==========================================================================
# Trainer
# ==========================================================================
class Trainer:
    def __init__(
        self,
        train_config: TrainConfig | None = None,
        mappo_config: MAPPOConfig | None = None,
        reward_config: RewardConfig | None = None,
        baseline_config: BaselineConfig | None = None,
    ) -> None:
        self.cfg = train_config or TrainConfig()
        self.mappo_cfg = mappo_config or MAPPOConfig()
        self.reward_cfg = reward_config or RewardConfig()
        self.baseline_cfg = baseline_config or BaselineConfig()

        if not self.cfg.run_id:
            self.cfg.run_id = time.strftime("run-%Y%m%d-%H%M%S")
        self.run_dir = Path(self.cfg.checkpoint_dir) / self.cfg.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._seed_everything(self.cfg.seed)
        torch.set_num_threads(max(1, int(self.cfg.torch_threads)))

        self.shaper = RewardShaper(self.reward_cfg)
        self.vec = VecClusterEnv(
            env_fn=self._make_train_env,
            n_envs=self.cfg.n_envs,
            seeds=[self.cfg.seed + 1000 * i for i in range(self.cfg.n_envs)],
        )
        self.policy = MAPPO(
            obs_dim=self.vec.obs_dim,
            state_dim=self.vec.state_dim,
            n_agents=self.vec.n_agents,
            n_actions=N_ACTIONS,
            config=self.mappo_cfg,
            device=self.cfg.device,
        )
        self.buffer = RolloutBuffer(
            n_steps=self.cfg.rollout_steps,
            n_envs=self.cfg.n_envs,
            n_agents=self.vec.n_agents,
            obs_dim=self.vec.obs_dim,
            state_dim=self.vec.state_dim,
            component_names=COMPONENT_NAMES,
        )
        self.logger = RunLogger(self.run_dir / "metrics.jsonl", self.cfg.log_every)
        self._write_config()

        self.env_steps = 0
        self.agent_steps = 0
        self.t_start = time.perf_counter()
        self._recent_episodes: list[dict[str, float]] = []

    # ------------------------------------------------------------- plumbing
    def _seed_everything(self, seed: int) -> None:
        np.random.seed(seed % (2**31 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _make_train_env(self) -> ClusterEnv:
        return make_scenario_env(
            self.cfg.train_scenario,
            self.reward_cfg,
            max_cycles=self.cfg.max_cycles,
            fixed_topology=self.cfg.fixed_topology,
        )

    def _make_eval_env(self, scenario: str):
        def factory() -> ClusterEnv:
            return make_scenario_env(
                scenario, self.reward_cfg, max_cycles=self.cfg.max_cycles
            )

        return factory

    def config_payload(self) -> dict[str, Any]:
        return {
            "train": self.cfg.as_dict(),
            "mappo": self.mappo_cfg.as_dict(),
            "reward": self.reward_cfg.as_dict(),
            "baseline": self.baseline_cfg.as_dict(),
            "provenance": _provenance(),
        }

    def _write_config(self) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def checkpoint(self, update: int, tag: str | None = None) -> Path:
        """Weights + optimizer + the config that produced them, in one file."""
        name = tag or f"update_{update:05d}"
        path = self.run_dir / f"{name}.pt"
        torch.save(
            {
                "update": update,
                "env_steps": self.env_steps,
                "obs_dim": self.vec.obs_dim,
                "state_dim": self.vec.state_dim,
                "n_agents": self.vec.n_agents,
                "n_actions": N_ACTIONS,
                "config": self.config_payload(),
                **self.policy.state_dict(),
            },
            path,
        )
        return path

    # -------------------------------------------------------------- rollout
    def collect(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fill the buffer with ``rollout_steps`` on-policy transitions.

        ``VecClusterEnv`` hands back live internal buffers, so the rollout keeps
        its own copies: ``step`` overwrites ``obs``/``state`` in place and the
        transition we are about to store must be the *pre*-step one.
        """
        obs, state = obs.copy(), state.copy()
        self.buffer.reset()
        self.policy.train()
        for _ in range(self.cfg.rollout_steps):
            actions, logprobs, values = self.policy.act(obs, state)
            (
                next_obs,
                next_state,
                raw,
                terminated,
                truncated,
                final_state,
            ) = self.vec.step(actions)
            reward, components = self.shaper.shape(raw)
            self.buffer.add(
                obs=obs,
                state=state,
                action=actions,
                logprob=logprobs,
                value=values,
                reward=reward,
                components=components,
                terminated=terminated,
                truncated=truncated,
                final_state=final_state,
            )
            obs, state = next_obs.copy(), next_state.copy()
            self.env_steps += self.cfg.n_envs
            self.agent_steps += self.cfg.n_envs * self.vec.n_agents
        self._recent_episodes.extend(self.vec.drain_episode_stats())
        return obs, state

    def advantages(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GAE over the buffer, bootstrapping the tail with V(s_T)."""
        buf = self.buffer
        last_values = self.policy.value(state)                       # (E, N)
        # V(s_final) for episodes that ended inside the rollout. Only read where
        # `truncated`; compute_gae zeroes it wherever `terminated`.
        flat_final = buf.final_states.reshape(-1, buf.final_states.shape[-1])
        final_values = self.policy.value(flat_final).reshape(
            buf.n_steps, buf.n_envs, buf.n_agents
        )
        terminated = np.repeat(buf.terminated[:, :, None], buf.n_agents, axis=2)
        truncated = np.repeat(buf.truncated[:, :, None], buf.n_agents, axis=2)
        return compute_gae(
            rewards=buf.rewards,
            values=buf.values,
            terminated=terminated,
            truncated=truncated,
            final_values=final_values,
            last_values=last_values,
            gamma=self.mappo_cfg.gamma,
            gae_lambda=self.mappo_cfg.gae_lambda,
        )

    # ----------------------------------------------------------------- loop
    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        obs, state = self.vec.reset()
        print(
            f"[aegis] run {cfg.run_id}: {cfg.updates} updates x "
            f"{cfg.env_steps_per_update} env steps = {cfg.total_env_steps:,} env steps "
            f"({cfg.total_env_steps * self.vec.n_agents:,} agent steps), "
            f"{self.policy.n_parameters():,} params, device={cfg.device}",
            flush=True,
        )
        print(f"[aegis] artifacts -> {self.run_dir}", flush=True)

        last_t = time.perf_counter()
        last_steps = 0
        for update in range(1, cfg.updates + 1):
            obs, state = self.collect(obs, state)
            adv, returns = self.advantages(state)
            stats = self.policy.update(
                self.buffer, adv, returns, progress=(update - 1) / max(cfg.updates, 1)
            )

            now = time.perf_counter()
            sps = (self.env_steps - last_steps) / max(now - last_t, 1e-9)
            last_t, last_steps = now, self.env_steps

            record = self._record(update, stats, adv, returns, sps)
            self.logger.log(record)

            if cfg.checkpoint_every and update % cfg.checkpoint_every == 0:
                self.checkpoint(update)
            if cfg.eval_every and update % cfg.eval_every == 0:
                self._mid_run_eval(update)

        final_path = self.checkpoint(cfg.updates, tag="final")
        wall = time.perf_counter() - self.t_start
        print(
            f"\n[aegis] training done in {wall:.1f}s - "
            f"{self.env_steps:,} env steps ({self.env_steps / wall:,.0f}/s), "
            f"{self.agent_steps:,} agent steps ({self.agent_steps / wall:,.0f}/s), "
            f"{cfg.updates / wall:.2f} updates/s",
            flush=True,
        )
        print(f"[aegis] final checkpoint: {final_path}", flush=True)
        return {
            "run_dir": str(self.run_dir),
            "final_checkpoint": str(final_path),
            "wall_clock_s": wall,
            "env_steps": self.env_steps,
            "agent_steps": self.agent_steps,
            "env_steps_per_sec": self.env_steps / wall,
            "agent_steps_per_sec": self.agent_steps / wall,
            "updates_per_sec": cfg.updates / wall,
        }

    def _record(
        self,
        update: int,
        stats: dict[str, float],
        adv: np.ndarray,
        returns: np.ndarray,
        sps: float,
    ) -> dict[str, Any]:
        buf = self.buffer
        n_samples = buf.rewards.size
        # Per reward component, mean per agent-step. Separate keys, always.
        record: dict[str, Any] = {
            "update": update,
            "env_steps": self.env_steps,
            "agent_steps": self.agent_steps,
            "env_steps_per_sec": float(sps),
            "wall_clock_s": time.perf_counter() - self.t_start,
            "reward/total": float(buf.rewards.mean()),
            "reward/episode_return_est": float(buf.rewards.mean() * self.cfg.max_cycles),
        }
        for key in COMPONENT_NAMES:
            record[f"reward/{key}"] = float(buf.components[key].sum() / n_samples)
        # Reconstruction check: the components must add back up to the scalar.
        record["reward/component_sum"] = float(
            sum(record[f"reward/{k}"] for k in COMPONENT_NAMES)
        )

        actions = buf.actions.reshape(-1)
        counts = np.bincount(actions, minlength=N_ACTIONS) / max(actions.size, 1)
        for a in range(N_ACTIONS):
            record[f"action/{ACTION_NAMES[a]}"] = float(counts[a])

        eps = self._recent_episodes[-64:]
        record["episode/count"] = len(self._recent_episodes)
        record["episode/length"] = float(np.mean([e["length"] for e in eps])) if eps else 0.0
        record["episode/recovery_rate"] = (
            float(np.mean([e["recovered"] for e in eps])) if eps else 0.0
        )
        record["episode/collapse_rate"] = (
            float(np.mean([e["collapsed"] for e in eps])) if eps else 0.0
        )
        record["episode/sla_service_ticks"] = (
            float(np.mean([e["sla_service_ticks"] for e in eps])) if eps else 0.0
        )
        record["episode/final_mean_health"] = (
            float(np.mean([e["final_mean_health"] for e in eps])) if eps else 0.0
        )

        record["loss/policy"] = stats["policy_loss"]
        record["loss/value"] = stats["value_loss"]
        record["loss/entropy"] = stats["entropy"]
        record["loss/approx_kl"] = stats["approx_kl"]
        record["loss/clip_frac"] = stats["clip_frac"]
        record["loss/grad_norm"] = stats["grad_norm"]
        record["loss/explained_variance"] = stats["explained_variance"]
        record["loss/lr"] = stats["lr"]
        record["loss/early_stop"] = stats["early_stop"]
        record["adv/mean"] = float(adv.mean())
        record["adv/std"] = float(adv.std())
        record["value/return_mean"] = float(returns.mean())
        return record

    def _mid_run_eval(self, update: int) -> None:
        seeds = self._eval_seeds("mixed", self.cfg.eval_every_episodes)
        report, _ = evaluate(
            self._make_eval_env("mixed"),
            PolicyController(self.policy),
            seeds,
            "mixed",
            self.shaper,
        )
        print(
            f"  [eval @{update}] mixed: ttr={report.mean_ttr:.1f} "
            f"sla_svc_ticks={report.mean_sla_service_ticks:.1f} "
            f"recovered={100 * report.recovery_rate:.0f}%",
            flush=True,
        )
        self.policy.train()

    # ----------------------------------------------------------- evaluation
    def _eval_seeds(self, scenario: str, n: int) -> list[int]:
        """Fixed seeds, identical for every controller and every scenario.

        Same seeds across scenarios means the *topology* is held constant and
        only the fault family changes, so a per-scenario difference is a
        difference in fault handling and not in cluster shape.  Disjoint from
        :meth:`_tune_seeds` so baseline tuning cannot see the test set.
        """
        return [self.cfg.eval_seed_base + i for i in range(n)]

    def _tune_seeds(self, n: int) -> list[int]:
        return [self.cfg.tune_seed_base + i for i in range(n)]

    def compare(self) -> dict[str, Any]:
        """MAPPO vs the tuned rule-based baseline vs no-op, per fault scenario."""
        cfg = self.cfg
        baseline_cfg = self.baseline_cfg
        tuning_trace: list[dict[str, Any]] = []
        if cfg.tune_baseline:
            print("\n[aegis] tuning the rule-based baseline (disjoint seeds)...", flush=True)
            baseline_cfg, tuning_trace = tune_baseline(
                self._make_eval_env(cfg.train_scenario),
                self._tune_seeds(cfg.tune_episodes),
                scenario=cfg.train_scenario,
                base=self.baseline_cfg,
                verbose=True,
            )
            self.baseline_cfg = baseline_cfg
            self._write_config()
            print(f"[aegis] tuned baseline: {baseline_cfg.as_dict()}", flush=True)

        controllers = [
            PolicyController(self.policy, name="mappo"),
            RuleBasedController(baseline_cfg, name="baseline"),
            NoOpController(),
        ]

        reports: list[ScenarioReport] = []
        verdicts: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for scenario in cfg.eval_scenarios:
            seeds = self._eval_seeds(scenario, cfg.eval_episodes)
            factory = self._make_eval_env(scenario)
            per_scenario: dict[str, ScenarioReport] = {}
            for controller in controllers:
                report, _ = evaluate(factory, controller, seeds, scenario, self.shaper)
                per_scenario[controller.name] = report
                reports.append(report)
            verdicts.append(beats(per_scenario["mappo"], per_scenario["baseline"]))
        eval_wall = time.perf_counter() - t0

        print(
            format_comparison(
                reports,
                headline=(
                    f"MAPPO vs rule-based baseline - {cfg.eval_episodes} episodes/scenario, "
                    f"identical seeds, run {cfg.run_id}"
                ),
            )
        )
        print(format_reward_components(reports))
        print("\n-- verdict (PLAN.md Phase 4: beat the baseline on BOTH metrics) " + "-" * 30)
        print(
            f"{'scenario':<20}{'TTR delta':>12}{'TTR %':>9}"
            f"{'SLA delta':>12}{'SLA %':>9}{'beats both':>13}"
        )
        for v in verdicts:
            print(
                f"{v['scenario']:<20}{v['ttr_delta']:>12.1f}{v['ttr_pct']:>9.1f}"
                f"{v['sla_delta']:>12.1f}{v['sla_pct']:>9.1f}"
                f"{str(v['beats_both']):>13}"
            )
        n_both = sum(v["beats_both"] for v in verdicts)
        print(
            f"\n[aegis] MAPPO beats the baseline on both metrics in "
            f"{n_both}/{len(verdicts)} scenarios "
            f"(PLAN.md needs > 1). Evaluation wall clock: {eval_wall:.1f}s"
        )

        payload = {
            "run_id": cfg.run_id,
            "eval_episodes_per_scenario": cfg.eval_episodes,
            "reports": [r.as_dict() for r in reports],
            "verdicts": verdicts,
            "scenarios_won_on_both": n_both,
            "scenarios_total": len(verdicts),
            "baseline_config": baseline_cfg.as_dict(),
            "baseline_tuning_trace": tuning_trace,
            "eval_wall_clock_s": eval_wall,
        }
        (self.run_dir / "comparison.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[aegis] comparison -> {self.run_dir / 'comparison.json'}")
        return payload

    def close(self) -> None:
        self.logger.close()
        self.vec.close()


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aegis MAPPO training (PLAN.md Phase 4)")
    p.add_argument("--updates", type=int, default=None, help="number of PPO updates")
    p.add_argument("--rollout-steps", type=int, default=None, help="env steps per env per update")
    p.add_argument("--envs", type=int, default=None, help="parallel environments")
    p.add_argument("--total-env-steps", type=int, default=None,
                   help="alternative budget; overrides --updates")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-scenario", type=str, default=None)
    p.add_argument("--eval-scenarios", type=str, default=None,
                   help="comma-separated; default is all five")
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None,
                   help="mid-run probe every N updates (0 = off)")
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--tune-baseline", action="store_true")
    p.add_argument("--no-compare", action="store_true", help="skip the baseline comparison")
    p.add_argument("--smoke", action="store_true", help="tiny run, for CI / sanity")
    return p


def config_from_args(args: argparse.Namespace) -> tuple[TrainConfig, MAPPOConfig]:
    cfg = TrainConfig()
    mappo = MAPPOConfig()

    if args.smoke:
        cfg.updates = 3
        cfg.rollout_steps = 32
        cfg.n_envs = 2
        cfg.max_cycles = 60
        cfg.eval_episodes = 3
        cfg.eval_scenarios = ("pod_crash", "mixed")
        cfg.checkpoint_every = 2

    for attr, value in (
        ("updates", args.updates),
        ("rollout_steps", args.rollout_steps),
        ("n_envs", args.envs),
        ("seed", args.seed),
        ("train_scenario", args.train_scenario),
        ("eval_episodes", args.eval_episodes),
        ("eval_every", args.eval_every),
        ("max_cycles", args.max_cycles),
        ("checkpoint_every", args.checkpoint_every),
        ("checkpoint_dir", args.checkpoint_dir),
        ("run_id", args.run_id),
        ("device", args.device),
        ("torch_threads", args.torch_threads),
    ):
        if value is not None:
            setattr(cfg, attr, value)
    if args.tune_baseline:
        cfg.tune_baseline = True
    if args.eval_scenarios:
        cfg.eval_scenarios = tuple(s.strip() for s in args.eval_scenarios.split(",") if s.strip())
    if args.total_env_steps is not None:
        cfg.updates = max(1, args.total_env_steps // (cfg.rollout_steps * cfg.n_envs))

    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.ent_coef is not None:
        overrides["ent_coef"] = args.ent_coef
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if overrides:
        mappo = MAPPOConfig(**{**mappo.as_dict(), **overrides})
    return cfg, mappo


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, mappo_cfg = config_from_args(args)

    trainer = Trainer(cfg, mappo_cfg)
    try:
        summary = trainer.train()
        if not args.no_compare:
            comparison = trainer.compare()
            summary["scenarios_won_on_both"] = comparison["scenarios_won_on_both"]
            summary["scenarios_total"] = comparison["scenarios_total"]
        (trainer.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
