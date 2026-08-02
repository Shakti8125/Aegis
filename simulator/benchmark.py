"""Throughput harness for the simulated cluster.

Beyond PLAN.md section 2's file list, on purpose: reporting episodes/sec after
any change that could affect it is a standing requirement of the `sim-engineer`
role, so it deserves a repeatable script rather than an ad-hoc one-liner.

Method (all four of these matter for the number to mean anything):

* one **warm-up** run, discarded - it pays for numpy's first-touch costs
* **fixed seeds** - the same episodes are replayed by every timed run
* **uniformly random valid actions** over the full 6-action space, so the
  action mix is not biased toward cheap no-ops
* **best of N** timed runs over >= 200 episodes each, ``time.perf_counter``

"Step" is reported both ways, because they differ by a factor of ``n_services``:

* **env step** - one call to ``env.step(...)``, advancing every agent at once
* **agent step** - one agent's action within an env step (env steps x n_agents);
  this is the number an RL sample budget is usually quoted in

Two loops are timed:

* ``full``  - the real PettingZoo path: build an action dict, get five dicts back
* ``core``  - ``env._advance(np.ndarray)``, the same physics with no dict work

The gap between them is the per-step dict-construction overhead the ParallelEnv
API mandates.  **This design vectorizes within a step (across pods and
services), not across parallel envs** - so if that gap dominates, the fix for
Phase 4 is a vector-env wrapper around N independent ClusterEnvs, not breaking
the API.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from simulator.cluster_env import N_ACTIONS, ClusterConfig, ClusterEnv  # noqa: E402


@dataclass
class RunResult:
    episodes: int
    env_steps: int
    seconds: float

    @property
    def episodes_per_sec(self) -> float:
        return self.episodes / self.seconds

    @property
    def env_steps_per_sec(self) -> float:
        return self.env_steps / self.seconds

    @property
    def mean_episode_length(self) -> float:
        return self.env_steps / self.episodes

    @property
    def usec_per_env_step(self) -> float:
        return 1e6 * self.seconds / self.env_steps


def _run_full(env: ClusterEnv, episodes: int, seed: int) -> RunResult:
    """The real API path: caller builds an action dict, env returns five dicts."""
    rng = np.random.default_rng(seed)
    n = env.n_services
    env.reset(seed=seed)
    env_steps = 0
    done = 0
    t0 = time.perf_counter()
    while done < episodes:
        arr = rng.integers(0, N_ACTIONS, n)
        actions = {a: int(x) for a, x in zip(env.agents, arr)}
        env.step(actions)
        env_steps += 1
        if not env.agents:
            done += 1
            env.reset()
    elapsed = time.perf_counter() - t0
    return RunResult(episodes=done, env_steps=env_steps, seconds=elapsed)


def _run_core(env: ClusterEnv, episodes: int, seed: int) -> RunResult:
    """Identical physics and identical action stream, with no dict construction."""
    rng = np.random.default_rng(seed)
    n = env.n_services
    env.reset(seed=seed)
    env_steps = 0
    done = 0
    t0 = time.perf_counter()
    while done < episodes:
        arr = rng.integers(0, N_ACTIONS, n)
        env._advance(arr)
        env_steps += 1
        if env._terminated or env._truncated:
            done += 1
            env.reset()
    elapsed = time.perf_counter() - t0
    return RunResult(episodes=done, env_steps=env_steps, seconds=elapsed)


def _best_of(fn, env: ClusterEnv, episodes: int, seed: int, repeats: int) -> RunResult:
    fn(env, max(10, episodes // 10), seed)  # warm-up, discarded
    runs = [fn(env, episodes, seed) for _ in range(repeats)]
    return max(runs, key=lambda r: r.env_steps_per_sec)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20240101)
    p.add_argument("--services", type=int, default=None)
    args = p.parse_args()

    cfg = ClusterConfig() if args.services is None else ClusterConfig(n_services=args.services)
    env = ClusterEnv(cfg)

    print("=" * 74)
    print("Aegis simulator throughput")
    print("=" * 74)
    print(f"  machine        : {platform.platform()}")
    print(f"  python / numpy : {platform.python_version()} / {np.__version__}")
    print(
        f"  cluster        : {cfg.n_services} services (= {cfg.n_services} agents), "
        f"{cfg.n_nodes} nodes, <= {cfg.n_services * cfg.max_replicas} pod slots"
    )
    print(f"  obs / state    : {env.obs_dim} / {env.state_dim} floats")
    print(f"  max_cycles     : {cfg.max_cycles}")
    print(f"  faults enabled : {', '.join(f.name for f in cfg.enabled_faults)}")
    print(
        f"  protocol       : warm-up discarded, fixed seed {args.seed}, "
        f"uniform random actions, best of {args.repeats} x {args.episodes} episodes"
    )
    print()

    full = _best_of(_run_full, env, args.episodes, args.seed, args.repeats)
    core = _best_of(_run_core, env, args.episodes, args.seed, args.repeats)

    n_agents = cfg.n_services
    print("-- full PettingZoo path (env.step with action/return dicts) ----------")
    print(f"  episodes/sec           : {full.episodes_per_sec:12,.1f}")
    print(f"  env steps/sec          : {full.env_steps_per_sec:12,.0f}")
    print(f"  agent steps/sec        : {full.env_steps_per_sec * n_agents:12,.0f}")
    print(f"  mean episode length    : {full.mean_episode_length:12.1f} env steps")
    print(f"  time per env step      : {full.usec_per_env_step:12.1f} us")
    print()

    print("-- core physics only (env._advance, no dicts) ------------------------")
    print(f"  episodes/sec           : {core.episodes_per_sec:12,.1f}")
    print(f"  env steps/sec          : {core.env_steps_per_sec:12,.0f}")
    print(f"  time per env step      : {core.usec_per_env_step:12.1f} us")
    print()

    overhead = full.usec_per_env_step - core.usec_per_env_step
    share = 100.0 * overhead / full.usec_per_env_step
    print("-- PettingZoo dict overhead ------------------------------------------")
    print(f"  cost per env step      : {overhead:12.1f} us  ({share:.1f}% of a step)")
    print(f"  cost per agent step    : {overhead / n_agents:12.2f} us")
    print(
        f"  headroom if removed    : {core.env_steps_per_sec / full.env_steps_per_sec:12.2f}x"
    )
    print()
    print("  'env step'   = one env.step() call, all agents advanced together")
    print(f"  'agent step' = one agent within an env step (x{n_agents})")
    print(
        "  This design vectorizes WITHIN a step, not across envs. If the dict\n"
        "  overhead above dominates, Phase 4 should wrap N envs in a vector-env\n"
        "  rather than break the ParallelEnv API."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
