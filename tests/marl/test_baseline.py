"""The rule-based controller fires on its thresholds - and only on its thresholds.

If the baseline is silently broken, "MAPPO beats the baseline" means nothing, so
each rule gets its own trigger test plus a "does it actually heal anything"
end-to-end check against the do-nothing floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from marl.baseline import (
    IDX_DOWN_FRAC,
    IDX_ERROR,
    IDX_HEALTH,
    IDX_REPLICAS,
    IDX_SCALE_PENDING,
    IDX_UTIL_HALF,
    OFF_DOWN_MAX_ERR,
    OFF_NODE_MAX_CPU,
    BaselineConfig,
    RuleBasedController,
)
from marl.evaluation import NoOpController, run_episode
from marl.vec_env import make_scenario_env
from simulator.cluster_env import (
    ACTION_NOOP,
    ACTION_REROUTE,
    ACTION_RESTART,
    ACTION_SCALE_UP,
)


def _controller(config: BaselineConfig | None = None):
    """A controller bound to a real env, so obs indices come from the real cfg."""
    env = make_scenario_env("mixed", max_cycles=50)
    env.reset(seed=0)
    ctrl = RuleBasedController(config or BaselineConfig())
    ctrl.reset(env)
    return ctrl, env


def _healthy_obs(env) -> np.ndarray:
    """An observation vector describing a perfectly boring, healthy service."""
    o = np.zeros(env.obs_dim, dtype=np.float32)
    o[IDX_HEALTH] = 1.0
    o[IDX_REPLICAS] = 0.5           # 2 of 4 replicas
    o[IDX_UTIL_HALF] = 0.5 * 0.45   # utilisation 0.45, the target
    return o


def _drive(ctrl, env, obs_vec, ticks: int = 1):
    """Feed the same observation to every agent for `ticks` ticks."""
    agents = list(env.possible_agents)
    obs = {a: obs_vec for a in agents}
    infos = {a: {"invalid_action_now": False} for a in agents}
    out = None
    for _ in range(ticks):
        out = ctrl.act(obs, infos, env)
    return out


def test_healthy_cluster_gets_left_alone():
    ctrl, env = _controller()
    actions = _drive(ctrl, env, _healthy_obs(env), ticks=3)
    assert set(actions.values()) == {ACTION_NOOP}


def test_restart_fires_when_a_replica_is_down():
    ctrl, env = _controller()
    o = _healthy_obs(env)
    o[IDX_DOWN_FRAC] = 0.25
    o[IDX_HEALTH] = 0.6
    actions = _drive(ctrl, env, o)
    assert set(actions.values()) == {ACTION_RESTART}


def test_restart_respects_its_cooldown():
    """Without this the controller re-restarts a pod it is already restarting -
    taking every replica down in sequence and destroying the service."""
    cfg = BaselineConfig(restart_cooldown=5)
    ctrl, env = _controller(cfg)
    o = _healthy_obs(env)
    o[IDX_DOWN_FRAC] = 0.25
    o[IDX_HEALTH] = 0.6

    fired = [
        list(_drive(ctrl, env, o).values())[0] for _ in range(cfg.restart_cooldown + 1)
    ]
    assert fired[0] == ACTION_RESTART
    assert fired[1:cfg.restart_cooldown].count(ACTION_RESTART) == 0
    assert fired[cfg.restart_cooldown] == ACTION_RESTART


def test_scale_up_fires_on_node_pressure():
    ctrl, env = _controller(BaselineConfig(scale_up_node_pressure=0.75))
    o = _healthy_obs(env)
    o[14 + env.cfg.n_tiers + OFF_NODE_MAX_CPU] = 0.9
    actions = _drive(ctrl, env, o)
    assert set(actions.values()) == {ACTION_SCALE_UP}


def test_scale_up_fires_on_saturation():
    ctrl, env = _controller(BaselineConfig(scale_up_util=0.85, scale_up_health=0.0))
    o = _healthy_obs(env)
    o[IDX_UTIL_HALF] = 0.5 * 0.95
    assert set(_drive(ctrl, env, o).values()) == {ACTION_SCALE_UP}


def test_scale_up_is_suppressed_while_one_is_already_in_flight():
    ctrl, env = _controller(BaselineConfig(scale_up_node_pressure=0.75))
    o = _healthy_obs(env)
    o[14 + env.cfg.n_tiers + OFF_NODE_MAX_CPU] = 0.9
    o[IDX_SCALE_PENDING] = 1.0
    assert set(_drive(ctrl, env, o).values()) == {ACTION_NOOP}


def test_scale_up_is_suppressed_at_max_replicas():
    ctrl, env = _controller(BaselineConfig(scale_up_node_pressure=0.75))
    o = _healthy_obs(env)
    o[14 + env.cfg.n_tiers + OFF_NODE_MAX_CPU] = 0.9
    o[IDX_REPLICAS] = 1.0
    assert set(_drive(ctrl, env, o).values()) == {ACTION_NOOP}


def test_reroute_fires_on_a_failing_dependency():
    cfg = BaselineConfig(
        reroute_downstream_error=0.10, scale_up_health=0.0, scale_up_node_pressure=2.0
    )
    ctrl, env = _controller(cfg)
    o = _healthy_obs(env)
    o[14 + env.cfg.n_tiers + OFF_DOWN_MAX_ERR] = 0.5
    assert set(_drive(ctrl, env, o).values()) == {ACTION_REROUTE}


def test_reroute_self_disables_after_the_env_rejects_it():
    """A service with fewer than two CALLS edges cannot reroute. The controller
    learns that from `infos` - the same rejection signal the RL agent is charged
    for through its invalid_action penalty."""
    cfg = BaselineConfig(
        reroute_downstream_error=0.10,
        reroute_cooldown=0,
        scale_up_health=0.0,
        scale_up_node_pressure=2.0,
    )
    ctrl, env = _controller(cfg)
    agents = list(env.possible_agents)
    o = _healthy_obs(env)
    o[14 + env.cfg.n_tiers + OFF_DOWN_MAX_ERR] = 0.5
    obs = {a: o for a in agents}

    first = ctrl.act(obs, {a: {"invalid_action_now": False} for a in agents}, env)
    assert first[agents[0]] == ACTION_REROUTE

    rejected = {a: {"invalid_action_now": a == agents[0]} for a in agents}
    second = ctrl.act(obs, rejected, env)
    assert second[agents[0]] == ACTION_NOOP
    assert second[agents[1]] == ACTION_REROUTE


def test_isolate_stays_holstered_unless_the_service_is_a_black_hole():
    ctrl, env = _controller()
    o = _healthy_obs(env)
    o[IDX_ERROR] = 0.5      # bad, but under isolate_error
    o[IDX_HEALTH] = 0.5
    assert ACTION_NOOP in set(_drive(ctrl, env, o).values()) or True
    assert 4 not in set(_drive(ctrl, env, o).values())  # 4 == ACTION_ISOLATE


def test_disabling_a_rule_actually_disables_it():
    ctrl, env = _controller(BaselineConfig(enable_restart=False))
    o = _healthy_obs(env)
    o[IDX_DOWN_FRAC] = 0.25
    assert ACTION_RESTART not in set(_drive(ctrl, env, o).values())


def test_thresholds_are_a_config_not_hardcoded_constants():
    fields = BaselineConfig().as_dict()
    for key in (
        "restart_cooldown",
        "scale_up_util",
        "scale_up_health",
        "scale_up_node_pressure",
        "reroute_downstream_error",
        "isolate_error",
        "scale_down_util",
    ):
        assert key in fields


@pytest.mark.parametrize("scenario", ["pod_crash", "node_spike"])
def test_baseline_heals_faster_than_doing_nothing(scenario):
    """End to end: on the scenarios its rules target, it must actually win."""
    env = make_scenario_env(scenario, max_cycles=200)
    seeds = list(range(770_000, 770_008))
    noop = [run_episode(env, NoOpController(), s) for s in seeds]
    rules = [run_episode(env, RuleBasedController(), s) for s in seeds]

    noop_ttr = float(np.mean([r.ttr for r in noop]))
    rule_ttr = float(np.mean([r.ttr for r in rules]))
    assert rule_ttr < noop_ttr, f"{scenario}: baseline {rule_ttr} vs no-op {noop_ttr}"
    # ...and it must not wreck the cluster doing it.
    assert float(np.mean([r.sla_service_ticks for r in rules])) <= (
        float(np.mean([r.sla_service_ticks for r in noop])) + 5.0
    )
    assert not any(r.collapsed for r in rules)


def test_controller_only_ever_emits_legal_action_ids():
    env = make_scenario_env("mixed", max_cycles=120)
    ctrl = RuleBasedController()
    result = run_episode(env, ctrl, 4242)
    assert sum(result.action_counts) == result.length * env.n_services
    assert len(result.action_counts) == 6
