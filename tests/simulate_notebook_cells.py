"""Empirical Cell-by-Cell Simulation of notebooks/aegis_training.ipynb."""

import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "aegis_training.ipynb"

def test_cell_by_cell():
    print("=== Testing Cell 1: Environment / CUDA detection ===")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active device: {device}")

    print("\n=== Testing Cell 3: Stage 1 GraphSAGE probe ===")
    from encoder.pretrain import PretrainConfig
    from encoder.probe import run_probe, ProbeConfig, format_report
    cfg = ProbeConfig.quick()
    encoder, probe_report = run_probe(cfg, verbose=False)
    print(f"Probe passed: {probe_report.passed}")

    print("\n=== Testing Cell 4: Stage 1 HGT pretraining ===")
    from encoder.hgt_encoder import HGTGraphEncoder
    from encoder.features import FEATURE_DIMS, NODE_TYPES
    from encoder.dataset import TRAIN_SIZES, collect_sized_dataset, iter_all
    from encoder.gnn_model import EncoderConfig

    train_dataset = collect_sized_dataset(TRAIN_SIZES)
    train_graphs = list(iter_all(train_dataset))
    
    # Try the notebook's code vs what is valid
    try:
        hgt_encoder_broken = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2).to(device)
        print("Broken HGT call unexpectedly succeeded!")
    except Exception as e:
        print(f"Confirmed notebook Bug in Stage 1 HGT: {type(e).__name__}: {e}")

    # Valid HGT call
    hgt_encoder = HGTGraphEncoder(EncoderConfig(hidden_dim=64, num_layers=2)).to(device)
    decoders = nn.ModuleDict({
        ntype: nn.Linear(64, FEATURE_DIMS[ntype]) for ntype in NODE_TYPES
    }).to(device)
    print("Valid HGT initialization succeeded.")

    print("\n=== Testing Cell 5 & 6: Stage 2 Offline trajectories & DT ===")
    from simulator.cluster_env import ClusterConfig, ClusterEnv
    from marl.vec_env import scenario_overrides
    from marl.reward import RewardConfig, RewardShaper
    from marl.decision_transformer import DecisionTransformer

    cfg_kwargs = scenario_overrides("pod_crash", max_cycles=10)
    env = ClusterEnv(config=ClusterConfig(**cfg_kwargs))
    obs, infos = env.reset(seed=42)
    reward_shaper = RewardShaper(RewardConfig())

    states, actions, ep_rewards = [], [], []
    for step in range(5):
        state_vec = env.state()
        act_dict = {agent: env.action_space(agent).sample() for agent in env.agents}
        next_obs, raw_rews, terms, truncs, infos = env.step(act_dict)
        r_scalar, _ = reward_shaper.shape_infos(infos, env.possible_agents)

        states.append(state_vec)
        actions.append([act_dict.get(f"service_{i}", 0) for i in range(env.n_services)])
        ep_rewards.append(float(r_scalar.mean()))

    env.close()
    returns_to_go = []
    discounted_sum = 0.0
    for r in reversed(ep_rewards):
        discounted_sum = r + 0.99 * discounted_sum
        returns_to_go.insert(0, discounted_sum)

    state_dim = np.array(states).shape[1]
    n_actions = 6
    dt_model = DecisionTransformer(state_dim=state_dim, n_actions=n_actions).to(device)
    print("Decision Transformer initialized and tested.")

    print("\n=== Testing Cell 7 & 8: Stage 3 HAPPO & QMIX ===")
    from marl.happo import HAPPO
    from marl.qmix import QMIX
    from marl.mappo import RolloutBuffer
    from marl.reward import COMPONENT_NAMES

    obs_dim = 38
    state_dim = 143
    n_agents = 12
    happo_trainer = HAPPO(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)
    qmix_trainer = QMIX(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)

    # Test broken RolloutBuffer call from notebook
    try:
        buf_broken = RolloutBuffer(n_steps=4, n_envs=2, n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim)
        print("Broken RolloutBuffer call unexpectedly succeeded!")
    except Exception as e:
        print(f"Confirmed notebook Bug in Stage 3 RolloutBuffer: {type(e).__name__}: {e}")

    buf_valid = RolloutBuffer(n_steps=4, n_envs=2, n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim, component_names=COMPONENT_NAMES)
    print("Valid RolloutBuffer call succeeded.")

    print("\n=== Testing Cell 9: Stage 4 Evaluation & Benchmark ===")
    from marl.evaluation import evaluate, beats, PolicyController, NoOpController, format_comparison, format_reward_components
    from marl.baseline import RuleBasedController
    from marl.mappo import MAPPO

    def make_env(scenario="pod_crash"):
        return ClusterEnv(config=ClusterConfig(**scenario_overrides(scenario, max_cycles=10)))

    eval_seeds = [900001, 900002]
    noop_ctrl = NoOpController()
    rule_ctrl = RuleBasedController()
    mappo_model = MAPPO(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)
    policy_ctrl = PolicyController(mappo_model, name="mappo")

    sc = "pod_crash"
    report_noop, _ = evaluate(lambda: make_env(sc), noop_ctrl, eval_seeds, scenario=sc)
    report_rule, _ = evaluate(lambda: make_env(sc), rule_ctrl, eval_seeds, scenario="pod_crash")
    report_policy, _ = evaluate(lambda: make_env(sc), policy_ctrl, eval_seeds, scenario="pod_crash")

    reports = [report_noop, report_rule, report_policy]
    verdict_policy_vs_rule = beats(report_policy, report_rule)
    verdict_rule_vs_noop = beats(report_rule, report_noop)
    print("Comparison table:")
    print(format_comparison(reports))
    print("Reward breakdown table:")
    print(format_reward_components(reports))
    print("Stage 4 evaluation succeeded.")

if __name__ == "__main__":
    test_cell_by_cell()
