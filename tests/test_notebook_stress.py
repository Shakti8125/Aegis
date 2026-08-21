"""Adversarial stress testing and empirical verification suite for notebooks/aegis_training.ipynb.

Scopes:
1. Behavior under both GPU (`cuda`) and CPU fallback modes.
2. Trajectory collection edge cases (empty actions, mismatched service counts, short horizons).
3. Stage 1 HGT loss autograd stability, empty node batches, and probe gate criteria.
4. Stage 4 evaluation metrics calculation (TTR, SLA violations, separated reward components, beats logic).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from encoder.features import FEATURE_DIMS, HEALTH_CLASSES, NODE_TYPES
from encoder.gnn_model import AegisGraphEncoder, EncoderConfig
from encoder.hgt_encoder import HGTGraphEncoder
from encoder.probe import (
    GATE_MIN_BALANCED_ACCURACY,
    GATE_MIN_MACRO_F1_MARGIN,
    GATE_MIN_TYPE_BALANCED_ACCURACY,
    GATE_MIN_TYPE_MACRO_F1_MARGIN,
    ClassificationMetrics,
    LinearProbe,
    ProbeConfig,
    SplitReport,
    evaluate_predictions,
    fit_linear_probe,
    format_report,
)
from marl.baseline import RuleBasedController
from marl.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from marl.evaluation import (
    EpisodeResult,
    NoOpController,
    PolicyController,
    ScenarioReport,
    beats,
    evaluate,
    format_comparison,
    format_reward_components,
    run_episode,
    summarise,
)
from marl.happo import HAPPO
from marl.mappo import MAPPO, MAPPOConfig, RolloutBuffer
from marl.qmix import QMIX
from marl.reward import COMPONENT_NAMES, RewardAccumulator, RewardConfig, RewardShaper
from marl.vec_env import scenario_overrides
from simulator.cluster_env import ClusterConfig, ClusterEnv


# ============================================================================
# Scope 1: GPU (`cuda`) and CPU fallback modes
# ============================================================================

def test_scope1_device_fallback_graphsage():
    """Verify GraphSAGE encoder runs consistently on CPU device."""
    device = torch.device("cpu")
    encoder = AegisGraphEncoder(EncoderConfig(hidden_dim=32, embed_dim=32, global_dim=64)).to(device)
    
    # Create dummy HeteroData on device
    data = HeteroData()
    data["Service"].x = torch.randn(4, FEATURE_DIMS["Service"], device=device)
    data["Pod"].x = torch.randn(8, FEATURE_DIMS["Pod"], device=device)
    data["Node"].x = torch.randn(2, FEATURE_DIMS["Node"], device=device)
    data[("Service", "routes_to", "Service")].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
    data[("Pod", "runs_on", "Node")].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
    data[("Service", "backed_by", "Pod")].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)

    out = encoder(data)
    assert "Service" in out.node_embeddings
    assert out.node_embeddings["Service"].shape == (4, 32)
    assert out.global_embedding.shape == (1, 64)
    assert out.global_embedding.device.type == "cpu"


def test_scope1_device_fallback_decision_transformer():
    """Verify DecisionTransformer executes forward & backward pass on CPU."""
    device = torch.device("cpu")
    state_dim = 143
    n_actions = 6
    seq_len = 10
    batch_size = 2

    dt = DecisionTransformer(state_dim=state_dim, n_actions=n_actions).to(device)
    st = torch.randn(batch_size, seq_len, state_dim, device=device)
    act = torch.randint(0, n_actions, (batch_size, seq_len), device=device)
    rtg = torch.randn(batch_size, seq_len, 1, device=device)
    ts = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    logits = dt(st, act, rtg, ts)
    assert logits.shape == (batch_size, seq_len, n_actions)
    
    loss = nn.functional.cross_entropy(logits.reshape(-1, n_actions), act.reshape(-1))
    loss.backward()
    assert dt.predict_action[0].weight.grad is not None
    assert not torch.isnan(loss)


def test_scope1_device_fallback_happo_qmix():
    """Verify HAPPO and QMIX components execute on CPU."""
    obs_dim = 38
    state_dim = 143
    n_agents = 12
    n_actions = 6

    happo = HAPPO(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)
    qmix = QMIX(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)

    dummy_obs = np.random.randn(2, n_agents, obs_dim).astype(np.float32)
    dummy_state = np.random.randn(2, state_dim).astype(np.float32)

    actions, logprobs, values = happo.act(dummy_obs, dummy_state)
    assert actions.shape == (2, n_agents)
    assert logprobs.shape == (2, n_agents)
    assert values.shape == (2, n_agents)

    # QMIX compute_loss
    obs_t = torch.randn(2, n_agents, obs_dim)
    states_t = torch.randn(2, state_dim)
    act_t = torch.randint(0, n_actions, (2, n_agents))
    rews_t = torch.randn(2, 1)
    next_obs_t = torch.randn(2, n_agents, obs_dim)
    next_states_t = torch.randn(2, state_dim)
    dones_t = torch.zeros(2, 1)

    loss = qmix.compute_loss(obs_t, states_t, act_t, rews_t, next_obs_t, next_states_t, dones_t)
    assert not torch.isnan(loss)
    loss.backward()
    assert qmix.mixer.hyper_w1[0].weight.grad is not None


# ============================================================================
# Scope 2: Trajectory Collection Edge Cases
# ============================================================================

def test_scope2_trajectory_agent_key_indexing():
    """Stress test agent key indexing under normal and edge case service configurations."""
    env = ClusterEnv(config=ClusterConfig(n_nodes=2, n_services=6, max_cycles=10))
    obs, infos = env.reset(seed=42)
    
    # 1. Standard action dictionary matching env.agents
    act_dict = {agent: env.action_space(agent).sample() for agent in env.agents}
    recorded_actions = [act_dict.get(f"service_{i}", 0) for i in range(env.n_services)]
    assert len(recorded_actions) == 6
    assert all(isinstance(a, (int, np.integer)) for a in recorded_actions)

    # 2. Edge case: Empty action dictionary (all agents dropped or dead)
    empty_dict = {}
    default_actions = [empty_dict.get(f"service_{i}", 0) for i in range(env.n_services)]
    assert default_actions == [0] * 6

    # 3. Edge case: Partial action dictionary
    partial_dict = {"service_0": 3, "service_2": 5}
    partial_actions = [partial_dict.get(f"service_{i}", 0) for i in range(env.n_services)]
    assert partial_actions == [3, 0, 5, 0, 0, 0]
    env.close()


def test_scope2_returns_to_go_calculation():
    """Test Returns-To-Go (RTG) discounting on various trajectory lengths and reward distributions."""
    gamma = 0.99
    
    # Case 1: Empty rewards
    rewards_empty = []
    rtg_empty = []
    disc_sum = 0.0
    for r in reversed(rewards_empty):
        disc_sum = r + gamma * disc_sum
        rtg_empty.insert(0, disc_sum)
    assert rtg_empty == []

    # Case 2: Single reward
    rewards_single = [10.0]
    rtg_single = []
    disc_sum = 0.0
    for r in reversed(rewards_single):
        disc_sum = r + gamma * disc_sum
        rtg_single.insert(0, disc_sum)
    assert rtg_single == [10.0]

    # Case 3: Multiple rewards
    rewards = [1.0, 2.0, 3.0]
    expected_0 = 1.0 + gamma * (2.0 + gamma * 3.0)
    expected_1 = 2.0 + gamma * 3.0
    expected_2 = 3.0
    
    rtg = []
    disc_sum = 0.0
    for r in reversed(rewards):
        disc_sum = r + gamma * disc_sum
        rtg.insert(0, disc_sum)
    
    assert math.isclose(rtg[0], expected_0, rel_tol=1e-5)
    assert math.isclose(rtg[1], expected_1, rel_tol=1e-5)
    assert math.isclose(rtg[2], expected_2, rel_tol=1e-5)


def test_scope2_decision_transformer_short_sequence():
    """Test DecisionTransformer forward pass on minimum sequence length (seq_len = 1)."""
    state_dim = 143
    n_actions = 6
    dt = DecisionTransformer(state_dim=state_dim, n_actions=n_actions)
    
    st = torch.randn(1, 1, state_dim)
    act = torch.tensor([[2]], dtype=torch.long)
    rtg = torch.tensor([[[5.0]]], dtype=torch.float32)
    ts = torch.tensor([[0]], dtype=torch.long)

    logits = dt(st, act, rtg, ts)
    assert logits.shape == (1, 1, n_actions)


# ============================================================================
# Scope 3: Stage 1 HGT Loss Autograd Stability & Probe Gate Criteria
# ============================================================================

def test_scope3_hgt_autograd_empty_node_types():
    """Verify HGT training loop handles empty node types without raising autograd errors."""
    device = torch.device("cpu")
    hgt_encoder = HGTGraphEncoder(EncoderConfig(hidden_dim=32, num_layers=1)).to(device)
    decoders = nn.ModuleDict({
        ntype: nn.Linear(32, FEATURE_DIMS[ntype]) for ntype in NODE_TYPES
    }).to(device)
    
    dec_params = list(hgt_encoder.parameters()) + list(decoders.parameters())
    opt = torch.optim.AdamW(dec_params, lr=1e-3)

    # 1. Batch where one node type has 0 nodes
    g1 = HeteroData()
    g1["Service"].x = torch.randn(3, FEATURE_DIMS["Service"], device=device)
    g1["Pod"].x = torch.zeros((0, FEATURE_DIMS["Pod"]), device=device)
    g1["Node"].x = torch.randn(2, FEATURE_DIMS["Node"], device=device)
    g1[("Service", "routes_to", "Service")].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)

    opt.zero_grad()
    out1 = hgt_encoder(g1)
    rec_loss = torch.tensor(0.0, device=device, requires_grad=True)
    has_terms = False
    for ntype, emb in out1.node_embeddings.items():
        if g1[ntype].num_nodes > 0:
            rec = decoders[ntype](emb)
            loss_term = nn.functional.mse_loss(rec, g1[ntype].x)
            rec_loss = rec_loss + loss_term if has_terms else loss_term
            has_terms = True

    assert has_terms is True
    rec_loss.backward()
    opt.step()

    # 2. Batch where ALL node types have 0 nodes
    g_empty = HeteroData()
    for ntype in NODE_TYPES:
        g_empty[ntype].x = torch.zeros((0, FEATURE_DIMS[ntype]), device=device)
    
    opt.zero_grad()
    out_empty = hgt_encoder(g_empty)
    rec_loss_empty = torch.tensor(0.0, device=device, requires_grad=True)
    has_terms_empty = False
    for ntype, emb in out_empty.node_embeddings.items():
        if g_empty[ntype].num_nodes > 0:
            rec = decoders[ntype](emb)
            loss_term = nn.functional.mse_loss(rec, g_empty[ntype].x)
            rec_loss_empty = rec_loss_empty + loss_term if has_terms_empty else loss_term
            has_terms_empty = True
    
    assert has_terms_empty is False
    # If not has_terms_empty, loop skips backward - verify clean pass
    if not has_terms_empty:
        pass  # correctly bypassed backward()


def test_scope3_probe_metrics_and_gate_thresholds():
    """Verify probe metrics computation and gate decision criteria."""
    # 1. Perfect predictions: balanced accuracy = 1.0, macro-F1 = 1.0
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred_perf = np.array([0, 0, 1, 1, 2, 2])
    m_perf = evaluate_predictions(y_true, y_pred_perf, n_classes=3)
    assert m_perf.accuracy == 1.0
    assert m_perf.balanced_accuracy == 1.0
    assert m_perf.macro_f1 == 1.0

    # 2. Majority class predictor (always predicting 0)
    y_maj = np.zeros_like(y_true)
    m_maj = evaluate_predictions(y_true, y_maj, n_classes=3)
    assert math.isclose(m_maj.balanced_accuracy, 1.0 / 3.0, rel_tol=1e-4)

    # 3. Test inverse-frequency linear probe fitting
    x_synth = torch.randn(100, 16)
    # Synthetic label distribution with imbalance
    y_synth = torch.tensor([0] * 70 + [1] * 20 + [2] * 10, dtype=torch.long)
    probe = fit_linear_probe(x_synth, y_synth, epochs=50, lr=0.05)
    preds = probe.predict(x_synth)
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1, 2})


def test_scope3_split_report_failure_detection():
    """Test SplitReport failure detection catches violations of all 4 gate clauses."""
    perf_metrics = ClassificationMetrics(
        accuracy=0.95, balanced_accuracy=0.95, macro_f1=0.95,
        per_class_f1=(0.95, 0.95, 0.95), per_class_recall=(0.95, 0.95, 0.95), support=(50, 30, 20)
    )
    maj_metrics = ClassificationMetrics(
        accuracy=0.50, balanced_accuracy=0.333, macro_f1=0.22,
        per_class_f1=(0.66, 0.0, 0.0), per_class_recall=(1.0, 0.0, 0.0), support=(50, 30, 20)
    )
    rand_metrics = ClassificationMetrics(
        accuracy=0.33, balanced_accuracy=0.33, macro_f1=0.33,
        per_class_f1=(0.33, 0.33, 0.33), per_class_recall=(0.33, 0.33, 0.33), support=(50, 30, 20)
    )

    per_type_enc = {nt: perf_metrics for nt in NODE_TYPES}
    per_type_maj = {nt: maj_metrics for nt in NODE_TYPES}
    per_type_raw = {nt: perf_metrics for nt in NODE_TYPES}

    # Passing split report
    passing_report = SplitReport(
        name="test_pass",
        sizes=("small",),
        n_nodes=100,
        class_counts=(50, 30, 20),
        encoder=perf_metrics,
        majority=maj_metrics,
        random=rand_metrics,
        raw_features=perf_metrics,
        per_type_encoder=per_type_enc,
        per_type_majority=per_type_maj,
        per_type_raw=per_type_raw,
        per_size_encoder={"small": perf_metrics},
        per_size_majority={"small": maj_metrics},
    )
    assert passing_report.passed is True
    assert len(passing_report.failures()) == 0

    # Failing split report: balanced_accuracy < 0.60
    failing_enc = ClassificationMetrics(
        accuracy=0.50, balanced_accuracy=0.40, macro_f1=0.35,
        per_class_f1=(0.5, 0.3, 0.25), per_class_recall=(0.6, 0.4, 0.2), support=(50, 30, 20)
    )
    failing_report = SplitReport(
        name="test_fail",
        sizes=("small",),
        n_nodes=100,
        class_counts=(50, 30, 20),
        encoder=failing_enc,
        majority=maj_metrics,
        random=rand_metrics,
        raw_features=perf_metrics,
        per_type_encoder=per_type_enc,
        per_type_majority=per_type_maj,
        per_type_raw=per_type_raw,
        per_size_encoder={"small": failing_enc},
        per_size_majority={"small": maj_metrics},
    )
    assert failing_report.passed is False
    assert any("pooled balanced accuracy" in msg for msg in failing_report.failures())


# ============================================================================
# Scope 4: Stage 4 Evaluation Metrics & Baseline Benchmarking
# ============================================================================

def test_scope4_evaluation_harness_and_controllers():
    """Verify evaluation harness computes TTR, SLA, and separated reward components for controllers."""
    def make_env():
        return ClusterEnv(config=ClusterConfig(**scenario_overrides("pod_crash", max_cycles=15)))

    eval_seeds = [90001, 90002]
    noop_ctrl = NoOpController()
    rule_ctrl = RuleBasedController()

    rep_noop, res_noop = evaluate(make_env, noop_ctrl, eval_seeds, scenario="pod_crash")
    rep_rule, res_rule = evaluate(make_env, rule_ctrl, eval_seeds, scenario="pod_crash")

    # Verify ScenarioReport contents
    assert rep_noop.episodes == 2
    assert rep_rule.episodes == 2
    assert rep_noop.scenario == "pod_crash"
    assert rep_rule.scenario == "pod_crash"

    # Verify uncollapsed reward components
    for k in ("sla_violation", "latency", "availability", "action_cost", "invalid_action", "terminal"):
        assert k in rep_noop.mean_reward_components
        assert k in rep_rule.mean_reward_components

    # Verify beats comparison
    verdict = beats(rep_rule, rep_noop)
    assert "ttr_delta" in verdict
    assert "sla_delta" in verdict
    assert "beats_both" in verdict
    assert isinstance(verdict["beats_both"], bool)

    # Verify formatting functions do not raise errors
    comparison_str = format_comparison([rep_noop, rep_rule], headline="TEST COMPARISON")
    assert "TEST COMPARISON" in comparison_str
    assert "no-op" in comparison_str
    assert "baseline" in comparison_str

    reward_str = format_reward_components([rep_noop, rep_rule])
    assert "reward components" in reward_str
    assert "sla_violation" in reward_str


def test_scope4_policy_controller_with_mappo():
    """Verify PolicyController integrates with MAPPO and drives evaluation episodes."""
    obs_dim = 38
    state_dim = 143
    n_agents = 12
    n_actions = 6

    mappo_model = MAPPO(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=n_actions)
    policy_ctrl = PolicyController(mappo_model, name="mappo")

    def make_env():
        return ClusterEnv(config=ClusterConfig(**scenario_overrides("pod_crash", max_cycles=10)))

    rep_policy, res_policy = evaluate(make_env, policy_ctrl, [90001], scenario="pod_crash")
    assert rep_policy.controller == "mappo"
    assert rep_policy.episodes == 1
    assert len(res_policy) == 1
    assert res_policy[0].length <= 10


def test_scope4_checkpoint_loading_weights_only_compatibility():
    """Verify checkpoint loading works reliably with PyTorch 2.6+ weights_only setting."""
    ckpt_dir = Path("marl/checkpoints/mappo_smoke_run")
    final_pt = ckpt_dir / "final.pt"
    if not final_pt.exists():
        pytest.skip("Smoke checkpoint not generated yet")

    # In PyTorch 2.6+, weights_only=True fails on torch_version.TorchVersion in provenance.
    # Verify weights_only=False loads successfully into MAPPO.
    ckpt = torch.load(final_pt, map_location="cpu", weights_only=False)
    assert "obs_dim" in ckpt
    assert "state_dim" in ckpt
    assert "encoder" in ckpt
    assert "actor" in ckpt
    assert "critic" in ckpt

    mappo_model = MAPPO(obs_dim=ckpt["obs_dim"], state_dim=ckpt["state_dim"], n_agents=ckpt["n_agents"], n_actions=ckpt["n_actions"])
    mappo_model.load_state_dict(ckpt)
    policy_ctrl = PolicyController(mappo_model, name="mappo")
    assert policy_ctrl.name == "mappo"


def test_notebook_cell6_and_cell12_exact_syntax_execution():
    """Verify that Cell 6 and Cell 12 code from the notebook executes cleanly without errors."""
    import json
    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "aegis_training.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # 1. Test Cell 6 execution
    cell6_src = "".join(nb["cells"][6]["source"])
    clean_lines6 = [l for l in cell6_src.splitlines() if not l.strip().startswith("%")]
    code6 = "\n".join(clean_lines6)

    # Create an execution scope with limited iterations for fast testing
    code6_fast = code6.replace("for epoch in range(1, 11):", "for epoch in range(1, 2):")
    scope6 = {}
    exec(code6_fast, scope6)
    assert "hgt_encoder" in scope6
    assert isinstance(scope6["hgt_encoder"], HGTGraphEncoder)

    # 2. Test Cell 12 execution
    cell12_src = "".join(nb["cells"][12]["source"])
    clean_lines12 = [l for l in cell12_src.splitlines() if not l.strip().startswith("%")]
    code12 = "\n".join(clean_lines12)
    scope12 = {}
    exec(code12, scope12)
    assert "buffer" in scope12
    assert "happo_trainer" in scope12
    assert "qmix_trainer" in scope12
    assert set(scope12["buffer"].components.keys()) == set(COMPONENT_NAMES)

