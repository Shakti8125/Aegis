"""Empirical Adversarial Stress Verification Suite for Milestone 1.

Targeting:
1. simulator/cluster_env.py: Seeding determinism, reset/step cycles, 6 actions x 5 fault types, reward component breakdown.
2. encoder/gnn_model.py & encoder/features.py: PyG HeteroData forward passes, dynamic cluster sizes, batching, edge attributes.
3. ops_layer/: LLM fallback resilience across narrator, safety supervisor, log parser, ask aegis, and post mortem.
"""

from __future__ import annotations

import copy
import json
import math
import urllib.error
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from encoder.features import (
    EDGE_DIMS,
    FEATURE_DIMS,
    NODE_TYPES,
    RELATIONS,
    snapshot_to_hetero_data,
)
from encoder.gnn_model import AegisGraphEncoder, EncoderConfig
from ops_layer.ask_aegis import AskAegisAssistant, validate_cypher_security
from ops_layer.autonomy_engine import (
    ActionProposal,
    AutonomyDecision,
    AutonomyLevel,
    DecisionOutcome,
    GraduatedAutonomyEngine,
)
from ops_layer.llm_client import GeminiClient, LLMClient, LLMError, OllamaClient, StubClient
from ops_layer.log_parser import EventType, GraphEvent, LogParser
from ops_layer.narrator import (
    ActionContext,
    DependencyEdge,
    Narration,
    Narrator,
    ServiceSnapshot,
)
from ops_layer.post_mortem import FactGroundedPostMortemGenerator, IncidentPostMortem
from ops_layer.react_agent import ReActDiagnosticAgent
from ops_layer.safety_supervisor import (
    Policy,
    SafetySupervisor,
    VetoDecision,
    VetoResult,
)
from simulator.cluster_env import (
    ACTION_COST,
    ACTION_ISOLATE,
    ACTION_NAMES,
    ACTION_NOOP,
    ACTION_REROUTE,
    ACTION_RESTART,
    ACTION_SCALE_DOWN,
    ACTION_SCALE_UP,
    N_ACTIONS,
    ClusterConfig,
    ClusterEnv,
    make_env,
)
from simulator.fault_injection import FaultEvent, FaultType, build_fault_schedule


# ==============================================================================
# 1. SIMULATOR EMPIRICAL STRESS & DETERMINISM TESTS
# ==============================================================================
class TestSimulatorEmpiricalStress:
    """Stress tests on simulator reset/step cycles, seeding, actions, and rewards."""

    @pytest.mark.parametrize("seed", [42, 100, 777])
    def test_seeding_determinism_multi_step_trajectory(self, seed: int):
        """Identical seeds must yield bitwise/float identical states, obs, rewards, and infos."""
        cfg = ClusterConfig(
            n_services=12,
            n_nodes=6,
            enabled_faults=(
                FaultType.POD_CRASH,
                FaultType.NODE_CPU_SPIKE,
                FaultType.NODE_MEM_SPIKE,
                FaultType.NETWORK_PARTITION,
                FaultType.CASCADING_LATENCY,
            ),
            n_faults_range=(2, 4),
            fault_start_tick=2,
        )

        env1 = ClusterEnv(cfg)
        env2 = ClusterEnv(cfg)

        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)

        assert env1.agents == env2.agents
        for agent in env1.agents:
            np.testing.assert_array_equal(
                obs1[agent], obs2[agent], err_msg=f"Initial obs mismatch for {agent}"
            )

        rng = np.random.default_rng(seed + 999)
        n_steps = 40

        for step in range(n_steps):
            # Same random actions applied to both environments
            actions = {agent: rng.integers(0, N_ACTIONS) for agent in env1.agents}

            step_obs1, r1, term1, trunc1, inf1 = env1.step(actions)
            step_obs2, r2, term2, trunc2, inf2 = env2.step(actions)

            assert term1 == term2, f"Termination mismatch at step {step}"
            assert trunc1 == trunc2, f"Truncation mismatch at step {step}"

            for agent in env1.agents:
                np.testing.assert_array_equal(
                    step_obs1[agent],
                    step_obs2[agent],
                    err_msg=f"Obs mismatch at step {step} for agent {agent}",
                )
                assert (
                    r1[agent] == r2[agent]
                ), f"Reward mismatch at step {step} for agent {agent}: {r1[agent]} != {r2[agent]}"

                rc1 = inf1[agent]["reward_components"]
                rc2 = inf2[agent]["reward_components"]
                for k in rc1:
                    assert math.isclose(
                        rc1[k], rc2[k], rel_tol=1e-6, abs_tol=1e-6
                    ), f"Reward component {k} mismatch at step {step} for {agent}"

            if not env1.agents:
                break

    def test_env_multiple_consecutive_reset_step_cycles(self):
        """Repeated reset-step cycles should not leak state, crash, or violate spaces."""
        env = make_env(n_services=8, n_nodes=4)
        for cycle in range(5):
            obs, infos = env.reset(seed=1000 + cycle)
            assert len(env.agents) == 8
            assert len(obs) == 8

            for _ in range(15):
                actions = {agent: ACTION_NOOP for agent in env.agents}
                obs, rewards, term, trunc, infos = env.step(actions)
                for agent in env.agents:
                    space = env.observation_space(agent)
                    assert space.contains(obs[agent]), f"Obs out of bounds on cycle {cycle}"
                    assert not np.isnan(obs[agent]).any(), "NaN found in observation vector"
                    assert not np.isinf(obs[agent]).any(), "Inf found in observation vector"

    @pytest.mark.parametrize(
        "fault_type",
        [
            FaultType.POD_CRASH,
            FaultType.NODE_CPU_SPIKE,
            FaultType.NODE_MEM_SPIKE,
            FaultType.NETWORK_PARTITION,
            FaultType.CASCADING_LATENCY,
        ],
    )
    def test_all_six_actions_under_each_fault_type(self, fault_type: FaultType):
        """All 6 actions must execute cleanly without exceptions under each specific fault type."""
        cfg = ClusterConfig(
            n_services=6,
            n_nodes=3,
            enabled_faults=(fault_type,),
            n_faults_range=(2, 3),
            fault_start_tick=1,
            max_cycles=30,
        )
        env = ClusterEnv(cfg)
        obs, infos = env.reset(seed=42)

        # Test each action explicitly across 6 steps
        actions_sequence = [
            ACTION_NOOP,
            ACTION_RESTART,
            ACTION_SCALE_UP,
            ACTION_SCALE_DOWN,
            ACTION_ISOLATE,
            ACTION_REROUTE,
        ]

        for act in actions_sequence:
            action_dict = {agent: act for agent in env.agents}
            obs, rewards, term, trunc, infos = env.step(action_dict)
            assert len(obs) == len(env.agents)
            for agent in env.agents:
                assert np.isfinite(obs[agent]).all()
                assert np.isfinite(rewards[agent])
                # Check reward components
                rc = infos[agent]["reward_components"]
                assert isinstance(rc, dict)
                for k, v in rc.items():
                    assert np.isfinite(v), f"Non-finite reward component {k}={v}"

    def test_reward_dictionary_structure_and_component_separation(self):
        """Verify reward dictionary structure and confirm that separate components are strictly preserved."""
        env = make_env(n_services=6, n_nodes=3)
        obs, infos = env.reset(seed=123)

        # Apply aggressive scaling and restarts to trigger operational costs and SLA impacts
        actions = {
            env.agents[0]: ACTION_SCALE_UP,
            env.agents[1]: ACTION_RESTART,
            env.agents[2]: ACTION_ISOLATE,
            env.agents[3]: ACTION_REROUTE,
            env.agents[4]: ACTION_SCALE_DOWN,
            env.agents[5]: ACTION_NOOP,
        }

        obs, rewards, term, trunc, infos = env.step(actions)

        for agent in env.agents:
            info = infos[agent]
            assert "reward_components" in info, f"Missing reward_components in info for {agent}"
            rc = info["reward_components"]
            assert isinstance(rc, dict), "reward_components must be a dictionary"

            # Check presence of distinct component keys from simulator/reward.py
            expected_keys = {
                "sla_violation",
                "latency",
                "availability",
                "action_cost",
                "invalid_action",
                "terminal",
            }
            # All expected components must be present
            for k in expected_keys:
                assert k in rc, f"Missing key {k} in reward_components: {rc.keys()}"
                assert isinstance(rc[k], (float, int, np.floating)), f"Component {k} is not float"

            # Ensure action cost is logged separately and matches action if valid, or zeroes if rejected
            agent_idx = env.agent_name_to_index[agent]
            action_taken = actions[agent]
            expected_cost = float(ACTION_COST[action_taken])
            if rc["invalid_action"] == 0.0:
                assert math.isclose(
                    abs(rc["action_cost"]), expected_cost, rel_tol=1e-5
                ), f"Action cost magnitude mismatch for valid action: got {rc['action_cost']}, expected {expected_cost}"
            else:
                assert math.isclose(
                    rc["action_cost"], 0.0, abs_tol=1e-5
                ), f"Action cost must be 0 for invalid/rejected action, got {rc['action_cost']}"


# ==============================================================================
# 2. ENCODER GNN & FEATURES EMPIRICAL STRESS TESTS
# ==============================================================================
class TestGNNEncoderEmpiricalStress:
    """Stress tests on PyG HeteroData encoding across cluster sizes, batching, and edge attributes."""

    @pytest.mark.parametrize(
        "n_services, n_nodes",
        [
            (3, 2),    # Tiny cluster
            (6, 3),    # Small cluster
            (12, 6),   # Default cluster
            (24, 12),  # Medium cluster
            (40, 15),  # Large cluster
        ],
    )
    def test_gnn_forward_pass_dynamic_cluster_sizes(self, n_services: int, n_nodes: int):
        """Graph encoder must produce valid embeddings for any cluster size without dimension mismatches."""
        cfg = ClusterConfig(n_services=n_services, n_nodes=n_nodes, max_replicas=4)
        env = ClusterEnv(cfg)
        env.reset(seed=42)

        snapshot = env.graph_snapshot()
        hetero_data = snapshot_to_hetero_data(snapshot, with_labels=True)

        encoder = AegisGraphEncoder(
            EncoderConfig(hidden_dim=32, embed_dim=32, global_dim=64, num_layers=2)
        )
        encoder.eval()

        with torch.no_grad():
            output = encoder(hetero_data)

        # Verify Service embeddings shape = (n_services, embed_dim)
        assert output.node_embeddings["Service"].shape == (n_services, 32)
        # Verify Node embeddings shape = (n_nodes, embed_dim)
        assert output.node_embeddings["Node"].shape == (n_nodes, 32)
        # Verify Global embedding shape = (1, global_dim)
        assert output.global_embedding.shape == (1, 64)

        # Numerical sanity: no NaNs, no Infs
        for ntype, emb in output.node_embeddings.items():
            assert torch.isfinite(emb).all(), f"NaN or Inf in {ntype} embeddings"
        assert torch.isfinite(output.global_embedding).all(), "NaN or Inf in global_embedding"

    def test_gnn_batching_multiple_heterogeneous_graphs(self):
        """Batching heterogeneous graphs of varying sizes with PyG Batch must correctly preserve batch dims."""
        graphs = []
        cluster_sizes = [(4, 2), (8, 4), (12, 6)]
        total_services = sum(s for s, _ in cluster_sizes)

        for s, n in cluster_sizes:
            cfg = ClusterConfig(n_services=s, n_nodes=n)
            env = ClusterEnv(cfg)
            env.reset(seed=s * 10)
            snapshot = env.graph_snapshot()
            graphs.append(snapshot_to_hetero_data(snapshot, with_labels=False))

        batch = Batch.from_data_list(graphs)

        encoder = AegisGraphEncoder(
            EncoderConfig(hidden_dim=32, embed_dim=32, global_dim=64, num_layers=2)
        )
        encoder.train()

        output = encoder(batch)

        # Global embedding should have batch dimension = 3
        assert output.global_embedding.shape == (3, 64)
        # Service embeddings should concatenate across graphs
        assert output.node_embeddings["Service"].shape == (total_services, 32)

        # Test backward pass through both heads
        loss = output.global_embedding.sum() + output.node_embeddings["Service"].sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    def test_gnn_edge_attributes_and_zero_pod_resilience(self):
        """GNN encoder must handle edge attributes on CALLS, empty edge attributes, and empty pod sets."""
        cfg = ClusterConfig(n_services=4, n_nodes=2)
        env = ClusterEnv(cfg)
        env.reset(seed=42)
        snapshot = env.graph_snapshot()

        # Artificially remove all pods from snapshot
        snapshot_no_pods = copy.deepcopy(snapshot)
        snapshot_no_pods["nodes"]["Pod"] = []
        snapshot_no_pods["relationships"]["INSTANCE_OF"] = []
        snapshot_no_pods["relationships"]["RUNS_ON"] = []

        data_no_pods = snapshot_to_hetero_data(snapshot_no_pods, with_labels=False)
        assert data_no_pods["Pod"].num_nodes == 0

        encoder = AegisGraphEncoder(
            EncoderConfig(hidden_dim=32, embed_dim=32, global_dim=64, num_layers=2)
        )
        encoder.eval()

        with torch.no_grad():
            out = encoder(data_no_pods)

        assert out.node_embeddings["Service"].shape == (4, 32)
        assert out.global_embedding.shape == (1, 64)
        assert torch.isfinite(out.global_embedding).all()


# ==============================================================================
# 3. OPS LAYER LLM FALLBACK RESILIENCE TESTS
# ==============================================================================
class TestOpsLayerLLMFallbackResilience:
    """Stress tests on ops_layer fallback mechanisms when external LLMs fail."""

    def _sample_action_context(self, action: int = ACTION_RESTART) -> ActionContext:
        svc = ServiceSnapshot(
            service_id="svc-02",
            health=0.42,
            cpu_pct=0.88,
            mem_pct=0.65,
            p99_latency_ms=350.0,
            error_rate=0.08,
            replicas=3,
            ready_replicas=2,
            tier="mid",
        )
        dep = DependencyEdge(
            source_id="svc-02",
            target_id="svc-05",
            relation="CALLS",
            p99_latency_ms=420.0,
            error_rate=0.15,
        )
        return ActionContext(
            tick=14,
            agent_id="svc-02",
            action=action,
            target_service=svc,
            dependencies=[dep],
            dependents=[],
            active_faults=[{"type": "pod_crash", "target": "svc-02"}],
        )

    def test_narrator_fallback_when_llm_is_none(self):
        """Narrator must cleanly generate template narration when llm_client=None."""
        narrator = Narrator(llm_client=None)
        ctx = self._sample_action_context(ACTION_RESTART)
        narration = narrator.narrate(ctx)

        assert isinstance(narration, Narration)
        assert narration.model == "fallback/template"
        assert narration.grounded is True
        assert "Restarting unhealthiest pod on svc-02" in narration.text
        assert narration.cited_edge_source == "svc-02"
        assert narration.cited_edge_target == "svc-05"

    def test_narrator_fallback_on_llm_network_or_api_error(self):
        """Narrator must gracefully fall back to templates when LLM throws LLMError or network errors."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model_name = "mock/error_model"
        mock_llm.complete.side_effect = LLMError("Connection timeout to LLM endpoint")

        narrator = Narrator(llm_client=mock_llm)
        ctx = self._sample_action_context(ACTION_SCALE_UP)
        narration = narrator.narrate(ctx)

        assert narration.model == "fallback/template"
        assert "Scaling up svc-02" in narration.text
        assert narrator.stats["fallback_narrations"] == 1

    def test_narrator_grounding_verification_rejects_hallucinations(self):
        """Narrator must reject LLM responses that cite hallucinated service IDs not in context."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.model_name = "mock/hallucinating_model"
        # LLM invents svc-99 which is not in context
        hallucinated_json = json.dumps(
            {
                "text": "Restarting svc-02 because svc-99 failed unexpectedly.",
                "cited_facts": ["svc-02", "svc-99"],
                "cited_edge_source": "svc-02",
                "cited_edge_target": "svc-99",
            }
        )
        mock_llm.complete.return_value = hallucinated_json

        narrator = Narrator(llm_client=mock_llm)
        ctx = self._sample_action_context(ACTION_RESTART)
        narration = narrator.narrate(ctx)

        # Grounding check should fail and force template fallback
        assert narration.model == "fallback/template"
        assert "svc-99" not in narration.text

    def test_safety_supervisor_rule_and_llm_failure_modes(self):
        """SafetySupervisor must handle LLM errors according to on_llm_failure configuration."""
        ctx = self._sample_action_context(ACTION_RESTART)

        # 1. on_llm_failure="no_op" (default: safety veto on LLM error)
        failing_llm = MagicMock(spec=LLMClient)
        failing_llm.complete.side_effect = LLMError("API rate limit exceeded")

        supervisor_safe = SafetySupervisor(
            llm_client=failing_llm,
            llm_policies=["Do not restart during high memory pressure"],
            on_llm_failure="no_op",
        )
        res_safe = supervisor_safe.check(ctx)
        assert res_safe.vetoed is True
        assert "LLM error" in res_safe.reason

        # 2. on_llm_failure="allow" (permissive fallback)
        supervisor_allow = SafetySupervisor(
            llm_client=failing_llm,
            llm_policies=["Do not restart during high memory pressure"],
            on_llm_failure="allow",
        )
        res_allow = supervisor_allow.check(ctx)
        assert res_allow.vetoed is False

    def test_log_parser_fallback_on_unstructured_and_failed_llm(self):
        """LogParser must parse structured lines with regex, and handle LLM failures on unknown lines without crash."""
        failing_llm = MagicMock(spec=LLMClient)
        failing_llm.complete.side_effect = LLMError("Local Ollama down")
        parser = LogParser(llm_client=failing_llm)

        # Structured line -> regex succeeds without LLM
        line1 = "[TICK 10] SERVICE svc-01-mid: health=0.90 cpu=0.35 latency=25.0"
        events1 = parser.parse(line1)
        assert len(events1) == 1
        assert events1[0].event_type == EventType.METRIC_UPDATE
        assert events1[0].entity_id == "svc-01-mid"

        # Completely unstructured noisy line -> LLM fails -> returns empty list gracefully
        line2 = "Unexpected kernel panic: Out of memory killer invoked on cgroup"
        events2 = parser.parse(line2)
        assert isinstance(events2, list)
        assert len(events2) == 0

    def test_ask_aegis_cypher_security_and_fallback_synthesis(self):
        """Ask Aegis must block mutating Cypher, and use rule fallback when LLM fails."""
        # 1. Security check
        assert validate_cypher_security("MATCH (s:Service) RETURN s")[0] is True
        assert validate_cypher_security("MATCH (s:Service) DELETE s")[0] is False
        assert validate_cypher_security("CREATE (n:Pod {id: 'hack'})")[0] is False
        assert validate_cypher_security("MATCH (s:Service) SET s.health = 1.0")[0] is False

        # 2. Fallback text-to-cypher & answer synthesis
        failing_llm = MagicMock(spec=LLMClient)
        failing_llm.complete.side_effect = LLMError("Gemini quota exceeded")
        assistant = AskAegisAssistant(llm_client=failing_llm, neo4j_driver=None)

        resp = assistant.query("Which services are currently unhealthy?")
        assert resp.is_safe is True
        assert "MATCH (s:Service) WHERE s.health < 0.8" in resp.cypher_query
        assert "is currently degraded" in resp.answer
        assert len(resp.raw_results) > 0

    def test_post_mortem_and_autonomy_engine_llm_failure_resilience(self):
        """FactGroundedPostMortemGenerator and GraduatedAutonomyEngine must degrade gracefully without crashing when LLM is unavailable."""
        failing_llm = MagicMock(spec=LLMClient)
        failing_llm.complete.side_effect = LLMError("LLM unavailable")

        # PostMortemGenerator fallback
        pmg = FactGroundedPostMortemGenerator(llm_client=failing_llm)
        report = pmg.generate(
            incident_data={"incident_id": "INC-101", "severity": "SEV-1", "target_service": "svc-03"},
            graph_facts={"target_service": "svc-03", "services": [{"id": "svc-03", "health": 0.4}]},
        )
        assert isinstance(report, IncidentPostMortem)
        assert report.incident_id == "INC-101"
        assert "fallback" in report.model_used
        assert report.verification_status is True

        # AutonomyEngine fallback
        autonomy = GraduatedAutonomyEngine(level=AutonomyLevel.LEVEL_2_HITL)
        proposal = ActionProposal(agent_id="svc-02", action_name="scale_up", target_service="svc-02", probabilities=[0.7, 0.2, 0.1])
        autonomy_ctx = self._sample_action_context(ACTION_SCALE_UP)
        decision = autonomy.evaluate_action(proposal, autonomy_ctx)
        assert isinstance(decision, AutonomyDecision)
        assert decision.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.REQUIRES_APPROVAL, DecisionOutcome.VETOED_BY_SAFETY)

    def test_react_diagnostic_agent_llm_fallback(self):
        """ReActDiagnosticAgent must fall back to deterministic diagnostics when LLM fails or is None."""
        failing_llm = MagicMock(spec=LLMClient)
        failing_llm.model_name = "failing-model"
        failing_llm.complete.side_effect = LLMError("LLM offline")

        agent = ReActDiagnosticAgent(llm_client=failing_llm)
        result = agent.diagnose("svc-03", "High p99 latency observed")

        assert result.target_service == "svc-03"
        assert result.root_cause != ""
        assert len(result.steps) > 0
        assert result.model_used == "rule-fallback"

    def test_gnn_fit_normalization_and_disconnected_graphs(self):
        """Test GNN encoder fit_normalization on sample graphs and forward pass on disconnected graph."""
        cfg = ClusterConfig(n_services=6, n_nodes=3)
        env = ClusterEnv(cfg)
        env.reset(seed=42)
        snapshot = env.graph_snapshot()
        g1 = snapshot_to_hetero_data(snapshot, with_labels=False)

        encoder = AegisGraphEncoder(
            EncoderConfig(hidden_dim=32, embed_dim=32, global_dim=64, num_layers=2)
        )
        assert not bool(encoder.get_buffer("normalization_fitted").item())

        encoder.fit_normalization([g1])
        assert bool(encoder.get_buffer("normalization_fitted").item())

        # Forward pass on empty edge relations
        g_disconnected = copy.deepcopy(g1)
        for rel in RELATIONS:
            if rel in g_disconnected.edge_types:
                g_disconnected[rel].edge_index = torch.zeros((2, 0), dtype=torch.long)
                if rel in EDGE_DIMS and EDGE_DIMS[rel] > 0:
                    g_disconnected[rel].edge_attr = torch.zeros((0, EDGE_DIMS[rel]), dtype=torch.float32)

        encoder.eval()
        with torch.no_grad():
            out = encoder(g_disconnected)
        assert out.node_embeddings["Service"].shape == (6, 32)
        assert out.global_embedding.shape == (1, 64)
        assert torch.isfinite(out.global_embedding).all()

