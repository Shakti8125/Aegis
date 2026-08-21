# Milestone 1 Review Report: Bug Resolution & Core Library Verification

**Reviewer**: Reviewer 2 (`reviewer_m1_2`)  
**Roles**: Reviewer & Adversarial Critic  
**Milestone**: Milestone 1 (Aegis Bug Resolution & Core Library Verification)  
**Target Handoff**: `c:\Users\Shakti\Documents\Aegis\.agents\worker_m1\handoff.md`  
**Verdict**: **PASS / APPROVE**  
**Integrity Audit**: **PASS (Clean)** — Zero hardcoded cheats, zero facade implementations, zero bypass shortcuts.

---

## 1. Observation

Direct inspection of the Aegis repository, core library APIs, and independent test execution confirmed the following facts:

### 1.1 `marl/action_mask.py`
- **Lines 54–59**: In `MaskedCategorical.entropy()`, the calculation computes:
  ```python
  def entropy(self) -> torch.Tensor:
      """Compute entropy ignoring zero-probability (masked) actions safely."""
      log_p = torch.log(self.probs.clamp_min(1e-12))
      p_log_p = torch.where(self.probs > 0, self.probs * log_p, torch.zeros_like(self.probs))
      return -p_log_p.sum(dim=-1)
  ```
  This replaces the prior faulty `self.probs * self.logits` calculation (which multiplied probabilities by unnormalized logits shifted by $-10^9$) with exact Shannon entropy $H(P) = -\sum_{i=1}^K p_i \ln(p_i)$ where $\lim_{p \to 0^+} p \ln(p) = 0$.
- **Lines 80–106**: In `compute_action_mask_from_obs()`:
  - Vector observation indices directly correspond to `simulator/cluster_env.py`:
    - Index 6: `replicas / max_replicas` (with 1 min replica yielding 0.1)
    - Index 8: `isolate_timer / isolate_duration`
  - Action 3 (`SCALE_DOWN`) is masked when `replica_frac <= 0.1`.
  - Action 4 (`ISOLATE`) is masked when `isolate_timer > 0.0`.
  - Action 5 is correctly preserved as `REROUTE` (unconstrained by service isolation status), fixing the prior erroneous `RECONNECT` mapping.
  - An explicit guard `if obs.shape[-1] >= 9:` prevents index out-of-bounds errors on truncated observation vectors.

### 1.2 `marl/ppo_lagrangian.py`
- **Lines 184–190 & 214–216**: `PPOLagrangian.update()` explicitly queries `device = next(self.actor.parameters()).device` and applies `device=device` to all tensor instantiations (`obs_t`, `state_t`, `actions_t`, `old_logprobs_t`, `tot_adv_t`) and MSE loss targets (`reward_returns`, `sla_returns`, `action_cost_returns`), eliminating device mismatch exceptions across CUDA and CPU environments.

### 1.3 `marl/qmix.py`
- **Lines 163–165**: `QMIX.act()` extracts `device = next(self.agent_net.parameters()).device` and sets `device=device` during tensor conversion.
- **Lines 189–195**: `QMIX.compute_loss()` handles both 1D joint rewards `(B,)` and 2D per-agent rewards `(B, N)`:
  ```python
  if rewards.dim() > 1:
      rewards = rewards.sum(dim=-1, keepdim=True)
  else:
      rewards = rewards.view(b_size, 1)
  ```
- **Lines 91–103**: `QMixer` enforces non-negative weights $W_1 = |W_1(S)|$ and $W_2 = |W_2(S)|$, satisfying the strict monotonicity condition $\frac{\partial Q_{tot}}{\partial q_i} \ge 0$.

### 1.4 `demo/kubectl_adapter.py` & `tests/demo/test_kubectl_adapter.py`
- Added dry-run command translation for all 6 discrete actions (`NOOP`, `RESTART`, `SCALE_UP`, `SCALE_DOWN`, `ISOLATE`, `REROUTE`).
- Strict regex validation (`K8S_NAME_REGEX = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")`) ensures compliance with RFC 1123 DNS subdomain standards and prevents command injection.

### 1.5 Independent Test Execution
- Executed `.venv\Scripts\pytest tests/simulator tests/encoder tests/marl tests/ops_layer tests/graph tests/backend tests/demo -v` independently:
  ```text
  ================ 335 passed, 47 skipped, 831 warnings in 39.11s ================
  ```
  (The 47 skipped tests are live Neo4j database integration tests designed to skip cleanly when Neo4j is offline; all static Cypher schema, migration, and query syntax tests passed).

---

## 2. Logic Chain

1. **Entropy Correction**:
   - For a discrete distribution with unmasked probabilities $p_i$ and masked entries ($p_j = 0$, $\text{logit}_j = \text{raw}_j - 10^9$), computing $p \cdot \text{logits}$ yielded invalid unnormalized quantities.
   - The clamped formulation `log_p = torch.log(self.probs.clamp_min(1e-12))` combined with `torch.where(self.probs > 0, ...)` guarantees that $0 \cdot \ln(0) \to 0$ without NaN or infinite values, while active categories evaluate to true Shannon entropy $H(P)$.
2. **Action Space & Feature Index Alignment**:
   - `simulator/cluster_env.py` (lines 943–950) defines observation slots: slot 6 is `svc_replicas / max_replicas`, and slot 8 is `svc_isolate_timer / isolate_duration`. Action 5 corresponds to `ACTION_REROUTE`.
   - Aligning `marl/action_mask.py` with these indices ensures that the policy gradient and action selection receive valid preconditions rather than corrupting action 5 or referencing non-existent features.
3. **Device Placement Consistency**:
   - Querying the parameter device of `self.actor` in `PPOLagrangian` and `self.agent_net` in `QMIX` guarantees that all tensors created from numpy buffers or python lists are placed on the active computation device.
4. **QMIX Multi-Dimensional Loss Support**:
   - MARL environments commonly emit per-agent reward arrays `(B, N)`. Reducing across the agent dimension via `rewards.sum(dim=-1, keepdim=True)` correctly produces `(B, 1)` team returns compatible with `target_q_tot` and TD loss computation.
5. **Architectural & Non-Negotiable Convention Compliance**:
   - **Separate Reward Logging**: Verified in `marl/reward.py` and `simulator/cluster_env.py` (`infos[agent]["reward_components"]`). Unit signals (`sla_violation`, `latency`, `availability`, `action_cost`, `invalid_action`, `terminal`) are preserved and never collapsed.
   - **Baseline Benchmark**: Verified in `marl/baseline.py`. A threshold-triggered rule-based controller evaluates identical observations and seeds to benchmark MAPPO recovery time and SLA violations.
   - **LLM Grounding**: Verified in `ops_layer/narrator.py` (`ActionContext`, `ServiceSnapshot`, `DependencyEdge`). Prompts strictly instruct models to cite only real graph facts.
   - **Numbered Cypher Migrations**: Verified in `graph/migrations/001_initial_schema.cypher` and executed by `graph/migrate.py`.

---

## 3. Core Library Verification Audit

| Core Library | Component / Module | Compliance Status | Evidence / Verification Method |
|---|---|---|---|
| **PettingZoo** | `simulator/cluster_env.py` (`ParallelEnv`) | **COMPLIANT** | Passes `test_parallel_api_conformance`, `test_parallel_seed_conformance`, and step/reset contracts |
| **PyTorch Geometric** | `encoder/gnn_model.py`, `encoder/hgt_encoder.py` (`HeteroData`, `MessagePassing`, `HeteroConv`) | **COMPLIANT** | Passes `test_conv_reduces_to_pyg_sageconv`, `test_conv_handles_bipartite_relations`, `test_hgt_encoder_forward_and_memory` |
| **PyTorch (MAPPO / CTDE / GAE)** | `marl/mappo.py`, `marl/actor_critic.py`, `marl/replay_buffer.py`, `marl/action_mask.py`, `marl/ppo_lagrangian.py`, `marl/qmix.py` | **COMPLIANT** | Passes `test_action_masking_logits_and_distribution`, `test_qmix_monotonicity_and_loss`, `test_ppo_lagrangian_primal_dual`, `test_mappo.py` |
| **Neo4j Cypher** | `graph/schema.cypher`, `graph/migrations/`, `graph/migrate.py` | **COMPLIANT** | Passes `test_migrations_are_numbered_uniquely_and_in_order`, `test_every_migration_statement_is_idempotent`, `test_schema_cypher_matches_the_migrations` |
| **FastAPI / WebSockets** | `backend/main.py`, `backend/ws.py`, `backend/models.py` | **COMPLIANT** | Passes `test_backend.py` REST endpoint validations and WebSocket simulation runner frames |

---

## 4. Adversarial Challenge & Stress-Testing

### 4.1 Masked Categorical Entropy Invariants
- **Challenge**: Evaluated entropy behavior across extreme logit ranges ($-1000$ to $+1000$) and highly sparse binary masks.
- **Result**: Entropy remains strictly finite and non-negative ($H \ge 0.0$). Single-valid action distributions evaluate to $H = 0.0 \pm 10^{-6}$, and $M$ equal-logit valid actions evaluate to $H = \ln(M) \pm 10^{-5}$.

### 4.2 QMIX Monotonicity Gradient Verification
- **Challenge**: Stress-tested the monotonicity condition $\frac{\partial Q_{tot}}{\partial q_i} \ge 0$ under adversarial state vectors and agent Q-values.
- **Result**: Autograd gradients $\frac{\partial Q_{tot}}{\partial q_i}$ are strictly positive across all batch dimensions ($\min(\nabla) > 0$), confirming that individual greedy improvements monotonically improve joint action-value estimates.

### 4.3 PPO-Lagrangian Dual Updates & Device Placement
- **Challenge**: Evaluated dual step updates and device consistency across CPU/GPU offloading.
- **Result**: Lagrange multipliers $\lambda_{sla}, \lambda_{cost}$ maintain strict non-negativity via log-space parametrization and clamp guards ($[-10, 5]$). All tensors and loss targets remain consistent on the target device.

---

## 5. Integrity & Non-Negotiable Conventions Audit

| Check Category | Standard / Requirement | Result | Finding |
|---|---|---|---|
| **Integrity: Hardcoded Outputs** | Source code must contain no hardcoded test outputs or mock bypasses | **PASS** | Audited all source files; zero hardcoded mock returns found |
| **Integrity: Facade Implementations** | Code must contain real algorithmic logic, not empty facades | **PASS** | Full implementations verified across MARL, GNN, Ops, Backend |
| **Integrity: Shortcut Bypasses** | Core functionality must be genuinely implemented | **PASS** | Verified GAE, CTDE, HeteroConv, Action Masking, and KubectlAdapter |
| **Convention: Separate Reward Logs** | Reward components must be logged separately and never collapsed | **PASS** | Verified `marl/reward.py` and `simulator/cluster_env.py` |
| **Convention: Numbered Cypher Migrations** | Schema migrations must be numbered files under `graph/migrations/` | **PASS** | Verified `001_initial_schema.cypher` and `graph/migrate.py` |
| **Convention: LLM Grounding** | LLM prompts must cite only real graph facts | **PASS** | Verified `ops_layer/narrator.py` prompt templates and context boundaries |

---

## 6. Caveats

- **Neo4j Offline Skipping**: 47 integration tests in `tests/graph/` and `tests/encoder/test_graph_source.py` are configured to skip when a live Neo4j instance is not running locally. All static schema verification, migration order, idempotency, and query syntax tests execute and pass offline.
- **CODE_ONLY Network Mode**: Execution and verification took place entirely offline with local virtual environment dependencies (`Python 3.13.14`, `torch 2.6.0`, `torch_geometric 2.6.1`, `gymnasium 1.3.0`, `pettingzoo 1.26.1`, `pytest 9.1.1`).

---

## 7. Conclusion

**Verdict: PASS (APPROVE)**

Milestone 1 satisfies all functional, architectural, adversarial, and integrity requirements. All bugs identified in MARL action masking, PPO-Lagrangian device placement, and QMIX reward handling have been resolved with mathematical precision and zero regressions.

---

## 8. Verification Method

To independently reproduce and verify this review:

1. **Run Full Test Suite**:
   ```powershell
   .venv\Scripts\pytest tests/simulator tests/encoder tests/marl tests/ops_layer tests/graph tests/backend tests/demo -v
   ```
   **Expected**: `335 passed, 47 skipped` with 0 failures.

2. **Run MARL Component Tests**:
   ```powershell
   .venv\Scripts\pytest tests/marl/test_marl_components.py -v
   ```
   **Expected**: `9 passed` in ~2.7s.

3. **Run Demo Kubectl Adapter Tests**:
   ```powershell
   .venv\Scripts\pytest tests/demo/test_kubectl_adapter.py -v
   ```
   **Expected**: `2 passed` in ~0.1s.
