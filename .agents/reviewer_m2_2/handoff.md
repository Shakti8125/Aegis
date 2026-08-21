# Milestone 2 Independent Review Report (Reviewer 2)

**Verdict**: **REQUEST_CHANGES (FAIL)**

---

## 1. Observation

Direct observations from codebase inspection, Python AST compilation, and isolated runtime execution of notebook cells:

### 1.1 Test Suite & AST Baseline
- Full pytest test suite command `.venv\Scripts\pytest tests/` passes with **432 passed, 47 skipped, 0 failed** in 110.41s.
- `tests/test_notebook_structure.py` passes 9/9 tests in 0.06s.
- Python AST compilation (`compile(source, "<cell>", "exec")`) passes across all 12 code cells in `notebooks/aegis_training.ipynb`.

### 1.2 Runtime Failures in Notebook Code Cells
When executing the code cells against the actual codebase modules, two critical runtime exceptions occur:

1. **Cell 6 (`hgt_pretrain` / Stage 1 HGT Pretraining)**:
   - Location: `notebooks/aegis_training.ipynb` lines 192-196:
     ```python
     hgt_encoder = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2).to(device)

     decoders = nn.ModuleDict({
         ntype: nn.Linear(64, hgt_encoder.feature_dims[ntype]) for ntype in hgt_encoder.node_types
     }).to(device)
     ```
   - **Verbatim Error 1**:
     ```
     TypeError: HGTGraphEncoder.__init__() got an unexpected keyword argument 'hidden_dim'
     ```
   - **Verbatim Error 2** (upon fixing init signature):
     ```
     AttributeError: 'HGTGraphEncoder' object has no attribute 'node_types'
     ```
     and
     ```
     AttributeError: 'HGTGraphEncoder' object has no attribute 'feature_dims'
     ```
   - **Root Cause**: `HGTGraphEncoder.__init__` in `encoder/hgt_encoder.py` accepts `config: EncoderConfig | None = None, *, feature_dims=None, edge_dims=None`. It does not accept `hidden_dim`, `num_heads`, `num_layers` directly as keyword parameters. Furthermore, `FEATURE_DIMS` and `NODE_TYPES` are defined in `encoder.features`, and are never assigned as instance attributes `self.node_types` or `self.feature_dims` on `HGTGraphEncoder`.

2. **Cell 12 (`happo_qmix_module` / Stage 3 HAPPO & QMIX Multi-Agent RL)**:
   - Location: `notebooks/aegis_training.ipynb` line 471:
     ```python
     buffer = RolloutBuffer(n_steps=4, n_envs=2, n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim)
     ```
   - **Verbatim Error**:
     ```
     TypeError: RolloutBuffer.__init__() missing 1 required positional argument: 'component_names'
     ```
   - **Root Cause**: `RolloutBuffer.__init__` in `marl/mappo.py` strictly requires 6 positional arguments: `(n_steps, n_envs, n_agents, obs_dim, state_dim, component_names: tuple[str, ...])`. In Cell 12, `component_names` was omitted.

### 1.3 Test Suite Blind Spot
- `tests/test_notebook_structure.py` relies on `compile(cleaned_source, ..., "exec")` in `test_notebook_all_code_cells_compile()`. Python's `compile()` only checks syntactic AST validity, not runtime symbol resolution, object attributes, or function argument arity.
- `test_stage1_hgt_pretraining_float_loss_fix()` performs weak substring matching (`assert "HGTGraphEncoder" in full_code`), which allowed broken runtime logic to pass verification undetected.

---

## 2. Logic Chain

1. **Step 1 — AST vs Runtime Verification**:
   The worker's handoff asserted that the notebook was fully functional end-to-end. However, AST compilation alone is insufficient for interactive Jupyter notebooks where cells are executed sequentially.
2. **Step 2 — Stage 1 Analysis**:
   In `encoder/hgt_encoder.py`, `HGTGraphEncoder` takes an `EncoderConfig` instance. Calling `HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)` causes an immediate crash when the user runs the cell in Google Colab. Accessing `hgt_encoder.feature_dims` and `hgt_encoder.node_types` crashes because those constants reside in `encoder.features`.
3. **Step 3 — Stage 3 Analysis**:
   In `marl/mappo.py`, `RolloutBuffer` enforces multi-component reward tracking by requiring `component_names` upon initialization. Cell 12 omits `component_names=COMPONENT_NAMES`, causing an immediate `TypeError`.
4. **Step 4 — Conclusion**:
   A user running `notebooks/aegis_training.ipynb` in Google Colab will encounter hard runtime failures in Stage 1 and Stage 3. Therefore, Milestone 2 cannot be approved in its current state.

---

## 3. Caveats

- The other cells in `notebooks/aegis_training.ipynb` (Colab setup, PyG GraphSAGE probe gate, Decision Transformer offline trajectory collection & training, MAPPO training subprocess, baseline evaluation, and Google Drive checkpoint sync) were tested and verified to execute cleanly.
- Live Neo4j integration tests (47 tests in `tests/graph/`) require a live Neo4j daemon and are skipped as expected in standard offline test environments.

---

## 4. Conclusion

**Verdict**: **REQUEST_CHANGES (FAIL)**

Milestone 2 requires the following fixes before approval:

### Required Changes:

1. **Fix Cell 6 (`id: "hgt_pretrain"`) in `notebooks/aegis_training.ipynb`**:
   Replace:
   ```python
   from encoder.hgt_encoder import HGTGraphEncoder
   from encoder.dataset import TRAIN_SIZES, collect_sized_dataset, iter_all

   ...
   hgt_encoder = HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2).to(device)

   decoders = nn.ModuleDict({
       ntype: nn.Linear(64, hgt_encoder.feature_dims[ntype]) for ntype in hgt_encoder.node_types
   }).to(device)
   ```
   With:
   ```python
   from encoder.gnn_model import EncoderConfig
   from encoder.features import FEATURE_DIMS, NODE_TYPES
   from encoder.hgt_encoder import HGTGraphEncoder
   from encoder.dataset import TRAIN_SIZES, collect_sized_dataset, iter_all

   ...
   hgt_encoder = HGTGraphEncoder(EncoderConfig(hidden_dim=64, num_layers=2)).to(device)

   decoders = nn.ModuleDict({
       ntype: nn.Linear(64, FEATURE_DIMS[ntype]) for ntype in NODE_TYPES
   }).to(device)
   ```

2. **Fix Cell 12 (`id: "happo_qmix_module"`) in `notebooks/aegis_training.ipynb`**:
   Add import and pass `component_names`:
   ```python
   from marl.happo import HAPPO, HAPPOConfig
   from marl.qmix import QMIX, QMixer, QMIXConfig
   from marl.mappo import RolloutBuffer
   from marl.reward import COMPONENT_NAMES
   ...
   buffer = RolloutBuffer(
       n_steps=4,
       n_envs=2,
       n_agents=n_agents,
       obs_dim=obs_dim,
       state_dim=state_dim,
       component_names=COMPONENT_NAMES,
   )
   ```

3. **Enhance `tests/test_notebook_structure.py`**:
   Add test assertions verifying that `EncoderConfig`, `FEATURE_DIMS`, `NODE_TYPES`, and `COMPONENT_NAMES` are correctly utilized in the respective notebook cells.

---

## 5. Verification Method

To independently verify these findings:

1. **Verify Cell 6 Bug**:
   ```bash
   .venv\Scripts\python -c "from encoder.hgt_encoder import HGTGraphEncoder; HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)"
   ```
   *Expected result*: `TypeError: HGTGraphEncoder.__init__() got an unexpected keyword argument 'hidden_dim'`.

2. **Verify Cell 12 Bug**:
   ```bash
   .venv\Scripts\python -c "from marl.mappo import RolloutBuffer; RolloutBuffer(n_steps=4, n_envs=2, n_agents=12, obs_dim=38, state_dim=143)"
   ```
   *Expected result*: `TypeError: RolloutBuffer.__init__() missing 1 required positional argument: 'component_names'`.

3. **Verify Full Test Suite**:
   ```bash
   .venv\Scripts\pytest tests/
   ```
   *Expected result*: 432 passed, 47 skipped.

---

## Quality Review Report

### Findings

#### [Critical] Finding 1: Broken `HGTGraphEncoder` Call & Missing Attributes in Cell 6
- **What**: `HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)` and `hgt_encoder.feature_dims` / `hgt_encoder.node_types` raise `TypeError` and `AttributeError`.
- **Where**: `notebooks/aegis_training.ipynb` (Cell 6, lines 192-196).
- **Why**: `HGTGraphEncoder` expects `EncoderConfig`. `FEATURE_DIMS` and `NODE_TYPES` belong to `encoder.features`.
- **Suggestion**: Import `EncoderConfig`, `FEATURE_DIMS`, `NODE_TYPES` and construct encoder with `HGTGraphEncoder(EncoderConfig(hidden_dim=64, num_layers=2))`.

#### [Critical] Finding 2: Missing `component_names` Argument in `RolloutBuffer` in Cell 12
- **What**: `RolloutBuffer` constructor is missing the mandatory `component_names` positional argument.
- **Where**: `notebooks/aegis_training.ipynb` (Cell 12, line 471).
- **Why**: `RolloutBuffer.__init__` requires `component_names: tuple[str, ...]`.
- **Suggestion**: Import `COMPONENT_NAMES` from `marl.reward` and pass `component_names=COMPONENT_NAMES`.

#### [Major] Finding 3: `tests/test_notebook_structure.py` Blind to Runtime Execution Errors
- **What**: Tests only checked AST compilation and substring presence, allowing crashing code to pass.
- **Where**: `tests/test_notebook_structure.py` (lines 34-57, 92-106).
- **Why**: AST compilation does not validate argument signatures or module attribute lookups.
- **Suggestion**: Add explicit assertions for `EncoderConfig`, `FEATURE_DIMS`, `NODE_TYPES`, and `COMPONENT_NAMES`.

---

## Adversarial Review / Stress Test Report

- **Stress Test 1 (Cell 6 Signature & Attribute Evaluation)**: `HGTGraphEncoder(hidden_dim=64, num_heads=4, num_layers=2)` -> **FAILED** (`TypeError`).
- **Stress Test 2 (Cell 12 Buffer Initialization)**: `RolloutBuffer(n_steps=4, n_envs=2, n_agents=12, obs_dim=38, state_dim=143)` -> **FAILED** (`TypeError`).
- **Stress Test 3 (Cell 9 Decision Transformer Forward Pass)**: `DecisionTransformer(state_dim=143, n_actions=6)` -> **PASSED**.
- **Stress Test 4 (Cell 14 MAPPO Checkpoint Loading & Benchmark Evaluation)**: `evaluate(make_env, RuleBasedController(), ...)` -> **PASSED**.
